from typing import Callable, List

from transformers import PreTrainedTokenizer

from vllm.core.scheduler import Scheduler
from vllm.engine.output_processor.interfaces import (
    SequenceGroupOutputProcessor)
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.sequence import (Logprob, Sequence, SequenceGroup,
                           SequenceGroupOutput, SequenceOutput, SequenceStatus)
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.utils import Counter

logger = init_logger(__name__)
# 这段代码定义了一个名为 MultiStepOutputProcessor 的类，它是 SequenceGroupOutputProcessor 的一个具体实现。此类专门用于处理多步解码（multi-step decoding），在这种模式下，vLLM 的工作节点可能每次调用生成多个令牌。
# 这与诸如 beam search 这样的高级采样技术是互斥的，因此将这种逻辑与单步输出处理器分开。

class MultiStepOutputProcessor(SequenceGroupOutputProcessor):
    """SequenceGroupOutputProcessor which handles logic related to
    detokenization and stopping conditions. It specializes to "multi-step
    decoding", where vLLM's worker may generate multiple tokens per invocation.
    This is currently mutually exclusive with advanced sampling techniques like
    beam search, which motivates the separation of this logic from the single
    step output processor.

    This class is responsible for things such as correctly appending all new
    token ids to their sequence, detokenizing new token ids, truncating new
    output tokens after an eos token, and correctly handling the case where the
    number of new output tokens per sequence differs in a single batch.
    """

    # __init__ 方法接收以下参数：
    # detokenizer：用于将令牌ID解码回文本字符串。
    # scheduler：调度器，管理序列组的执行和资源分配。
    # seq_counter：用于跟踪序列的计数器。
    # get_tokenizer_for_seq：一个函数，根据给定的序列返回适当的分词器（PreTrainedTokenizer）。
    # stop_checker：用于检查是否应该停止序列的对象。
    def __init__(
        self,
        detokenizer: Detokenizer,
        scheduler: Scheduler,
        seq_counter: Counter,
        get_tokenizer_for_seq: Callable[[Sequence], PreTrainedTokenizer],
        stop_checker: StopChecker,
    ):
        self.detokenizer = detokenizer
        self.scheduler = scheduler
        self.seq_counter = seq_counter
        self.get_tokenizer_for_seq = get_tokenizer_for_seq
        self.stop_checker = stop_checker

    def process_prompt_logprob(self, seq_group: SequenceGroup,
                               outputs: List[SequenceGroupOutput]) -> None:
        # TODO(sang): Prompt logprob currently not implemented in multi step
        # workers.
        logger.warning(
            "Prompt logprob is not supported by multi step workers. "
            "(e.g., speculative decode uses multi step workers).")
        pass
    # process_outputs 方法接收一个序列组和一系列的 SequenceGroupOutput 对象作为输入。此方法专门处理以下情况：
    # 确保只有一个正在运行的序列（因为多步解码不支持 beam search）。
    # 从输出中提取有效样本。
    # 截断超出 max_tokens 的令牌。
    # 管理在 EOS 令牌后生成的令牌。
    # 将新的令牌追加到序列中，并在适当的时候停止序列。
    def process_outputs(self, sequence_group: SequenceGroup,
                        outputs: List[SequenceGroupOutput]) -> None:
        """Append new tokens in the outputs to sequences in the sequence group.

        This only supports sequence groups of size 1. It supports greater than
        one new token per sequence.

        This applies logic like stop condition checking and detokenization,
        including freeing finished sequences. It also handles cases where there
        are tokens emitted after the EOS token.
        """
        seqs = sequence_group.get_seqs(status=SequenceStatus.RUNNING)

        assert seqs, "expected running sequences"
        assert len(seqs) == 1, (
            "Beam search not supported in multi-step decoding.")
        seq = seqs[0]

        # Since there's only one sequence per sequence group, we can take the
        # first sample.
        samples = [outputs[step].samples[0] for step in range(len(outputs))]

        # -1 means the output token is not valid (eg. due to spec decode
        # rejecting tokens).
        valid_samples = [
            sample for sample in samples if sample.output_token != -1
        ]
        assert valid_samples

        self._process_seq_outputs(seq, valid_samples,
                                  sequence_group.sampling_params)
    # _process_seq_outputs 方法处理从有效样本中提取的输出令牌ID：
    #
    # 首先，根据序列的 sampling_params（采样参数）检查是否需要忽略 EOS 令牌。
    # 对输出令牌进行迭代，并逐个追加到序列中。
    # 对于每个新令牌，如果启用了 detokenization（解码），则调用 detokenizer 更新序列的文本表示。
    # 使用 stop_checker 检查是否应该停止序列。如果序列完成，则通过调度器释放序列的资源。
    def _process_seq_outputs(self, seq: Sequence,
                             valid_samples: List[SequenceOutput],
                             sampling_params: SamplingParams) -> None:
        output_token_ids = [sample.output_token for sample in valid_samples]

        # Truncate to max_tokens if necessary.
        remaining_tokens = sampling_params.max_tokens - (seq.get_output_len() +
                                                         len(output_token_ids))
        if remaining_tokens < 0:
            valid_samples = valid_samples[:remaining_tokens]
            output_token_ids = output_token_ids[:remaining_tokens]

        # Truncate any tokens after EOS. This is required as spec decode
        # generates a fixed number of tokens without evaluating stopping
        # conditions within the block. This can cause an eos token to be
        # unintentionally ignored.
        if not sampling_params.ignore_eos:
            eos_token_id = self.get_tokenizer_for_seq(seq).eos_token_id
            # Avoiding .index calls as exception throwing in the happy path
            # is expensive.
            for i in range(len(output_token_ids)):
                if output_token_ids[i] == eos_token_id:
                    output_token_ids = output_token_ids[:i + 1]
                    valid_samples = valid_samples[:i + 1]
                    break

        # Incrementally append tokens to the sequence, as if we had only one new
        # token.
        for output_token_id in output_token_ids:
            seq.append_token_id(
                token_id=output_token_id,
                # TODO emit logprobs in multi-step decoding.
                logprobs={output_token_id: Logprob(0.0)},
            )

            new_char_count = 0
            if sampling_params.detokenize:
                new_char_count = self.detokenizer.decode_sequence_inplace(
                    seq, sampling_params)

            self.stop_checker.maybe_stop_sequence(
                seq,
                new_char_count=new_char_count,
                sampling_params=sampling_params)
            if seq.is_finished():
                break

        if seq.is_finished():
            self.scheduler.free_seq(seq)
