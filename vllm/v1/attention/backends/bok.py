# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashInfer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, List

import torch
import numpy as np
import math

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
)
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.block_table import BlockTable

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from py_bok import (
    create_op,
    forward_inplace,
    calculate_workspace_size,
    AttentionLayerDimensions,
    DeviceDataType,
    RotaryEmbedding,
    RotaryScalingType,
    RotaryPositionalEmbeddingType,
    PrefixCacheConfiguration,
    AttentionOp,
)

logger = init_logger(__name__)


class BokAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "BOK"

    @staticmethod
    def get_impl_cls() -> type[BokImpl]:
        return BokImpl

    @staticmethod
    def get_metadata_cls() -> type[BokMetadata]:
        return BokMetadata

    @staticmethod
    def get_builder_cls() -> type[BokMetadataBuilder]:
        return BokMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)


@dataclass
class BokMetadata:
    op: AttentionOp
    attention_layer_dimensions: AttentionLayerDimensions
    num_prefill_requests: int
    seq_lens_gpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    input_sequence_lengths_host: torch.Tensor
    input_sequence_lengths_device: torch.Tensor
    workspace: torch.Tensor
    rotary_cos_sin: torch.Tensor
    block_table: BlockTable
    num_reqs: int
    context_chunk_size: int
    fp8_output_buffer: torch.Tensor


class BokMetadataBuilder:

    def __init__(
        self,
        runner: GPUModelRunner,
        kv_cache_spec: AttentionSpec,
        block_table: BlockTable,
    ):
        self.runner = runner
        self._workspace_buffer = None

        self.vllm_config = runner.vllm_config
        self.kv_cache_spec = kv_cache_spec
        self.block_table = block_table
        self.context_chunk_size = runner.scheduler_config.max_num_batched_tokens

        self.attention_layer_dimensions = AttentionLayerDimensions()
        self.attention_layer_dimensions.numQHeads = runner.num_query_heads
        self.attention_layer_dimensions.numKVHeads = kv_cache_spec.num_kv_heads
        self.attention_layer_dimensions.headSize = kv_cache_spec.head_size

        # TODO: find exact values.
        rotary_positional_embedding = RotaryEmbedding()

        rope_scaling = getattr(
            runner.vllm_config.model_config.hf_config, "rope_scaling", None
        )
        scaling_factor = rope_scaling.get("factor", 1.0) if rope_scaling else 1.0
        rotary_positional_embedding.rotaryEmbeddingScale = scaling_factor
        mapping_dict = {
            "none": RotaryScalingType.NONE,
            "linear": RotaryScalingType.LINEAR,
            "dynamic": RotaryScalingType.DYNAMIC,
            "longrope": RotaryScalingType.LONG,
            "llama3": RotaryScalingType.LLAMA3,
        }
        rotary_positional_embedding.rotaryScalingType = (
            RotaryScalingType.LLAMA3
        )  # TODO: fix.

        max_position_embeddings = getattr(
            runner.vllm_config.model_config.hf_config, "max_position_embeddings", 8192
        )
        rotary_positional_embedding.rotaryEmbeddingMaxPositions = (
            max_position_embeddings
        )

        rope_theta = getattr(
            runner.vllm_config.model_config.hf_config, "rope_theta", 10000
        )
        rotary_positional_embedding.rotaryEmbeddingBase = rope_theta

        rotary_positional_embedding.rotaryEmbeddingDim = (
            runner.vllm_config.model_config.get_head_size()
        )
        rotary_positional_embedding.type = (
            RotaryPositionalEmbeddingType.GPT_NEOX
        )  # TODO: what to do with this?

        prefix_cache_configuration = PrefixCacheConfiguration()
        prefix_cache_configuration.dataType = (
            DeviceDataType.FP8_E4M3
        )  # TODO: needs to take actual dtype
        prefix_cache_configuration.numTokensPerBlock = (
            kv_cache_spec.block_size
        )  # TODO: check correctness
        prefix_cache_configuration.maxNumBlocksPerSequence = (
            block_table.max_num_blocks_per_req  # TODO: check correctness
        )

        max_attention_window_size = runner.vllm_config.model_config.max_model_len
        cyclic_attention_window_size = runner.vllm_config.model_config.max_model_len

        # TODO: check if max_num_seqs == max_batch_size
        self.multi_block_semaphores = torch.zeros(
            runner.num_query_heads * runner.scheduler_config.max_num_seqs,
            device=torch.device("cuda"),
            dtype=torch.int32,
            requires_grad=False,
        ).contiguous()

        # Create a representation of the fixed parameters of the attention operation.
        self.op = create_op(
            inputDataType=DeviceDataType.BF16,
            outputDataType=DeviceDataType.FP8_E4M3,
            attentionLayerDimensions=self.attention_layer_dimensions,
            rotaryEmbedding=rotary_positional_embedding,
            prefixCacheConfiguration=prefix_cache_configuration,
            qScaling=1.0,
            maxAttentionWindowSize=max_attention_window_size,
            cyclicAttentionWindowSize=cyclic_attention_window_size,
            outputScalingFactor=1.0,
            multiBlockSemaphores=self.multi_block_semaphores,
            enableSpeculativeDecoding=False,
        )
        # TODO: check if max_num_seqs == max_batch_size
        workspace_size = calculate_workspace_size(
            self.op,
            self.runner.max_num_tokens,
            self.runner.scheduler_config.max_num_seqs,
        )

        self.workspace = torch.zeros(
            workspace_size,
            device=torch.device("cuda"),
            dtype=torch.int8,
            requires_grad=False,
        ).contiguous()

        _, self.rotary_cos_sin_ndarray = (
            create_sinusoidal_positions_for_attention_plugin(
                rotary_positional_embedding.rotaryEmbeddingMaxPositions,
                rotary_positional_embedding.rotaryEmbeddingDim,
                rotary_positional_embedding.rotaryEmbeddingBase,
                rotary_positional_embedding.rotaryEmbeddingScale,
                rotary_positional_embedding.rotaryScalingType,
                rope_scaling_config=rope_scaling,
            )
        )
        self.rotary_cos_sin = torch.tensor(
            self.rotary_cos_sin_ndarray,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        ).contiguous()

        self.fp8_output_buffer = torch.zeros(
            self.runner.max_num_tokens,
            self.runner.num_query_heads,
            self.runner.vllm_config.model_config.get_head_size(),
            device=torch.device("cuda"),
            dtype=torch.float8_e4m3fn,
            requires_grad=False,
        ).contiguous()

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False  # TODO: implement this

    def reorder_batch(
        self, input_batch: InputBatch, scheduler_output: SchedulerOutput
    ) -> bool:
        # We now want to reorder the batch so that the "decode" requests are and
        # the front and the "prefill" requests are at the using the least amount
        # swaps possible. (NOTE for now we loosely use "decode" to mean requests
        # where attention is likely memory-bound and "prefill" to mean requests
        # where attention is likely compute-bound
        decodes = []
        prefills = []
        num_decode_tokens = 0
        num_prefill_tokens = 0

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # for now treat 1 scheduled token as "decode" even if its not,
            # we should update this to something like < 8 in the future but
            # currently the decode run only supports num_tokens = 1
            if num_tokens == 1:
                decodes.append(i)
                num_decode_tokens += num_tokens
            else:
                prefills.append(i)
                num_prefill_tokens += num_tokens

        # We hope that this is fairly minimal since decodes
        # should be around for a number of iterations so hopefully they are
        # relatively stationary (and new request are generally appended to the
        # persistent batch so already should be at the back)
        # To achieve this we loop over the decodes in descending order and
        # the prefills in ascending order. We swap decodes from the  "back"
        # i.e. past where the last decode should be in the reodorered with
        # prefills from the front of the batch.
        # `decodes` and `prefills` are already in ascending order just based on
        # the above loop
        num_decodes = len(decodes)
        num_prefills = len(prefills)
        modified_batch = False

        # TODO: doing the wrong thing currently. Fix when moving to multiple sequences cases.
        # for i in range(1, min(num_decodes, num_prefills) + 1):
        #     # If the decode is at the "back" of the batch, i, we can swap it
        #     # with the prefill closest to the front of the batch
        #     decode_idx = decodes[num_decodes - i]
        #     if decode_idx < num_decodes:
        #         break

        #     input_batch.swap_states(prefills[i - 1], decode_idx)
        #     modified_batch = True

        self._num_decode_requests = num_decodes
        self._num_prefill_requests = num_prefills
        self._num_decode_tokens = num_decode_tokens
        self._num_prefill_tokens = num_prefill_tokens

        return False

    def build(
        self,
        num_reqs: int,
        num_actual_tokens: int,
        max_query_len: int,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
    ):
        assert self._num_decode_requests + self._num_prefill_requests == num_reqs
        assert self._num_decode_tokens + self._num_prefill_tokens == num_actual_tokens

        seq_lens_gpu = common_attn_metadata.seq_lens
        seq_lens_cpu = self.runner.seq_lens_cpu[:num_reqs]
        input_sequence_lengths_device = common_attn_metadata.query_start_loc.diff().to(
            dtype=torch.uint32
        )
        input_sequence_lengths_host = input_sequence_lengths_device.to(
            device=torch.device("cpu"),
        )
        attn_metadata = BokMetadata(
            op=self.op,
            attention_layer_dimensions=self.attention_layer_dimensions,
            num_prefill_requests=self._num_prefill_requests,
            seq_lens_gpu=seq_lens_gpu,
            seq_lens_cpu=seq_lens_cpu,
            input_sequence_lengths_host=input_sequence_lengths_host,
            input_sequence_lengths_device=input_sequence_lengths_device,
            workspace=self.workspace,
            rotary_cos_sin=self.rotary_cos_sin,
            block_table=self.block_table,
            num_reqs=num_reqs,
            context_chunk_size=self.context_chunk_size,
            fp8_output_buffer=self.fp8_output_buffer,
        )

        return attn_metadata


class BokImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[int] = None,
        use_irope: bool = False,
    ) -> None:
        self.scale = scale
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        if use_irope:
            logger.warning_once(
                "Using irope in FlashInfer is not supported yet, it will fall"
                " back to global attention for long context."
            )
        self.scale = float(scale)
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashInferImpl"
            )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: BokMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashInfer.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache = [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            # Profiling run.
            return output

        kv_cache_block_offsets = attn_metadata.block_table.block_table[
            : attn_metadata.num_reqs, :
        ]
        seq_lens_cpu_uint32 = attn_metadata.seq_lens_cpu.to(torch.uint32)
        seq_lens_gpu_uint32 = attn_metadata.seq_lens_gpu.to(torch.uint32)
        kv_cache_block_offsets_uint32 = kv_cache_block_offsets.to(torch.uint32)
        kv_cache_data_ptr = kv_cache.data_ptr()
        cuda_stream = torch.cuda.current_stream().cuda_stream

        forward_inplace(
            op=attn_metadata.op,
            qkv=query,
            numContextRequests=attn_metadata.num_prefill_requests,
            contextChunkSize=attn_metadata.context_chunk_size,
            # TODO: see how to get actual input seqlens
            # TODO: see how to skip / workaround the uint cast
            inputSequenceLengthsHost=attn_metadata.input_sequence_lengths_host,
            inputSequenceLengthsDevice=attn_metadata.input_sequence_lengths_device,
            sequenceLengthsDevice=seq_lens_gpu_uint32,
            sequenceLengthsHost=seq_lens_cpu_uint32,
            kvCacheBlockOffsets=kv_cache_block_offsets_uint32,
            kvCachePoolPtr=kv_cache_data_ptr,
            rotaryCosSin=attn_metadata.rotary_cos_sin,
            output=attn_metadata.fp8_output_buffer.view(torch.int8),
            workspace=attn_metadata.workspace,
            stream=cuda_stream,
        )

        # TODO: copying fp8 attention outputs to the bf16 output tensor expected by vLLM, or figure out how to enable fp8 output, although it seems tied to input dtype at this point.
        output.copy_(attn_metadata.fp8_output_buffer[: output.shape[0], :, :])
        return output


def apply_llama3_scaling(inv_freqs: np.ndarray, rope_scaling_config: dict):

    scale_factor = rope_scaling_config.get("factor", 8.0)
    low_freq_factor = rope_scaling_config.get("low_freq_factor", 1.0)
    high_freq_factor = rope_scaling_config.get("high_freq_factor", 4.0)
    old_context_len = rope_scaling_config.get("original_max_position_embeddings", 8192)

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_inv_freqs = []
    for inv_freq in inv_freqs:
        wavelen = 2 * math.pi / inv_freq
        if wavelen < high_freq_wavelen:
            new_inv_freqs.append(inv_freq)
        elif wavelen > low_freq_wavelen:
            new_inv_freqs.append(inv_freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_inv_freqs.append(
                (1 - smooth) * inv_freq / scale_factor + smooth * inv_freq
            )
    return np.array(new_inv_freqs, dtype=inv_freqs.dtype)


def create_sinusoidal_positions_for_attention_plugin(
    num_pos: int,
    dim: int,
    theta: float = 10000.0,
    scale: float = 1.0,
    scale_type: RotaryScalingType = RotaryScalingType.NONE,
    # Other scaling configs that only used by certain scaling types.
    rope_scaling_config: dict = None,
    dtype=np.float32,
) -> List[np.ndarray]:
    if scale_type == RotaryScalingType.LINEAR:
        scale = 1.0 / scale
    if scale_type == RotaryScalingType.LLAMA3:
        assert (
            rope_scaling_config is not None
        ), "rotary_scaling config must be provided."
        inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2) / dim)).astype(dtype)
        inv_freq = apply_llama3_scaling(inv_freq, rope_scaling_config)
    else:
        inv_freq = scale / (theta ** (np.arange(0, dim, 2) / dim)).astype(dtype)
    sinusoid_inp = np.expand_dims(
        np.einsum(
            "i , j -> i j", np.arange(num_pos, dtype=dtype), inv_freq, dtype=dtype
        ),
        axis=-1,
    )
    # fuse cos/sin into float2 (cos, sin).
    concat = np.concatenate(
        (np.cos(sinusoid_inp), np.sin(sinusoid_inp)), axis=-1
    )  # np.cos(sinusoid_inp).shape = (32768, 64, 1)

    return inv_freq, concat.astype(dtype).reshape(num_pos, dim)
