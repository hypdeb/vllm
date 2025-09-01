# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashInfer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from py_tke import (AttentionLayerDimensions, BlockOffsetLayout,
                    DeviceDataType, PrefixCacheConfiguration, RotaryEmbedding,
                    RotaryPositionalEmbeddingType, RotaryScalingType,
                    calculate_workspace_size, create_op, run_context_inplace,
                    run_generation_inplace)

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer, AttentionType,
                                              InputLayout)
from vllm.config import SpeculativeConfig, VllmConfig
from vllm.config.cache import CacheConfig
from vllm.config.scheduler import SchedulerConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding.config import (
    RotaryEmbeddingConfig)
from vllm.utils import current_stream
from vllm.v1.attention.backends.utils import (AttentionMetadataBuilder,
                                              CommonAttentionMetadata,
                                              split_decodes_and_prefills)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

rope_scaling_type_mapping = {
    "none": RotaryScalingType.NONE,
    "llama3": RotaryScalingType.LLAMA3,
}

# Global cache for rotary cos/sin tensors
_ROTARY_COS_SIN_CACHE: dict[tuple, torch.Tensor] = {}


class TkeAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "TKE"

    @staticmethod
    def get_impl_cls() -> type[TkeImpl]:
        return TkeImpl

    @staticmethod
    def get_metadata_cls() -> type[TkeMetadata]:
        return TkeMetadata

    @staticmethod
    def get_builder_cls() -> type[TkeMetadataBuilder]:
        return TkeMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order() -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets us from
        # `get_kv_cache_shape` to the actual memory layout we want.
        # NOTE: 3 and 2 are swapped because the TRTLLM layout within blocks is [num_heads, num_tokens, dimension]
        return (0, 1, 3, 2, 4)

    @staticmethod
    def get_output_dtype(kv_cache_dtype: torch.dtype) -> torch.dtype:
        if kv_cache_dtype == torch.uint8:
            return torch.float8_e4m3fn
        return kv_cache_dtype

    @staticmethod
    def get_input_layout() -> InputLayout:
        return InputLayout.CONTIGUOUS_QKV

    @staticmethod
    def get_backend_applies_rotary_embedding() -> bool:
        return True


@dataclass(frozen=True)
class TkeMetadata:
    # The shared attention metadata.
    common_attn_metadata: CommonAttentionMetadata

    # Pre-computed sequence lengths for the prefill/context sequences in the batch.
    context_sequence_lengths_device: torch.Tensor

    # Pre-computed input sequence lengths for the prefill/context sequences in the batch.
    context_input_sequence_lengths_device: torch.Tensor

    # Pre-computed block table tensor for the prefill/context sequences in the batch.
    context_block_table_tensor: torch.Tensor

    # Length of queries for all sequences in the batch.
    query_lens: torch.Tensor

    # Pre-computed max sequence length for the prefill/context sequences in the batch.
    context_max_sequence_length: int

    # Pre-computed max sequence length for the generation sequences in the batch.
    generation_max_sequence_length: int

    # Pre-computed max input sequence length for the generation sequences in the batch.
    generation_max_input_sequence_length: int

    # The number of sequences in context phase.
    num_context_sequences: int

    # The number of context tokens in the batch.
    num_context_tokens: int

    # The number of generation sequences in the batch.
    num_generation_sequences: int

    # The number of generation tokens in the batch.
    num_generation_tokens: int


def _torch_to_device_data_type(dtype: torch.dtype) -> DeviceDataType:
    if dtype == torch.float8_e4m3fn:
        return DeviceDataType.FP8_E4M3
    if (
            dtype == torch.uint8
    ):  # FIXME: hmmm... I don't know why the dtype would be uint8 in the first place.
        return DeviceDataType.FP8_E4M3
    if dtype == torch.bfloat16:
        return DeviceDataType.BF16
    raise RuntimeError(f"Unsupported dtype: {dtype}")


def _str_to_device_data_type(dtype: str) -> DeviceDataType:
    if dtype == "fp8_e4m3":
        return DeviceDataType.FP8_E4M3
    if dtype == "fp8":
        return DeviceDataType.FP8_E4M3
    if dtype == "bfloat16":
        return DeviceDataType.BF16
    raise RuntimeError(f"Unsupported dtype: {dtype}")


class TkeMetadataBuilder(AttentionMetadataBuilder[TkeMetadata]):

    def reorder_batch_threshold(self) -> Optional[int]:
        return self._reorder_batch_threshold

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):

        # For this backend, the data type of the kv-cache determines the data type in which we perform operation.
        self.kv_cache_device_data_type = _torch_to_device_data_type(
            kv_cache_spec.dtype)

        self.xqa_enabled = self.kv_cache_device_data_type == DeviceDataType.FP8_E4M3

        # When doing speculative decoding, consider all input sequences with fewer tokens than the configured number of
        # speculative tokens as "decode" requests and pull them to the front of the batch.
        # Also, we do not need to do any reordering if we do not use generation optimized kernels (XQA) for generation.
        if self.xqa_enabled:
            if vllm_config.speculative_config is not None and vllm_config.speculative_config.num_speculative_tokens is not None:
                self._reorder_batch_threshold: int | None = vllm_config.speculative_config.num_speculative_tokens
            else:
                self._reorder_batch_threshold = 1
        else:
            self._reorder_batch_threshold = None

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False  # TODO: implement this

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ):
        if self.xqa_enabled:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens =\
                split_decodes_and_prefills(common_attn_metadata)
        else:
            num_decodes = 0
            num_prefills = common_attn_metadata.num_reqs
            num_decode_tokens = 0
            num_prefill_tokens = common_attn_metadata.num_actual_tokens

        query_lens_cpu = common_attn_metadata.query_start_loc_cpu[
            1:] - common_attn_metadata.query_start_loc_cpu[:-1]
        query_lens = common_attn_metadata.query_start_loc[
            1:] - common_attn_metadata.query_start_loc[:-1]

        context_max_sequence_length = (
            common_attn_metadata.seq_lens_cpu[num_decodes:].max().item()
            if num_prefills > 0 else 0)
        generation_max_sequence_length = (
            common_attn_metadata.seq_lens_cpu[:num_decodes].max().item()
            if num_decodes > 0 else 0)
        generation_max_input_sequence_length = (
            query_lens_cpu[:num_decodes].max().item()
            if num_decodes > 0 else 0)

        return TkeMetadata(
            common_attn_metadata=common_attn_metadata,
            context_sequence_lengths_device=common_attn_metadata.
            seq_lens[num_decodes:],
            context_input_sequence_lengths_device=query_lens[num_decodes:],
            context_block_table_tensor=common_attn_metadata.
            block_table_tensor[num_decodes:],
            query_lens=query_lens,
            context_max_sequence_length=int(context_max_sequence_length),
            generation_max_sequence_length=int(generation_max_sequence_length),
            generation_max_input_sequence_length=int(
                generation_max_input_sequence_length),
            num_context_sequences=num_prefills,
            num_context_tokens=num_prefill_tokens,
            num_generation_sequences=num_decodes,
            num_generation_tokens=num_decode_tokens,
        )


class TkeImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[int] = None,
        use_irope: bool = False,
        rotary_embedding_config: Optional[RotaryEmbeddingConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        speculative_decoding_config: Optional[SpeculativeConfig] = None,
        scheduler_config: Optional[SchedulerConfig] = None,
    ) -> None:
        if rotary_embedding_config is None:
            raise RuntimeError(
                "The TKE backend needs to have a configured RoPE.")

        if cache_config is None:
            raise RuntimeError(
                "The TKE backend needs to know about the cache configuration.")

        if scheduler_config is None:
            raise RuntimeError(
                "The TKE backend needs to know about the scheduler configuration."
            )

        kv_cache_device_data_type = _str_to_device_data_type(kv_cache_dtype)

        # Extract the dimensions of the attention layer.
        self.attention_layer_dimensions = AttentionLayerDimensions()
        self.attention_layer_dimensions.numQHeads = num_heads
        self.attention_layer_dimensions.numKVHeads = num_kv_heads
        self.attention_layer_dimensions.headSize = head_size

        # Extract the configuration of the KV-cache.
        prefix_cache_configuration = PrefixCacheConfiguration()
        prefix_cache_configuration.dataType = kv_cache_device_data_type
        prefix_cache_configuration.numTokensPerBlock = cache_config.block_size
        prefix_cache_configuration.maxNumBlocksPerSequence = (
            rotary_embedding_config.max_positions // cache_config.block_size)
        prefix_cache_configuration.blockOffsetLayout = BlockOffsetLayout.VLLM

        # Internal quantity used to size the kv-cache TMA descriptor.
        # TODO: calculate this value from the size of the kv-cache. It needs to be large enough that the TMA descriptor can fit the whole kv-cache tensor.
        # I couldn't find a way to access the actual size of the kv-cache at this time. The 'num_blocks' on the kv-cache config is not set at this point.
        prefix_cache_configuration.maxNumSequences = 8192

        # Extract RoPE configuration.
        self.rotary_embedding = RotaryEmbedding()
        self.rotary_embedding.rotaryEmbeddingBase = rotary_embedding_config.base
        self.rotary_embedding.rotaryEmbeddingMaxPositions = rotary_embedding_config.max_positions
        self.rotary_embedding.rotaryEmbeddingDim = rotary_embedding_config.dimension
        self.rotary_embedding.rotaryEmbeddingScale = 1.0  # TODO: for now only support no scaling.
        self.rotary_embedding.rotaryScalingType = RotaryScalingType.NONE  # TODO: ditto.
        self.rotary_embedding.type = RotaryPositionalEmbeddingType.GPT_NEOX  # TODO: only support GPT_NEOX for now.

        # Create cache key for rotary cos/sin tensor
        cache_key = (
            self.rotary_embedding.rotaryEmbeddingMaxPositions,
            self.rotary_embedding.rotaryEmbeddingDim,
            self.rotary_embedding.rotaryEmbeddingBase,
            self.rotary_embedding.rotaryEmbeddingScale,
            self.rotary_embedding.rotaryScalingType,
        )

        # Check if we already have the rotary cos/sin tensor in cache
        if cache_key in _ROTARY_COS_SIN_CACHE:
            self.rotary_cos_sin = _ROTARY_COS_SIN_CACHE[cache_key]
        else:
            _, self.rotary_cos_sin_ndarray = (
                create_sinusoidal_positions_for_attention_plugin(
                    self.rotary_embedding.rotaryEmbeddingMaxPositions,
                    self.rotary_embedding.rotaryEmbeddingDim,
                    self.rotary_embedding.rotaryEmbeddingBase,
                    self.rotary_embedding.rotaryEmbeddingScale,
                    self.rotary_embedding.rotaryScalingType,
                ))
            self.rotary_cos_sin = torch.tensor(
                self.rotary_cos_sin_ndarray,
                dtype=torch.float32,
                device="cuda",
                requires_grad=False,
            ).contiguous()
            # Cache the tensor for future use
            _ROTARY_COS_SIN_CACHE[cache_key] = self.rotary_cos_sin
        self.rotary_embedding.rotaryCosSinCache = self.rotary_cos_sin.data_ptr(
        )

        # FIXME: should be moved to forward().
        self.output_scaling_factor = torch.tensor(
            [1.0],
            dtype=torch.float32,
            device=torch.device("cuda"),
            requires_grad=False,
        )

        # NOTE: According to modelopt team, 1.0 is almost always the optimal value.
        # TODO: There should also be the equivalent dequantization factor. Add support for that.
        self.kv_cache_dequantization_factor = torch.tensor(
            [1.0],
            dtype=torch.float32,
            device=torch.device("cuda"),
            requires_grad=False,
        )

        # NOTE: XQA BF16 is not enabled yet, meaning that for BF16 attention, we always use FMHA_V2.
        self.xqa_enabled = (
            kv_cache_device_data_type == DeviceDataType.FP8_E4M3)

        # An internal buffer used by the XQA kernel. Needs to be initialized to 0.
        max_num_seqs = scheduler_config.max_num_seqs
        self.multi_block_semaphores = torch.zeros(
            num_heads * max_num_seqs,
            device=torch.device("cuda"),
            dtype=torch.int32,
            requires_grad=False,
        ).contiguous()

        # Create a representation of the fixed parameters of the attention operation.
        self.op = create_op(
            inputDataType=DeviceDataType.BF16,
            outputDataType=kv_cache_device_data_type,
            attentionLayerDimensions=self.attention_layer_dimensions,
            prefixCacheConfiguration=prefix_cache_configuration,
            qScaling=
            1.0,  # TODO: seems to be 1.0 most of the time, still, set correctly ultimately.
            maxAttentionWindowSize=rotary_embedding_config.
            max_positions,  # TODO: set correctly.
            cyclicAttentionWindowSize=rotary_embedding_config.
            max_positions,  # TODO: set correctly.
            outputScalingFactor=self.output_scaling_factor,
            kvCacheDequantizationFactor=self.kv_cache_dequantization_factor,
            multiBlockSemaphores=self.multi_block_semaphores,
            enableMultiTokenGeneration=speculative_decoding_config is not None,
            xqaEnabled=self.xqa_enabled,
        )

        # The size in bytes of the workspace needed by FMHA and XQA.
        max_num_tokens = scheduler_config.max_num_batched_tokens
        workspace_size = calculate_workspace_size(
            self.op,
            max_num_tokens,
            max_num_seqs,
        )

        # Allocate the workspace.
        self.workspace = torch.zeros(
            workspace_size,
            device=torch.device("cuda"),
            dtype=torch.int8,
            requires_grad=False,
        ).contiguous()

        self.scale = scale
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = float(scale)
        self.alibi_slopes = None
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention is not implemented for TKE.")

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TkeMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with TensorRT-LLM Kernel Export.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: should be None for this backend, unused, qkv is passed as a single tensor as query.
            value: should be None for this backend, unused, qkv is passed as a single tensor as query.
            kv_cache = [num_blocks, 2, num_kv_heads, block_size, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            return output

        # NOTE: we have removed all calls to tensor slicing and viewing from this function intentionally, at the cost of making the API of TKE a bit more complex.
        cuda_stream = current_stream()
        if attn_metadata.num_context_sequences > 0:
            run_context_inplace(
                op=self.op,
                numContextSequences=attn_metadata.num_context_sequences,
                numContextTokens=attn_metadata.num_context_tokens,
                maxSequenceLength=attn_metadata.context_max_sequence_length,
                tokenOffset=attn_metadata.
                num_generation_tokens,  # Generation or decode tokens come first in the batch.
                qkv=query,
                sequenceLengthsDevice=attn_metadata.
                context_sequence_lengths_device,
                inputSequenceLengthsDevice=attn_metadata.
                context_input_sequence_lengths_device,
                kvCacheBlockOffsets=attn_metadata.context_block_table_tensor,
                kvCachePoolPtr=kv_cache.data_ptr(),
                rotaryEmbedding=self.rotary_embedding,
                outputPtr=output.data_ptr(),
                workspace=self.workspace,
                stream=cuda_stream.cuda_stream,
            )

        if attn_metadata.num_generation_sequences > 0:
            run_generation_inplace(
                op=self.op,
                numGenerationSequences=attn_metadata.num_generation_sequences,
                numGenerationTokens=attn_metadata.num_generation_tokens,
                maxSequenceLength=attn_metadata.generation_max_sequence_length,
                maxInputSequenceLength=attn_metadata.
                generation_max_input_sequence_length,
                tokenOffset=
                0,  # Generation or decode tokens come first in the batch.
                qkv=query,
                sequenceLengthsDevice=attn_metadata.common_attn_metadata.
                seq_lens,
                inputSequenceLengthsDevice=attn_metadata.query_lens,
                kvCacheBlockOffsets=attn_metadata.common_attn_metadata.
                block_table_tensor,
                kvCachePoolPtr=kv_cache.data_ptr(),
                rotaryEmbedding=self.rotary_embedding,
                outputPtr=output.data_ptr(),
                workspace=self.workspace,
                stream=cuda_stream.cuda_stream,
            )

        return output


def apply_llama3_scaling(inv_freqs: np.ndarray, rope_scaling_config: dict):
    scale_factor = rope_scaling_config.get("factor", 8.0)
    low_freq_factor = rope_scaling_config.get("low_freq_factor", 1.0)
    high_freq_factor = rope_scaling_config.get("high_freq_factor", 4.0)
    old_context_len = rope_scaling_config.get(
        "original_max_position_embeddings", 8192)

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
            smooth = (old_context_len / wavelen -
                      low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_inv_freqs.append((1 - smooth) * inv_freq / scale_factor +
                                 smooth * inv_freq)
    return np.array(new_inv_freqs, dtype=inv_freqs.dtype)


def create_sinusoidal_positions_for_attention_plugin(
    num_pos: int,
    dim: int,
    theta: float,
    scale: float,
    scale_type: RotaryScalingType,
    # Other scaling configs that only used by certain scaling types.
    rope_scaling_config: Optional[dict] = None,
    dtype=np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    if scale_type == RotaryScalingType.LINEAR:
        scale = 1.0 / scale
    if scale_type == RotaryScalingType.LLAMA3:
        assert rope_scaling_config is not None, (
            "rotary_scaling config must be provided.")
        inv_freq = 1.0 / (theta**(np.arange(0, dim, 2) / dim)).astype(dtype)
        inv_freq = apply_llama3_scaling(inv_freq, rope_scaling_config)
    else:
        inv_freq = scale / (theta**(np.arange(0, dim, 2) / dim)).astype(dtype)
    sinusoid_inp = np.expand_dims(
        np.einsum(
            "i , j -> i j",
            np.arange(num_pos, dtype=dtype),
            inv_freq,
            dtype=dtype,
        ),
        axis=-1,
    )
    # fuse cos/sin into float2 (cos, sin).
    concat = np.concatenate(
        (np.cos(sinusoid_inp), np.sin(sinusoid_inp)),
        axis=-1)  # np.cos(sinusoid_inp).shape = (32768, 64, 1)

    return inv_freq, concat.astype(dtype).reshape(num_pos, dim)
