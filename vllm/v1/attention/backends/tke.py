# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashInfer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from py_tke import (
    AttentionLayerDimensions,
    BlockOffsetLayout,
    DeviceDataType,
    InputScales,
    PrefixCacheConfiguration,
    RotaryEmbedding,
    RotaryPositionalEmbeddingType,
    RotaryScalingType,
    calculate_workspace_size,
    create_op,
    run_context_inplace,
    run_generation_inplace,
)

from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
    InputLayout,
)
from vllm.config import SpeculativeConfig, VllmConfig
from vllm.config.cache import CacheConfig, CacheDType
from vllm.config.scheduler import SchedulerConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding.abstractions import (
    RotaryEmbeddingConfig,
    VanillaRotaryEmbeddingConfig,
    YarnRotaryEmbeddingConfig,
)
from vllm.model_executor.layers.rotary_embedding.common import yarn_get_mscale
from vllm.utils.torch_utils import current_stream
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# The RoPE coefficients are read-only and identical given identical
# configurations. We cache them to reuse them across layers to save memory.
_ROPE_COEFF_CACHE: dict[RotaryEmbeddingConfig, torch.Tensor] = {}

# The kernel workspace is a slice of memory used in turns by each layer.
# The kernels should make no assumption about the content of this
# memory, therefore, we can reuse them across layers.
# We only allocate the largest workspace required.
_WORKSPACE: tuple[int, torch.Tensor] | None = None

# Unlike the workspace, the semaphores should be zero on kernel launch.
# Hence, we manage them separately, but we can still share them across
# layers, as the kernels do zero them out after execution.
_MULTI_BLOCK_SEMAPHORES: dict[tuple, torch.Tensor] = {}


class TKEAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128, 256]

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        if kv_cache_dtype.startswith("fp8"):
            return True
        return kv_cache_dtype in ["auto"]

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
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order() -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets us from
        # `get_kv_cache_shape` to the actual memory layout we want.
        # NOTE: 3 and 2 are swapped because the TRTLLM layout
        # within blocks is [num_heads, num_tokens, dimension]
        return (0, 1, 3, 2, 4)

    @staticmethod
    def get_output_dtype(kv_cache_dtype: torch.dtype) -> torch.dtype:
        if kv_cache_dtype == torch.float8_e4m3fn:
            return torch.float8_e4m3fn
        return torch.bfloat16

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

    # Pre-computed input sequence lengths for
    # the prefill/context sequences in the batch.
    context_input_sequence_lengths_device: torch.Tensor

    # Pre-computed block table tensor for the
    # prefill/context sequences in the batch.
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

    # vLLM uses uint8 to represent fp8 as kv-cache datatype sometimes.
    # NOTE: if we ever need a real uint8 kv-cache, this will need addressing.
    if dtype == torch.uint8:
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

    # This should resolve to a real data type earlier, but somehow it doesn't.
    # We make the assumption that 'auto' will be BF16.
    if dtype == "auto":
        return DeviceDataType.BF16
    raise RuntimeError(f"Unsupported dtype: {dtype}")


class TkeMetadataBuilder(AttentionMetadataBuilder[TkeMetadata]):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        # For this backend, the data type of the kv-cache determines
        # the data type in which we perform operation.
        self.kv_cache_device_data_type = _torch_to_device_data_type(kv_cache_spec.dtype)

        self.xqa_enabled = self.kv_cache_device_data_type == DeviceDataType.FP8_E4M3

        # When doing speculative decoding, consider all input sequences
        # with fewer tokens than the configured number of
        # speculative tokens as "decode" requests and pull them to
        # the front of the batch.
        # Also, we do not need to do any reordering if we do not
        # use generation optimized kernels (XQA) for generation.
        if self.xqa_enabled:
            if (
                vllm_config.speculative_config is not None
                and vllm_config.speculative_config.num_speculative_tokens is not None
            ):
                self._reorder_batch_threshold: int | None = (
                    vllm_config.speculative_config.num_speculative_tokens
                )
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
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(common_attn_metadata)
            )
        else:
            num_decodes = 0
            num_prefills = common_attn_metadata.num_reqs
            num_decode_tokens = 0
            num_prefill_tokens = common_attn_metadata.num_actual_tokens

        query_lens_cpu = (
            common_attn_metadata.query_start_loc_cpu[1:]
            - common_attn_metadata.query_start_loc_cpu[:-1]
        )
        query_lens = (
            common_attn_metadata.query_start_loc[1:]
            - common_attn_metadata.query_start_loc[:-1]
        )

        context_max_sequence_length = (
            common_attn_metadata.seq_lens_cpu[num_decodes:].max().item()
            if num_prefills > 0
            else 0
        )
        generation_max_sequence_length = (
            common_attn_metadata.seq_lens_cpu[:num_decodes].max().item()
            if num_decodes > 0
            else 0
        )
        generation_max_input_sequence_length = (
            query_lens_cpu[:num_decodes].max().item() if num_decodes > 0 else 0
        )

        return TkeMetadata(
            common_attn_metadata=common_attn_metadata,
            context_sequence_lengths_device=common_attn_metadata.seq_lens[num_decodes:],
            context_input_sequence_lengths_device=query_lens[num_decodes:],
            context_block_table_tensor=common_attn_metadata.block_table_tensor[
                num_decodes:
            ],
            query_lens=query_lens,
            context_max_sequence_length=int(context_max_sequence_length),
            generation_max_sequence_length=int(generation_max_sequence_length),
            generation_max_input_sequence_length=int(
                generation_max_input_sequence_length
            ),
            num_context_sequences=num_prefills,
            num_context_tokens=num_prefill_tokens,
            num_generation_sequences=num_decodes,
            num_generation_tokens=num_decode_tokens,
        )


class TkeImpl(AttentionImpl):
    rotary_embedding: RotaryEmbedding
    input_scales: InputScales | None = None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        use_irope: bool = False,
        rotary_embedding_config: RotaryEmbeddingConfig | None = None,
        cache_config: CacheConfig | None = None,
        speculative_decoding_config: SpeculativeConfig | None = None,
        scheduler_config: SchedulerConfig | None = None,
    ) -> None:
        # A dummy output scale in case it is not provided to forward().
        self.dummy_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        global _WORKSPACE, _ROPE_COEFF_CACHE, _MULTI_BLOCK_SEMAPHORES

        if rotary_embedding_config is None:
            raise RuntimeError("The TKE backend needs to have a configured RoPE.")

        if cache_config is None:
            raise RuntimeError(
                "The TKE backend needs to know about the cache configuration."
            )

        if scheduler_config is None:
            raise RuntimeError(
                "The TKE backend needs to know about the scheduler configuration."
            )

        kv_cache_device_data_type = _str_to_device_data_type(kv_cache_dtype)

        # FIXME: handle this in TKE instead.
        if kv_cache_device_data_type == DeviceDataType.BF16:
            self.input_scales = InputScales()
            self.input_scales.qScale = 1.0
            self.input_scales.qScaleDevice = self.dummy_scale.data_ptr()
            self.input_scales.kScale = 1.0
            self.input_scales.kScaleDevice = self.dummy_scale.data_ptr()
            self.input_scales.vScale = 1.0
            self.input_scales.vScaleDevice = self.dummy_scale.data_ptr()

        # Extract the dimensions of the attention layer.
        self.attention_layer_dimensions = AttentionLayerDimensions()
        self.attention_layer_dimensions.numQHeads = num_heads
        self.attention_layer_dimensions.numKVHeads = num_kv_heads
        self.attention_layer_dimensions.headSize = head_size

        self._setup_rope(rotary_embedding_config)

        # Extract the configuration of the KV-cache.
        self.prefix_cache_configuration = PrefixCacheConfiguration()
        self.prefix_cache_configuration.dataType = kv_cache_device_data_type
        self.prefix_cache_configuration.numTokensPerBlock = cache_config.block_size
        self.prefix_cache_configuration.maxNumBlocksPerSequence = (
            self.rotary_embedding.rotaryEmbeddingMaxPositions // cache_config.block_size
        )
        self.prefix_cache_configuration.blockOffsetLayout = BlockOffsetLayout.VLLM

        # Internal quantity used to size the kv-cache TMA descriptor.
        # TODO: calculate this value from the size of the kv-cache.
        # It needs to be large enough that the TMA descriptor can
        # fit the whole kv-cache tensor.
        # I couldn't find a way to access the actual size
        # of the kv-cache at this time. The 'num_blocks'
        # on the kv-cache config is not set at this point.
        self.prefix_cache_configuration.maxNumSequences = 8192

        # NOTE: XQA BF16 is not enabled yet, meaning that for
        # BF16 attention, we always use FMHA_V2.
        self.xqa_enabled = kv_cache_device_data_type == DeviceDataType.FP8_E4M3

        # An internal buffer used by the XQA kernel. Needs to be initialized to 0.
        # This is the reason why it is handled separately,
        # instead of as part of the workspace,
        # as we cannot guarantee the value of anything in the workspace.
        max_num_seqs = scheduler_config.max_num_seqs
        if (num_heads, max_num_seqs) not in _MULTI_BLOCK_SEMAPHORES:
            _MULTI_BLOCK_SEMAPHORES[(num_heads, max_num_seqs)] = torch.zeros(
                num_heads * max_num_seqs,
                device=torch.device("cuda"),
                dtype=torch.int32,
                requires_grad=False,
            ).contiguous()
        self.multi_block_semaphores = _MULTI_BLOCK_SEMAPHORES[(num_heads, max_num_seqs)]

        # Create a representation of the fixed parameters of the attention operation.
        self.op = create_op(
            inputDataType=DeviceDataType.BF16,
            outputDataType=kv_cache_device_data_type,
            attentionLayerDimensions=self.attention_layer_dimensions,
            prefixCacheConfiguration=self.prefix_cache_configuration,
            multiBlockSemaphores=self.multi_block_semaphores,
            enableSpeculativeDecoding=speculative_decoding_config is not None,
            xqaEnabled=self.xqa_enabled,
        )

        # The size in bytes of the workspace needed by FMHA and XQA.
        max_num_tokens = scheduler_config.max_num_batched_tokens
        workspace_size = calculate_workspace_size(
            self.op,
            max_num_tokens,
            max_num_seqs,
        )
        if _WORKSPACE is None or _WORKSPACE[0] < workspace_size:
            _WORKSPACE = (
                workspace_size,
                torch.zeros(
                    workspace_size,
                    device=torch.device("cuda"),
                    dtype=torch.int8,
                    requires_grad=False,
                ).contiguous(),
            )
        self.workspace = _WORKSPACE[1]

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
                "Encoder self-attention is not implemented for TKE."
            )

    def _setup_rope(self, rotary_embedding_config: RotaryEmbeddingConfig) -> None:
        # Create cache key for rotary cos/sin tensor.
        # Crucially, frozen dataclass, which is hashable.
        cache_key = rotary_embedding_config
        self.rotary_embedding = RotaryEmbedding()

        match rotary_embedding_config:
            case VanillaRotaryEmbeddingConfig() as simple_rope:
                self.rotary_embedding.rotaryEmbeddingBase = simple_rope.base
                self.rotary_embedding.rotaryEmbeddingMaxPositions = (
                    simple_rope.max_positions
                )
                self.rotary_embedding.rotaryEmbeddingDim = simple_rope.dimension
                self.rotary_embedding.rotaryEmbeddingScale = 1.0
                self.rotary_embedding.rotaryScalingType = RotaryScalingType.NONE
                self.rotary_embedding.type = RotaryPositionalEmbeddingType.GPT_NEOX

                if cache_key in _ROPE_COEFF_CACHE:
                    self.rotary_cos_sin = _ROPE_COEFF_CACHE[cache_key]
                else:
                    self.rotary_cos_sin_ndarray = (
                        create_rope_coefficient_cache_for_simple_rope(
                            self.rotary_embedding.rotaryEmbeddingMaxPositions,
                            self.rotary_embedding.rotaryEmbeddingDim,
                            self.rotary_embedding.rotaryEmbeddingBase,
                        )
                    )
                    self.rotary_cos_sin = torch.tensor(
                        self.rotary_cos_sin_ndarray,
                        dtype=torch.float32,
                        device="cuda",
                        requires_grad=False,
                    )
                    _ROPE_COEFF_CACHE[cache_key] = self.rotary_cos_sin
                self.rotary_embedding.rotaryCosSinCache = self.rotary_cos_sin.data_ptr()
            case YarnRotaryEmbeddingConfig() as yarn_rope:
                self.rotary_embedding.rotaryEmbeddingBase = (
                    yarn_rope.rotary_embedding_config.base
                )
                self.rotary_embedding.rotaryEmbeddingMaxPositions = (
                    yarn_rope.yarn_scaling_config.extended_max_positions
                )
                self.rotary_embedding.rotaryEmbeddingDim = (
                    yarn_rope.rotary_embedding_config.dimension
                )
                self.rotary_embedding.rotaryEmbeddingScale = (
                    yarn_rope.yarn_scaling_config.scaling_factor
                )
                self.rotary_embedding.rotaryScalingType = RotaryScalingType.YARN
                self.rotary_embedding.type = RotaryPositionalEmbeddingType.GPT_NEOX

                if cache_key in _ROPE_COEFF_CACHE:
                    self.rotary_cos_sin = _ROPE_COEFF_CACHE[cache_key]
                else:
                    self.rotary_cos_sin_ndarray = create_sinusoidal_positions_yarn(
                        yarn_rope.yarn_scaling_config.extended_max_positions,
                        yarn_rope.rotary_embedding_config.dimension,
                        yarn_rope.rotary_embedding_config.base,
                        yarn_rope.yarn_scaling_config.scaling_factor,
                        original_max_position_embeddings=yarn_rope.rotary_embedding_config.max_positions,
                        beta_fast=yarn_rope.yarn_scaling_config.beta_fast,
                        beta_slow=yarn_rope.yarn_scaling_config.beta_slow,
                    )
                    self.rotary_cos_sin = torch.tensor(
                        self.rotary_cos_sin_ndarray,
                        dtype=torch.float32,
                        device="cuda",
                        requires_grad=False,
                    )
                    _ROPE_COEFF_CACHE[cache_key] = self.rotary_cos_sin
                self.rotary_embedding.rotaryCosSinCache = self.rotary_cos_sin.data_ptr()
            case _:
                raise NotImplementedError(
                    f"The TKE backend does not support RoPE configured "
                    f"with a {type(rotary_embedding_config)}"
                )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TkeMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with TensorRT-LLM Kernel Export.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: should be None for this backend, unused,
            qkv is passed as a single tensor as query.
            value: should be None for this backend, unused,
            qkv is passed as a single tensor as query.
            kv_cache = [num_blocks, 2, num_kv_heads, block_size, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            return output

        # Assuming the input scales are per-layer and don't change.
        if self.input_scales is None:
            self.input_scales = InputScales()
            self.input_scales.qScale = layer._q_scale_float
            self.input_scales.qScaleDevice = layer._q_scale.data_ptr()
            self.input_scales.kScale = layer._k_scale_float
            self.input_scales.kScaleDevice = layer._k_scale.data_ptr()
            self.input_scales.vScale = layer._v_scale_float
            self.input_scales.vScaleDevice = layer._v_scale.data_ptr()

        # NOTE: we have removed all calls to tensor slicing and viewing
        # from this function intentionally, at the cost of
        # making the API of TKE a bit more complex.
        cuda_stream = current_stream()

        if attn_metadata.num_context_sequences > 0:
            run_context_inplace(
                op=self.op,
                numContextSequences=attn_metadata.num_context_sequences,
                numContextTokens=attn_metadata.num_context_tokens,
                maxSequenceLength=attn_metadata.context_max_sequence_length,
                tokenOffset=attn_metadata.num_generation_tokens,
                qkv=query,
                sequenceLengthsDevice=attn_metadata.context_sequence_lengths_device,
                inputSequenceLengthsDevice=attn_metadata.context_input_sequence_lengths_device,
                kvCacheBlockOffsets=attn_metadata.context_block_table_tensor,
                kvCachePoolPtr=kv_cache.data_ptr(),
                rotaryEmbedding=self.rotary_embedding,
                inputScales=self.input_scales,
                outputScalingFactor=self.dummy_scale.data_ptr(),
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
                maxInputSequenceLength=attn_metadata.generation_max_input_sequence_length,
                tokenOffset=0,  # Generation or decode tokens come first in the batch.
                qkv=query,
                sequenceLengthsDevice=attn_metadata.common_attn_metadata.seq_lens,
                inputSequenceLengthsDevice=attn_metadata.query_lens,
                kvCacheBlockOffsets=attn_metadata.common_attn_metadata.block_table_tensor,
                kvCachePoolPtr=kv_cache.data_ptr(),
                rotaryEmbedding=self.rotary_embedding,
                inputScales=self.input_scales,
                outputScalingFactor=self.dummy_scale.data_ptr(),
                outputPtr=output.data_ptr(),
                workspace=self.workspace,
                stream=cuda_stream.cuda_stream,
            )

        return output


def create_rope_coefficient_cache_for_simple_rope(
    num_pos: int,
    dim: int,
    theta: float,
    dtype=np.float32,
) -> np.ndarray:
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2) / dim)).astype(dtype)
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
    concat = np.concatenate((np.cos(sinusoid_inp), np.sin(sinusoid_inp)), axis=-1)

    return concat.astype(dtype).reshape(num_pos, dim)


# Below is copied from TensorRT-LLM and trimmed a bit.
def create_sinusoidal_positions_yarn(
    num_pos: int,
    dim: int,
    base: float = 10000,
    scaling_factor: float = 1.0,
    original_max_position_embeddings: int = 4096,
    beta_fast: float = 32,
    beta_slow: float = 1,
    dtype=torch.float32,
) -> np.ndarray:
    # Copy from https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py
    # Inverse dim formula to find dim based on number of rotations
    def yarn_find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        return (
            dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
        ) / (2 * math.log(base))

    # Find dim range bounds based on rotations
    def yarn_find_correction_range(
        low_rot, high_rot, dim, base, max_position_embeddings
    ):
        low = math.floor(
            yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
        )
        high = math.ceil(
            yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
        )
        if low < 0:
            low = 0
        if high > dim - 1:
            high = dim - 1
        return low, high  # Clamp values just in case

    def yarn_linear_ramp_mask(min, max, dim):
        if min == max:
            max += 0.001  # Prevent singularity

        linear_func = (torch.arange(dim, dtype=dtype, device="cpu") - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=dtype, device="cpu") / dim)
    freq_extra = 1.0 / pos_freqs
    freq_inter = 1.0 / (scaling_factor * pos_freqs)

    low, high = yarn_find_correction_range(
        beta_fast,
        beta_slow,
        dim,
        base,
        original_max_position_embeddings,
    )
    inv_freq_mask = 1 - yarn_linear_ramp_mask(low, high, dim // 2)
    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
    t = torch.arange(num_pos, dtype=dtype, device="cpu")
    sinusoid_inp = torch.einsum("i,j -> ij", t, inv_freq).unsqueeze(-1)

    _mscale = float(yarn_get_mscale(scaling_factor))

    concat = torch.cat(
        (torch.cos(sinusoid_inp) * _mscale, torch.sin(sinusoid_inp) * _mscale), dim=-1
    )
    return concat.reshape((1, -1)).to(dtype).numpy()
