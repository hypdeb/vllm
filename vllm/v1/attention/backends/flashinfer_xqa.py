# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashInfer, with XQA kernels for Hopper."""

from typing import List, Optional
import torch

from vllm.attention.backends.abstract import AttentionBackend, AttentionImpl, AttentionLayer, AttentionMetadata, AttentionType
from vllm.config.cache import CacheConfig
from vllm.v1.attention.backends.utils import AttentionMetadataBuilder

from flashinfer.xqa import xqa

class FlashInferXQABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_XQA_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["FlashInferXQAImpl"]:
        return FlashInferXQAImpl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        return FlashInferXQAMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        return FlashInferXQAMetadataBuilder

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
        return (0, 1, 3, 2, 4)

class FlashInferXQAMetadata(AttentionMetadata):
    pass

class FlashInferXQAMetadataBuilder(
        AttentionMetadataBuilder[FlashInferXQAMetadata]):
    pass

class FlashInferXQAImpl(AttentionImpl):

    use_fp16: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        cache_config: Optional[CacheConfig] = None,
    ):
        if cache_config is None:
            raise ValueError("The FlashInfer XQA backend needs the configuration of the KV cache.")
        self.num_tokens_per_block = cache_config.block_size

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferXQAMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        assert output is not None, "Output tensor must be provided."

        # This happens during profiling.
        if attn_metadata is None:
            return output

        if attn_metadata.num_context_sequences > 0:
            # TODO: Run an appropriate context optimized kernel
            pass

        if attn_metadata.num_generation_sequences > 0:
            # TODO: Run XQA
            xqa(
                use_fp16=self.use_fp16,
                token_per_page=self.num_tokens_per_block
            )

        return output