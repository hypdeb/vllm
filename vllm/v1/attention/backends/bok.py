# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashInfer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from flashinfer import (BatchDecodeWithPagedKVCacheWrapper,
                        BatchPrefillWithPagedKVCacheWrapper,
                        MultiLevelCascadeAttentionWrapper)

import vllm.envs as envs
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionType)
from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_attn import use_cascade_attention
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
    AttentionOp
)

USING_BOK = True

BOK_WORKSPACE_BUFFER_SIZE = 256 * 1024 * 1024

logger = init_logger(__name__)


class BokAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "BOK_VLLM_V1"

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
class PerLayerParameters:
    """
    Currently, Bok backend only support models in which all layers share
    the same values for the following hyperparameters.
    """

    window_left: int
    logits_soft_cap: Optional[float]
    sm_scale: float


def get_per_layer_parameters(
        vllm_config: VllmConfig) -> dict[str, PerLayerParameters]:
    """
    Scan all attention layers and determine some hyperparameters
    to use during `plan`.
    """

    layers = get_layers_from_vllm_config(vllm_config, Attention)
    per_layer_params: dict[str, PerLayerParameters] = {}

    for key, layer in layers.items():
        impl = layer.impl
        assert isinstance(impl, BokImpl)

        # Infer hyperparameters from the attention layer
        window_size = impl.sliding_window
        window_left = window_size[0] if window_size is not None else -1
        logits_soft_cap = impl.logits_soft_cap
        sm_scale = impl.scale

        per_layer_params[key] = PerLayerParameters(window_left,
                                                   logits_soft_cap, sm_scale)

    return per_layer_params


def infer_global_hyperparameters(
        per_layer_params: dict[str, PerLayerParameters]) -> PerLayerParameters:
    """
    Currently, Bok backend only support models in which all layers share
    the same values for the following hyperparameters:
    - `window_left`
    - `logits_soft_cap`
    - `sm_scale`

    So this function asserts that all layers share the same values for these
    hyperparameters and returns the global values.
    """

    assert len(per_layer_params) > 0, "No attention layers found in the model."

    param_sets = list(per_layer_params.values())
    global_params = param_sets[0]
    for params in param_sets:
        assert params == global_params, (
            "Bok backend currently only supports models in which all "
            "layers share the same values for the following hyperparameters: "
            "`window_left`, `logits_soft_cap`, `sm_scale`.")

    return global_params


@dataclass
class BokMetadata:
    op: AttentionOp
    attention_layer_dimensions: AttentionLayerDimensions


class BokMetadataBuilder:

    def __init__(self, runner: GPUModelRunner, kv_cache_spec: AttentionSpec,
                 block_table: BlockTable):
        self.runner = runner
        self._workspace_buffer = None

        # Global hyperparameters shared by all attention layers
        self.global_hyperparameters: Optional[PerLayerParameters] = None

        self.vllm_config = runner.vllm_config
        self.kv_cache_spec = kv_cache_spec
        self.block_table = block_table

        self.attention_layer_dimensions = AttentionLayerDimensions()
        self.attention_layer_dimensions.numQHeads = runner.num_query_heads
        self.attention_layer_dimensions.numKVHeads = kv_cache_spec.num_kv_heads
        self.attention_layer_dimensions.headSize = kv_cache_spec.head_size

        # TODO: find exact values.
        rotary_positional_embedding = RotaryEmbedding()
        rotary_positional_embedding.rotaryEmbeddingScale = 1.0
        rotary_positional_embedding.rotaryEmbeddingMaxPositions = 1 << 14
        rotary_positional_embedding.rotaryEmbeddingBase = 210000
        rotary_positional_embedding.rotaryScalingType = RotaryScalingType.NONE
        
        prefix_cache_configuration = PrefixCacheConfiguration()
        prefix_cache_configuration.dataType = DeviceDataType.FP8_E4M3
        prefix_cache_configuration.numTokensPerBlock = 32 # TODO: find exact value.
        prefix_cache_configuration.maxNumBlocksPerSequence = (
            1024 # TODO: find exact value.
        )

        max_attention_window_size = runner.vllm_config.model_config.max_model_len
        print("max_attention_window_size", max_attention_window_size)
        cyclic_attention_window_size = runner.vllm_config.model_config.max_model_len
        print("cyclic_attention_window_size", cyclic_attention_window_size)

        fp8_output_scaling = torch.tensor(
            [1.0], device=torch.device("cuda"), dtype=torch.float32
        )
        kv_scale_orig_quant = torch.tensor(
            [1.0], device=torch.device("cuda"), dtype=torch.float32
        )
        kv_scale_quant_orig = torch.tensor(
            [1.0], device=torch.device("cuda"), dtype=torch.float32
        )
        multi_block_semaphores = torch.zeros(
            runner.num_query_heads * runner.scheduler_config.max_num_seqs, device=torch.device("cuda"), dtype=torch.int32
        )

        # Create a representation of the fixed parameters of the attention operation.
        self.op = create_op(
            inputDataType=DeviceDataType.BF16,
            outputDataType=DeviceDataType.FP8_E4M3,
            attentionLayerDimensions=attention_layer_dimensions,
            rotaryEmbedding=rotary_positional_embedding,
            prefixCacheConfiguration=prefix_cache_configuration,
            qScaling=1.0,
            maxAttentionWindowSize=max_attention_window_size,
            cyclicAttentionWindowSize=cyclic_attention_window_size,
            fp8OutputScaling=fp8_output_scaling,
            kvScaleOrigQuant=kv_scale_orig_quant,
            kvScaleQuantOrig=kv_scale_quant_orig,
            multiBlockSemaphores=multi_block_semaphores,
        )
    def reorder_batch(self, input_batch: InputBatch,
                      scheduler_output: SchedulerOutput) -> bool:
        # We now want to reorder the batch so that the "decode" requests are and
        # the front and the "prefill" requests are at the using the least amount
        # swaps possible. (NOTE for now we loosely use "decode" to mean requests
        # where attention is likely memory-bound and "prefill" to mean requests
        # where attention is likely compute-bound, TODO(lucas): figure out a
        # better naming here)
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

        for i in range(1, min(num_decodes, num_prefills) + 1):
            # If the decode is at the "back" of the batch, i, we can swap it
            # with the prefill closest to the front of the batch
            decode_idx = decodes[num_decodes - i]
            if decode_idx < num_decodes:
                break

            input_batch.swap_states(prefills[i - 1], decode_idx)
            modified_batch = True

        # Save for next `build` call
        # TODO(lucas): this is a bit of a hack, we should probably have a
        # better way of doing this
        self._num_decodes = num_decodes
        self._num_prefills = num_prefills
        self._num_decode_tokens = num_decode_tokens
        self._num_prefill_tokens = num_prefill_tokens

        return modified_batch


    def build(self, num_reqs: int, num_actual_tokens: int, max_query_len: int,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata):
        assert self._num_decodes + self._num_prefills == num_reqs
        assert (self._num_decode_tokens +
                self._num_prefill_tokens == num_actual_tokens)
        page_size = self.kv_cache_spec.block_size
        device = self.runner.device
        qo_indptr = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = self.block_table.get_device_tensor()[:num_reqs]
        slot_mapping = self.block_table.slot_mapping_cpu[:num_actual_tokens].to(
            self.runner.device, non_blocking=True).long()

        block_table_bounds = (seq_lens + page_size - 1) // page_size

        use_cascade = common_prefix_len > 0
        if use_cascade:
            # Grab the blocks of the shared prefix from the first request.
            assert common_prefix_len % page_size == 0
            num_common_kv_blocks = common_prefix_len // page_size
            shared_qo_indptr = torch.tensor([0, num_actual_tokens],
                                            dtype=torch.int32,
                                            device=device)
            shared_kv_page_indptr = torch.tensor([0, num_common_kv_blocks],
                                                 dtype=torch.int32,
                                                 device=device)
            shared_kv_page_indices = block_table_tensor[
                0, :num_common_kv_blocks]
            shared_kv_last_page_len = torch.tensor([page_size],
                                                   dtype=torch.int32,
                                                   device=device)
            # Remove the blocks of the shared prefix from all requests.
            block_table_tensor = block_table_tensor[:, num_common_kv_blocks:]
            block_table_bounds -= num_common_kv_blocks
        else:
            shared_qo_indptr = None
            shared_kv_page_indptr = None
            shared_kv_page_indices = None
            shared_kv_last_page_len = None

        mask = (torch.arange(block_table_tensor.size(1),
                             dtype=block_table_tensor.dtype,
                             device=block_table_tensor.device).unsqueeze(0)
                < block_table_bounds.unsqueeze(1))
        paged_kv_indices = block_table_tensor[mask]

        paged_kv_indptr = torch.cat([
            torch.zeros(1,
                        dtype=block_table_bounds.dtype,
                        device=block_table_bounds.device),
            block_table_bounds.cumsum(dim=0, dtype=torch.int32)
        ])

        paged_kv_last_page_len = seq_lens % page_size
        paged_kv_last_page_len = torch.where(paged_kv_last_page_len == 0,
                                             page_size, paged_kv_last_page_len)

        attn_metadata = BokMetadata(
            op=self.op,
            attention_layer_dimensions=self.attention_layer_dimensions,
            
        )


        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        if self.kv_cache_spec.dtype != self.runner.model_config.dtype:
            # TODO: The cascade wrapper currently does not support setting
            # kv cache dtype to something different from query dtype.
            return False
        return use_cascade_attention(*args, **kwargs)


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
        if use_irope:
            logger.warning_once(
                "Using irope in FlashInfer is not supported yet, it will fall"
                " back to global attention for long context.")
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
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashInferImpl")

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

        # Create qkv tensor and reshape to desired view
        query_ptr = query.data_ptr()
        query_flattened = query.view(-1, self.num_heads * self.head_size)
        key_flattened = key.view(-1, self.num_kv_heads * self.head_size)
        value_flattened = value.view(-1, self.num_kv_heads * self.head_size)

        # Reshape to [num_tokens, (num_heads + 2*num_kv_heads)*head_size]
        qkv = torch.cat([query_flattened, key_flattened, value_flattened], dim=-1)
        
        forward_inplace(
            attn_metadata.op,
            qkv,
            kv_cache,
            num_context_requests,
            input_sequence_lengths_host.to(torch.uint32),
            input_sequence_lengths_device.to(torch.uint32),
            sequence_lengths_device.to(torch.uint32),
            sequence_lengths_host.to(torch.uint32),
            kv_cache_block_offsets.to(torch.uint32),
            actual_kv_cache_pools[layer_index].data_ptr(),
            output_scaling_factor,
            rotary_cos_sin,
            output,
            workspace,
            stream.cuda_stream,
        )
        return output_padded
