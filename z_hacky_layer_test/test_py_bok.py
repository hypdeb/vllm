from dataclasses import dataclass
import dataclasses
from typing import Generator
import pytest
import torch

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
)

from utils import (
    generate_constant_attention_input,
    identity_rotary_cos_sin,
    pack_attention_input,
)


@dataclass(frozen=True)
class ContextRequest:
    sequence_length: int


@dataclass(frozen=True)
class GenerationRequest:
    sequence_length: int


type Request = ContextRequest | GenerationRequest


@dataclass(frozen=True)
class ForwardInplaceTestCase:
    num_layers: int
    max_batch_size: int
    max_num_tokens: int

    attention_layer_dimensions: AttentionLayerDimensions

    rotary_embedding_dim: int
    rotary_embedding_base: int
    rotary_embedding_max_positions: int

    max_attention_window_size: int

    num_tokens_per_block: int
    max_num_blocks_per_sequence: int

    requests: tuple[Request, ...]

    output_scaling_factor: float


_base_test_cases = [
    ForwardInplaceTestCase(
        num_layers=1,
        max_batch_size=64,
        max_num_tokens=(1 << 14),
        attention_layer_dimensions=AttentionLayerDimensions(),
        rotary_embedding_dim=128,
        rotary_embedding_base=10000,
        rotary_embedding_max_positions=2048,
        max_attention_window_size=(1 << 15),
        num_tokens_per_block=32,
        max_num_blocks_per_sequence=512,
        requests=(ContextRequest(sequence_length=1024),),
        output_scaling_factor=1.0,
    ),
    ForwardInplaceTestCase(
        num_layers=1,
        max_batch_size=64,
        max_num_tokens=(1 << 14),
        attention_layer_dimensions=AttentionLayerDimensions(),
        rotary_embedding_dim=128,
        rotary_embedding_base=10000,
        rotary_embedding_max_positions=2048,
        max_attention_window_size=(1 << 15),
        num_tokens_per_block=32,
        max_num_blocks_per_sequence=512,
        requests=(
            ContextRequest(sequence_length=1024),
            ContextRequest(sequence_length=1024),
        ),
        output_scaling_factor=1.0,
    ),
    ForwardInplaceTestCase(
        num_layers=1,
        max_batch_size=64,
        max_num_tokens=(1 << 14),
        attention_layer_dimensions=AttentionLayerDimensions(),
        rotary_embedding_dim=128,
        rotary_embedding_base=10000,
        rotary_embedding_max_positions=2048,
        max_attention_window_size=(1 << 15),
        num_tokens_per_block=32,
        max_num_blocks_per_sequence=512,
        requests=(GenerationRequest(sequence_length=1024),),
        output_scaling_factor=1.0,
    ),
    ForwardInplaceTestCase(
        num_layers=1,
        max_batch_size=64,
        max_num_tokens=(1 << 14),
        attention_layer_dimensions=AttentionLayerDimensions(),
        rotary_embedding_dim=128,
        rotary_embedding_base=10000,
        rotary_embedding_max_positions=2048,
        max_attention_window_size=(1 << 15),
        num_tokens_per_block=32,
        max_num_blocks_per_sequence=512,
        requests=(
            GenerationRequest(sequence_length=1024),
            GenerationRequest(sequence_length=1024),
        ),
        output_scaling_factor=1.0,
    ),
    ForwardInplaceTestCase(
        num_layers=1,
        max_batch_size=64,
        max_num_tokens=(1 << 14),
        attention_layer_dimensions=AttentionLayerDimensions(),
        rotary_embedding_dim=128,
        rotary_embedding_base=10000,
        rotary_embedding_max_positions=2048,
        max_attention_window_size=(1 << 15),
        num_tokens_per_block=32,
        max_num_blocks_per_sequence=512,
        requests=(
            ContextRequest(sequence_length=1024),
            GenerationRequest(sequence_length=1024),
        ),
        output_scaling_factor=1.0,
    ),
    ForwardInplaceTestCase(
        num_layers=1,
        max_batch_size=64,
        max_num_tokens=(1 << 14),
        attention_layer_dimensions=AttentionLayerDimensions(),
        rotary_embedding_dim=128,
        rotary_embedding_base=10000,
        rotary_embedding_max_positions=2048,
        max_attention_window_size=(1 << 15),
        num_tokens_per_block=32,
        max_num_blocks_per_sequence=512,
        requests=(
            ContextRequest(sequence_length=128),
            ContextRequest(sequence_length=1024),
            GenerationRequest(sequence_length=1024),
            GenerationRequest(sequence_length=512),
        ),
        output_scaling_factor=1.0,
    ),
]


def _attention_layer_dimensions() -> Generator[AttentionLayerDimensions, None, None]:
    small = AttentionLayerDimensions()
    small.numQHeads = 32
    small.numKVHeads = 4
    small.headSize = 64
    yield small
    mistral_large = AttentionLayerDimensions()
    mistral_large.numQHeads = 96
    mistral_large.numKVHeads = 8
    mistral_large.headSize = 128
    yield mistral_large


def _num_tokens_per_block() -> Generator[int, None, None]:
    yield 32


def _max_num_blocks_per_sequence() -> Generator[int, None, None]:
    yield 512


def _num_layers() -> Generator[int, None, None]:
    yield 1
    yield 2
    yield 4


test_cases = [
    dataclasses.replace(
        test_case,
        attention_layer_dimensions=attention_layer_dimensions,
        num_tokens_per_block=num_tokens_per_block,
        max_num_blocks_per_sequence=max_num_blocks_per_sequence,
        num_layers=num_layers,
    )
    for test_case in _base_test_cases
    for attention_layer_dimensions in _attention_layer_dimensions()
    for num_tokens_per_block in _num_tokens_per_block()
    for max_num_blocks_per_sequence in _max_num_blocks_per_sequence()
    for num_layers in _num_layers()
]


def _input_sequence_length(request: Request) -> int:
    match request:
        case ContextRequest(sequence_length=sequence_length):
            return sequence_length
        case GenerationRequest(sequence_length=sequence_length):
            return 1
        case _:
            raise ValueError(f"Invalid request: {request}")


def _sequence_length(request: Request) -> int:
    match request:
        case ContextRequest(sequence_length=sequence_length):
            return sequence_length
        case GenerationRequest(sequence_length=sequence_length):
            return sequence_length
        case _:
            raise ValueError(f"Invalid request: {request}")


@pytest.mark.parametrize("test_case", test_cases)
def test_forward_inplace(test_case: ForwardInplaceTestCase):
    stream = torch.cuda.current_stream()

    rotary_embedding = RotaryEmbedding()
    rotary_embedding.type = RotaryPositionalEmbeddingType.GPT_NEOX
    rotary_embedding.rotaryEmbeddingDim = test_case.rotary_embedding_dim
    rotary_embedding.rotaryEmbeddingBase = test_case.rotary_embedding_base
    rotary_embedding.rotaryEmbeddingScale = 0
    rotary_embedding.rotaryEmbeddingMaxPositions = (
        test_case.rotary_embedding_max_positions
    )
    rotary_embedding.rotaryScalingType = RotaryScalingType.NONE

    prefix_cache_configuration = PrefixCacheConfiguration()
    prefix_cache_configuration.numTokensPerBlock = test_case.num_tokens_per_block
    prefix_cache_configuration.maxNumBlocksPerSequence = (
        test_case.max_num_blocks_per_sequence
    )
    prefix_cache_configuration.dataType = DeviceDataType.FP8_E4M3

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
        test_case.max_batch_size * test_case.attention_layer_dimensions.numQHeads,
        device=torch.device("cuda"),
        dtype=torch.int32,
    )

    # Create a representation of the fixed parameters of the attention operation.
    op = create_op(
        inputDataType=DeviceDataType.BF16,
        outputDataType=DeviceDataType.FP8_E4M3,
        attentionLayerDimensions=test_case.attention_layer_dimensions,
        rotaryEmbedding=rotary_embedding,
        prefixCacheConfiguration=prefix_cache_configuration,
        qScaling=1.0,
        maxAttentionWindowSize=test_case.max_attention_window_size,
        cyclicAttentionWindowSize=test_case.max_attention_window_size,
        fp8OutputScaling=fp8_output_scaling,
        kvScaleOrigQuant=kv_scale_orig_quant,
        kvScaleQuantOrig=kv_scale_quant_orig,
        multiBlockSemaphores=multi_block_semaphores,
    )

    # Calculate how much workspace memory is needed for the operation.
    workspace_size = calculate_workspace_size(
        op, test_case.max_num_tokens, test_case.max_batch_size
    )
    workspace = torch.zeros(
        workspace_size, device=torch.device("cuda"), dtype=torch.int8
    )

    q_dimension = (
        test_case.attention_layer_dimensions.numQHeads
        * test_case.attention_layer_dimensions.headSize
    )

    sequence_lengths_host = torch.tensor(
        [_sequence_length(request) for request in test_case.requests],
        dtype=torch.int32,
    )

    # Create a representation of the rotary positional embedding cosine and sine cache.
    rotary_cos_sin = identity_rotary_cos_sin(rotary_embedding)

    output_scaling_factor = torch.tensor(
        [test_case.output_scaling_factor],
        device=torch.device("cuda"),
        dtype=torch.float32,
    )

    num_blocks_in_kv_cache = (
        test_case.max_batch_size * prefix_cache_configuration.maxNumBlocksPerSequence
    )
    actual_kv_cache_pools = [
        torch.zeros(
            num_blocks_in_kv_cache,
            2,
            test_case.attention_layer_dimensions.numKVHeads,
            prefix_cache_configuration.numTokensPerBlock,
            test_case.attention_layer_dimensions.headSize,
            device=torch.device("cuda"),
            dtype=torch.float8_e4m3fn,
        )
        for _ in range(test_case.num_layers)
    ]

    input_sequence_lengths_host = torch.tensor(
        [_input_sequence_length(request) for request in test_case.requests],
        dtype=torch.int32,
    )
    input_sequence_lengths_device = input_sequence_lengths_host.cuda()
    num_tokens = input_sequence_lengths_host.sum().item()
    sequence_lengths_device = sequence_lengths_host.cuda()

    num_context_requests = sum(
        isinstance(request, ContextRequest) for request in test_case.requests
    )

    # Rough simulation of a realistic workload where the operation would be called for each layer.
    for layer_index in range(test_case.num_layers):

        (q, k, v) = generate_constant_attention_input(
            num_tokens,
            test_case.attention_layer_dimensions,
            DeviceDataType.BF16,
            1.0,
        )

        qkv = pack_attention_input(q, k, v)

        # TODO: it will be a bit more complicated here to set the state of the KV-cache correctly.
        # kv_cache_block_offsets = write_kv_cache_at_contiguous_offsets(
        #     k.to(torch.float8_e4m3fn),
        #     v.to(torch.float8_e4m3fn),
        #     sequence_lengths_device,
        #     actual_kv_cache_pools[layer_index],
        #     test_case.max_num_blocks_per_sequence,
        # )
        num_sequences = input_sequence_lengths_host.shape[0]
        num_blocks_required = num_sequences * test_case.max_num_blocks_per_sequence
        kv_cache_block_offsets = torch.arange(
            num_blocks_required,
            device=torch.device("cuda"),
            dtype=torch.int32,
        ).reshape(num_sequences, test_case.max_num_blocks_per_sequence)

        # Run the attention operation.
        output = torch.zeros(
            num_tokens,
            q_dimension,
            device=torch.device("cuda"),
            dtype=torch.int8,
        )
        stream.synchronize()

        # Debug prints to understand parameter types and shapes
        qkv = qkv[0]
        print(f"DEBUG forward_inplace parameters (test file):")
        print(f"  op type: {type(op)}")
        print(f"  qkv type: {type(qkv)}, shape: {qkv.shape}, dtype: {qkv.dtype}, device: {qkv.device}")
        print(f"  num_context_requests type: {type(num_context_requests)}, value: {num_context_requests}")
        
        input_seq_lens_host_uint32 = input_sequence_lengths_host.to(torch.uint32)
        input_seq_lens_device_uint32 = input_sequence_lengths_device.to(torch.uint32)
        seq_lens_device_uint32 = sequence_lengths_device.to(torch.uint32)
        seq_lens_host_uint32 = sequence_lengths_host.to(torch.uint32)
        kv_cache_offsets_uint32 = kv_cache_block_offsets.to(torch.uint32)
        
        print(f"  input_seq_lens_host_uint32 type: {type(input_seq_lens_host_uint32)}, shape: {input_seq_lens_host_uint32.shape}, dtype: {input_seq_lens_host_uint32.dtype}, device: {input_seq_lens_host_uint32.device}")
        print(f"  input_seq_lens_device_uint32 type: {type(input_seq_lens_device_uint32)}, shape: {input_seq_lens_device_uint32.shape}, dtype: {input_seq_lens_device_uint32.dtype}, device: {input_seq_lens_device_uint32.device}")
        print(f"  seq_lens_device_uint32 type: {type(seq_lens_device_uint32)}, shape: {seq_lens_device_uint32.shape}, dtype: {seq_lens_device_uint32.dtype}, device: {seq_lens_device_uint32.device}")
        print(f"  seq_lens_host_uint32 type: {type(seq_lens_host_uint32)}, shape: {seq_lens_host_uint32.shape}, dtype: {seq_lens_host_uint32.dtype}, device: {seq_lens_host_uint32.device}")
        print(f"  kv_cache_offsets_uint32 type: {type(kv_cache_offsets_uint32)}, shape: {kv_cache_offsets_uint32.shape}, dtype: {kv_cache_offsets_uint32.dtype}, device: {kv_cache_offsets_uint32.device}")
        
        kv_cache_data_ptr = actual_kv_cache_pools[layer_index].data_ptr()
        print(f"  kv_cache_data_ptr type: {type(kv_cache_data_ptr)}, value: {kv_cache_data_ptr}")
        print(f"  output_scaling_factor type: {type(output_scaling_factor)}, shape: {output_scaling_factor.shape}, dtype: {output_scaling_factor.dtype}, device: {output_scaling_factor.device}")
        print(f"  rotary_cos_sin type: {type(rotary_cos_sin)}, shape: {rotary_cos_sin.shape}, dtype: {rotary_cos_sin.dtype}, device: {rotary_cos_sin.device}")
        print(f"  output type: {type(output)}, shape: {output.shape}, dtype: {output.dtype}, device: {output.device}")
        print(f"  workspace type: {type(workspace)}, shape: {workspace.shape}, dtype: {workspace.dtype}, device: {workspace.device}")
        
        cuda_stream = stream.cuda_stream
        print(f"  cuda_stream type: {type(cuda_stream)}, value: {cuda_stream}")
        
        print(f"Test file parameters summary:")
        print(f"  Using concatenated QKV tensor: {qkv.shape}")
        print(f"  Layer index: {layer_index}")
        print(f"  Num tokens: {num_tokens}")
        print(f"  Num sequences: {len(test_case.requests)}")

        forward_inplace(
            op,
            qkv,
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
        stream.synchronize()
