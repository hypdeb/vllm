import math
from vllm import LLM
import argparse
import torch
import torch.distributed as dist
import os
from vllm.sampling_params import SamplingParams
from datetime import datetime
import uuid

# Set environment variable for insecure serialization
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

BACKEND_DIR_FORMAT = "/scratch/usr/z_hacky_layer_test/captures/{output_dir}"
VARIANT_DIR_FORMAT = BACKEND_DIR_FORMAT + "/{variant}"

SEQLEN_DIR_FORMAT = VARIANT_DIR_FORMAT + "/seq_len_{seq_len}"
PASS_DIR_FORMAT = SEQLEN_DIR_FORMAT + "/pass_{current_pass_count}"

LOG_PRE_FORWARD_HOOK = False
LOG_KV_CACHE = False

# Use a class to ensure the same object reference is maintained
class GlobalState:
    def __init__(self):
        self.sequence_length = 0
        self.variant = None
        self.chunk_idx_per_layer = [0] * 1024
        self.output_dir = None

GLOBAL_STATE = GlobalState()


def multiply_1000(tensor):
    tensor = tensor * 1000
    tensor = tensor / 1000
    return tensor.to(torch.bfloat16)


def identity(tensor):
    return tensor


def fake_fp8(tensor):
    return (tensor.to(torch.float8_e4m3fn)).to(torch.bfloat16)


VARIANT_TO_ERROR_FN = {
    "default": identity,
    "multiply_1000": multiply_1000,
    "fake_fp8": fake_fp8,
    "true_fp8": identity,
}


def attach_hooks(self):
    # Initialize state on self if not exists
    if not hasattr(self, '_cache_layer_state'):
        self._cache_layer_state = GlobalState()
    
    layers = self.model_runner.model.language_model.model.layers
    num_layers = len(layers)

    def get_input_fn_hooks():
        def pre_forward_hook(module, input_args):
            error_fn = VARIANT_TO_ERROR_FN[self._cache_layer_state.variant]
            query, key, value = input_args[0], input_args[1], input_args[2]

            query = error_fn(query)

            if key is not None:
                key = error_fn(key)
            if value is not None:
                value = error_fn(value)

            return (query, key, value)

        return pre_forward_hook

    for layer in layers:
        layer.self_attn.attn.register_forward_pre_hook(get_input_fn_hooks())

    first_layer = layers[0]
    first_attn: torch.nn.Module = first_layer.self_attn.attn
    last_layer_idx = num_layers - 1
    last_layer = layers[last_layer_idx]
    last_attn: torch.nn.Module = last_layer.self_attn.attn

    def get_output_saving_hooks(layer_idx: int):
        def output_saving_hook(module, input_args, output_tensor):
            # Only run on rank 0
            rank = dist.get_rank() if dist.is_initialized() else 0
            
            state = self._cache_layer_state
            current_chunk_idx = state.chunk_idx_per_layer[layer_idx]
            last_chunk_index = math.ceil(state.sequence_length / 16384) - 1
            
            # Increment chunk index for next call
            state.chunk_idx_per_layer[layer_idx] += 1
            
            if current_chunk_idx != last_chunk_index:
                return

            seq_len = state.sequence_length
            file_name = f"{state.variant}_{seq_len}_{layer_idx}_{rank}_out.pt"
            output_path = os.path.join(state.output_dir, file_name)

            torch.save(output_tensor.detach().cpu()[-1], output_path)

        return output_saving_hook

    first_attn.register_forward_hook(get_output_saving_hooks(0))
    last_attn.register_forward_hook(get_output_saving_hooks(last_layer_idx))


def run_for_variant(
    model: str,
    tokenizer: str,
    variant: str,
    output_dir: str,
    start_seq_len: int,
    end_seq_len: int,
    step_seq_len: int,
):
    match variant:
        case "default":
            dtype = "auto"
        case "true_fp8":
            dtype = "fp8"
        case "fake_fp8":
            dtype = "auto"
        case _:
            raise ValueError(f"Unknown variant: {variant}")

    # Initialize model
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        enforce_eager=True,
        gpu_memory_utilization=0.9,
        kv_cache_dtype=dtype,
        enable_prefix_caching=False,
        tensor_parallel_size=4,
        config_format="mistral",
        load_format="mistral",
        tokenizer_mode="mistral",
        block_size=32,
    )
    
    def get_set_sequence_length_callback(seq_len: int, var: str, out_dir: str):
        def set_sequence_length(self):
            # Initialize state on self if not exists
            if not hasattr(self, '_cache_layer_state'):
                self._cache_layer_state = GlobalState()
            
            # Mutate the state object's attributes
            state = self._cache_layer_state
            state.sequence_length = seq_len
            state.chunk_idx_per_layer[:] = [0] * len(state.chunk_idx_per_layer)
            state.variant = var
            state.output_dir = out_dir

        return set_sequence_length
        
    llm.collective_rpc(get_set_sequence_length_callback(start_seq_len, variant, output_dir))
    llm.collective_rpc(attach_hooks)

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        top_k=1,
    )

    with open("/scratch/usr/michelangelo.txt") as file:
        book = file.read()

    tokenized_book_full = llm.get_tokenizer().encode(book)

    for seq_len in range(start_seq_len, end_seq_len + 1, step_seq_len):
        llm.collective_rpc(get_set_sequence_length_callback(seq_len, variant, output_dir))

        # Truncate to a reasonable length if needed
        tokenized_book = tokenized_book_full[:seq_len]

        detokenized_book = llm.get_tokenizer().decode(tokenized_book)
        _ = llm.generate(detokenized_book, sampling_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract first LlamaAttention layer"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        required=False,
        default=None,
        help="Tokenizer name or path",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--start-seq-len", type=int, default=2000, help="Start sequence length"
    )
    parser.add_argument(
        "--end-seq-len", type=int, default=128000, help="End sequence length"
    )
    parser.add_argument(
        "--step-seq-len", type=int, default=1000, help="Step sequence length"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="output directory",
    )
    args = parser.parse_args()

    # Create run ID with current datetime and 4 characters from GUID
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:4]
    )
    print(f"Run ID: {run_id}")

    output_dir = os.path.join(args.output_dir, run_id)
    print("Saving results to: " + output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for variant in ("default", "true_fp8", "fake_fp8"):
        GLOBAL_STATE.variant = variant
        GLOBAL_STATE.output_dir = output_dir
        GLOBAL_STATE.chunk_idx_per_layer[:] = [0] * len(GLOBAL_STATE.chunk_idx_per_layer)
        GLOBAL_STATE.sequence_length = args.start_seq_len
        run_for_variant(
            args.model,
            args.tokenizer,
            variant,
            output_dir,
            args.start_seq_len,
            args.end_seq_len,
            args.step_seq_len,
        )