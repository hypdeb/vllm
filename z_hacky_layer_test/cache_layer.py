
from vllm import LLM
import argparse
import torch
import os
import shutil
from vllm.sampling_params import SamplingParams

# Set environment variable for insecure serialization
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

BACKEND_DIR_FORMAT = "z_hacky_layer_test/captures/{output_dir}"
VARIANT_DIR_FORMAT = BACKEND_DIR_FORMAT + "/{variant}"

SEQLEN_DIR_FORMAT = VARIANT_DIR_FORMAT + "/seq_len_{seq_len}"
PASS_DIR_FORMAT = SEQLEN_DIR_FORMAT + "/pass_{current_pass_count}"

LOG_PRE_FORWARD_HOOK = False
LOG_KV_CACHE = False

HOOK_COUNTER = [0]

def multiply_1000(tensor):
    tensor = tensor * 1000
    tensor = tensor / 1000
    return tensor.to(torch.bfloat16)

def identity(tensor):
    return tensor

VARIANT_TO_ERROR_FN = {
    "default": identity,
    "multiply_1000": multiply_1000,
}

def main():
    parser = argparse.ArgumentParser(description="Extract first LlamaAttention layer")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path",
    )

    parser.add_argument(
        "--max-tokens", type=int, default=1, help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--enforce-eager", action="store_true", help="Enforce eager execution"
    )
    parser.add_argument(
        "--variant", type=str, default="default", help="variant of inference, eg `downcast` to downcast to fp8 before op"
    )

    parser.add_argument(
        "--start-seq-len", type=int, default=4000, help="Start sequence length"
    )
    parser.add_argument(
        "--end-seq-len", type=int, default=64000, help="End sequence length"
    )
    parser.add_argument(
        "--step-seq-len", type=int, default=10000, help="Step sequence length"
    )
    args = parser.parse_args()

    # Create the output directory name based on the backend
    backend = os.environ['VLLM_ATTENTION_BACKEND']

    # Instantiate a normal vLLM engine
    print(f"Loading model: {args.model}")

    quantization = "modelopt"
    kv_cache_dtype = "fp8"
    # Initialize model
    llm = LLM(
        model=args.model,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        quantization=quantization,
        kv_cache_dtype=kv_cache_dtype,
        max_model_len=131072,
        enable_prefix_caching=False,
        tensor_parallel_size=1,
        block_size=32,
        enable_chunked_prefill=False,
    )

    with open("pg2600.txt", "r") as file:
        book = file.read()

    variant_dir = VARIANT_DIR_FORMAT.format(output_dir=backend,variant=args.variant)
    if os.path.isdir(variant_dir) and not os.path.islink(variant_dir):
        shutil.rmtree(variant_dir)
    elif os.path.exists(variant_dir):
        os.remove(variant_dir)

    def attach_hook(self):
        # print("self.model_runner.model.model.__class__", self.model_runner.model.model.__class__)
        # print("self.model_runner.model.model.layers", self.model_runner.model.model.layers)
        layer = self.model_runner.model.model.layers[0]
        attn_layer: torch.nn.Module = layer.self_attn.attn
        # print("layer:", layer)
        # print("layer.self_attn:", layer.self_attn)
        # print("layer.self_attn.attn:", layer.self_attn.attn)


        # Counter to keep track of forward passes


        def pre_forward_hook(module, input_args):
            global HOOK_COUNTER
            current_pass_count = HOOK_COUNTER[0]
            print(
                f"Pre-forward hook for pass {current_pass_count}, saving pre-execution tensors.",
                flush=True,
            )

            query, key, value = input_args[0], input_args[1], input_args[2]
            
            print("query dtype:", query.dtype, flush=True)
            print("query:", query.shape, flush=True)
            query = VARIANT_TO_ERROR_FN[args.variant](query)
            # print("key dtype:", key.dtype, flush=True)
            # print("key:", key.shape, flush=True)
            # print("value dtype:", value.dtype, flush=True)
            # print("value:", value.shape, flush=True)
            # # Get the forward context and kv_cache
            # from vllm.forward_context import get_forward_context

            # forward_context = get_forward_context()
            # kv_cache_pre = module.kv_cache[forward_context.virtual_engine]


            # # Save tensors to files using torch.save
            # if query is not None and LOG_PRE_FORWARD_HOOK:
            #     print("query:", query.shape, flush=True)
            #     torch.save(query.detach().cpu(), f"{pass_dir}/query.pt")

            # if key is not None and LOG_PRE_FORWARD_HOOK:
            #     print("key:", key.shape, flush=True)
            #     torch.save(key.detach().cpu(), f"{pass_dir}/key.pt")
            # else:
            #     torch.save(None, f"{pass_dir}/key.pt")

            # if value is not None and LOG_PRE_FORWARD_HOOK:
            #     print("value:", value.shape, flush=True)
            #     torch.save(value.detach().cpu(), f"{pass_dir}/value.pt")
            # else:
            #     torch.save(None, f"{pass_dir}/value.pt")

            # # Save the KV cache *before* forward
            # if isinstance(kv_cache_pre, (list, tuple)) and LOG_PRE_FORWARD_HOOK:
            #     print("kv_cache_pre:", len(kv_cache_pre), flush=True)
            #     torch.save(len(kv_cache_pre), f"{pass_dir}/kv_cache_pre_list_len.pt")
            #     kv_cache_pre_cpu = []
            #     for tensor_item in kv_cache_pre:
            #         if hasattr(tensor_item, "detach"):
            #             kv_cache_pre_cpu.append(tensor_item.detach().cpu())
            #         else:
            #             kv_cache_pre_cpu.append(tensor_item)
            #     torch.save(kv_cache_pre_cpu, f"{pass_dir}/kv_cache_pre.pt")
            # else:
            #     if hasattr(kv_cache_pre, "detach"):
            #         torch.save(
            #             kv_cache_pre.detach().cpu(), f"{pass_dir}/kv_cache_pre.pt"
            #         )
            #     else:
            #         torch.save(kv_cache_pre, f"{pass_dir}/kv_cache_pre.pt")

            # # Also save the forward context's virtual engine
            # torch.save(forward_context.virtual_engine, f"{pass_dir}/virtual_engine.pt")

            return (query, key, value)

        def post_forward_hook(module, args_input, output_tensor):
            global HOOK_COUNTER
            # hook_counter[0] was already incremented by the pre_forward_hook for this pass
            pass_dir = PASS_DIR_FORMAT.format(output_dir=backend,variant=args.variant, current_pass_count=HOOK_COUNTER[0], seq_len=output_tensor.shape[0])
            os.makedirs(pass_dir, exist_ok=True)

            current_pass_count = HOOK_COUNTER[0]
            print(
                f"Post-forward hook for pass {current_pass_count}, saving post-execution tensors.",
                flush=True,
            )

            print("detached output_tensor:", output_tensor.shape, flush=True)
            torch.save(output_tensor.detach().cpu(), f"{pass_dir}/output.pt")

            # Save the KV cache *after* forward pass
            from vllm.forward_context import get_forward_context

            forward_context = get_forward_context()
            kv_cache_post = module.kv_cache[forward_context.virtual_engine]

            if isinstance(kv_cache_post, (list, tuple)) and LOG_KV_CACHE:
                print("kv_cache_post:", len(kv_cache_post), flush=True)
                torch.save(len(kv_cache_post), f"{pass_dir}/kv_cache_post_list_len.pt")
                kv_cache_post_cpu = []
                for tensor_item in kv_cache_post:
                    if hasattr(tensor_item, "detach"):
                        kv_cache_post_cpu.append(tensor_item.detach().cpu())
                    else:
                        kv_cache_post_cpu.append(tensor_item)
                torch.save(kv_cache_post_cpu, f"{pass_dir}/kv_cache_post.pt")
            else:
                if hasattr(kv_cache_post, "detach"):
                    torch.save(
                        kv_cache_post.detach().cpu(), f"{pass_dir}/kv_cache_post.pt"
                    )
                else:
                    torch.save(kv_cache_post, f"{pass_dir}/kv_cache_post.pt")

            HOOK_COUNTER[0] += 1

        # Register the pre-forward and post-forward hooks
        pre_hook = attn_layer.register_forward_pre_hook(pre_forward_hook)
        post_hook = attn_layer.register_forward_hook(post_forward_hook)

        print(
            f"Attached pre-forward and post-forward hooks to {attn_layer}", flush=True
        )
        # To store hooks if you plan to remove them later:
        # if not hasattr(self, 'registered_hooks'):
        #     self.registered_hooks = []
        # self.registered_hooks.extend([pre_hook, post_hook])

    llm.collective_rpc(attach_hook)

    # Generate output with the model to trigger the hook

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_k=1,
    )
        # Tokenize the book text
    tokenized_book_full = llm.get_tokenizer().encode(book)
    print(f"Book tokenized to {len(tokenized_book_full)} tokens")
    
    for seq_len in range(args.start_seq_len, args.end_seq_len+1, args.step_seq_len):
        HOOK_COUNTER[0] = 0
        print(f"Reset HOOK_COUNTER to 0 for seq_len {seq_len}", flush=True)
        # Truncate to a reasonable length if needed
        tokenized_book = tokenized_book_full[:seq_len]
        print(f"Truncated to {len(tokenized_book)} tokens")
        
        detokenized_book = llm.get_tokenizer().decode(tokenized_book)
        output = llm.generate(detokenized_book, sampling_params)
        print("Input:", output[0].prompt[-100:], f"\noutput: {output[0].outputs[0].text}", flush=True)

    # print("ret:", ret)


if __name__ == "__main__":
    main()
