#!/usr/bin/env python3
# Benchmarking script for VLLM models with multiple prompts

import argparse
import time
import json
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.attention.selector import global_force_attn_backend_context_manager, _Backend
import torch

from py_tke import (
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


class SomeClass:
    def __init__(self):
        print("here 1", flush=True)
        from z_hacky_layer_test.test_py_tke import (
            ForwardInplaceTestCase,
            ContextRequest,
            GenerationRequest,
            _sequence_length,
            _input_sequence_length,
        )
        from z_hacky_layer_test.utils import (
            identity_rotary_cos_sin,
            generate_constant_attention_input,
            pack_attention_input,
        )

        self.stream = torch.cuda.current_stream()
        attention_layer_dimensions = AttentionLayerDimensions()
        attention_layer_dimensions.numQHeads = 32
        attention_layer_dimensions.numKVHeads = 4
        attention_layer_dimensions.headSize = 64
        self.test_case = ForwardInplaceTestCase(
            num_layers=1,
            max_batch_size=64,
            max_num_tokens=(1 << 14),
            attention_layer_dimensions=attention_layer_dimensions,
            rotary_embedding_dim=128,
            rotary_embedding_base=10000,
            rotary_embedding_max_positions=2048,
            max_attention_window_size=(1 << 15),
            num_tokens_per_block=32,
            max_num_blocks_per_sequence=512,
            requests=(ContextRequest(sequence_length=1024),),
            output_scaling_factor=1.0,
        )
        print("here 2", flush=True)
        self.rotary_embedding = RotaryEmbedding()
        self.rotary_embedding.type = RotaryPositionalEmbeddingType.GPT_NEOX
        self.rotary_embedding.rotaryEmbeddingDim = 128  # TODO: make this configurable
        self.rotary_embedding.rotaryEmbeddingBase = (
            10000  # TODO: make this configurable
        )
        self.rotary_embedding.rotaryEmbeddingScale = 0  # TODO: make this configurable
        self.rotary_embedding.rotaryEmbeddingMaxPositions = (
            2048  # TODO: make this configurable
        )
        self.rotary_embedding.rotaryScalingType = (
            RotaryScalingType.NONE
        )  # TODO: make this configurable

        self.prefix_cache_configuration = PrefixCacheConfiguration()
        self.prefix_cache_configuration.numTokensPerBlock = (
            32  # TODO: make this configurable
        )
        self.prefix_cache_configuration.maxNumBlocksPerSequence = (
            512  # TODO: make this configurable
        )
        self.prefix_cache_configuration.dataType = DeviceDataType.FP8_E4M3

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
            self.test_case.max_batch_size
            * self.test_case.attention_layer_dimensions.numQHeads,
            device=torch.device("cuda"),
            dtype=torch.int32,
        )
        print("here 3", flush=True)
        # Create a representation of the fixed parameters of the attention operation.
        self.op = create_op(
            inputDataType=DeviceDataType.BF16,
            outputDataType=DeviceDataType.FP8_E4M3,
            attentionLayerDimensions=self.test_case.attention_layer_dimensions,
            rotaryEmbedding=self.rotary_embedding,
            prefixCacheConfiguration=self.prefix_cache_configuration,
            qScaling=1.0,
            maxAttentionWindowSize=self.test_case.max_attention_window_size,
            cyclicAttentionWindowSize=self.test_case.max_attention_window_size,
            fp8OutputScaling=fp8_output_scaling,
            kvScaleOrigQuant=kv_scale_orig_quant,
            kvScaleQuantOrig=kv_scale_quant_orig,
            multiBlockSemaphores=multi_block_semaphores,
        )
        print("here 4", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark VLLM model with multiple prompts"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument(
        "--prompts", type=str, nargs="+", help="List of prompts to process"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        help="Path to file containing prompts (one per line)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for processing"
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--num-iters", type=int, default=5, help="Number of iterations to run"
    )
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=2,
        help="Number of iterations for warmup",
    )
    parser.add_argument(
        "--enforce-eager", action="store_true", help="Enforce eager execution"
    )
    parser.add_argument(
        "--output-json", type=str, help="Path to save the results in JSON format"
    )
    parser.add_argument(
        "--tp", type=int, default=1, help="Number of TP"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load prompts from file if specified
    prompts = args.prompts or []
    prompt_configs = []  # List of (output_length, prompt) tuples

    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse format: output_length, prompt
                if "," in line:
                    parts = line.split(",", 1)  # Split only on first comma
                    try:
                        output_length = int(parts[0].strip())
                        prompt = parts[1].strip() if parts[1].strip() else ""
                        prompt_configs.append((output_length, prompt))
                        prompts.append(prompt)  # Keep for backward compatibility
                    except ValueError:
                        print(f"Warning: Invalid output length in line: {line}")
                        # Treat as regular prompt if parsing fails
                        prompts.append(line)
                        prompt_configs.append((args.output_len, line))
                else:
                    # No comma, treat as regular prompt with default output length
                    prompts.append(line)
                    prompt_configs.append((args.output_len, line))

    # Handle command line prompts (use default output length)
    for prompt in args.prompts or []:
        if prompt not in prompts:  # Avoid duplicates
            prompts.append(prompt)
            prompt_configs.append((args.output_len, prompt))

    if not prompts:
        raise ValueError("No prompts provided. Use --prompts or --prompts-file")

    print(f"Model: {args.model}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output length: {args.output_len}")
    print(f"Iterations: {args.num_iters}")
    print(f"Warmup iterations: {args.num_iters_warmup}")

    # SomeClass()
    if "FP8" in args.model:
        quantization = "modelopt"
        kv_cache_dtype = "fp8"
    else:
        quantization = None
        kv_cache_dtype = "auto"
    # Initialize model
    llm = LLM(
        model=args.model,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        quantization=quantization,
        kv_cache_dtype=kv_cache_dtype,
        block_size=32,
        tensor_parallel_size=args.tp,
    )

    # Process prompts as a batch with individual sampling parameters
    def process_prompts():
        # Create list of prompts and corresponding sampling parameters
        batch_prompts = []
        batch_sampling_params = []

        for output_length, prompt in prompt_configs:
            batch_prompts.append(prompt)
            batch_sampling_params.append(
                SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=output_length,
                )
            )

        # Process all prompts in one batch with individual sampling parameters
        outputs = llm.generate(batch_prompts, batch_sampling_params)
        return outputs

    # Warmup runs
    print("\nWarming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        process_prompts()

    # Benchmark runs
    print("\nBenchmarking...")
    latencies = []
    for i in tqdm(range(args.num_iters), desc="Benchmark iterations"):
        start_time = time.perf_counter()
        results = process_prompts()
        end_time = time.perf_counter()
        latency = end_time - start_time
        latencies.append(latency)

    # Calculate statistics
    latencies = np.array(latencies)
    avg_latency = np.mean(latencies)
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies, percentages)

    # Print results
    print(f"\nResults for processing {len(prompts)} prompts:")
    print(f"Average latency: {avg_latency:.4f} seconds")
    print(f"Latency per prompt: {avg_latency / len(prompts):.4f} seconds")
    print("\nPrompt-Output Pairs:")

    # Display results in original order
    for i, (output_length, prompt) in enumerate(prompt_configs):
        if i < len(results):
            print(f"\nPair {i+1}:")
            print(f"Output Length:{output_length}")
            print(f"Prompt:{prompt}")
            print(f"Output:{results[i].outputs[0].text}")

    for percentage, percentile in zip(percentages, percentiles):
        print(f"{percentage}% percentile latency: {percentile:.4f} seconds")

    # Output JSON results if specified
    if args.output_json:
        results_data = (
            {  # Renamed to avoid conflict with 'results' from process_prompts
                "model": args.model,
                "num_prompts": len(prompts),
                "batch_size": args.batch_size,
                "output_len": args.output_len,
                "avg_latency": float(avg_latency),
                "avg_latency_per_prompt": float(avg_latency / len(prompts)),
                "latencies": latencies.tolist(),
                "percentiles": dict(zip(percentages, percentiles.tolist())),
            }
        )
        with open(args.output_json, "w") as f:
            json.dump(results_data, f, indent=4)
        print(f"Results saved to {args.output_json}")

    # Example of a single forward pass with a forced backend
    if False:
        if hasattr(_Backend, "FLASHINFER"):
            print("\nPerforming an extra forward pass with FLASHINFER backend...")
            with global_force_attn_backend_context_manager(_Backend.FLASHINFER):
                del llm
                # Initialize model
                llm = LLM(
                    model=args.model,
                    enforce_eager=args.enforce_eager,
                    trust_remote_code=True,
                )
                start_time_extra = time.perf_counter()
                extra_outputs = process_prompts()
                end_time_extra = time.perf_counter()
                latency_extra = end_time_extra - start_time_extra
                print(
                    f"Extra forward pass with FLASHINFER took: {latency_extra:.4f} seconds"
                )
                # Optionally, do something with extra_outputs
        else:
            print("\nFLASHINFER backend not available, skipping extra forward pass.")


if __name__ == "__main__":
    main()
