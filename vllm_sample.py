#!/usr/bin/env python3
# Benchmarking script for VLLM models with multiple prompts

import argparse
import time
import json
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig

def print_speculative_decoding_metrics(llm:LLM):
    """Extract and print speculative decoding acceptance rate metrics"""
    print("\n" + "="*60)
    print("SPECULATIVE DECODING METRICS")
    print("="*60)
    
    try:
        metrics = llm.get_metrics()
        
        # Initialize counters
        num_drafts = 0
        num_draft_tokens = 0
        num_accepted_tokens = 0
        acceptance_counts = [0] * 10  # Support up to 10 speculative tokens
        
        # Extract metrics from the metrics list
        for metric in metrics:
            if hasattr(metric, 'name') and hasattr(metric, 'value'):
                if "spec_decode_num_drafts" in metric.name:
                    num_drafts += metric.value
                elif "spec_decode_num_draft_tokens" in metric.name:
                    num_draft_tokens += metric.value
                elif "spec_decode_num_accepted_tokens" in metric.name and "per_pos" not in metric.name:
                    num_accepted_tokens += metric.value
                elif "spec_decode_num_accepted_tokens_per_pos" in metric.name:
                    # Handle per-position metrics if available
                    print(metric)
                    if hasattr(metric, 'values'):
                        for pos, count in enumerate(metric.values):
                            if pos < len(acceptance_counts):
                                acceptance_counts[pos] += count
        
        # Calculate and display metrics
        if num_draft_tokens > 0:
            acceptance_rate = (num_accepted_tokens / num_draft_tokens) * 100
            mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1
            
            print(f"📊 Draft Acceptance Rate: {acceptance_rate:.2f}%")
            print(f"📏 Mean Acceptance Length: {mean_acceptance_length:.2f} tokens")
            print(f"✅ Total Accepted Tokens: {num_accepted_tokens:,}")
            print(f"📝 Total Draft Tokens: {num_draft_tokens:,}")
            print(f"🎯 Number of Drafts: {num_drafts:,}")
            
            # Calculate efficiency
            efficiency = (num_accepted_tokens / num_draft_tokens) if num_draft_tokens > 0 else 0
            print(f"⚡ Speculative Efficiency: {efficiency:.3f}")
            
            # Per-position acceptance rates
            print(f"\n📍 Per-Position Acceptance Rates:")
            for i, count in enumerate(acceptance_counts[:5]):  # Show first 5 positions
                if num_drafts > 0 and count > 0:
                    pos_rate = (count / num_drafts) * 100
                    print(f"   Position {i}: {pos_rate:.1f}%")
                    
        else:
            print("❌ No speculative decoding metrics available")
            print("   (Check if speculative decoding is enabled and working)")
            
    except Exception as e:
        print(f"⚠️  Error extracting metrics: {e}")
        print("   Trying alternative method...")
        
        # Alternative method: try to access stat loggers directly
        try:
            if hasattr(llm.llm_engine, 'stat_loggers'):
                stat_logger = llm.llm_engine.stat_loggers.get('prometheus')
                if stat_logger and hasattr(stat_logger, 'spec_decode_metrics'):
                    metrics = stat_logger.spec_decode_metrics
                    if metrics:
                        print(f"📊 Draft Acceptance Rate: {metrics.draft_acceptance_rate:.2f}%")
                        print(f"⚡ System Efficiency: {metrics.system_efficiency:.3f}")
                        print(f"✅ Accepted Tokens: {metrics.accepted_tokens:,}")
                        print(f"📝 Draft Tokens: {metrics.draft_tokens:,}")
                    else:
                        print("❌ No metrics available from stat logger")
                else:
                    print("❌ Stat logger not available")
            else:
                print("❌ No stat loggers found")
        except Exception as e2:
            print(f"❌ Alternative method also failed: {e2}")
    
    print("="*60)

def monitor_realtime_metrics(llm, interval=2.0, duration=10.0):
    """Monitor acceptance rate in real-time during generation"""
    print(f"\n🔄 Real-time monitoring for {duration}s (interval: {interval}s)")
    
    try:
        start_time = time.time()
        while time.time() - start_time < duration:
            metrics = llm.get_metrics()
            
            num_draft_tokens = 0
            num_accepted_tokens = 0
            
            for metric in metrics:
                if hasattr(metric, 'name') and hasattr(metric, 'value'):
                    if "spec_decode_num_draft_tokens" in metric.name:
                        num_draft_tokens += metric.value
                    elif "spec_decode_num_accepted_tokens" in metric.name and "per_pos" not in metric.name:
                        num_accepted_tokens += metric.value
            
            if num_draft_tokens > 0:
                rate = (num_accepted_tokens / num_draft_tokens) * 100
                print(f"⏱️  Current Acceptance Rate: {rate:.1f}% ({num_accepted_tokens}/{num_draft_tokens})")
            else:
                print("⏱️  No speculative tokens yet...")
                
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n⏹️  Real-time monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Real-time monitoring error: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark VLLM model with multiple prompts")
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="Model name or path")
    parser.add_argument(
        "--prompts-file",
        type=str,
        help="Path to file containing prompts (one per line)",
    )
    parser.add_argument("--batch-size",
                        type=int,
                        default=1,
                        help="Batch size for processing")
    parser.add_argument("--num-iters",
                        type=int,
                        default=5,
                        help="Number of iterations to run")
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=2,
        help="Number of iterations for warmup",
    )
    parser.add_argument("--enforce-eager",
                        action="store_true",
                        help="Enforce eager execution")
    parser.add_argument(
        "--output-json",
        type=str,
        help="Path to save the results in JSON format",
    )
    parser.add_argument("--max-model-len",
                        type=int,
                        help="Maximum model length")
    parser.add_argument("--tensor-parallel-size",
                        type=int,
                        help="Tensor parallel size")
    parser.add_argument("--kv-cache-dtype",
                        type=str,
                        required=False,
                        help="KV cache dtype")
    parser.add_argument("--draft-model-path",
                        type=str,
                        required=False,
                        help="Draft model path")
    parser.add_argument("--num-speculative-tokens",
                        type=int,
                        required=False,
                        help="Number of speculative tokens")
    parser.add_argument("--enable-metrics",
                        action="store_true",
                        help="Enable speculative decoding metrics collection")
    parser.add_argument("--realtime-monitoring",
                        action="store_true", 
                        help="Enable real-time acceptance rate monitoring")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load prompts from file if specified
    prompts = []
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
                        prompts.append(
                            prompt)  # Keep for backward compatibility
                    except ValueError:
                        print(
                            f"Warning: Invalid output length in line: {line}")
                        # Treat as regular prompt if parsing fails
                        prompts.append(line)
                        prompt_configs.append((100, line))  # Default to 100 tokens
                else:
                    # No comma, treat as regular prompt with default output length
                    prompts.append(line)
                    prompt_configs.append((100, line))  # Default to 100 tokens

    # prompt_configs.append((1, "a"+prompt_configs[0][1]))
    if not prompts:
        raise ValueError(
            "No prompts provided. Use --prompts or --prompts-file")

    print(f"Model: {args.model}")
    print(f"Number of prompts: {len(prompts)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Iterations: {args.num_iters}")
    print(f"Warmup iterations: {args.num_iters_warmup}")
    print(f"Speculative Decoding: N-gram (5 tokens, max 4-gram)")
    print(f"Metrics Enabled: {args.enable_metrics or args.realtime_monitoring}")

    effective_kv_cache_dtype = args.kv_cache_dtype if args.kv_cache_dtype else "auto"
    print(f"Effective kv_cache_dtype: {effective_kv_cache_dtype}")
    # Initialize model
    speculative_config = None
    if args.num_speculative_tokens and args.draft_model_path:
        speculative_config = {
            "model":
            args.draft_model_path,
            "method":
            "draft_model",
            "num_speculative_tokens":
            args.num_speculative_tokens,
            "target_model_config":
            ModelConfig(model=args.model),
            "target_parallel_config":
            ParallelConfig(tensor_parallel_size=args.tensor_parallel_size)
        }
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        quantization="modelopt",
        kv_cache_dtype=effective_kv_cache_dtype,
        block_size=32,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        enable_prefix_caching=False,
        enforce_eager=args.enforce_eager,
        disable_log_stats=False,  # Enable metrics collection
        speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    })

    
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
                    ignore_eos=True,
                ))

        batch_prompts = batch_prompts * 1
        batch_sampling_params = batch_sampling_params * 1
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
    
    # Start real-time monitoring if requested
    if args.realtime_monitoring:
        import threading
        
        def monitor_background():
            monitor_realtime_metrics(llm, interval=3.0, duration=args.num_iters * 5)
        
        monitor_thread = threading.Thread(target=monitor_background, daemon=True)
        monitor_thread.start()
    
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
    
    # Print speculative decoding metrics
    if args.enable_metrics or args.realtime_monitoring:
        print_speculative_decoding_metrics(llm)
    else:
        # Always show basic metrics to confirm speculative decoding is working
        try:
            metrics = llm.get_metrics()
            num_draft_tokens = 0
            num_accepted_tokens = 0
            
            for metric in metrics:
                if hasattr(metric, 'name') and hasattr(metric, 'value'):
                    if "spec_decode_num_draft_tokens" in metric.name:
                        num_draft_tokens += metric.value
                    elif "spec_decode_num_accepted_tokens" in metric.name and "per_pos" not in metric.name:
                        num_accepted_tokens += metric.value
            
            if num_draft_tokens > 0:
                rate = (num_accepted_tokens / num_draft_tokens) * 100
                print(f"\n🎯 N-gram Acceptance Rate: {rate:.1f}% ({num_accepted_tokens}/{num_draft_tokens} tokens)")
            else:
                print("\n⚠️  No speculative decoding detected (may be normal for short sequences)")
        except:
            print("\n⚠️  Could not extract basic metrics")
    
    print("\nPrompt-Output Pairs:")

    # Display results in original order
    for i, (output_length, prompt) in enumerate(prompt_configs):
        if i < len(results):
            print(f"\nPair {i + 1}:")
            print(f"Output Length:{output_length}")
            print(f"Prompt:{prompt}")
            print(f"Output:{results[i].outputs[0].text}")

    for percentage, percentile in zip(percentages, percentiles):
        print(f"{percentage}% percentile latency: {percentile:.4f} seconds")

    # Output JSON results if specified
    if args.output_json:
        results_data = {  # Renamed to avoid conflict with 'results' from process_prompts
            "model": args.model,
            "num_prompts": len(prompts),
            "batch_size": args.batch_size,
            "avg_latency": float(avg_latency),
            "avg_latency_per_prompt": float(avg_latency / len(prompts)),
            "latencies": latencies.tolist(),
            "percentiles": dict(zip(percentages, percentiles.tolist())),
        }
        with open(args.output_json, "w") as f:
            json.dump(results_data, f, indent=4)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
