#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Comprehensive benchmarking script for vLLM serving performance.

This script starts vLLM servers with different attention backends and runs
benchmark_serving.py against them with various configurations. It now supports
both eager and non-eager execution modes.

Usage:
    python benchmark_serving_runner.py --model /path/to/model --tp-size 4
    python benchmark_serving_runner.py --model /path/to/model --tp-size 4 --execution-modes eager non-eager

Environment variables:
    MODEL_PATH: Default model path
    TP_SIZE: Default tensor parallel size
    OUTPUT_PATH: Output directory for results
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests


class VLLMServerManager:
    """Manages vLLM server lifecycle."""

    def __init__(self, model_path: str, tp_size: int, port: int = 8000):
        self.model_path = model_path
        self.tp_size = tp_size
        self.port = port
        self.process: subprocess.Popen | None = None
        self.log_file: str | None = None

    def start_server(
        self,
        backend_env: dict[str, str],
        backend_name: str,
        enforce_eager: bool,
        full_cuda_graph: bool,
    ) -> bool:
        """Start vLLM server with specified backend environment and execution mode."""
        execution_mode = "eager" if enforce_eager else "non-eager"
        print(
            f"Starting {backend_name} server ({execution_mode} mode) on port {self.port}..."
        )

        cuda_graph_config = "full" if full_cuda_graph else "none"

        # Set up environment
        env = os.environ.copy()
        env.update(backend_env)

        # Prepare command
        nsys_cmd_prefix = (
            [
                "nsys",
                "profile",
                "-o",
                f"{backend_name.lower()}_{execution_mode}_{cuda_graph_config}_server_no_reordering.nsys-rep",
                "--trace-fork-before-exec=true",
                "--cuda-graph-trace=node",
                "--force-overwrite=true",
            ]
            if os.getenv("NSYS_PROFILE", "0") == "1"
            else []
        )
        cmd = nsys_cmd_prefix + [
            "vllm",
            "serve",
            self.model_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(self.tp_size),
            "--quantization",
            "modelopt",
            "--kv-cache-dtype",
            "fp8",
            "--disable-log-requests",
            "--disable-log-stats",
            "--trust-remote-code",
        ]

        # Add enforce-eager flag if needed
        if enforce_eager:
            cmd.append("--enforce-eager")
        else:
            if full_cuda_graph:
                # Add compilation config for full CUDA graph when not in eager mode
                cmd.extend(["--compilation-config", '{"full_cuda_graph": true}'])

        # Start server with improved process management
        self.log_file = (
            f"{backend_name.lower()}_{execution_mode}_{cuda_graph_config}_server.log"
        )

        print(f"Starting server with command: {' '.join(cmd)}")
        print(f"Environment: {backend_env}")
        print(f"Logging to: {self.log_file}")

        try:
            # Open log file in a way that persists after process creation
            log_fd = os.open(
                self.log_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644
            )

            # Create a simple preexec function that's safer in containerized environments
            def safe_preexec():
                try:
                    # Try to create a new process group, but don't fail if it doesn't work
                    os.setpgrp()
                except (OSError, AttributeError):
                    # If setpgrp fails, continue without it
                    pass

            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_fd,
                stderr=subprocess.STDOUT,
                preexec_fn=safe_preexec,  # Use safer preexec function
            )

            # Close our copy of the file descriptor - the subprocess has its own
            os.close(log_fd)

            print(f"Server process started with PID: {self.process.pid}")

        except Exception as e:
            print(f"Failed to start server process: {e}")
            return False

        # Wait for server to be ready
        success = self._wait_for_ready(timeout=300)

        if success:
            print(f"Server is ready and stable. Process PID: {self.process.pid}")
            # Give the server a moment to fully stabilize
            time.sleep(5)

            # Verify the process is still running
            if self.process.poll() is not None:
                print(
                    f"Error: Server process exited with code {self.process.returncode}"
                )
                self._print_recent_logs()
                return False

            print("Server verification complete - process is stable")
        else:
            print("Server failed to start or become ready")
            self._print_recent_logs()

        return success

    def _wait_for_ready(self, timeout: int = 60) -> bool:
        """Wait for server to be ready to accept requests."""
        print(f"Waiting for server to be ready on port {self.port}...")

        start_time = time.time()
        consecutive_successes = 0
        required_successes = 3  # Require multiple successful checks

        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"http://localhost:{self.port}/v1/models", timeout=5
                )
                if response.status_code == 200:
                    consecutive_successes += 1
                    print(
                        f"Health check {consecutive_successes}/{required_successes} successful"
                    )
                    if consecutive_successes >= required_successes:
                        print("Server is ready!")
                        return True
                else:
                    consecutive_successes = 0
                    print(f"Health check failed with status {response.status_code}")
            except (
                requests.exceptions.RequestException,
                requests.exceptions.Timeout,
            ):
                consecutive_successes = 0

            # Check if process is still running
            if self.process and self.process.poll() is not None:
                print(f"Server process exited with code {self.process.returncode}")
                self._print_recent_logs()
                return False

            elapsed = int(time.time() - start_time)
            print(f"Waiting... ({elapsed}s elapsed)")
            time.sleep(2)

        print(f"Server failed to start within {timeout} seconds")
        self._print_recent_logs()
        return False

    def _print_recent_logs(self):
        """Print recent server logs for debugging."""
        if self.log_file and os.path.exists(self.log_file):
            print(f"\n--- Recent server logs from {self.log_file} ---")
            try:
                with open(self.log_file) as f:
                    lines = f.readlines()
                    # Print last 20 lines
                    for line in lines[-20:]:
                        print(line.rstrip())
            except Exception as e:
                print(f"Failed to read log file: {e}")
            print("--- End of logs ---\n")

    def stop_server(self):
        """Stop the vLLM server."""
        if self.process:
            print(f"Stopping server (PID: {self.process.pid})...")
            try:
                # Check if process is still running
                if self.process.poll() is None:
                    # Try graceful shutdown first
                    print("Sending SIGTERM to process...")

                    # Try to terminate the process group if possible, otherwise just the process
                    try:
                        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                        print("Sent SIGTERM to process group")
                    except (OSError, ProcessLookupError):
                        # If process group termination fails, just terminate the main process
                        self.process.terminate()
                        print("Sent SIGTERM to main process")

                    # Wait for graceful shutdown
                    try:
                        self.process.wait(timeout=15)
                        print("Server shut down gracefully")
                    except subprocess.TimeoutExpired:
                        print("Graceful shutdown timed out, forcing kill...")
                        try:
                            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                        except (OSError, ProcessLookupError):
                            self.process.kill()
                        self.process.wait(timeout=5)
                else:
                    print(f"Process already exited with code {self.process.returncode}")

            except (ProcessLookupError, OSError) as e:
                print(f"Process cleanup error (expected): {e}")
            finally:
                self.process = None

        # Clean up any remaining processes more carefully
        try:
            result = subprocess.run(
                ["pgrep", "-f", f"vllm.*serve.*{self.model_path}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                pids = result.stdout.strip().split("\n")
                print(f"Cleaning up remaining vLLM processes: {pids}")
                for pid in pids:
                    if pid.strip():
                        subprocess.run(["kill", "-9", pid.strip()], check=False)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        print("Server cleanup complete")
        time.sleep(3)  # Give time for cleanup

    def is_running(self) -> bool:
        """Check if the server process is still running."""
        return self.process is not None and self.process.poll() is None


class BenchmarkRunner:
    """Runs benchmark_serving.py with various configurations."""

    def __init__(self, model_path: str, output_path: str, port: int = 8000):
        self.model_path = model_path
        self.output_path = Path(output_path)
        self.port = port
        self.output_path.mkdir(parents=True, exist_ok=True)

    def run_benchmark(self, config: dict, server_manager: VLLMServerManager) -> bool:
        """Run a single benchmark configuration."""
        concurrency = config["concurrency"]
        input_len = config["input_len"]
        output_len = config["output_len"]
        backend_name = config["backend_name"]
        execution_mode = config["execution_mode"]
        num_requests = concurrency * 10

        # Include CUDA graph info in filename if available
        # Use underscores as separators for unambiguous parsing
        filename_parts = [
            str(concurrency),
            str(input_len),
            str(output_len),
            backend_name.lower(),
            execution_mode,
        ]
        if "cuda_graph" in config:
            filename_parts.append(config["cuda_graph"])

        result_file = self.output_path / f"{'_'.join(filename_parts)}.json"

        print(
            f"Running benchmark: concurrency={concurrency}, input_len={input_len}, "
            f"output_len={output_len}, backend={backend_name}, mode={execution_mode}"
            + (f", cuda_graph={config['cuda_graph']}" if "cuda_graph" in config else "")
        )

        # Verify server is still running before starting benchmark
        if not server_manager.is_running():
            print("✗ Server is not running, cannot run benchmark")
            return False

        cmd = [
            "python",
            "benchmarks/benchmark_serving.py",
            "--backend",
            "vllm",
            "--model",
            self.model_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--dataset-name",
            "random",
            "--num-prompts",
            str(num_requests),
            "--random-input-len",
            str(input_len),
            "--random-output-len",
            str(output_len),
            "--max-concurrency",
            str(concurrency),
            "--result-filename",
            str(result_file),
            "--save-result",
            "--save-detailed",
            "--ignore-eos",
        ]

        try:
            print(f"Benchmark command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            if result.returncode == 0:
                print(f"✓ Benchmark completed: {result_file}")
                return True
            else:
                print(f"✗ Benchmark failed with return code {result.returncode}")
                print(f"stderr: {result.stderr}")
                print(f"stdout: {result.stdout}")
                return False
        except subprocess.TimeoutExpired:
            print("✗ Benchmark timed out")
            return False
        except Exception as e:
            print(f"✗ Benchmark failed with exception: {e}")
            return False


def generate_test_configurations() -> list[dict]:
    """Generate test configurations for benchmarking."""
    concurrency_levels = [2, 4]
    output_lengths = [500]
    input_lengths = [5000, 64000, 128000]

    configs = []
    for concurrency in concurrency_levels:
        for output_len in output_lengths:
            for input_len in input_lengths:
                configs.append(
                    {
                        "concurrency": concurrency,
                        "input_len": input_len,
                        "output_len": output_len,
                    }
                )

    return configs


def run_backend_benchmarks(
    server_manager: VLLMServerManager,
    benchmark_runner: BenchmarkRunner,
    backend_config: tuple[dict[str, str], str],
    test_configs: list[dict],
    execution_modes: list[str],
) -> dict[str, list[bool]]:
    """Run all benchmarks for a specific backend in both execution modes."""
    backend_env, backend_name = backend_config

    all_results = {}

    for execution_mode in execution_modes:
        enforce_eager = execution_mode == "eager"

        # Determine CUDA graph configurations to test
        if backend_name == "flash-attn" and not enforce_eager:
            # For flash-attn in non-eager mode, test both with and without full CUDA graph
            cuda_graph_configs = [True]
        else:
            # For all other cases, only test without full CUDA graph
            cuda_graph_configs = [False]

        for full_cuda_graph in cuda_graph_configs:
            mode_results = []

            # Create a unique key for results that includes CUDA graph info when relevant
            if backend_name == "flash-attn" and not enforce_eager:
                result_key = f"{execution_mode}_{'full_cuda_graph' if full_cuda_graph else 'no_cuda_graph'}"
                cuda_graph_desc = (
                    "with full CUDA graph"
                    if full_cuda_graph
                    else "without full CUDA graph"
                )
            else:
                result_key = execution_mode
                cuda_graph_desc = ""

            print(f"\n{'=' * 40}")
            if cuda_graph_desc:
                print(
                    f"Testing {backend_name} in {execution_mode} mode {cuda_graph_desc}"
                )
            else:
                print(f"Testing {backend_name} in {execution_mode} mode")
            print(f"{'=' * 40}")

            # Start server for this backend and execution mode
            server_started = False
            try:
                server_started = server_manager.start_server(
                    backend_env, backend_name, enforce_eager, full_cuda_graph
                )

                if not server_started:
                    desc_suffix = f" {cuda_graph_desc}" if cuda_graph_desc else ""
                    print(
                        f"Failed to start {backend_name} server in {execution_mode} mode{desc_suffix}"
                    )
                    mode_results = [False] * len(test_configs)
                else:
                    desc_suffix = f" {cuda_graph_desc}" if cuda_graph_desc else ""
                    print(
                        f"Successfully started {backend_name} server in {execution_mode} mode{desc_suffix}"
                    )

                    # Run all test configurations
                    for i, config in enumerate(test_configs):
                        config["backend_name"] = backend_name
                        config["execution_mode"] = execution_mode
                        # Add CUDA graph info to config for result file naming
                        if backend_name == "flash-attn" and not enforce_eager:
                            config["cuda_graph"] = "full" if full_cuda_graph else "none"

                        print(f"\nRunning test {i + 1}/{len(test_configs)}")

                        # Verify server is still running before each test
                        if not server_manager.is_running():
                            print(
                                "Server stopped running, cannot continue with remaining tests"
                            )
                            mode_results.extend(
                                [False] * (len(test_configs) - len(mode_results))
                            )
                            break

                        success = benchmark_runner.run_benchmark(config, server_manager)
                        mode_results.append(success)

                        if not success:
                            print("Benchmark failed, checking server status...")
                            if not server_manager.is_running():
                                print(
                                    "Server crashed during benchmark, stopping remaining tests"
                                )
                                mode_results.extend(
                                    [False] * (len(test_configs) - len(mode_results))
                                )
                                break
                            else:
                                print("Server still running, continuing with next test")

            except Exception as e:
                desc_suffix = f" {cuda_graph_desc}" if cuda_graph_desc else ""
                print(
                    f"Exception occurred while running {backend_name} ({execution_mode}{desc_suffix}): {e}"
                )
                mode_results = [False] * len(test_configs)
            finally:
                # Always stop the server, regardless of what happened
                if server_started:
                    desc_suffix = f" {cuda_graph_desc}" if cuda_graph_desc else ""
                    print(
                        f"\nShutting down {backend_name} server ({execution_mode} mode{desc_suffix})"
                    )
                    server_manager.stop_server()
                else:
                    print("Server was never started, no cleanup needed")

            all_results[result_key] = mode_results

    return all_results


def main():
    # Set up signal handling to prevent accidental server shutdowns
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}. Cleaning up...")
        # Don't propagate signals to child processes immediately
        # Let them finish gracefully
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="Run vLLM serving benchmarks")
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_PATH", "/scratch/usr/quantized_model"),
        help="Model path",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=int(os.getenv("TP_SIZE", "4")),
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--output-path",
        default=os.getenv("OUTPUT_PATH", "."),
        help="Output directory for results",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port for vLLM server")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["tke", "flash-attn", "flashinfer"],
        default=["tke", "flash-attn"],
        help="Backends to test",
    )
    parser.add_argument(
        "--execution-modes",
        nargs="+",
        choices=["eager", "non-eager"],
        default=["non-eager"],
        help="Execution modes to test (default: both eager and non-eager)",
    )

    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model):
        print(f"Error: Model path does not exist: {args.model}")
        return 1

    # Backend configurations
    backend_configs = {
        "tke": (
            {
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
                "VLLM_ATTENTION_BACKEND": "TKE",
                # "CUDA_LAUNCH_BLOCKING": "1",
            },
            "TKE",
        ),
        "flash-attn": (
            {
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
                "VLLM_ATTENTION_BACKEND": "FLASH_ATTN_VLLM_V1",
            },
            "flash-attn",
        ),
        "flashinfer": (
            {
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
                "VLLM_ATTENTION_BACKEND": "FLASHINFER_VLLM_V1",
            },
            "FlashInfer",
        ),
    }

    test_configs = generate_test_configurations()

    print(
        f"Running {len(test_configs)} test configurations per backend per execution mode"
    )
    print(f"Testing backends: {args.backends}")
    print(f"Testing execution modes: {args.execution_modes}")
    print(f"Model: {args.model}")
    print(f"TP Size: {args.tp_size}")
    print(f"Output Path: {args.output_path}")
    print()

    # Initialize managers
    server_manager = VLLMServerManager(args.model, args.tp_size, args.port)
    benchmark_runner = BenchmarkRunner(args.model, args.output_path, args.port)

    # Run benchmarks for each backend
    all_results = {}

    try:
        for backend_name in args.backends:
            if backend_name not in backend_configs:
                print(f"Warning: Unknown backend '{backend_name}', skipping")
                continue

            print(f"\n{'=' * 60}")
            print(f"Running benchmarks for {backend_configs[backend_name][1]} backend")
            print(f"{'=' * 60}")

            try:
                results = run_backend_benchmarks(
                    server_manager,
                    benchmark_runner,
                    backend_configs[backend_name],
                    test_configs,
                    args.execution_modes,
                )

                all_results[backend_name] = results

                # Print results for this backend
                for result_key, result_list in results.items():
                    successful = sum(result_list)
                    total = len(result_list)

                    # Format the result key for display
                    if result_key.endswith("_full_cuda_graph"):
                        execution_mode = result_key[: -len("_full_cuda_graph")]
                        display_key = f"{execution_mode} (full CUDA graph)"
                    elif result_key.endswith("_no_cuda_graph"):
                        execution_mode = result_key[: -len("_no_cuda_graph")]
                        display_key = f"{execution_mode} (no CUDA graph)"
                    else:
                        # Regular result key
                        display_key = result_key

                    print(
                        f"\n{backend_configs[backend_name][1]} ({display_key}) Results: {successful}/{total} successful"
                    )
            except Exception as e:
                print(f"Failed to run benchmarks for {backend_name}: {e}")
                # Ensure server is stopped even if there's an error
                try:
                    server_manager.stop_server()
                except:
                    pass

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error during benchmarking: {e}")
        return 1
    finally:
        # Final cleanup - make sure no servers are left running
        try:
            server_manager.stop_server()
        except:
            pass

        # Extra cleanup for any remaining vLLM processes
        try:
            subprocess.run(["pkill", "-f", "vllm serve"], check=False, timeout=10)
        except:
            pass

    # Print final summary
    print(f"\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    for backend_name, backend_results in all_results.items():
        backend_display = backend_configs[backend_name][1]
        for result_key, results in backend_results.items():
            successful = sum(results)
            total = len(results)
            success_rate = (successful / total * 100) if total > 0 else 0

            # Format the result key for display
            if result_key.endswith("_full_cuda_graph"):
                execution_mode = result_key[: -len("_full_cuda_graph")]
                display_key = f"{execution_mode} (full CG)"
            elif result_key.endswith("_no_cuda_graph"):
                execution_mode = result_key[: -len("_no_cuda_graph")]
                display_key = f"{execution_mode} (no CG)"
            else:
                # Regular result key
                display_key = result_key

            print(
                f"{backend_display:12} ({display_key:12}): {successful:3}/{total:3} successful ({success_rate:5.1f}%)"
            )

    print(f"\nResults saved to: {args.output_path}")

    # Return non-zero exit code if any benchmarks failed
    total_failed = 0
    for backend_results in all_results.values():
        for mode_results in backend_results.values():
            total_failed += len(mode_results) - sum(mode_results)

    return min(total_failed, 1)


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        # Clean up any remaining processes
        try:
            subprocess.run(["pkill", "-f", "vllm serve"], check=False, timeout=10)
        except:
            pass
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        # Clean up any remaining processes
        try:
            subprocess.run(["pkill", "-f", "vllm serve"], check=False, timeout=10)
        except:
            pass
        sys.exit(1)
