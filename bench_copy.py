import sys

import torch

def _concat_k_nope_k_pe(
    k_nope: torch.Tensor, k_pe: torch.Tensor, k: torch.Tensor
):
    """
    Efficiently concatenate k_nope and k_pe tensors along the last dimension.

    This function avoids the performance penalty of torch.cat with expanded
    non-contiguous tensors by pre-allocating the output and using direct copies.

    Args:
        k_nope: Tensor of shape [..., nope_dim]
        k_pe: Tensor to broadcast and concatenate, typically shape [..., 1, pe_dim]
            or [..., pe_dim]

    Returns:
        Tensor of shape [..., nope_dim + pe_dim]
    """

    # Direct copies with efficient broadcasting
    k[..., : k_nope.shape[-1]] = k_nope
    k[..., k_nope.shape[-1] :] = k_pe

def main() -> int:
    return 0

if __name__ == "__main__":
    k_head_dim = 128
    rope_dim = 64
    num_heads = 128
    num_tokens = 100000
    num_iterations = 1000

    k_nope = torch.randn((num_tokens, num_heads, k_head_dim), device="cuda").to(torch.bfloat16)
    k_pe = torch.randn((num_tokens, num_heads, k_head_dim - rope_dim), device="cuda").to(torch.bfloat16)

    k = torch.empty(
        (*k_nope.shape[:-1], k_nope.shape[-1] + k_pe.shape[-1]),
        dtype=k_nope.dtype,
        device=k_nope.device,
    )

    # Warmup
    _concat_k_nope_k_pe(k_nope, k_pe, k)
    torch.cuda.synchronize()

    # Capture CUDA graph
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Warmup in capture stream
        _concat_k_nope_k_pe(k_nope, k_pe, k)
    torch.cuda.synchronize()

    # Capture the graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(num_iterations):
            _concat_k_nope_k_pe(k_nope, k_pe, k)

    # Benchmark the graph replay
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    print(f"{num_iterations} iterations in CUDA graph: {elapsed_ms:.3f} ms")
    time_per_iter = elapsed_ms / num_iterations
    print(f"Average per iteration: {time_per_iter:.6f} ms")

    amount_of_data_read = (
        k_nope.numel() + k_pe.numel()
    ) * k_nope.element_size()

    amount_of_data_written = amount_of_data_read
    total_data = amount_of_data_read + amount_of_data_written
    bandwidth = total_data / (time_per_iter / 1000) / 1e9  # GB/s
    print(f"Effective Bandwidth: {bandwidth:.2f} GB/s")

    sys.exit(main())