from typing import Any

import argparse
import glob
import os

import torch
import matplotlib.pyplot as plt

def load_output_tensor_from_pass(pass_dir: str) -> torch.Tensor | None:
    """Load all tensors from a specific pass directory."""
    pt_files = glob.glob(os.path.join(pass_dir, "*.pt"))
    assert len(pt_files) == 1, f"Expected 1 tensor file in {pass_dir}, got {len(pt_files)}"
    file_path = pt_files[0]

    tensor_name = os.path.splitext(os.path.basename(file_path))[0]
    try:
        return torch.load(
            file_path, map_location="cpu"
        )  # Added map_location for safety
    except Exception as e:
        print(f"Error loading tensor {tensor_name} from {file_path}: {e}")
        return None  # Or handle error as appropriate


def last_token_diff(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Compare two tensors/values and return the per-token difference. Assumes first dimension is tokens."""
    return torch.norm(tensor1[-1] - tensor2[-1], p=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare outputs from two different runs.")
    parser.add_argument("--variant_dir_left", type=str, required=True, help="Path to first directory")
    parser.add_argument("--variant_dir_right", type=str, required=True, help="Path to second directory")
    args = parser.parse_args()

    print(f"Comparing {args.variant_dir_left} and {args.variant_dir_right}")
    """Compare outputs from two different runs."""
    # Get all pass directories from both runs
    passes1 = sorted(glob.glob(os.path.join(args.variant_dir_left, "seq_len_*/pass_*")))
    print(f"Passes in {args.variant_dir_left}: {passes1}")
    passes2 = sorted(glob.glob(os.path.join(args.variant_dir_right, "seq_len_*/pass_*")))
    print(f"Passes in {args.variant_dir_right}: {passes2}")

    if len(passes1) != len(passes2):
        print(f"Different number of passes: {len(passes1)} vs {len(passes2)}")
        exit(1)

    # Collect data for scatter plot
    num_tokens_list = []
    diff_list = []

    for pass_idx, (pass1_dir, pass2_dir) in enumerate(zip(passes1, passes2)):
        print(f"\nComparing pass {pass_idx + 1}:")

        # Load tensors from both passes
        tensor1 = load_output_tensor_from_pass(pass1_dir)
        tensor2 = load_output_tensor_from_pass(pass2_dir)

        # Compare each tensor
        num_tokens = tensor1.shape[0]
        diff = last_token_diff(
            tensor1, tensor2
        )
        
        # Collect data for plotting
        num_tokens_list.append(num_tokens)
        diff_list.append(diff.item())
        
        print(f"Tokens: {num_tokens}, Diff: {diff.item():.6f}")
        print(f"  {pass1_dir}")

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(num_tokens_list, diff_list, alpha=0.6, s=50)
    plt.xlabel('Number of Tokens')
    plt.ylabel('Difference (L2 Norm)')
    plt.title('Token Count vs. Difference between Variants')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('token_diff_scatter.png', dpi=150)
    print(f"\nScatter plot saved to token_diff_scatter.png")
    # plt.show()