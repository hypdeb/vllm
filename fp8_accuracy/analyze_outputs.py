# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_output_tensor_from_pass(pass_dir: str) -> torch.Tensor | None:
    """Load all tensors from a specific pass directory."""
    pt_files = glob.glob(os.path.join(pass_dir, "*.pt"))
    assert len(pt_files) == 1, (
        f"Expected 1 tensor file in {pass_dir}, got {len(pt_files)}"
    )
    file_path = pt_files[0]

    tensor_name = os.path.splitext(os.path.basename(file_path))[0]
    try:
        return torch.load(file_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading tensor {tensor_name} from {file_path}: {e}")
        return None


def parse_filename(filename: str) -> tuple[str, int, int, int]:
    """Parse filename with template:
    {variant}_{seq_len}_{layer_idx}_{rank}_out.pt

    Returns tuple: (variant, seq_len, layer_idx, rank)
    """
    # Remove .pt extension if present
    base_name = filename.replace(".pt", "")

    # Pattern: variant_seqlen_layeridx_rank_out
    # Split by underscore and get the last 4 parts before 'out'
    parts = base_name.split("_")

    # Find the 'out' suffix and work backwards
    if parts[-1] == "out":
        rank = int(parts[-2])
        layer_idx = int(parts[-3])
        seq_len = int(parts[-4])
        # variant is everything before that (could contain underscores)
        variant = "_".join(parts[:-4])

        return variant, seq_len, layer_idx, rank
    else:
        raise ValueError(f"Filename '{filename}' does not match expected pattern")


def euclidean_distance(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """Calculate euclidean distance between two tensors."""
    return torch.norm(tensor1 - tensor2, p=2).item()


def normalized_euclidean_distance(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> float:
    """Calculate euclidean distance normalized by the norm of tensor2."""
    distance = torch.norm(tensor1 - tensor2, p=2).item()
    norm = torch.norm(tensor2, p=2).item()
    if norm == 0:
        return np.nan
    return distance / norm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare outputs from two different runs."
    )
    parser.add_argument(
        "--attn_outputs_dir",
        type=str,
        required=True,
        help="Path to attn outputs directory",
    )
    args = parser.parse_args()

    # Get all .pt files
    all_attn_output_files = sorted(
        glob.glob(os.path.join(args.attn_outputs_dir, "*.pt"))
    )

    if not all_attn_output_files:
        print(f"No .pt files found in {args.attn_outputs_dir}")
        exit(1)

    # Parse all files and organize data
    data = {}  # key: (layer_idx, rank, seq_len, variant) -> tensor
    all_variants = set()
    all_layers = set()
    all_ranks = set()
    all_seq_lens = set()

    print(f"Loading {len(all_attn_output_files)} files...")
    for file in all_attn_output_files:
        variant, seq_len, layer_idx, rank = parse_filename(os.path.basename(file))

        # Load tensor
        try:
            tensor = torch.load(file, map_location="cpu")
            data[(layer_idx, rank, seq_len, variant)] = tensor

            all_variants.add(variant)
            all_layers.add(layer_idx)
            all_ranks.add(rank)
            all_seq_lens.add(seq_len)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

    # Check if default variant exists
    if "default" not in all_variants:
        print("'default' variant not found to compare everything else with it.")
        exit(1)

    # Get non-default variants
    non_default_variants = sorted([v for v in all_variants if v != "default"])

    print(f"Found {len(all_variants)} variants: {sorted(all_variants)}")
    print(f"Found {len(all_layers)} layers: {sorted(all_layers)}")
    print(f"Found {len(all_ranks)} ranks: {sorted(all_ranks)}")
    print(f"Found {len(all_seq_lens)} sequence lengths: {sorted(all_seq_lens)}")

    # For each layer and rank, create a dataframe
    for layer_idx in sorted(all_layers):
        for rank in sorted(all_ranks):
            print(f"\nProcessing Layer {layer_idx}, Rank {rank}...")

            # Collect data for this layer and rank
            df_data_absolute = {}
            df_data_normalized = {}
            seq_lens_for_df = []

            for seq_len in sorted(all_seq_lens):
                # Get default variant vector
                default_key = (layer_idx, rank, seq_len, "default")
                if default_key not in data:
                    print(
                        f"  Warning: Missing default for seq_len={seq_len}, skipping..."
                    )
                    continue

                default_tensor = data[default_key]
                seq_lens_for_df.append(seq_len)

                # Calculate distances for each non-default variant
                for variant in non_default_variants:
                    variant_key = (layer_idx, rank, seq_len, variant)
                    if variant_key not in data:
                        print(f"  Warning: Missing {variant} for seq_len={seq_len}")
                        distance_abs = np.nan
                        distance_norm = np.nan
                    else:
                        variant_tensor = data[variant_key]
                        distance_abs = euclidean_distance(
                            variant_tensor, default_tensor
                        )
                        distance_norm = normalized_euclidean_distance(
                            variant_tensor, default_tensor
                        )

                    if variant not in df_data_absolute:
                        df_data_absolute[variant] = []
                        df_data_normalized[variant] = []
                    df_data_absolute[variant].append(distance_abs)
                    df_data_normalized[variant].append(distance_norm)

            # Create scatter plots
            if seq_lens_for_df:
                markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*"]

                # Plot 1: Absolute euclidean distance
                plt.figure(figsize=(10, 6))
                for idx, variant in enumerate(non_default_variants):
                    if variant in df_data_absolute:
                        marker = markers[idx % len(markers)]
                        plt.scatter(
                            seq_lens_for_df,
                            df_data_absolute[variant],
                            label=variant,
                            marker=marker,
                            s=50,
                            alpha=0.7,
                        )

                plt.xlabel("Sequence Length", fontsize=12)
                plt.ylabel("Euclidean Distance from Default", fontsize=12)
                plt.title(
                    f"Layer {layer_idx}, Rank {rank}: "
                    f"Absolute Distance vs Sequence Length",
                    fontsize=14,
                )
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                output_filename_abs = (
                    f"layer_{layer_idx}_rank_{rank}_distances_absolute.png"
                )
                output_path_abs = os.path.join(
                    args.attn_outputs_dir, output_filename_abs
                )
                plt.savefig(output_path_abs, dpi=150, bbox_inches="tight")
                plt.close()

                # Plot 2: Normalized euclidean distance
                plt.figure(figsize=(10, 6))
                for idx, variant in enumerate(non_default_variants):
                    if variant in df_data_normalized:
                        marker = markers[idx % len(markers)]
                        plt.scatter(
                            seq_lens_for_df,
                            df_data_normalized[variant],
                            label=variant,
                            marker=marker,
                            s=50,
                            alpha=0.7,
                        )

                plt.xlabel("Sequence Length", fontsize=12)
                plt.ylabel("Normalized Distance (Distance / Default Norm)", fontsize=12)
                plt.title(
                    f"Layer {layer_idx}, Rank {rank}: "
                    f"Normalized Distance vs Sequence Length",
                    fontsize=14,
                )
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                output_filename_norm = (
                    f"layer_{layer_idx}_rank_{rank}_distances_normalized.png"
                )
                output_path_norm = os.path.join(
                    args.attn_outputs_dir, output_filename_norm
                )
                plt.savefig(output_path_norm, dpi=150, bbox_inches="tight")
                plt.close()

                print(f"  Saved absolute plot to {output_filename_abs}")
                print(f"  Saved normalized plot to {output_filename_norm}")
                print(
                    f"  Plots contain {len(df_data_absolute)} variants and "
                    f"{len(seq_lens_for_df)} sequence lengths"
                )
            else:
                print("  No data to plot for this layer/rank combination")

    print(f"\nAnalysis complete! Results saved to {args.attn_outputs_dir}")
