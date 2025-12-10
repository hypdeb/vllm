#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LLM Output Comparison Tool

This script compares outputs from two LLM datasets using multiple similarity measures:
- Token overlap ratio
- Edit distance (Levenshtein)
- BLEU score
- ROUGE scores
- Longest Common Subsequence (LCS)
- Jaccard similarity
"""

import json
import statistics
from typing import Any


# BLEU score implementation
def edit_distance(seq1: list[str], seq2: list[str]) -> int:
    """Compute edit distance (Levenshtein distance) between two token sequences."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union if union > 0 else 0.0


def token_overlap_ratio(tokens1: list[str], tokens2: list[str]) -> float:
    """Compute ratio of overlapping tokens."""
    equalities = [token1 == token2 for token1, token2 in zip(tokens1, tokens2)]
    return sum(equalities) / len(equalities) if equalities else 0.0


def compare_outputs(
    text1: str, text2: str, tokens1: list[int], tokens2: list[int]
) -> dict[str, float]:
    """Compare two outputs using multiple similarity measures."""
    # Tokenize text

    # Convert token IDs to strings for processing
    token_ids1 = [str(token) for token in tokens1]
    token_ids2 = [str(token) for token in tokens2]

    results = {}

    # Token ID-based similarity measures
    results["token_id_overlap"] = token_overlap_ratio(token_ids1, token_ids2)
    results["token_id_edit_distance"] = edit_distance(token_ids1, token_ids2)
    results["token_id_edit_distance_normalized"] = 1.0 - (
        results["token_id_edit_distance"] / max(len(token_ids1), len(token_ids2), 1)
    )
    results["token_id_jaccard"] = jaccard_similarity(set(token_ids1), set(token_ids2))

    return results


def load_dataset(filepath: str) -> dict[str, Any]:
    """Load dataset from JSON file."""
    with open(filepath, encoding="utf-8") as f:
        json_data = json.load(f)
        return json_data["prompt_output_pairs"]


def compare_datasets(dataset1_path: str, dataset2_path: str) -> None:
    """Compare two datasets and print comprehensive similarity analysis."""
    print("Loading datasets...")
    dataset1 = load_dataset(dataset1_path)
    dataset2 = load_dataset(dataset2_path)

    print(f"Number of prompt-output pairs: {len(dataset1)}")
    print()

    all_similarities = []

    # Compare each pair
    for i, (pair1, pair2) in enumerate(zip(dataset1, dataset2)):
        if pair1["pair_id"] != pair2["pair_id"]:
            print(
                f"Warning: Mismatched pair IDs at index {i}: {pair1['pair_id']} vs {pair2['pair_id']}"
            )

        # Parse token IDs (they're stored as strings)
        try:
            tokens1 = json.loads(pair1["output_token_ids"])
            tokens2 = json.loads(pair2["output_token_ids"])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing tokens for pair {pair1['pair_id']}: {e}")
            continue

        similarities = compare_outputs(
            pair1["output_text"], pair2["output_text"], tokens1, tokens2
        )

        similarities["pair_id"] = pair1["pair_id"]
        all_similarities.append(similarities)

    # Compute aggregate statistics
    print("=" * 50)
    print("AGGREGATE STATISTICS")
    print("=" * 50)

    metrics = [
        "token_id_overlap",
        "token_id_edit_distance_normalized",
        "token_id_jaccard",
    ]

    for metric in metrics:
        values = [sim[metric] for sim in all_similarities if metric in sim]

        if not values:
            print(f"{metric.replace('_', ' ').title()}: Not available")
            continue

        mean_val = statistics.mean(values)
        median_val = statistics.median(values)
        min_val = min(values)
        max_val = max(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0

        print(f"{metric.replace('_', ' ').title()}:")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Median: {median_val:.4f}")
        print(f"  Min: {min_val:.4f}")
        print(f"  Max: {max_val:.4f}")
        print(f"  Std Dev: {std_val:.4f}")
        print()

    # Find most and least similar pairs
    token_overlaps = [
        (sim["pair_id"], sim["token_id_overlap"]) for sim in all_similarities
    ]
    token_overlaps.sort(key=lambda x: x[1], reverse=True)

    print("Most similar pairs (by token overlap):")
    for pair_id, overlap in token_overlaps[:3]:
        print(f"  Pair {pair_id}: {overlap:.4f}")

    print("\nLeast similar pairs (by token overlap):")
    for pair_id, overlap in token_overlaps[-3:]:
        print(f"  Pair {pair_id}: {overlap:.4f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python llm_comparison.py <dataset1.json> <dataset2.json>")
        sys.exit(1)

    dataset1_path = sys.argv[1]
    dataset2_path = sys.argv[2]

    compare_datasets(dataset1_path, dataset2_path)
