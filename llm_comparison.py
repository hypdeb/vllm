#!/usr/bin/env python3
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
import re
from typing import List, Dict, Tuple, Any
from collections import Counter
import statistics

# BLEU score implementation
def compute_bleu_score(reference: List[str], candidate: List[str], max_n: int = 4) -> float:
    """Compute BLEU score between reference and candidate token sequences."""
    if not reference or not candidate:
        return 0.0
    
    # Calculate n-gram precision for n=1 to max_n
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = [tuple(reference[i:i+n]) for i in range(len(reference) - n + 1)]
        cand_ngrams = [tuple(candidate[i:i+n]) for i in range(len(candidate) - n + 1)]
        
        if not cand_ngrams:
            precisions.append(0.0)
            continue
            
        ref_counter = Counter(ref_ngrams)
        cand_counter = Counter(cand_ngrams)
        
        overlap = sum(min(ref_counter[ngram], cand_counter[ngram]) for ngram in cand_counter)
        precision = overlap / len(cand_ngrams)
        precisions.append(precision)
    
    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        bleu = (precisions[0] * precisions[1] * precisions[2] * precisions[3]) ** 0.25
    else:
        bleu = 0.0
    
    # Brevity penalty
    bp = min(1.0, len(candidate) / len(reference)) if len(reference) > 0 else 0.0
    
    return bp * bleu

def edit_distance(seq1: List[str], seq2: List[str]) -> int:
    """Compute edit distance (Levenshtein distance) between two token sequences."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]

def longest_common_subsequence(seq1: List[str], seq2: List[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

def compute_rouge_l(reference: List[str], candidate: List[str]) -> float:
    """Compute ROUGE-L score based on longest common subsequence."""
    if not reference or not candidate:
        return 0.0
    
    lcs_length = longest_common_subsequence(reference, candidate)
    precision = lcs_length / len(candidate) if candidate else 0.0
    recall = lcs_length / len(reference) if reference else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def compute_rouge_n(reference: List[str], candidate: List[str], n: int = 1) -> float:
    """Compute ROUGE-n score for n-grams."""
    if not reference or not candidate:
        return 0.0
    
    ref_ngrams = [tuple(reference[i:i+n]) for i in range(len(reference) - n + 1)]
    cand_ngrams = [tuple(candidate[i:i+n]) for i in range(len(candidate) - n + 1)]
    
    if not ref_ngrams:
        return 0.0
    
    ref_counter = Counter(ref_ngrams)
    cand_counter = Counter(cand_ngrams)
    
    overlap = sum(min(ref_counter[ngram], cand_counter[ngram]) for ngram in cand_counter)
    recall = overlap / len(ref_ngrams)
    
    return recall

def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def tokenize_text(text: str) -> List[str]:
    """Simple tokenization by splitting on whitespace and punctuation."""
    # Remove extra whitespace and split
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
    return [token for token in tokens if token.strip()]

def token_overlap_ratio(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute ratio of overlapping tokens."""
    if not tokens1 and not tokens2:
        return 1.0
    
    if not tokens1 or not tokens2:
        return 0.0
    
    counter1 = Counter(tokens1)
    counter2 = Counter(tokens2)
    
    overlap = sum(min(counter1[token], counter2[token]) for token in counter1 if token in counter2)
    total = max(len(tokens1), len(tokens2))
    
    return overlap / total

def compare_outputs(text1: str, text2: str, tokens1: List[int], tokens2: List[int]) -> Dict[str, float]:
    """Compare two outputs using multiple similarity measures."""
    # Tokenize text
    text_tokens1 = tokenize_text(text1)
    text_tokens2 = tokenize_text(text2)
    
    # Convert token IDs to strings for processing
    token_ids1 = [str(token) for token in tokens1]
    token_ids2 = [str(token) for token in tokens2]
    
    results = {}
    
    # Text-based similarity measures
    results['token_overlap_ratio'] = token_overlap_ratio(text_tokens1, text_tokens2)
    results['edit_distance'] = edit_distance(text_tokens1, text_tokens2)
    results['edit_distance_normalized'] = 1.0 - (results['edit_distance'] / max(len(text_tokens1), len(text_tokens2), 1))
    results['bleu_score'] = compute_bleu_score(text_tokens1, text_tokens2)
    results['rouge_1'] = compute_rouge_n(text_tokens1, text_tokens2, n=1)
    results['rouge_2'] = compute_rouge_n(text_tokens1, text_tokens2, n=2)
    results['rouge_l'] = compute_rouge_l(text_tokens1, text_tokens2)
    results['lcs_length'] = longest_common_subsequence(text_tokens1, text_tokens2)
    results['lcs_ratio'] = results['lcs_length'] / max(len(text_tokens1), len(text_tokens2), 1)
    results['jaccard_similarity'] = jaccard_similarity(set(text_tokens1), set(text_tokens2))
    
    # Token ID-based similarity measures
    results['token_id_overlap'] = token_overlap_ratio(token_ids1, token_ids2)
    results['token_id_edit_distance'] = edit_distance(token_ids1, token_ids2)
    results['token_id_edit_distance_normalized'] = 1.0 - (results['token_id_edit_distance'] / max(len(token_ids1), len(token_ids2), 1))
    results['token_id_jaccard'] = jaccard_similarity(set(token_ids1), set(token_ids2))
    
    return results

def load_dataset(filepath: str) -> Dict[str, Any]:
    """Load dataset from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_datasets(dataset1_path: str, dataset2_path: str) -> None:
    """Compare two datasets and print comprehensive similarity analysis."""
    print("Loading datasets...")
    dataset1 = load_dataset(dataset1_path)
    dataset2 = load_dataset(dataset2_path)
    
    print(f"Dataset 1: {dataset1['model']}")
    print(f"Dataset 2: {dataset2['model']}")
    print(f"Number of prompt-output pairs: {len(dataset1['prompt_output_pairs'])}")
    print()
    
    all_similarities = []
    
    # Compare each pair
    for i, (pair1, pair2) in enumerate(zip(dataset1['prompt_output_pairs'], dataset2['prompt_output_pairs'])):
        if pair1['pair_id'] != pair2['pair_id']:
            print(f"Warning: Mismatched pair IDs at index {i}: {pair1['pair_id']} vs {pair2['pair_id']}")
        
        # Parse token IDs (they're stored as strings)
        try:
            tokens1 = json.loads(pair1['output_token_ids'])
            tokens2 = json.loads(pair2['output_token_ids'])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing tokens for pair {pair1['pair_id']}: {e}")
            continue
        
        similarities = compare_outputs(
            pair1['output_text'], 
            pair2['output_text'],
            tokens1,
            tokens2
        )
        
        similarities['pair_id'] = pair1['pair_id']
        all_similarities.append(similarities)
        
        print(f"Pair {pair1['pair_id']}:")
        print(f"  Token overlap ratio: {similarities['token_overlap_ratio']:.4f}")
        print(f"  Edit distance (normalized): {similarities['edit_distance_normalized']:.4f}")
        print(f"  BLEU score: {similarities['bleu_score']:.4f}")
        print(f"  ROUGE-1: {similarities['rouge_1']:.4f}")
        print(f"  ROUGE-2: {similarities['rouge_2']:.4f}")
        print(f"  ROUGE-L: {similarities['rouge_l']:.4f}")
        print(f"  Jaccard similarity: {similarities['jaccard_similarity']:.4f}")
        print(f"  Token ID overlap: {similarities['token_id_overlap']:.4f}")
        print()
    
    # Compute aggregate statistics
    print("=" * 50)
    print("AGGREGATE STATISTICS")
    print("=" * 50)
    
    metrics = [
        'token_overlap_ratio', 'edit_distance_normalized', 'bleu_score',
        'rouge_1', 'rouge_2', 'rouge_l', 'jaccard_similarity', 'token_id_overlap'
    ]
    
    for metric in metrics:
        values = [sim[metric] for sim in all_similarities]
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
    token_overlaps = [(sim['pair_id'], sim['token_overlap_ratio']) for sim in all_similarities]
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