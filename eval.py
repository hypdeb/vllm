# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import csv
import glob
import json
import os

import numpy as np
import requests


def score_answer(model_answer: str, true_target: str):
    print(f"model_answer: {model_answer}, true_target: {true_target}")
    if true_target.startswith("["):
        return float(int(str(model_answer) == true_target))
    try:
        model_answer_int = int(model_answer)
        norm = np.abs(float(true_target))
        err = min(
            1.0,
            np.abs(float(true_target) - model_answer_int) / (1e-10 + norm),
        )
    except (ValueError, TypeError):
        err = 1.0
    return 1.0 - err


def create_request(question):
    prompt = (
        f"What is the output of the print statement at the end of this "
        f"Python code? Respond with just the output of the print statement "
        f"and nothing else.\n{question}"
    )
    data = {
        "model": "default_model",
        "temperature": 0,
        "max_tokens": 200,
        "top_p": 1,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "response_format": {"type": "text"},
        "stop": ["\n", "."],
        "skip_special_tokens": True,
        "prediction": {"type": "content", "content": ""},
    }
    return data


def get_response(data):
    url = "http://localhost:8000/v1/chat/completions"
    response = requests.post(url, json=data, timeout=600)
    response.raise_for_status()
    result = response.json()
    print(f"result: {result}")
    # Transform OpenAI format to expected format
    return {"message": {"content": result["choices"][0]["message"]["content"]}}


def evaluate_request(item):
    """Evaluate a single request and return the results."""
    question = item["data"]["question"]
    answer = item["data"]["answer"][
        0
    ]  # The answer is always one pre-concatenated string.
    uid = item["uid"]

    if not item["use_for_test"]:
        return None

    try:
        data = create_request(question)
        response = get_response(data)
        model_answer = response["message"]["content"].strip().strip(" >")
        score = score_answer(model_answer, answer)

        return {
            "uid": uid,
            "expected_output": answer,
            "actual_output": model_answer,
            "score": score,
        }
    except Exception as e:
        return {
            "uid": uid,
            "expected_output": answer,
            "actual_output": f"ERROR: {str(e)}",
            "score": 0.0,
        }


def main():
    """
    Process all .jsonl files in /scratch/samples/eval/.

    Output results to CSV file.
    """
    input_dir = "/scratch/samples/eval/"
    output_file = "evaluation_results.csv"

    # Find all .jsonl files in the directory
    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl"))

    if not jsonl_files:
        print(f"No .jsonl files found in {input_dir}")
        return

    print(f"Found {len(jsonl_files)} .jsonl file(s)")

    # Collect all results
    results = []

    for jsonl_file in sorted(jsonl_files):
        print(f"Processing {os.path.basename(jsonl_file)}...")

        with open(jsonl_file) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())

                    # Only process test items
                    if item.get("use_for_test", False):
                        print(f"  Evaluating item {line_num}: {item['uid']}")
                        result = evaluate_request(item)
                        if result:
                            results.append(result)
                except json.JSONDecodeError as e:
                    print(f"  Warning: Failed to parse line {line_num}: {e}")
                except Exception as e:
                    print(f"  Warning: Error processing line {line_num}: {e}")

    # Write results to CSV
    if results:
        with open(output_file, "w", newline="") as csvfile:
            fieldnames = ["uid", "expected_output", "actual_output", "score"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in results:
                writer.writerow(result)

        print(f"\nEvaluation complete! Results written to {output_file}")
        print(f"Total evaluations: {len(results)}")

        # Print summary statistics
        scores = [r["score"] for r in results]
        print(f"Average score: {np.mean(scores):.4f}")
        print(f"Min score: {np.min(scores):.4f}")
        print(f"Max score: {np.max(scores):.4f}")
    else:
        print("No test items found to evaluate.")


if __name__ == "__main__":
    main()
