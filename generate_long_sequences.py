#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Generate realistic text sequences of varying token lengths for testing purposes.
Produces sequences of 40k+ tokens using sample prompts as seeds.
"""

import random

from transformers import AutoTokenizer


class RealisticTextGenerator:
    def __init__(
        self,
        prompts_file: str = "sample_prompts.txt",
        tokenizer_path: str = "/trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct-FP8",
    ):
        self.prompts = self._load_prompts(prompts_file)

        # Load the tokenizer
        print(f"Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"Tokenizer loaded successfully. Vocab size: {len(self.tokenizer)}")

        # Common text patterns for realistic expansion
        self.sentence_starters = [
            "Furthermore, ",
            "In addition, ",
            "Moreover, ",
            "However, ",
            "Nevertheless, ",
            "On the other hand, ",
            "For instance, ",
            "For example, ",
            "Similarly, ",
            "In contrast, ",
            "As a result, ",
            "Consequently, ",
            "Therefore, ",
            "It should be noted that ",
            "Interestingly, ",
            "Remarkably, ",
            "Studies have shown that ",
            "Research indicates that ",
            "Experts suggest that ",
            "According to recent findings, ",
            "It has been observed that ",
        ]

        self.transition_phrases = [
            "moving forward",
            "in the next phase",
            "building upon this",
            "expanding on this concept",
            "diving deeper into the subject",
            "exploring further",
            "continuing this line of thought",
            "examining the implications",
            "considering alternative perspectives",
            "analyzing the broader context",
            "investigating related aspects",
        ]

        self.filler_content = [
            "This comprehensive analysis delves into the multifaceted nature of the subject matter, "
            "providing detailed insights and thorough examination of various components.",
            "The intricate relationships between different elements create a complex web of "
            "interdependencies that require careful consideration and systematic evaluation.",
            "Through extensive research and careful observation, patterns emerge that help us "
            "understand the underlying mechanisms and driving forces at play.",
            "The methodology employed in this investigation follows established protocols "
            "while incorporating innovative approaches to ensure comprehensive coverage.",
            "Data collection and analysis reveal significant trends that have important "
            "implications for future development and strategic planning initiatives.",
            "Contemporary perspectives on this topic have evolved considerably, reflecting "
            "changing societal needs and technological advancements in the field.",
            "The implementation of these concepts requires careful planning and systematic "
            "execution to achieve optimal results and minimize potential complications.",
            "Historical context provides valuable insights into the evolution of current "
            "practices and helps identify successful strategies for future application.",
        ]

    def _load_prompts(self, filename: str) -> list[tuple[int, str]]:
        """Load prompts from file and parse them."""
        prompts = []
        try:
            with open(filename, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and "," in line:
                        parts = line.split(",", 1)
                        if len(parts) == 2:
                            try:
                                score = int(parts[0])
                                prompt = parts[1].strip()
                                prompts.append((score, prompt))
                            except ValueError:
                                continue
        except FileNotFoundError:
            print(f"Warning: {filename} not found. Using default prompts.")
            prompts = [
                (50, "Write about artificial intelligence and its impact on society."),
                (75, "Describe a complex scientific process in detail."),
                (100, "Create a comprehensive analysis of modern technology trends."),
                (60, "Explain the importance of sustainable development."),
                (80, "Discuss the evolution of communication technologies."),
            ]
        return prompts

    def _get_token_count(self, text: str) -> int:
        """Get the token count for the given text."""
        return len(self.tokenizer.encode(text))

    def _expand_text(self, seed_text: str, target_token_length: int) -> str:
        """Expand seed text to reach target token length with realistic content."""
        result = seed_text + " "

        while self._get_token_count(result) < target_token_length:
            # Add transition phrase
            addition = random.choice(self.sentence_starters)

            # Add filler content
            addition += random.choice(self.filler_content) + " "

            # Add some variation with transition phrases
            if random.random() < 0.3:
                addition += (
                    f"When {random.choice(self.transition_phrases)}, we observe that "
                )
                addition += random.choice(self.filler_content) + " "

            # Add numbered points occasionally (without newlines)
            if random.random() < 0.15:
                addition += f"{random.randint(1, 10)}. "
                addition += random.choice(self.filler_content) + " "

            # Add some repetition with variation
            current_token_count = self._get_token_count(result)
            if current_token_count > 1000 and random.random() < 0.2:
                # Take a previous sentence and modify it slightly
                sentences = result.split(". ")
                if len(sentences) > 3:
                    base_sentence = sentences[random.randint(0, len(sentences) // 2)]
                    variations = [
                        f"As previously mentioned, {base_sentence.lower()}",
                        f"Returning to the concept that {base_sentence.lower()}",
                        f"It's worth reiterating that {base_sentence.lower()}",
                    ]
                    addition += random.choice(variations) + ". "

            # Check if adding this would exceed our target
            test_result = result + addition
            if self._get_token_count(test_result) > target_token_length:
                # Add words one by one until we reach the target
                words = addition.split()
                for word in words:
                    test_word = result + word + " "
                    if self._get_token_count(test_word) <= target_token_length:
                        result = test_word
                    else:
                        break
                break
            else:
                result += addition

        # Remove any newlines and clean up the text
        result = " ".join(result.split())  # Remove all newlines and extra spaces

        # Final token count check and trim if necessary
        while self._get_token_count(result) > target_token_length:
            words = result.split()
            if len(words) > 1:
                result = " ".join(words[:-1])
            else:
                break

        return result

    def generate_sequence(
        self, target_token_length: int, complexity_range: tuple[int, int] | None = None
    ) -> str:
        """Generate a single realistic text sequence of specified token length."""
        if complexity_range:
            min_complexity, max_complexity = complexity_range
            suitable_prompts = [
                p for p in self.prompts if min_complexity <= p[0] <= max_complexity
            ]
        else:
            suitable_prompts = self.prompts

        if not suitable_prompts:
            suitable_prompts = self.prompts

        # Select a random prompt as seed
        score, prompt = random.choice(suitable_prompts)

        # Expand the prompt to target token length
        expanded_text = self._expand_text(prompt, target_token_length)

        return expanded_text

    def generate_multiple_sequences(
        self,
        count: int,
        min_token_length: int = 10000,
        max_token_length: int = 25000,
        complexity_range: tuple[int, int] | None = None,
    ) -> list[str]:
        """Generate multiple sequences with varying token lengths."""
        sequences = []

        for i in range(count):
            # Random token length within the specified range
            target_token_length = random.randint(min_token_length, max_token_length)

            # Generate sequence
            sequence = self.generate_sequence(target_token_length, complexity_range)
            actual_token_count = self._get_token_count(sequence)

            sequences.append(sequence)

            print(
                f"Generated sequence {i + 1}/{count} - Target tokens: {target_token_length:,}, Actual tokens: {actual_token_count:,}, Characters: {len(sequence):,}"
            )

        return sequences

    def save_sequences_to_single_file(
        self, sequences: list[str], filename: str = "generated_sequences.txt"
    ):
        """Save generated sequences to a single file in sample_prompts format."""
        with open(filename, "w", encoding="utf-8") as f:
            for sequence in sequences:
                # Format: length,text (no newlines in text)
                f.write(f"{2},{sequence}\n")
        print(f"Saved {len(sequences)} sequences to: {filename}")

    def save_sequences(self, sequences: list[str], base_filename: str = "sequence"):
        """Save generated sequences to separate files (legacy method)."""
        for i, sequence in enumerate(sequences):
            token_count = self._get_token_count(sequence)
            filename = f"{base_filename}_{i + 1}_{token_count}tokens.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(sequence)
            print(f"Saved: {filename}")


def main():
    """Main function to demonstrate the text generator."""
    generator = RealisticTextGenerator()

    print("Realistic Text Sequence Generator (Token-based)")
    print("=" * 50)

    # Generate sequences with different configurations
    print("\nGenerating sequences with varying token lengths ...")
    sequences = generator.generate_multiple_sequences(
        count=1, min_token_length=64000, max_token_length=64000
    )

    # # Generate a few very long sequences
    # print("\nGenerating 2 very long sequences (20k-40k tokens)...")
    # long_sequences = generator.generate_multiple_sequences(
    #     count=1,
    #     min_token_length=20000,
    #     max_token_length=40000,
    #     complexity_range=(50, 100)  # Higher complexity prompts
    # )

    # Combine all sequences

    all_sequences = sequences

    # Save all sequences to a single file in sample_prompts format
    print("\nSaving all sequences to single file...")
    generator.save_sequences_to_single_file(all_sequences, "generated_sequences.txt")

    # Print statistics
    total_tokens = sum(generator._get_token_count(s) for s in all_sequences)
    total_chars = sum(len(s) for s in all_sequences)

    print("\nGeneration complete!")
    print(f"Total sequences generated: {len(all_sequences)}")
    print(f"Total tokens generated: {total_tokens:,}")
    print(f"Total characters generated: {total_chars:,}")
    print(f"Average tokens per sequence: {total_tokens // len(all_sequences):,}")
    print(f"Average characters per token: {total_chars / total_tokens:.2f}")
    print("Sequences saved in sample_prompts.txt format to: generated_sequences.txt")


if __name__ == "__main__":
    main()
