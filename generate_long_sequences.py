#!/usr/bin/env python3
"""
Generate realistic text sequences of varying lengths for testing purposes.
Produces sequences of 40k+ characters using sample prompts as seeds.
"""

import random
import re
from typing import List, Tuple, Optional
from pathlib import Path

class RealisticTextGenerator:
    def __init__(self, prompts_file: str = "sample_prompts.txt"):
        self.prompts = self._load_prompts(prompts_file)
        
        # Common text patterns for realistic expansion
        self.sentence_starters = [
            "Furthermore, ", "In addition, ", "Moreover, ", "However, ", "Nevertheless, ",
            "On the other hand, ", "For instance, ", "For example, ", "Similarly, ",
            "In contrast, ", "As a result, ", "Consequently, ", "Therefore, ",
            "It should be noted that ", "Interestingly, ", "Remarkably, ",
            "Studies have shown that ", "Research indicates that ", "Experts suggest that ",
            "According to recent findings, ", "It has been observed that ",
        ]
        
        self.transition_phrases = [
            "moving forward", "in the next phase", "building upon this",
            "expanding on this concept", "diving deeper into the subject",
            "exploring further", "continuing this line of thought",
            "examining the implications", "considering alternative perspectives",
            "analyzing the broader context", "investigating related aspects"
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
            "practices and helps identify successful strategies for future application."
        ]

    def _load_prompts(self, filename: str) -> List[Tuple[int, str]]:
        """Load prompts from file and parse them."""
        prompts = []
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and ',' in line:
                        parts = line.split(',', 1)
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
                (80, "Discuss the evolution of communication technologies.")
            ]
        return prompts

    def _expand_text(self, seed_text: str, target_length: int) -> str:
        """Expand seed text to reach target length with realistic content."""
        result = seed_text + " "
        
        while len(result) < target_length:
            # Add transition phrase
            result += random.choice(self.sentence_starters)
            
            # Add filler content
            result += random.choice(self.filler_content) + " "
            
            # Add some variation with transition phrases
            if random.random() < 0.3:
                result += f"When {random.choice(self.transition_phrases)}, we observe that "
                result += random.choice(self.filler_content) + " "
            
            # Add numbered points occasionally (without newlines)
            if random.random() < 0.15:
                result += f"{random.randint(1, 10)}. "
                result += random.choice(self.filler_content) + " "
            
            # Add some repetition with variation
            if len(result) > 1000 and random.random() < 0.2:
                # Take a previous sentence and modify it slightly
                sentences = result.split('. ')
                if len(sentences) > 3:
                    base_sentence = sentences[random.randint(0, len(sentences)//2)]
                    variations = [
                        f"As previously mentioned, {base_sentence.lower()}",
                        f"Returning to the concept that {base_sentence.lower()}",
                        f"It's worth reiterating that {base_sentence.lower()}"
                    ]
                    result += random.choice(variations) + ". "
        
        # Remove any newlines and clean up the text
        result = result[:target_length]
        result = ' '.join(result.split())  # Remove all newlines and extra spaces
        return result

    def generate_sequence(self, target_length: int, complexity_range: Optional[Tuple[int, int]] = None) -> str:
        """Generate a single realistic text sequence of specified length."""
        if complexity_range:
            min_complexity, max_complexity = complexity_range
            suitable_prompts = [p for p in self.prompts if min_complexity <= p[0] <= max_complexity]
        else:
            suitable_prompts = self.prompts
        
        if not suitable_prompts:
            suitable_prompts = self.prompts
        
        # Select a random prompt as seed
        score, prompt = random.choice(suitable_prompts)
        
        # Expand the prompt to target length
        expanded_text = self._expand_text(prompt, target_length)
        
        return expanded_text

    def generate_multiple_sequences(self, 
                                  count: int, 
                                  min_length: int = 40000, 
                                  max_length: int = 100000,
                                  complexity_range: Optional[Tuple[int, int]] = None) -> List[str]:
        """Generate multiple sequences with varying lengths."""
        sequences = []
        
        for i in range(count):
            # Random length within the specified range
            target_length = random.randint(min_length, max_length)
            
            # Generate sequence
            sequence = self.generate_sequence(target_length, complexity_range)
            sequences.append(sequence)
            
            print(f"Generated sequence {i+1}/{count} - Length: {len(sequence):,} characters")
        
        return sequences

    def save_sequences_to_single_file(self, sequences: List[str], filename: str = "generated_sequences.txt"):
        """Save generated sequences to a single file in sample_prompts format."""
        with open(filename, 'w', encoding='utf-8') as f:
            for sequence in sequences:
                # Format: length,text (no newlines in text)
                f.write(f"{2},{sequence}\n")
        print(f"Saved {len(sequences)} sequences to: {filename}")
        
    def save_sequences(self, sequences: List[str], base_filename: str = "sequence"):
        """Save generated sequences to separate files (legacy method)."""
        for i, sequence in enumerate(sequences):
            filename = f"{base_filename}_{i+1}_{len(sequence)}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(sequence)
            print(f"Saved: {filename}")

def main():
    """Main function to demonstrate the text generator."""
    generator = RealisticTextGenerator()
    
    print("Realistic Text Sequence Generator")
    print("=" * 40)
    
    # Generate sequences with different configurations
    print("\nGenerating 5 sequences with varying lengths (40k-80k characters)...")
    sequences = generator.generate_multiple_sequences(
        count=1,
        min_length=200000,
        max_length=200000
    )
    
    # # Generate a few very long sequences
    # print("\nGenerating 2 very long sequences (80k-150k characters)...")
    # long_sequences = generator.generate_multiple_sequences(
    #     count=1,
    #     min_length=40000,
    #     max_length=40000,
    #     complexity_range=(50, 100)  # Higher complexity prompts
    # )
    
    # Combine all sequences
    
    all_sequences = sequences
    
    # Save all sequences to a single file in sample_prompts format
    print(f"\nSaving all sequences to single file...")
    generator.save_sequences_to_single_file(all_sequences, "generated_sequences.txt")
    
    print(f"\nGeneration complete!")
    print(f"Total sequences generated: {len(all_sequences)}")
    print(f"Total characters generated: {sum(len(s) for s in all_sequences):,}")
    print(f"Sequences saved in sample_prompts.txt format to: generated_sequences.txt")

if __name__ == "__main__":
    main() 