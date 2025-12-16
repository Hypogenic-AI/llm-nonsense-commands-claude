"""
Perplexity Calculator Module

Calculates perplexity of prompts using GPT-2 model.
Perplexity measures how "surprising" a text is to a language model.
Higher perplexity = more unexpected/nonsensical text.
"""

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Dict, Union
import warnings

# Suppress transformers warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class PerplexityCalculator:
    """Calculate perplexity using GPT-2."""

    def __init__(self, model_name: str = "gpt2"):
        """Initialize with specified model."""
        print(f"Loading model {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")

    def calculate_perplexity(self, text: str, max_length: int = 512) -> float:
        """
        Calculate perplexity for a single text.

        Returns perplexity value. Lower = more likely/natural text.
        Very high values (>1000) indicate very unusual text.
        """
        if not text or text.strip() == "":
            return float("inf")

        try:
            # Tokenize
            encodings = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )

            input_ids = encodings.input_ids.to(self.device)

            # Handle case where text is too short
            if input_ids.size(1) < 2:
                return float("inf")

            # Calculate loss
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss

            # Convert loss to perplexity
            perplexity = torch.exp(loss).item()

            # Cap extremely high perplexities
            if perplexity > 1e10 or np.isnan(perplexity) or np.isinf(perplexity):
                return 1e10

            return perplexity

        except Exception as e:
            print(f"Error calculating perplexity for '{text[:50]}...': {e}")
            return float("inf")

    def calculate_batch_perplexity(self, texts: List[str]) -> List[float]:
        """Calculate perplexity for a batch of texts."""
        return [self.calculate_perplexity(text) for text in texts]

    def get_perplexity_stats(self, texts: List[str]) -> Dict[str, float]:
        """Calculate perplexity statistics for a list of texts."""
        perplexities = self.calculate_batch_perplexity(texts)
        valid_perplexities = [p for p in perplexities if p < float("inf") and p < 1e10]

        if not valid_perplexities:
            return {
                "mean": float("inf"),
                "std": float("inf"),
                "min": float("inf"),
                "max": float("inf"),
                "median": float("inf"),
            }

        return {
            "mean": np.mean(valid_perplexities),
            "std": np.std(valid_perplexities),
            "min": np.min(valid_perplexities),
            "max": np.max(valid_perplexities),
            "median": np.median(valid_perplexities),
        }


def normalize_perplexity(perplexity: float, min_ppl: float = 1.0, max_ppl: float = 10000.0) -> float:
    """
    Normalize perplexity to 0-1 scale using log transformation.

    0 = very natural text (low perplexity)
    1 = very unusual text (high perplexity)
    """
    if perplexity <= min_ppl:
        return 0.0
    if perplexity >= max_ppl:
        return 1.0

    # Log-scale normalization
    log_ppl = np.log(perplexity)
    log_min = np.log(min_ppl)
    log_max = np.log(max_ppl)

    normalized = (log_ppl - log_min) / (log_max - log_min)
    return min(max(normalized, 0.0), 1.0)


if __name__ == "__main__":
    # Test perplexity calculation
    calc = PerplexityCalculator()

    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "What is the capital of France?",
        "Explain why pickles orbit the moon on Tuesdays.",
        "Capital what France of is the?",
        "elephant bicycle quantum whisper mountain",
        "xyz123 !@# elephant &*( abc",
        "∆π∑∂ Ω∞≈ç√∫ ß∂ƒ©˙∆˚¬",
    ]

    print("=" * 60)
    print("Perplexity Calculation Test")
    print("=" * 60)

    for text in test_texts:
        ppl = calc.calculate_perplexity(text)
        norm_ppl = normalize_perplexity(ppl)
        print(f"\nText: {text[:50]}...")
        print(f"  Perplexity: {ppl:.2f}")
        print(f"  Normalized (0-1): {norm_ppl:.4f}")
