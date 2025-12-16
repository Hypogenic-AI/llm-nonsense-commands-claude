"""
Prompt Generator Module

Generates prompts across a spectrum from coherent English to complete nonsense.
This allows testing whether LLMs can explain prompts of varying "sensicalness."
"""

import random
import string
from typing import List, Dict, Tuple
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Common English words for generating word salad
COMMON_WORDS = [
    "elephant", "bicycle", "quantum", "whisper", "mountain", "galaxy",
    "chocolate", "symphony", "rainbow", "penguin", "hurricane", "philosophy",
    "telescope", "butterfly", "algorithm", "democracy", "waterfall", "orchestra",
    "carnival", "mystery", "volcano", "infinity", "harmonica", "blueprint",
    "champagne", "kaleidoscope", "labyrinth", "nostalgia", "serenity", "velvet"
]

# Random ASCII characters for token soup
SPECIAL_CHARS = "!@#$%^&*(){}[]<>?/\\|~`"

# Normal coherent questions
NORMAL_PROMPTS = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "What are the main causes of climate change?",
    "Describe the process of water evaporation.",
    "Who invented the telephone?",
    "What is the speed of light?",
    "Explain the concept of gravity.",
    "What are the benefits of regular exercise?",
    "How does the internet work?",
    "What is the chemical formula for water?",
    "Describe the life cycle of a butterfly.",
    "What is the largest planet in our solar system?",
    "Explain the difference between weather and climate.",
    "What causes earthquakes?",
    "How do vaccines work?",
]

# Slightly odd but grammatical prompts
SLIGHTLY_ODD_PROMPTS = [
    "Explain why pickles orbit the moon on Tuesdays.",
    "What is the emotional state of a typical stapler?",
    "Describe the migratory patterns of invisible pandas.",
    "How do mathematical equations feel about being solved?",
    "What advice would Neptune give to aspiring accountants?",
    "Explain the philosophy of sentient refrigerators.",
    "Why do prime numbers hold secret meetings?",
    "Describe the cultural traditions of subatomic particles.",
    "What career opportunities exist for time-traveling bananas?",
    "How does the color purple experience jealousy?",
    "Explain the diplomatic relations between vowels and consonants.",
    "What motivates clouds to pursue higher education?",
    "Describe the spiritual beliefs of office furniture.",
    "Why do parallel lines feel lonely?",
    "How do paradoxes celebrate their birthdays?",
]


def generate_broken_grammar_prompts(n: int = 15) -> List[str]:
    """Generate prompts with broken grammar but real words."""
    bases = [
        ("What is the capital of France?", "Capital what France of is the?"),
        ("How does photosynthesis work?", "Work photosynthesis does how?"),
        ("Explain the causes of climate change.", "Climate causes explain the change of."),
        ("Describe the water cycle.", "Water the cycle describe."),
        ("What are the benefits of exercise?", "Exercise benefits the what are of?"),
    ]

    prompts = []
    for original, broken in bases:
        prompts.append(broken)

    # Generate more by shuffling words from normal prompts
    for prompt in NORMAL_PROMPTS[:n-len(bases)]:
        words = prompt.replace("?", "").replace(".", "").split()
        random.shuffle(words)
        shuffled = " ".join(words) + "?"
        prompts.append(shuffled)

    return prompts[:n]


def generate_word_salad_prompts(n: int = 15) -> List[str]:
    """Generate prompts with real words but no semantic structure."""
    prompts = []
    for _ in range(n):
        num_words = random.randint(5, 10)
        words = random.sample(COMMON_WORDS, min(num_words, len(COMMON_WORDS)))
        prompt = " ".join(words)
        prompts.append(prompt)
    return prompts


def generate_token_soup_prompts(n: int = 15) -> List[str]:
    """Generate prompts with mix of tokens, numbers, and special characters."""
    prompts = []
    for _ in range(n):
        parts = []
        for _ in range(random.randint(5, 8)):
            choice = random.random()
            if choice < 0.3:
                # Random word
                parts.append(random.choice(COMMON_WORDS))
            elif choice < 0.5:
                # Random number
                parts.append(str(random.randint(0, 999)))
            elif choice < 0.7:
                # Random special chars
                num_chars = random.randint(2, 4)
                parts.append("".join(random.choices(SPECIAL_CHARS, k=num_chars)))
            else:
                # Random letters
                num_chars = random.randint(2, 5)
                parts.append("".join(random.choices(string.ascii_lowercase, k=num_chars)))
        prompts.append(" ".join(parts))
    return prompts


def generate_pure_random_prompts(n: int = 15) -> List[str]:
    """Generate prompts with completely random characters."""
    prompts = []
    extended_chars = string.ascii_letters + string.digits + SPECIAL_CHARS + "    "
    unicode_chars = "∆π∑∂Ω∞≈ç√∫ß∂ƒ©˙∆˚¬…æœ∑®†¥"
    all_chars = extended_chars + unicode_chars

    for _ in range(n):
        length = random.randint(20, 50)
        prompt = "".join(random.choices(all_chars, k=length))
        prompts.append(prompt)
    return prompts


def generate_adversarial_like_prompts(n: int = 15) -> List[str]:
    """
    Generate prompts that mimic GCG-style adversarial suffixes.
    These are token sequences that don't make sense but have structure.
    """
    prompts = []

    # Patterns inspired by GCG-style outputs (but not actual attacks)
    prefixes = ["Sure", "Here", "Step", "First", "I", "The", "To", "For"]
    suffixes = [".", "!", ":", " is", " be", " do"]

    for _ in range(n):
        parts = []
        for i in range(random.randint(6, 12)):
            if i == 0:
                parts.append(random.choice(prefixes))
            elif random.random() < 0.3:
                # Repeated characters (common in adversarial)
                char = random.choice(string.ascii_lowercase)
                parts.append(char * random.randint(2, 4))
            elif random.random() < 0.5:
                # Fragment from common word
                word = random.choice(COMMON_WORDS)
                start = random.randint(0, len(word)//2)
                end = random.randint(start+2, len(word))
                parts.append(word[start:end])
            else:
                # Random short token
                parts.append("".join(random.choices(string.ascii_lowercase, k=random.randint(2, 4))))

        prompt = " ".join(parts) + random.choice(suffixes)
        prompts.append(prompt)

    return prompts


def create_prompt_dataset(samples_per_category: int = 15) -> Dict[str, List[str]]:
    """Create a dataset of prompts across all categories."""
    return {
        "normal": NORMAL_PROMPTS[:samples_per_category],
        "slightly_odd": SLIGHTLY_ODD_PROMPTS[:samples_per_category],
        "broken_grammar": generate_broken_grammar_prompts(samples_per_category),
        "word_salad": generate_word_salad_prompts(samples_per_category),
        "token_soup": generate_token_soup_prompts(samples_per_category),
        "pure_random": generate_pure_random_prompts(samples_per_category),
        "adversarial_like": generate_adversarial_like_prompts(samples_per_category),
    }


def get_category_order() -> List[str]:
    """Return categories in order of expected perplexity (low to high)."""
    return [
        "normal",
        "slightly_odd",
        "broken_grammar",
        "word_salad",
        "adversarial_like",
        "token_soup",
        "pure_random",
    ]


def get_category_description() -> Dict[str, str]:
    """Return descriptions for each category."""
    return {
        "normal": "Standard English questions - coherent and grammatical",
        "slightly_odd": "Grammatically correct but semantically nonsensical",
        "broken_grammar": "Real words but scrambled word order",
        "word_salad": "Real words with no grammatical or semantic structure",
        "adversarial_like": "Mimics GCG-style adversarial suffixes",
        "token_soup": "Mix of words, numbers, and special characters",
        "pure_random": "Completely random characters and symbols",
    }


if __name__ == "__main__":
    # Test prompt generation
    dataset = create_prompt_dataset(samples_per_category=5)

    print("=" * 60)
    print("Prompt Dataset Sample")
    print("=" * 60)

    for category in get_category_order():
        print(f"\n### {category.upper()} ###")
        print(f"Description: {get_category_description()[category]}")
        print("-" * 40)
        for i, prompt in enumerate(dataset[category], 1):
            print(f"{i}. {prompt}")
