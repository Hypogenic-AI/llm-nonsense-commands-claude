"""
Main Experiment Runner

Runs the complete experiment to test whether LLMs can understand/explain
prompts across a spectrum from coherent to nonsensical.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt_generator import create_prompt_dataset, get_category_order, get_category_description
from perplexity_calculator import PerplexityCalculator, normalize_perplexity
from llm_client import OpenRouterClient, ExplanationAssessor, LLMResponse

# Configuration
SAMPLES_PER_CATEGORY = 10  # Number of prompts per category
MODEL_TO_TEST = "openai/gpt-4o-mini"  # Model for generating explanations
MODEL_FOR_JUDGE = "openai/gpt-4o-mini"  # Model for assessing explanations

# Set random seeds
np.random.seed(42)


def run_experiment(
    output_dir: str = "results",
    samples_per_category: int = SAMPLES_PER_CATEGORY,
) -> pd.DataFrame:
    """Run the complete experiment."""

    print("=" * 70)
    print("LLM Nonsense Understanding Experiment")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Model being tested: {MODEL_TO_TEST}")
    print(f"Samples per category: {samples_per_category}")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1: Generate prompts
    print("\n[Phase 1] Generating prompt dataset...")
    prompt_dataset = create_prompt_dataset(samples_per_category=samples_per_category)

    total_prompts = sum(len(prompts) for prompts in prompt_dataset.values())
    print(f"Generated {total_prompts} prompts across {len(prompt_dataset)} categories")

    # Phase 2: Calculate perplexities
    print("\n[Phase 2] Calculating perplexities...")
    ppl_calc = PerplexityCalculator()

    perplexity_data = {}
    for category in get_category_order():
        prompts = prompt_dataset[category]
        perplexities = ppl_calc.calculate_batch_perplexity(prompts)
        perplexity_data[category] = perplexities
        mean_ppl = np.mean([p for p in perplexities if p < 1e10])
        print(f"  {category}: mean perplexity = {mean_ppl:.2f}")

    # Phase 3: Get LLM explanations
    print(f"\n[Phase 3] Getting LLM explanations using {MODEL_TO_TEST}...")

    # Initialize clients
    explanation_client = OpenRouterClient(model=MODEL_TO_TEST)
    judge_client = OpenRouterClient(model=MODEL_FOR_JUDGE)
    assessor = ExplanationAssessor(judge_client)

    results = []
    categories = get_category_order()

    for category in categories:
        prompts = prompt_dataset[category]
        perplexities = perplexity_data[category]

        print(f"\n  Processing category: {category} ({len(prompts)} prompts)")

        for i, (prompt, ppl) in enumerate(tqdm(zip(prompts, perplexities), total=len(prompts), desc=f"  {category}")):
            # Get explanation from LLM
            explanation_query = f'What does this prompt mean? Explain in detail:\n\n"{prompt}"'

            explanation_response = explanation_client.query(
                prompt=explanation_query,
                max_tokens=300,
                temperature=0.1,
            )

            # Small delay to avoid rate limiting
            time.sleep(0.5)

            # Assess explanation quality
            if explanation_response.success and explanation_response.content:
                assessment = assessor.assess_explanation(prompt, explanation_response.content)
            else:
                assessment = {
                    "coherence": 0, "relevance": 0, "confidence": 0,
                    "insightfulness": 0, "accuracy": 0, "overall": 0,
                    "rationale": f"Explanation failed: {explanation_response.error}",
                    "success": False,
                }

            # Small delay
            time.sleep(0.3)

            # Test self-awareness (can LLM identify prompt type?)
            self_awareness = assessor.assess_self_awareness(prompt)
            time.sleep(0.3)

            # Compile result
            result = {
                "prompt": prompt,
                "category": category,
                "category_order": categories.index(category),
                "perplexity": ppl,
                "normalized_perplexity": normalize_perplexity(ppl),
                "explanation": explanation_response.content if explanation_response.success else "",
                "explanation_success": explanation_response.success,
                "explanation_length": len(explanation_response.content) if explanation_response.content else 0,
                "coherence": assessment.get("coherence", 0),
                "relevance": assessment.get("relevance", 0),
                "confidence": assessment.get("confidence", 0),
                "insightfulness": assessment.get("insightfulness", 0),
                "accuracy": assessment.get("accuracy", 0),
                "overall_quality": assessment.get("overall", 0),
                "assessment_rationale": assessment.get("rationale", ""),
                "self_awareness_classification": self_awareness.get("classification", "unknown"),
                "self_awareness_confidence": self_awareness.get("confidence", 0.0),
                "self_awareness_reasoning": self_awareness.get("reasoning", ""),
            }

            results.append(result)

    # Phase 4: Save results
    print("\n[Phase 4] Saving results...")

    df = pd.DataFrame(results)

    # Save to CSV
    csv_path = os.path.join(output_dir, "experiment_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved CSV to: {csv_path}")

    # Save to JSON for detailed data
    json_path = os.path.join(output_dir, "experiment_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved JSON to: {json_path}")

    # Save configuration
    config = {
        "model_tested": MODEL_TO_TEST,
        "model_judge": MODEL_FOR_JUDGE,
        "samples_per_category": samples_per_category,
        "categories": categories,
        "category_descriptions": get_category_description(),
        "total_prompts": total_prompts,
        "timestamp": datetime.now().isoformat(),
    }
    config_path = os.path.join(output_dir, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config to: {config_path}")

    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("=" * 70)

    # Print summary statistics
    print("\n### Summary Statistics ###\n")

    print("Mean Quality Scores by Category:")
    print("-" * 50)
    summary_cols = ["coherence", "relevance", "confidence", "insightfulness", "accuracy", "overall_quality"]
    summary = df.groupby("category")[summary_cols + ["perplexity", "normalized_perplexity"]].mean()

    # Reorder by category order
    summary = summary.reindex(categories)
    print(summary.round(2).to_string())

    print("\nCorrelation: Perplexity vs Overall Quality")
    print("-" * 50)
    from scipy import stats
    valid_idx = df["perplexity"] < 1e9
    correlation, p_value = stats.spearmanr(
        df.loc[valid_idx, "normalized_perplexity"],
        df.loc[valid_idx, "overall_quality"]
    )
    print(f"Spearman correlation: r = {correlation:.4f}")
    print(f"P-value: {p_value:.6f}")

    return df


if __name__ == "__main__":
    # Run experiment
    results_df = run_experiment()

    print("\n\nFirst few results:")
    print(results_df[["category", "prompt", "perplexity", "overall_quality"]].head(10))
