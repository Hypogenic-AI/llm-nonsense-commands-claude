"""
Analysis script for the nonsense commands experiment.
Produces statistical tests, visualizations, and summary tables.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

RESULTS_DIR = Path("/workspaces/llm-nonsense-commands-claude/results")
FIGURES_DIR = Path("/workspaces/llm-nonsense-commands-claude/figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Ordered categories from low to high perplexity
CATEGORY_ORDER = ["natural", "obfuscated", "jailbreak", "gcg_suffix", "random"]
CATEGORY_LABELS = {
    "natural": "Natural\nLanguage",
    "obfuscated": "Obfuscated\n(Leet/PigLatin)",
    "jailbreak": "Human\nJailbreaks",
    "gcg_suffix": "GCG\nSuffixes",
    "random": "Random\nTokens",
}


def load_results():
    """Load experiment results."""
    df = pd.read_csv(RESULTS_DIR / "experiment_results.csv")
    return df


def analyze_perplexity_by_category(df):
    """Analyze perplexity distributions across categories."""
    print("\n" + "=" * 60)
    print("PERPLEXITY BY CATEGORY")
    print("=" * 60)

    summary = df.groupby("category")["perplexity"].agg(
        ["mean", "std", "median", "min", "max", "count"]
    ).reindex(CATEGORY_ORDER)
    print(summary.to_string())

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    cat_data = [df[df["category"] == c]["perplexity"].values for c in CATEGORY_ORDER]
    # Use log scale for better visualization
    cat_data_log = [np.log10(np.clip(d, 1, None)) for d in cat_data]

    bp = ax.boxplot(cat_data_log, labels=[CATEGORY_LABELS[c] for c in CATEGORY_ORDER],
                     patch_artist=True)
    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6", "#95a5a6"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("log₁₀(Perplexity)", fontsize=12)
    ax.set_title("Prompt Perplexity by Category", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "perplexity_by_category.png", dpi=150)
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'perplexity_by_category.png'}")
    return summary


def analyze_explanation_quality(df):
    """Analyze explanation quality scores across categories."""
    print("\n" + "=" * 60)
    print("EXPLANATION QUALITY BY CATEGORY")
    print("=" * 60)

    # Filter out failed judgments
    valid = df[df["explanation_score"] > 0].copy()

    summary = valid.groupby("category")["explanation_score"].agg(
        ["mean", "std", "median", "count"]
    ).reindex(CATEGORY_ORDER)
    print(summary.to_string())

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    cat_data = [valid[valid["category"] == c]["explanation_score"].values
                for c in CATEGORY_ORDER]

    bp = ax.boxplot(cat_data, labels=[CATEGORY_LABELS[c] for c in CATEGORY_ORDER],
                     patch_artist=True)
    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6", "#95a5a6"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Explanation Quality Score (1-5)", fontsize=12)
    ax.set_title("LLM Explanation Quality by Prompt Category", fontsize=14)
    ax.set_ylim(0.5, 5.5)
    ax.grid(axis="y", alpha=0.3)

    # Add mean markers
    means = [np.mean(d) if len(d) > 0 else 0 for d in cat_data]
    ax.scatter(range(1, len(means)+1), means, color="red", zorder=5,
               marker="D", s=50, label="Mean")
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "explanation_quality_by_category.png", dpi=150)
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'explanation_quality_by_category.png'}")
    return summary


def analyze_perplexity_explanation_correlation(df):
    """Compute correlation between perplexity and explanation quality."""
    print("\n" + "=" * 60)
    print("PERPLEXITY-EXPLANATION CORRELATION")
    print("=" * 60)

    valid = df[(df["explanation_score"] > 0) & (df["perplexity"] < float("inf"))].copy()
    valid["log_perplexity"] = np.log10(np.clip(valid["perplexity"], 1, None))

    # Spearman correlation
    rho, p_value = stats.spearmanr(valid["log_perplexity"], valid["explanation_score"])
    print(f"Spearman ρ = {rho:.4f}, p = {p_value:.6f}")

    # Pearson correlation on log perplexity
    r, p_pearson = stats.pearsonr(valid["log_perplexity"], valid["explanation_score"])
    print(f"Pearson r = {r:.4f}, p = {p_pearson:.6f}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 7))
    color_map = {"natural": "#2ecc71", "obfuscated": "#f39c12", "jailbreak": "#e74c3c",
                 "gcg_suffix": "#9b59b6", "random": "#95a5a6"}

    for cat in CATEGORY_ORDER:
        subset = valid[valid["category"] == cat]
        ax.scatter(subset["log_perplexity"], subset["explanation_score"],
                   c=color_map[cat], label=CATEGORY_LABELS[cat].replace("\n", " "),
                   s=80, alpha=0.7, edgecolors="black", linewidth=0.5)

    # Trend line
    z = np.polyfit(valid["log_perplexity"], valid["explanation_score"], 1)
    x_line = np.linspace(valid["log_perplexity"].min(), valid["log_perplexity"].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "k--", alpha=0.5,
            label=f"Trend (ρ={rho:.2f}, p={p_value:.4f})")

    ax.set_xlabel("log₁₀(GPT-2 Perplexity)", fontsize=12)
    ax.set_ylabel("Explanation Quality Score (1-5)", fontsize=12)
    ax.set_title("Prompt Perplexity vs. LLM Explanation Quality", fontsize=14)
    ax.legend(fontsize=9)
    ax.set_ylim(0.5, 5.5)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "perplexity_vs_explanation.png", dpi=150)
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'perplexity_vs_explanation.png'}")

    return {"spearman_rho": rho, "spearman_p": p_value,
            "pearson_r": r, "pearson_p": p_pearson}


def run_statistical_tests(df):
    """Run hypothesis tests across categories."""
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)

    valid = df[df["explanation_score"] > 0].copy()
    results = {}

    # Kruskal-Wallis test across all categories
    groups = [valid[valid["category"] == c]["explanation_score"].values
              for c in CATEGORY_ORDER]
    groups = [g for g in groups if len(g) > 0]
    h_stat, p_kw = stats.kruskal(*groups)
    print(f"\nKruskal-Wallis H = {h_stat:.4f}, p = {p_kw:.6f}")
    results["kruskal_wallis"] = {"H": h_stat, "p": p_kw}

    # Pairwise Mann-Whitney U tests
    n_comparisons = len(CATEGORY_ORDER) * (len(CATEGORY_ORDER) - 1) // 2
    alpha_bonf = 0.05 / n_comparisons
    print(f"\nPairwise Mann-Whitney U tests (Bonferroni α = {alpha_bonf:.4f}):")
    pairwise = []
    for i, cat1 in enumerate(CATEGORY_ORDER):
        for cat2 in CATEGORY_ORDER[i+1:]:
            g1 = valid[valid["category"] == cat1]["explanation_score"].values
            g2 = valid[valid["category"] == cat2]["explanation_score"].values
            if len(g1) > 0 and len(g2) > 0:
                u_stat, p_mw = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                # Effect size (rank-biserial correlation)
                effect = 1 - (2 * u_stat) / (len(g1) * len(g2))
                sig = "***" if p_mw < alpha_bonf else ("*" if p_mw < 0.05 else "ns")
                print(f"  {cat1} vs {cat2}: U={u_stat:.1f}, p={p_mw:.4f}, "
                      f"r={effect:.3f} {sig}")
                pairwise.append({
                    "cat1": cat1, "cat2": cat2,
                    "U": u_stat, "p": p_mw, "effect_r": effect,
                    "significant_bonf": p_mw < alpha_bonf
                })

    results["pairwise"] = pairwise
    return results


def create_summary_table(df):
    """Create a summary table for the report."""
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)

    valid = df[df["explanation_score"] > 0].copy()

    summary = []
    for cat in CATEGORY_ORDER:
        subset = valid[valid["category"] == cat]
        ppl_vals = subset["perplexity"].values
        score_vals = subset["explanation_score"].values
        summary.append({
            "Category": CATEGORY_LABELS[cat].replace("\n", " "),
            "N": len(subset),
            "Perplexity (median)": f"{np.median(ppl_vals):.1f}",
            "Perplexity (IQR)": f"{np.percentile(ppl_vals, 25):.1f}-{np.percentile(ppl_vals, 75):.1f}",
            "Explanation Score": f"{np.mean(score_vals):.2f} ± {np.std(score_vals):.2f}",
            "Score Median": f"{np.median(score_vals):.1f}",
        })

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
    return summary_df


def create_combined_figure(df):
    """Create a combined figure with key results."""
    valid = df[(df["explanation_score"] > 0) & (df["perplexity"] < float("inf"))].copy()
    valid["log_perplexity"] = np.log10(np.clip(valid["perplexity"], 1, None))

    color_map = {"natural": "#2ecc71", "obfuscated": "#f39c12", "jailbreak": "#e74c3c",
                 "gcg_suffix": "#9b59b6", "random": "#95a5a6"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: Perplexity distribution
    cat_data_log = [np.log10(np.clip(
        valid[valid["category"] == c]["perplexity"].values, 1, None))
        for c in CATEGORY_ORDER]
    bp = axes[0].boxplot(cat_data_log,
                          labels=[CATEGORY_LABELS[c] for c in CATEGORY_ORDER],
                          patch_artist=True)
    for patch, color in zip(bp["boxes"], list(color_map.values())):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_ylabel("log₁₀(Perplexity)")
    axes[0].set_title("A) Perplexity by Category")
    axes[0].grid(axis="y", alpha=0.3)

    # Panel B: Explanation quality
    cat_scores = [valid[valid["category"] == c]["explanation_score"].values
                  for c in CATEGORY_ORDER]
    bp2 = axes[1].boxplot(cat_scores,
                           labels=[CATEGORY_LABELS[c] for c in CATEGORY_ORDER],
                           patch_artist=True)
    for patch, color in zip(bp2["boxes"], list(color_map.values())):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    means = [np.mean(d) if len(d) > 0 else 0 for d in cat_scores]
    axes[1].scatter(range(1, len(means)+1), means, color="red", zorder=5,
                     marker="D", s=50, label="Mean")
    axes[1].set_ylabel("Explanation Quality (1-5)")
    axes[1].set_title("B) Explanation Quality by Category")
    axes[1].set_ylim(0.5, 5.5)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    # Panel C: Scatter with correlation
    for cat in CATEGORY_ORDER:
        subset = valid[valid["category"] == cat]
        axes[2].scatter(subset["log_perplexity"], subset["explanation_score"],
                         c=color_map[cat],
                         label=CATEGORY_LABELS[cat].replace("\n", " "),
                         s=60, alpha=0.7, edgecolors="black", linewidth=0.5)
    rho, p_val = stats.spearmanr(valid["log_perplexity"], valid["explanation_score"])
    z = np.polyfit(valid["log_perplexity"], valid["explanation_score"], 1)
    x_line = np.linspace(valid["log_perplexity"].min(), valid["log_perplexity"].max(), 100)
    axes[2].plot(x_line, np.polyval(z, x_line), "k--", alpha=0.5)
    axes[2].set_xlabel("log₁₀(Perplexity)")
    axes[2].set_ylabel("Explanation Quality (1-5)")
    axes[2].set_title(f"C) Correlation (ρ={rho:.2f}, p={p_val:.4f})")
    axes[2].set_ylim(0.5, 5.5)
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=7)

    plt.suptitle("Do LLMs Understand Nonsense Commands?", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "combined_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'combined_results.png'}")


def analyze_example_explanations(df):
    """Show representative examples from each category."""
    print("\n" + "=" * 60)
    print("REPRESENTATIVE EXAMPLES")
    print("=" * 60)

    valid = df[df["explanation_score"] > 0].copy()
    examples = []

    for cat in CATEGORY_ORDER:
        subset = valid[valid["category"] == cat].sort_values("explanation_score")
        if len(subset) > 0:
            # Pick median-scored example
            mid_idx = len(subset) // 2
            row = subset.iloc[mid_idx]
            examples.append({
                "category": cat,
                "prompt": row["text"][:150],
                "explanation": row["explanation"][:200],
                "score": row["explanation_score"],
                "perplexity": row["perplexity"],
            })
            print(f"\n--- {cat} (score={row['explanation_score']}, ppl={row['perplexity']:.1f}) ---")
            print(f"  Prompt: {row['text'][:120]}")
            print(f"  Explanation: {row['explanation'][:200]}")

    pd.DataFrame(examples).to_csv(RESULTS_DIR / "example_explanations.csv", index=False)


def run_analysis():
    """Run the full analysis pipeline."""
    print("=" * 70)
    print("ANALYSIS: Do LLMs Understand Nonsense Commands?")
    print("=" * 70)

    df = load_results()
    print(f"Loaded {len(df)} results")

    ppl_summary = analyze_perplexity_by_category(df)
    score_summary = analyze_explanation_quality(df)
    correlation = analyze_perplexity_explanation_correlation(df)
    stat_tests = run_statistical_tests(df)
    summary_table = create_summary_table(df)
    create_combined_figure(df)
    analyze_example_explanations(df)

    # Save all analysis results
    analysis_results = {
        "correlation": correlation,
        "kruskal_wallis": stat_tests["kruskal_wallis"],
        "pairwise_tests": stat_tests["pairwise"],
    }
    with open(RESULTS_DIR / "analysis_results.json", "w") as f:
        json.dump(analysis_results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

    return analysis_results


if __name__ == "__main__":
    run_analysis()
