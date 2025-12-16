"""
Results Analysis Module

Performs statistical analysis and generates visualizations for the experiment results.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple


def load_results(results_dir: str = "results") -> pd.DataFrame:
    """Load experiment results from CSV."""
    csv_path = os.path.join(results_dir, "experiment_results.csv")
    return pd.read_csv(csv_path)


def load_config(results_dir: str = "results") -> Dict:
    """Load experiment configuration."""
    config_path = os.path.join(results_dir, "experiment_config.json")
    with open(config_path) as f:
        return json.load(f)


def perform_statistical_analysis(df: pd.DataFrame) -> Dict:
    """Perform comprehensive statistical analysis."""
    results = {}

    # Filter out infinite perplexities
    valid_df = df[df["perplexity"] < 1e9].copy()

    # 1. Correlation Analysis: Perplexity vs Quality Metrics
    print("\n### Correlation Analysis ###\n")

    quality_metrics = ["coherence", "relevance", "confidence", "insightfulness", "accuracy", "overall_quality"]

    correlations = {}
    for metric in quality_metrics:
        r, p = stats.spearmanr(valid_df["normalized_perplexity"], valid_df[metric])
        correlations[metric] = {"spearman_r": r, "p_value": p}
        print(f"Perplexity vs {metric}: r = {r:.4f}, p = {p:.6f}")

    results["correlations"] = correlations

    # 2. Group Comparisons
    print("\n### Group Comparisons ###\n")

    categories = df["category"].unique()
    category_order = ["normal", "slightly_odd", "broken_grammar", "word_salad",
                      "adversarial_like", "token_soup", "pure_random"]
    categories = [c for c in category_order if c in categories]

    # Kruskal-Wallis H test across all categories
    groups = [df[df["category"] == cat]["overall_quality"].values for cat in categories]
    h_stat, kw_p = stats.kruskal(*groups)
    print(f"Kruskal-Wallis H test: H = {h_stat:.4f}, p = {kw_p:.6f}")
    results["kruskal_wallis"] = {"h_statistic": h_stat, "p_value": kw_p}

    # Mann-Whitney U: Normal vs each other category
    print("\nMann-Whitney U: Normal vs Other Categories")
    print("-" * 50)
    normal_scores = df[df["category"] == "normal"]["overall_quality"].values

    mann_whitney_results = {}
    for cat in categories[1:]:  # Skip normal
        other_scores = df[df["category"] == cat]["overall_quality"].values
        u_stat, mw_p = stats.mannwhitneyu(normal_scores, other_scores, alternative="greater")

        # Effect size (rank-biserial correlation)
        n1, n2 = len(normal_scores), len(other_scores)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)

        mann_whitney_results[cat] = {
            "u_statistic": u_stat,
            "p_value": mw_p,
            "effect_size": effect_size,
        }
        print(f"  Normal vs {cat}: U = {u_stat:.1f}, p = {mw_p:.6f}, effect = {effect_size:.3f}")

    results["mann_whitney"] = mann_whitney_results

    # 3. Self-Awareness Analysis
    print("\n### Self-Awareness Analysis ###\n")

    self_awareness_accuracy = {}
    for cat in categories:
        cat_df = df[df["category"] == cat]
        classifications = cat_df["self_awareness_classification"].value_counts()
        total = len(cat_df)

        # Determine expected classification
        if cat == "normal":
            expected = "normal"
        elif cat in ["pure_random", "token_soup", "word_salad"]:
            expected = "nonsense"
        else:
            expected = "adversarial" if cat == "adversarial_like" else "normal"

        correct = classifications.get(expected, 0) / total if total > 0 else 0
        self_awareness_accuracy[cat] = {
            "expected": expected,
            "accuracy": correct,
            "distribution": classifications.to_dict(),
        }
        print(f"  {cat}: expected={expected}, accuracy={correct:.2%}")

    results["self_awareness"] = self_awareness_accuracy

    # 4. Descriptive Statistics by Category
    print("\n### Descriptive Statistics by Category ###\n")

    category_stats = {}
    for cat in categories:
        cat_df = df[df["category"] == cat]
        cat_stats = {
            "n": len(cat_df),
            "perplexity_mean": cat_df["perplexity"].mean(),
            "perplexity_std": cat_df["perplexity"].std(),
            "quality_mean": cat_df["overall_quality"].mean(),
            "quality_std": cat_df["overall_quality"].std(),
            "explanation_length_mean": cat_df["explanation_length"].mean(),
        }
        category_stats[cat] = cat_stats

    results["category_stats"] = category_stats

    # Print summary table
    summary_df = pd.DataFrame(category_stats).T
    print(summary_df.round(2).to_string())

    return results


def create_visualizations(df: pd.DataFrame, output_dir: str = "figures") -> List[str]:
    """Create and save visualizations."""

    os.makedirs(output_dir, exist_ok=True)
    saved_figures = []

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")

    category_order = ["normal", "slightly_odd", "broken_grammar", "word_salad",
                      "adversarial_like", "token_soup", "pure_random"]
    category_order = [c for c in category_order if c in df["category"].unique()]

    # 1. Box plot: Quality by Category
    fig, ax = plt.subplots(figsize=(12, 6))
    box_plot = sns.boxplot(
        data=df, x="category", y="overall_quality",
        order=category_order, ax=ax
    )
    ax.set_xlabel("Prompt Category", fontsize=12)
    ax.set_ylabel("Overall Explanation Quality (0-10)", fontsize=12)
    ax.set_title("Explanation Quality by Prompt Category", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = os.path.join(output_dir, "quality_by_category_boxplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    saved_figures.append(path)
    plt.close()

    # 2. Scatter plot: Perplexity vs Quality
    fig, ax = plt.subplots(figsize=(10, 6))
    valid_df = df[df["perplexity"] < 1e9].copy()

    scatter = ax.scatter(
        valid_df["normalized_perplexity"],
        valid_df["overall_quality"],
        c=valid_df["category_order"],
        cmap="viridis",
        alpha=0.6,
        s=50
    )

    # Add regression line
    z = np.polyfit(valid_df["normalized_perplexity"], valid_df["overall_quality"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, label=f"Trend line")

    # Add correlation annotation
    r, pval = stats.spearmanr(valid_df["normalized_perplexity"], valid_df["overall_quality"])
    ax.text(0.05, 0.95, f"Spearman r = {r:.3f}\np = {pval:.4f}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Normalized Perplexity (0=natural, 1=random)", fontsize=12)
    ax.set_ylabel("Overall Explanation Quality (0-10)", fontsize=12)
    ax.set_title("Perplexity vs Explanation Quality", fontsize=14)
    plt.colorbar(scatter, label="Category (ordered by perplexity)")
    plt.tight_layout()

    path = os.path.join(output_dir, "perplexity_vs_quality_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    saved_figures.append(path)
    plt.close()

    # 3. Heatmap: Quality Metrics by Category
    quality_metrics = ["coherence", "relevance", "confidence", "insightfulness", "accuracy", "overall_quality"]
    mean_scores = df.groupby("category")[quality_metrics].mean()
    mean_scores = mean_scores.reindex(category_order)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        mean_scores, annot=True, fmt=".1f", cmap="RdYlGn",
        vmin=0, vmax=10, ax=ax, cbar_kws={"label": "Score (0-10)"}
    )
    ax.set_xlabel("Quality Metric", fontsize=12)
    ax.set_ylabel("Prompt Category", fontsize=12)
    ax.set_title("Mean Quality Scores by Category", fontsize=14)
    plt.tight_layout()

    path = os.path.join(output_dir, "quality_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    saved_figures.append(path)
    plt.close()

    # 4. Bar plot: Self-Awareness Classification
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    for i, cat in enumerate(category_order):
        if i >= len(axes):
            break
        cat_df = df[df["category"] == cat]
        class_counts = cat_df["self_awareness_classification"].value_counts()

        ax = axes[i]
        colors = {"normal": "green", "nonsense": "orange", "adversarial": "red", "unknown": "gray"}
        bar_colors = [colors.get(c, "blue") for c in class_counts.index]

        ax.bar(class_counts.index, class_counts.values, color=bar_colors)
        ax.set_title(cat, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

    # Hide empty subplots
    for i in range(len(category_order), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Self-Awareness Classification by Category", fontsize=14, y=1.02)
    plt.tight_layout()

    path = os.path.join(output_dir, "self_awareness_classification.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    saved_figures.append(path)
    plt.close()

    # 5. Line plot: Mean Quality across Category Spectrum
    fig, ax = plt.subplots(figsize=(10, 5))

    mean_quality = df.groupby("category")["overall_quality"].agg(["mean", "std"])
    mean_quality = mean_quality.reindex(category_order)

    x = range(len(category_order))
    ax.errorbar(x, mean_quality["mean"], yerr=mean_quality["std"],
                marker="o", capsize=5, capthick=2, linewidth=2, markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(category_order, rotation=45, ha="right")
    ax.set_xlabel("Prompt Category (Low → High Perplexity)", fontsize=12)
    ax.set_ylabel("Mean Explanation Quality (± std)", fontsize=12)
    ax.set_title("Explanation Quality Decreases with Prompt Nonsensicality", fontsize=14)
    ax.set_ylim(0, 10)
    plt.tight_layout()

    path = os.path.join(output_dir, "quality_trend_line.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    saved_figures.append(path)
    plt.close()

    # 6. Violin plot: Perplexity distribution by category
    fig, ax = plt.subplots(figsize=(12, 6))
    valid_df = df[df["perplexity"] < 1e9].copy()

    # Use log scale for perplexity
    valid_df["log_perplexity"] = np.log10(valid_df["perplexity"] + 1)

    sns.violinplot(
        data=valid_df, x="category", y="log_perplexity",
        order=category_order, ax=ax
    )
    ax.set_xlabel("Prompt Category", fontsize=12)
    ax.set_ylabel("Log10(Perplexity)", fontsize=12)
    ax.set_title("Perplexity Distribution by Category", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = os.path.join(output_dir, "perplexity_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    saved_figures.append(path)
    plt.close()

    print(f"\nSaved {len(saved_figures)} figures to {output_dir}/")
    return saved_figures


def generate_analysis_report(df: pd.DataFrame, stats_results: Dict, output_dir: str = "results") -> str:
    """Generate a text summary of the analysis."""

    report = []
    report.append("=" * 70)
    report.append("ANALYSIS REPORT: LLM Understanding of Nonsense Commands")
    report.append("=" * 70)

    # Key Findings
    report.append("\n## KEY FINDINGS\n")

    # Correlation finding
    corr = stats_results["correlations"]["overall_quality"]
    report.append(f"1. PERPLEXITY-QUALITY CORRELATION")
    report.append(f"   Spearman r = {corr['spearman_r']:.4f}")
    report.append(f"   P-value = {corr['p_value']:.6f}")
    if corr["spearman_r"] < -0.3 and corr["p_value"] < 0.05:
        report.append("   → Significant NEGATIVE correlation: higher perplexity = lower explanation quality")
    elif corr["p_value"] >= 0.05:
        report.append("   → No significant correlation detected")

    # Group difference finding
    kw = stats_results["kruskal_wallis"]
    report.append(f"\n2. GROUP DIFFERENCES (Kruskal-Wallis)")
    report.append(f"   H-statistic = {kw['h_statistic']:.4f}")
    report.append(f"   P-value = {kw['p_value']:.6f}")
    if kw["p_value"] < 0.05:
        report.append("   → Significant differences in quality across prompt categories")

    # Normal vs nonsense comparison
    if "pure_random" in stats_results["mann_whitney"]:
        mw = stats_results["mann_whitney"]["pure_random"]
        report.append(f"\n3. NORMAL vs RANDOM PROMPTS (Mann-Whitney U)")
        report.append(f"   Effect size = {mw['effect_size']:.3f}")
        report.append(f"   P-value = {mw['p_value']:.6f}")

    # Self-awareness
    report.append("\n4. SELF-AWARENESS")
    sa = stats_results["self_awareness"]
    normal_acc = sa.get("normal", {}).get("accuracy", 0)
    random_acc = sa.get("pure_random", {}).get("accuracy", 0)
    report.append(f"   Normal prompts correctly identified: {normal_acc:.1%}")
    report.append(f"   Random prompts correctly identified: {random_acc:.1%}")

    # Category Statistics
    report.append("\n## CATEGORY STATISTICS\n")
    cat_stats = stats_results["category_stats"]
    for cat, s in cat_stats.items():
        report.append(f"{cat}:")
        report.append(f"  Mean perplexity: {s['perplexity_mean']:.1f}")
        report.append(f"  Mean quality: {s['quality_mean']:.2f} (±{s['quality_std']:.2f})")

    report_text = "\n".join(report)

    # Save report
    report_path = os.path.join(output_dir, "analysis_summary.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    print(report_text)
    return report_text


def run_full_analysis(results_dir: str = "results", figures_dir: str = "figures"):
    """Run complete analysis pipeline."""

    print("\n" + "=" * 70)
    print("Running Full Analysis")
    print("=" * 70)

    # Load data
    df = load_results(results_dir)
    config = load_config(results_dir)

    print(f"\nLoaded {len(df)} results from experiment")
    print(f"Model tested: {config.get('model_tested', 'unknown')}")

    # Perform statistical analysis
    stats_results = perform_statistical_analysis(df)

    # Save statistics
    stats_path = os.path.join(results_dir, "statistical_analysis.json")
    with open(stats_path, "w") as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(stats_results, f, indent=2, default=convert)

    # Create visualizations
    saved_figures = create_visualizations(df, figures_dir)

    # Generate text report
    generate_analysis_report(df, stats_results, results_dir)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)

    return df, stats_results, saved_figures


if __name__ == "__main__":
    run_full_analysis()
