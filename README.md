# Do LLMs Understand Nonsense Commands?

An empirical investigation into whether large language models can explain prompts across a spectrum from coherent English to complete nonsense.

## Key Findings

- **Strong negative correlation (r = -0.60, p < 0.0001)** between prompt perplexity and explanation quality
- LLMs provide significantly lower quality explanations for nonsensical prompts compared to normal English
- Effect sizes are large: adversarial-like and token soup prompts show near-perfect separation (d = -1.0) from normal prompts
- LLMs recognize prompts as "nonsense" but still attempt to rationalize them
- LLMs rarely identify prompts as "adversarial" - they default to classifying unusual input as "nonsense"

These findings support the hypothesis that adversarial prompts exploit mechanisms distinct from standard language understanding.

## Quick Results Summary

| Category | Mean Perplexity | Explanation Quality (0-10) |
|----------|-----------------|---------------------------|
| Normal | 44 | 9.0 |
| Slightly Odd | 310 | 8.3 |
| Broken Grammar | 2,893 | 8.5 |
| Word Salad | 53,370 | 7.3 |
| Adversarial-like | 1,909 | 6.7 |
| Token Soup | 10,497 | 6.4 |
| Pure Random | 502 | 7.6 |

## Repository Structure

```
llm-nonsense-commands-claude/
├── REPORT.md                    # Full research report with methodology and findings
├── README.md                    # This file
├── planning.md                  # Research plan and experimental design
├── pyproject.toml               # Project dependencies
├── src/
│   ├── prompt_generator.py      # Generates prompts across perplexity spectrum
│   ├── perplexity_calculator.py # Calculates perplexity using GPT-2
│   ├── llm_client.py            # OpenRouter API client for LLM queries
│   ├── run_experiment.py        # Main experiment runner
│   └── analyze_results.py       # Statistical analysis and visualization
├── results/
│   ├── experiment_results.csv   # Raw experimental data
│   ├── experiment_results.json  # Detailed results with explanations
│   ├── experiment_config.json   # Experiment configuration
│   ├── statistical_analysis.json# Statistical test results
│   └── analysis_summary.txt     # Text summary of analysis
├── figures/
│   ├── quality_trend_line.png   # Quality vs category trend
│   ├── perplexity_vs_quality_scatter.png
│   ├── quality_heatmap.png
│   ├── quality_by_category_boxplot.png
│   ├── perplexity_distribution.png
│   └── self_awareness_classification.png
├── literature_review.md         # Background literature review
├── resources.md                 # Available resources catalog
├── datasets/                    # Reference datasets (AdvBench, HarmBench)
└── papers/                      # Reference papers
```

## How to Reproduce

### 1. Set up environment

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install pandas numpy matplotlib scipy openai requests tqdm seaborn transformers tiktoken torch
```

### 2. Set API key

```bash
export OPENROUTER_API_KEY="your-key-here"
```

### 3. Run experiment

```bash
python src/run_experiment.py
```

This will:
- Generate 70 prompts across 7 categories
- Calculate perplexity for each prompt using GPT-2
- Query GPT-4o-mini for explanations
- Assess explanation quality using LLM-as-judge
- Save results to `results/` directory

### 4. Run analysis

```bash
python src/analyze_results.py
```

This will:
- Perform statistical analysis (correlations, group comparisons)
- Generate visualizations in `figures/`
- Create summary report

## Methodology

1. **Prompt Generation**: Created 7 categories of prompts spanning from normal English to pure random characters
2. **Perplexity Calculation**: Used GPT-2 to measure how "surprising" each prompt is
3. **Explanation Collection**: Asked GPT-4o-mini "What does this prompt mean? Explain in detail."
4. **Quality Assessment**: Used GPT-4o-mini as judge to rate explanations on coherence, relevance, confidence, insightfulness, and accuracy
5. **Self-Awareness Testing**: Asked the LLM to classify prompts as normal/nonsense/adversarial
6. **Statistical Analysis**: Computed correlations, effect sizes, and significance tests

## Key Statistical Results

- **Spearman correlation (perplexity vs quality)**: r = -0.60, p < 0.0001
- **Kruskal-Wallis test**: H = 52.19, p < 0.0001
- **Effect size (Normal vs Token Soup)**: d = -1.00

## Citation

If you use this research, please cite:

```
@misc{llm-nonsense-2024,
  title={Do LLMs Understand Nonsense Commands?},
  year={2024},
  note={Investigating LLM explanation quality across prompt perplexity spectrum}
}
```

## Related Work

- Cherepanova & Zou (2024). "Talking Nonsense: Probing LLMs' Understanding of Adversarial Gibberish Inputs"
- Zou et al. (2023). "Universal and Transferable Adversarial Attacks on Aligned Language Models" (GCG)
- Liu et al. (2024). "AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned LLMs"
- Chao et al. (2023). "Jailbreaking Black Box LLMs in Twenty Queries" (PAIR)
