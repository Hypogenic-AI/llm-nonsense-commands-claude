# Do LLMs Understand Nonsense Commands?

An empirical investigation into whether LLMs can explain prompts across a perplexity spectrum—from natural English to GCG adversarial suffixes to random tokens—and what this reveals about the nature of adversarial attacks.

## Key Findings

- **U-shaped explanation quality**: Models explain natural language (4.5/5) and pure random (5.0/5) well, but struggle with *partially structured* nonsense like GCG suffixes (3.75/5). The hardest prompts to explain are not the most nonsensical, but those in the "uncanny valley" with enough structure to trigger confabulation.
- **Two explanation mechanisms**: Models use *understanding* for natural language and *meta-recognition* ("this is random") for pure nonsense. GCG suffixes fall in the gap between these mechanisms.
- **GCG triggers cross-lingual responses**: Adversarial suffixes containing tokens from Malay, Indonesian, and German caused GPT-4.1 to respond in those languages, suggesting gradient optimization discovers cross-lingual trigger tokens.
- **Directed nonsense is extremely hard to find**: Random search produced task-relevant outputs in only 5% of trials, confirming that adversarial prompt optimization (GCG) is essential.
- **Statistically significant** category differences (Kruskal-Wallis H=14.96, p=0.005)

## Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai transformers torch scipy pandas matplotlib seaborn numpy

# Run experiment (requires OPENAI_API_KEY)
python src/experiment.py

# Run analysis
python src/analyze.py
```

## File Structure

```
├── REPORT.md                    # Full research report with results
├── planning.md                  # Experimental design and hypotheses
├── literature_review.md         # Survey of 15 related papers
├── resources.md                 # Catalog of datasets, papers, code
├── src/
│   ├── experiment.py            # Main experiment (perplexity + API calls)
│   └── analyze.py               # Statistical analysis + visualization
├── results/
│   ├── experiment_results.csv   # All prompt results with scores
│   ├── experiment_results.json  # Same in JSON format
│   ├── directed_search_*.csv    # Directed nonsense search results
│   ├── analysis_results.json    # Statistical test outputs
│   └── summary_table.csv        # Category-level summary
├── figures/
│   ├── combined_results.png     # Three-panel main figure
│   ├── u_shape_explanation.png  # U-shaped pattern visualization
│   └── perplexity_*.png         # Additional plots
├── papers/                      # 15 downloaded research papers
├── datasets/                    # AdvBench, JailbreakBench, etc.
└── code/                        # Cloned repos (GCG, AutoDAN, etc.)
```

## Method

40 prompts across 5 perplexity categories tested on GPT-4.1:
1. Compute GPT-2 perplexity for each prompt
2. Send prompt to GPT-4.1, record response
3. Ask GPT-4.1 to explain the prompt's meaning
4. Score explanation quality (1-5) via LLM-as-judge

See [REPORT.md](REPORT.md) for full methodology and results.
