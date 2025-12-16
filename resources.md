# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project: **"Do LLMs Understand Nonsense Commands?"**

**Research Hypothesis**: Large language models may not be able to explain or interpret prompts that are optimized to produce outputs with high perplexity or outputs that do not resemble English, suggesting that directed prompt-based "jailbreaking" is fundamentally different from standard prompt understanding.

---

## Papers

**Total papers downloaded**: 8

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Talking Nonsense: Probing LLMs' Understanding of Adversarial Gibberish Inputs | Cherepanova, Zou | 2024 | `papers/2404.17120_talking_nonsense_llm_gibberish.pdf` | **Most relevant** - studies LM Babel |
| 2 | Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG) | Zou et al. | 2023 | `papers/2307.15043_universal_transferable_attacks.pdf` | Foundational GCG attack |
| 3 | AutoDAN: Generating Stealthy Jailbreak Prompts | Liu et al. | 2024 | `papers/2310.04451_autodan_stealthy_jailbreak.pdf` | Low-perplexity attacks |
| 4 | Jailbreaking Black Box LLMs in Twenty Queries (PAIR) | Chao et al. | 2023 | `papers/2310.08419_pair_jailbreaking_twenty_queries.pdf` | Semantic jailbreaks |
| 5 | HarmBench: A Standardized Evaluation Framework | Mazeika et al. | 2024 | `papers/2402.04249_harmbench.pdf` | Evaluation benchmark |
| 6 | COLD-Attack: Jailbreaking LLMs with Stealthiness | Guo et al. | 2024 | `papers/2402.08679_cold_attack.pdf` | Controllable attacks |
| 7 | JailbreakBench: An Open Robustness Benchmark | Various | 2024 | `papers/2404.01318_jailbreakbench.pdf` | Additional benchmark |
| 8 | AutoDAN: Interpretable Gradient-Based Attacks | Zhu et al. | 2024 | `papers/2310.15140_autodan_interpretable.pdf` | Interpretable attacks |

See `papers/README.md` for detailed descriptions.

---

## Datasets

**Total datasets downloaded**: 3

| # | Name | Source | Size | Task | Location | Notes |
|---|------|--------|------|------|----------|-------|
| 1 | AdvBench Harmful Behaviors | llm-attacks repo | 520 behaviors, 82KB | Jailbreak evaluation | `datasets/advbench_harmful_behaviors.csv` | Standard benchmark |
| 2 | AdvBench Harmful Strings | llm-attacks repo | 574 strings, 35KB | Target matching | `datasets/advbench_harmful_strings.csv` | Exact match targets |
| 3 | HarmBench Behaviors | HarmBench repo | 1528 behaviors, 199KB | Comprehensive evaluation | `datasets/harmbench_behaviors.csv` | Multi-category |

See `datasets/README.md` for detailed descriptions and download instructions.

---

## Code Repositories

**Total repositories cloned**: 4

| # | Name | URL | Purpose | Location | Notes |
|---|------|-----|---------|----------|-------|
| 1 | llm-attacks | github.com/llm-attacks/llm-attacks | GCG attack implementation | `code/llm-attacks/` | Primary for nonsense generation |
| 2 | AutoDAN | github.com/SheltonLiu-N/AutoDAN | Stealthy jailbreak attacks | `code/AutoDAN/` | Low-perplexity contrast |
| 3 | HarmBench | github.com/centerforaisafety/HarmBench | Evaluation framework | `code/HarmBench/` | Standardized evaluation |
| 4 | JailbreakingLLMs | github.com/patrickrchao/JailbreakingLLMs | PAIR attack | `code/JailbreakingLLMs/` | Semantic attacks |

See `code/README.md` for detailed descriptions and usage instructions.

---

## Resource Gathering Notes

### Search Strategy

1. **Literature Search**:
   - Searched arXiv for "LLM jailbreaking", "adversarial prompts", "perplexity attacks"
   - Searched Semantic Scholar and Papers with Code
   - Focused on 2023-2024 papers
   - Prioritized papers with code availability

2. **Dataset Search**:
   - Identified AdvBench as standard benchmark from GCG paper
   - Found HarmBench as comprehensive evaluation dataset
   - Downloaded from official GitHub repositories

3. **Code Search**:
   - Cloned official implementations from paper authors
   - Prioritized well-maintained repositories with documentation

### Selection Criteria

**Papers selected based on**:
- Direct relevance to nonsense prompt understanding
- Citation count and venue quality
- Code availability
- Recency (2023-2024)

**Datasets selected based on**:
- Standard use in literature
- Availability and accessibility
- Coverage of harmful behaviors
- Size and diversity

**Code repositories selected based on**:
- Official implementations from paper authors
- Documentation quality
- Active maintenance
- Compatibility with modern frameworks

### Challenges Encountered

1. **Most Relevant Paper**: The "Talking Nonsense" paper (2404.17120) is the only work directly studying LLM understanding of gibberish prompts. Most other work focuses on attack success, not interpretation.

2. **Dataset Availability**: JailbreakBench dataset structure was unclear; relied on AdvBench and HarmBench instead.

3. **No Explanation Datasets**: No existing dataset specifically tests whether LLMs can explain nonsense prompts - this is a gap for our research.

### Gaps and Workarounds

| Gap | Workaround |
|-----|------------|
| No dataset for prompt explanation | Generate with GCG, create custom evaluation |
| Limited understanding metrics | Define new metrics: explanation quality, coherence |
| Most work focuses on attack success | Use "Talking Nonsense" methodology as basis |

---

## Recommendations for Experiment Design

Based on gathered resources, recommend:

### 1. Primary Dataset(s)
- **AdvBench Harmful Behaviors**: Standard, well-established, used in GCG/AutoDAN
- **HarmBench**: For comprehensive category coverage

### 2. Baseline Methods
- **GCG** (llm-attacks): Generate high-perplexity nonsense suffixes
- **AutoDAN**: Generate low-perplexity readable attacks (contrast)
- **Random tokens**: True nonsense baseline
- **Natural prompts**: Control group

### 3. Evaluation Metrics

| Metric | Purpose | Implementation |
|--------|---------|----------------|
| Prompt Perplexity | Measure "nonsensicalness" | GPT-2/LLaMA perplexity |
| Attack Success Rate | Standard effectiveness | Exact match / classifier |
| Explanation Quality | LLM understanding | Human eval or LLM-as-judge |
| Explanation-Success Correlation | Core hypothesis | Statistical analysis |

### 4. Code to Adapt/Reuse
- `code/llm-attacks/` - For generating GCG adversarial suffixes
- `code/HarmBench/` - For standardized evaluation
- `code/AutoDAN/` - For generating readable attack prompts

### 5. Recommended Experiment Structure

```
Experiment 1: Generate Nonsense Prompts
├── Use GCG to generate adversarial suffixes
├── Vary perplexity levels
└── Collect successful and failed attacks

Experiment 2: Test LLM Explanation Ability
├── Present nonsense prompts to LLMs
├── Ask "What does this prompt mean?"
├── Measure explanation quality

Experiment 3: Correlation Analysis
├── Correlate perplexity with explanation quality
├── Correlate explanation quality with attack success
└── Test hypothesis: harder to explain = better attack?
```

---

## File Structure Summary

```
workspace/
├── papers/                          # Downloaded PDFs
│   ├── README.md
│   ├── 2404.17120_talking_nonsense_llm_gibberish.pdf
│   ├── 2307.15043_universal_transferable_attacks.pdf
│   ├── 2310.04451_autodan_stealthy_jailbreak.pdf
│   ├── 2310.08419_pair_jailbreaking_twenty_queries.pdf
│   ├── 2402.04249_harmbench.pdf
│   ├── 2402.08679_cold_attack.pdf
│   ├── 2404.01318_jailbreakbench.pdf
│   └── 2310.15140_autodan_interpretable.pdf
│
├── datasets/                        # Downloaded data
│   ├── README.md
│   ├── .gitignore
│   ├── advbench_harmful_behaviors.csv
│   ├── advbench_harmful_strings.csv
│   └── harmbench_behaviors.csv
│
├── code/                           # Cloned repositories
│   ├── README.md
│   ├── llm-attacks/               # GCG implementation
│   ├── AutoDAN/                   # Stealthy attacks
│   ├── HarmBench/                 # Evaluation framework
│   └── JailbreakingLLMs/          # PAIR implementation
│
├── literature_review.md           # Comprehensive review
├── resources.md                   # This file
└── .resource_finder_complete      # Completion marker
```

---

## Quick Start for Experiment Runner

1. **Read literature review**: `literature_review.md` for background
2. **Load datasets**: See `datasets/README.md` for loading code
3. **Generate prompts**: Use `code/llm-attacks/` GCG implementation
4. **Evaluate**: Use `code/HarmBench/` evaluation framework
5. **Custom experiments**: Test LLM explanation ability

---

## Version Information

- **Resource gathering date**: December 2024
- **Papers**: 2023-2024 publications
- **Repositories**: Latest main branches (shallow clone)
- **Datasets**: As of December 2024
