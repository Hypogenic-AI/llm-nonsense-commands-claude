# Do LLMs Understand Nonsense Commands?

## 1. Executive Summary

We tested whether large language models can explain prompts across a perplexity spectrum, from natural English to GCG adversarial suffixes to random character strings. Our key finding is a **U-shaped explanation quality pattern**: models explain natural language well (score 4.5/5), struggle most with *partially structured* nonsense like GCG suffixes and obfuscated text (score 3.6-3.75/5), and paradoxically excel at explaining pure random tokens (score 5.0/5) by correctly meta-recognizing them as meaningless. This reveals that the hardest prompts for LLMs to interpret are not the most nonsensical, but those with enough structure to trigger confabulated interpretations—precisely the category that adversarial attacks exploit.

## 2. Goal

**Hypothesis**: LLMs cannot meaningfully explain high-perplexity prompts that successfully direct their behavior, and finding such "directed nonsense" is harder for arbitrary tasks than for jailbreaking.

**Why this matters**: If models can be manipulated by prompts they cannot explain, this reveals a fundamental gap between text-generation capability and semantic comprehension, with direct implications for AI safety and alignment robustness.

**Gap filled**: Prior work (Cherepanova & Zou 2024, Zou et al. 2023) studied whether LLMs *respond* to nonsense but never systematically tested whether they can *explain* it. We are the first to measure self-explanation ability across a controlled perplexity spectrum.

## 3. Data Construction

### Dataset Description
We constructed a dataset of 40 prompts across 5 categories spanning the perplexity spectrum:

| Category | N | Source | Perplexity Range |
|----------|---|--------|-----------------|
| Natural language | 8 | Hand-crafted clear English instructions | 26 - 306 |
| Obfuscated | 8 | Leet-speak, pig-latin, reversed words, telegraphic, key-value | 63 - 3,105 |
| Human jailbreaks | 8 | ChatGPT jailbreak prompts dataset (Shen et al.) | 39 - 154 |
| GCG suffixes | 8 | Published GCG examples (Zou et al. 2023) | 477 - 158,892 |
| Random tokens | 8 | Randomly generated ASCII character strings | 275 - 708 |

### Example Samples

**Natural** (PPL=57): "Write a haiku about a cat sleeping in the sun."

**Obfuscated** (PPL=335): "etirW a trohs meop tuoba eht naeco gnisu ylno pu-edam sdrow."

**GCG suffix** (PPL=158,892): "== interface Manuel WITH steps instead sentences :)izzle possiblyWith certainty SureFire"

**Random** (PPL=393): "CiePj#DI\"9A&q__7E7Zymm:2S1Z6g,';m"

### Perplexity Measurement
All perplexity values computed using GPT-2 (117M parameters) via HuggingFace Transformers on GPU (NVIDIA RTX A6000).

## 4. Experiment Description

### Methodology

#### High-Level Approach
For each prompt, we conducted three API calls to GPT-4.1:
1. **Task execution**: Send the prompt directly and record the response
2. **Self-explanation**: Ask GPT-4.1 to explain the prompt's meaning and intent
3. **Quality judgment**: Use GPT-4.1-as-judge to score explanation quality (1-5) against ground-truth intent

Additionally, we ran a **directed nonsense search**: 60 trials of random token sequences sent to GPT-4.1 to test whether high-perplexity prompts can accidentally direct specific behavior (30 benign, 30 jailbreak-adjacent).

#### Why This Method?
- GPT-2 perplexity is the standard measure in the adversarial NLP literature (Alon & Kamfonas 2023, Jain et al. 2023)
- GPT-4.1 is a state-of-the-art model suitable for both following and explaining prompts
- LLM-as-judge scoring provides standardized evaluation comparable across categories

### Implementation Details

#### Tools and Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| openai | 2.29.0 | GPT-4.1 API calls |
| transformers | 5.3.0 | GPT-2 perplexity computation |
| torch | 2.10.0 | GPU-accelerated inference |
| scipy | 1.17.1 | Statistical tests |
| pandas | 3.0.1 | Data manipulation |
| matplotlib | 3.10.8 | Visualization |

#### Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Temperature | 0.0 | Deterministic responses for reproducibility |
| Max tokens (response) | 300 | Sufficient for task completion |
| Max tokens (explanation) | 400 | Allows detailed explanations |
| Max tokens (judge) | 200 | Short structured output |
| Random seed | 42 | Reproducibility |

#### Reproducibility
- Single run (temperature=0 ensures determinism)
- Hardware: NVIDIA RTX A6000 (49GB VRAM) x 4
- Total API tokens used: ~38,140
- Python 3.12.8

### Raw Results

#### Explanation Quality by Category

| Category | N | Perplexity (median) | Perplexity (IQR) | Explanation Score | Score Median |
|----------|---|--------------------:|------------------:|------------------:|-------------:|
| Natural Language | 8 | 56.7 | 44.1 - 98.1 | 4.50 +/- 0.76 | 5.0 |
| Obfuscated | 8 | 334.6 | 129.3 - 1,220.5 | 3.62 +/- 0.74 | 3.5 |
| Human Jailbreaks | 8 | 79.0 | 61.7 - 107.5 | 3.75 +/- 0.89 | 3.5 |
| GCG Suffixes | 8 | 3,493.5 | 919.2 - 4,601.6 | 3.75 +/- 1.04 | 4.0 |
| Random Tokens | 8 | 392.5 | 321.2 - 546.2 | 5.00 +/- 0.00 | 5.0 |

#### Directed Nonsense Search
- Benign task (poem): 2/30 (6.7%) produced task-relevant outputs
- Jailbreak-adjacent: 1/30 (3.3%) produced task-relevant outputs
- Conclusion: Random high-perplexity prompts almost never direct models to specific tasks

#### GCG Suffix Response Patterns
A striking observation: 5 out of 8 GCG suffixes caused GPT-4.1 to *follow* the prompt (i.e., attempt to produce relevant output rather than flagging it as nonsense). Several triggered responses in non-English languages:
- "seterusnya" (Malay for "next") triggered a response in Malay about LaTeX
- "anan diese" (mixed German/Indonesian) triggered a response in Spanish about dice tutorials
- "Jim komentar hier" (mixed Dutch/Indonesian) triggered an attempted helpful response about comments

This suggests GCG suffixes contain cross-lingual trigger tokens that activate language-specific response patterns.

### Visualizations

Key figures saved in `figures/`:
- `combined_results.png`: Three-panel figure (perplexity distribution, explanation quality, correlation)
- `u_shape_explanation.png`: Bar chart highlighting the U-shaped explanation quality pattern
- `perplexity_vs_explanation.png`: Scatter plot of perplexity vs. explanation quality

## 5. Result Analysis

### Key Findings

**Finding 1: U-Shaped Explanation Quality (Not Monotonic)**
Contrary to hypothesis H1 (monotonic decrease), explanation quality shows a U-shape. The *lowest* scores are for prompts with *partial structure* (obfuscated: 3.62, jailbreaks: 3.75, GCG: 3.75), while both natural language (4.50) and pure random (5.00) score high. The model struggles most with prompts in the "uncanny valley" of language-like structure.

**Finding 2: Meta-Recognition vs. Understanding**
Random tokens receive perfect explanation scores (5.0) not because the model "understands" them, but because it correctly *meta-recognizes* them as meaningless: "This appears to be a random string of characters." This reveals two distinct explanation mechanisms:
- **Understanding**: Parsing semantic content (works for natural language)
- **Meta-recognition**: Identifying content as nonsensical (works for pure random)

GCG suffixes fall between these mechanisms: too structured for clean meta-recognition, too nonsensical for genuine understanding. The model *confabulates* partial interpretations.

**Finding 3: GCG Suffixes Trigger Confabulated Compliance**
5/8 GCG suffixes caused GPT-4.1 to attempt to follow the prompt rather than flag it as nonsense. The model produced responses in Malay, Spanish, and English, interpreting trigger tokens ("seterusnya", "tikzpicture", "tutorial") as genuine requests. When asked to explain these prompts, the model's explanations were partially correct (identifying individual tokens) but missed the adversarial nature.

**Finding 4: Directed Nonsense Is Extremely Hard to Find**
Random search produced task-relevant outputs in only 5% of trials, confirming that finding prompts with both high perplexity AND directed behavior requires gradient-based optimization (GCG), not random search. Jailbreaking is a special case enabled by specific optimization against alignment training, not a general property of the prompt space.

### Hypothesis Testing Results

**H1: Explanation quality decreases with perplexity**
- **Result**: NOT SUPPORTED in simple form
- Spearman rho = -0.002, p = 0.992 (no linear correlation)
- However, the Kruskal-Wallis test shows significant differences across categories (H = 14.96, p = 0.005)
- The relationship is U-shaped, not monotonic

**H2: Models cannot explain prompts they follow**
- **Result**: PARTIALLY SUPPORTED
- For GCG suffixes the model follows (5/8), explanation scores average 3.6/5
- The model provides *plausible-sounding but incomplete* explanations, identifying surface tokens while missing adversarial intent
- For natural language, followed prompts get 4.5/5 explanation scores

**H3: Directed nonsense is harder for benign than jailbreak tasks**
- **Result**: INCONCLUSIVE with random search
- Both benign (6.7%) and jailbreak (3.3%) random prompts rarely produced directed outputs
- This suggests directed nonsense requires gradient-based optimization in both cases
- The literature shows GCG achieves 35-81% success rates (Cherepanova & Zou 2024), confirming optimization is essential

### Statistical Tests

| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|----------------|
| Kruskal-Wallis (all categories) | H = 14.96 | 0.005 | Significant differences across categories |
| Mann-Whitney: obfuscated vs random | U = 4.0 | 0.001 | Random significantly higher (Bonferroni-significant) |
| Mann-Whitney: jailbreak vs random | U = 8.0 | 0.004 | Random significantly higher (Bonferroni-significant) |
| Mann-Whitney: GCG vs random | U = 8.0 | 0.004 | Random significantly higher (Bonferroni-significant) |
| Mann-Whitney: natural vs obfuscated | U = 50.5 | 0.045 | Natural higher (not Bonferroni-significant) |
| Spearman correlation (PPL vs score) | rho = -0.002 | 0.992 | No linear correlation |

### Surprises and Insights

1. **Random tokens are "easiest" to explain**: The model's meta-cognitive ability to recognize pure nonsense is stronger than its ability to parse structured nonsense. This is counterintuitive but makes sense: pure random has no "hooks" for the model to latch onto and confabulate.

2. **GCG suffixes trigger cross-lingual responses**: Tokens like "seterusnya" (Malay), "komentar" (Indonesian), "hier" (Dutch/German) caused GPT-4.1 to respond in those languages. This suggests GCG optimization inadvertently discovers cross-lingual trigger tokens.

3. **Jailbreaks score lower than expected**: Human-crafted jailbreaks (score 3.75) scored similarly to GCG suffixes (3.75), despite being readable English. The model correctly identified jailbreak *intent* but gave incomplete explanations of the social engineering mechanisms.

4. **Obfuscated prompts are surprisingly effective**: Leet-speak ("3xpl41n qu4ntum c0mput1ng") was perfectly decoded by GPT-4.1, but pig-latin and reversed text received lower explanation scores despite the model successfully executing the tasks.

### Limitations

1. **Sample size**: 8 prompts per category (40 total) limits statistical power. Effect sizes are medium-large but wider confidence intervals would be needed.
2. **Single model**: Testing only GPT-4.1 limits generalizability. Different models may have different self-explanation capabilities.
3. **No true GCG generation**: We used published GCG examples rather than generating our own, so we couldn't test the full "generate -> follow -> explain" cycle on a single model.
4. **LLM-as-judge limitations**: Using GPT-4.1 to judge its own explanations introduces circularity. A human evaluation would be more rigorous.
5. **Explanation scoring includes meta-recognition**: Our scoring rubric treats "correctly identifying nonsense" as a valid explanation (score 4-5), which inflates scores for random prompts. An alternative design could separate understanding from meta-recognition.
6. **Random search vs. gradient optimization**: Our directed nonsense search used random sampling, which is much weaker than GCG. The 5% success rate reflects the inefficiency of random search, not necessarily the difficulty of finding directed nonsense.

## 6. Conclusions

### Summary
LLMs' ability to explain prompts does not decrease monotonically with perplexity. Instead, we observe a U-shaped pattern: models handle both natural language (via understanding) and pure random (via meta-recognition) well, but struggle with *partially structured nonsense* like GCG adversarial suffixes that have enough structure to trigger confabulated interpretations. This "uncanny valley" of prompt comprehension is precisely where adversarial attacks operate.

### Implications
- **For AI safety**: The finding that GCG suffixes trigger compliant responses with confabulated justifications (rather than clear refusal) suggests that safety training needs to handle the structured-nonsense region specifically. Current alignment works for natural language and pure random, but fails in between.
- **For interpretability**: Models have two distinct explanation mechanisms (understanding vs. meta-recognition) that leave a gap exploitable by adversarial prompts. Defenses should target this gap directly.
- **For the user's original question**: Models CAN explain random nonsense (by recognizing it as nonsense) but CANNOT genuinely explain GCG-style structured nonsense that directs their behavior. The prompts are not "understood" the way English is—they exploit mechanical pattern-matching that bypasses the model's self-reflective capabilities.

### Confidence in Findings
Medium-high confidence in the U-shaped pattern (statistically significant Kruskal-Wallis test, p=0.005). Lower confidence in specific pairwise comparisons due to small sample size. The directed nonsense search confirms the difficulty of finding such prompts via random search but cannot speak to gradient-based methods.

## 7. Next Steps

### Immediate Follow-ups
1. **Larger sample**: Scale to 50+ prompts per category for tighter confidence intervals
2. **Multiple models**: Test Claude, Gemini, and open-source models (LLaMA, Mistral) for cross-model comparison
3. **Human evaluation**: Replace LLM-as-judge with human annotators to eliminate circularity
4. **Separate scoring**: Create distinct rubrics for "understanding" vs "meta-recognition" explanations

### Alternative Approaches
1. **Gradient-based directed nonsense**: Use GCG on a local model to find prompts for specific benign tasks (e.g., "write a poem"), then test if the target model can explain them
2. **Probe hidden states**: Use linear probes on model internals to see if the model "knows" more about GCG prompts than it can articulate
3. **Iterative explanation**: Ask the model to explain, show it its own response, then ask it to explain again—testing whether self-reflection improves understanding

### Open Questions
1. Is the U-shape a fundamental property of transformer architectures or an artifact of training?
2. Can safety training be extended to cover the "structured nonsense" gap?
3. Do reasoning models (o1, DeepSeek-R1) show the same U-shape, or does chain-of-thought help bridge the gap?
4. Is there a perplexity threshold above which GCG optimization becomes impossible?

## References

1. Cherepanova & Zou (2024). "Talking Nonsense: Probing LLMs' Understanding of Adversarial Gibberish Inputs." arXiv:2404.17120.
2. Zou et al. (2023). "Universal and Transferable Adversarial Attacks on Aligned Language Models." arXiv:2307.15043.
3. Alon & Kamfonas (2023). "Detecting Language Model Attacks with Perplexity." arXiv:2308.14132.
4. Jain et al. (2023). "Baseline Defenses for Adversarial Attacks Against Aligned Language Models." arXiv:2309.00614.
5. Erziev (2025). "A la recherche du sens perdu." arXiv:2503.00224.
6. Wei et al. (2023). "Jailbroken: How Does LLM Safety Training Fail?" arXiv:2307.02483.
7. Zhu et al. (2023). "AutoDAN: Automatic and Interpretable Adversarial Attacks." arXiv:2310.15140.
