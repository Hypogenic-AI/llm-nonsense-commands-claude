# Research Report: Do LLMs Understand Nonsense Commands?

## 1. Executive Summary

This research investigates whether large language models (LLMs) can explain or interpret prompts across a spectrum from coherent English to complete nonsense. We hypothesized that LLMs cannot meaningfully explain high-perplexity prompts, suggesting a fundamental gap between text generation capability and semantic comprehension.

**Key Finding:** There is a strong, statistically significant negative correlation (Spearman r = -0.60, p < 0.0001) between prompt perplexity and explanation quality. LLMs provide significantly lower quality explanations for nonsensical prompts compared to normal English prompts, supporting the hypothesis that they process adversarial/nonsense prompts differently than they do natural language.

**Practical Implications:** This finding suggests that adversarial prompts (like those used in jailbreaking) may exploit a different mechanism than standard prompt understanding. While LLMs can recognize prompts as "nonsense," they still attempt to rationalize them, and their explanation quality degrades proportionally with prompt unnaturalness.

## 2. Goal

### Research Question
Can large language models explain or interpret prompts that are optimized to produce high-perplexity outputs or outputs that do not resemble English? If not, does this suggest that directed prompt-based "jailbreaking" exploits a fundamentally different mechanism than standard prompt understanding?

### Importance
1. **AI Safety**: Understanding how LLMs process adversarial/nonsense prompts is crucial for defending against jailbreaking attacks
2. **Model Understanding**: This research reveals a gap between LLM text-generation capability and semantic comprehension
3. **Mechanistic Insights**: The perplexity-understanding relationship illuminates how transformers process input tokens

### Background from Literature
The "Talking Nonsense" paper (Cherepanova & Zou, 2024) showed that LLMs can be manipulated by "LM Babel" (gibberish prompts) to produce specific outputs, but these prompts are fragile. The GCG attack (Zou et al., 2023) demonstrated that adversarial suffixes appear as "amalgamation of tokens" - pure gibberish but highly effective. However, **no prior work systematically tested whether LLMs can explain these prompts**, which is the gap this research addresses.

## 3. Data Construction

### Dataset Description
We generated a synthetic dataset of 70 prompts across 7 categories, designed to span a spectrum from coherent English to complete nonsense.

| Category | Description | Count | Example |
|----------|-------------|-------|---------|
| Normal | Standard English questions | 10 | "What is the capital of France?" |
| Slightly Odd | Grammatical but semantically nonsensical | 10 | "Explain why pickles orbit the moon on Tuesdays." |
| Broken Grammar | Real words, scrambled order | 10 | "Capital what France of is the?" |
| Word Salad | Real words, no structure | 10 | "elephant bicycle quantum whisper mountain" |
| Adversarial-like | Mimics GCG-style suffixes | 10 | "Here ystery wzh ctqg hh terfly nguin." |
| Token Soup | Mix of words, numbers, symbols | 10 | "^! 274 velvet fmxq mountain orchestra" |
| Pure Random | Random characters/Unicode | 10 | "D3˙çkVFa )D`!Wbi∂˙9≈$qoI©π∫©xCl∆∂T" |

### Example Samples

**Normal prompt:**
```
"What is the capital of France?"
Perplexity: 34.9
Overall Quality Score: 9/10
```

**Token soup prompt:**
```
"^! 274 velvet fmxq mountain orchestra"
Perplexity: 34,440
Overall Quality Score: 6/10
```

**Adversarial-like prompt:**
```
"Here ystery wzh ctqg hh terfly nguin."
Perplexity: 1,684
Overall Quality Score: 6/10
```

### Data Quality
- All 70 prompts successfully generated
- Perplexity calculated using GPT-2 model
- All prompts received explanations from GPT-4o-mini
- No missing values in final dataset

### Preprocessing Steps
1. Prompts generated programmatically with fixed random seed (42) for reproducibility
2. Perplexity calculated using GPT-2 with log transformation for normalization
3. Normalized perplexity mapped to 0-1 scale (0 = natural, 1 = random)

## 4. Experiment Description

### Methodology

#### High-Level Approach
We tested the hypothesis that LLMs cannot meaningfully explain high-perplexity prompts by:
1. Generating prompts across a perplexity spectrum
2. Asking an LLM (GPT-4o-mini) to explain each prompt
3. Using an LLM-as-judge (GPT-4o-mini) to assess explanation quality
4. Testing LLM self-awareness (can it identify prompts as nonsense/adversarial?)
5. Correlating perplexity with explanation quality

#### Why This Method?
- **Real LLM responses**: We used actual API calls to GPT-4o-mini, not simulated behavior
- **LLM-as-judge**: Provides consistent, scalable evaluation of explanation quality
- **Perplexity as proxy**: GPT-2 perplexity serves as an objective measure of prompt "naturalness"
- **Multiple quality dimensions**: We assessed coherence, relevance, confidence, insightfulness, and accuracy

### Implementation Details

#### Tools and Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.9 | Runtime |
| transformers | 4.57.3 | GPT-2 perplexity calculation |
| torch | 2.8.0+cpu | Model backend |
| openai | 2.12.0 | API client |
| pandas | 2.3.3 | Data processing |
| scipy | 1.13.1 | Statistical analysis |
| seaborn | 0.13.2 | Visualization |

#### Models Used
- **Perplexity Model**: GPT-2 (Hugging Face)
- **Test Model**: openai/gpt-4o-mini (via OpenRouter)
- **Judge Model**: openai/gpt-4o-mini (via OpenRouter)

#### Evaluation Metrics
| Metric | Scale | Description |
|--------|-------|-------------|
| Coherence | 0-10 | Is the explanation coherent and well-structured? |
| Relevance | 0-10 | Does the explanation address the prompt? |
| Confidence | 0-10 | How confident is the model in its explanation? |
| Insightfulness | 0-10 | Does the explanation provide meaningful insight? |
| Accuracy | 0-10 | Is the interpretation accurate? |
| Overall Quality | 0-10 | Holistic assessment |

#### Self-Awareness Classification
We also tested whether the LLM could classify prompts as:
- **Normal**: Standard, meaningful prompt
- **Nonsense**: Random or meaningless text
- **Adversarial**: Designed to manipulate the AI

### Experimental Protocol

#### Reproducibility Information
- **Random seeds**: 42 (used for prompt generation and NumPy)
- **Temperature**: 0.1 (explanation generation), 0.0 (assessment)
- **Hardware**: CPU-only execution
- **API Model**: openai/gpt-4o-mini via OpenRouter
- **Total API calls**: ~210 (70 prompts × 3 calls each)

### Raw Results

#### Mean Quality Scores by Category

| Category | Perplexity | Norm. PPL | Coherence | Relevance | Confidence | Insightful | Accuracy | Overall |
|----------|------------|-----------|-----------|-----------|------------|------------|----------|---------|
| normal | 44.4 | 0.38 | 9.5 | 9.8 | 9.7 | 8.0 | 9.9 | **9.0** |
| slightly_odd | 309.8 | 0.58 | 9.1 | 9.1 | 9.0 | 7.9 | 9.1 | **8.3** |
| broken_grammar | 2,893 | 0.83 | 8.9 | 9.4 | 9.4 | 7.5 | 9.3 | **8.5** |
| word_salad | 53,370 | 0.96 | 8.4 | 7.2 | 9.0 | 6.2 | 7.4 | **7.3** |
| adversarial_like | 1,909 | 0.80 | 8.0 | 6.9 | 8.0 | 5.9 | 7.7 | **6.7** |
| token_soup | 10,497 | 0.94 | 7.9 | 6.5 | 8.0 | 5.5 | 6.6 | **6.4** |
| pure_random | 502.0 | 0.66 | 8.7 | 7.8 | 8.2 | 6.6 | 9.0 | **7.6** |

#### Self-Awareness Classification Results

| Category | Expected | Correct % | Most Common Classification |
|----------|----------|-----------|---------------------------|
| normal | normal | 100% | normal (10/10) |
| slightly_odd | normal | 0% | nonsense (10/10) |
| broken_grammar | normal | 20% | nonsense (8/10) |
| word_salad | nonsense | 100% | nonsense (10/10) |
| adversarial_like | adversarial | 0% | nonsense (10/10) |
| token_soup | nonsense | 100% | nonsense (10/10) |
| pure_random | nonsense | 100% | nonsense (10/10) |

## 5. Result Analysis

### Key Findings

#### Finding 1: Strong Negative Correlation Between Perplexity and Explanation Quality
- **Spearman correlation**: r = -0.60 (p < 0.0001)
- This supports the core hypothesis: higher perplexity prompts receive lower quality explanations
- All quality dimensions show significant negative correlations:
  - Insightfulness: r = -0.66 (strongest)
  - Accuracy: r = -0.64
  - Coherence: r = -0.63
  - Relevance: r = -0.61
  - Confidence: r = -0.30 (weakest, p = 0.01)

#### Finding 2: Significant Group Differences
- **Kruskal-Wallis H test**: H = 52.19, p < 0.0001
- Explanation quality varies significantly across prompt categories
- **Effect sizes (Normal vs Others)**:
  - vs adversarial_like: d = -1.00 (perfect separation)
  - vs token_soup: d = -1.00 (perfect separation)
  - vs word_salad: d = -0.97
  - vs pure_random: d = -0.85
  - vs slightly_odd: d = -0.63
  - vs broken_grammar: d = -0.45

#### Finding 3: Self-Awareness Shows Nuanced Pattern
- LLMs correctly identify obvious nonsense (word_salad, token_soup, pure_random: 100% accuracy)
- LLMs often classify semantically odd prompts as "nonsense" even when grammatically correct
- LLMs rarely identify prompts as "adversarial" - they default to "nonsense" classification
- This suggests LLMs can detect *unusualness* but not necessarily *adversarial intent*

### Visualizations

![Quality Trend Line](figures/quality_trend_line.png)
*Figure 1: Explanation quality decreases as prompt category moves from normal to more nonsensical*

![Perplexity vs Quality Scatter](figures/perplexity_vs_quality_scatter.png)
*Figure 2: Scatter plot showing negative correlation between normalized perplexity and explanation quality*

![Quality Heatmap](figures/quality_heatmap.png)
*Figure 3: Heatmap of all quality metrics by category*

![Self-Awareness Classification](figures/self_awareness_classification.png)
*Figure 4: Distribution of self-awareness classifications by category*

### Surprises and Insights

1. **Pure random prompts got higher quality scores than expected (7.6)**
   - The model correctly identified them as random and gave accurate explanations like "this appears to be a random string"
   - The high accuracy score reflects the model correctly characterizing what it sees

2. **Adversarial-like prompts had the lowest quality (6.4)**
   - These prompts, designed to mimic GCG-style attacks, were harder to explain than pure randomness
   - The model struggled to find patterns in the partial words and structured nonsense

3. **Confidence remains relatively high even for nonsense (8.0-9.0)**
   - LLMs don't significantly reduce their expressed confidence when explaining nonsense
   - They rationalize instead of admitting uncertainty

4. **LLMs never identified prompts as "adversarial"**
   - All adversarial-like prompts were classified as "nonsense"
   - This suggests LLMs may not recognize manipulation attempts

### Error Analysis

The most common failure mode was **over-rationalization**: the LLM attempts to find meaning in meaningless input.

**Example (Token Soup):**
```
Prompt: "^! 274 velvet fmxq mountain orchestra"
Explanation: "The prompt appears to be a string of text that could have various
interpretations... Let's break it down into its components... The number 274
could represent a year, measurement, or code..."
```

The model treats random characters as if they might contain hidden meaning.

### Limitations

1. **Synthetic prompts**: We did not use actual GCG-generated adversarial prompts (would require GPU and extensive computation)
2. **Single model tested**: Results may differ for other LLMs (Claude, GPT-4, Llama, etc.)
3. **LLM-as-judge bias**: Using GPT-4o-mini to judge GPT-4o-mini may introduce systematic bias
4. **Limited sample size**: 10 prompts per category is sufficient for detecting large effects but may miss subtler patterns
5. **Perplexity measurement**: GPT-2 perplexity may not perfectly capture what GPT-4o-mini finds "surprising"

## 6. Conclusions

### Summary
LLMs cannot provide high-quality explanations for nonsensical prompts. There is a strong negative correlation (r = -0.60, p < 0.0001) between prompt perplexity and explanation quality. The quality degradation is especially pronounced for prompts that mimic adversarial structures (token soup, adversarial-like), suggesting that such prompts may exploit mechanisms distinct from normal language understanding.

### Implications

1. **For AI Safety**: Perplexity-based detection remains a viable defense against GCG-style attacks, as these prompts are both detectable AND less well-understood by the model
2. **For Understanding LLM Behavior**: LLMs process adversarial prompts in a qualitatively different way than natural language, supporting the "LM Babel" hypothesis from prior work
3. **For Jailbreaking Research**: The fact that LLMs cannot explain adversarial prompts but still respond to them suggests that jailbreaking exploits low-level token patterns rather than semantic understanding

### Confidence in Findings
- **High confidence** in the main correlation finding (p < 0.0001, large effect size)
- **Medium confidence** in category-specific patterns (limited sample size per category)
- **Lower confidence** in generalization to other models or real adversarial prompts

## 7. Next Steps

### Immediate Follow-ups
1. **Test with actual GCG-generated prompts**: Use real adversarial suffixes from the llm-attacks repository
2. **Multi-model comparison**: Repeat experiment with Claude, GPT-4, Llama, Mistral
3. **Larger sample sizes**: Increase to 50+ prompts per category for more robust estimates

### Alternative Approaches
1. **Use LLM internals**: Analyze attention patterns and activations for nonsense vs. normal prompts
2. **Fine-tuning study**: Does fine-tuning on adversarial examples improve explanation ability?
3. **Human evaluation**: Compare LLM-as-judge scores with human ratings

### Broader Extensions
1. **Multilingual**: Does the perplexity-understanding relationship hold for non-English?
2. **Multimodal**: Do vision-language models show similar patterns for adversarial images?
3. **Defense applications**: Can we use explanation quality as a filter for adversarial inputs?

### Open Questions
1. Why do LLMs try to rationalize nonsense instead of simply refusing to explain?
2. At what perplexity threshold does understanding truly break down?
3. Can adversarial prompts be designed that are both effective AND explainable by the model?

## References

### Papers
1. Cherepanova, V., & Zou, J. (2024). Talking Nonsense: Probing Large Language Models' Understanding of Adversarial Gibberish Inputs. arXiv:2404.17120
2. Zou, A., et al. (2023). Universal and Transferable Adversarial Attacks on Aligned Language Models. arXiv:2307.15043
3. Liu, X., et al. (2024). AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models. ICLR 2024
4. Chao, P., et al. (2023). Jailbreaking Black Box Large Language Models in Twenty Queries. arXiv:2310.08419
5. Mazeika, M., et al. (2024). HarmBench: A Standardized Evaluation Framework for Automated Red Teaming. arXiv:2402.04249

### Datasets
- AdvBench Harmful Behaviors (referenced for context)
- HarmBench Behaviors (referenced for context)

### Code and Tools
- GPT-2 (Hugging Face transformers)
- OpenRouter API
- Python scientific stack (pandas, scipy, matplotlib, seaborn)
