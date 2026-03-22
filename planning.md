# Research Plan: Do LLMs Understand Nonsense Commands?

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs can be manipulated by adversarial prompts that appear as gibberish to humans (GCG suffixes, Babel prompts). Understanding whether models can *explain* these prompts—not just respond to them—reveals whether adversarial attacks exploit genuine linguistic understanding or merely trigger mechanical pattern responses. This has direct implications for AI safety: if models can't explain what nonsense prompts "mean," it suggests alignment mechanisms are superficial and can be bypassed by inputs the model processes but doesn't "understand."

### Gap in Existing Work
The literature extensively studies whether LLMs *respond* to nonsense (Cherepanova & Zou 2024, Zou et al. 2023) and whether nonsense can be *detected* (Alon & Kamfonas 2023). However, no study systematically tests whether models can *explain* or *interpret* the nonsense prompts they successfully follow. The "Talking Nonsense" paper shows Babel prompts have hidden structure, but doesn't ask models to articulate that structure.

### Our Novel Contribution
1. **Self-explanation test**: Systematically measure whether LLMs can explain prompts across a perplexity spectrum
2. **Directed nonsense search**: Test whether high-perplexity prompts can direct benign tasks or only jailbreaking
3. **Explanation quality × perplexity correlation**: Quantify the comprehension threshold

### Experiment Justification
- **Experiment 1 (Explanation Quality Spectrum)**: Tests the core hypothesis across 5 prompt categories
- **Experiment 2 (Directed Nonsense Search)**: Tests whether prompt optimization generalizes beyond jailbreaking
- **Experiment 3 (Cross-Model Explanation)**: Tests whether nonsense encodes transferable meaning

## Research Question
Can LLMs explain or interpret high-perplexity prompts that successfully direct their behavior?

## Hypothesis Decomposition
- **H1**: Explanation quality decreases with prompt perplexity
- **H2**: Models that follow nonsense prompts cannot explain what they "mean"
- **H3**: Finding directed nonsense for benign tasks is harder than for jailbreaking

## Proposed Methodology

### Prompt Categories (5 levels)
1. **Natural instructions**: Clear English task descriptions
2. **Obfuscated instructions**: Leet-speak, word scrambles, pig-latin
3. **Human-crafted jailbreaks**: From datasets (medium perplexity, readable)
4. **GCG-style adversarial suffixes**: Published examples (high perplexity)
5. **Random token strings**: Control baseline (very high perplexity)

### Experiments
1. Measure GPT-2 perplexity for all prompts
2. Send each to GPT-4.1 and assess task completion
3. Ask GPT-4.1 to explain each prompt's meaning
4. Score explanation quality via LLM-as-judge (1-5 scale)
5. Search for high-perplexity prompts for benign vs. jailbreak tasks
6. Statistical analysis of perplexity-explanation correlation

### Evaluation Metrics
- GPT-2 perplexity, task success rate, explanation quality (1-5), Spearman ρ

### Statistical Plan
- Spearman correlation, Kruskal-Wallis test, Mann-Whitney U pairwise
- Bootstrap CIs, α = 0.05 with Bonferroni correction
