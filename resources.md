# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project: **"Do LLMs Understand Nonsense Commands?"**

**Research Hypothesis**: Large language models may not be able to explain or interpret prompts that are optimized to produce outputs with high perplexity or that do not resemble English, suggesting that directed jailbreaks are difficult to find and may not be explainable by the models in the same way as natural language prompts.

## Papers

Total papers downloaded: 23 (15 unique papers, some with duplicate filenames from prior downloads)

| # | Title | Authors | Year | File | Relevance |
|---|-------|---------|------|------|-----------|
| 1 | Talking Nonsense: Probing LLMs' Understanding of Adversarial Gibberish | Cherepanova, Zou | 2024 | `talking_nonsense.pdf` | Core paper - LM Babel study |
| 2 | Detecting Language Model Attacks with Perplexity | Alon, Kamfonas | 2023 | `detecting_perplexity.pdf` | Perplexity-based detection |
| 3 | A la recherche du sens perdu | Erziev | 2025 | `sens_perdu.pdf` | Unicode encoding understanding |
| 4 | ASETF: From Noise to Clarity | Wang et al. | 2024 | `noise_to_clarity.pdf` | Embedding translation of suffixes |
| 5 | Deciphering the Chaos | - | 2024 | `deciphering_chaos.pdf` | Adversarial prompt translation |
| 6 | Universal and Transferable Adversarial Attacks (GCG) | Zou et al. | 2023 | `gcg_attack.pdf` | Foundational GCG attack |
| 7 | Refusal Is Mediated by a Single Direction | - | 2024 | `refusal_single_direction.pdf` | Mechanistic refusal analysis |
| 8 | AutoDAN: Interpretable Adversarial Attacks | Zhu et al. | 2023 | `autodan.pdf` | Readable adversarial prompts |
| 9 | Jailbroken: How Does LLM Safety Training Fail? | Wei et al. | 2023 | `jailbroken.pdf` | Failure mode taxonomy |
| 10 | Baseline Defenses for Adversarial Attacks | Jain et al. | 2023 | `baseline_defenses.pdf` | Perplexity filter defense |
| 11 | SmoothLLM | - | 2023 | `smoothllm.pdf` | Randomized smoothing defense |
| 12 | Universal Jailbreak Suffixes Are Attention Hijackers | - | 2025 | `attention_hijackers.pdf` | Mechanistic analysis |
| 13 | Between the Bars: Jailbreaks are Bugs that Induce Features | - | 2024 | `between_the_bars.pdf` | Bugs vs features debate |
| 14 | Toward Understanding Transferability of Adversarial Suffixes | - | 2025 | `transferability_suffixes.pdf` | Transfer mechanisms |
| 15 | Adversarial Manipulation of Reasoning Models | - | 2025 | `adversarial_reasoning.pdf` | Reasoning model attacks |

See `papers/README.md` for detailed descriptions and `literature_review.md` for synthesis.

## Datasets

Total datasets downloaded: 6 (+ 2 gated datasets documented)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| AdvBench | GCG paper | 520+574 items | Harmful behaviors/strings | `datasets/advbench/` | Standard adversarial benchmark |
| JailbreakBench | JailbreakBench | 100+100 items | Harmful+benign behaviors | `datasets/jailbreakbench/` | Includes benign control set |
| HarmBench | CAIS | 400 behaviors | Multi-category harms | `datasets/harmbench/` | Broadest category coverage |
| ChatGPT Jailbreak Prompts | Shen et al. | 79 prompts | Named jailbreaks | `datasets/chatgpt_jailbreak_prompts/` | Human-crafted, readable |
| In-the-Wild Jailbreaks | verazuo | 1405+390 items | Jailbreaks+questions | `datasets/jailbreak_llms/` | Largest human-crafted collection |
| GCG Suffix Info | llm-attacks | 387 behaviors | Transfer experiments | `datasets/gcg_suffixes/` | Metadata only; generate with llm-attacks |

See `datasets/README.md` for detailed descriptions and download instructions.

## Code Repositories

Total repositories cloned: 11

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| llm-attacks | github.com/llm-attacks/llm-attacks | GCG attack implementation | `code/llm-attacks/` |
| AutoDAN | github.com/SheltonLiu-N/AutoDAN | Readable adversarial prompts | `code/autodan/` |
| SmoothLLM | github.com/arobey1/smooth-llm | Randomized smoothing defense | `code/smooth-llm/` |
| JailbreakBench | github.com/JailbreakBench/jailbreakbench | Standardized benchmark | `code/jailbreakbench/` |
| HarmBench | github.com/centerforaisafety/HarmBench | Red teaming framework | `code/harmbench/` |
| AmpleGCG | github.com/OSU-NLP-Group/AmpleGCG | Mass suffix generation | `code/amplecgc/` |
| Baseline Defenses | github.com/neelsjain/baseline-defenses | Perplexity filter | `code/baseline-defenses/` |
| Gibberish Detector | github.com/sp-uhh/gibberish | Perplexity scoring | `code/gibberish-detector/` |
| PAIR (JailbreakingLLMs) | github.com/patrickrchao/JailbreakingLLMs | Black-box jailbreaking | `code/JailbreakingLLMs/` |

See `code/README.md` for detailed descriptions and key files.

## Resource Gathering Notes

### Search Strategy
- Used paper-finder service with multiple queries: "adversarial nonsense prompts LLM jailbreak perplexity", "LLM understanding gibberish adversarial suffixes interpretability", "perplexity filtering adversarial attacks language models safety"
- Deep-read the 4 most relevant papers (Talking Nonsense, Detecting with Perplexity, Sens Perdu, ASETF)
- Skimmed abstracts of 30+ papers, selected top 15 for download

### Selection Criteria
- Prioritized papers directly studying LLM comprehension of nonsensical inputs
- Included both attack and defense perspectives
- Covered foundational works (GCG, Jailbroken) and recent advances (2024-2025)
- Selected datasets that provide both adversarial (GCG-style) and human-readable jailbreaks for controlled comparison

### Key Findings
1. **LLMs can respond to gibberish but it's unclear if they "understand" it**: Babel prompts have structure (lower entropy than random) but appear nonsensical. Models respond correctly but removing single tokens breaks them.
2. **Perplexity is a strong signal**: GCG suffixes have extremely high perplexity (measurable by GPT-2), making them detectable but also confirming they don't resemble natural language.
3. **Some nonsense IS interpretable**: Unicode encoding attacks (Erziev) show LLMs can decode certain systematic encodings. Embedding translation (ASETF) shows adversarial embeddings carry interpretable meaning.
4. **Alignment fails for OOD inputs**: Harmful text is easier to elicit via Babel than benign text, and alignment doesn't generalize to nonsensical prompts.

## Recommendations for Experiment Design

1. **Primary dataset(s)**: AdvBench behaviors (standard adversarial benchmark) + JailbreakBench (harmful+benign pairs) + ChatGPT Jailbreak Prompts (readable control group)
2. **Baseline methods**: GCG for generating nonsense suffixes, perplexity measurement via GPT-2, AutoDAN for readable adversarial comparison
3. **Evaluation metrics**: GPT-2 perplexity of prompts, explanation quality score (LLM-as-judge), exact match rate, attack success rate
4. **Code to adapt/reuse**: llm-attacks (GCG generation), baseline-defenses (perplexity filtering), HarmBench (evaluation framework)
5. **Experimental design**:
   - Generate GCG suffixes for AdvBench behaviors
   - Collect human-crafted jailbreaks as readable control
   - Present both to LLMs with "explain this prompt" instruction
   - Measure whether models can interpret nonsense they respond to
   - Correlate explanation quality with prompt perplexity
