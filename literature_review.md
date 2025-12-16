# Literature Review: Do LLMs Understand Nonsense Commands?

## Research Area Overview

This literature review examines the intersection of adversarial prompting, jailbreaking, and LLM understanding of nonsense/gibberish inputs. The research hypothesis under investigation is that LLMs may not be able to explain or interpret prompts that are optimized to produce high-perplexity outputs or outputs that do not resemble English, suggesting that directed prompt-based "jailbreaking" is fundamentally different from standard prompt understanding.

The field has seen rapid development in both attack methods (GCG, AutoDAN, PAIR, COLD-Attack) and defense mechanisms (perplexity filtering, adversarial training). A key finding across this literature is that LLMs can be manipulated by seemingly nonsensical prompts to generate coherent—and often harmful—outputs, yet the models cannot explain or interpret these prompts when asked directly.

---

## Key Papers

### Paper 1: Talking Nonsense: Probing Large Language Models' Understanding of Adversarial Gibberish Inputs

- **Authors**: Valeriia Cherepanova, James Zou
- **Year**: 2024
- **Source**: arXiv:2404.17120 (under review at ICML)
- **Key Contribution**: Most directly relevant paper to our hypothesis. Systematically studies "LM Babel"—nonsensical prompts that compel LLMs to generate coherent responses.
- **Methodology**: Uses Greedy Coordinate Gradient (GCG) optimizer to craft prompts that induce specific target text generation. Analyzes success rates across different text types (Wikipedia, CC-News, AESLC, AdvBench).
- **Datasets Used**: Wikipedia, CC-News, AESLC (corporate emails), AdvBench (harmful strings)
- **Key Findings**:
  - Manipulation efficiency depends on target text's length and perplexity
  - Babel prompts often located in lower loss minima compared to natural prompts
  - Generating harmful texts is NOT more difficult than benign texts, suggesting lack of alignment for OOD prompts
  - Success rate significantly decreases with minor alterations (below 20% with single token removal)
  - Vicuna models more susceptible than LLaMA models
- **Results**: Success rates: Vicuna-7B 35-81%, LLaMA-7B 20-55% across datasets
- **Code Available**: Not explicitly mentioned
- **Relevance to Our Research**: **Highly relevant** - directly addresses whether LLMs "understand" their own gibberish language

---

### Paper 2: Universal and Transferable Adversarial Attacks on Aligned Language Models (GCG)

- **Authors**: Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, Matt Fredrikson
- **Year**: 2023
- **Source**: arXiv:2307.15043
- **Key Contribution**: Introduces the Greedy Coordinate Gradient (GCG) attack—the foundational method for automatic adversarial suffix generation
- **Methodology**:
  1. Target affirmative responses (e.g., "Sure, here is...")
  2. Combined greedy and gradient-based discrete token optimization
  3. Multi-prompt and multi-model attack training for transferability
- **Datasets Used**: Custom harmful behavior prompts (100 behaviors), AdvBench
- **Key Findings**:
  - Adversarial suffixes are highly transferable across models
  - 88% exact match rate on Vicuna, transfers to ChatGPT (84%), GPT-4, Bard, Claude (2.1%)
  - Generated suffixes appear as "gibberish" or "amalgamation of tokens"
  - Easily detectable by perplexity-based defenses
- **Results**: 99/100 harmful behaviors generated on Vicuna
- **Code Available**: Yes - github.com/llm-attacks/llm-attacks
- **Relevance to Our Research**: **Foundational** - provides the attack method (GCG) and demonstrates that gibberish prompts can manipulate LLMs

---

### Paper 3: AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models

- **Authors**: Xiaogeng Liu, Nan Xu, Muhao Chen, Chaowei Xiao
- **Year**: 2024 (ICLR 2024)
- **Source**: arXiv:2310.04451
- **Key Contribution**: Addresses the stealthiness problem of GCG by generating semantically meaningful jailbreak prompts using genetic algorithms
- **Methodology**:
  - Hierarchical genetic algorithm operating at sentence and paragraph level
  - Uses handcrafted jailbreak prompts (DAN series) as initialization
  - Momentum word scoring scheme for fine-grained search
- **Datasets Used**: AdvBench harmful behaviors
- **Key Findings**:
  - Bypasses perplexity-based defenses effectively
  - Superior transferability and universality vs. GCG
  - 60% improvement over GCG baseline when considering defense
  - Produces human-readable prompts unlike GCG
- **Results**: Improves attack success rate by 10%+ on robust models like Llama2
- **Code Available**: Yes - https://github.com/SheltonLiu-N/AutoDAN
- **Relevance to Our Research**: **Important** - shows that stealthy (low-perplexity) attacks bypass detection, while GCG's high-perplexity outputs are detectable

---

### Paper 4: Jailbreaking Black Box Large Language Models in Twenty Queries (PAIR)

- **Authors**: Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, Eric Wong
- **Year**: 2023
- **Source**: arXiv:2310.08419
- **Key Contribution**: Black-box jailbreaking using LLM-vs-LLM approach with semantic, human-readable prompts
- **Methodology**:
  - Uses an "attacker" LLM to iteratively refine prompts against a "target" LLM
  - Inspired by social engineering attacks
  - Requires only ~20 queries (250x more efficient than GCG)
- **Datasets Used**: AdvBench (subset of 50 behaviors)
- **Key Findings**:
  - Generates semantic (meaningful) jailbreaks vs. gibberish
  - Competitive success rates with GCG but much faster
  - Highly transferable to GPT-3.5/4, Vicuna, PaLM
- **Results**: <20 queries for successful jailbreak
- **Code Available**: Yes - https://github.com/patrickrchao/JailbreakingLLMs
- **Relevance to Our Research**: **Important** - provides contrast between semantic (understandable) and nonsensical (GCG-style) attacks

---

### Paper 5: HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal

- **Authors**: Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, et al.
- **Year**: 2024
- **Source**: arXiv:2402.04249
- **Key Contribution**: Standardized benchmark for evaluating red teaming attacks and defenses
- **Methodology**:
  - 510 carefully curated harmful behaviors across diverse categories
  - 18 red teaming methods and 33 target LLMs evaluated
  - Introduces adversarial training method for robust refusal
- **Datasets Used**: Novel benchmark with behaviors from AdvBench, Trojan Red Teaming Competition, and new categories (contextual, copyright, multimodal)
- **Key Findings**:
  - No current attack or defense is uniformly effective
  - Robustness is independent of model size
  - Attack success rate is stable within model families but variable across families
  - Number of generated tokens drastically impacts ASR measurement
- **Results**: Large-scale comparison of 18 methods × 33 models
- **Code Available**: Yes - https://github.com/centerforaisafety/HarmBench
- **Relevance to Our Research**: **Essential** - provides standardized evaluation framework and AdvBench dataset

---

### Paper 6: COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability

- **Authors**: Xingang Guo, Fangxu Yu, Huan Zhang, Lianhui Qin, Bin Hu
- **Year**: 2024 (ICML 2024)
- **Source**: arXiv:2402.08679
- **Key Contribution**: Unifies controllability and stealthiness in white-box attacks via energy-based constrained decoding
- **Methodology**:
  - Adapts COLD (Energy-based Constrained Decoding with Langevin Dynamics)
  - Continuous logit space optimization instead of discrete tokens
  - Supports multiple constraints: fluency, sentiment, coherence, position
- **Datasets Used**: AdvBench, custom attack scenarios
- **Key Findings**:
  - Can generate fluent suffix attacks, paraphrase attacks, and positional attacks
  - More efficient than GCG (no greedy search step)
  - Maintains stealthiness while being controllable
  - Good transferability to GPT-3.5, GPT-4
- **Results**: Outperforms AutoDAN-Zhu in suffix attack setting
- **Code Available**: Yes - https://github.com/Yu-Fangxu/COLD-Attack
- **Relevance to Our Research**: **Useful** - shows the spectrum from gibberish to fluent attacks

---

### Paper 7: JailbreakBench: An Open Robustness Benchmark for Jailbreaking Language Models

- **Authors**: Various (NeurIPS 2024 Datasets and Benchmarks Track)
- **Year**: 2024
- **Source**: arXiv:2404.01318
- **Key Contribution**: Open-source robustness benchmark with 100 distinct misuse behaviors
- **Methodology**: Standardized evaluation pipeline with behaviors from AdvBench, Trojan Red Teaming Competition, and Shah et al.
- **Datasets Used**: 100 harmful behaviors benchmark
- **Key Findings**: Provides standardized comparison across attacks
- **Code Available**: Yes - https://github.com/JailbreakBench/jailbreakbench
- **Relevance to Our Research**: **Useful** - additional benchmark dataset

---

### Paper 8: AutoDAN: Interpretable Gradient-Based Adversarial Attacks on Large Language Models

- **Authors**: Zhu et al.
- **Year**: 2024
- **Source**: arXiv:2310.15140
- **Key Contribution**: Different approach to AutoDAN using gradient-based interpretable attacks
- **Methodology**: Generates semantically coherent suffixes from scratch using gradients
- **Key Findings**:
  - Generated prompts are interpretable and diverse
  - Emerging strategies similar to manual jailbreaks
  - Better black-box transfer than unreadable counterparts
  - Bypasses perplexity filters better than GCG
- **Code Available**: Yes
- **Relevance to Our Research**: **Useful** - interpretable attacks provide contrast to nonsense attacks

---

## Common Methodologies

### Attack Generation Methods
1. **Token-level Gradient Optimization (GCG)**: Used in GCG, AutoDAN-Zhu
   - Pros: Effective, transferable
   - Cons: Produces gibberish, detectable by perplexity filters

2. **Genetic Algorithms (AutoDAN-Liu)**: Used in AutoDAN
   - Pros: Produces readable prompts, bypasses perplexity defense
   - Cons: Requires initialization with handcrafted prompts

3. **LLM-based Iterative Refinement (PAIR)**: Used in PAIR, TAP
   - Pros: Efficient (~20 queries), black-box
   - Cons: Requires attacker LLM

4. **Energy-based Constrained Decoding (COLD)**: Used in COLD-Attack
   - Pros: Controllable, efficient, stealthy
   - Cons: More complex setup

### Defense Methods
1. **Perplexity Filtering**: Simple but effective against GCG-style attacks
2. **Adversarial Training**: Most robust but computationally expensive
3. **Input/Output Filtering**: Used in production but can be bypassed

---

## Standard Baselines

| Baseline | Type | Typical ASR | Notes |
|----------|------|-------------|-------|
| GCG | White-box | 88% (Vicuna) | Gibberish outputs |
| AutoDAN | White-box | ~60% higher than GCG vs. defense | Readable prompts |
| PAIR | Black-box | Competitive with GCG | Semantic prompts |
| Random search | Baseline | ~10-30% | Simple baseline |
| Human jailbreaks | Manual | Variable | DAN series |

---

## Evaluation Metrics

1. **Attack Success Rate (ASR)**: Percentage of test cases that elicit harmful behavior
2. **Exact Match Rate**: Target text exactly reproduced
3. **Conditional Perplexity**: How "unexpected" the target text is given the prompt
4. **Keyword/Substring Matching**: Presence of specific harmful content
5. **Classifier-based Detection**: LLM judges whether output is harmful

---

## Datasets in the Literature

| Dataset | Description | Size | Used In |
|---------|-------------|------|---------|
| **AdvBench** | Harmful behaviors/strings | 500+ behaviors | GCG, AutoDAN, HarmBench, PAIR |
| **HarmBench** | Standardized harmful behaviors | 510 behaviors | HarmBench evaluation |
| **JailbreakBench** | Misuse behaviors | 100 behaviors | JailbreakBench |
| **Wikipedia** | Benign text | Variable | Talking Nonsense |
| **CC-News** | News articles | Variable | Talking Nonsense |
| **AESLC** | Corporate emails | Variable | Talking Nonsense |

---

## Gaps and Opportunities

### Gap 1: Understanding vs. Exploitation
Current literature focuses on *exploiting* LLMs with nonsense prompts but rarely investigates whether LLMs can *explain* or *interpret* these prompts. The "Talking Nonsense" paper is the only work that systematically studies this.

### Gap 2: Bidirectional Analysis
No paper systematically tests whether LLMs can:
- Explain what a nonsense prompt "means"
- Predict what output a nonsense prompt would generate
- Identify that a prompt is adversarial

### Gap 3: Perplexity-Understanding Correlation
While perplexity is used for detection, no work correlates prompt perplexity with LLM's ability to explain the prompt.

### Gap 4: Semantic vs. Non-semantic Attack Understanding
PAIR and AutoDAN generate semantic attacks; GCG generates nonsense. No work compares LLM understanding of both types.

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **Primary**: AdvBench (standard in field, includes harmful behaviors)
2. **Secondary**: Wikipedia/CC-News (benign targets from "Talking Nonsense")
3. **Supplementary**: HarmBench behaviors (broader coverage)

### Recommended Baselines
1. **GCG** - generates high-perplexity nonsense suffixes (most relevant)
2. **AutoDAN** - generates low-perplexity readable prompts (contrast)
3. **Random tokens** - baseline for true nonsense

### Recommended Metrics
1. **Prompt Perplexity**: Measure how "nonsensical" the prompt appears
2. **LLM Explanation Quality**: Can the LLM explain what the prompt means?
3. **Attack Success Rate**: Does the prompt achieve its intended effect?
4. **Explanation-Success Correlation**: Are explainable prompts less effective?

### Methodological Considerations
1. Use multiple LLMs (Llama-2, Vicuna, Mistral) for generalizability
2. Include both successful and failed attack prompts
3. Test explanation ability before and after attack execution
4. Control for prompt length and complexity
5. Consider using the GCG codebase for generating nonsense prompts

---

## Key References for Implementation

1. **GCG Implementation**: https://github.com/llm-attacks/llm-attacks
2. **AutoDAN Implementation**: https://github.com/SheltonLiu-N/AutoDAN
3. **PAIR Implementation**: https://github.com/patrickrchao/JailbreakingLLMs
4. **HarmBench Framework**: https://github.com/centerforaisafety/HarmBench
5. **COLD-Attack**: https://github.com/Yu-Fangxu/COLD-Attack
6. **JailbreakBench**: https://github.com/JailbreakBench/jailbreakbench

---

## Summary

The literature strongly supports the research hypothesis that LLMs respond to nonsense prompts without truly "understanding" them. The "Talking Nonsense" paper provides the most direct evidence, showing that:

1. LLMs can be manipulated by gibberish (LM Babel) to produce any target text
2. These prompts are fragile—minor changes break them
3. Alignment doesn't protect against out-of-distribution (nonsense) prompts
4. Harmful text generation is not harder than benign text generation

This suggests that prompt-based jailbreaking exploits a different mechanism than standard prompt understanding—the model is optimized for token prediction, not semantic comprehension of its inputs.
