# Cloned Repositories

This directory contains code repositories relevant to the research project: "Do LLMs Understand Nonsense Commands?"

---

## Repository 1: llm-attacks (GCG)

- **URL**: https://github.com/llm-attacks/llm-attacks
- **Purpose**: Primary implementation of Greedy Coordinate Gradient (GCG) attack for generating adversarial suffixes
- **Location**: `code/llm-attacks/`

### Key Files
- `llm_attacks/gcg/gcg_attack.py` - Core GCG attack implementation
- `llm_attacks/base/attack_manager.py` - Attack manager class
- `experiments/` - Example experiment scripts
- `data/advbench/` - AdvBench dataset

### Dependencies
```bash
pip install -e .
# Requires: transformers, torch, fschat
```

### Usage Example
```python
from llm_attacks.gcg import GCGAttackPrompt

# See experiments/configs/ for configuration examples
# Generates adversarial suffixes that produce high-perplexity nonsense
```

### Relevance
Most relevant for generating nonsense prompts. GCG produces gibberish adversarial suffixes that can be used to test LLM understanding.

---

## Repository 2: AutoDAN

- **URL**: https://github.com/SheltonLiu-N/AutoDAN
- **Purpose**: Stealthy jailbreak attack using genetic algorithms - produces readable (low-perplexity) prompts
- **Location**: `code/AutoDAN/`

### Key Files
- `autodan_hga_eval.py` - Main evaluation script
- `autodan_ga_eval.py` - Genetic algorithm variant
- `prompts/` - Jailbreak prompt templates

### Dependencies
```bash
pip install -r requirements.txt
# Requires: transformers, torch, openai
```

### Usage Example
```python
# Run AutoDAN attack
python autodan_hga_eval.py --model vicuna --prompt_type DAN
```

### Relevance
Provides contrast to GCG - generates human-readable attacks. Useful for comparing LLM understanding of semantic vs. nonsensical prompts.

---

## Repository 3: HarmBench

- **URL**: https://github.com/centerforaisafety/HarmBench
- **Purpose**: Standardized evaluation framework for automated red teaming
- **Location**: `code/HarmBench/`

### Key Files
- `baselines/` - Implementation of various attack methods (GCG, AutoDAN, PAIR, etc.)
- `evaluate.py` - Main evaluation script
- `data/behavior_datasets/` - HarmBench behavior datasets
- `classifiers/` - Harm classifiers for evaluation

### Dependencies
```bash
pip install -e .
# Requires: transformers, torch, vllm (optional)
```

### Usage Example
```bash
# Evaluate an attack method
python evaluate.py --method GCG --model llama2 --behaviors standard
```

### Relevance
Provides comprehensive evaluation framework. Can compare multiple attack methods on same benchmark.

---

## Repository 4: JailbreakingLLMs (PAIR)

- **URL**: https://github.com/patrickrchao/JailbreakingLLMs
- **Purpose**: PAIR (Prompt Automatic Iterative Refinement) - black-box LLM jailbreaking
- **Location**: `code/JailbreakingLLMs/`

### Key Files
- `PAIR/` - PAIR attack implementation
- `TAP/` - Tree of Attacks implementation (extension of PAIR)
- `main.py` - Main entry point

### Dependencies
```bash
pip install -r requirements.txt
# Requires: openai, anthropic, google-generativeai
```

### Usage Example
```python
# Run PAIR attack
python main.py --attack PAIR --target-model vicuna --attacker-model gpt-4
```

### Relevance
Provides semantic (readable) jailbreak prompts. Useful for comparing LLM understanding of semantic vs. GCG-style nonsense.

---

## Usage Recommendations for Research

### Generating Nonsense Prompts
1. **Primary**: Use `llm-attacks` (GCG) to generate high-perplexity adversarial suffixes
2. **Contrast**: Use `AutoDAN` to generate low-perplexity readable attacks
3. **Evaluation**: Use `HarmBench` for standardized comparison

### Experiment Workflow
```python
# 1. Generate nonsense prompts with GCG
# code/llm-attacks/experiments/

# 2. Evaluate on target models
# code/HarmBench/evaluate.py

# 3. Test LLM understanding of generated prompts
# Custom experiments using generated prompts
```

### Key Research Questions to Address
1. Can LLMs explain what GCG-generated prompts "mean"?
2. How does perplexity correlate with LLM understanding?
3. Are low-perplexity attacks (AutoDAN, PAIR) more interpretable?

---

## Notes

- All repositories are cloned with `--depth 1` for space efficiency
- See individual repository READMEs for detailed setup instructions
- These tools are for AI safety research only
- Requires GPU for running attacks on large models
