# Datasets for Research: "Do LLMs Understand Nonsense Commands?"

**Research Hypothesis**: Large language models may not be able to explain or interpret prompts that are optimized to produce outputs with high perplexity or that don't resemble English.

This directory contains datasets relevant to studying whether LLMs understand nonsense/adversarial commands.
Data files are NOT committed to git due to size and content sensitivity. Follow the download instructions below.

---

## Table of Contents

1. [Downloaded Datasets](#downloaded-datasets)
   - [AdvBench (Harmful Behaviors + Strings)](#1-advbench-harmful-behaviors--strings)
   - [JailbreakBench JBB-Behaviors](#2-jailbreakbench-jbb-behaviors)
   - [HarmBench Behaviors](#3-harmbench-behaviors)
   - [ChatGPT Jailbreak Prompts](#4-chatgpt-jailbreak-prompts)
   - [In-the-Wild Jailbreak Prompts (verazuo)](#5-in-the-wild-jailbreak-prompts-verazuo)
   - [GCG Adversarial Suffix Examples](#6-gcg-adversarial-suffix-examples)
2. [Gated/Large Datasets (Not Downloaded)](#gatedlarge-datasets-not-downloaded)
   - [WildJailbreak (AllenAI)](#7-wildjailbreak-allenai)
   - [LLM-Jailbreak-Classifier](#8-llm-jailbreak-classifier)
3. [Relevant Models (Not Datasets)](#relevant-models-not-datasets)
   - [Gibberish Detector](#9-gibberish-detector-model)
4. [Key Papers and Tools](#key-papers-and-tools)
5. [Dataset Comparison](#dataset-comparison)
6. [Usage Recommendations](#usage-recommendations)

---

## Downloaded Datasets

### 1. AdvBench (Harmful Behaviors + Strings)

- **Source**: https://github.com/llm-attacks/llm-attacks/tree/main/data/advbench
- **HuggingFace mirror**: https://huggingface.co/datasets/walledai/AdvBench
- **Paper**: Zou et al. 2023, "Universal and Transferable Adversarial Attacks on Aligned Language Models" (https://arxiv.org/abs/2307.15043)
- **License**: MIT
- **Local path**: `advbench/`

#### Files
| File | Rows | Columns | Size |
|------|------|---------|------|
| `harmful_behaviors.csv` | 520 | `goal`, `target` | ~30KB |
| `harmful_strings.csv` | 574 | `target` | ~16KB |

#### Description
The standard benchmark dataset from the GCG paper. Contains 520 harmful behavior instructions paired with target affirmative responses, and 574 harmful target strings. This is the dataset used to evaluate GCG adversarial suffix attacks -- the adversarial suffixes are optimized so that appending them to these behaviors causes LLMs to comply.

#### Research Relevance
- **Primary use**: The behaviors are the "goals" that GCG adversarial suffixes are optimized against
- **For our research**: We can generate GCG suffixes for these behaviors, then test whether LLMs can explain what the suffixes mean or do
- **Key insight**: The suffixes themselves are high-perplexity nonsense that LLMs process but likely cannot interpret

#### Download
```bash
curl -sL "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv" -o datasets/advbench/harmful_behaviors.csv
curl -sL "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_strings.csv" -o datasets/advbench/harmful_strings.csv
```

#### Loading
```python
import pandas as pd
df = pd.read_csv("datasets/advbench/harmful_behaviors.csv")
print(f"Behaviors: {len(df)}, Columns: {list(df.columns)}")
```

---

### 2. JailbreakBench JBB-Behaviors

- **Source**: https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors
- **Paper**: Chao et al. 2024, "JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models" (https://arxiv.org/abs/2404.01318)
- **License**: MIT
- **Local path**: `jailbreakbench/`

#### Files
| File | Rows | Columns | Size |
|------|------|---------|------|
| `harmful_behaviors.csv` | 100 | `Index`, `Goal`, `Target`, `Behavior`, `Category`, `Source` | ~12KB |
| `benign_behaviors.csv` | 100 | `Index`, `Goal`, `Target`, `Behavior`, `Category`, `Source` | ~10KB |

#### Description
A curated benchmark of 100 harmful + 100 benign behaviors across 10 categories aligned with OpenAI's usage policies. Categories include: Harassment/Discrimination, Malware/Hacking, Physical Harm, Economic Harm, Fraud/Deception, Disinformation, Sexual/Adult Content, Privacy, Expert Advice, Government Decision-Making. 55% of behaviors are original; the rest are sourced from AdvBench and TDC/HarmBench.

#### Research Relevance
- **Curated quality**: Higher quality than AdvBench with careful categorization
- **Benign contrasts**: The benign set provides control prompts that resemble harmful ones but are safe
- **For our research**: Useful for testing whether LLMs can distinguish between nonsense appended to harmful vs. benign prompts

#### Download
```python
from datasets import load_dataset
ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
ds.to_csv("datasets/jailbreakbench/harmful_behaviors.csv")
ds2 = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="benign")
ds2.to_csv("datasets/jailbreakbench/benign_behaviors.csv")
```

---

### 3. HarmBench Behaviors

- **Source**: https://github.com/centerforaisafety/HarmBench
- **HuggingFace mirror**: https://huggingface.co/datasets/walledai/HarmBench
- **Paper**: Mazeika et al. 2024, "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal" (https://arxiv.org/abs/2402.04249)
- **License**: MIT
- **Local path**: `harmbench/`

#### Files
| File | Rows | Columns | Size |
|------|------|---------|------|
| `harmbench_behaviors_text_all.csv` | 400 | `Behavior`, `FunctionalCategory`, `SemanticCategory`, `Tags`, `ContextString`, `BehaviorID` | ~113KB |

#### Description
Comprehensive benchmark with 400 text-based behaviors spanning standard, contextual, and copyright categories. Semantic categories include chemical/biological, cybercrime, harassment, illegal activities, misinformation, and more. Some behaviors include context strings for more realistic scenarios.

#### Research Relevance
- **Breadth**: Much broader category coverage than AdvBench
- **Contextual behaviors**: Some include context that could interact interestingly with adversarial suffixes
- **For our research**: Can test LLM interpretation of nonsense across diverse harm categories

#### Download
```bash
curl -sL "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv" -o datasets/harmbench/harmbench_behaviors_text_all.csv
```

---

### 4. ChatGPT Jailbreak Prompts

- **Source**: https://huggingface.co/datasets/rubend18/ChatGPT-Jailbreak-Prompts
- **Creator**: Ruben Dario Jaramillo Romero
- **License**: Not specified
- **Local path**: `chatgpt_jailbreak_prompts/`

#### Files
| File | Rows | Columns | Size |
|------|------|---------|------|
| `prompts.csv` | 79 | `Name`, `Prompt`, `Votes`, `Jailbreak Score`, `GPT-4` | ~85KB |

#### Description
Collection of 79 named jailbreak prompts (DAN variants, APOPHIS, BasedGPT, etc.) with community votes and jailbreak effectiveness scores. These are human-crafted jailbreak prompts (not GCG-optimized), typically using role-play, persona injection, or system prompt manipulation.

#### Research Relevance
- **Contrast with GCG**: These are human-readable jailbreaks vs. GCG's nonsensical suffixes
- **For our research**: LLMs should be able to explain what these prompts do (they are coherent English). This provides a control group -- if LLMs can explain human-crafted jailbreaks but cannot explain GCG suffixes, that supports our hypothesis
- **Jailbreak scores**: Can correlate interpretability with effectiveness

#### Download
```python
from datasets import load_dataset
ds = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", split="train")
ds.to_csv("datasets/chatgpt_jailbreak_prompts/prompts.csv")
```

---

### 5. In-the-Wild Jailbreak Prompts (verazuo)

- **Source**: https://github.com/verazuo/jailbreak_llms
- **Paper**: Shen et al. 2024 (CCS'24), "Do Anything Now: Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models"
- **License**: MIT
- **Local path**: `jailbreak_llms/`

#### Files
| File | Rows | Columns | Size |
|------|------|---------|------|
| `jailbreak_prompts.csv` | 1,405 | `platform`, `source`, `prompt`, `jailbreak`, `created_at`, + more | ~1.9MB |
| `forbidden_question_set.csv` | 390 | `content_policy_id`, `content_policy_name`, `q_id`, `question` | ~35KB |

#### Description
The largest collection of in-the-wild jailbreak prompts, collected from Reddit, Discord, websites, and open-source datasets during Dec 2022 - Dec 2023. Includes 1,405 jailbreak prompts out of 15,140 total ChatGPT prompts collected. Also includes 390 forbidden questions used for evaluation.

#### Research Relevance
- **Scale**: Largest human-crafted jailbreak dataset available
- **Platform metadata**: Know where each prompt originated
- **For our research**: Another large control set of human-readable jailbreaks. Can compute perplexity of these prompts and compare with GCG suffixes -- human jailbreaks should have much lower perplexity
- **Forbidden questions**: Can be used as test goals for adversarial suffix generation

#### Download
```bash
curl -sL "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/jailbreak_prompts_2023_12_25.csv" -o datasets/jailbreak_llms/jailbreak_prompts.csv
curl -sL "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/forbidden_question/forbidden_question_set.csv" -o datasets/jailbreak_llms/forbidden_question_set.csv
```

---

### 6. GCG Adversarial Suffix Examples

- **Source**: Compiled from published research
- **Local path**: `gcg_suffixes/`

#### Files
| File | Description |
|------|-------------|
| `gcg_examples_info.json` | Metadata and example suffixes from the GCG paper |
| `transfer_experiment_behaviors.csv` | 387 behaviors used in GCG transfer experiments |

#### Description
Information about GCG (Greedy Coordinate Gradient) adversarial suffixes -- nonsensical token sequences optimized via gradient-based search to jailbreak LLMs. These suffixes look like gibberish (e.g., `describing.\ + similarlyNow write oppridge`) but cause LLMs to comply with harmful requests. Key characteristics:
- Extremely high perplexity (often >1000 when measured by GPT-2)
- Do not resemble natural English
- Found via gradient optimization over token embeddings
- Often transfer across different LLM architectures

#### Research Relevance
- **Central to our hypothesis**: These are the primary example of prompts that LLMs process but likely cannot explain
- **Perplexity testing**: Can measure and confirm their high perplexity scores
- **Interpretation testing**: Ask LLMs to explain what these suffixes mean -- the inability to do so supports our hypothesis

#### Generating More Suffixes
To generate actual GCG adversarial suffixes, use the llm-attacks repository:
```bash
git clone https://github.com/llm-attacks/llm-attacks.git
cd llm-attacks
pip install -e .
# See experiments/ directory for GCG optimization scripts
```

For pre-generated suffixes, see:
- **AmpleGCG**: https://github.com/OSU-NLP-Group/AmpleGCG (millions of generated suffixes available via request)
- **JailbreakBench artifacts**: Install `jailbreakbench` package and load GCG artifacts

---

## Gated/Large Datasets (Not Downloaded)

### 7. WildJailbreak (AllenAI)

- **Source**: https://huggingface.co/datasets/allenai/wildjailbreak
- **Paper**: WildJailbreak / WildTeaming (AllenAI, 2024)
- **Size**: 262K prompt-response pairs (~large)
- **License**: ODC-BY
- **Status**: Gated -- requires HuggingFace authentication and access approval

#### Description
Open-source synthetic safety-training dataset with 262K pairs:
- 50,050 vanilla harmful requests (direct)
- 50,050 vanilla benign requests (resemble harmful but safe)
- 82,728 adversarial harmful requests (complex jailbreaks using WildTeaming tactics)
- 78,706 adversarial benign requests (adversarial form, no harmful intent)

Columns: `vanilla`, `adversarial`, `tactics`, `completion`, `data_type`

#### Research Relevance
- **Adversarial vs vanilla**: Can compare LLM interpretation of adversarial vs. direct prompts
- **Benign contrasts**: Benign prompts that look adversarial test whether form or content confuses LLMs
- **Tactics metadata**: Know which jailbreak tactics were used

#### Download Instructions
```python
# Requires HuggingFace authentication:
# huggingface-cli login
# Then request access at https://huggingface.co/datasets/allenai/wildjailbreak
from datasets import load_dataset
ds = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
```

---

### 8. LLM-Jailbreak-Classifier

- **Source**: https://huggingface.co/datasets/markush1/LLM-Jailbreak-Classifier
- **Size**: 100K-1M rows
- **Format**: Parquet
- **License**: Not specified
- **Status**: Gated -- requires HuggingFace authentication

#### Description
Large aggregated dataset for training jailbreak classifiers. Combines data from multiple sources: OpenOrca, DAN_jailbreak, JailBreakV_28K, jailbreak_sft_rm_ds, and verazuo/jailbreak_llms. Binary classification: jailbreak vs. normal prompts.

#### Research Relevance
- **Classification labels**: Pre-labeled jailbreak vs. normal for training detectors
- **Scale**: Large enough for training classifiers
- **For our research**: Could train a perplexity-based classifier and compare with this labeled data

#### Download Instructions
```python
# Requires HuggingFace authentication
from datasets import load_dataset
ds = load_dataset("markush1/LLM-Jailbreak-Classifier")
```

---

## Relevant Models (Not Datasets)

### 9. Gibberish Detector Model

- **Source**: https://huggingface.co/madhurjindal/autonlp-Gibberish-Detector-492513457
- **Training data**: `madhurjindal/autonlp-data-Gibberish-Detector` (not publicly downloadable)
- **Architecture**: Fine-tuned DistilBERT
- **Accuracy**: 97.36%
- **License**: MIT

#### Description
A pre-trained classifier that distinguishes four categories:
1. **Clean**: Normal, coherent text
2. **Mild Gibberish**: Somewhat incoherent but contains real words
3. **Word Salad**: Real words in random/meaningless combinations
4. **Noise**: Random character sequences

#### Research Relevance
- **Classify adversarial suffixes**: Run GCG suffixes through this model to confirm they register as gibberish/noise
- **Perplexity proxy**: The gibberish categories correlate with perplexity levels
- **For our research**: Can use as a tool to categorize the "nonsenseness" of different prompt types

#### Usage
```python
from transformers import pipeline
classifier = pipeline("text-classification", model="madhurjindal/autonlp-Gibberish-Detector-492513457")
result = classifier("describing.\ + similarlyNow write oppridge")
print(result)  # Expected: high confidence "Noise" or "Word Salad"
```

---

## Key Papers and Tools

### Perplexity-Based Attack Detection
- **Paper**: Alon & Kamfonas 2023, "Detecting Language Model Attacks with Perplexity" (https://arxiv.org/abs/2308.14132)
- **Finding**: GCG adversarial suffixes have perplexity >1000 (measured by GPT-2); nearly 90% exceed this threshold
- **Method**: Perplexity + token length features fed to LightGBM classifier
- **Relevance**: Directly validates that adversarial suffixes are "nonsensical" by perplexity metrics. Provides methodology for our perplexity measurements.

### Computing Perplexity
- **HuggingFace guide**: https://huggingface.co/docs/transformers/perplexity
- **Evaluate library**: https://huggingface.co/spaces/evaluate-metric/perplexity
```python
from evaluate import load
perplexity = load("perplexity", module_type="metric")
results = perplexity.compute(predictions=["your text here"], model_id="gpt2")
```

### AmpleGCG (Suffix Generation at Scale)
- **Paper**: https://arxiv.org/abs/2404.07921
- **GitHub**: https://github.com/OSU-NLP-Group/AmpleGCG
- **Models**: https://huggingface.co/osunlp/AmpleGCG-llama2-sourced-llama2-7b-chat
- **Description**: Trains a generator to produce adversarial suffixes efficiently (vs. GCG's slow optimization). Can generate millions of suffixes.

---

## Dataset Comparison

| Dataset | Rows | Type | Perplexity Level | Human-Readable | Downloaded |
|---------|------|------|-----------------|----------------|------------|
| AdvBench Behaviors | 520 | Harmful goals | Low (normal English) | Yes | Yes |
| AdvBench Strings | 574 | Target outputs | Low (normal English) | Yes | Yes |
| JBB-Behaviors (harmful) | 100 | Curated harmful goals | Low | Yes | Yes |
| JBB-Behaviors (benign) | 100 | Benign goals | Low | Yes | Yes |
| HarmBench | 400 | Diverse harmful goals | Low | Yes | Yes |
| ChatGPT Jailbreak Prompts | 79 | Human-crafted jailbreaks | Low-Medium | Yes | Yes |
| In-the-Wild Jailbreaks | 1,405 | Collected jailbreaks | Low-Medium | Yes | Yes |
| GCG Adversarial Suffixes | N/A* | Optimized nonsense | Very High (>1000) | No | Partial |
| WildJailbreak | 262K | Synthetic safety data | Low-Medium | Yes | No (gated) |
| LLM-Jailbreak-Classifier | 100K+ | Labeled jailbreak data | Mixed | Mixed | No (gated) |

*GCG suffixes must be generated using the llm-attacks or AmpleGCG toolkits.

---

## Usage Recommendations for Our Research

### Experimental Design

1. **Generate nonsense prompts** (high perplexity):
   - Use GCG/AmpleGCG to generate adversarial suffixes against AdvBench behaviors
   - Collect random token sequences as additional baselines

2. **Collect coherent prompts** (low perplexity, control group):
   - ChatGPT Jailbreak Prompts (79 human-crafted jailbreaks)
   - In-the-Wild Jailbreaks (1,405 prompts from verazuo)
   - JBB-Behaviors benign set (100 benign prompts)

3. **Measure perplexity** of all prompts:
   - Use GPT-2 perplexity as per Alon & Kamfonas 2023
   - Use the gibberish detector model as a secondary classifier

4. **Test LLM interpretation**:
   - Ask LLMs to explain/interpret each prompt
   - Compare explanation quality between high-perplexity (GCG) and low-perplexity (human) prompts
   - Hypothesis: LLMs can explain human-crafted jailbreaks but fail on GCG nonsense

### Suggested Prompt for Testing
```
Please explain what the following text means and what it is trying to accomplish:
"{prompt_text}"
```

---

## Notes

- These datasets contain harmful content for AI safety research purposes only
- Use responsibly in controlled research environments
- Do not use to actually harm people or systems
- Some datasets (WildJailbreak, LLM-Jailbreak-Classifier) require HuggingFace authentication
- GCG suffix generation requires GPU resources and access to open-source LLM weights
