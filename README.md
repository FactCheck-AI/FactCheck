# FactCheck: Knowledge Graph Validation via Large Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-VLDB%202025-green.svg)](#citation)

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Datasets](#-datasets)
- [Methodologies](#-methodologies)
- [Configuration](#-configuration)
- [Usage Examples](#-usage-examples)
- [Results](#-results)
- [Web Platform](#-web-platform)
- [Repository Structure](#-repository-structure)
- [Contributing](#-contributing)
- [Extra Experiments](https://github.com/FactCheck-AI/FactCheck/tree/main/extra-experiments)

## 🎯 Overview

**FactCheck** is a comprehensive benchmark designed to evaluate Large Language Models (LLMs) capabilities in Knowledge Graph (KG) fact verification. This repository implements multiple verification methodologies and provides extensive evaluation across three real-world KG datasets with over 13,530 facts.

### Key Research Questions

**FactCheck** addresses three fundamental research questions in the domain of LLM-based knowledge graph fact verification:

1. **RQ1: Internal Knowledge Assessment**: How effective are Large Language Models at knowledge graph fact-checking when relying solely on their internal knowledge representations acquired during pre-training?

2. **RQ2: External Evidence Integration**: Does augmenting LLMs with external evidence through Retrieval-Augmented Generation (RAG) methodologies improve their capability to verify knowledge graph facts, and at what computational cost?

3. **RQ3: Multi-Model Consensus**: Does aggregating predictions from multiple LLMs through consensus mechanisms lead to more reliable and robust verification of knowledge graph facts compared to individual model performance?

### Main Findings

Our evaluation across three real-world KG datasets (**FactBench**, **YAGO**, **DBpedia**) reveals several critical insights:

#### Performance Capabilities
- **Promising but Limited**: While LLMs demonstrate promising fact verification capabilities, they remain insufficient for reliable deployment in real-world KG validation scenarios
- **Model Hierarchy**: Open-source models achieve competitive performance, with Gemma2 consistently outperforming others (balanced accuracy up to 0.90 on FactBench with RAG)
- **Dataset Sensitivity**: Performance varies significantly across datasets due to class imbalance and schema complexity

#### External Evidence Integration
- **Inconsistent Improvements**: RAG integration provides fluctuating performance gains, offering inconsistent improvements over streamlined internal knowledge approaches
- **Computational Trade-offs**: External evidence integration increases computational overhead by approximately **10× processing time** (0.3s → 2.0s+ per fact)
- **Context Dependency**: RAG effectiveness is highly dependent on dataset characteristics and retrieval quality

#### Multi-Model Consensus
- **Modest Gains**: Consensus strategies provide **average 4.5% balanced accuracy improvement** over individual models in knowledge-constrained scenarios
- **Limited Consistency**: "Wisdom of the crowd" approaches fail to consistently outperform individual models across all experimental conditions
- **Resource Implications**: Consensus methods require 1-1.5× additional computational resources while providing marginal accuracy improvements

#### Methodological Insights
- **Structured Prompting**: Few-shot guided iterative verification (GIV-F) consistently outperforms zero-shot approaches
- **Evidence Quality**: External evidence effectiveness is constrained by retrieval noise and source reliability
- **Class Imbalance Challenges**: Extreme class imbalance (as in YAGO with 99.2% correct facts) presents significant verification challenges

#### Research Implications
These findings underscore the **urgent need for systematic benchmarking** and highlight the complexity of automated KG fact verification, emphasizing that current LLM-based approaches require substantial advancement before practical deployment.

---

### 📊 Detailed Analysis:
- 🔍 For comprehensive ablation study results: [Ablation Study Results](extra-experiments/ablation_study_results/README.md)
- 💰 For token usage analysis and computational cost implications: [Token Usage Analysis](extra-experiments/token/Readme.md)

## 🚀 Features

- **Multiple LLM Support**: Both open-source (Gemma2, Qwen2.5, Llama3.1, Mistral) and commercial (GPT-4o mini) models
- **Diverse Methodologies**: Direct Knowledge Assessment (DKA), Guided Iterative Verification (GIV), RAG, and Multi-model Consensus
- **Real-world Datasets**: FactBench, YAGO, and DBpedia with 13,530 total facts
- **RAG Dataset**: 2+ million documents specifically curated for KG fact verification
- **Mock API**: Simulated API for testing and development -- refer to [FactCheck MockAPI](https://github.com/FactCheck-AI/FactCheck-MockAPI).
- **Interactive Platform**: [Web-based](https://factcheck.dei.unipd.it) exploration tool for verification analysis
- **Comprehensive Evaluation**: Balanced accuracy, F1-macro, efficiency metrics, and cost analysis

## 📑 Prompt templates
Prompt templates for each methodology are available in the [prompts directory](prompts/README.md).

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- Ollama (for open-source models)
- Azure OpenAI API access (for commercial models)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/FactCheck-AI/factcheck-benchmark
cd factcheck-benchmark
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Ollama** (for open-source models)
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve
```

4. **Download required models**
```bash
ollama pull gemma2:9b
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
ollama pull mistral:7b
```

## 🚀 Quick Start

### Basic Fact Verification

```bash
python main.py # Run with default configuration in `config.yml`
```

## 📊 Datasets

### Supported Datasets

| Dataset | Facts | Predicates | Gold Accuracy | Description |
|---------|-------|------------|---------------|-------------|
| **FactBench** | 2,800 | 10 | 0.54 | Systematically generated with balanced true/false distribution |
| **YAGO** | 1,386 | 16 | 0.99 | High-quality facts with extreme class imbalance |
| **DBpedia** | 9,344 | 1,092 | 0.85 | Diverse schema with extensive predicate coverage |

### RAG Dataset

- **130,820 questions** generated from KG facts
- **2,090,305 documents** from Google SERP
- **87.4% text coverage** rate
- Similarity scores for question relevance ranking

## 🔬 Methodologies

### 1. Direct Knowledge Assessment (DKA)
Basic fact verification using only LLM's internal knowledge without external guidance.

```yaml
method:
  name: "DKA"
```

### 2. Guided Iterative Verification (GIV)
Enhanced verification with structured guidelines and examples.

**GIV-Z (Zero-shot)**:
```yaml
method:
  name: "GIV-Z"
```

**GIV-F (Few-shot)**:
```yaml
method:
  name: "GIV-F"
```

### 3. Retrieval-Augmented Generation (RAG)
Verification using external evidence from web search results.

```yaml
method:
  name: "RAG"

rag:
  embedding_model: 'bge-small-en-v1.5'
  chunking_strategy: 'sliding_window'
  window_size: 3
  similarity_cutoff: 0.3
  top_k: 6
```

### 4. Multi-model Consensus
Combines predictions from multiple models using majority voting with tie-breaking.

```yaml
majority_vote:
  mode: 'commercial'  # Options: commercial, open_source
  final_tie_breaker: 'most_consistent' # Options: least_consistent, most_consistent, Null (for commercial)
  num_votes: 3 # Number of votes for each model
  llms:
    - "mistral:7B"
    - "qwen2.5:7B"
    - "llama3.1:7B"
    - "gemma2:9B"
  higher_parameter_model:
    qwen2.5:7b: 'qwen2.5:7b'
    mistral:7b: 'mistral:7b'
    llama3.1:7b: 'llama3.1:latest'
    gemma2:9b: 'gemma2:9b'
  commercial_model:
    - "gpt-4o-mini"
```

## ⚙️ Configuration

### Example Configuration (`config.yml`)

```yaml
# Dataset configuration
dataset:
  name: "FactBench"  # Options: DBpedia, YAGO, FactBench

# Method configuration
method:
  name: "DKA"  # Options: DKA, GIV-Z, GIV-F, RAG

# LLM configuration
llm:
  mode: "open_source"  # Options: commercial, open_source
  model: "gemma2:9B"
  parameters:
    temperature: 0.75
    top_p: 0.9
    max_tokens: 512

# Evaluation configuration
evaluation:
  metrics:
    accuracy: 'balanced'  # Options: balanced, normal
    f1_score: "macro"     # Options: micro, macro, weighted

# Knowledge Graph configuration
knowledge_graph:
  kg_ids: ['correct_death_00106', 'correct_death_00040']

# Output configuration
output:
  directory: "./results"
```

### Azure OpenAI Configuration

For commercial models, configure Azure OpenAI:

```yaml
OpenAI:
  azure_endpoint: "https://your-resource.openai.azure.com/"
  api_key: "your-api-key"
  api_version: "2024-02-15-preview"
```

## 💻 Usage Examples

### 1. Single Model Evaluation

```bash
python main.py
```

### 2. Batch Evaluation

```bash
python evaluation.py --file results/factbench_results.json
```

For full evaluation use `--full` flag to include all metrics.

```bash
python evaluation.py --file results/factbench_results.json --full
```

### 3. Majority Vote Consensus
This module is interactive. You can run it as follows:
```bash
python consensus.py --dataset FactBench
```
if you don't specify files it will ask you to enter which files you want to use for the consensus.
The output example will be:
```markdown
Found 3 files for FactBench:
   1. FactBench_open-source_gemma2:9b_rag_20250527-103716.json
      Model: open-source_gemma2:9b, Method: rag
      Facts: 2, Success: 100.0%
   2. FactBench_open-source_qwen2.5:7B_rag_20250527-103404.json
      Model: open-source_qwen2.5:7B, Method: rag
      Facts: 2, Success: 100.0%
   3. FactBench_open-source_qwen2.5:7B_rag_20250527-103603.json
      Model: open-source_qwen2.5:7B, Method: rag
      Facts: 2, Success: 100.0%

How many files do you want to select? (1-3): 
```
Or you can simply define the files you want to use for the consensus:
```bash
python consensus.py --files results/factbench_open-source_gemma2:9b_rag_20250527-103716.json results/factbench_open-source_qwen2.5:7B_rag_20250527-103404.json
```

#### 🚧 Todo:
- [ ] add parallel processing for the consensus

## 📈 Results

### Performance Summary

| Method | FactBench BAcc | YAGO BAcc | DBpedia BAcc | Avg Time/Fact |
|--------|----------------|-----------|--------------|---------------|
| **DKA** | 0.72 | 0.53 | 0.64 | ~0.3s |
| **GIV-F** | 0.74 | 0.58 | 0.65 | ~0.8s |
| **RAG** | **0.90** | 0.56 | **0.67** | ~2.3s |
| **Consensus** | **0.90** | **0.64** | **0.68** | ~1.5s |

### Key Insights

1. **Model Rankings**: Gemma2 > Qwen2.5 > Mistral > Llama3.1 > GPT-4o mini
2. **Dataset Difficulty**: FactBench (easiest) > DBpedia > YAGO (hardest due to class imbalance)
3. **Cost-Performance Trade-off**: RAG provides best accuracy but 10× computational cost
4. **Consensus Benefits**: 1-5% improvement over individual models

## 🌐 Web Platform

Explore verification results interactively at: **https://factcheck.dei.unipd.it/**

### Features:
- **Fact Search**: Find specific KG triples and their verification results
- **Step-by-step Analysis**: Inspect RAG pipeline components
- **Model Comparison**: Compare reasoning patterns across different LLMs
- **Error Analysis**: Categorized failure analysis with systematic insights
- **User Feedback**: Collaborative annotation and feedback system

## 📁 Repository Structure

```
factcheck-benchmark/
├── config.yml                 # Main configuration file
├── main.py                    # Entry point for experiments
├── config.py                  # Configuration validation and management
├── data_loader.py             # Dataset loading and preprocessing
├── llm_client.py              # LLM client implementations
├── evaluate.py                # Evaluation metrics and analysis
├── requirements.txt           # Python dependencies
├── prompts/                   # Prompt templates for each methodology
├── consensus.py               # Multi-model consensus implementation
├── rag_dataset/              # RAG dataset -- filtered -- for complete dataset refer to mockapi
├── methods/
│   ├── dka.py                 # Direct Knowledge Assessment
│   ├── giv.py                 # Guided Iterative Verification
│   └── rag.py                 # Retrieval-Augmented Generation
├── dataset/
│   ├── FactBench/
│   ├── YAGO/
│   └── DBpedia/
├── results/                   # Output directory for results
└── README.md                  # This file
```

### Key Files

- **`config.py`**: Comprehensive configuration validation with support for multiple LLM providers
- **`evaluate.py`**: Scikit-learn based evaluation with balanced accuracy and F1-macro metrics
- **`methods/`**: Implementation of all verification methodologies
- **`prompts/`**: Contains prompt templates for each methodology

## 📊 Evaluation Metrics

### Implemented Metrics

```python
# Balanced Accuracy (addresses class imbalance)
BAcc = (Sensitivity + Specificity) / 2

# F1-Macro Score (unweighted average across classes)  
F1_macro = (1/N) * Σ(2 * Precision_i * Recall_i / (Precision_i + Recall_i))

# Consistency (model agreement)
Consistency = |{f ∈ F | response(m,f) = majorityVote(f)}| / |F|

# Efficiency
Time_per_fact = average_response_time_excluding_outliers
```

## 🛠️ Troubleshooting

### Common Issues

1. **Ollama Connection Error**
```bash
# Ensure Ollama is running
ollama serve

# Check available models
ollama list
```

2. **Memory Issues**
```python
# Reduce batch size or use smaller models
config["llm"]["model"] = "gemma2:2b"  # Instead of 9b
```

[//]: # (### Performance Optimization)

[//]: # ()
[//]: # (- [ ] **Parallel Processing**: Use consensus methods for better accuracy)

[//]: # (- [ ] **Caching**: Enable result caching for repeated evaluations)

[//]: # (- [ ] **Model Selection**: Choose models based on accuracy vs. speed requirements)

## 🤝 Contributing

We welcome contributions!

### Areas for Contribution

- **New Datasets**: Integration of additional KG datasets
- **Model Support**: Adding support for new LLM architectures
- **Evaluation Metrics**: Implementation of additional evaluation measures
- **Optimization**: Performance improvements and efficiency enhancements -- **important aspect**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Paper**: [2025 Proceedings](#)
- **Dataset**: [Hugging Face Repository](https://huggingface.co/FactCheck-AI)
- **Web Platform**: [https://factcheck.dei.unipd.it/](https://factcheck.dei.unipd.it/)
- **Issues**: [GitHub Issues](https://github.com/FactCheck-AI/factcheck-benchmark/issues)

## 👥 Authors

- **Farzad Shami** - University of Padua - [farzad.shami@studenti.unipd.it](mailto:farzad.shami@studenti.unipd.it)
- **Stefano Marchesin** - University of Padua - [stefano.marchesin@unipd.it](mailto:stefano.marchesin@unipd.it)  
- **Gianmaria Silvello** - University of Padua - [gianmaria.silvello@unipd.it](mailto:gianmaria.silvello@unipd.it)

---

*This work is partially supported by the HEREDITARY Project (EU Horizon Europe Grant Agreement No. GA 101137074).*
