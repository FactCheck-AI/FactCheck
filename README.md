# FactCheck: Knowledge Graph Validation via Large Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-VLDB%202025-green.svg)](#citation)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Datasets](#datasets)
- [Methodologies](#methodologies)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Results](#results)
- [Web Platform](#web-platform)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [Contributing](#contributing)

## 🎯 Overview

**FactCheck** is a comprehensive benchmark designed to evaluate Large Language Models (LLMs) capabilities in Knowledge Graph (KG) fact verification. This repository implements multiple verification methodologies and provides extensive evaluation across three real-world KG datasets with over 13,530 facts.

### Key Research Questions

1. **RQ1**: How effective are LLMs in fact-checking KGs using only their internal knowledge?
2. **RQ2**: Can LLMs effectively fact-check KGs using external evidence through RAG?
3. **RQ3**: Do multi-model consensus approaches improve KG fact verification accuracy?

### Main Findings

- ✅ Open-source LLMs can effectively verify KG facts (up to 0.90 balanced accuracy)
- ✅ RAG integration improves accuracy but increases computational cost (~10×)
- ✅ Multi-model consensus consistently outperforms individual models (+4.5% improvement)
- 🚧 For ablation study results, see [Ablation Study Results](results/ablation_study_results/README.md).

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

```python
from config import ConfigReader, run_fact_check_experiment

# Load configuration
config_reader = ConfigReader("config.yml")
config = config_reader.load_config()

# Validate configuration
validation_result = config_reader.validate_config()
if validation_result.is_valid:
    # Run experiment
    run_fact_check_experiment(config)
```

### Command Line Usage

```bash
# Run with default configuration
python main.py

# Evaluate results
python evaluate.py --file results/your_results.json

# Full evaluation against ground truth
python evaluate.py --file results/your_results.json --full
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
  mode: 'open_source'
  num_votes: 3
  llms:
    - "mistral:7B"
    - "qwen2.5:7B"
    - "llama3.1:8B"
    - "gemma2:9B"
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

```python
from llm_client import LLMClient
from data_loader import load_dataset

# Load configuration
config = {
    "llm": {
        "mode": "open_source",
        "model": "gemma2:9B",
        "parameters": {"temperature": 0.7}
    }
}

# Initialize client
client = LLMClient(config)

# Load dataset
facts = load_dataset("FactBench")

# Evaluate single fact
for fact_id, fact in facts[:5]:
    prompt = f"Fact: {fact['s']} {fact['p']} {fact['o']}\nIs this fact true? Reply T or F:"
    response = client.generate_response(prompt)
    print(f"Fact: {fact_id}, Prediction: {response.content}")
```

### 2. Batch Evaluation

```python
from methods.dka import run_dka_method

# Configure for batch processing
config = {
    "dataset": {"name": "FactBench"},
    "method": {"name": "DKA"},
    "llm": {"mode": "open_source", "model": "gemma2:9B"},
    "knowledge_graph": {"kg_ids": None}  # Process all facts
}

# Run DKA method
run_dka_method(config)
```

### 3. Custom Evaluation

```python
from evaluate import evaluate_results, load_results_file

# Load results
results = load_results_file("results/factbench_results.json")

# Evaluate with custom metrics
config = {
    "evaluation": {
        "metrics": {
            "accuracy": "balanced",
            "f1_score": "macro"
        }
    }
}

# Get evaluation metrics
eval_results = evaluate_results(results, config, "FactBench")
print(f"Balanced Accuracy: {eval_results['accuracy']:.3f}")
print(f"F1-Score: {eval_results['f1_score']:.3f}")
```

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
4. **Consensus Benefits**: 3-5% improvement over individual models

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
- **`data_loader.py`**: Unified dataset loading with filtering and preprocessing capabilities
- **`llm_client.py`**: Abstracted LLM client supporting both Ollama and Azure OpenAI
- **`evaluate.py`**: Scikit-learn based evaluation with balanced accuracy and F1-macro metrics
- **`methods/`**: Implementation of all verification methodologies

## 🔬 Advanced Usage

### Custom RAG Configuration

```python
# Advanced RAG setup
config = {
    "method": {"name": "RAG"},
    "rag": {
        "embedding_model": "bge-small-en-v1.5",
        "chunking_strategy": "sliding_window",
        "chunk_size": 512,
        "window_size": 3,
        "similarity_cutoff": 0.3,
        "top_k": 6
    }
}
```

### Multi-model Consensus Setup

```python
# Configure consensus with tie-breaking
config = {
    "majority_vote": {
        "mode": "open_source",
        "num_votes": 4,
        "llms": ["gemma2:9B", "qwen2.5:7B", "llama3.1:8B", "mistral:7B"],
        "higher_parameter_model": {
            "gemma2:9b": "gemma2:27b",
            "qwen2.5:7b": "qwen2.5:14b"
        }
    }
}
```

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

3. **Azure OpenAI Rate Limits**
```yaml
# Add retry configuration
llm:
  parameters:
    max_retries: 3
    timeout: 60
```

### Performance Optimization

- **Parallel Processing**: Use consensus methods for better accuracy
- **Caching**: Enable result caching for repeated evaluations
- **Model Selection**: Choose models based on accuracy vs. speed requirements

## 📚 Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{shami2025factcheck,
  title={Knowledge Graph Validation via Large Language Models},
  author={Shami, Farzad and Marchesin, Stefano and Silvello, Gianmaria},
  journal={Proceedings of the VLDB Endowment},
  volume={14},
  number={1},
  pages={XXX-XXX},
  year={2025},
  publisher={VLDB Endowment}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution

- **New Datasets**: Integration of additional KG datasets
- **Model Support**: Adding support for new LLM architectures
- **Evaluation Metrics**: Implementation of additional evaluation measures
- **Optimization**: Performance improvements and efficiency enhancements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Paper**: [VLDB 2025 Proceedings](#)
- **Dataset**: [Hugging Face Repository](https://huggingface.co/FactCheck-AI)
- **Web Platform**: [https://factcheck.dei.unipd.it/](https://factcheck.dei.unipd.it/)
- **Issues**: [GitHub Issues](https://github.com/FactCheck-AI/factcheck-benchmark/issues)

## 👥 Authors

- **Farzad Shami** - University of Padua - [farzad.shami@studenti.unipd.it](mailto:farzad.shami@studenti.unipd.it)
- **Stefano Marchesin** - University of Padua - [stefano.marchesin@unipd.it](mailto:stefano.marchesin@unipd.it)  
- **Gianmaria Silvello** - University of Padua - [gianmaria.silvello@unipd.it](mailto:gianmaria.silvello@unipd.it)

---

*This work is partially supported by the HEREDITARY Project (EU Horizon Europe Grant Agreement No. GA 101137074).*