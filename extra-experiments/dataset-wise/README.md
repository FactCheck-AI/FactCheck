# Performance Analysis: Fact Verification Across Three Datasets

This document provides detailed analysis of the performance variations observed across three benchmark datasets in our fact verification evaluation: **FactBench**, **YAGO**, and **DBpedia**.

## Overview

The performance plots illustrate how four different methodologies (DKA, GIV-Z, GIV-F, and RAG) perform across multiple language models (Qwen2.5, Llama3.1, Mistral, Gemma2, and GPT-4o mini) on three distinct fact verification datasets. Each dataset presents unique challenges that significantly impact verification accuracy.

## Dataset Analysis

### FactBench Performance
<div align="center">

<a href="https://github.com/FactCheck-AI/FactCheck/blob/main/extra-experiments/dataset-wise/README.md"><picture>
    <img src="https://raw.githubusercontent.com/FactCheck-AI/FactCheck/e14644f1734599440b01a1fb41da8ff7c6765a40/analysis/factbench_comparison.png"  alt="FactBench Analysis" />
</picture></a>
</div>

### YAGO Performance
<div align="center">

<a href="https://github.com/FactCheck-AI/FactCheck/blob/main/extra-experiments/dataset-wise/README.md"><picture>
<img src="https://raw.githubusercontent.com/FactCheck-AI/FactCheck/e14644f1734599440b01a1fb41da8ff7c6765a40/analysis/yago_comparison.png"  alt="Yago Analysis" />
</picture></a>
</div>

### DBpedia Performance
<div align="center">

<a href="https://github.com/FactCheck-AI/FactCheck/blob/main/extra-experiments/dataset-wise/README.md"><picture>
<img src="https://raw.githubusercontent.com/FactCheck-AI/FactCheck/e14644f1734599440b01a1fb41da8ff7c6765a40/analysis/dbpedia_comparison.png"  alt="Dbpedia Analysis" />
</picture></a>
</div>


---
### RAG (Retrieval-Augmented Generation)
- **Best Performance**: FactBench (90% BAcc)
- **Most Challenging**: YAGO (54-57% BAcc)
- **Key Insight**: External knowledge retrieval most effective with balanced datasets

### GIV-F (Generate-In-Verify with Facts)
- Shows consistent moderate performance across datasets
- Less sensitive to class imbalance compared to RAG
- Provides stable baseline performance

### GIV-Z (Generate-In-Verify with Zero-shot)
- Performance varies significantly by model and dataset
- Particularly effective on FactBench with certain models (Gemma2, Mistral)

### DKA (Direct Knowledge Assessment)
- Generally lower performance across all datasets
- Struggles particularly with YAGO's class imbalance
- Most consistent performance on FactBench

## Key Insights

1. **Dataset Characteristics Drive Performance**: The fundamental properties of each dataset (class balance, error types, predicate diversity) significantly impact verification accuracy more than model choice alone.

2. **RAG Benefits from Balance**: Retrieval-augmented approaches show dramatic improvements on balanced datasets but offer minimal gains on highly imbalanced ones.

3. **Model Consistency Varies**: Different models show varying levels of robustness across datasets, with some performing consistently (GPT-4o mini) while others show high variance (Qwen2.5).
