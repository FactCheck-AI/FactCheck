# Performance Analysis: Fact Verification Across Three Datasets

This document provides detailed analysis of the performance variations observed across three benchmark datasets in our fact verification evaluation: **FactBench**, **YAGO**, and **DBpedia**.

## Overview

The performance plots illustrate how four different methodologies (DKA, GIV-Z, GIV-F, and RAG) perform across multiple language models (Qwen2.5, Llama3.1, Mistral, Gemma2, and GPT-4o mini) on three distinct fact verification datasets. Each dataset presents unique challenges that significantly impact verification accuracy.

## Dataset Analysis

### FactBench Performance
<div align="center">

<a href="https://factcheck.dei.unipd.it"><picture>
    <img src="https://github.com/FactCheck-AI/FactCheck/tree/main/analysis/factbench_comparison.png"  alt="FactBench Analysis" />
</picture></a>
</div>

### YAGO Performance
<div align="center">

<a href="https://factcheck.dei.unipd.it"><picture>
<img src="https://github.com/FactCheck-AI/FactCheck/tree/main/analysis/yago_comparison.png"  alt="FactBench Analysis" />
</picture></a>
</div>

### DBpedia Performance
<div align="center">

<a href="https://factcheck.dei.unipd.it"><picture>
<img src="https://github.com/FactCheck-AI/FactCheck/tree/main/analysis/dbpedia_comparison.png"  alt="FactBench Analysis" />
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

2. **Class Imbalance is Critical**: YAGO's extreme imbalance (99% correct facts) creates a ceiling effect where detecting the rare 1% of incorrect facts becomes extremely difficult.

3. **RAG Benefits from Balance**: Retrieval-augmented approaches show dramatic improvements on balanced datasets but offer minimal gains on highly imbalanced ones.

4. **Model Consistency Varies**: Different models show varying levels of robustness across datasets, with some performing consistently (GPT-4o mini) while others show high variance (Qwen2.5).

## Implications for Fact Verification Systems

- **Benchmark Selection**: FactBench provides the clearest performance differentiation for comparing methods
- **Real-world Deployment**: YAGO-like challenges (rare errors in mostly correct data) represent common real-world scenarios
- **Method Selection**: RAG excels with balanced data but may not justify complexity gains on imbalanced datasets
- **Evaluation Strategy**: Multiple datasets with varying characteristics are essential for comprehensive method evaluation