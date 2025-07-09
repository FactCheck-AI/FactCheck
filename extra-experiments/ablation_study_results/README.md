## 🛠️ NOTE
This results doesn't follow the same way of json format as the other results.

## 🤝 Todo:
- [ ] Convert this to the same format as the other results.
- [ ] Add more details about the experiments.

---
## 🔬 Ablation Study

We conducted comprehensive ablation studies to optimize the RAG pipeline components and identify the most effective configurations. All experiments were performed using the **FactBench** dataset with the **Gemma2:9B** model, measuring both accuracy and computational efficiency.

### 📊 Study Overview

The ablation study focused on five key components of the RAG methodology:

1. **Document Selection Methods**
2. **Embedding Models**
3. **Chunking Strategies**
4. **Similarity Cut-off Mechanisms**
5. **Top-K Retrieval Configuration**

### 🔍 Component Analysis

#### 1. Document Selection Methods

We evaluated both unsupervised and supervised retrieval approaches:

**Unsupervised Methods:**
- **BM25**: Traditional term frequency-based retrieval
- **Contriever (MS-MARCO)**: Dense vector representations through contrastive learning

**Supervised Methods:**
- **Jina.ai Reranker**: Cross-encoder architecture with multilingual support (26 languages)
- **MS MARCO MiniLM-L-6-v2**: BERT-based architecture optimized for ranking

**Results:** MS MARCO MiniLM-L-6-v2 achieved the highest accuracy (90.14%) with acceptable latency (0.8172s).

#### 2. Embedding Models

Tested diverse embedding models with varying sizes and capabilities:

| Model | Accuracy | Latency | Notes |
|-------|----------|---------|-------|
| **bge-small-en-v1.5** | **90.14%** | **1.70s** | ✅ Best overall performance |
| stella_en_1.5B_v5 | 89.61% | 17.69s | Large model, slower inference |
| multilingual-e5-large-instruct | 89.54% | 5.00s | Good multilingual support |
| jina-embeddings-v3* | 88.52% | 4.87s | Memory constraints |
| gte-large-en-v1.5* | 89.71% | 5.86s | Memory constraints |

*Limited evaluation due to memory constraints

#### 3. Chunking Strategies

Compared three main approaches for text segmentation:

**Fixed Chunking:**
- Chunk sizes: 256, 512, 1024 tokens
- Best: 1024 tokens (89.46% accuracy, 0.024s latency)

**Small2Big Hierarchical:**
- Multi-tier: 128, 256, 512 tokens with parent-child relations
- Result: 88.89% accuracy, 0.192s latency

**Sliding Window (Recommended):**
- Window size 3: **90.14% accuracy**, 0.031s latency
- Window size 6: 90.14% accuracy, 0.035s latency

**Results:** Sliding window with size 3 provides optimal balance of context preservation and performance.

#### 4. Similarity Cut-off Analysis

Evaluated filtering mechanisms to retain only relevant chunks:

| Method | Accuracy | Latency Improvement |
|--------|----------|-------------------|
| No cut-off (baseline) | 89.71% | - |
| **Original score cut-off** | **90.18%** | **-0.22s** |
| Re-ranked score cut-off | 90.14% | -0.35s |

**Threshold:** 0.3 (optimal balance between quality and coverage)

#### 5. Top-K Retrieval Configuration

Compared different numbers of retrieved chunks:

| Configuration | Accuracy | Latency | Trade-off |
|---------------|----------|---------|-----------|
| Top-K 3 | 90.18% | **5.21s** | Faster, slightly lower accuracy |
| **Top-K 6** | **90.32%** | 7.03s | Higher accuracy, increased latency |

### 🏆 Optimal Configuration

Based on ablation study results, the recommended RAG configuration is:

```yaml
rag:
  # Document selection
  document_selector: "ms-marco-MiniLM-L-6-v2"
  
  # Embedding model
  embedding_model: "bge-small-en-v1.5"
  
  # Chunking strategy
  chunking_strategy: "sliding_window"
  window_size: 3
  
  # Similarity filtering
  similarity_cutoff: 0.3
  cutoff_method: "original_score"
  
  # Retrieval configuration
  top_k: 6
  selected_documents: 10