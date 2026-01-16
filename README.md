# Gemini Embeddings - Complete Guide

This repository contains comprehensive examples and scripts demonstrating Gemini embeddings usage, including task type nuances, normalization, and retrieval patterns.

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your GEMINI_API_KEY
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Tests**
   ```bash
   python scripts/test_gemini_embeddings.py
   python scripts/test_gemini_embed_retrieval.py
   ```

## 📁 Repository Structure

```
gemini-embeddings/
├── README.md                    # This file
├── .env.example                 # Environment variables template
├── requirements.txt             # Python dependencies
├── llm_handler/                # LLM handler library
│   ├── __init__.py
│   └── llm_handler.py          # Core LLM handling logic
├── scripts/                     # Test scripts
│   ├── test_gemini_embeddings.py      # Basic embeddings test
│   ├── test_gemini_embed_retrieval.py  # Full retrieval pipeline
│   ├── test_gemini_tokens.py          # Token usage analysis
│   ├── test_gemini_streaming.py       # Complete streaming test suite
│   ├── gemini_embeddings_normalization_python.py  # Manual normalization
│   └── gemini_embeddings_normalization_numpy.py   # NumPy normalization
└── examples/                    # Code examples for article
    ├── manual_normalization.py
    ├── numpy_normalization.py
    └── task_type_comparison.py
```

## 🧪 Test Scripts

### 1. Basic Embeddings (`test_gemini_embeddings.py`)
- Demonstrates basic Gemini embeddings generation
- Shows manual vs NumPy normalization
- Prints vector values and magnitudes

### 2. Retrieval Pipeline (`test_gemini_embed_retrieval.py`)
- **Complete test of task types:**
  - Document indexing: `RETRIEVAL_DOCUMENT` vs no task_type
  - Query search: `RETRIEVAL_QUERY` vs `QUESTION_ANSWERING` vs no task_type
- **Qdrant integration** for similarity search
- **Vector comparisons** and performance analysis

### 3. Token Analysis (`test_gemini_tokens.py`)
- Token usage analysis for different text lengths
- Cost estimation examples

### 4. Normalization Examples
- **`gemini_embeddings_normalization_python.py`** - Pure Python manual normalization
- **`gemini_embeddings_normalization_numpy.py`** - NumPy optimization with performance comparison

### 5. Streaming Tests
- **`test_gemini_streaming.py`** - Complete Gemini streaming test suite
  - Multiple model testing (flash, pro, flash-lite)
  - Streaming vs non-streaming comparison
  - Embeddings with task types
  - Environment validation

## 🔧 Key Concepts Demonstrated

### Task Type Nuances
```python
# Document indexing
task_type="RETRIEVAL_DOCUMENT"

# User queries  
task_type="RETRIEVAL_QUERY"

# Question answering
task_type="QUESTION_ANSWERING"
```

### Normalization Methods
```python
# Manual L2 normalization
def l2_normalize(vec):
    mag = math.sqrt(sum(x * x for x in vec))
    return [x / mag for x in vec] if mag > 0 else vec

# NumPy L2 normalization
def l2_normalize_np(vec):
    v = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(v)
    return (v / norm).tolist() if norm > 0 else vec
```

### Vector Magnitude Analysis
All scripts show vector magnitudes to verify normalization:
```python
magnitude = math.sqrt(sum(x * x for x in embedding))
print(f"L2 norm: {magnitude:.6f}")
```

## 📊 What You'll Learn

1. **Task Type Impact**: How `RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY` affects search performance
2. **Normalization**: Manual vs NumPy approaches and their equivalence
3. **Vector Analysis**: First 10 values and magnitude inspection
4. **Qdrant Integration**: Complete indexing and retrieval pipeline
5. **Best Practices**: Production-ready embedding patterns

## 🎯 Use Cases

- **RAG Systems**: Optimal document/query embedding pairs
- **Semantic Search**: Task type selection for different scenarios
- **Vector Databases**: Qdrant integration patterns
- **Production Systems**: Error handling and retry logic

## 📈 Performance Insights

The tests reveal:
- `RETRIEVAL_QUERY` typically outperforms `RETRIEVAL_DOCUMENT` for user queries
- Normalization is critical for cosine similarity calculations
- Task type selection impacts search relevance scores

## 🔗 LinkedIn Article Companion

This codebase accompanies the LinkedIn article discussing:
- Gemini embeddings task type nuances
- Production-ready implementation patterns
- Vector database integration best practices
- Performance optimization techniques

## 📝 License

MIT License - Feel free to use these examples in your projects!

## 🤝 Contributing

Found issues or improvements? Please open an issue or submit a PR!

---

**Keywords**: Gemini, Embeddings, Vector Search, RAG, Qdrant, Semantic Search, Normalization
