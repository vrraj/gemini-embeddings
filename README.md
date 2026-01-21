# Gemini Embeddings - Truncation, Normalization and Retrieval

> This repository contains the complete code implementation supporting the article: [RAG with Gemini Embeddings: What Changes When You Move Beyond OpenAI](gemini-embeddings.md)

This repository contains examples and scripts demonstrating Gemini embeddings usage, including task type nuances, normalization, and retrieval patterns that complement the practical insights discussed in the article.

## Available Tests

- **Basic Embeddings** - Generate and normalize embeddings with manual vs NumPy methods
- **Normalization Examples** - Pure Python vs NumPy normalization with performance comparison
- **Retrieval Pipeline** - Complete indexing and search with optional Qdrant integration

## Quick Start

### Prerequisites

Ensure your environment meets these requirements before proceeding:
- **Python 3.10+**: Required for all scripts and examples
- **Git** – required to clone the repository. Install: https://git-scm.com/downloads
- **Gemini API Key:** Required for embeddings and streaming tests. [Get one here](https://aistudio.google.com/app/apikey)
- **Optional: Docker & Docker Compose:** Required only for Qdrant vector database (optional for retrieval tests)

### 1. **Clone the Repository**
   ```bash
   git clone https://github.com/vrraj/gemini-embeddings.git
   cd gemini-embeddings
   
   ```

### 2. **Setup Environment - GEMINI_API_KEY**
   ```bash
   cp .env.example .env

   ```

### 3. **Create Virtual Environment**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   
   ```

   **On Windows:**
   ```
   venv\Scripts\activate
   ```

### 4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

   ```

### 5. **Optional: Setup Qdrant (for retrieval tests)**

**Option A: Docker (Recommended)**
```bash
# Ensure Docker daemon is running first
# Then start Qdrant container
docker run -p 6333:6333 qdrant/qdrant
```

**Option B: Local Python Installation**
```bash
# Install Qdrant client library (works without Docker)
pip install qdrant-client
```

> **Note**: Docker requires the Docker daemon to be running on your system. If you prefer not to use Docker, use Option B which installs the Python client directly.

### 6. **Run Tests**
   ```bash
   python scripts/test_gemini_embeddings.py
   python scripts/gemini_embeddings_normalization_python.py
   python scripts/gemini_embeddings_normalization_numpy.py
   # Need the Qdrant setup for embeddings and retrieval tests
   python scripts/test_gemini_embed_retrieval.py

   ```

### Troubleshooting

**ImportError: cannot import name 'genai' from 'google'**
```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep google-generativeai
```

**ModuleNotFoundError: No module named 'dotenv'**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Install missing dependency
pip install python-dotenv
```

**Permission Issues (macOS/Linux)**
```bash
# Use pip with user flag if needed
pip install --user -r requirements.txt

# Or fix virtual environment permissions
chmod -R 755 venv/
    
```

## Repository Structure

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
│   ├── gemini_embeddings_normalization_python.py  # Manual normalization
│   └── gemini_embeddings_normalization_numpy.py   # NumPy normalization
```

## Test Scripts

### 1. Basic Embeddings (`test_gemini_embeddings.py`)
- Demonstrates basic Gemini embeddings generation
- Shows manual vs NumPy normalization
- Prints vector values and magnitudes

### 2. Normalization Examples
- **`gemini_embeddings_normalization_python.py`** - Pure Python manual normalization
- **`gemini_embeddings_normalization_numpy.py`** - NumPy optimization with performance comparison

### 3. Retrieval Pipeline (`test_gemini_embed_retrieval.py`)
- **Complete test of task types:**
  - Document indexing: `RETRIEVAL_DOCUMENT` vs no task_type
  - Query search: `RETRIEVAL_QUERY` vs `QUESTION_ANSWERING` vs no task_type
- **Optional Qdrant integration** for similarity search (works without Qdrant too)
- **Vector comparisons** and performance analysis
- **Graceful fallback** if Qdrant not available



## Key Concepts Demonstrated

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

## What these scripts demonstrate

1. **Task Type Impact**: How `RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY` affects search performance
2. **Normalization**: Truncated dimensions and Manual vs NumPy approaches and their equivalence
3. **Vector Analysis**: First 10 values and magnitude inspection
4. **Qdrant Integration**: Complete indexing and retrieval pipeline


## Use Cases

- **RAG Systems**: Optimal document/query embedding pairs
- **Semantic Search**: Task type selection for different scenarios
- **Vector Databases**: Qdrant integration patterns


## Performance Insights

The tests reveal:
- Task type selection  (`RETRIEVAL_QUERY` , "QUESTION_ANSWERING", `RETRIEVAL_DOCUMENT`) can impact search relevance scores and performance.
- Normalization can affect cosine similarity calculations.
- 

## Article Companion

This codebase accompanies the article discussing:
- Gemini embeddings task type nuances
- Normalization
- Vector database integration best practices
- Retrieval optimization

## License

MIT License - Feel free to use these examples in your projects!

## Contributing

Found issues or improvements? Please open an issue or submit a PR!

---

**Keywords**: Gemini, Embeddings, Vector Search, RAG, Qdrant, Semantic Search, Normalization
