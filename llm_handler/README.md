# Gemini Embeddings Repository

This repository contains standalone code for demonstrating Gemini embeddings, extracted from the main chat-with-rag project.

## Usage

```python
from llm_handler import GeminiEmbeddingHandler

# Initialize handler
handler = GeminiEmbeddingHandler(api_key="your_key")

# Create embeddings
result = handler.create_embedding(
    text="Your text here",
    task_type="RETRIEVAL_DOCUMENT"
)

print(f"Embedding dimension: {len(result.embedding)}")
print(f"Magnitude: {result.magnitude}")
print(f"Normalized: {result.normalized}")
```

## Quick Function

```python
from llm_handler import create_gemini_embeddings

# Quick embedding creation
embedding = create_gemini_embeddings(
    "Sample text",
    task_type="RETRIEVAL_QUERY"
)
```
