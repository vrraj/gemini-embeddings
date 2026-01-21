# Release Notes

## v1.0.0 - 2025-01-21

### Release Overview

This repository contains code examples that demonstrate the concepts discussed in the article **"RAG with Gemini Embeddings: What Changes When You Move Beyond OpenAI"**.

### What's Included

- **Gemini Embeddings Examples**
  - Basic embedding generation with task types
  - Manual vs NumPy normalization comparison
  - Simple performance testing

- **Retrieval Pipeline Demo**
  - RAG implementation with Qdrant integration
  - Task type testing (`RETRIEVAL_DOCUMENT`, `RETRIEVAL_QUERY`, `QUESTION_ANSWERING`)
  - Basic error handling

- **Normalization Analysis**
  - Truncated dimension handling (1536, 768)
  - L2 normalization implementations (Python & NumPy)
  - Vector magnitude verification

### Files

- **`test_gemini_embeddings.py`** - Basic embedding generation
- **`test_gemini_embed_retrieval.py`** - Retrieval pipeline demo
- **`gemini_embeddings_normalization_*.py`** - Normalization examples
- **`gemini-embeddings.md`** - Article with technical details

### Purpose

This repository serves as a companion to the article, demonstrating:
- RAG system migration from OpenAI to Gemini
- Task type effects on retrieval quality
- Normalization approaches for vector databases
- Simple performance comparisons

---

**Focus**: Educational examples for Gemini embeddings concepts
