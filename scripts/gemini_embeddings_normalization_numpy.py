import math
import os
from google import genai
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize client with API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY environment variable not set")
    print("Please set it in .env file or export GEMINI_API_KEY=your_key_here")
    exit(1)

client = genai.Client(api_key=api_key)

texts = [
    "RAG systems combine retrieval with generation",
    "Gemini embeddings support multiple output dimensions",
    "Normalization matters for cosine similarity"
]

# Request embeddings
response = client.models.embed_content(
    model="models/gemini-embedding-001",
    contents=texts,
    config=genai.types.EmbedContentConfig(
        output_dimensionality=1536,
        task_type="RETRIEVAL_DOCUMENT"
    ),
)

def l2_normalize_numpy(vec):
    """L2 normalization using NumPy for better performance"""
    v = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(v)
    return (v / norm).tolist() if norm > 0 else vec

embeddings = [
    l2_normalize_numpy(e.values)
    for e in response.embeddings
]

# Print results for numpy normalization
print("=" * 60)
print("NUMPY NORMALIZATION")
print("=" * 60)
print(f"Generated {len(embeddings)} embeddings (numpy normalization)")
print(f"Embedding dimension: {len(embeddings[0])}")
print(f"First embedding (first 5 values): {embeddings[0][:5]}")
print(f"L2 norm of first embedding: {math.sqrt(sum(x * x for x in embeddings[0])):.6f}")

# Verify all embeddings are normalized
for i, embedding in enumerate(embeddings):
    norm = math.sqrt(sum(x * x for x in embedding))
    print(f"Embedding {i+1} L2 norm: {norm:.6f}")

print(f"\nAll embeddings normalized: {all(abs(math.sqrt(sum(x * x for x in emb)) - 1.0) < 1e-10 for emb in embeddings)}")

# Performance comparison
print("\n" + "=" * 60)
print("PERFORMANCE COMPARISON")
print("=" * 60)

import time

# Test manual normalization performance
start_time = time.time()
for _ in range(100):
    manual_norm = [math.sqrt(sum(x * x for x in e.values)) for e in response.embeddings]
manual_time = time.time() - start_time

# Test numpy normalization performance
start_time = time.time()
for _ in range(100):
    numpy_norm = [np.linalg.norm(e.values) for e in response.embeddings]
numpy_time = time.time() - start_time

print(f"Manual normalization (100 iterations): {manual_time:.4f}s")
print(f"NumPy normalization (100 iterations): {numpy_time:.4f}s")
print(f"NumPy speedup: {manual_time/numpy_time:.2f}x")
