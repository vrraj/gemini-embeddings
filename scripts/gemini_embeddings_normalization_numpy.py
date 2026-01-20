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
    return (v / norm).tolist() if norm > 0 else vec, float(norm)

print("=" * 60)
print("NUMPY NORMALIZATION - COMPREHENSIVE ANALYSIS")
print("=" * 60)

# Process each embedding
for i, embedding_obj in enumerate(response.embeddings):
    raw_embedding = embedding_obj.values
    normalized_embedding, magnitude = l2_normalize_numpy(raw_embedding)
    
    print(f"\n--- Embedding {i+1}: '{texts[i]}' ---")
    print(f"Raw magnitude: {magnitude:.6f}")
    print(f"Raw first 5 values: {[round(x, 6) for x in raw_embedding[:5]]}")
    
    # Verify normalization
    normalized_magnitude = math.sqrt(sum(x * x for x in normalized_embedding))
    print(f"Normalized magnitude: {normalized_magnitude:.6f}")
    print(f"Normalized first 5 values: {[round(x, 6) for x in normalized_embedding[:5]]}")
    print(f"Normalization successful: {abs(normalized_magnitude - 1.0) < 1e-10}")

# Comparison analysis
print(f"\n{'='*60}")
print("NORMALIZED vs NON-NORMALIZED COMPARISON")
print(f"{'='*60}")

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (mag1 * mag2)

# Compare first two texts
text1_raw = response.embeddings[0].values
text2_raw = response.embeddings[1].values

text1_norm, _ = l2_normalize_numpy(text1_raw)
text2_norm, _ = l2_normalize_numpy(text2_raw)

# Raw similarity
raw_similarity = cosine_similarity(text1_raw, text2_raw)

# Normalized similarity (should be same as cosine similarity)
norm_similarity = sum(a * b for a, b in zip(text1_norm, text2_norm))

print(f"Comparing: '{texts[0]}' vs '{texts[1]}'")
print(f"Raw vectors cosine similarity: {raw_similarity:.6f}")
print(f"Normalized vectors dot product: {norm_similarity:.6f}")
print(f"Difference: {abs(raw_similarity - norm_similarity):.10f}")
print(f"Methods equivalent: {abs(raw_similarity - norm_similarity) < 1e-10}")

# Magnitude impact analysis
print(f"\n{'='*60}")
print("MAGNITUDE IMPACT ANALYSIS")
print(f"{'='*60}")

print("Text magnitudes (before normalization):")
for i, embedding_obj in enumerate(response.embeddings):
    raw_embedding = embedding_obj.values
    magnitude = math.sqrt(sum(x * x for x in raw_embedding))
    print(f"  Text {i+1}: {magnitude:.6f}")

print(f"\nAfter normalization - all magnitudes should be 1.0:")
for i, embedding_obj in enumerate(response.embeddings):
    raw_embedding = embedding_obj.values
    normalized_embedding, _ = l2_normalize_numpy(raw_embedding)
    normalized_magnitude = math.sqrt(sum(x * x for x in normalized_embedding))
    print(f"  Text {i+1}: {normalized_magnitude:.6f}")

# Verify all embeddings are normalized
all_normalized = all(abs(math.sqrt(sum(x * x for x in l2_normalize_numpy(e.values)[0]) - 1.0) < 1e-10 for e in response.embeddings)

print(f"\nAll embeddings normalized: {all_normalized}")

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
