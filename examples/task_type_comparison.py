"""
Examples for LinkedIn article - Gemini Embeddings Deep Dive
"""

import math
import os
from google import genai
import numpy as np
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Initialize client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Set GEMINI_API_KEY in .env file")

client = genai.Client(api_key=api_key)

# Example 1: Manual vs NumPy Normalization
def example_normalization_comparison():
    """Demonstrate manual vs NumPy L2 normalization"""
    
    print("=" * 60)
    print("EXAMPLE 1: Manual vs NumPy Normalization")
    print("=" * 60)
    
    text = "RAG systems enhance AI responses with external knowledge"
    
    # Get embedding
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=[text],
        config=genai.types.EmbedContentConfig(
            output_dimensionality=1536,
            task_type="RETRIEVAL_DOCUMENT"
        ),
    )
    
    raw_embedding = response.embeddings[0].values
    
    # Manual normalization
    def l2_normalize_manual(vec):
        mag = math.sqrt(sum(x * x for x in vec))
        return [x / mag for x in vec] if mag > 0 else vec
    
    # NumPy normalization  
    def l2_normalize_numpy(vec):
        v = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(v)
        return (v / norm).tolist() if norm > 0 else vec
    
    manual_norm = l2_normalize_manual(raw_embedding)
    numpy_norm = l2_normalize_numpy(raw_embedding)
    
    # Compare
    diff = max(abs(a - b) for a, b in zip(manual_norm, numpy_norm))
    
    print(f"Raw embedding magnitude: {math.sqrt(sum(x * x for x in raw_embedding)):.6f}")
    print(f"Manual normalized magnitude: {math.sqrt(sum(x * x for x in manual_norm)):.6f}")
    print(f"NumPy normalized magnitude: {math.sqrt(sum(x * x for x in numpy_norm)):.6f}")
    print(f"Max difference: {diff:.10f}")
    print(f"Methods equivalent: {diff < 1e-10}")

# Example 2: Task Type Impact
def example_task_type_impact():
    """Demonstrate how task types affect embeddings"""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Task Type Impact on Embeddings")
    print("=" * 60)
    
    text = "What are the components of a RAG system?"
    
    task_types = [None, "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "QUESTION_ANSWERING"]
    embeddings = {}
    
    for task_type in task_types:
        config = {"output_dimensionality": 1536}
        if task_type:
            config["task_type"] = task_type
        
        response = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=[text],
            config=genai.types.EmbedContentConfig(**config),
        )
        
        embeddings[task_type or "None"] = response.embeddings[0].values
    
    # Compare all pairs
    print("Embedding comparisons (max differences):")
    print("-" * 40)
    
    task_names = list(embeddings.keys())
    for i, task1 in enumerate(task_names):
        for j, task2 in enumerate(task_names[i+1:], i+1):
            diff = max(abs(a - b) for a, b in zip(embeddings[task1], embeddings[task2]))
            print(f"{task1} vs {task2}: {diff:.8f}")

# Example 3: Vector Analysis
def example_vector_analysis():
    """Show detailed vector analysis"""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Detailed Vector Analysis")
    print("=" * 60)
    
    text = "Machine learning models require large datasets"
    
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=[text],
        config=genai.types.EmbedContentConfig(
            output_dimensionality=1536,
            task_type="RETRIEVAL_DOCUMENT"
        ),
    )
    
    embedding = response.embeddings[0].values
    
    # Analysis
    magnitude = math.sqrt(sum(x * x for x in embedding))
    first_10 = embedding[:10]
    min_val = min(embedding)
    max_val = max(embedding)
    
    # Normalize
    normalized = [x / magnitude for x in embedding]
    normalized_magnitude = math.sqrt(sum(x * x for x in normalized))
    
    print(f"Text: '{text}'")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"Raw magnitude: {magnitude:.6f}")
    print(f"Value range: [{min_val:.6f}, {max_val:.6f}]")
    print(f"First 10 values: {[round(x, 6) for x in first_10]}")
    print(f"Normalized magnitude: {normalized_magnitude:.6f}")
    print(f"First 10 normalized: {[round(x, 6) for x in normalized[:10]]}")

# Example 4: Cosine Similarity
def example_cosine_similarity():
    """Calculate cosine similarity between different texts"""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Cosine Similarity Analysis")
    print("=" * 60)
    
    texts = [
        "RAG combines retrieval with generation",
        "Retrieval-augmented generation enhances AI responses",
        "The weather is sunny today",
        "Machine learning requires training data"
    ]
    
    # Get embeddings
    embeddings = []
    for text in texts:
        response = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=[text],
            config=genai.types.EmbedContentConfig(
                output_dimensionality=1536,
                task_type="RETRIEVAL_DOCUMENT"
            ),
        )
        embeddings.append(response.embeddings[0].values)
    
    # Calculate similarities
    def cosine_similarity(vec1, vec2):
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        return dot_product / (mag1 * mag2)
    
    print("Cosine Similarity Matrix:")
    print("-" * 50)
    
    for i, text1 in enumerate(texts):
        similarities = []
        for j, text2 in enumerate(texts):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(f"{sim:.3f}")
        
        # Truncate text for display
        short_text = text1[:30] + "..." if len(text1) > 30 else text1
        print(f"{short_text:<35} {'  '.join(similarities)}")

if __name__ == "__main__":
    try:
        example_normalization_comparison()
        example_task_type_impact()
        example_vector_analysis()
        example_cosine_similarity()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
