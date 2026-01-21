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
try:
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=texts,
        config=genai.types.EmbedContentConfig(
            output_dimensionality=1536,
            task_type="RETRIEVAL_DOCUMENT"
        ),
    )
    
    # Get Gemini native embeddings (not normalized)
    native_embeddings = [e.values for e in response.embeddings]
    
    # Show native magnitude
    native_magnitude = math.sqrt(sum(x * x for x in native_embeddings[0]))
    print(f"Gemini native embedding magnitude (should be ~0.69): {native_magnitude:.6f}")
    print(f"First native embedding (first 5 values): {native_embeddings[0][:5]}")
    
    # Manual L2 normalization
    def l2_normalize(vec):
        mag = math.sqrt(sum(x * x for x in vec))
        return [x / mag for x in vec] if mag > 0 else vec

    embeddings_manual = [
        l2_normalize(vec)
        for vec in native_embeddings
    ]
    
    # NumPy L2 normalization
    def l2_normalize_np(vec):
        v = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(v)
        return (v / norm).tolist() if norm > 0 else vec

    embeddings_numpy = [
        l2_normalize_np(vec)
        for vec in native_embeddings
    ]
    
    print(f"\nAfter manual normalization:")
    print(f"First embedding (first 5 values): {embeddings_manual[0][:5]}")
    print(f"L2 norm of first embedding: {math.sqrt(sum(x * x for x in embeddings_manual[0])):.6f}")
    
    print(f"\nAfter numpy normalization:")
    print(f"First embedding (first 5 values): {embeddings_numpy[0][:5]}")
    print(f"L2 norm of first embedding: {math.sqrt(sum(x * x for x in embeddings_numpy[0])):.6f}")
    
    # Show magnitude difference
    magnitude_change = native_magnitude - 1.0
    print(f"\nMagnitude change: {native_magnitude:.6f} → 1.000000 (difference: {abs(magnitude_change):.6f})")
    
    # Compare results
    diff = max(abs(a - b) for a, b in zip(embeddings_manual[0], embeddings_numpy[0]))
    print(f"Max difference between manual and numpy normalization: {diff:.10f}")
    
except Exception as e:
    print(f"Error generating embeddings: {e}")
    exit(1)
