import math
import os
from google import genai
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

def l2_normalize(vec):
    """Manual L2 normalization using pure Python math"""
    mag = math.sqrt(sum(x * x for x in vec))
    return [x / mag for x in vec] if mag > 0 else vec

embeddings = [
    l2_normalize(e.values)
    for e in response.embeddings
]

# Print results for manual normalization
print("=" * 60)
print("MANUAL PYTHON NORMALIZATION")
print("=" * 60)
print(f"Generated {len(embeddings)} embeddings (manual normalization)")
print(f"Embedding dimension: {len(embeddings[0])}")
print(f"First embedding (first 5 values): {embeddings[0][:5]}")
print(f"L2 norm of first embedding: {math.sqrt(sum(x * x for x in embeddings[0])):.6f}")

# Verify all embeddings are normalized
for i, embedding in enumerate(embeddings):
    norm = math.sqrt(sum(x * x for x in embedding))
    print(f"Embedding {i+1} L2 norm: {norm:.6f}")

print(f"\nAll embeddings normalized: {all(abs(math.sqrt(sum(x * x for x in emb)) - 1.0) < 1e-10 for emb in embeddings)}")
