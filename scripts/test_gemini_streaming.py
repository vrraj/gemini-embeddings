#!/usr/bin/env python3
"""
Test script for Gemini streaming functionality.
Tests both regular and reasoning models with streaming output.
"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path for standalone repo
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

try:
    from llm_handler import GeminiEmbeddingHandler
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"📁 Script directory: {script_dir}")
    print(f"📁 Project root: {project_root}")
    print(f"📁 Project root exists: {project_root.exists()}")
    sys.exit(1)

# Import Google GenAI for streaming
try:
    from google import genai
    from google.genai import types
except ImportError as e:
    print(f"❌ Google GenAI import error: {e}")
    print("Please install: pip install google-generativeai")
    sys.exit(1)


def test_gemini_streaming():
    """Test Gemini streaming with different models."""
    
    print("🚀 Testing Gemini Streaming")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set in environment variables")
        return
    
    # Initialize client
    client = genai.Client(api_key=api_key)
    
    # Test cases
    test_cases = [
        {
            "name": "Gemini Fast Model (gemini-2.5-flash-lite)",
            "model": "models/gemini-2.5-flash-lite",
            "prompt": "Write a short poem about artificial intelligence",
            "expected_tokens": 150
        },
        {
            "name": "Gemini Pro Model (gemini-2.5-pro)",
            "model": "models/gemini-2.5-pro", 
            "prompt": "Explain quantum computing in simple terms",
            "expected_tokens": 200
        },
        {
            "name": "Gemini Flash Model (gemini-2.5-flash)",
            "model": "models/gemini-2.5-flash",
            "prompt": "Solve this step by step: If a train travels 120 km in 2 hours, and another train travels 180 km in 3 hours, which train is faster and by how much?",
            "expected_tokens": 300
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['name']}")
        print("-" * 40)
        print(f"Prompt: {test_case['prompt']}")
        print(f"Expected tokens: ~{test_case['expected_tokens']}")
        print("\n🔄 Streaming output:")
        print("-" * 20)
        
        try:
            # Create streaming request
            response = client.models.generate_content(
                model=test_case["model"],
                contents=test_case["prompt"],
                config=types.GenerateContentConfig(
                    max_output_tokens=test_case["expected_tokens"],
                    temperature=0.7,
                )
            )
            
            # Process stream
            full_text = ""
            chunk_count = 0
            
            for chunk in response:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    full_text += chunk.text
                    chunk_count += 1
            
            print("\n\n✅ Stream completed!")
            print(f"📊 Stats:")
            print(f"   - Total chunks: {chunk_count}")
            print(f"   - Total characters: {len(full_text)}")
            print(f"   - Total words: {len(full_text.split())}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print(f"   Type: {type(e).__name__}")
            
        print("\n" + "=" * 50)


def test_gemini_non_streaming():
    """Test Gemini non-streaming for comparison."""
    
    print("\n\n🔍 Testing Gemini Non-Streaming (Comparison)")
    print("=" * 50)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        return
    
    client = genai.Client(api_key=api_key)
    
    try:
        response = client.models.generate_content(
            model="models/gemini-2.5-flash-lite",
            contents="What is the capital of France? Give a brief explanation.",
            config=types.GenerateContentConfig(
                max_output_tokens=100,
                temperature=0.3,
            )
        )
        
        print("📄 Non-streaming response:")
        print("-" * 30)
        print(response.text)
        
        print(f"\n📊 Response info:")
        print(f"   - Model: gemini-2.5-flash-lite")
        print(f"   - Total characters: {len(response.text)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def test_gemini_embeddings():
    """Test Gemini embeddings."""
    
    print("\n\n🧠 Testing Gemini Embeddings")
    print("=" * 30)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        return
    
    client = genai.Client(api_key=api_key)
    
    try:
        response = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=["Hello world", "Goodbye world", "Artificial intelligence"],
            config=types.EmbedContentConfig(
                output_dimensionality=1536,
                task_type="RETRIEVAL_DOCUMENT"
            )
        )
        
        print("✅ Embeddings generated successfully!")
        print(f"   - Number of embeddings: {len(response.embeddings)}")
        print(f"   - Embedding dimension: {len(response.embeddings[0].values)}")
        
        # Show first few values of first embedding
        first_embedding = response.embeddings[0].values
        print(f"   - First embedding sample: {first_embedding[:5]}...")
        
        # Calculate magnitude
        import math
        magnitude = math.sqrt(sum(x * x for x in first_embedding))
        print(f"   - L2 magnitude: {magnitude:.6f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def test_gemini_task_types():
    """Test Gemini embeddings with different task types."""
    
    print("\n\n🎯 Testing Gemini Task Types")
    print("=" * 30)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        return
    
    client = genai.Client(api_key=api_key)
    
    test_text = "What are the main components of a RAG system?"
    task_types = [None, "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "QUESTION_ANSWERING"]
    
    for task_type in task_types:
        try:
            config_params = {"output_dimensionality": 1536}
            if task_type:
                config_params["task_type"] = task_type
            
            response = client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=[test_text],
                config=types.EmbedContentConfig(**config_params)
            )
            
            embedding = response.embeddings[0].values
            magnitude = math.sqrt(sum(x * x for x in embedding))
            
            print(f"Task type: {task_type or 'None'}")
            print(f"   - Dimension: {len(embedding)}")
            print(f"   - Magnitude: {magnitude:.6f}")
            print(f"   - First 5 values: {[round(x, 6) for x in embedding[:5]]}")
            print()
            
        except Exception as e:
            print(f"Task type {task_type}: ❌ Error: {e}")


def check_environment():
    """Check if required environment variables are set."""
    
    print("🔧 Environment Check")
    print("-" * 20)
    
    required_vars = ["GEMINI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            masked = value[:8] + "*" * (len(value) - 8) if len(value) > 8 else "*" * len(value)
            print(f"✅ {var}: {masked}")
        else:
            print(f"❌ {var}: Not set")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("Please set them in your environment or .env file")
        return False
    
    return True


def main():
    """Main test function."""
    
    print("🧪 Gemini LLM Handler Test Suite")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Exiting.")
        return
    
    print("\n✅ Environment check passed!")
    
    # Run tests
    try:
        test_gemini_streaming()
        test_gemini_non_streaming()
        test_gemini_embeddings()
        test_gemini_task_types()
        
        print("\n\n🎉 All tests completed!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
