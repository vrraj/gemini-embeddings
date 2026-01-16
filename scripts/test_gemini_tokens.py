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

def analyze_token_usage():
    """Analyze token usage for different text lengths and task types"""
    
    test_cases = [
        ("Short text", "Hello world"),
        ("Medium text", "RAG systems combine retrieval with generation to provide more accurate and contextually relevant answers by leveraging external knowledge bases."),
        ("Long text", "Retrieval-Augmented Generation (RAG) represents a significant advancement in the field of artificial intelligence and natural language processing. This innovative approach combines the strengths of large language models with external knowledge retrieval systems, enabling more accurate, factual, and contextually relevant responses. By integrating real-time information retrieval with generative AI, RAG systems can access up-to-date information, reduce hallucinations, and provide verifiable sources for their responses. This architecture has proven particularly valuable in enterprise applications, research, and educational contexts where accuracy and reliability are paramount."),
    ]
    
    task_types = [None, "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "QUESTION_ANSWERING"]
    
    print("=" * 80)
    print("GEMINI EMBEDDINGS TOKEN USAGE ANALYSIS")
    print("=" * 80)
    
    for desc, text in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST CASE: {desc}")
        print(f"Text length: {len(text)} characters")
        print(f"{'='*60}")
        
        for task_type in task_types:
            try:
                config_params = {"output_dimensionality": 1536}
                if task_type:
                    config_params["task_type"] = task_type
                
                response = client.models.embed_content(
                    model="models/gemini-embedding-001",
                    contents=[text],
                    config=genai.types.EmbedContentConfig(**config_params),
                )
                
                # Note: Gemini SDK may not return usage info in all cases
                # This is a placeholder for when usage tracking is available
                print(f"Task type: {task_type or 'None'}")
                print(f"  Embedding dimension: {len(response.embeddings[0].values)}")
                print(f"  Status: Success")
                
            except Exception as e:
                print(f"Task type: {task_type or 'None'}")
                print(f"  Error: {e}")

def estimate_costs():
    """Estimate costs for different usage scenarios"""
    
    # Gemini embedding pricing (example rates - check current rates)
    PRICE_PER_1M_TOKENS = 0.10  # Example price
    
    scenarios = [
        ("Small documents", 100, 500),      # 100 docs, 500 tokens each
        ("Medium documents", 1000, 1000),   # 1000 docs, 1000 tokens each  
        ("Large documents", 10000, 2000),  # 10000 docs, 2000 tokens each
        ("High volume", 100000, 1500),      # 100k docs, 1500 tokens each
    ]
    
    print(f"\n{'='*80}")
    print("COST ESTIMATION EXAMPLES")
    print(f"Price: ${PRICE_PER_1M_TOKENS} per 1M tokens")
    print(f"{'='*80}")
    
    for desc, num_docs, avg_tokens in scenarios:
        total_tokens = num_docs * avg_tokens
        cost = (total_tokens / 1_000_000) * PRICE_PER_1M_TOKENS
        
        print(f"\n{desc}:")
        print(f"  Documents: {num_docs:,}")
        print(f"  Average tokens/doc: {avg_tokens:,}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Estimated cost: ${cost:.2f}")

if __name__ == "__main__":
    try:
        analyze_token_usage()
        estimate_costs()
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETED")
        print("Note: Actual token usage may vary based on text content and language")
        print("Check Google AI pricing for current rates")
        print(f"{'='*80}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
