"""
Minimal LLM Handler for Gemini Embeddings
Standalone version for the gemini-embeddings repository
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from google import genai

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResult:
    """Result from embedding generation"""
    embedding: List[float]
    model: str
    provider: str
    magnitude: Optional[float] = None
    normalized: bool = False

class GeminiEmbeddingHandler:
    """Simplified Gemini embeddings handler for standalone use"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key"""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = "models/gemini-embedding-001"
    
    def create_embedding(
        self,
        text: Union[str, List[str]],
        task_type: Optional[str] = None,
        output_dimensionality: int = 1536,
        normalize: bool = True
    ) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """
        Create embeddings for text(s)
        
        Args:
            text: Single text or list of texts
            task_type: Optional task type (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, etc.)
            output_dimensionality: Embedding dimension (default: 1536)
            normalize: Whether to L2 normalize the embeddings
            
        Returns:
            EmbeddingResult or list of EmbeddingResult
        """
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]
        
        # Prepare config
        config_params = {"output_dimensionality": output_dimensionality}
        if task_type:
            config_params["task_type"] = task_type
        
        try:
            response = self.client.models.embed_content(
                model=self.model,
                contents=texts,
                config=genai.types.EmbedContentConfig(**config_params),
            )
            
            results = []
            for i, embedding_obj in enumerate(response.embeddings):
                embedding = embedding_obj.values
                
                # Calculate magnitude
                magnitude = None
                if normalize:
                    magnitude = self._calculate_magnitude(embedding)
                    embedding = self._normalize_embedding(embedding, magnitude)
                
                result = EmbeddingResult(
                    embedding=embedding,
                    model=self.model,
                    provider="gemini",
                    magnitude=magnitude,
                    normalized=normalize
                )
                results.append(result)
            
            return results[0] if not is_batch else results
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def _calculate_magnitude(self, vector: List[float]) -> float:
        """Calculate L2 magnitude of vector"""
        import math
        return math.sqrt(sum(x * x for x in vector))
    
    def _normalize_embedding(self, vector: List[float], magnitude: float) -> List[float]:
        """L2 normalize vector"""
        if magnitude > 0:
            return [x / magnitude for x in vector]
        return vector
    
    def compare_embeddings(
        self,
        text1: str,
        text2: str,
        task_type1: Optional[str] = None,
        task_type2: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare two embeddings with different task types
        
        Returns comparison metrics including cosine similarity
        """
        emb1 = self.create_embedding(text1, task_type=task_type1)
        emb2 = self.create_embedding(text2, task_type=task_type2)
        
        # Calculate cosine similarity
        similarity = self._cosine_similarity(emb1.embedding, emb2.embedding)
        
        # Calculate difference
        diff = max(abs(a - b) for a, b in zip(emb1.embedding, emb2.embedding))
        
        return {
            "text1": text1,
            "text2": text2,
            "task_type1": task_type1,
            "task_type2": task_type2,
            "cosine_similarity": similarity,
            "max_difference": diff,
            "magnitude1": emb1.magnitude,
            "magnitude2": emb2.magnitude,
            "normalized1": emb1.normalized,
            "normalized2": emb2.normalized
        }
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)

# Convenience function for quick usage
def create_gemini_embeddings(
    texts: Union[str, List[str]],
    task_type: Optional[str] = None,
    api_key: Optional[str] = None
) -> Union[EmbeddingResult, List[EmbeddingResult]]:
    """
    Quick function to create embeddings
    
    Args:
        texts: Text or list of texts to embed
        task_type: Optional task type
        api_key: Optional API key (defaults to environment variable)
        
    Returns:
        EmbeddingResult(s)
    """
    handler = GeminiEmbeddingHandler(api_key=api_key)
    return handler.create_embedding(texts, task_type=task_type)
