"""
Embedding Service: Generates embeddings using SentenceTransformers.
"""

from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer


from loguru import logger


class EmbeddingService:
    """Service for generating text embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding service.

        Args:
            model_name: Name of the SentenceTransformer model
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded (dimension: {self.dimension})")

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            query: Input text

        Returns:
            Embedding vector
        """
        logger.debug(f"Embedding query: {query[:100]}...")
        embedding = self.model.encode(query, convert_to_numpy=True, show_progress_bar=False)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        logger.debug(f"Embedding {len(texts)} documents")
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10
        )
        return embeddings.tolist()
