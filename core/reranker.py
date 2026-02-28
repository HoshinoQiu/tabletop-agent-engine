"""
Reranker Service: Cross-encoder based reranking for improved retrieval quality.
"""

from typing import List, Dict, Any

from config.settings import settings


from loguru import logger


class RerankerService:
    """Reranker using cross-encoder model for more accurate relevance scoring."""

    def __init__(self):
        """Initialize reranker with cross-encoder model."""
        self.model = None
        self.enabled = settings.RERANKER_ENABLED

        if self.enabled:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading reranker model: {settings.RERANKER_MODEL}")
                self.model = CrossEncoder(settings.RERANKER_MODEL)
                logger.info("Reranker model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load reranker model: {e}. Reranker disabled.")
                self.enabled = False
        else:
            logger.info("Reranker is disabled by configuration")

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder scoring.

        Args:
            query: The search query
            documents: List of document dicts with 'document' and 'score' keys
            top_k: Number of top results to return after reranking

        Returns:
            Reranked list of documents, sorted by cross-encoder score
        """
        if not self.enabled or not self.model or not documents:
            return documents[:top_k]

        logger.info(f"Reranking {len(documents)} candidates, selecting top {top_k}")

        # Build query-document pairs for cross-encoder
        pairs = [(query, doc["document"]) for doc in documents]

        try:
            # Get cross-encoder scores
            scores = self.model.predict(pairs)

            # Attach reranker scores to documents
            for doc, score in zip(documents, scores):
                doc["reranker_score"] = float(score)

            # Sort by reranker score (descending)
            reranked = sorted(documents, key=lambda x: x["reranker_score"], reverse=True)

            logger.info(f"Reranking complete. Top score: {reranked[0]['reranker_score']:.4f}")
            return reranked[:top_k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}. Returning original order.")
            return documents[:top_k]
