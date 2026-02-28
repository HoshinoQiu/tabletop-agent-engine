"""
RAG Engine: Handles text chunking, embedding generation, and vector storage.
"""

import re
from typing import List, Dict, Any, Tuple
from pathlib import Path

from .embeddings import EmbeddingService
from .reranker import RerankerService
from .vector_store import VectorStore
from config.settings import settings


from loguru import logger


# Kept for backward compatibility with tests/imports.
BILINGUAL_DICT = {
    "卡牌": ["card", "cards"],
    "规则": ["rule", "rules"],
    "回合": ["turn", "round"],
    "得分": ["score", "scoring", "points"],
}


class RAGEngine:
    """Main RAG orchestrator with hybrid search support."""

    def __init__(self, embedding_model: str = None):
        model = embedding_model or settings.EMBEDDING_MODEL
        store_path = settings.VECTOR_STORE_PATH
        logger.info(f"Initializing RAG engine with model: {model}")
        self.embedding_service = EmbeddingService(model_name=model)
        self.vector_store = VectorStore(
            embedding_dimension=self.embedding_service.dimension,
            store_path=store_path
        )

        # Optional reranker (cross-encoder) for final ranking refinement.
        self.reranker = RerankerService() if settings.RERANKER_ENABLED else None
        self.cache = None

        logger.info("RAG engine initialized successfully")

    def ingest_document(self, text: str, metadata: Dict[str, Any] = None,
                        batch_size: int = 32) -> int:
        """Ingest a document: chunk, embed in batches, and store."""
        logger.info("Ingesting document into vector store")
        chunks = self._chunk_text(text, metadata)
        logger.info(f"Generated {len(chunks)} chunks before filtering")

        # Filter out garbled/unreadable chunks
        good_chunks = [c for c in chunks if self._is_chunk_readable(c.text)]
        filtered = len(chunks) - len(good_chunks)
        if filtered > 0:
            logger.info(f"Filtered out {filtered} unreadable chunks")
        chunks = good_chunks

        if not chunks:
            logger.warning("No readable chunks after filtering")
            return 0

        # Add game name prefix to each chunk for better retrieval
        game_name = (metadata or {}).get("game_name", "")
        if game_name:
            for chunk in chunks:
                chunk.text = f"[{game_name}] {chunk.text}"

        # Embed and store in batches to avoid long blocks
        total = len(chunks)
        stored = 0
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = [c.text for c in batch]
            batch_metas = [c.metadata for c in batch]

            embeddings = self.embedding_service.embed_documents(batch_texts)
            self.vector_store.add_embeddings(
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_metas,
            )
            stored += len(batch)
            logger.info(f"Embedded batch {i // batch_size + 1}: {stored}/{total} chunks")

        logger.info(f"Successfully stored {stored} chunks")
        return stored

    @staticmethod
    def _is_chunk_readable(text: str) -> bool:
        """Check if a chunk contains meaningful readable text (not garbled/repetitive)."""
        if not text or len(text.strip()) < 20:
            return False
        sample = text[:500]
        # Count Chinese characters and normal ASCII letters/digits/punctuation
        readable = sum(1 for c in sample if '\u4e00' <= c <= '\u9fff' or c.isalnum() or c in ' \n\t.,;:!?()-/')
        ratio = readable / len(sample)
        if ratio < 0.5:
            return False
        # Reject chunks with excessive repetition (OCR/PDF extraction artifacts)
        # Use both word-level and character n-gram checks
        words = sample.split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                return False
        # Check for character-level repetition (e.g. "om\nom\nom\n")
        lines = [l.strip() for l in sample.split('\n') if l.strip()]
        if len(lines) > 5:
            unique_lines = len(set(lines))
            if unique_lines / len(lines) < 0.4:
                return False
        return True

    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.3) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Retrieve relevant chunks using hybrid search.

        Returns:
            Tuple of (results list with citations, is_relevant flag)
        """
        logger.info(f"Retrieving top {top_k} results for query: {query}")

        query_embedding = self.embedding_service.embed_query(query)

        reranker_enabled = bool(
            self.reranker and getattr(self.reranker, "enabled", False)
        )
        candidate_k = max(top_k, settings.RERANKER_CANDIDATES) if reranker_enabled else top_k

        # Use hybrid search if enabled, otherwise vector-only
        if settings.HYBRID_SEARCH_ENABLED:
            results = self.vector_store.search_hybrid(query, query_embedding, top_k=candidate_k)
        else:
            results = self.vector_store.search(query_embedding, top_k=candidate_k)

        if reranker_enabled and results:
            logger.info(
                f"Applying reranker on {len(results)} candidates, selecting top {top_k}"
            )
            results = self.reranker.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

        # Build citations
        for r in results:
            meta = r.get("metadata", {})
            r["citation"] = {
                "document_name": meta.get("source", meta.get("document_name", "unknown")),
                "page_number": meta.get("page_number", 0),
                "section_title": meta.get("section_title", ""),
                "chunk_text": r["document"][:150],
                "relevance_score": r["score"],
            }

        # Check relevance against the strongest fused score in current results.
        is_relevant = False
        if results:
            best_score = max(r.get("score", 0.0) for r in results)
            is_relevant = best_score >= min_score
            logger.info(f"Best score: {best_score:.4f}, threshold: {min_score}")

        logger.info(f"Retrieved {len(results)} results, is_relevant: {is_relevant}")
        return results, is_relevant

    def _chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Any]:
        """Split text into chunks using simple TextChunker."""
        from .chunking import TextChunker
        chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        chunks = chunker.chunk(text, metadata or {})
        logger.info(f"Text chunked into {len(chunks)} pieces")
        return chunks
