"""
Cache Layer: Semantic cache for query results and embedding cache.
"""

import hashlib
import threading
from collections import OrderedDict
from typing import Any, Optional, List, Tuple

import numpy as np

from config.settings import settings


from loguru import logger


class EmbeddingCache:
    """LRU cache for embedding vectors keyed by text hash."""

    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, text: str) -> Optional[List[float]]:
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None

    def put(self, text: str, embedding: List[float]):
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
            self.cache[key] = embedding

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "size": len(self.cache)}


class SemanticCache:
    """Cache that matches queries by embedding similarity."""

    def __init__(self, embedding_service, max_size: int = None, threshold: float = None):
        self.embedding_service = embedding_service
        self.max_size = max_size or settings.CACHE_MAX_SIZE
        self.threshold = threshold or settings.CACHE_SIMILARITY_THRESHOLD
        self.entries: OrderedDict = OrderedDict()  # key: query_text -> (embedding, result)
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get(self, query: str) -> Optional[Any]:
        """Check if a semantically similar query exists in cache."""
        if not self.entries:
            self.misses += 1
            return None

        query_embedding = np.array(self.embedding_service.embed_query(query))

        with self.lock:
            best_sim = 0.0
            best_key = None
            for key, (emb, _) in self.entries.items():
                sim = self._cosine_similarity(query_embedding, emb)
                if sim > best_sim:
                    best_sim = sim
                    best_key = key

            if best_sim >= self.threshold and best_key is not None:
                self.entries.move_to_end(best_key)
                self.hits += 1
                logger.info(f"Semantic cache hit (similarity: {best_sim:.4f})")
                return self.entries[best_key][1]

        self.misses += 1
        return None

    def put(self, query: str, result: Any):
        """Store a query result in the cache."""
        embedding = np.array(self.embedding_service.embed_query(query))
        with self.lock:
            if query in self.entries:
                self.entries.move_to_end(query)
            else:
                if len(self.entries) >= self.max_size:
                    self.entries.popitem(last=False)
            self.entries[query] = (embedding, result)

    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.entries.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "size": len(self.entries)}
