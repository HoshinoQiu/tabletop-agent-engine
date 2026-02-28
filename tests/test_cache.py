"""Tests for caching layer."""

from unittest.mock import MagicMock
import numpy as np
from core.cache import EmbeddingCache, SemanticCache


class TestEmbeddingCache:
    def test_put_and_get(self):
        cache = EmbeddingCache(max_size=10)
        cache.put("hello", [0.1, 0.2, 0.3])
        result = cache.get("hello")
        assert result == [0.1, 0.2, 0.3]

    def test_cache_miss(self):
        cache = EmbeddingCache(max_size=10)
        result = cache.get("nonexistent")
        assert result is None

    def test_lru_eviction(self):
        cache = EmbeddingCache(max_size=2)
        cache.put("a", [1.0])
        cache.put("b", [2.0])
        cache.put("c", [3.0])  # Should evict "a"
        assert cache.get("a") is None
        assert cache.get("b") == [2.0]
        assert cache.get("c") == [3.0]

    def test_stats(self):
        cache = EmbeddingCache()
        cache.put("x", [1.0])
        cache.get("x")  # hit
        cache.get("y")  # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1


class TestSemanticCache:
    def test_put_and_get_exact(self):
        mock_service = MagicMock()
        mock_service.embed_query.return_value = [1.0, 0.0, 0.0]
        cache = SemanticCache(mock_service, max_size=10, threshold=0.95)

        cache.put("test query", "test result")
        result = cache.get("test query")
        assert result == "test result"

    def test_cache_miss_different_query(self):
        mock_service = MagicMock()
        call_count = [0]
        def side_effect(q):
            call_count[0] += 1
            if call_count[0] <= 1:
                return [1.0, 0.0, 0.0]
            return [0.0, 1.0, 0.0]  # Very different embedding
        mock_service.embed_query.side_effect = side_effect

        cache = SemanticCache(mock_service, max_size=10, threshold=0.95)
        cache.put("query A", "result A")
        result = cache.get("query B")
        assert result is None

    def test_clear(self):
        mock_service = MagicMock()
        mock_service.embed_query.return_value = [1.0, 0.0]
        cache = SemanticCache(mock_service, max_size=10, threshold=0.9)
        cache.put("q", "r")
        cache.clear()
        assert cache.stats()["size"] == 0

    def test_stats(self):
        mock_service = MagicMock()
        mock_service.embed_query.return_value = [1.0, 0.0]
        cache = SemanticCache(mock_service, max_size=10, threshold=0.9)
        stats = cache.stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "size" in stats
