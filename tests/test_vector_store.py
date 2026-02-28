"""Tests for vector store."""

import numpy as np


class TestVectorStore:
    def test_add_and_search(self, temp_vector_store):
        embeddings = [np.random.randn(384).tolist() for _ in range(5)]
        documents = [f"Document {i}" for i in range(5)]
        metadatas = [{"source": f"doc{i}.pdf"} for i in range(5)]

        temp_vector_store.add_embeddings(embeddings, documents, metadatas)
        assert temp_vector_store.index.ntotal == 5

        results = temp_vector_store.search(embeddings[0], top_k=3)
        assert len(results) > 0
        assert "score" in results[0]
        assert "document" in results[0]

    def test_save_and_load(self, temp_vector_store):
        embeddings = [np.random.randn(384).tolist() for _ in range(3)]
        documents = ["Doc A", "Doc B", "Doc C"]
        metadatas = [{"source": "a.pdf"}, {"source": "b.pdf"}, {"source": "c.pdf"}]

        temp_vector_store.add_embeddings(embeddings, documents, metadatas)
        temp_vector_store.save()

        from core.vector_store import VectorStore
        loaded = VectorStore(embedding_dimension=384, store_path=str(temp_vector_store.store_path))
        assert loaded.index.ntotal == 3
        assert len(loaded.documents) == 3

    def test_hybrid_search(self, temp_vector_store):
        embeddings = [np.random.randn(384).tolist() for _ in range(3)]
        documents = ["玩家每回合抽两张牌", "攻击阶段可以选择目标", "游戏结束时计算分数"]
        metadatas = [{"source": "game.pdf"}] * 3

        temp_vector_store.add_embeddings(embeddings, documents, metadatas)
        results = temp_vector_store.search_hybrid("抽牌", embeddings[0], top_k=2)
        assert len(results) > 0

    def test_bm25_search(self, temp_vector_store):
        embeddings = [np.random.randn(384).tolist() for _ in range(3)]
        documents = ["玩家每回合抽两张牌", "攻击阶段可以选择目标", "游戏结束时计算分数"]
        metadatas = [{"source": "game.pdf"}] * 3

        temp_vector_store.add_embeddings(embeddings, documents, metadatas)
        results = temp_vector_store.search_bm25("抽牌", top_k=2)
        assert isinstance(results, list)

    def test_remove_by_source(self, temp_vector_store):
        embeddings = [np.random.randn(384).tolist() for _ in range(4)]
        documents = ["Doc A1", "Doc A2", "Doc B1", "Doc B2"]
        metadatas = [{"source": "a.pdf"}, {"source": "a.pdf"}, {"source": "b.pdf"}, {"source": "b.pdf"}]

        temp_vector_store.add_embeddings(embeddings, documents, metadatas)
        assert temp_vector_store.index.ntotal == 4

        temp_vector_store.remove_by_source("a.pdf")
        assert temp_vector_store.index.ntotal == 2
        assert all(m["source"] == "b.pdf" for m in temp_vector_store.metadatas)

    def test_empty_search(self, temp_vector_store):
        results = temp_vector_store.search([0.1] * 384, top_k=5)
        assert results == []

    def test_get_chunk_by_id(self, temp_vector_store):
        embeddings = [np.random.randn(384).tolist()]
        documents = ["Test doc"]
        metadatas = [{"chunk_id": "test-123", "source": "test.pdf"}]

        temp_vector_store.add_embeddings(embeddings, documents, metadatas)
        result = temp_vector_store.get_chunk_by_id("test-123")
        assert result is not None
        assert result["document"] == "Test doc"

        result = temp_vector_store.get_chunk_by_id("nonexistent")
        assert result is None
