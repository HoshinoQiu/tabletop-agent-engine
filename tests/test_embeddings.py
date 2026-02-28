"""Tests for embedding service."""

from unittest.mock import patch, MagicMock


class TestEmbeddingService:
    def test_embed_query(self, mock_embedding_service):
        result = mock_embedding_service.embed_query("test")
        assert len(result) == 384

    def test_embed_documents(self, mock_embedding_service):
        result = mock_embedding_service.embed_documents(["doc1", "doc2"])
        assert len(result) == 1  # Mock returns single embedding
        assert len(result[0]) == 384

    def test_dimension(self, mock_embedding_service):
        assert mock_embedding_service.dimension == 384
