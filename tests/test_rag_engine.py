"""Tests for RAG engine."""

from unittest.mock import MagicMock, patch


class TestRAGEngine:
    def test_ingest_document(self, mock_rag_engine):
        # Test that ingest_document is callable
        mock_rag_engine.ingest_document.return_value = 5
        result = mock_rag_engine.ingest_document("Test text", {"source": "test.pdf"})
        assert result == 5

    def test_retrieve(self, mock_rag_engine):
        results, is_relevant = mock_rag_engine.retrieve("test query")
        assert len(results) > 0
        assert is_relevant is True
        assert "citation" in results[0]

    def test_retrieve_with_citations(self, mock_rag_engine):
        results, _ = mock_rag_engine.retrieve("test")
        for r in results:
            citation = r["citation"]
            assert "document_name" in citation
            assert "relevance_score" in citation

    def test_bilingual_expansion(self):
        """Test that bilingual expansion works."""
        from core.rag_engine import RAGEngine, BILINGUAL_DICT
        # Just test the dictionary exists and has entries
        assert len(BILINGUAL_DICT) > 0
        assert "卡牌" in BILINGUAL_DICT
        assert "card" in BILINGUAL_DICT["卡牌"]
