"""Shared pytest fixtures for the test suite."""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service that returns fixed-dimension vectors."""
    service = MagicMock()
    service.dimension = 384
    service.embed_query.return_value = [0.1] * 384
    service.embed_documents.return_value = [[0.1] * 384]
    return service


@pytest.fixture
def temp_vector_store(tmp_path):
    """Create a temporary vector store for testing."""
    from core.vector_store import VectorStore
    store = VectorStore(embedding_dimension=384, store_path=str(tmp_path / "test_store"))
    return store


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {"text": "玩家每回合可以抽两张牌。", "metadata": {"source": "game1.pdf", "page_number": 1}},
        {"text": "The player draws two cards each turn.", "metadata": {"source": "game1.pdf", "page_number": 1}},
        {"text": "胜利条件是收集10个资源。", "metadata": {"source": "game2.pdf", "page_number": 5}},
        {"text": "攻击阶段，玩家可以选择一个目标进行攻击。", "metadata": {"source": "game1.pdf", "page_number": 3}},
        {"text": "游戏结束时，分数最高的玩家获胜。", "metadata": {"source": "game2.pdf", "page_number": 10}},
    ]


@pytest.fixture
def mock_rag_engine(mock_embedding_service):
    """Mock RAG engine for testing."""
    engine = MagicMock()
    engine.embedding_service = mock_embedding_service
    engine.vector_store = MagicMock()
    engine.vector_store.metadatas = [
        {"source": "game1.pdf"},
        {"source": "game2.pdf"},
    ]
    engine.retrieve.return_value = (
        [{"document": "测试规则内容", "score": 0.85, "metadata": {"source": "test.pdf"}, "citation": {"document_name": "test.pdf", "page_number": 1, "section_title": "", "chunk_text": "测试规则内容", "relevance_score": 0.85}}],
        True,
    )
    return engine


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    from unittest.mock import patch, MagicMock
    import anyio
    import httpx

    class SyncASGIClient:
        """Sync facade for httpx.AsyncClient over ASGITransport (httpx>=0.28 compatible)."""

        def __init__(self, app):
            self._client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            )

        def request(self, method: str, url: str, **kwargs):
            async def _request():
                return await self._client.request(method, url, **kwargs)

            return anyio.run(_request)

        def get(self, url: str, **kwargs):
            return self.request("GET", url, **kwargs)

        def post(self, url: str, **kwargs):
            return self.request("POST", url, **kwargs)

        def delete(self, url: str, **kwargs):
            return self.request("DELETE", url, **kwargs)

        def close(self):
            anyio.run(self._client.aclose)

    mock_engine = MagicMock()
    mock_engine.cache = None
    mock_agent = MagicMock()
    mock_agent.tool_registry = MagicMock()
    mock_agent.tool_registry.list_tools.return_value = ["retrieve_rules", "list_games"]
    mock_agent.tool_registry.get_tools_prompt.return_value = "- retrieve_rules: 检索规则\n- list_games: 列出游戏"
    mock_agent.query.return_value = {
        "query": "test",
        "game_state": None,
        "iterations": 1,
        "thought_chain": [],
        "final_response": "测试回答",
        "citations": [],
    }

    mock_session_mgr = MagicMock()
    mock_session_mgr.active_count.return_value = 0
    mock_session_mgr.create_session.return_value = "test-session-id"
    mock_session_mgr.get_session.return_value = MagicMock()
    mock_session_mgr.get_session.return_value.get_history.return_value = []

    mock_doc_mgr = MagicMock()
    mock_doc_mgr.documents = {}
    mock_doc_mgr.list_documents.return_value = []

    with patch("api.main.rag_engine", mock_engine), \
         patch("api.main.react_agent", mock_agent), \
         patch("api.main.session_manager", mock_session_mgr), \
         patch("api.main.document_manager", mock_doc_mgr):
        from api.main import app
        client = SyncASGIClient(app)
        yield client
        client.close()
