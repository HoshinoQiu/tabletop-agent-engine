"""Tests for API endpoints."""


class TestHealthEndpoints:
    def test_status(self, test_client):
        resp = test_client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


class TestQueryEndpoints:
    def test_query(self, test_client):
        resp = test_client.post("/api/query", json={
            "query": "游戏怎么玩？",
            "game_state": {"hand_cards": 0, "phase": "setup", "other_info": {}}
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "final_response" in data
        assert "session_id" in data

    def test_ask(self, test_client):
        resp = test_client.post("/api/ask", json={"query": "规则是什么？"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data

    def test_query_simple(self, test_client):
        resp = test_client.post("/api/query/simple", json={
            "query": "怎么获胜？"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data


class TestSessionEndpoints:
    def test_clear_session(self, test_client):
        resp = test_client.post("/api/session/clear?session_id=test-123")
        assert resp.status_code == 200


class TestDocumentEndpoints:
    def test_list_documents(self, test_client):
        resp = test_client.get("/api/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert "documents" in data
