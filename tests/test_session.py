"""Tests for session management."""

import time
from core.session_manager import SessionManager, Session


class TestSession:
    def test_create_session(self):
        session = Session("test-id")
        assert session.session_id == "test-id"
        assert len(session.history) == 0

    def test_add_turn(self):
        session = Session("test-id")
        session.add_turn("user", "Hello")
        session.add_turn("assistant", "Hi there")
        assert len(session.history) == 2
        assert session.history[0]["role"] == "user"

    def test_history_trimming(self):
        session = Session("test-id")
        for i in range(30):
            session.add_turn("user", f"Message {i}")
            session.add_turn("assistant", f"Reply {i}")
        # Should be trimmed to MAX_HISTORY_TURNS * 2
        from config.settings import settings
        assert len(session.history) <= settings.MAX_HISTORY_TURNS * 2

    def test_to_dict(self):
        session = Session("test-id")
        d = session.to_dict()
        assert d["session_id"] == "test-id"
        assert "created_at" in d


class TestSessionManager:
    def test_create_and_get(self):
        mgr = SessionManager()
        sid = mgr.create_session()
        session = mgr.get_session(sid)
        assert session is not None
        assert session.session_id == sid

    def test_auto_create(self):
        mgr = SessionManager()
        session = mgr.get_session("new-session")
        assert session is not None

    def test_clear_session(self):
        mgr = SessionManager()
        sid = mgr.create_session()
        assert mgr.clear_session(sid) is True
        assert mgr.clear_session("nonexistent") is False

    def test_active_count(self):
        mgr = SessionManager()
        mgr.create_session()
        mgr.create_session()
        assert mgr.active_count() == 2
