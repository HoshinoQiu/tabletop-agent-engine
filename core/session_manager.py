"""
Session Manager: Manages multi-turn conversation sessions.
"""

import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from config.settings import settings


from loguru import logger


class Session:
    """Represents a single conversation session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        self.history: List[Dict[str, str]] = []

    def add_turn(self, role: str, content: str):
        """Add a conversation turn."""
        self.history.append({"role": role, "content": content})
        self.last_active = datetime.now()
        # Trim to max turns
        max_entries = settings.MAX_HISTORY_TURNS * 2  # Each turn = user + assistant
        if len(self.history) > max_entries:
            self.history = self.history[-max_entries:]

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return list(self.history)

    def is_expired(self) -> bool:
        """Check if session has expired."""
        ttl = timedelta(seconds=settings.SESSION_TTL_SECONDS)
        return datetime.now() - self.last_active > ttl

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "turns": len(self.history) // 2,
        }


class SessionManager:
    """Thread-safe session manager with TTL-based expiration."""

    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self.lock = threading.Lock()
        logger.info("SessionManager initialized")

    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        with self.lock:
            self._cleanup_expired()
            self.sessions[session_id] = Session(session_id)
        logger.info(f"Created session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID, creating one if it doesn't exist."""
        with self.lock:
            self._cleanup_expired()
            if session_id not in self.sessions:
                self.sessions[session_id] = Session(session_id)
                logger.info(f"Auto-created session: {session_id}")
            session = self.sessions[session_id]
            if session.is_expired():
                del self.sessions[session_id]
                self.sessions[session_id] = Session(session_id)
                logger.info(f"Session expired, recreated: {session_id}")
                session = self.sessions[session_id]
            session.last_active = datetime.now()
            return session

    def clear_session(self, session_id: str) -> bool:
        """Clear a specific session."""
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Cleared session: {session_id}")
                return True
            return False

    def _cleanup_expired(self):
        """Remove expired sessions (called within lock)."""
        expired = [sid for sid, s in self.sessions.items() if s.is_expired()]
        for sid in expired:
            del self.sessions[sid]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

    def active_count(self) -> int:
        """Return count of active sessions."""
        with self.lock:
            self._cleanup_expired()
            return len(self.sessions)
