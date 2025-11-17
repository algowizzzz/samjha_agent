"""
Session Manager for Agent Queries
Tracks active sessions and handles cancellation
"""
import threading
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentSessionManager:
    """Manages active agent sessions and cancellation"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.active_sessions: Dict[str, Dict] = {}  # session_id -> {thread, cancelled, start_time, user_id}
        self._session_lock = threading.Lock()
    
    def register_session(self, session_id: str, thread: threading.Thread, user_id: Optional[str] = None) -> None:
        """Register an active session"""
        with self._session_lock:
            self.active_sessions[session_id] = {
                'thread': thread,
                'cancelled': False,
                'start_time': datetime.utcnow(),
                'user_id': user_id
            }
            logger.info(f"[SessionManager] Registered session {session_id}")
    
    def cancel_session(self, session_id: str) -> bool:
        """Mark session as cancelled"""
        with self._session_lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['cancelled'] = True
                logger.info(f"[SessionManager] Cancelled session {session_id}")
                return True
            logger.warning(f"[SessionManager] Session {session_id} not found for cancellation")
            return False
    
    def is_cancelled(self, session_id: str) -> bool:
        """Check if session is cancelled"""
        with self._session_lock:
            return self.active_sessions.get(session_id, {}).get('cancelled', False)
    
    def unregister_session(self, session_id: str) -> None:
        """Remove session from active list"""
        with self._session_lock:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                logger.info(f"[SessionManager] Unregistered session {session_id}")
    
    def get_active_sessions(self) -> Dict[str, Dict]:
        """Get all active sessions (for debugging)"""
        with self._session_lock:
            return dict(self.active_sessions)

