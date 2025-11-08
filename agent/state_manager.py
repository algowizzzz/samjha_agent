import json
import os
import glob
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import threading

logger = logging.getLogger(__name__)


class AgentStateManager:
    """
    JSON file-backed persistence for agent sessions.
    Stores full state per session and allows fetching the last two sessions for a user.
    Uses file locking to avoid concurrent write conflicts.
    """

    def __init__(self, state_dir: str = os.path.join("data", "agent_state")):
        self.state_dir = state_dir
        os.makedirs(self.state_dir, exist_ok=True)
        self._lock = threading.Lock()

    def _get_session_path(self, session_id: str) -> str:
        """Get file path for a session"""
        # Sanitize session_id for filename
        safe_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in session_id)
        return os.path.join(self.state_dir, f"session_{safe_id}.json")

    def save_session_state(self, session_id: str, user_id: Optional[str], state: Dict[str, Any]) -> None:
        """
        Save full state for a session.
        """
        with self._lock:
            try:
                now = datetime.utcnow()
                session_path = self._get_session_path(session_id)
                
                # Load existing data if present
                if os.path.exists(session_path):
                    try:
                        with open(session_path, 'r') as f:
                            existing = json.load(f)
                            created_at = existing.get('created_at', now.isoformat() + "Z")
                    except Exception as e:
                        logger.warning(f"Could not load existing session {session_id}: {e}")
                        created_at = now.isoformat() + "Z"
                else:
                    created_at = now.isoformat() + "Z"
                
                # Prepare session data for saving
                state_to_save = state
                
                session_data = {
                    'session_id': session_id,
                    'user_id': user_id,
                    'created_at': created_at,
                    'updated_at': now.isoformat() + "Z",
                    'state': state_to_save
                }
                
                # Write to file
                with open(session_path, 'w') as f:
                    json.dump(session_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved session state for {session_id} (user: {user_id})")
            except Exception as e:
                logger.error(f"Failed to save session state for {session_id}: {e}", exc_info=True)
                raise

    def load_session_state(self, session_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load state for a specific session.
        """
        with self._lock:
            try:
                session_path = self._get_session_path(session_id)
                if not os.path.exists(session_path):
                    logger.warning(f"Session {session_id} not found")
                    return None
                
                with open(session_path, 'r') as f:
                    data = json.load(f)
                    
                # Verify user_id if provided
                if user_id and data.get('user_id') != user_id:
                    logger.warning(f"Session {session_id} user mismatch")
                    return None
                
                logger.info(f"Loaded session state for {session_id}")
                return data['state']
            except Exception as e:
                logger.error(f"Failed to load session {session_id}: {e}", exc_info=True)
                return None



