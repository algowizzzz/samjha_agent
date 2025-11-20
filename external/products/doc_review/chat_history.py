"""
Chat history management for document review.
Handles storage and retrieval of chat messages between user and RiskGPT.
"""
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
import json

from .store import DocReviewStore


def _timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


class ChatHistoryManager:
    """Manages chat history for document review."""
    
    def __init__(self, store: DocReviewStore):
        self.store = store
    
    def _get_messages_from_state(self, file_id: str) -> List[Dict[str, Any]]:
        """Get chat messages array from document state."""
        record = self.store.load(file_id)
        if not record:
            return []
        
        state = record.get("state", {})
        return state.get("chat_messages", [])
    
    def _save_messages_to_state(self, file_id: str, messages: List[Dict[str, Any]]) -> bool:
        """Save chat messages array to document state."""
        record = self.store.load(file_id)
        if not record:
            return False
        
        state = record.get("state", {})
        state["chat_messages"] = messages
        record["state"] = state
        record["updated_at"] = _timestamp()
        
        path = self.store._state_path(file_id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        
        return True
    
    def list_messages(self, file_id: str) -> List[Dict[str, Any]]:
        """
        List all chat messages for a document.
        
        Args:
            file_id: Document ID
            
        Returns:
            List of message dictionaries
        """
        return self._get_messages_from_state(file_id)
    
    def add_message(
        self,
        file_id: str,
        role: str,  # 'user' or 'assistant'
        content: str,
        context: str = None  # Optional selected text context
    ) -> Dict[str, Any]:
        """
        Add a new chat message.
        
        Args:
            file_id: Document ID
            role: Message role ('user' or 'assistant')
            content: Message content
            context: Optional selected text context
            
        Returns:
            The created message dictionary
        """
        messages = self._get_messages_from_state(file_id)
        
        message = {
            "id": f"msg{len(messages) + 1}_{int(datetime.utcnow().timestamp() * 1000)}",
            "role": role,
            "content": content,
            "timestamp": _timestamp(),
        }
        
        if context:
            message["context"] = context
        
        messages.append(message)
        self._save_messages_to_state(file_id, messages)
        
        return message
    
    def clear_messages(self, file_id: str) -> bool:
        """
        Clear all chat messages for a document.
        
        Args:
            file_id: Document ID
            
        Returns:
            True if successful
        """
        return self._save_messages_to_state(file_id, [])


