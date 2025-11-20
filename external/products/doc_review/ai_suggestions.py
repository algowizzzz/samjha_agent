"""
AI Suggestions management for document review.
Handles CRUD operations for AI-suggested text improvements.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from .store import DocReviewStore


def _timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


class AISuggestionsManager:
    """Manages AI suggestions for document review."""
    
    def __init__(self, store: DocReviewStore):
        self.store = store
    
    def _get_suggestions_from_state(self, file_id: str) -> List[Dict[str, Any]]:
        """Get AI suggestions array from document state."""
        record = self.store.load(file_id)
        if not record:
            return []
        
        state = record.get("state", {})
        return state.get("ai_suggestions", [])
    
    def _save_suggestions_to_state(self, file_id: str, suggestions: List[Dict[str, Any]]) -> bool:
        """Save AI suggestions array to document state."""
        record = self.store.load(file_id)
        if not record:
            return False
        
        state = record.get("state", {})
        state["ai_suggestions"] = suggestions
        record["state"] = state
        record["updated_at"] = _timestamp()
        
        path = self.store._state_path(file_id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        
        return True
    
    def list_suggestions(self, file_id: str, block_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all AI suggestions for a document, optionally filtered by block_id.
        
        Args:
            file_id: Document ID
            block_id: Optional block ID to filter suggestions
            
        Returns:
            List of suggestion dictionaries
        """
        suggestions = self._get_suggestions_from_state(file_id)
        
        if block_id:
            suggestions = [s for s in suggestions if s.get("block_id") == block_id]
        
        return suggestions
    
    def add_suggestion(
        self,
        file_id: str,
        block_id: str,
        selection_text: str,
        improved_text: str,
        status: str = "pending",  # pending, accepted, rejected
        start_offset: Optional[int] = None,
        end_offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Add a new AI suggestion.
        
        Args:
            file_id: Document ID
            block_id: Block ID where suggestion was made
            selection_text: Original selected text
            improved_text: AI-suggested improved text
            status: Suggestion status (pending, accepted, rejected)
            start_offset: Optional character offset where selection starts in block
            end_offset: Optional character offset where selection ends in block
            
        Returns:
            The created suggestion dictionary
        """
        suggestions = self._get_suggestions_from_state(file_id)
        
        suggestion = {
            "id": f"ai{len(suggestions) + 1}_{int(datetime.utcnow().timestamp() * 1000)}",
            "block_id": block_id,
            "selection_text": selection_text,
            "improved_text": improved_text,
            "status": status,
            "timestamp": _timestamp(),
        }
        
        if start_offset is not None:
            suggestion["start_offset"] = start_offset
        
        if end_offset is not None:
            suggestion["end_offset"] = end_offset
        
        suggestions.append(suggestion)
        self._save_suggestions_to_state(file_id, suggestions)
        
        return suggestion
    
    def update_status(
        self,
        file_id: str,
        suggestion_id: str,
        status: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Update the status of an AI suggestion (pending -> accepted/rejected).
        
        Args:
            file_id: Document ID
            suggestion_id: Suggestion ID
            status: New status (pending, accepted, rejected)
            
        Returns:
            Updated suggestion dictionary or None if not found
        """
        suggestions = self._get_suggestions_from_state(file_id)
        
        for suggestion in suggestions:
            if suggestion["id"] == suggestion_id:
                suggestion["status"] = status
                suggestion["updated_at"] = _timestamp()
                self._save_suggestions_to_state(file_id, suggestions)
                return suggestion
        
        return None
    
    def delete_suggestion(self, file_id: str, suggestion_id: str) -> bool:
        """
        Delete an AI suggestion.
        
        Args:
            file_id: Document ID
            suggestion_id: Suggestion ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        suggestions = self._get_suggestions_from_state(file_id)
        original_count = len(suggestions)
        
        suggestions = [s for s in suggestions if s["id"] != suggestion_id]
        
        if len(suggestions) < original_count:
            self._save_suggestions_to_state(file_id, suggestions)
            return True
        
        return False
    
    def get_suggestion(self, file_id: str, suggestion_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific AI suggestion by ID.
        
        Args:
            file_id: Document ID
            suggestion_id: Suggestion ID
            
        Returns:
            Suggestion dictionary or None if not found
        """
        suggestions = self._get_suggestions_from_state(file_id)
        
        for suggestion in suggestions:
            if suggestion["id"] == suggestion_id:
                return suggestion
        
        return None

