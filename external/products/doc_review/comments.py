"""
Comments management for document review.
Handles CRUD operations for block-level comments and replies.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from .store import DocReviewStore


def _timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


class CommentsManager:
    """Manages comments for document review blocks."""
    
    def __init__(self, store: DocReviewStore):
        self.store = store
    
    def _get_comments_from_state(self, file_id: str) -> List[Dict[str, Any]]:
        """Get comments array from document state."""
        record = self.store.load(file_id)
        if not record:
            return []
        
        state = record.get("state", {})
        return state.get("comments", [])
    
    def _save_comments_to_state(self, file_id: str, comments: List[Dict[str, Any]]) -> bool:
        """Save comments array to document state."""
        record = self.store.load(file_id)
        if not record:
            return False
        
        state = record.get("state", {})
        state["comments"] = comments
        record["state"] = state
        record["updated_at"] = _timestamp()
        
        path = self.store._state_path(file_id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        
        return True
    
    def list_comments(self, file_id: str, block_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all comments for a document, optionally filtered by block_id.
        
        Args:
            file_id: Document ID
            block_id: Optional block ID to filter comments
            
        Returns:
            List of comment dictionaries
        """
        comments = self._get_comments_from_state(file_id)
        
        if block_id:
            comments = [c for c in comments if c.get("block_id") == block_id]
        
        return comments
    
    def add_comment(
        self,
        file_id: str,
        block_id: str,
        block_title: str,
        content: str,
        author: str = "User",
        selection_text: Optional[str] = None,
        start_offset: Optional[int] = None,
        end_offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Add a new comment to a block.
        
        Args:
            file_id: Document ID
            block_id: Block ID to comment on
            block_title: Title/preview of the block
            content: Comment content
            author: Comment author name
            selection_text: Optional selected text that comment refers to
            start_offset: Optional character offset where selection starts in block
            end_offset: Optional character offset where selection ends in block
            
        Returns:
            The created comment dictionary
        """
        comments = self._get_comments_from_state(file_id)
        
        comment = {
            "id": f"c{len(comments) + 1}_{int(datetime.utcnow().timestamp() * 1000)}",
            "block_id": block_id,
            "block_title": block_title,
            "author": author,
            "timestamp": _timestamp(),
            "content": content,
            "resolved": False,
            "replies": [],
        }
        
        if selection_text:
            comment["selection_text"] = selection_text
        
        if start_offset is not None:
            comment["start_offset"] = start_offset
        
        if end_offset is not None:
            comment["end_offset"] = end_offset
        
        comments.append(comment)
        self._save_comments_to_state(file_id, comments)
        
        return comment
    
    def add_reply(
        self,
        file_id: str,
        comment_id: str,
        content: str,
        author: str = "User",
    ) -> Optional[Dict[str, Any]]:
        """
        Add a reply to an existing comment.
        
        Args:
            file_id: Document ID
            comment_id: Comment ID to reply to
            content: Reply content
            author: Reply author name
            
        Returns:
            The updated comment with the new reply, or None if comment not found
        """
        comments = self._get_comments_from_state(file_id)
        
        for comment in comments:
            if comment["id"] == comment_id:
                reply = {
                    "id": f"r{len(comment['replies']) + 1}_{int(datetime.utcnow().timestamp() * 1000)}",
                    "author": author,
                    "timestamp": _timestamp(),
                    "content": content,
                }
                comment["replies"].append(reply)
                self._save_comments_to_state(file_id, comments)
                return comment
        
        return None
    
    def resolve_comment(self, file_id: str, comment_id: str) -> Optional[Dict[str, Any]]:
        """
        Mark a comment as resolved or unresolved (toggle).
        
        Args:
            file_id: Document ID
            comment_id: Comment ID to resolve
            
        Returns:
            The updated comment, or None if not found
        """
        comments = self._get_comments_from_state(file_id)
        
        for comment in comments:
            if comment["id"] == comment_id:
                comment["resolved"] = not comment.get("resolved", False)
                self._save_comments_to_state(file_id, comments)
                return comment
        
        return None
    
    def delete_comment(self, file_id: str, comment_id: str) -> bool:
        """
        Delete a comment and all its replies.
        
        Args:
            file_id: Document ID
            comment_id: Comment ID to delete
            
        Returns:
            True if comment was deleted, False if not found
        """
        comments = self._get_comments_from_state(file_id)
        original_count = len(comments)
        
        comments = [c for c in comments if c["id"] != comment_id]
        
        if len(comments) < original_count:
            self._save_comments_to_state(file_id, comments)
            return True
        
        return False
    
    def update_comment(
        self,
        file_id: str,
        comment_id: str,
        content: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update a comment's content.
        
        Args:
            file_id: Document ID
            comment_id: Comment ID to update
            content: New content (if None, no update)
            
        Returns:
            The updated comment, or None if not found
        """
        if content is None:
            return None
        
        comments = self._get_comments_from_state(file_id)
        
        for comment in comments:
            if comment["id"] == comment_id:
                comment["content"] = content
                comment["updated_at"] = _timestamp()
                self._save_comments_to_state(file_id, comments)
                return comment
        
        return None
    
    def get_comment_count_by_block(self, file_id: str) -> Dict[str, int]:
        """
        Get comment count for each block.
        
        Args:
            file_id: Document ID
            
        Returns:
            Dictionary mapping block_id to unresolved comment count
        """
        comments = self._get_comments_from_state(file_id)
        count_map: Dict[str, int] = {}
        
        for comment in comments:
            if not comment.get("resolved", False):
                block_id = comment.get("block_id", "")
                count_map[block_id] = count_map.get(block_id, 0) + 1
        
        return count_map

