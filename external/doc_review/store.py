from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from external.doc_review.types import AgentState
from external.tools.doc_processing.utils import (
    ensure_directory,
    generate_md_file_id,
    write_text_file,
)

logger = logging.getLogger(__name__)

_STATE_DIR = Path("external/data/doc_review/state")
_STATE_DIR.mkdir(parents=True, exist_ok=True)
_IMPROVED_DIR = Path("external/data/doc_review/improved")
_IMPROVED_DIR.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


class DocReviewStore:
    """File-based persistence for document review runs."""

    def __init__(self, base_dir: Path = _STATE_DIR):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _state_path(self, file_id: str) -> Path:
        return self.base_dir / f"{file_id}.json"

    def list_documents(self) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []
        for path in sorted(self.base_dir.glob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                metadata = {
                    "file_id": data.get("file_id"),
                    "source_path": data.get("source_path"),
                    "status": data.get("status", "unknown"),
                    "updated_at": data.get("updated_at"),
                    "file_metadata": data.get("state", {}).get("file_metadata"),
                    "state": data.get("state", {}),  # Include full state for Phase 2 filtering
                }
                documents.append(metadata)
            except Exception as exc:
                logger.error("Failed to read doc review state %s: %s", path, exc)
        return documents

    def load(self, file_id: str) -> Optional[Dict[str, Any]]:
        path = self._state_path(file_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, file_id: str, source_path: str, state: AgentState, status: str) -> Dict[str, Any]:
        payload = {
            "file_id": file_id,
            "source_path": source_path,
            "status": status,
            "updated_at": _timestamp(),
            "state": state,
        }
        path = self._state_path(file_id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload

    def update_status(self, file_id: str, status: str) -> None:
        entry = self.load(file_id)
        if not entry:
            return
        entry["status"] = status
        entry["updated_at"] = _timestamp()
        path = self._state_path(file_id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)

    def delete(self, file_id: str) -> bool:
        path = self._state_path(file_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def update_markdown(
        self, 
        file_id: str, 
        markdown: str, 
        toc_markdown: Optional[str] = None,
        block_metadata: Optional[list] = None,
        accepted_suggestions: Optional[list] = None,
        rejected_suggestions: Optional[list] = None
    ) -> Optional[Dict[str, Any]]:
        record = self.load(file_id)
        if not record:
            return None

        state: Dict[str, Any] = record.get("state") or {}
        if not isinstance(state, dict):
            state = {}

        improved_id = state.get("improved_md_file_id")
        if not improved_id:
            improved_id = generate_md_file_id(f"{file_id}_manual")
            state["improved_md_file_id"] = improved_id

        improved_path = _IMPROVED_DIR / f"{improved_id}.md"
        ensure_directory(improved_path.parent)
        write_text_file(improved_path, markdown)

        state["improved_markdown"] = markdown
        state["improved_md_path"] = str(improved_path)
        if toc_markdown is not None:
            state["toc_markdown"] = toc_markdown
        
        # Save block metadata and suggestion states
        if block_metadata is not None:
            state["block_metadata"] = block_metadata
        if accepted_suggestions is not None:
            state["accepted_suggestions"] = accepted_suggestions
        if rejected_suggestions is not None:
            state["rejected_suggestions"] = rejected_suggestions

        record["state"] = state
        record["status"] = record.get("status", "completed")
        record["updated_at"] = _timestamp()

        dest = self._state_path(file_id)
        with dest.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        return record

    def append_chat_message(
        self,
        file_id: str,
        message: str,
        response: str,
        selection: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        record = self.load(file_id)
        if not record:
            return None

        state: Dict[str, Any] = record.get("state") or {}
        if not isinstance(state, dict):
            state = {}

        chat_entry = {
            "timestamp": _timestamp(),
            "message": message,
            "response": response,
        }
        if selection:
            chat_entry["selection"] = selection

        history: List[Dict[str, Any]] = state.setdefault("chat_history", [])
        history.append(chat_entry)
        # keep history bounded
        if len(history) > 20:
            del history[:-20]

        record["state"] = state
        record["updated_at"] = _timestamp()

        dest = self._state_path(file_id)
        with dest.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        return record
