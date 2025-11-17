from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from external.model_doc.types import AgentState

logger = logging.getLogger(__name__)

_STATE_DIR = Path("external/data/model_doc/state")
_STATE_DIR.mkdir(parents=True, exist_ok=True)
_OUTPUT_DIR = Path("external/data/model_doc/output")
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


class ModelDocStore:
    """File-based persistence for model documentation runs."""

    def __init__(self, base_dir: Path = _STATE_DIR):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _state_path(self, codebase_id: str) -> Path:
        return self.base_dir / f"{codebase_id}.json"

    def list_codebases(self) -> List[Dict[str, Any]]:
        """List all registered codebases."""
        codebases: List[Dict[str, Any]] = []
        for path in sorted(self.base_dir.glob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                metadata = {
                    "codebase_id": data.get("codebase_id"),
                    "codebase_path": data.get("codebase_path"),
                    "status": data.get("status", "unknown"),
                    "updated_at": data.get("updated_at"),
                    "codebase_metadata": data.get("state", {}).get("codebase_metadata"),
                    "state": data.get("state", {}),
                }
                codebases.append(metadata)
            except Exception as exc:
                logger.error("Failed to read model doc state %s: %s", path, exc)
        return codebases

    def load(self, codebase_id: str) -> Optional[Dict[str, Any]]:
        """Load a codebase record."""
        path = self._state_path(codebase_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def save(
        self, codebase_id: str, codebase_path: str, state: AgentState, status: str
    ) -> Dict[str, Any]:
        """Save or update a codebase record."""
        payload = {
            "codebase_id": codebase_id,
            "codebase_path": codebase_path,
            "status": status,
            "updated_at": _timestamp(),
            "state": state,
        }
        path = self._state_path(codebase_id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload

    def update_status(self, codebase_id: str, status: str) -> None:
        """Update the status of a codebase."""
        entry = self.load(codebase_id)
        if not entry:
            return
        entry["status"] = status
        entry["updated_at"] = _timestamp()
        path = self._state_path(codebase_id)
        with path.open("w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)

    def delete(self, codebase_id: str) -> bool:
        """Delete a codebase record."""
        path = self._state_path(codebase_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def update_documentation(
        self, codebase_id: str, documentation: str, output_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Update the final documentation in a codebase record."""
        record = self.load(codebase_id)
        if not record:
            return None

        state: Dict[str, Any] = record.get("state") or {}
        if not isinstance(state, dict):
            state = {}

        state["final_documentation"] = documentation
        if output_path:
            state["final_documentation_path"] = output_path

        record["state"] = state
        record["status"] = record.get("status", "completed")
        record["updated_at"] = _timestamp()

        dest = self._state_path(codebase_id)
        with dest.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        return record

    def append_chat_message(
        self,
        codebase_id: str,
        message: str,
        response: str,
        selection: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Append a chat message to the codebase history."""
        record = self.load(codebase_id)
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
        # Keep history bounded
        if len(history) > 20:
            del history[:-20]

        record["state"] = state
        record["updated_at"] = _timestamp()

        dest = self._state_path(codebase_id)
        with dest.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        return record

