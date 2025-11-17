from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from external.doc_review.types import AgentState


def _slugify_section(title: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in title).strip("_") or "section"


class DocReviewVFSAdapter:
    """Virtual file system facade for AgentState."""

    def __init__(self, state: Optional[AgentState]):
        self.state = state or {}
        self.structure = self.state.setdefault("structure", {})
        self.phase1 = self.state.setdefault("phase1", {})
        self.phase2 = self.state.setdefault("phase2", {"chunks": {}, "reviews": {}})
        self.changes = self.state.setdefault("changes", {})

    # ------------------------------------------------------------------ #
    # Directory helpers
    # ------------------------------------------------------------------ #
    def list_dir(self, path: str) -> List[Dict[str, str]]:
        path = self._normalize(path)
        if path == "/":
            return self._root_entries()
        if path == "/original":
            return self._directory_entries(
                [("document.md", self.structure.get("raw_text"))], file_extension="md"
            )
        if path == "/phase1":
            files = [
                ("doc_summary.json", self.phase1.get("doc_summary")),
                ("toc_review.json", self.phase1.get("toc_review")),
                ("template_fitness.json", self.phase1.get("template_fitness_report")),
                ("section_strategy.json", self.phase1.get("section_strategy")),
            ]
            return self._directory_entries(files, file_extension="json")
        if path == "/phase2":
            entries: List[Dict[str, str]] = []
            if self.phase2.get("chunks"):
                entries.append({"name": "sections", "type": "directory"})
            if self.phase2.get("reviews"):
                entries.append({"name": "reviews", "type": "directory"})
            if self.phase2.get("summary_report"):
                entries.append({"name": "summary_report.json", "type": "file"})
            return entries
        if path == "/phase2/sections":
            return self._list_section_chunks(kind="sections")
        if path == "/phase2/reviews":
            return self._list_section_chunks(kind="reviews")
        if path == "/changes":
            files = [
                ("suggested_changes.json", self.changes.get("suggested_changes")),
                ("applied_changes.json", self.changes.get("applied_change_ids")),
            ]
            return self._directory_entries(files, file_extension="json")
        if path == "/versions":
            entries = [
                ("current.md", self.structure.get("raw_text")),
            ]
            if self.changes.get("_pre_apply_text"):
                entries.append(("previous.md", self.changes.get("_pre_apply_text")))
            return self._directory_entries(entries, file_extension="md")
        raise FileNotFoundError(path)

    def stat(self, path: str) -> Dict[str, Optional[int]]:
        path = self._normalize(path)
        if path.endswith("/"):
            path = path.rstrip("/")
        if path in {"/", "/original", "/phase1", "/phase2", "/phase2/sections", "/phase2/reviews", "/changes", "/versions"}:
            return {"path": path or "/", "type": "directory", "size": 0}
        content = self.read_file(path)
        return {"path": path, "type": "file", "size": len(content)}

    def read_file(self, path: str) -> str:
        path = self._normalize(path)
        if path == "/original/document.md":
            return self.structure.get("raw_text", "") or ""
        if path == "/phase1/doc_summary.json":
            return self._to_json(self.phase1.get("doc_summary"))
        if path == "/phase1/toc_review.json":
            return self._to_json(self.phase1.get("toc_review"))
        if path == "/phase1/template_fitness.json":
            return self._to_json(self.phase1.get("template_fitness_report"))
        if path == "/phase1/section_strategy.json":
            return self._to_json(self.phase1.get("section_strategy"))
        if path == "/phase2/summary_report.json":
            return self._to_json(self.phase2.get("summary_report"))
        if path.startswith("/phase2/reviews/"):
            section = self._resolve_section_from_path(path, suffix=".json")
            reviews = self.phase2.get("reviews") or {}
            return self._to_json(reviews.get(section))
        if path.startswith("/phase2/sections/"):
            section = self._resolve_section_from_path(path, suffix=".md")
            chunks = self.phase2.get("chunks") or {}
            chunk = chunks.get(section, {})
            return chunk.get("text", "") or ""
        if path == "/changes/suggested_changes.json":
            return self._to_json(self.changes.get("suggested_changes"))
        if path == "/changes/applied_changes.json":
            applied = {
                "applied_change_ids": self.changes.get("applied_change_ids", []),
                "failed_changes": self.changes.get("failed_changes", []),
            }
            return self._to_json(applied)
        if path == "/versions/current.md":
            return self.structure.get("raw_text", "") or ""
        if path == "/versions/previous.md":
            previous = self.changes.get("_pre_apply_text")
            if not previous:
                raise FileNotFoundError(path)
            return previous
        raise FileNotFoundError(path)

    def write_file(self, path: str, data: str) -> None:
        path = self._normalize(path)
        if path == "/original/document.md":
            self.structure["raw_text"] = data
            return
        if path == "/changes/suggested_changes.json":
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError as exc:
                raise ValueError("suggested_changes must be valid JSON") from exc
            if not isinstance(parsed, list):
                raise ValueError("suggested_changes must be a list")
            self.changes["suggested_changes"] = parsed
            return
        raise PermissionError(f"Path '{path}' is read-only")

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    def _normalize(self, path: Optional[str]) -> str:
        if not path:
            return "/"
        if not path.startswith("/"):
            path = "/" + path
        return path.rstrip() or "/"

    def _root_entries(self) -> List[Dict[str, str]]:
        entries = []
        if self.structure.get("raw_text"):
            entries.append({"name": "original", "type": "directory"})
        if any(self.phase1.get(key) for key in ("doc_summary", "toc_review", "template_fitness_report", "section_strategy")):
            entries.append({"name": "phase1", "type": "directory"})
        if self.phase2.get("chunks") or self.phase2.get("reviews") or self.phase2.get("summary_report"):
            entries.append({"name": "phase2", "type": "directory"})
        entries.append({"name": "changes", "type": "directory"})
        entries.append({"name": "versions", "type": "directory"})
        return entries

    def _directory_entries(self, files: List[Tuple[str, Optional[object]]], file_extension: str) -> List[Dict[str, str]]:
        entries = []
        for name, data in files:
            if data is None:
                continue
            entries.append({"name": name, "type": "file"})
        return entries

    def _list_section_chunks(self, kind: str) -> List[Dict[str, str]]:
        chunks = self.phase2.get("chunks") or {}
        if not chunks:
            return []
        entries = []
        for title in sorted(chunks.keys()):
            slug = _slugify_section(title)
            suffix = ".md" if kind == "sections" else ".json"
            entries.append({"name": f"{slug}{suffix}", "type": "file"})
        return entries

    def _resolve_section_from_path(self, path: str, suffix: str) -> str:
        parts = path.split("/")
        if len(parts) < 4:
            raise FileNotFoundError(path)
        filename = parts[-1]
        if not filename.endswith(suffix):
            raise FileNotFoundError(path)
        slug = filename[: -len(suffix)]
        chunks = self.phase2.get("chunks") or {}
        for title in chunks.keys():
            if _slugify_section(title) == slug:
                return title
        raise FileNotFoundError(path)

    def _to_json(self, data: object) -> str:
        return json.dumps(data if data is not None else {}, indent=2, ensure_ascii=False)


