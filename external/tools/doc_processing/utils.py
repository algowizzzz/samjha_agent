from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional

_WORDS_PER_PAGE = 450


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text_file(path: Path, content: str) -> None:
    ensure_directory(path.parent)
    path.write_text(content, encoding="utf-8")


def generate_md_file_id(file_id: str) -> str:
    return f"{file_id}_md_{uuid.uuid4().hex[:8]}"


def generate_chunk_id(file_id: str, index: int) -> str:
    return f"{file_id}_chunk_{index:04d}"


def estimate_page_count(word_count: int) -> int:
    if word_count <= 0:
        return 1
    return max(1, (word_count + _WORDS_PER_PAGE - 1) // _WORDS_PER_PAGE)


_heading_pattern = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.+?)\s*$")


def parse_markdown_headings(markdown: str) -> List[Dict[str, object]]:
    headings: List[Dict[str, object]] = []
    path_stack: List[Dict[str, object]] = []

    for line_no, line in enumerate(markdown.splitlines(), start=1):
        match = _heading_pattern.match(line.strip())
        if not match:
            continue

        level = len(match.group("hashes"))
        text = match.group("text").strip()
        heading_entry = {
            "level": level,
            "heading_level": f"h{level}",
            "heading_text": text,
            "line_number": line_no,
        }

        # Maintain hierarchical path
        while path_stack and path_stack[-1]["level"] >= level:
            path_stack.pop()
        heading_entry["heading_path"] = [p["heading_text"] for p in path_stack] + [text]
        path_stack.append({"level": level, "heading_text": text})

        headings.append(heading_entry)
    return headings


def headings_to_levels(headings: Iterable[Dict[str, object]]) -> List[str]:
    levels = sorted({h.get("heading_level") for h in headings if h.get("heading_level")})
    level_order = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
    return sorted(levels, key=lambda lvl: level_order.get(lvl, 10))


def normalize_heading_level(level: str) -> str:
    level = level.lower()
    if level.startswith("h") and level[1:].isdigit():
        return level
    raise ValueError(f"Unsupported heading level: {level}")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or uuid.uuid4().hex[:8]


def build_heading_anchor(heading_path: List[str]) -> str:
    combined = "-".join(slugify(part) for part in heading_path)
    digest = hashlib.sha1(combined.encode("utf-8")).hexdigest()[:10]
    return f"sec-{digest}"


def derive_template_path(template_id: str) -> Path:
    # Primary location: config/doc_review/outline_templates/
    base_dir = Path("config/doc_review/outline_templates")
    if base_dir.exists():
        template_path = base_dir / f"{template_id}.json"
        if template_path.exists():
            return template_path
    # Fallback to legacy location: external/doc_review/templates/
    base_dir = Path("external/doc_review/templates")
    return base_dir / f"{template_id}.json"


def resolve_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def read_json(path: Path) -> Dict[str, object]:
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
