"""
Agent Registry

Stores and manages configurable agent *instances* (same core capability, different domain + data).

All persisted state lives under:
- external/config/agents/
- external/config/domains/
- external/datawarehouse/
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


AGENTS_DIR = Path("external/config/agents")
DOMAINS_DIR = Path("external/config/domains")
DATAWAREHOUSE_DIR = Path("external/datawarehouse")

AGENTS_INDEX_FILE = AGENTS_DIR / "agents.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True))


def slugify_name(name: str) -> str:
    """
    Create a stable agent_id from a display name.
    - lowercases
    - converts non-alnum to underscores
    - collapses repeats
    - trims underscores
    """
    base = (name or "").strip().lower()
    base = re.sub(r"[^a-z0-9]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    return base or "agent"


def validate_safe_name(value: str, field: str) -> None:
    """
    Prevent path traversal / separators; allow alnum, underscore, dash.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    v = value.strip()
    if ".." in v or "/" in v or "\\" in v:
        raise ValueError(f"Invalid {field}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", v):
        raise ValueError(f"Invalid {field} (allowed: letters, numbers, '_' '-')")


def validate_agent_type(agent_type: str) -> None:
    if agent_type not in ("structured", "unstructured", "external", "excel", "deep_research", "quick_search"):
        raise ValueError("Invalid agent_type")


def ensure_default_agent() -> None:
    """
    Ensure there's at least one structured agent configured for the existing ecomm setup.
    """
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    DOMAINS_DIR.mkdir(parents=True, exist_ok=True)
    DATAWAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)

    ecommerce_id = "ecommerce_agent"
    ecommerce_cfg_file = AGENTS_DIR / f"{ecommerce_id}.json"

    # Only create if missing
    if not ecommerce_cfg_file.exists():
        # Keep existing domain file name if present; otherwise create a copy under the new naming.
        existing_domain = DOMAINS_DIR / "ecomm_domain.md"
        domain_file = f"{ecommerce_id}_domain.md"
        target_domain = DOMAINS_DIR / domain_file
        if existing_domain.exists() and not target_domain.exists():
            _atomic_write_text(target_domain, existing_domain.read_text())

        cfg = {
            "id": ecommerce_id,
            "name": "Ecommerce Agent",
            "description": "Analyze sales and customer data (structured parquet/csv via DuckDB).",
            "agent_type": "structured",
            "domain_file": domain_file if target_domain.exists() else "ecomm_domain.md",
            "data_folder": "ECommerce",
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
        }
        _atomic_write_json(ecommerce_cfg_file, cfg)

    # Ensure index exists
    if not AGENTS_INDEX_FILE.exists():
        index = {"agents": []}
        _atomic_write_json(AGENTS_INDEX_FILE, index)

    # Ensure ecommerce agent is in index (read directly to avoid recursion)
    try:
        if AGENTS_INDEX_FILE.exists():
            idx = json.loads(AGENTS_INDEX_FILE.read_text())
        else:
            idx = {"agents": []}
    except Exception:
        idx = {"agents": []}
    
    agents = idx.get("agents") or []
    if "ecommerce_agent" not in agents:
        agents.append("ecommerce_agent")
        _atomic_write_json(AGENTS_INDEX_FILE, {"agents": sorted(set(agents))})


def load_agents_index() -> Dict[str, Any]:
    """Load agents index, ensuring default agent exists first."""
    ensure_default_agent()
    try:
        return json.loads(AGENTS_INDEX_FILE.read_text())
    except Exception:
        return {"agents": ["ecommerce_agent"]}


def list_agents() -> List[Dict[str, Any]]:
    ensure_default_agent()
    idx = load_agents_index()
    out: List[Dict[str, Any]] = []
    for agent_id in idx.get("agents") or []:
        cfg = get_agent(agent_id)
        if cfg:
            out.append(cfg)
    return out


def list_agents_by_type(agent_type: str) -> List[Dict[str, Any]]:
    validate_agent_type(agent_type)
    return [a for a in list_agents() if a.get("agent_type") == agent_type]


def get_agent(agent_id: str) -> Optional[Dict[str, Any]]:
    ensure_default_agent()
    validate_safe_name(agent_id, "agent_id")
    path = AGENTS_DIR / f"{agent_id}.json"
    if not path.exists():
        return None
    try:
        cfg = json.loads(path.read_text())
        return cfg if isinstance(cfg, dict) else None
    except Exception:
        return None


def _ensure_unique_agent_id(base_id: str) -> str:
    validate_safe_name(base_id, "agent_id")
    existing = set((load_agents_index().get("agents") or []))
    if base_id not in existing and not (AGENTS_DIR / f"{base_id}.json").exists():
        return base_id
    # suffix
    i = 2
    while True:
        candidate = f"{base_id}-{i}"
        if candidate not in existing and not (AGENTS_DIR / f"{candidate}.json").exists():
            return candidate
        i += 1


def create_agent(
    *,
    name: str,
    description: str,
    agent_type: str,
    domain_text: str,
    data_folder: str,
) -> Dict[str, Any]:
    ensure_default_agent()
    validate_agent_type(agent_type)
    validate_safe_name(data_folder, "data_folder")

    base_id = slugify_name(name)
    agent_id = _ensure_unique_agent_id(base_id)

    # Write domain file
    domain_file = f"{agent_id}_domain.md"
    domain_path = DOMAINS_DIR / domain_file
    _atomic_write_text(domain_path, domain_text or "")

    # Create data folder if missing
    folder_path = DATAWAREHOUSE_DIR / data_folder
    folder_path.mkdir(parents=True, exist_ok=True)

    cfg = {
        "id": agent_id,
        "name": name.strip(),
        "description": (description or "").strip(),
        "agent_type": agent_type,
        "domain_file": domain_file,
        "data_folder": data_folder,
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
    }

    _atomic_write_json(AGENTS_DIR / f"{agent_id}.json", cfg)

    idx = load_agents_index()
    agents = set(idx.get("agents") or [])
    agents.add(agent_id)
    _atomic_write_json(AGENTS_INDEX_FILE, {"agents": sorted(agents)})

    return cfg


def delete_agent(agent_id: str) -> None:
    validate_safe_name(agent_id, "agent_id")
    cfg_path = AGENTS_DIR / f"{agent_id}.json"
    if cfg_path.exists():
        cfg_path.unlink()
    # remove from index
    idx = load_agents_index()
    agents = [a for a in (idx.get("agents") or []) if a != agent_id]
    _atomic_write_json(AGENTS_INDEX_FILE, {"agents": agents})


