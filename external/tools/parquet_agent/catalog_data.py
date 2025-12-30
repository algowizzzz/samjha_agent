"""
Catalog Data Tool - Deterministically list available data files and their registered view names
scoped to a single agent's data folder.

This tool exists to prevent the Decider from guessing file paths / view names, especially for
subfolder-based agents (e.g., monthly snapshots).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from tools.base_mcp_tool import BaseMCPTool


def _derive_view_name(rel_path_no_suffix: str) -> str:
    """
    Must match ExecuteSQLTool's view naming:
    - Use relative path (without extension)
    - Replace / and \\ with _
    - Sanitize -, . into _
    """
    view_name = rel_path_no_suffix.replace("/", "_").replace("\\", "_")
    view_name = view_name.replace("-", "_").replace(".", "_")
    return view_name


class CatalogDataTool(BaseMCPTool):
    def __init__(self, config=None):
        default_config = {
            "name": "catalog_data",
            "description": "Catalog available data files and corresponding DuckDB view names for an agent folder",
            "version": "1.0.0",
            "enabled": True,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        self.base_path = Path(self.config.get("data_directory", "external/datawarehouse"))

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["agent_data_folder"],
            "properties": {
                "agent_data_folder": {
                    "type": "string",
                    "description": "Agent data folder under data_directory (e.g., 'ECommerce' or 'ecommerce_advanced')",
                }
            },
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "required": ["entries"],
            "properties": {
                "entries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["file_path", "view_name", "ext"],
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path relative to agent_data_folder (agent-relative)",
                            },
                            "view_name": {
                                "type": "string",
                                "description": "DuckDB view name that execute_sql will register for this file",
                            },
                            "ext": {"enum": ["csv", "parquet"]},
                        },
                    },
                },
                "error": {"type": ["string", "null"]},
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        agent_data_folder = (arguments.get("agent_data_folder") or "").strip()
        if not agent_data_folder:
            return {"entries": [], "error": "Missing agent_data_folder"}

        agent_dir = self.base_path / agent_data_folder
        if not agent_dir.exists() or not agent_dir.is_dir():
            return {"entries": [], "error": f"Agent data folder not found: {agent_data_folder}"}

        entries: List[Dict[str, str]] = []

        for file in agent_dir.glob("**/*.csv"):
            rel_path = file.relative_to(agent_dir).as_posix()
            rel_no_suffix = str(Path(rel_path).with_suffix(""))
            entries.append(
                {
                    "file_path": rel_path,
                    "view_name": _derive_view_name(rel_no_suffix),
                    "ext": "csv",
                }
            )

        for file in agent_dir.glob("**/*.parquet"):
            rel_path = file.relative_to(agent_dir).as_posix()
            rel_no_suffix = str(Path(rel_path).with_suffix(""))
            entries.append(
                {
                    "file_path": rel_path,
                    "view_name": _derive_view_name(rel_no_suffix),
                    "ext": "parquet",
                }
            )

        entries.sort(key=lambda x: x["file_path"])
        return {"entries": entries, "error": None}


