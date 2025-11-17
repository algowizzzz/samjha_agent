"""
Copyright All rights Reserved 2025-2030, Ashutosh Sinha, Email: ajsinha@gmail.com
Model Documentation MCP Tool - Save Documentation
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from tools.base_mcp_tool import BaseMCPTool


class ModelDocSaveDocTool(BaseMCPTool):
    """Save generated documentation to a file."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "model_doc_save_doc",
            "description": "Save generated markdown documentation to a timestamped output directory.",
            "inputSchema": self.get_input_schema(),
            "outputSchema": self.get_output_schema(),
        }
        if config:
            tool_config.update(config)
        super().__init__(tool_config)

    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "markdown_content": {
                    "type": "string",
                    "description": "Complete markdown documentation to save.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Base output directory (default: external/data/model_doc/output/).",
                    "default": "external/data/model_doc/output/",
                },
                "filename": {
                    "type": ["string", "null"],
                    "description": "Output filename (default: final_documentation.md).",
                    "default": "final_documentation.md",
                },
                "create_timestamped_dir": {
                    "type": "boolean",
                    "description": "Whether to create a timestamped subdirectory (default: true).",
                    "default": True,
                },
                "codebase_id": {
                    "type": ["string", "null"],
                    "description": "Optional codebase ID for subdirectory naming.",
                },
            },
            "required": ["markdown_content"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "output_directory": {"type": "string"},
                "filename": {"type": "string"},
                "timestamp": {"type": "string"},
            },
            "required": ["file_path", "output_directory", "timestamp"],
            "additionalProperties": False,
        }

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        markdown_content = arguments["markdown_content"]
        output_dir = Path(arguments.get("output_dir", "external/data/model_doc/output/")).expanduser()
        filename = arguments.get("filename", "final_documentation.md")
        create_timestamped_dir = arguments.get("create_timestamped_dir", True)
        codebase_id = arguments.get("codebase_id")

        # Ensure filename ends with .md
        if not filename.endswith(".md"):
            filename = f"{filename}.md"

        # Create output directory structure
        if create_timestamped_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if codebase_id:
                subdir_name = f"{codebase_id}_{timestamp}"
            else:
                subdir_name = f"model_doc_{timestamp}"
            output_path = output_dir / subdir_name
        else:
            output_path = output_dir

        output_path.mkdir(parents=True, exist_ok=True)

        # Write file
        file_path = output_path / filename
        try:
            with file_path.open("w", encoding="utf-8") as f:
                f.write(markdown_content)
        except Exception as e:
            raise ValueError(f"Failed to write documentation to {file_path}: {e}")

        timestamp_iso = datetime.now().isoformat() + "Z"

        self.logger.info(
            f"model_doc_save_doc: file_path={file_path} size={len(markdown_content)} bytes"
        )

        return {
            "file_path": str(file_path),
            "output_directory": str(output_path),
            "filename": filename,
            "timestamp": timestamp_iso,
        }

