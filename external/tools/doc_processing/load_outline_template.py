from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from tools.base_mcp_tool import BaseMCPTool

from .utils import derive_template_path, read_json, resolve_path


class LoadOutlineTemplateTool(BaseMCPTool):
    """Load a predefined outline template from disk."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "load_outline_template",
            "description": "Loads the outline template JSON used for mapping document headings.",
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
                "template_id": {"type": "string"},
                "template_path": {"type": ["string", "null"]},
            },
            "required": ["template_id"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "template": {"type": "object"},
                "template_id": {"type": "string"},
                "template_path": {"type": "string"},
            },
            "required": ["template", "template_id", "template_path"],
            "additionalProperties": False,
        }

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        template_id = arguments["template_id"]
        template_path_str = arguments.get("template_path")

        path: Path
        if template_path_str:
            path = resolve_path(template_path_str)
        else:
            path = derive_template_path(template_id)

        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")

        template = read_json(path)

        self.logger.info(
            "load_outline_template: template_id=%s path=%s sections=%d",
            template_id,
            path,
            len(template.get("sections", [])),
        )

        return {
            "template": template,
            "template_id": template_id,
            "template_path": str(path),
        }
