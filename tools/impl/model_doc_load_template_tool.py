"""
Copyright All rights Reserved 2025-2030, Ashutosh Sinha, Email: ajsinha@gmail.com
Model Documentation MCP Tool - Load Documentation Template
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from tools.base_mcp_tool import BaseMCPTool


class ModelDocLoadTemplateTool(BaseMCPTool):
    """Load a documentation template from JSON file."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "model_doc_load_template",
            "description": "Load a documentation template JSON structure from disk.",
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
                "template_id": {
                    "type": "string",
                    "description": "ID of the template to load (filename without .json extension).",
                },
                "template_path": {
                    "type": ["string", "null"],
                    "description": "Optional explicit path to template file.",
                },
                "default_path": {
                    "type": ["string", "null"],
                    "description": "Default directory to search for templates (default: config/doc_review/outline_templates/).",
                    "default": "config/doc_review/outline_templates/",
                },
            },
            "required": ["template_id"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "template": {
                    "type": "object",
                    "description": "Template JSON structure.",
                },
                "template_id": {"type": "string"},
                "template_path": {"type": "string"},
            },
            "required": ["template", "template_id", "template_path"],
            "additionalProperties": False,
        }

    def _resolve_template_path(
        self, template_id: str, template_path: Optional[str] = None, default_path: Optional[str] = None
    ) -> Path:
        """Resolve the template file path."""
        if template_path:
            path = Path(template_path).expanduser().resolve()
        else:
            base_dir = Path(default_path or "config/doc_review/outline_templates/")
            # Try with .json extension
            path = base_dir / f"{template_id}.json"
            if not path.exists():
                # Try without extension
                path = base_dir / template_id
        
        return path

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        template_id = arguments["template_id"]
        template_path_str = arguments.get("template_path")
        default_path = arguments.get("default_path", "config/doc_review/outline_templates/")

        path = self._resolve_template_path(template_id, template_path_str, default_path)

        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as f:
                template = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in template file {path}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load template from {path}: {e}")

        self.logger.info(
            f"model_doc_load_template: template_id={template_id} path={path} sections={len(template.get('sections', []))}"
        )

        return {
            "template": template,
            "template_id": template_id,
            "template_path": str(path),
        }

