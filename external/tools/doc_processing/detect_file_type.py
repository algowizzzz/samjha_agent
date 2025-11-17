from pathlib import Path
import mimetypes
from typing import Any, Dict, Optional

from tools.base_mcp_tool import BaseMCPTool


class DetectFileTypeTool(BaseMCPTool):
    """Determine the logical file_type for an uploaded document."""

    DEFAULT_EXTENSION_MAP = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".pptx": "pptx",
        ".md": "md",
        ".markdown": "md",
        ".txt": "txt",
        ".csv": "csv",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "detect_file_type",
            "description": "Detects the canonical file type for a source document.",
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
                "file_id": {
                    "type": "string",
                    "description": "Internal identifier for the file.",
                },
                "source_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the uploaded file.",
                },
            },
            "required": ["file_id", "source_path"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "file_type": {
                    "type": "string",
                    "description": "One of pdf|docx|pptx|md|txt|csv|unknown.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Rough confidence score between 0 and 1.",
                },
                "detected_via": {
                    "type": "string",
                    "description": "Detection mechanism used (extension|mimetype|fallback).",
                },
            },
            "required": ["file_id", "file_type"],
            "additionalProperties": False,
        }

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        file_id = arguments["file_id"]
        source_path = Path(arguments["source_path"]).expanduser().resolve()

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        confidence = 0.3
        detected_via = "fallback"

        ext = source_path.suffix.lower()
        if ext in self.DEFAULT_EXTENSION_MAP:
            file_type = self.DEFAULT_EXTENSION_MAP[ext]
            confidence = 0.9
            detected_via = "extension"
        else:
            mimetype, _ = mimetypes.guess_type(str(source_path))
            if mimetype:
                detected_via = "mimetype"
                confidence = 0.6
                if "pdf" in mimetype:
                    file_type = "pdf"
                elif "word" in mimetype or "doc" in mimetype:
                    file_type = "docx"
                elif "presentation" in mimetype or "powerpoint" in mimetype:
                    file_type = "pptx"
                elif "markdown" in mimetype:
                    file_type = "md"
                elif mimetype.startswith("text/"):
                    file_type = "txt"
                else:
                    file_type = "unknown"
            else:
                file_type = "unknown"

        self.logger.info(
            "detect_file_type: file_id=%s path=%s ext=%s mimetype=%s type=%s",
            file_id,
            source_path,
            ext,
            mimetypes.guess_type(str(source_path))[0],
            file_type,
        )

        return {
            "file_id": file_id,
            "file_type": file_type,
            "confidence": confidence,
            "detected_via": detected_via,
        }
