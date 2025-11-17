"""
Copyright All rights Reserved 2025-2030, Ashutosh Sinha, Email: ajsinha@gmail.com
Model Documentation MCP Tool - Read Code File
"""

from pathlib import Path
from typing import Any, Dict, Optional

from tools.base_mcp_tool import BaseMCPTool


class ModelDocReadFileTool(BaseMCPTool):
    """Read a code file and detect its encoding."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "model_doc_read_file",
            "description": "Read a code file, detect encoding, and return content with metadata.",
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
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read.",
                },
                "encoding": {
                    "type": "string",
                    "description": "Optional encoding to use (if not provided, will be detected).",
                },
                "max_size_mb": {
                    "type": "number",
                    "description": "Maximum file size in MB to read (default: 10).",
                    "default": 10,
                },
            },
            "required": ["file_path"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
                "encoding": {"type": "string"},
                "line_count": {"type": "integer"},
                "file_size": {"type": "integer"},
                "detected_encoding": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["file_path", "content", "encoding", "line_count"],
            "additionalProperties": False,
        }

    def _detect_encoding(self, file_path: Path) -> tuple[str, float]:
        """Detect file encoding with fallback to utf-8."""
        # Try common encodings
        encodings_to_try = ["utf-8", "latin-1", "cp1252", "ascii"]
        
        for encoding in encodings_to_try:
            try:
                with file_path.open("r", encoding=encoding) as f:
                    f.read(1000)  # Try reading first 1KB
                return encoding, 0.9 if encoding == "utf-8" else 0.7
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception:
                pass
        
        # Fallback to utf-8 with low confidence
        return "utf-8", 0.5

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        file_path = Path(arguments["file_path"]).expanduser().resolve()
        encoding = arguments.get("encoding")
        max_size_mb = arguments.get("max_size_mb", 10)
        max_size_bytes = max_size_mb * 1024 * 1024

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        # Check file size
        file_size = file_path.stat().st_size
        if file_size > max_size_bytes:
            raise ValueError(
                f"File too large: {file_size} bytes (max: {max_size_bytes} bytes)"
            )

        # Detect encoding if not provided
        detected_encoding = None
        confidence = 0.0
        if not encoding:
            detected_encoding, confidence = self._detect_encoding(file_path)
            encoding = detected_encoding
        else:
            detected_encoding = encoding
            confidence = 1.0

        # Try reading with detected/provided encoding
        try:
            with file_path.open("r", encoding=encoding, errors="replace") as f:
                content = f.read()
        except (UnicodeDecodeError, LookupError) as e:
            # Fallback to utf-8 with error replacement
            self.logger.warning(f"Failed to read with {encoding}, falling back to utf-8: {e}")
            encoding = "utf-8"
            detected_encoding = "utf-8"
            confidence = 0.5
            with file_path.open("r", encoding="utf-8", errors="replace") as f:
                content = f.read()

        line_count = content.count("\n") + (1 if content else 0)

        self.logger.info(
            f"model_doc_read_file: file_path={file_path} lines={line_count} encoding={encoding}"
        )

        return {
            "file_path": str(file_path),
            "content": content,
            "encoding": encoding,
            "line_count": line_count,
            "file_size": file_size,
            "detected_encoding": detected_encoding or encoding,
            "confidence": confidence,
        }

