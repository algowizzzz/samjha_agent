from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from tools.base_mcp_tool import BaseMCPTool


class BuildFileMetadataTool(BaseMCPTool):
    """Aggregate deterministic ingestion outputs into a FileMetadata record."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "build_file_metadata",
            "description": "Builds the FileMetadata object that captures ingestion statistics and file details.",
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
                "file_id": {"type": "string"},
                "source_path": {"type": "string"},
                "file_type": {"type": "string"},
                "md_file_id": {"type": "string"},
                "md_path": {"type": "string"},
                "images": {
                    "type": "array",
                    "items": {"type": "object"},
                },
                "page_count": {"type": "integer"},
                "word_count": {"type": "integer"},
                "heading_levels": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "file_id",
                "source_path",
                "file_type",
                "md_file_id",
                "md_path",
                "images",
                "page_count",
                "word_count",
                "heading_levels",
            ],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_metadata": {"type": "object"},
            },
            "required": ["file_metadata"],
            "additionalProperties": False,
        }

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        timestamp = datetime.utcnow().isoformat() + "Z"
        file_metadata = {
            "file_id": arguments["file_id"],
            "source_path": arguments["source_path"],
            "file_type": arguments["file_type"],
            "md_file_id": arguments["md_file_id"],
            "md_path": arguments["md_path"],
            "images": arguments["images"],
            "page_count": arguments["page_count"],
            "word_count": arguments["word_count"],
            "heading_levels": arguments["heading_levels"],
            "chunking": {
                "should_chunk": None,
                "strategy": None,
                "reason": None,
                "max_chunk_tokens": None,
            },
            "stats": {
                "created_at": timestamp,
                "ingestion_version": "v1",
            },
        }

        self.logger.info(
            "build_file_metadata: file_id=%s type=%s words=%d pages=%d",
            file_metadata["file_id"],
            file_metadata["file_type"],
            file_metadata["word_count"],
            file_metadata["page_count"],
        )

        return {"file_metadata": file_metadata}
