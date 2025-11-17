"""
Copyright All rights Reserved 2025-2030, Ashutosh Sinha, Email: ajsinha@gmail.com
Model Documentation MCP Tool - List Codebase Files
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.base_mcp_tool import BaseMCPTool


class ModelDocListFilesTool(BaseMCPTool):
    """List all Python files in a codebase directory."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "model_doc_list_files",
            "description": "Recursively scan a codebase directory for Python files and return their metadata.",
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
                "codebase_path": {
                    "type": "string",
                    "description": "Path to the codebase root directory.",
                },
                "file_extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file extensions to include (default: ['.py']).",
                    "default": [".py"],
                },
                "exclude_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of directory/file patterns to exclude (e.g., ['__pycache__', '*.pyc']).",
                    "default": ["__pycache__", "*.pyc", ".git", ".venv", "venv", ".env"],
                },
            },
            "required": ["codebase_path"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "codebase_path": {"type": "string"},
                "file_count": {"type": "integer"},
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "relative_path": {"type": "string"},
                            "file_size": {"type": "integer"},
                            "modified_time": {"type": "string"},
                            "extension": {"type": "string"},
                        },
                        "required": ["file_path", "relative_path", "file_size"],
                    },
                },
            },
            "required": ["codebase_path", "file_count", "files"],
            "additionalProperties": False,
        }

    def _should_exclude(self, path: Path, exclude_patterns: List[str]) -> bool:
        """Check if a path should be excluded based on patterns."""
        path_str = str(path)
        path_parts = path.parts
        
        for pattern in exclude_patterns:
            if "*" in pattern:
                # Simple glob pattern matching
                if pattern.replace("*", "") in path_str:
                    return True
            else:
                # Exact match in path parts
                if pattern in path_parts:
                    return True
        return False

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        codebase_path = Path(arguments["codebase_path"]).expanduser().resolve()
        file_extensions = arguments.get("file_extensions", [".py"])
        exclude_patterns = arguments.get("exclude_patterns", ["__pycache__", "*.pyc", ".git"])

        if not codebase_path.exists():
            raise FileNotFoundError(f"Codebase directory not found: {codebase_path}")
        if not codebase_path.is_dir():
            raise ValueError(f"Path is not a directory: {codebase_path}")

        files = []
        file_count = 0

        # Normalize extensions (ensure they start with .)
        normalized_extensions = [ext if ext.startswith(".") else f".{ext}" for ext in file_extensions]

        try:
            for root, dirs, filenames in os.walk(codebase_path):
                # Filter out excluded directories
                dirs[:] = [d for d in dirs if not self._should_exclude(Path(root) / d, exclude_patterns)]
                
                for filename in filenames:
                    file_path = Path(root) / filename
                    
                    # Check if file should be excluded
                    if self._should_exclude(file_path, exclude_patterns):
                        continue
                    
                    # Check extension
                    if file_path.suffix.lower() not in normalized_extensions:
                        continue
                    
                    try:
                        stat = file_path.stat()
                        relative_path = file_path.relative_to(codebase_path)
                        
                        files.append({
                            "file_path": str(file_path),
                            "relative_path": str(relative_path),
                            "file_size": stat.st_size,
                            "modified_time": str(file_path.stat().st_mtime),
                            "extension": file_path.suffix,
                        })
                        file_count += 1
                    except (OSError, PermissionError) as e:
                        self.logger.warning(f"Could not access file {file_path}: {e}")
                        continue

            self.logger.info(
                f"model_doc_list_files: codebase_path={codebase_path} found={file_count} files"
            )

            return {
                "codebase_path": str(codebase_path),
                "file_count": file_count,
                "files": sorted(files, key=lambda x: x["relative_path"]),
            }

        except Exception as e:
            self.logger.error(f"Failed to list files in codebase: {e}")
            raise ValueError(f"Failed to list files: {str(e)}")

