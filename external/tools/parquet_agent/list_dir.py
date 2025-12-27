"""
List Directory Tool - Discover datasets in a domain folder.
"""

from pathlib import Path
from tools.base_mcp_tool import BaseMCPTool


class ListDirTool(BaseMCPTool):
    def __init__(self, config=None):
        default_config = {
            "name": "list_dir",
            "description": "List files and directories in a domain folder",
            "version": "1.0.0",
            "enabled": True,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        self.base_path = Path(self.config.get("data_directory", "external/datawarehouse"))

    def get_input_schema(self):
        return {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {"type": "string", "description": "Domain folder path"}
            }
        }

    def get_output_schema(self):
        return {
            "type": "object",
            "required": ["entries"],
            "properties": {
                "entries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "path", "type"],
                        "properties": {
                            "name": {"type": "string"},
                            "path": {"type": "string"},
                            "type": {"enum": ["file", "dir"]}
                        }
                    }
                }
            }
        }

    def execute(self, arguments):
        path = arguments["path"]
        target = self.base_path / path
        
        if not target.exists():
            return {"entries": []}
        
        entries = []
        for item in target.iterdir():
            entries.append({
                "name": item.name,
                "path": str(item.relative_to(self.base_path)),
                "type": "dir" if item.is_dir() else "file"
            })
        
        return {"entries": sorted(entries, key=lambda x: x["name"])}

