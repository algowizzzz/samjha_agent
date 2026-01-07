"""
List Excel Files Tool - Discover Excel files in a data folder.
"""

from pathlib import Path
from tools.base_mcp_tool import BaseMCPTool


class ListExcelFilesTool(BaseMCPTool):
    def __init__(self, config=None):
        default_config = {
            "name": "list_excel_files",
            "description": "List Excel files (.xlsx, .xls) in a data folder",
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
            "properties": {
                "agent_data_folder": {
                    "type": "string",
                    "description": "Optional agent-specific data folder to scope search"
                },
                "path": {
                    "type": "string",
                    "description": "Optional subdirectory path within data folder"
                }
            }
        }

    def get_output_schema(self):
        return {
            "type": "object",
            "required": ["files"],
            "properties": {
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "path", "size_bytes"],
                        "properties": {
                            "name": {"type": "string"},
                            "path": {"type": "string"},
                            "size_bytes": {"type": "number"}
                        }
                    }
                }
            }
        }

    def execute(self, arguments):
        agent_data_folder = arguments.get("agent_data_folder")
        subpath = arguments.get("path", "")
        
        if agent_data_folder:
            target = self.base_path / agent_data_folder
            if subpath:
                target = target / subpath
        else:
            target = self.base_path
            if subpath:
                target = target / subpath
        
        if not target.exists():
            return {"files": []}
        
        files = []
        for item in target.rglob("*.xlsx"):
            if item.is_file():
                files.append({
                    "name": item.name,
                    "path": str(item.relative_to(self.base_path)),
                    "size_bytes": item.stat().st_size
                })
        for item in target.rglob("*.xls"):
            if item.is_file():
                files.append({
                    "name": item.name,
                    "path": str(item.relative_to(self.base_path)),
                    "size_bytes": item.stat().st_size
                })
        
        return {"files": sorted(files, key=lambda x: x["name"])}

