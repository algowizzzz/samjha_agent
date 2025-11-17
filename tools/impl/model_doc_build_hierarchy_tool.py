"""
Copyright All rights Reserved 2025-2030, Ashutosh Sinha, Email: ajsinha@gmail.com
Model Documentation MCP Tool - Build File Hierarchy
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.base_mcp_tool import BaseMCPTool


class ModelDocBuildHierarchyTool(BaseMCPTool):
    """Build a directory tree hierarchy from a list of files."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "model_doc_build_hierarchy",
            "description": "Build a directory tree structure with module relationships from a file list.",
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
                "file_list": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "relative_path": {"type": "string"},
                        },
                        "required": ["file_path", "relative_path"],
                    },
                    "description": "List of files with paths from list_codebase_files.",
                },
                "codebase_path": {
                    "type": "string",
                    "description": "Root path of the codebase.",
                },
            },
            "required": ["file_list"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "hierarchy": {
                    "type": "object",
                    "description": "Tree structure with directory/file hierarchy.",
                },
                "modules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of identified Python modules.",
                },
                "packages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of identified Python packages (directories with __init__.py).",
                },
            },
            "required": ["hierarchy"],
            "additionalProperties": False,
        }

    def _build_tree(self, file_list: List[Dict[str, Any]], codebase_path: Optional[str] = None) -> Dict[str, Any]:
        """Build a tree structure from file paths."""
        root = {"type": "directory", "name": "", "children": {}, "files": []}
        
        for file_info in file_list:
            relative_path = file_info.get("relative_path", file_info.get("file_path", ""))
            file_path_obj = Path(relative_path)
            
            # Build directory structure
            current = root
            parts = file_path_obj.parts[:-1] if file_path_obj.name else file_path_obj.parts
            
            for part in parts:
                if part not in current["children"]:
                    current["children"][part] = {
                        "type": "directory",
                        "name": part,
                        "path": str(Path(*file_path_obj.parts[:file_path_obj.parts.index(part) + 1])),
                        "children": {},
                        "files": [],
                    }
                current = current["children"][part]
            
            # Add file to appropriate directory
            filename = file_path_obj.name
            current["files"].append({
                "name": filename,
                "path": relative_path,
                "file_path": file_info.get("file_path", relative_path),
            })
        
        return root

    def _find_modules_and_packages(self, hierarchy: Dict[str, Any], codebase_path: Optional[str] = None) -> tuple[List[str], List[str]]:
        """Identify Python modules and packages."""
        modules = []
        packages = []
        codebase_path_obj = Path(codebase_path) if codebase_path else None
        
        def traverse(node: Dict[str, Any], current_path: Path = Path("")):
            # Check if this is a package (has __init__.py)
            if any(f["name"] == "__init__.py" for f in node.get("files", [])):
                package_path = str(current_path) if current_path else "."
                packages.append(package_path)
            
            # Add Python files as modules
            for file_info in node.get("files", []):
                if file_info["name"].endswith(".py"):
                    module_path = str(current_path / file_info["name"][:-3]) if current_path else file_info["name"][:-3]
                    module_path = module_path.replace("\\", ".").replace("/", ".")
                    modules.append(module_path)
            
            # Recursively process children
            for child_name, child_node in node.get("children", {}).items():
                traverse(child_node, current_path / child_name if current_path else Path(child_name))
        
        traverse(hierarchy)
        return modules, packages

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        file_list = arguments["file_list"]
        codebase_path = arguments.get("codebase_path")

        if not file_list:
            return {
                "hierarchy": {"type": "directory", "name": "", "children": {}, "files": []},
                "modules": [],
                "packages": [],
            }

        hierarchy = self._build_tree(file_list, codebase_path)
        modules, packages = self._find_modules_and_packages(hierarchy, codebase_path)

        self.logger.info(
            f"model_doc_build_hierarchy: files={len(file_list)} modules={len(modules)} packages={len(packages)}"
        )

        return {
            "hierarchy": hierarchy,
            "modules": sorted(modules),
            "packages": sorted(packages),
        }

