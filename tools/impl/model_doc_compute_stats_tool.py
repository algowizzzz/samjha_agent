"""
Copyright All rights Reserved 2025-2030, Ashutosh Sinha, Email: ajsinha@gmail.com
Model Documentation MCP Tool - Compute Code Statistics
"""

from typing import Any, Dict, List, Optional

from tools.base_mcp_tool import BaseMCPTool


class ModelDocComputeStatsTool(BaseMCPTool):
    """Compute statistics from file content and AST structure."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "model_doc_compute_stats",
            "description": "Calculate lines of code, complexity metrics, and counts from file content and AST structure.",
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
                "file_content": {
                    "type": "string",
                    "description": "Content of the Python file.",
                },
                "ast_structure": {
                    "type": "object",
                    "description": "AST structure from parse_code_structure.",
                    "properties": {
                        "classes": {"type": "array"},
                        "functions": {"type": "array"},
                        "imports": {"type": "array"},
                    },
                },
            },
            "required": ["file_content", "ast_structure"],
            "additionalProperties": False,
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "total_lines": {"type": "integer"},
                "code_lines": {"type": "integer"},
                "comment_lines": {"type": "integer"},
                "blank_lines": {"type": "integer"},
                "class_count": {"type": "integer"},
                "function_count": {"type": "integer"},
                "method_count": {"type": "integer"},
                "import_count": {"type": "integer"},
                "complexity_score": {"type": "number"},
            },
            "required": ["total_lines", "code_lines", "class_count", "function_count"],
            "additionalProperties": False,
        }

    def _count_lines(self, content: str) -> Dict[str, int]:
        """Count different types of lines."""
        lines = content.split("\n")
        total = len(lines)
        code = 0
        comments = 0
        blank = 0
        
        in_multiline = False
        multiline_char = None
        
        for line in lines:
            stripped = line.strip()
            
            # Skip blank lines
            if not stripped:
                blank += 1
                continue
            
            # Check for multiline strings/comments
            if '"""' in stripped or "'''" in stripped:
                # Simple detection - could be improved
                if stripped.count('"""') % 2 != 0 or stripped.count("'''") % 2 != 0:
                    in_multiline = not in_multiline
            
            # Count comments
            if stripped.startswith("#"):
                comments += 1
            elif in_multiline and (stripped.startswith('"""') or stripped.startswith("'''")):
                comments += 1
            else:
                code += 1
        
        return {
            "total_lines": total,
            "code_lines": code,
            "comment_lines": comments,
            "blank_lines": blank,
        }

    def _calculate_complexity(self, ast_structure: Dict[str, Any]) -> float:
        """Calculate a simple complexity score based on structure."""
        classes = ast_structure.get("classes", [])
        functions = ast_structure.get("functions", [])
        
        # Simple complexity: classes * 2 + functions + methods
        method_count = sum(len(c.get("methods", [])) for c in classes)
        complexity = len(classes) * 2 + len(functions) + method_count
        
        return float(complexity)

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        file_content = arguments["file_content"]
        ast_structure = arguments["ast_structure"]

        # Count lines
        line_counts = self._count_lines(file_content)

        # Extract counts from AST
        classes = ast_structure.get("classes", [])
        functions = ast_structure.get("functions", [])
        imports = ast_structure.get("imports", [])
        
        method_count = sum(len(c.get("methods", [])) for c in classes)

        # Calculate complexity
        complexity_score = self._calculate_complexity(ast_structure)

        self.logger.info(
            f"model_doc_compute_stats: lines={line_counts['code_lines']} classes={len(classes)} functions={len(functions)}"
        )

        return {
            "total_lines": line_counts["total_lines"],
            "code_lines": line_counts["code_lines"],
            "comment_lines": line_counts["comment_lines"],
            "blank_lines": line_counts["blank_lines"],
            "class_count": len(classes),
            "function_count": len(functions),
            "method_count": method_count,
            "import_count": len(imports),
            "complexity_score": complexity_score,
        }

