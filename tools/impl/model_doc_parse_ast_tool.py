"""
Copyright All rights Reserved 2025-2030, Ashutosh Sinha, Email: ajsinha@gmail.com
Model Documentation MCP Tool - Parse Code Structure (AST)
"""

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.base_mcp_tool import BaseMCPTool


class ModelDocParseASTTool(BaseMCPTool):
    """Parse Python code using AST to extract structure (classes, functions, imports)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        tool_config = {
            "name": "model_doc_parse_ast",
            "description": "Parse Python code using AST to extract classes, functions, methods, imports, and docstrings.",
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
                    "description": "Path to the Python file.",
                },
                "file_content": {
                    "type": "string",
                    "description": "Content of the Python file (optional, will read from file_path if not provided).",
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
                "classes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "line_number": {"type": "integer"},
                            "docstring": {"type": ["string", "null"]},
                            "methods": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "line_number": {"type": "integer"},
                                        "args": {"type": "array"},
                                        "docstring": {"type": ["string", "null"]},
                                    },
                                },
                            },
                            "base_classes": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "functions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "line_number": {"type": "integer"},
                            "args": {"type": "array"},
                            "docstring": {"type": ["string", "null"]},
                            "decorators": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "imports": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "module": {"type": "string"},
                            "names": {"type": "array", "items": {"type": "string"}},
                            "type": {"type": "string", "enum": ["import", "from"]},
                            "line_number": {"type": "integer"},
                        },
                    },
                },
                "docstring": {"type": ["string", "null"]},
                "ast_errors": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["file_path", "classes", "functions", "imports"],
            "additionalProperties": False,
        }

    def _extract_docstring(self, node: ast.AST) -> Optional[str]:
        """Extract docstring from an AST node."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Str)
            ):
                return node.body[0].value.s
            elif (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                return node.body[0].value.value
        return None

    def _extract_args(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[str]:
        """Extract argument names from a function node."""
        args = [arg.arg for arg in node.args.args]
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        return args

    def _extract_decorators(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> List[str]:
        """Extract decorator names."""
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(ast.unparse(decorator) if hasattr(ast, 'unparse') else str(decorator))
        return decorators

    def _extract_base_classes(self, node: ast.ClassDef) -> List[str]:
        """Extract base class names."""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(ast.unparse(base) if hasattr(ast, 'unparse') else str(base))
        return bases

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        file_path = Path(arguments["file_path"])
        file_content = arguments.get("file_content")

        # Read file if content not provided
        if not file_content:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            with file_path.open("r", encoding="utf-8", errors="replace") as f:
                file_content = f.read()

        classes: List[Dict[str, Any]] = []
        functions: List[Dict[str, Any]] = []
        imports: List[Dict[str, Any]] = []
        ast_errors: List[str] = []
        module_docstring: Optional[str] = None

        try:
            tree = ast.parse(file_content, filename=str(file_path))
            
            # Extract module docstring
            module_docstring = self._extract_docstring(tree)

            # Use a visitor to track parent nodes
            class Visitor(ast.NodeVisitor):
                def __init__(self):
                    self.classes = []
                    self.functions = []
                    self.imports = []
                    
                def visit_ClassDef(self, node):
                    # Extract methods
                    methods = []
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            methods.append({
                                "name": item.name,
                                "line_number": item.lineno,
                                "args": self._extract_args(item),
                                "docstring": self._extract_docstring_impl(item),
                            })
                    
                    self.classes.append({
                        "name": node.name,
                        "line_number": node.lineno,
                        "docstring": self._extract_docstring_impl(node),
                        "methods": methods,
                        "base_classes": self._extract_base_classes(node),
                    })
                    self.generic_visit(node)
                    
                def visit_FunctionDef(self, node):
                    # Only top-level functions (check if parent is module)
                    parent = getattr(node, 'parent', None)
                    if parent is None or isinstance(parent, ast.Module):
                        self.functions.append({
                            "name": node.name,
                            "line_number": node.lineno,
                            "args": self._extract_args(node),
                            "docstring": self._extract_docstring_impl(node),
                            "decorators": self._extract_decorators(node),
                        })
                    self.generic_visit(node)
                    
                def visit_AsyncFunctionDef(self, node):
                    parent = getattr(node, 'parent', None)
                    if parent is None or isinstance(parent, ast.Module):
                        self.functions.append({
                            "name": node.name,
                            "line_number": node.lineno,
                            "args": self._extract_args(node),
                            "docstring": self._extract_docstring_impl(node),
                            "decorators": self._extract_decorators(node),
                        })
                    self.generic_visit(node)
                    
                def visit_Import(self, node):
                    for alias in node.names:
                        self.imports.append({
                            "module": alias.name,
                            "names": [alias.asname or alias.name],
                            "type": "import",
                            "line_number": node.lineno,
                        })
                    self.generic_visit(node)
                    
                def visit_ImportFrom(self, node):
                    names = [alias.name for alias in node.names] if node.names else []
                    module = node.module or ""
                    self.imports.append({
                        "module": module,
                        "names": names,
                        "type": "from",
                        "line_number": node.lineno,
                    })
                    self.generic_visit(node)
                    
                def _extract_docstring_impl(self, node):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                        if (
                            node.body
                            and isinstance(node.body[0], ast.Expr)
                            and isinstance(node.body[0].value, ast.Str)
                        ):
                            return node.body[0].value.s
                        elif (
                            node.body
                            and isinstance(node.body[0], ast.Expr)
                            and isinstance(node.body[0].value, ast.Constant)
                            and isinstance(node.body[0].value.value, str)
                        ):
                            return node.body[0].value.value
                    return None
                    
                def _extract_args(self, node):
                    args = [arg.arg for arg in node.args.args]
                    if node.args.vararg:
                        args.append(f"*{node.args.vararg.arg}")
                    if node.args.kwarg:
                        args.append(f"**{node.args.kwarg.arg}")
                    return args
                    
                def _extract_decorators(self, node):
                    decorators = []
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Name):
                            decorators.append(decorator.id)
                        elif isinstance(decorator, ast.Attribute):
                            decorators.append(ast.unparse(decorator) if hasattr(ast, 'unparse') else str(decorator))
                    return decorators
                    
                def _extract_base_classes(self, node):
                    bases = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            bases.append(base.id)
                        elif isinstance(base, ast.Attribute):
                            bases.append(ast.unparse(base) if hasattr(ast, 'unparse') else str(base))
                    return bases

            # Traverse with parent tracking
            def add_parent_info(node, parent=None):
                if hasattr(node, '__dict__'):
                    node.parent = parent
                for child in ast.iter_child_nodes(node):
                    add_parent_info(child, node)
            
            add_parent_info(tree)
            visitor = Visitor()
            visitor.visit(tree)
            
            classes = visitor.classes
            functions = visitor.functions
            imports = visitor.imports

        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            ast_errors.append(error_msg)
            self.logger.warning(f"AST parsing error for {file_path}: {error_msg}")
        except Exception as e:
            error_msg = f"AST parsing error: {str(e)}"
            ast_errors.append(error_msg)
            self.logger.error(f"Failed to parse AST for {file_path}: {e}")

        self.logger.info(
            f"model_doc_parse_ast: file_path={file_path} classes={len(classes)} functions={len(functions)} imports={len(imports)}"
        )

        return {
            "file_path": str(file_path),
            "classes": classes,
            "functions": functions,
            "imports": imports,
            "docstring": module_docstring,
            "ast_errors": ast_errors if ast_errors else None,
        }

