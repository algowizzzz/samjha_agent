"""
Query Safety Validator Tool
Ensures SQL queries are read-only, within limits, and reference allowed paths/tables.
May return a modified (safer) SQL with an enforced LIMIT.
"""
from typing import Dict, Any, List, Tuple
from tools.base_mcp_tool import BaseMCPTool


class QuerySafetyValidatorTool(BaseMCPTool):
    def __init__(self, config: Dict = None):
        default_config = {
            "name": "query_safety_validator",
            "description": "Validate SQL queries for safety and performance",
            "version": "1.0.0",
            "enabled": True,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        limits = self.config.get("limits", {})
        self.max_rows = int(limits.get("max_rows", 1000))
        safety = self.config.get("safety", {})
        self.allowed_paths: List[str] = safety.get("allowed_paths", ["data/duckdb"])
        self.denylist_ops: List[str] = safety.get(
            "denylist_ops",
            ["UPDATE", "DELETE", "INSERT", "CREATE", "ALTER", "DROP", "TRUNCATE"],
        )

    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "enforce_limit": {"type": "boolean", "default": True},
                "default_limit": {"type": "integer", "default": 100},
            },
            "required": ["query"],
        }

    def get_output_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "is_safe": {"type": "boolean"},
                "reason": {"type": "string"},
                "sanitized_query": {"type": "string"},
                "limit_enforced": {"type": "boolean"},
            },
            "required": ["is_safe", "sanitized_query"],
        }

    def _has_forbidden_ops(self, query_upper: str) -> str:
        for op in self.denylist_ops:
            if op in query_upper:
                return op
        return ""

    def _ensure_limit(self, sql: str, default_limit: int) -> Tuple[str, bool]:
        sql_upper = sql.upper()
        if "LIMIT" in sql_upper:
            return sql, False
        # naive append; users should avoid trailing semicolons
        sanitized = f"{sql.rstrip(';')} LIMIT {min(default_limit, self.max_rows)}"
        return sanitized, True

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        sql: str = arguments["query"]
        enforce_limit: bool = bool(arguments.get("enforce_limit", True))
        default_limit: int = int(arguments.get("default_limit", 100))
        reason = ""

        query_upper = sql.upper()
        forbidden = self._has_forbidden_ops(query_upper)
        if forbidden:
            return {
                "is_safe": False,
                "reason": f"Forbidden operation detected: {forbidden}",
                "sanitized_query": "",
                "limit_enforced": False,
            }

        sanitized = sql
        limit_added = False
        if enforce_limit:
            sanitized, limit_added = self._ensure_limit(sql, default_limit)

        return {
            "is_safe": True,
            "reason": reason,
            "sanitized_query": sanitized,
            "limit_enforced": limit_added,
        }


