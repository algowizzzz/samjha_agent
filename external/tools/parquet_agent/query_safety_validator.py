"""
Query Safety Validator Tool - Adapted for spec schema.
Input: sql, policy_limits
Output: allowed, flags
"""

from typing import Dict, Any, List
from tools.base_mcp_tool import BaseMCPTool


class QuerySafetyValidatorTool(BaseMCPTool):
    def __init__(self, config: Dict = None):
        default_config = {
            "name": "query_safety_validator",
            "description": "Validate SQL queries for safety and performance",
            "version": "2.0.0",  # Updated version for spec compliance
            "enabled": True,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)

    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "required": ["sql", "policy_limits"],
            "properties": {
                "sql": {"type": "string"},
                "policy_limits": {
                    "type": "object",
                    "required": ["max_attempts", "max_rows", "timeout_seconds", "allow_cross_join"],
                    "properties": {
                        "max_attempts": {"type": "integer", "minimum": 1},
                        "max_rows": {"type": "integer", "minimum": 1},
                        "timeout_seconds": {"type": "integer", "minimum": 1},
                        "allow_cross_join": {"type": "boolean"}
                    }
                }
            }
        }

    def get_output_schema(self) -> Dict:
        return {
            "type": "object",
            "required": ["allowed", "flags"],
            "properties": {
                "allowed": {"type": "boolean"},
                "flags": {"type": "array", "items": {"type": "string"}}
            }
        }

    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        sql: str = arguments["sql"]
        policy_limits: Dict = arguments.get("policy_limits", {})
        
        max_rows = policy_limits.get("max_rows", 1000)
        allow_cross_join = policy_limits.get("allow_cross_join", False)
        
        flags = []
        allowed = True
        
        sql_upper = sql.upper()
        
        # Check forbidden operations
        denylist_ops = ["DELETE", "UPDATE", "INSERT", "DROP", "TRUNCATE", "CREATE", "ALTER"]
        for op in denylist_ops:
            if op in sql_upper:
                flags.append(f"FORBIDDEN_OP:{op}")
                allowed = False
        
        # Check cross join
        if "CROSS JOIN" in sql_upper and not allow_cross_join:
            flags.append("CROSS_JOIN_BLOCKED")
            allowed = False
        
        # Check for LIMIT
        if "LIMIT" not in sql_upper:
            flags.append(f"NO_LIMIT:will_enforce_{max_rows}")
        
        return {
            "allowed": allowed,
            "flags": flags
        }

