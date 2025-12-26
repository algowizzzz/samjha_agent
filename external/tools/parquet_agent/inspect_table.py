"""
Inspect Table Tool - Get schema information for a data file.
"""

import duckdb
from pathlib import Path
from tools.base_mcp_tool import BaseMCPTool


class InspectTableTool(BaseMCPTool):
    def __init__(self, config=None):
        default_config = {
            "name": "inspect_table",
            "description": "Get schema information for a data file",
            "version": "1.0.0",
            "enabled": True,
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        self.base_path = Path(self.config.get("data_directory", "mock_datawarehouse"))

    def get_input_schema(self):
        return {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {"type": "string"}
            }
        }

    def get_output_schema(self):
        return {
            "type": "object",
            "required": ["columns"],
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "type"],
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"}
                        }
                    }
                },
                "row_count_estimate": {"type": ["integer", "null"]},
                "primary_key_candidates": {"type": "array", "items": {"type": "string"}},
                "time_column_candidates": {"type": "array", "items": {"type": "string"}}
            }
        }

    def execute(self, arguments):
        path = arguments["path"]
        full_path = self.base_path / path
        
        if not full_path.exists():
            raise ValueError(f"File not found: {path}")
        
        conn = duckdb.connect(":memory:")
        
        try:
            # Detect file type and read
            if full_path.suffix == ".parquet":
                query = f"DESCRIBE SELECT * FROM read_parquet('{str(full_path)}')"
                count_query = f"SELECT COUNT(*) FROM read_parquet('{str(full_path)}')"
            else:  # CSV
                query = f"DESCRIBE SELECT * FROM read_csv_auto('{str(full_path)}')"
                count_query = f"SELECT COUNT(*) FROM read_csv_auto('{str(full_path)}')"
            
            # Get schema
            schema_result = conn.execute(query).fetchall()
            columns = [{"name": row[0], "type": row[1]} for row in schema_result]
            
            # Get row count
            row_count = conn.execute(count_query).fetchone()[0]
            
            # Heuristics for key/time columns
            pk_candidates = [c["name"] for c in columns if "_id" in c["name"].lower() or c["name"].lower() == "id"]
            time_candidates = [c["name"] for c in columns if "date" in c["name"].lower() or "time" in c["name"].lower()]
            
            return {
                "columns": columns,
                "row_count_estimate": row_count,
                "primary_key_candidates": pk_candidates,
                "time_column_candidates": time_candidates
            }
        finally:
            conn.close()

