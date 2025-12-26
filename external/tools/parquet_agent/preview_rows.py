"""
Preview Rows Tool - Sample rows from a data file.
"""

import duckdb
from pathlib import Path
from tools.base_mcp_tool import BaseMCPTool


class PreviewRowsTool(BaseMCPTool):
    def __init__(self, config=None):
        default_config = {
            "name": "preview_rows",
            "description": "Sample rows from a data file",
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
            "required": ["path", "limit"],
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100}
            }
        }

    def get_output_schema(self):
        return {
            "type": "object",
            "required": ["rows_preview"],
            "properties": {
                "rows_preview": {"type": "array", "items": {"type": "object"}}
            }
        }

    def execute(self, arguments):
        path = arguments["path"]
        limit = min(arguments.get("limit", 10), 100)
        full_path = self.base_path / path
        
        if not full_path.exists():
            raise ValueError(f"File not found: {path}")
        
        conn = duckdb.connect(":memory:")
        
        try:
            if full_path.suffix == ".parquet":
                query = f"SELECT * FROM read_parquet('{str(full_path)}') LIMIT {limit}"
            else:
                query = f"SELECT * FROM read_csv_auto('{str(full_path)}') LIMIT {limit}"
            
            df = conn.execute(query).fetchdf()
            
            # Convert to list of dicts
            rows = df.to_dict(orient="records")
            
            return {"rows_preview": rows}
        finally:
            conn.close()

