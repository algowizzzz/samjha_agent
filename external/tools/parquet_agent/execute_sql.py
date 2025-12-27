"""
Execute SQL Tool - Execute SQL query on data warehouse.
"""

import duckdb
from pathlib import Path
from tools.base_mcp_tool import BaseMCPTool


class ExecuteSQLTool(BaseMCPTool):
    def __init__(self, config=None):
        default_config = {
            "name": "execute_sql",
            "description": "Execute SQL query on data warehouse",
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
            "required": ["sql"],
            "properties": {
                "sql": {"type": "string"},
                "timeout_seconds": {"type": ["integer", "null"]},
                "max_rows": {"type": ["integer", "null"]}
            }
        }

    def get_output_schema(self):
        return {
            "type": "object",
            "required": ["row_count", "columns", "rows_preview"],
            "properties": {
                "row_count": {"type": "integer"},
                "columns": {"type": "array", "items": {"type": "string"}},
                "rows_preview": {"type": "array", "items": {"type": "object"}}
            }
        }

    def execute(self, arguments):
        sql = arguments["sql"]
        max_rows = arguments.get("max_rows", 100) or 100
        
        conn = duckdb.connect(":memory:")
        
        try:
            # Register data files as views
            for domain_dir in self.base_path.iterdir():
                if domain_dir.is_dir():
                    for file in domain_dir.glob("*.csv"):
                        view_name = file.stem
                        conn.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_csv_auto('{str(file)}')")
                    for file in domain_dir.glob("*.parquet"):
                        view_name = file.stem
                        conn.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{str(file)}')")
            
            # Execute query
            result = conn.execute(sql)
            df = result.fetchdf()
            
            row_count = len(df)
            columns = list(df.columns)
            rows_preview = df.head(max_rows).to_dict(orient="records")
            
            return {
                "row_count": row_count,
                "columns": columns,
                "rows_preview": rows_preview
            }
        finally:
            conn.close()

