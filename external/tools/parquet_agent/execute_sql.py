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
        max_rows = arguments.get("max_rows", 50) or 50
        agent_data_folder = arguments.get("agent_data_folder")  # Get agent's data folder for scoping
        
        conn = duckdb.connect(":memory:")
        
        try:
            created_views = set()
            # Register data files as views - scope to agent's folder if provided
            if agent_data_folder:
                # Only register views from the agent's specific folder
                agent_dir = self.base_path / agent_data_folder
                if agent_dir.exists() and agent_dir.is_dir():
                    # Recursively find all CSV and parquet files and register each as a separate view
                    # Each file gets its own view name (filename without extension)
                    # Domain MD guides the agent on how to query/aggregate these views
                    for file in agent_dir.glob("**/*.csv"):
                        # Use the full relative path (without extension) as view name to ensure uniqueness
                        # e.g., "jan012204/sales_01012024" becomes view name
                        rel_path = file.relative_to(agent_dir)
                        view_name = str(rel_path.with_suffix('')).replace('/', '_').replace('\\', '_')
                        # Sanitize view name (DuckDB doesn't allow certain characters in identifiers)
                        view_name = view_name.replace('-', '_').replace('.', '_')
                        conn.execute(f"CREATE OR REPLACE VIEW \"{view_name}\" AS SELECT * FROM read_csv_auto('{str(file)}')")
                        created_views.add(view_name)

                        # Also register a convenient alias view using the filename stem when safe.
                        # This helps the LLM, which often guesses `sales_jan012024` instead of `jan012024_sales_jan012024`.
                        alias = file.stem.replace('-', '_').replace('.', '_')
                        if alias and alias != view_name and alias not in created_views:
                            conn.execute(f"CREATE OR REPLACE VIEW \"{alias}\" AS SELECT * FROM \"{view_name}\"")
                            created_views.add(alias)
                    
                    for file in agent_dir.glob("**/*.parquet"):
                        rel_path = file.relative_to(agent_dir)
                        view_name = str(rel_path.with_suffix('')).replace('/', '_').replace('\\', '_')
                        view_name = view_name.replace('-', '_').replace('.', '_')
                        conn.execute(f"CREATE OR REPLACE VIEW \"{view_name}\" AS SELECT * FROM read_parquet('{str(file)}')")
                        created_views.add(view_name)

                        alias = file.stem.replace('-', '_').replace('.', '_')
                        if alias and alias != view_name and alias not in created_views:
                            conn.execute(f"CREATE OR REPLACE VIEW \"{alias}\" AS SELECT * FROM \"{view_name}\"")
                            created_views.add(alias)
            else:
                # Fallback: register all folders (legacy behavior)
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

