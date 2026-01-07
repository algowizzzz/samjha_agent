"""
Query Excel Data Tool - Execute queries on Excel data using pandas.
"""

from pathlib import Path
import pandas as pd
from tools.base_mcp_tool import BaseMCPTool


class QueryExcelDataTool(BaseMCPTool):
    def __init__(self, config=None):
        default_config = {
            "name": "query_excel_data",
            "description": "Execute data operations (filter, group, aggregate, etc.) on Excel data",
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
            "required": ["file_path", "sheet_name", "operation"],
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to Excel file relative to datawarehouse"
                },
                "sheet_name": {
                    "type": "string",
                    "description": "Sheet name to query"
                },
                "operation": {
                    "type": "string",
                    "enum": ["filter", "group_by", "aggregate", "sort", "select_columns", "head", "tail", "describe"],
                    "description": "Operation to perform"
                },
                "filters": {
                    "type": "object",
                    "description": "Filter conditions (for filter operation)"
                },
                "group_by_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to group by (for group_by/aggregate)"
                },
                "aggregations": {
                    "type": "object",
                    "description": "Aggregation functions (e.g., {'revenue': 'sum', 'count': 'count'})"
                },
                "sort_by": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to sort by (for sort operation)"
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to select (for select_columns)"
                },
                "n": {
                    "type": "integer",
                    "description": "Number of rows (for head/tail)"
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Maximum rows to return",
                    "default": 1000
                },
                "agent_data_folder": {
                    "type": "string",
                    "description": "Optional agent-specific data folder"
                }
            }
        }

    def get_output_schema(self):
        return {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "description": "Query results as rows"
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Column names"
                },
                "row_count": {
                    "type": "integer",
                    "description": "Number of rows returned"
                },
                "operation": {
                    "type": "string",
                    "description": "Operation performed"
                }
            }
        }

    def execute(self, arguments):
        file_path = arguments["file_path"]
        sheet_name = arguments["sheet_name"]
        operation = arguments["operation"]
        max_rows = arguments.get("max_rows", 1000)
        agent_data_folder = arguments.get("agent_data_folder")
        
        # Resolve file path
        if agent_data_folder:
            full_path = self.base_path / agent_data_folder / file_path
        else:
            full_path = self.base_path / file_path
        
        if not full_path.exists():
            return {"error": f"File not found: {full_path}"}
        
        try:
            # Read Excel sheet
            df = pd.read_excel(full_path, sheet_name=sheet_name)
            
            # Apply operation
            if operation == "filter":
                # Simple filter: expects filters dict like {"column": "value"} or {"column": {"op": "value"}}
                filters = arguments.get("filters", {})
                for col, condition in filters.items():
                    if isinstance(condition, dict):
                        op = condition.get("op", "==")
                        val = condition.get("value")
                        if op == "==":
                            df = df[df[col] == val]
                        elif op == "!=":
                            df = df[df[col] != val]
                        elif op == ">":
                            df = df[df[col] > val]
                        elif op == "<":
                            df = df[df[col] < val]
                        elif op == ">=":
                            df = df[df[col] >= val]
                        elif op == "<=":
                            df = df[df[col] <= val]
                    else:
                        df = df[df[col] == condition]
            
            elif operation == "group_by":
                group_cols = arguments.get("group_by_columns", [])
                if group_cols:
                    df = df.groupby(group_cols)
                aggregations = arguments.get("aggregations", {})
                if aggregations:
                    df = df.agg(aggregations).reset_index()
            
            elif operation == "aggregate":
                group_cols = arguments.get("group_by_columns", [])
                aggregations = arguments.get("aggregations", {})
                if group_cols:
                    df = df.groupby(group_cols).agg(aggregations).reset_index()
                else:
                    df = df.agg(aggregations).reset_index()
            
            elif operation == "sort":
                sort_cols = arguments.get("sort_by", [])
                if sort_cols:
                    df = df.sort_values(by=sort_cols, ascending=True)
            
            elif operation == "select_columns":
                cols = arguments.get("columns", [])
                if cols:
                    df = df[cols]
            
            elif operation == "head":
                n = arguments.get("n", 10)
                df = df.head(n)
            
            elif operation == "tail":
                n = arguments.get("n", 10)
                df = df.tail(n)
            
            elif operation == "describe":
                df = df.describe()
            
            # Limit rows
            if len(df) > max_rows:
                df = df.head(max_rows)
            
            # Convert to JSON-serializable format
            data = df.fillna("").to_dict("records")
            columns = list(df.columns)
            
            return {
                "data": data,
                "columns": columns,
                "row_count": len(data),
                "operation": operation
            }
        
        except Exception as e:
            return {"error": f"Error querying Excel data: {str(e)}"}

