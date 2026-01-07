"""
Read Excel Sheet Tool - Read and preview Excel sheet data.
"""

from pathlib import Path
import pandas as pd
from tools.base_mcp_tool import BaseMCPTool


class ReadExcelSheetTool(BaseMCPTool):
    def __init__(self, config=None):
        default_config = {
            "name": "read_excel_sheet",
            "description": "Read an Excel file and list its sheets, or read a specific sheet's data",
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
            "required": ["file_path"],
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to Excel file relative to datawarehouse"
                },
                "sheet_name": {
                    "type": "string",
                    "description": "Optional sheet name. If not provided, returns list of available sheets."
                },
                "max_rows": {
                    "type": "integer",
                    "description": "Maximum rows to read (default: 100)",
                    "default": 100
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
                "sheets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of sheet names (when sheet_name not provided)"
                },
                "data": {
                    "type": "array",
                    "description": "Sheet data as rows (when sheet_name provided)"
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Column names"
                },
                "row_count": {
                    "type": "integer",
                    "description": "Total number of rows in sheet"
                },
                "preview_row_count": {
                    "type": "integer",
                    "description": "Number of rows in preview"
                }
            }
        }

    def execute(self, arguments):
        file_path = arguments["file_path"]
        sheet_name = arguments.get("sheet_name")
        max_rows = arguments.get("max_rows", 100)
        agent_data_folder = arguments.get("agent_data_folder")
        
        # Resolve file path
        if agent_data_folder:
            full_path = self.base_path / agent_data_folder / file_path
        else:
            full_path = self.base_path / file_path
        
        if not full_path.exists():
            return {"error": f"File not found: {full_path}"}
        
        try:
            excel_file = pd.ExcelFile(full_path)
            
            # If no sheet specified, return list of sheets
            if not sheet_name:
                return {
                    "sheets": excel_file.sheet_names,
                    "file_path": file_path
                }
            
            # Read specified sheet
            if sheet_name not in excel_file.sheet_names:
                return {"error": f"Sheet '{sheet_name}' not found. Available sheets: {excel_file.sheet_names}"}
            
            df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=max_rows)
            
            # Convert to list of dicts for JSON serialization
            data = df.fillna("").to_dict("records")
            columns = list(df.columns)
            
            # Get total row count (read full sheet just for count)
            df_full = pd.read_excel(excel_file, sheet_name=sheet_name)
            total_rows = len(df_full)
            
            return {
                "data": data,
                "columns": columns,
                "row_count": total_rows,
                "preview_row_count": len(data),
                "sheet_name": sheet_name,
                "file_path": file_path
            }
        except Exception as e:
            return {"error": f"Error reading Excel file: {str(e)}"}

