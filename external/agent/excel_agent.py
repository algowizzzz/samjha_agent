"""
Excel Agent - Simple ReAct-style agent for querying Excel files.
Follows Reason → Act → Observe pattern.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

from external.agent.state_types import ControllerState
from tools.tools_registry import ToolsRegistry
from external.tools.excel_agent import ListExcelFilesTool, ReadExcelSheetTool, QueryExcelDataTool

logger = logging.getLogger(__name__)


class ExcelAgent:
    """
    Simple ReAct agent for Excel files.
    Pattern: Reason (plan) → Act (tools) → Observe (results) → Reason (synthesize)
    """
    
    def __init__(self):
        self.tools = {
            "list_excel_files": ListExcelFilesTool(),
            "read_excel_sheet": ReadExcelSheetTool(),
            "query_excel_data": QueryExcelDataTool(),
        }
    
    def handle_query(
        self,
        user_query: str,
        agent_data_folder: Optional[str] = None,
        max_iterations: int = 10,  # Increased for multi-file queries
    ) -> Dict[str, Any]:
        """
        Main query handler - implements ReAct loop.
        
        ReAct Pattern:
        1. Reason: Plan what files/sheets to examine
        2. Act: Call tools (list files, read sheets, query data)
        3. Observe: Process tool outputs
        4. Reason: Decide next action or synthesize answer
        """
        
        logger.info(f"Excel Agent query: {user_query}")
        
        # Step 1: REASON - Initial planning
        plan = self._reason_initial_plan(user_query, agent_data_folder)
        logger.info(f"Initial plan: {plan}")
        
        observations = []
        iteration = 0
        
        # ReAct loop
        while iteration < max_iterations:
            iteration += 1
            
            # Step 2: ACT - Execute planned actions
            action_results = []
            for action in plan.get("actions", []):
                result = self._act_execute_tool(action, agent_data_folder)
                action_results.append(result)
                observations.append({
                    "action": action,
                    "result": result
                })
            
            # Step 3: OBSERVE - Check if we have enough information
            if self._observe_has_answer(user_query, observations):
                # Step 4: REASON - Final synthesis
                answer = self._reason_synthesize_answer(user_query, observations)
                return {
                    "status": "SUCCESS",
                    "answer": answer,
                    "observations": observations,
                    "iterations": iteration
                }
            
            # Step 4: REASON - Plan next actions
            plan = self._reason_next_plan(user_query, observations, agent_data_folder)
            if not plan.get("actions"):
                # No more actions to take
                break
        
        # Max iterations reached or no more actions
        final_answer = self._reason_synthesize_answer(user_query, observations)
        return {
            "status": "SUCCESS" if observations else "INCOMPLETE",
            "answer": final_answer,
            "observations": observations,
            "iterations": iteration
        }
    
    def _reason_initial_plan(self, query: str, agent_data_folder: Optional[str]) -> Dict[str, Any]:
        """
        REASON: Create initial plan based on query.
        In a full implementation, this would use an LLM.
        For simplicity, we do heuristic planning.
        """
        plan = {
            "actions": []
        }
        
        query_lower = query.lower()
        
        # If query mentions specific file/sheet, we'll discover files first
        if "list" in query_lower or "show" in query_lower or "what files" in query_lower:
            plan["actions"].append({
                "tool": "list_excel_files",
                "arguments": {"agent_data_folder": agent_data_folder}
            })
        else:
            # For queries, start by listing files to see what's available
            plan["actions"].append({
                "tool": "list_excel_files",
                "arguments": {"agent_data_folder": agent_data_folder}
            })
        
        return plan
    
    def _reason_next_plan(self, query: str, observations: List[Dict], agent_data_folder: Optional[str]) -> Dict[str, Any]:
        """
        REASON: Plan next actions based on observations.
        Now handles multiple files and reads all of them.
        """
        plan = {"actions": []}
        
        # Track what we've done
        all_files = []
        files_read_sheets = set()
        files_read_data = set()
        
        query_lower = query.lower()
        needs_all_files = "all" in query_lower or "each" in query_lower or "relationship" in query_lower or "connected" in query_lower
        
        # Collect all files and their status
        for obs in observations:
            if obs["action"].get("tool") == "list_excel_files":
                files = obs["result"].get("files", [])
                for f in files:
                    file_path = f["path"]
                    if "/" in file_path:
                        file_path = file_path.split("/")[-1]
                    all_files.append({"name": f["name"], "path": file_path})
            
            elif obs["action"].get("tool") == "read_excel_sheet":
                file_path = obs["action"]["arguments"].get("file_path")
                if "/" in file_path:
                    file_path = file_path.split("/")[-1]
                
                result = obs["result"]
                if "sheets" in result:
                    files_read_sheets.add(file_path)
                if "data" in result:
                    files_read_data.add(file_path)
        
        # Strategy: Read all files if query asks for details about each or relationships
        if needs_all_files and all_files:
            # For each file, ensure we have sheets info and data
            for file_info in all_files:
                file_path = file_info["path"]
                
                if file_path not in files_read_sheets:
                    # Read sheets first
                    plan["actions"].append({
                        "tool": "read_excel_sheet",
                        "arguments": {
                            "file_path": file_path,
                            "agent_data_folder": agent_data_folder
                        }
                    })
                elif file_path in files_read_sheets and file_path not in files_read_data:
                    # We have sheets, now read data
                    # Find which sheets this file has
                    for obs in observations:
                        if (obs["action"].get("tool") == "read_excel_sheet" and 
                            obs["action"]["arguments"].get("file_path", "").split("/")[-1] == file_path and
                            "sheets" in obs["result"]):
                            sheets = obs["result"].get("sheets", [])
                            if sheets:
                                plan["actions"].append({
                                    "tool": "read_excel_sheet",
                                    "arguments": {
                                        "file_path": file_path,
                                        "sheet_name": sheets[0],
                                        "agent_data_folder": agent_data_folder,
                                        "max_rows": 5  # Preview rows
                                    }
                                })
                            break
        
        # For single file queries or if no action yet, use old logic
        elif all_files and len(all_files) == 1:
            file_path = all_files[0]["path"]
            if file_path not in files_read_sheets:
                plan["actions"].append({
                    "tool": "read_excel_sheet",
                    "arguments": {
                        "file_path": file_path,
                        "agent_data_folder": agent_data_folder
                    }
                })
            elif file_path in files_read_sheets and file_path not in files_read_data:
                for obs in observations:
                    if (obs["action"].get("tool") == "read_excel_sheet" and 
                        obs["action"]["arguments"].get("file_path", "").split("/")[-1] == file_path and
                        "sheets" in obs["result"]):
                        sheets = obs["result"].get("sheets", [])
                        if sheets:
                            plan["actions"].append({
                                "tool": "read_excel_sheet",
                                "arguments": {
                                    "file_path": file_path,
                                    "sheet_name": sheets[0],
                                    "agent_data_folder": agent_data_folder,
                                    "max_rows": 10
                                }
                            })
                        break
        
        return plan
    
    def _act_execute_tool(self, action: Dict, agent_data_folder: Optional[str]) -> Dict[str, Any]:
        """
        ACT: Execute a tool action.
        """
        tool_name = action["tool"]
        arguments = action.get("arguments", {})
        
        # Ensure agent_data_folder is passed if not in arguments
        if "agent_data_folder" not in arguments and agent_data_folder:
            arguments["agent_data_folder"] = agent_data_folder
        
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            result = self.tools[tool_name].execute(arguments)
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _observe_has_answer(self, query: str, observations: List[Dict]) -> bool:
        """
        OBSERVE: Determine if we have enough information to answer.
        """
        query_lower = query.lower()
        
        # Count what we have
        all_files = []
        files_with_data = set()
        
        for obs in observations:
            if obs["action"].get("tool") == "list_excel_files":
                files = obs["result"].get("files", [])
                all_files = files
            elif obs["action"].get("tool") == "read_excel_sheet":
                if "data" in obs["result"]:
                    file_path = obs["action"]["arguments"].get("file_path", "")
                    if "/" in file_path:
                        file_path = file_path.split("/")[-1]
                    files_with_data.add(file_path)
        
        num_files = len(all_files)
        num_files_with_data = len(files_with_data)
        
        # Check if this is a complex analytical query (needs data operations)
        is_analytical_query = any(word in query_lower for word in [
            "which", "what", "show me", "calculate", "total", "sum", "find", "filter",
            "purchased", "spending", "spent", "customers who", "products that"
        ])
        
        # If query asks about all files/each file/relationships, we need data from all
        if "all" in query_lower or "each" in query_lower or "relationship" in query_lower or "connected" in query_lower:
            # Need to have read data from all files
            return num_files > 0 and num_files_with_data == num_files
        
        # For analytical queries, we need data but also might need to execute queries
        if is_analytical_query:
            # For now, if we have data from relevant files, we can synthesize
            # In full implementation, would use query_excel_data tool
            return num_files_with_data >= 2  # At least 2 files for multi-file analysis
        
        # If query asks to "show data" or "what data is available", we need actual data preview
        if "show" in query_lower or "what data" in query_lower or ("data" in query_lower and "available" in query_lower):
            # Need to have read actual data from at least one file
            return num_files_with_data > 0
        
        # For simple "list files" queries, file list is enough
        if "list" in query_lower and "files" in query_lower and "data" not in query_lower:
            return num_files > 0
        
        # Default: If we have data, we're done
        return num_files_with_data > 0
    
    def _reason_synthesize_answer(self, query: str, observations: List[Dict]) -> str:
        """
        REASON: Synthesize final answer from observations.
        Analyzes all files and identifies relationships.
        """
        if not observations:
            return "I couldn't find any Excel files or data to answer your query."
        
        query_lower = query.lower()
        needs_relationships = "relationship" in query_lower or "connected" in query_lower or "each" in query_lower
        
        # Collect all file information
        files_info = {}
        all_files_list = []
        
        for obs in observations:
            action = obs["action"]
            result = obs["result"]
            
            if action["tool"] == "list_excel_files":
                files = result.get("files", [])
                all_files_list = files
                answer_parts = [f"Found {len(files)} Excel file(s):"]
                for f in files:
                    file_name = f['name']
                    files_info[file_name] = {
                        "name": file_name,
                        "path": f["path"],
                        "size": f["size_bytes"],
                        "columns": [],
                        "row_count": 0,
                        "sample_data": []
                    }
                    answer_parts.append(f"  - {file_name} ({f['size_bytes']} bytes)")
            
            elif action["tool"] == "read_excel_sheet":
                file_path = action["arguments"].get("file_path", "")
                if "/" in file_path:
                    file_path = file_path.split("/")[-1]
                
                # Find file name from path
                file_name = None
                for name, info in files_info.items():
                    if info["path"].split("/")[-1] == file_path:
                        file_name = name
                        break
                if not file_name:
                    file_name = file_path
                
                if "data" in result:
                    columns = result.get("columns", [])
                    row_count = result.get("row_count", 0)
                    data = result.get("data", [])
                    
                    if file_name not in files_info:
                        files_info[file_name] = {"name": file_name, "path": file_path}
                    
                    files_info[file_name]["columns"] = columns
                    files_info[file_name]["row_count"] = row_count
                    files_info[file_name]["sample_data"] = data[:3]  # Store sample rows
        
        # Build comprehensive answer
        answer_parts = []
        
        if needs_relationships or len(files_info) > 1:
            # Detailed analysis with relationships
            answer_parts.append(f"Excel Files Overview ({len(files_info)} files):")
            answer_parts.append("")
            
            for file_name, info in sorted(files_info.items()):
                answer_parts.append(f"📊 {file_name}:")
                answer_parts.append(f"   • Rows: {info.get('row_count', 'N/A')}")
                answer_parts.append(f"   • Columns ({len(info.get('columns', []))}): {', '.join(info.get('columns', [])[:8])}")
                
                # Identify key columns (IDs, foreign keys)
                columns = info.get("columns", [])
                key_cols = [c for c in columns if "id" in c.lower() or "customer" in c.lower() or "product" in c.lower()]
                if key_cols:
                    answer_parts.append(f"   • Key columns: {', '.join(key_cols)}")
                answer_parts.append("")
            
            # Analyze relationships
            if len(files_info) > 1:
                answer_parts.append("🔗 Relationships:")
                # Find common columns across files
                all_columns = {}
                for file_name, info in files_info.items():
                    for col in info.get("columns", []):
                        if col not in all_columns:
                            all_columns[col] = []
                        all_columns[col].append(file_name)
                
                # Identify relationship columns (appear in multiple files)
                relationships = []
                for col, files in all_columns.items():
                    if len(files) > 1 and ("id" in col.lower() or col.lower() in ["customer_id", "product_id", "order_id"]):
                        relationships.append(f"   • '{col}' connects: {', '.join(files)}")
                
                if relationships:
                    answer_parts.extend(relationships)
                else:
                    # Try to infer relationships
                    for file_name, info in files_info.items():
                        cols = info.get("columns", [])
                        if "customer" in file_name.lower():
                            if any("customer_id" in c.lower() for c in cols):
                                answer_parts.append(f"   • Customer data (customer_id) can link to sales/orders")
                        if "product" in file_name.lower() or "inventory" in file_name.lower():
                            if any("product" in c.lower() for c in cols):
                                answer_parts.append(f"   • Product/Inventory data can link to sales via product names/IDs")
                        if "sales" in file_name.lower():
                            answer_parts.append(f"   • Sales data contains customer_id and product references for linking")
        else:
            # Simple answer for single file queries
            if all_files_list:
                answer_parts.append(f"Found {len(all_files_list)} Excel file(s):")
                for f in all_files_list[:5]:
                    answer_parts.append(f"  - {f['name']} ({f['size_bytes']} bytes)")
            
            for file_name, info in files_info.items():
                if info.get("columns"):
                    answer_parts.append(f"\n{file_name}:")
                    answer_parts.append(f"  Rows: {info['row_count']}, Columns: {', '.join(info['columns'][:10])}")
        
        if not answer_parts:
            return "I processed your query but couldn't extract a clear answer from the available data."
        
        return "\n".join(answer_parts)


def handle_excel_query(
    user_query: str,
    agent_data_folder: Optional[str] = None,
    max_iterations: int = 5,
) -> Dict[str, Any]:
    """
    Main entry point for Excel agent queries.
    """
    agent = ExcelAgent()
    return agent.handle_query(user_query, agent_data_folder, max_iterations)

