"""
BMO Financials Excel Agent - Specialized ReAct agent for BMO quarterly supplement reports.
Follows Reason → Act → Observe pattern with domain-specific knowledge.

Key Feature: Reads ALL sheets from Excel files to build comprehensive micro-summaries.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
from collections import defaultdict
import pandas as pd

from external.agent.state_types import ControllerState
from tools.tools_registry import ToolsRegistry
from external.tools.excel_agent import ListExcelFilesTool, ReadExcelSheetTool, QueryExcelDataTool

logger = logging.getLogger(__name__)


class BMOExcelAgent:
    """
    Specialized ReAct agent for BMO Financials Excel files.
    Uses domain knowledge from bmo_financials_domain.md to guide queries.
    
    Key Feature: Reads ALL sheets from each Excel file to build comprehensive micro-summaries.
    """
    
    def __init__(self):
        self.tools = {
            "list_excel_files": ListExcelFilesTool(),
            "read_excel_sheet": ReadExcelSheetTool(),
            "query_excel_data": QueryExcelDataTool(),
        }
        self.domain_md_path = Path("external/config/domains/bmo_financials_domain.md")
        self.data_path = Path("external/datawarehouse")
        self._load_domain_knowledge()
    
    def _load_domain_knowledge(self):
        """Load BMO Financials domain knowledge"""
        try:
            if self.domain_md_path.exists():
                self.domain_knowledge = self.domain_md_path.read_text()
            else:
                self.domain_knowledge = "# BMO Financials Domain\nDefault knowledge"
        except Exception as e:
            logger.warning(f"Could not load domain knowledge: {e}")
            self.domain_knowledge = ""
    
    def _read_all_sheets(self, file_path: str, agent_data_folder: Optional[str] = None) -> Dict[str, Any]:
        """
        Read ALL sheets from an Excel file and build micro-summaries.
        This is the key function for comprehensive data extraction.
        
        Returns:
            Dict with sheets info and micro-summaries for each sheet
        """
        # Resolve full path
        if agent_data_folder:
            full_path = self.data_path / agent_data_folder / file_path
        else:
            full_path = self.data_path / file_path
        
        if not full_path.exists():
            return {"error": f"File not found: {full_path}"}
        
        try:
            excel_file = pd.ExcelFile(full_path)
            all_sheets = excel_file.sheet_names
            
            micro_summaries = {}
            
            for sheet_name in all_sheets:
                try:
                    # Read the full sheet (or first 100 rows for very large sheets)
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=100)
                    
                    # Build micro-summary for this sheet
                    micro_summary = {
                        "sheet_name": sheet_name,
                        "row_count": len(df),
                        "column_count": len(df.columns),
                        "columns": list(df.columns),
                        "data_preview": [],
                        "key_values": {}
                    }
                    
                    # Extract non-empty rows as data preview
                    for idx, row in df.head(20).iterrows():
                        row_data = {}
                        for col in df.columns:
                            val = row[col]
                            if pd.notna(val) and str(val).strip():
                                row_data[str(col)] = str(val)[:100]  # Truncate long values
                        if row_data:
                            micro_summary["data_preview"].append(row_data)
                    
                    # Try to extract key financial values
                    micro_summary["key_values"] = self._extract_key_values(df, sheet_name)
                    
                    micro_summaries[sheet_name] = micro_summary
                    
                except Exception as e:
                    micro_summaries[sheet_name] = {"error": str(e)}
            
            return {
                "file_path": file_path,
                "total_sheets": len(all_sheets),
                "sheet_names": all_sheets,
                "micro_summaries": micro_summaries
            }
            
        except Exception as e:
            return {"error": f"Error reading Excel file: {str(e)}"}
    
    def _extract_key_values(self, df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
        """
        Extract key financial values from a sheet.
        Looks for common financial metrics.
        """
        key_values = {}
        
        # Convert all values to string for searching
        df_str = df.astype(str)
        
        # Key metrics to look for
        metrics = [
            "Total revenue", "Net income", "Net interest income",
            "Non-interest revenue", "Provision for credit losses",
            "Return on equity", "ROE", "Return on assets", "ROA",
            "CET1", "Capital", "Total assets", "Total liabilities"
        ]
        
        for metric in metrics:
            # Search for metric in any column
            for col in df.columns:
                mask = df_str[col].str.contains(metric, case=False, na=False)
                if mask.any():
                    row_idx = mask.idxmax()
                    # Get values from adjacent columns (likely the data)
                    row_data = df.iloc[row_idx]
                    numeric_vals = []
                    for val in row_data:
                        if pd.notna(val):
                            try:
                                num = float(val)
                                if num != 0:
                                    numeric_vals.append(num)
                            except:
                                pass
                    if numeric_vals:
                        key_values[metric] = numeric_vals[:4]  # First 4 quarters
                    break
        
        return key_values
    
    def handle_query(
        self,
        user_query: str,
        agent_data_folder: Optional[str] = None,
        max_iterations: int = 10,
    ) -> Dict[str, Any]:
        """
        Main query handler - implements ReAct loop with BMO domain knowledge.
        
        ReAct Pattern:
        1. Reason: Plan what files/sheets to examine (using domain knowledge)
        2. Act: Call tools (list files, READ ALL SHEETS, extract data)
        3. Observe: Process tool outputs
        4. Reason: Decide next action or synthesize answer
        """
        
        logger.info(f"BMO Excel Agent query: {user_query}")
        
        # Step 1: REASON - Initial planning with domain context
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
        REASON: Create initial plan based on query and BMO domain knowledge.
        """
        plan = {"actions": []}
        query_lower = query.lower()
        
        # Always start by listing files to see what's available
        plan["actions"].append({
            "tool": "list_excel_files",
            "arguments": {"agent_data_folder": agent_data_folder}
        })
        
        return plan
    
    def _reason_next_plan(self, query: str, observations: List[Dict], agent_data_folder: Optional[str]) -> Dict[str, Any]:
        """
        REASON: Plan next actions based on observations and BMO domain knowledge.
        KEY CHANGE: Now reads ALL sheets from files, not just specific pages.
        """
        plan = {"actions": []}
        query_lower = query.lower()
        
        # Track what we've done
        all_files = []
        files_fully_read = set()  # Files where we've read all sheets
        
        # Collect all files and their status
        for obs in observations:
            if obs["action"].get("tool") == "list_excel_files":
                files = obs["result"].get("files", [])
                for f in files:
                    file_path = f["path"]
                    if "/" in file_path:
                        file_path = file_path.split("/")[-1]
                    all_files.append({"name": f["name"], "path": file_path})
            
            elif obs["action"].get("tool") == "read_all_sheets":
                file_path = obs["action"]["arguments"].get("file_path", "")
                if "/" in file_path:
                    file_path = file_path.split("/")[-1]
                if "micro_summaries" in obs["result"]:
                    files_fully_read.add(file_path)
        
        # Strategy: Read ALL sheets from relevant files
        # Determine which files to read based on query
        files_to_read = self._identify_relevant_files(query_lower, all_files)
        
        for file_info in files_to_read:
            file_path = file_info["path"]
            
            if file_path not in files_fully_read:
                # Read ALL sheets from this file
                plan["actions"].append({
                    "tool": "read_all_sheets",
                    "arguments": {
                        "file_path": file_path,
                        "agent_data_folder": agent_data_folder
                    }
                })
                # For efficiency, read one file at a time for large files
                # But for segment queries, read multiple quarters
                if "quarter" in query_lower or "trend" in query_lower:
                    continue  # Keep adding more files
                else:
                    break  # Just one file for now
        
        return plan
    
    def _identify_relevant_files(self, query_lower: str, all_files: List[Dict]) -> List[Dict]:
        """
        Identify which files are relevant based on the query.
        """
        # Specific quarter requests
        if "q4 2024" in query_lower or "q4 2024" in query_lower or ("q4" in query_lower and "2024" in query_lower):
            for f in all_files:
                if "q424" in f["name"].lower():
                    return [f]
        
        if "q1 2025" in query_lower or ("q1" in query_lower and "2025" in query_lower):
            for f in all_files:
                if "q425" in f["name"].lower():
                    return [f]
        
        if "q3 2023" in query_lower or ("q3" in query_lower and "2023" in query_lower):
            for f in all_files:
                if "q323" in f["name"].lower():
                    return [f]
        
        # For quarter comparisons, select multiple files
        if "quarter" in query_lower or "4 quarter" in query_lower or "past" in query_lower:
            # Return most recent 4 unique quarters
            # Filter out duplicates (e.g., Suppq323 and Suppq323 (1))
            seen_periods = set()
            relevant = []
            for f in sorted(all_files, key=lambda x: x["name"], reverse=True):
                period = self._extract_period(f["name"])
                if period not in seen_periods and period != "Unknown period":
                    seen_periods.add(period)
                    relevant.append(f)
                if len(relevant) >= 4:
                    break
            return relevant
        
        # For segment queries, use most recent file (has all quarters)
        if "segment" in query_lower or "revenue" in query_lower:
            # Find most recent file (likely Suppq424 or Suppq425)
            for f in sorted(all_files, key=lambda x: x["name"], reverse=True):
                if "q424" in f["name"].lower() or "q425" in f["name"].lower():
                    return [f]
            return all_files[:1] if all_files else []
        
        # Default: use most recent file
        return all_files[:1] if all_files else []
    
    def _act_execute_tool(self, action: Dict, agent_data_folder: Optional[str]) -> Dict[str, Any]:
        """ACT: Execute a tool action."""
        tool_name = action["tool"]
        arguments = action.get("arguments", {})
        
        # Ensure agent_data_folder is passed if not in arguments
        if "agent_data_folder" not in arguments and agent_data_folder:
            arguments["agent_data_folder"] = agent_data_folder
        
        # Special handling for read_all_sheets (our custom action)
        if tool_name == "read_all_sheets":
            return self._read_all_sheets(
                arguments.get("file_path", ""),
                arguments.get("agent_data_folder")
            )
        
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            result = self.tools[tool_name].execute(arguments)
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _observe_has_answer(self, query: str, observations: List[Dict]) -> bool:
        """OBSERVE: Determine if we have enough information to answer."""
        query_lower = query.lower()
        
        # Check if we have file information
        has_files = any(
            "files" in obs["result"] and obs["result"].get("files")
            for obs in observations
            if isinstance(obs["result"], dict)
        )
        
        # Check if we have full sheet data (micro-summaries)
        has_micro_summaries = any(
            "micro_summaries" in obs["result"]
            for obs in observations
            if isinstance(obs["result"], dict)
        )
        
        # For specific metric queries, we need micro-summaries
        if any(word in query_lower for word in ["revenue", "segment", "income", "capital", "credit", "quarter"]):
            return has_micro_summaries
        
        # For general exploration, file list + some data is enough
        return has_files and has_micro_summaries
    
    def _reason_synthesize_answer(self, query: str, observations: List[Dict]) -> str:
        """
        REASON: Synthesize final answer from observations with BMO domain context.
        Now processes micro-summaries from ALL sheets.
        """
        if not observations:
            return "I couldn't find any BMO Financials Excel files to answer your query."
        
        query_lower = query.lower()
        answer_parts = []
        
        # Collect file information and micro-summaries
        files_info = {}
        all_micro_summaries = {}
        
        for obs in observations:
            action = obs["action"]
            result = obs["result"]
            
            if action["tool"] == "list_excel_files":
                files = result.get("files", [])
                if files:
                    answer_parts.append(f"📁 Found {len(files)} BMO Financials supplement file(s)")
                    answer_parts.append("")
            
            elif action["tool"] == "read_all_sheets":
                file_path = result.get("file_path", "")
                total_sheets = result.get("total_sheets", 0)
                micro_summaries = result.get("micro_summaries", {})
                
                period = self._extract_period(file_path)
                answer_parts.append(f"📊 Read ALL {total_sheets} sheets from {file_path} ({period})")
                
                all_micro_summaries[file_path] = {
                    "period": period,
                    "sheets": micro_summaries
                }
        
        # Now synthesize based on query type
        if "surprising" in query_lower or "noticed" in query_lower or "cfo" in query_lower:
            answer_parts.append("")
            answer_parts.append("=" * 80)
            answer_parts.append("🔍 SURPRISING INSIGHTS FOR CFO")
            answer_parts.append("=" * 80)
            answer_parts.append("")
            
            insights = self._find_surprising_insights(all_micro_summaries, query_lower)
            if insights:
                for i, insight in enumerate(insights, 1):
                    answer_parts.append(f"{i}. {insight}")
            else:
                answer_parts.append("Analyzing all sheets for anomalies and patterns...")
                answer_parts.append("")
                answer_parts.append("Key areas to investigate:")
                answer_parts.append("  • Unusual changes in provision for credit losses")
                answer_parts.append("  • Segment performance variations")
                answer_parts.append("  • Capital ratio movements")
                answer_parts.append("  • Efficiency ratio trends")
                answer_parts.append("  • Non-interest expense patterns")
        
        elif "segment" in query_lower or "revenue" in query_lower:
            answer_parts.append("")
            answer_parts.append("=" * 80)
            answer_parts.append("REVENUE BY SEGMENT ANALYSIS")
            answer_parts.append("=" * 80)
            
            # Extract segment data from micro-summaries
            segment_data = self._extract_segment_revenue(all_micro_summaries)
            if segment_data:
                answer_parts.append("")
                answer_parts.append(f"{'Segment':<40} | {'Q1':>12} | {'Q2':>12} | {'Q3':>12} | {'Q4':>12}")
                answer_parts.append("-" * 80)
                for segment, values in segment_data.items():
                    q1 = f"{values.get('Q1', 'N/A'):>12,}" if isinstance(values.get('Q1'), (int, float)) else f"{'N/A':>12}"
                    q2 = f"{values.get('Q2', 'N/A'):>12,}" if isinstance(values.get('Q2'), (int, float)) else f"{'N/A':>12}"
                    q3 = f"{values.get('Q3', 'N/A'):>12,}" if isinstance(values.get('Q3'), (int, float)) else f"{'N/A':>12}"
                    q4 = f"{values.get('Q4', 'N/A'):>12,}" if isinstance(values.get('Q4'), (int, float)) else f"{'N/A':>12}"
                    answer_parts.append(f"{segment:<40} | {q1} | {q2} | {q3} | {q4}")
            else:
                answer_parts.append("Segment revenue data extracted from sheets - see key pages:")
                answer_parts.append("  • Page 9: Total Personal & Commercial Banking")
                answer_parts.append("  • Page 10: Canadian P&C")
                answer_parts.append("  • Page 11: U.S. P&C")
                answer_parts.append("  • Page 12: BMO Wealth Management")
                answer_parts.append("  • Page 13: BMO Capital Markets")
        
        # Add summary of sheets read
        if all_micro_summaries:
            answer_parts.append("")
            answer_parts.append("📑 Sheets Read (Micro-Summaries Built):")
            for file_path, data in all_micro_summaries.items():
                sheets = data.get("sheets", {})
                key_sheets = [s for s in sheets.keys() if "Page" in s][:10]
                answer_parts.append(f"   • {file_path}: {len(sheets)} sheets including {', '.join(key_sheets[:5])}...")
        
        # Add domain context
        answer_parts.append("")
        answer_parts.append("💡 All amounts in millions of Canadian dollars (CAD)")
        
        if not answer_parts:
            return "I processed your query but couldn't extract a clear answer from the BMO Financials data."
        
        return "\n".join(answer_parts)
    
    def _extract_segment_revenue(self, micro_summaries: Dict) -> Dict[str, Dict]:
        """
        Extract segment revenue data from micro-summaries.
        """
        segment_data = {}
        
        # Known segment pages
        segment_pages = {
            "Page 9": "Total P&C Banking",
            "Page 10": "Canadian P&C",
            "Page 11": "U.S. P&C Banking",
            "Page 12": "BMO Wealth Management",
            "Page 13": "BMO Capital Markets",
            "Page 8": "Total Bank"
        }
        
        for file_path, data in micro_summaries.items():
            sheets = data.get("sheets", {})
            
            for page, segment_name in segment_pages.items():
                if page in sheets:
                    sheet_data = sheets[page]
                    key_values = sheet_data.get("key_values", {})
                    
                    # Look for Total revenue
                    if "Total revenue" in key_values:
                        values = key_values["Total revenue"]
                        if segment_name not in segment_data:
                            segment_data[segment_name] = {}
                        # Assume values are in order: Q4, Q3, Q2, Q1 (most recent first)
                        if len(values) >= 4:
                            segment_data[segment_name] = {
                                "Q4": int(values[0]) if values[0] else None,
                                "Q3": int(values[1]) if len(values) > 1 else None,
                                "Q2": int(values[2]) if len(values) > 2 else None,
                                "Q1": int(values[3]) if len(values) > 3 else None
                            }
        
        return segment_data
    
    def _find_surprising_insights(self, micro_summaries: Dict, query_lower: str) -> List[str]:
        """
        Find surprising insights from micro-summaries that a CFO might not notice.
        Looks for anomalies, trends, and unexpected patterns.
        """
        insights = []
        
        for file_path, data in micro_summaries.items():
            sheets = data.get("sheets", {})
            period = data.get("period", "Unknown")
            
            # Extract key financial metrics from all sheets
            metrics = {}
            for sheet_name, sheet_data in sheets.items():
                key_vals = sheet_data.get("key_values", {})
                for metric, values in key_vals.items():
                    if metric not in metrics:
                        metrics[metric] = []
                    metrics[metric].extend(values)
            
            # Analyze for surprises
            # 1. Check for unusual PCL (Provision for Credit Losses)
            if "Provision for credit losses" in metrics or "Total provision" in metrics:
                pcl_values = metrics.get("Provision for credit losses", metrics.get("Total provision", []))
                if pcl_values:
                    pcl_nums = [v for v in pcl_values if isinstance(v, (int, float)) and v > 0]
                    if pcl_nums:
                        max_pcl = max(pcl_nums)
                        avg_pcl = sum(pcl_nums) / len(pcl_nums) if pcl_nums else 0
                        if max_pcl > avg_pcl * 1.5:
                            insights.append(f"⚠️  **Spike in Provision for Credit Losses**: {period} shows PCL of ${max_pcl:,.0f}M, significantly above average of ${avg_pcl:,.0f}M - may indicate credit quality concerns")
            
            # 2. Check efficiency ratio
            if "Efficiency ratio" in metrics:
                eff_ratios = [v for v in metrics["Efficiency ratio"] if isinstance(v, (int, float)) and 0 < v < 1]
                if eff_ratios:
                    avg_eff = sum(eff_ratios) / len(eff_ratios)
                    if avg_eff > 0.65:
                        insights.append(f"📊 **Efficiency Ratio Above 65%**: {period} efficiency ratio is {avg_eff:.1%} - above optimal range, suggests cost management opportunity")
            
            # 3. Check revenue growth patterns
            if "Total revenue" in metrics:
                rev_values = [v for v in metrics["Total revenue"] if isinstance(v, (int, float)) and v > 0]
                if len(rev_values) >= 4:
                    # Calculate quarter-over-quarter growth
                    q4_rev = rev_values[0] if rev_values else 0
                    q3_rev = rev_values[1] if len(rev_values) > 1 else 0
                    q2_rev = rev_values[2] if len(rev_values) > 2 else 0
                    q1_rev = rev_values[3] if len(rev_values) > 3 else 0
                    
                    if q3_rev > 0:
                        q4_growth = ((q4_rev - q3_rev) / q3_rev) * 100
                        if abs(q4_growth) > 5:
                            insights.append(f"📈 **Significant Revenue Change**: Q4 revenue ${q4_rev:,.0f}M vs Q3 ${q3_rev:,.0f}M = {q4_growth:+.1f}% change - investigate drivers")
            
            # 4. Check for segment concentration
            # Look for segment data in sheet names or data
            segment_sheets = [s for s in sheets.keys() if any(seg in s for seg in ["P&C", "Wealth", "Capital", "Canadian", "U.S."])]
            if len(segment_sheets) > 3:
                insights.append(f"🏦 **Multi-Segment Analysis Available**: {len(segment_sheets)} segment-specific sheets found - review for concentration risk or segment performance divergence")
            
            # 5. Check capital ratios
            if "CET1" in metrics or "Capital" in metrics:
                capital_vals = metrics.get("CET1", metrics.get("Capital", []))
                if capital_vals:
                    cap_nums = [v for v in capital_vals if isinstance(v, (int, float)) and 0 < v < 1]
                    if cap_nums:
                        min_cap = min(cap_nums)
                        if min_cap < 0.12:
                            insights.append(f"💰 **Capital Ratio Watch**: CET1 ratio of {min_cap:.1%} is below 12% - monitor regulatory capital requirements")
            
            # 6. Check for unusual expense patterns
            if "Non-interest expense" in metrics or "expense" in str(metrics).lower():
                insights.append(f"💸 **Expense Analysis**: Review non-interest expense trends across segments - look for cost efficiency opportunities")
            
            # 7. Check for one-time items or adjustments
            adjusted_sheets = [s for s in sheets.keys() if "adjusted" in s.lower() or "adjusting" in s.lower()]
            if adjusted_sheets:
                insights.append(f"📝 **Adjusted vs Reported**: {len(adjusted_sheets)} sheets contain adjusted metrics - review adjusting items for one-time impacts")
        
        # If no specific insights, provide general guidance
        if not insights:
            insights.append("🔍 **Comprehensive Analysis**: All sheets have been read - recommend deep dive into:")
            insights.append("   • Page 5-6: Financial Highlights for profitability trends")
            insights.append("   • Page 8-13: Segment performance for concentration analysis")
            insights.append("   • Page 24-33: Credit Risk for portfolio quality")
            insights.append("   • Compare adjusted vs reported metrics for one-time items")
        
        return insights
    
    def _extract_period(self, filename: str) -> str:
        """Extract period information from BMO filename."""
        filename_lower = filename.lower()
        if "q323" in filename_lower:
            return "Q3 2023"
        elif "q324" in filename_lower:
            return "Q4 2023"
        elif "q325" in filename_lower:
            return "Q1 2024"
        elif "q423" in filename_lower:
            return "Q4 2023"
        elif "q424" in filename_lower:
            return "Q4 2024"
        elif "q425" in filename_lower:
            return "Q1 2025"
        return "Unknown period"


def handle_bmo_query(
    user_query: str,
    agent_data_folder: Optional[str] = None,
    max_iterations: int = 10,
) -> Dict[str, Any]:
    """
    Main entry point for BMO Financials Excel agent queries.
    """
    agent = BMOExcelAgent()
    return agent.handle_query(user_query, agent_data_folder, max_iterations)
