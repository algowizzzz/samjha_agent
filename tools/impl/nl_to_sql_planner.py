"""
NL to SQL Planner Tool
Converts a natural language query and schema hints into a structured SQL plan.
Uses LLM when available, falls back to heuristics.
"""
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from tools.base_mcp_tool import BaseMCPTool
import json

try:
    import duckdb  # type: ignore
except ImportError:
    duckdb = None

try:
    from agent.llm_client import get_llm_client
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class NLToSQLPlannerTool(BaseMCPTool):
    def __init__(self, config: Dict = None):
        default_config = {
            "name": "nl_to_sql_planner",
            "description": "Convert natural language query to SQL with confidence score and explanation",
            "version": "1.0.0",
            "enabled": True,
            "use_llm": True,  # Set to False to force heuristics
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
        self.preview_rows = self.config.get("preview_rows", 100)
        self.data_directory = self.config.get("data_directory", "data/duckdb")
        self.db_path = f"{self.data_directory}/duckdb_analytics.db"
        self.use_llm = self.config.get("use_llm", True) and LLM_AVAILABLE
        
        # Initialize LLM if available
        self.llm_client = None
        if self.use_llm:
            try:
                self.llm_client = get_llm_client()
                if not self.llm_client.is_available():
                    print("⚠ LLM not available, using heuristics for NL-to-SQL")
                    self.use_llm = False
            except Exception as e:
                print(f"⚠ Failed to initialize LLM for planner: {e}")
                self.use_llm = False

    def get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query"},
                "table_schema": {"type": "object", "description": "Table schema metadata"},
                "docs_meta": {"type": "array", "description": "Optional doc metadata"},
                "previous_clarifications": {"type": "array", "description": "Past clarifications"},
            },
            "required": ["query"],
        }

    def get_output_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "plan": {"type": "object"},
                "plan_quality": {"type": "string"},
                "plan_explain": {"type": "string"},
                "clarification_questions": {"type": "array"},
                "timestamp": {"type": "string"},
            },
            "required": ["plan", "plan_quality", "plan_explain"],
        }

    def _list_tables(self) -> List[str]:
        if duckdb is None:
            return []
        try:
            conn = duckdb.connect(self.db_path, read_only=True)
            # Get both tables and views
            tables = [r[0] for r in conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()]
            views = [r[0] for r in conn.execute("SELECT table_name FROM information_schema.views WHERE table_schema = 'main'").fetchall()]
            conn.close()
            all_tables = list(set(tables + views))
            print(f"[NLToSQLPlanner] Found tables/views: {all_tables}")  # Debug
            return all_tables
        except Exception as e:
            print(f"[NLToSQLPlanner] Error listing tables: {e}")  # Debug
            return []

    def _simple_guess(self, nl_query: str, available_tables: List[str]) -> Dict[str, Any]:
        """
        Improved heuristic with better keyword matching
        """
        text = nl_query.lower()
        chosen = None
        
        # Keyword matching for common queries
        if "sales" in text or "revenue" in text or "sold" in text:
            for t in available_tables:
                if "sales" in t.lower():
                    chosen = t
                    break
        elif "customer" in text or "client" in text:
            for t in available_tables:
                if "customer" in t.lower():
                    chosen = t
                    break
        elif "inventory" in text or "stock" in text or "product" in text:
            for t in available_tables:
                if "inventory" in t.lower():
                    chosen = t
                    break
        
        # Fallback: check if table name is mentioned
        if not chosen:
            for t in available_tables:
                if t.lower() in text:
                    chosen = t
                    break
        
        # Last resort: use first available table
        if not chosen and available_tables:
            chosen = available_tables[0]
        
        # Build SQL with LIMIT
        if chosen:
            if "top" in text or "limit" in text:
                # Extract number if mentioned
                import re
                numbers = re.findall(r'\d+', text)
                limit = int(numbers[0]) if numbers else 10
                sql = f"SELECT * FROM {chosen} LIMIT {limit}"
            elif "total" in text or "sum" in text:
                sql = f"SELECT SUM(*) as total FROM {chosen} LIMIT 1"
            else:
                sql = f"SELECT * FROM {chosen} LIMIT {min(self.preview_rows, 10)}"
        else:
            sql = "SELECT 1"
        
        return {
            "type": "sql_plan",
            "sql": sql,
            "limits": {"preview_rows": self.preview_rows},
            "target_table": chosen,
        }

    def _llm_plan(self, nl_query: str, table_schema: Dict[str, Any], available_tables: List[str], docs_meta: List[Dict[str, Any]] = None, stream_callback: Optional[Callable[[str], None]] = None, conversation_history: str = "No previous conversation history.", previous_clarifications: List[str] = None) -> Dict[str, Any]:
        """Use LLM to generate SQL plan with three-stage approach:
        Stage 1: Classify query type (knowledge vs. data)
        Stage 2a: If knowledge, return answer from business glossary
        Stage 2b: If data query, clarification gate
        Stage 2c: If clear, generate SQL
        
        Args:
            conversation_history: Formatted conversation history for context
        """
        # Get prompts from config if available, otherwise use defaults
        from agent.config import QueryAgentConfig
        cfg = QueryAgentConfig()
        
        # =========================================================================
        # STAGE 1: CLASSIFY QUERY TYPE (knowledge vs. data)
        # =========================================================================
        print(f"[NLToSQLPlanner] STAGE 1: Classifying query type for: {nl_query}")
        
        # Build context for classifier (same as planner gets)
        schema_summary_for_classifier = "Available tables/views:\n"
        for table_name, schema_info in table_schema.items():
            cols = schema_info.get('columns', [])
            if cols:
                col_str = ', '.join([f"{c['name']}" for c in cols[:15]])
                schema_summary_for_classifier += f"\n{table_name}:\n"
                schema_summary_for_classifier += f"  Columns: {col_str}\n"
        
        if not table_schema:
            schema_summary_for_classifier = f"Available tables/views: {', '.join(available_tables)}\n"
        
        business_context_for_classifier = ""
        if docs_meta:
            for doc in docs_meta:
                if doc.get("type") == "business_glossary":
                    business_context_for_classifier += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    business_context_for_classifier += "📚 BUSINESS GLOSSARY:\n"
                    business_context_for_classifier += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    for term, definition in doc.get("glossary", {}).items():
                        business_context_for_classifier += f"  • {term}: {definition}\n"
                elif "table" in doc:
                    # Include table-level business context
                    table_name = doc.get("table", "")
                    table_context = doc.get("business_context", "")
                    if table_context:
                        business_context_for_classifier += f"\n{table_name}: {table_context}\n"
        
        # Also load full data dictionary context for comprehensive understanding
        try:
            import os
            data_dict_path = "config/data_dictionary_risk.json"
            if not os.path.exists(data_dict_path):
                data_dict_path = "config/data_dictionary.json"
            
            if os.path.exists(data_dict_path):
                with open(data_dict_path, 'r', encoding='utf-8') as f:
                    data_dict = json.load(f)
                    
                    # Add table relationships if available
                    if "table_relationships" in data_dict:
                        business_context_for_classifier += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        business_context_for_classifier += "🔗 TABLE RELATIONSHIPS:\n"
                        business_context_for_classifier += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        business_context_for_classifier += data_dict.get("table_relationships", "") + "\n"
                    
                    # Add business rules summary if available
                    if "business_rules" in data_dict:
                        business_context_for_classifier += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        business_context_for_classifier += "📐 BUSINESS RULES:\n"
                        business_context_for_classifier += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        business_context_for_classifier += data_dict.get("business_rules", "") + "\n"
        except Exception as e:
            print(f"[NLToSQLPlanner] Could not load full data dictionary context: {e}")
        
        # Get classifier prompts
        classifier_system = cfg.get_nested("prompts", "classifier_system", default="Classify as knowledge or data query.")
        classifier_user_template = cfg.get_nested("prompts", "classifier_user_template", default="User Query: {nl_query}\n\nClassify this query.")
        classifier_user_prompt = classifier_user_template.format(
            nl_query=nl_query,
            schema_summary=schema_summary_for_classifier,
            business_context=business_context_for_classifier,
            conversation_history=conversation_history
        )
        
        # Call LLM to classify
        try:
            classification_response = self.llm_client.invoke_with_prompt(
                classifier_system, 
                classifier_user_prompt, 
                response_format="json"
            )
            classification = json.loads(classification_response)
            query_type = classification.get("type", "data").lower()
            print(f"[NLToSQLPlanner] Classification result: {query_type} (confidence: {classification.get('confidence', 'unknown')})")
            print(f"[NLToSQLPlanner] Reasoning: {classification.get('reasoning', 'N/A')}")
        except Exception as e:
            print(f"[NLToSQLPlanner] Classification failed: {e}, defaulting to 'data' query")
            query_type = "data"
        
        # =========================================================================
        # STAGE 2a: If KNOWLEDGE QUESTION, generate answer from business glossary
        # =========================================================================
        if query_type == "knowledge":
            print(f"[NLToSQLPlanner] STAGE 2a: Generating knowledge answer")
            
            # Extract business glossary for knowledge answer
            business_glossary_text = ""
            if docs_meta:
                for doc in docs_meta:
                    if doc.get("type") == "business_glossary":
                        for term, definition in doc.get("glossary", {}).items():
                            business_glossary_text += f"{term}: {definition}\n"
            
            # Also load from data_dictionary
            try:
                import os
                data_dict_path = "config/data_dictionary_risk.json"
                if not os.path.exists(data_dict_path):
                    data_dict_path = "config/data_dictionary.json"
                
                if os.path.exists(data_dict_path):
                    with open(data_dict_path, 'r', encoding='utf-8') as f:
                        data_dict = json.load(f)
                        glossary = data_dict.get("business_glossary", {})
                        for term, term_data in glossary.items():
                            if isinstance(term_data, dict):
                                definition = term_data.get("definition", "")
                                usage = term_data.get("usage", "")
                                business_glossary_text += f"{term}: {definition}\n"
                                if usage:
                                    business_glossary_text += f"  Usage: {usage}\n"
            except Exception as e:
                print(f"[NLToSQLPlanner] Error loading business glossary: {e}")
            
            # Get knowledge answer prompts
            knowledge_system = cfg.get_nested("prompts", "knowledge_answer_system", default="Provide definitions from business glossary.")
            knowledge_user_template = cfg.get_nested("prompts", "knowledge_answer_user_template", default="User Query: {nl_query}\n\nBusiness Glossary:\n{business_glossary}\n\nProvide the definition.")
            knowledge_user_prompt = knowledge_user_template.format(
                nl_query=nl_query,
                business_glossary=business_glossary_text
            )
            
            # Call LLM to generate knowledge answer
            try:
                # Use streaming if callback provided, otherwise use regular invoke
                if stream_callback:
                    # Stream and accumulate full response
                    knowledge_answer = ""
                    for chunk in self.llm_client.stream_with_prompt(
                        knowledge_system, knowledge_user_prompt, callback=stream_callback
                    ):
                        knowledge_answer += chunk
                else:
                    knowledge_answer = self.llm_client.invoke_with_prompt(
                        knowledge_system,
                        knowledge_user_prompt
                    )
                
                print(f"[NLToSQLPlanner] Knowledge answer generated: {knowledge_answer[:100]}...")
                
                # Return as a special plan type
                return {
                    "type": "sql_plan",
                    "sql": "-- KNOWLEDGE QUESTION",
                    "plan_quality": "high",
                    "plan_explain": knowledge_answer,  # Note: use plan_explain (not plan_explanation) to match node expectations
  "clarification_questions": [],
                    "target_table": "none"
                }
            except Exception as e:
                print(f"[NLToSQLPlanner] Error generating knowledge answer: {e}")
                # Fall through to data query generation
        
        # =========================================================================
        # STAGE 2b: CLARIFICATION GATE - Check if query is clear enough
        # =========================================================================
        print(f"[NLToSQLPlanner] STAGE 2b: Checking if clarification needed")
        
        # Build previous clarifications context
        # Priority: Use parameter if provided (from replan_node), otherwise extract from query string
        previous_clarifications_text = ""
        if previous_clarifications:
            # Use the clarification questions that were asked (from state)
            previous_clarifications_text = "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            previous_clarifications_text += "❓ PREVIOUS CLARIFICATION QUESTIONS ASKED:\n"
            previous_clarifications_text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            for i, question in enumerate(previous_clarifications, 1):
                previous_clarifications_text += f"{i}. {question}\n"
            previous_clarifications_text += "\n📝 USER'S RESPONSES:\n"
            # Extract user responses from combined query
            if "\n\nClarification" in nl_query:
                parts = nl_query.split("\n\nClarification")
                if len(parts) > 1:
                    for i, part in enumerate(parts[1:], 1):
                        user_response = part.split(":", 1)[-1].strip() if ":" in part else part.strip()
                        previous_clarifications_text += f"{i}. {user_response}\n"
            previous_clarifications_text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        elif "\n\nClarification" in nl_query:
            # Fallback: Extract clarifications from combined query string
            parts = nl_query.split("\n\nClarification")
            if len(parts) > 1:
                previous_clarifications_text = "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                previous_clarifications_text += "📝 PREVIOUS CLARIFICATIONS PROVIDED:\n"
                previous_clarifications_text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                for i, part in enumerate(parts[1:], 1):
                    clarif = part.split(":", 1)[-1].strip() if ":" in part else part.strip()
                    previous_clarifications_text += f"{i}. {clarif}\n"
                previous_clarifications_text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        # Load procedural knowledge from data dictionary
        procedural_knowledge_text = ""
        try:
            import os
            data_dict_path = "config/data_dictionary_risk.json"
            if not os.path.exists(data_dict_path):
                data_dict_path = "config/data_dictionary.json"
            
            if os.path.exists(data_dict_path):
                with open(data_dict_path, 'r', encoding='utf-8') as f:
                    data_dict = json.load(f)
                    procedural_knowledge_text = data_dict.get("procedural_knowledge", "")
                    if procedural_knowledge_text:
                        procedural_knowledge_text = f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n📋 PROCEDURAL KNOWLEDGE:\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n{procedural_knowledge_text}\n"
        except Exception as e:
            print(f"[NLToSQLPlanner] Could not load procedural knowledge: {e}")
        
        # Reuse the context we built for classifier
        gate_system = cfg.get_nested("prompts", "clarification_gate_system", default="Assess if query is clear.")
        gate_user_template = cfg.get_nested("prompts", "clarification_gate_user_template", default="User Query: {nl_query}\n\nIs this clear?")
        gate_user_prompt = gate_user_template.format(
            nl_query=nl_query,
            previous_clarifications=previous_clarifications_text,
            schema_summary=schema_summary_for_classifier,
            business_context=business_context_for_classifier,
            procedural_knowledge=procedural_knowledge_text,
            conversation_history=conversation_history
        )
        
        try:
            # Call clarification gate
            if stream_callback:
                gate_response = ""
                for chunk in self.llm_client.stream_with_prompt(
                    gate_system, gate_user_prompt, response_format="json",
                    callback=stream_callback
                ):
                    gate_response += chunk
            else:
                gate_response = self.llm_client.invoke_with_prompt(
                    gate_system, gate_user_prompt, response_format="json"
                )
            
            gate_response = gate_response.strip()
            if gate_response.startswith('```'):
                lines = gate_response.split('\n')
                gate_response = '\n'.join(lines[1:-1]) if len(lines) > 2 else gate_response
            
            # Clean up potential control characters before JSON parsing
            import re
            gate_response_clean = re.sub(r'[\x00-\x1f\x7f]', '', gate_response)
            
            try:
                gate_result = json.loads(gate_response_clean)
            except json.JSONDecodeError as json_err:
                print(f"[NLToSQLPlanner] JSON parse error: {json_err}")
                print(f"[NLToSQLPlanner] Response snippet: {gate_response_clean[:500]}...")
                raise
            needs_clarification = gate_result.get("needs_clarification", False)
            
            print(f"[NLToSQLPlanner] Clarification gate: needs_clarification={needs_clarification}")
            
            if needs_clarification:
                questions = gate_result.get("questions", [])
                reasoning = gate_result.get("reasoning", "Query is ambiguous")
                
                print(f"[NLToSQLPlanner] Clarification needed: {reasoning}")
                print(f"[NLToSQLPlanner] Questions extracted from gate_result: {questions}")
                print(f"[NLToSQLPlanner] Full gate_result: {gate_result}")
                
                # Return early - don't generate SQL yet
                return {
                    "type": "clarification_needed",
                    "sql": None,  # No SQL generated yet
                    "plan_quality": "medium",  # Indicate needs clarification
                    "plan_explain": reasoning,
                    "clarification_questions": questions,
                    "target_table": None
                }
        except Exception as e:
            print(f"[NLToSQLPlanner] Clarification gate failed: {e}, proceeding to SQL generation")
            # Fall through to SQL generation on error
        
        # =========================================================================
        # STAGE 2c: SQL GENERATION - Generate SQL for clear query
        # =========================================================================
        print(f"[NLToSQLPlanner] STAGE 2c: Generating SQL for clear data query")
        
        # Use simpler SQL generator prompts (not the complex planner prompts)
        sql_system = cfg.get_nested("prompts", "sql_generator_system", default="Generate SQL for clear query.")

        # Build schema summary with business context
        schema_summary = "Available tables/views:\n"
        for table_name, schema_info in table_schema.items():
            cols = schema_info.get('columns', [])
            if cols:
                col_str = ', '.join([f"{c['name']}" for c in cols[:15]])  # Show column names
                schema_summary += f"\n{table_name}:\n"
                schema_summary += f"  Columns: {col_str}\n"
            else:
                schema_summary += f"- {table_name}\n"
        
        if not table_schema:
            schema_summary = f"Available tables/views: {', '.join(available_tables)}\n"
        
        # Add business context from docs_meta
        business_context = ""
        if docs_meta:
            for doc in docs_meta:
                if doc.get("type") == "business_glossary":
                    business_context += "\nBusiness Glossary (important!):\n"
                    for term, definition in doc.get("glossary", {}).items():
                        business_context += f"  - {term}: {definition}\n"
                elif "table" in doc:
                    business_context += f"\n{doc['table']} - {doc.get('business_context', '')}\n"
        
        # Load procedural knowledge from data_dictionary
        procedural_knowledge = ""
        try:
            import os
            # Try risk dictionary first, fall back to standard dictionary
            data_dict_path = "config/data_dictionary_risk.json"
            if not os.path.exists(data_dict_path):
                data_dict_path = "config/data_dictionary.json"
            
            if os.path.exists(data_dict_path):
                with open(data_dict_path, 'r', encoding='utf-8') as f:
                    data_dict = json.load(f)
                    procedural_knowledge = data_dict.get("procedural_knowledge", "")
        except Exception:
            pass  # Optional, graceful failure
        
        print(f"[NLToSQLPlanner] Schema summary for LLM:\n{schema_summary}")  # Debug

        # Get user prompt template for SQL generator (simpler than planner)
        sql_user_template = cfg.get_nested("prompts", "sql_generator_user_template", default="User Query: {nl_query}\n\nSchema:\n{schema_summary}\n\nGenerate SQL.")
        sql_user_prompt = sql_user_template.format(
            nl_query=nl_query,
            schema_summary=schema_summary,
            business_context=business_context,
            conversation_history=conversation_history
        )

        try:
            # Use streaming if callback provided, otherwise use regular invoke
            if stream_callback:
                # Stream and accumulate full response
                sql_response = ""
                for chunk in self.llm_client.stream_with_prompt(
                    sql_system, sql_user_prompt, response_format="json", callback=stream_callback
                ):
                    sql_response += chunk
            else:
                sql_response = self.llm_client.invoke_with_prompt(sql_system, sql_user_prompt, response_format="json")
            
            # Parse JSON from response (simpler structure than before)
            sql_response = sql_response.strip()
            if sql_response.startswith('```'):
                # Extract JSON from code block
                lines = sql_response.split('\n')
                sql_response = '\n'.join(lines[1:-1]) if len(lines) > 2 else sql_response
            
            sql_result = json.loads(sql_response)
            
            # Post-process SQL to fix common LLM mistakes
            sql = sql_result.get("sql", "SELECT 1")
            # Fix "SELECT TOP N" -> "SELECT" (remove TOP clause, rely on LIMIT at end)
            import re
            sql = re.sub(r'\bSELECT\s+TOP\s+\d+\s+', 'SELECT ', sql, flags=re.IGNORECASE)
            
            # Return with simpler structure (no quality assessment - we already passed gate)
            return {
                "type": "sql_plan",
                "sql": sql,
                "plan_quality": "high",  # Assume high since we passed clarification gate
                "plan_explain": sql_result.get("explanation", "SQL generated for clear query"),
                "clarification_questions": [],
                "target_table": sql_result.get("target_table")
            }
        except json.JSONDecodeError as e:
            print(f"⚠ SQL generator returned invalid JSON: {e}")
            print(f"  Response: {sql_response[:200]}")
            # Fall back to heuristic
            return self._simple_guess(nl_query, available_tables)
        except Exception as e:
            print(f"⚠ LLM planning failed: {e}")
            return self._simple_guess(nl_query, available_tables)

    def execute(self, arguments: Dict[str, Any], stream_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        nl_query: str = arguments["query"]
        table_schema: Dict[str, Any] = arguments.get("table_schema", {})
        
        # Prefer table_schema from arguments (avoids DuckDB lock issues)
        # Fall back to querying DuckDB only if schema not provided
        if table_schema:
            available_tables = list(table_schema.keys())
        else:
            available_tables = self._list_tables()
        
        # Use LLM if available, otherwise heuristics
        if self.use_llm and self.llm_client:
            print(f"[NLToSQLPlanner] Using LLM to plan query")
            conversation_history = arguments.get("conversation_history", "No previous conversation history.")
            previous_clarifications = arguments.get("previous_clarifications", [])
            llm_result = self._llm_plan(nl_query, table_schema, available_tables, arguments.get("docs_meta"), stream_callback=stream_callback, conversation_history=conversation_history, previous_clarifications=previous_clarifications)
            
            # Check if clarification is needed (Stage 2b returned early)
            if llm_result.get("type") == "clarification_needed":
                # Pass through clarification request directly
                return {
                    "type": llm_result.get("type"),
                    "plan": {"sql": None},  # No SQL yet
                    "plan_quality": llm_result.get("plan_quality", "medium"),
                    "plan_explain": llm_result.get("plan_explain", ""),
                    "clarification_questions": llm_result.get("clarification_questions", []),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
            
            # Otherwise, SQL was generated successfully
            plan = {
                "type": "sql_plan",
                "sql": llm_result.get("sql", "SELECT 1"),
                "limits": {"preview_rows": self.preview_rows},
                "target_table": llm_result.get("target_table"),
            }
            plan_quality = llm_result.get("plan_quality", "high")  # Default to high if gate passed
            # Check both plan_explain and plan_explanation for backward compatibility
            plan_explain = llm_result.get("plan_explain") or llm_result.get("plan_explanation", "SQL generated")
            clarify = llm_result.get("clarification_questions", [])
        else:
            print(f"[NLToSQLPlanner] Using heuristics to plan query")
            plan = self._simple_guess(nl_query, available_tables)
            plan_quality = "low" if plan.get("target_table") is None else "medium"
            plan_explain = "Heuristic plan based on detected table names and default preview limit."
            clarify = []
            if plan_quality == "low":
                clarify.append("Which table should I use?")
        
        return {
            "plan": plan,
            "plan_quality": plan_quality,
            "plan_explain": plan_explain,
            "clarification_questions": clarify,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


