from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from external.model_doc.types import AgentConfig, AgentState
from external.model_doc.mcp_client import call_mcp
from external.model_doc.llm import (
    LLMNotAvailableError,
    summarize_file,
    generate_hierarchical_summary,
    create_documentation_outline,
    draft_section,
)

logger = logging.getLogger(__name__)

_EVENT_EMITTER: ContextVar[Optional[Callable[[str, Dict[str, Any]], None]]] = ContextVar(
    "model_doc_event_emitter", default=None
)

_DEFAULT_CONFIG_PATH = Path("external/config/agent/model_doc_agent.json")
_DEFAULT_CONFIG: AgentConfig = {
    "codebase": {
        "file_extensions": [".py"],
        "exclude_patterns": ["__pycache__", "*.pyc", ".git", ".venv", "venv", ".env"],
    },
    "llm": {
        "model": "claude-3-opus-20240229",
        "temperature": 0.2,
    },
    "template": {
        "template_id": "bmo_model_documentation",
        "default_path": "config/doc_review/outline_templates/",
    },
    "output": {
        "base_dir": "external/data/model_doc/output",
        "create_timestamped_dir": True,
    },
}


class ModelDocAgent:
    """LangGraph-style orchestration for the model documentation workflow."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or _DEFAULT_CONFIG_PATH
        self.config = self._load_config(self.config_path)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def run(
        self,
        initial_state: AgentState,
        event_emitter: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> AgentState:
        token = _EVENT_EMITTER.set(event_emitter)
        state = self._initialise_state(initial_state)

        self._emit_event(
            "status",
            {
                "status": "running",
                "message": "Model documentation workflow started",
            },
        )

        try:
            # Phase 1: Codebase Discovery (MCP Tools)
            state = self._node_list_codebase_files(state)
            state = self._node_read_code_files(state)  # Batch read all files
            state = self._node_parse_code_structure(state)  # Batch parse AST
            state = self._node_build_file_hierarchy(state)
            state = self._node_compute_code_stats(state)

            # Phase 2: Summarization (LLM Nodes)
            state = self._node_summarize_file_llm(state)  # Summarizes all files
            state = self._node_generate_hierarchical_summary_llm(state)

            # Phase 3: Outline Generation (LLM Node)
            state = self._node_create_documentation_outline_llm(state)

            # Phase 4: Content Generation (LLM Node)
            state = self._node_draft_section_llm(state)  # Drafts all sections

            # Phase 5: Assembly (MCP Tools)
            state = self._node_load_documentation_template(state)
            state = self._node_assemble_documentation(state)
            state = self._node_save_documentation(state)

            self._emit_event(
                "status",
                {
                    "status": "completed",
                    "message": "Model documentation workflow completed",
                },
            )
            return state
        except Exception as exc:
            self._emit_event(
                "log",
                {
                    "node": "workflow",
                    "message": str(exc),
                    "level": "error",
                },
            )
            self._emit_event(
                "status",
                {
                    "status": "error",
                    "message": str(exc),
                },
            )
            raise
        finally:
            _EVENT_EMITTER.reset(token)

    def build_config(self, overrides: Optional[Dict[str, Any]] = None) -> AgentConfig:
        cfg = deepcopy(self.config)
        if overrides:
            self._deep_update(cfg, overrides)
        return cfg  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_config(self, path: Path) -> AgentConfig:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    user_cfg = json.load(f)
            except Exception as exc:
                logger.error("Failed to load model doc config %s: %s", path, exc)
                user_cfg = {}
        else:
            user_cfg = {}
        cfg = deepcopy(_DEFAULT_CONFIG)
        self._deep_update(cfg, user_cfg)
        return cfg  # type: ignore[return-value]

    def _deep_update(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = self._deep_update(base.get(key, {}), value)  # type: ignore[arg-type]
            else:
                base[key] = value
        return base

    def _initialise_state(self, state: AgentState) -> AgentState:
        state.setdefault("config", deepcopy(self.config))
        state.setdefault("logs", [])
        state.setdefault("file_list", [])
        state.setdefault("file_summaries", {})
        state.setdefault("section_drafts", [])
        state.setdefault("phase_stats", {})
        state.setdefault("phase_reports", {})
        return state

    def _emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        emitter = _EVENT_EMITTER.get()
        if not emitter:
            return
        try:
            data = dict(payload)
            data.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
            emitter(event_type, data)
        except Exception:  # pragma: no cover - emitter side effects shouldn't crash agent
            logger.exception("ModelDocAgent event emitter failed for %s", event_type)

    def _log(self, state: AgentState, node: str, message: str, level: str = "info") -> None:
        entry = f"{node}: {message}"
        state.setdefault("logs", []).append(entry)
        state["last_node"] = node
        logger.debug("[ModelDocAgent] %s - %s", node, message)
        self._emit_event(
            "log",
            {
                "node": node,
                "message": message,
                "level": level,
            },
        )

    def _call_tool(self, state: AgentState, tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return call_mcp(tool_name, payload)
        except Exception as exc:
            error_message = f"{tool_name} failed: {exc}"
            state.setdefault("logs", []).append(error_message)
            self._emit_event(
                "log",
                {
                    "node": tool_name,
                    "message": str(exc),
                    "level": "error",
                },
            )
            raise

    # ------------------------------------------------------------------
    # Phase 1: Codebase Discovery (MCP Tools)
    # ------------------------------------------------------------------
    def _node_list_codebase_files(self, state: AgentState) -> AgentState:
        """List all Python files in the codebase."""
        codebase_path = state["codebase_path"]
        config = state.get("config", {})
        codebase_config = config.get("codebase", {})
        
        result = self._call_tool(
            state,
            "model_doc_list_files",
            {
                "codebase_path": codebase_path,
                "file_extensions": codebase_config.get("file_extensions", [".py"]),
                "exclude_patterns": codebase_config.get("exclude_patterns", ["__pycache__", "*.pyc", ".git"]),
            },
        )
        
        state["file_list"] = result.get("files", [])
        self._log(state, "list_codebase_files", f"found {len(state['file_list'])} files")
        return state

    def _node_read_code_files(self, state: AgentState) -> AgentState:
        """Read all files from the file list."""
        file_list = state.get("file_list", [])
        file_metadata_list = []
        
        for file_info in file_list:
            try:
                result = self._call_tool(
                    state,
                    "model_doc_read_file",
                    {
                        "file_path": file_info.get("file_path"),
                    },
                )
                
                file_metadata = {
                    "file_path": file_info.get("file_path"),
                    "relative_path": file_info.get("relative_path"),
                    "file_size": result.get("file_size", 0),
                    "line_count": result.get("line_count", 0),
                    "encoding": result.get("encoding"),
                    "content": result.get("content"),
                }
                file_metadata_list.append(file_metadata)
                
            except Exception as e:
                self._log(state, "read_code_file", f"Failed to read {file_info.get('file_path')}: {e}", "warning")
                continue
        
        # Store file contents for later use
        state["file_contents"] = {f["file_path"]: f.get("content", "") for f in file_metadata_list}
        state["file_list"] = file_metadata_list
        self._log(state, "read_code_files", f"read {len(file_metadata_list)} files")
        return state

    def _node_parse_code_structure(self, state: AgentState) -> AgentState:
        """Parse AST structure for all files."""
        file_list = state.get("file_list", [])
        file_contents = state.get("file_contents", {})
        ast_structures = {}
        
        for file_info in file_list:
            file_path = file_info.get("file_path")
            content = file_contents.get(file_path, "")
            
            if not content:
                continue
            
            try:
                result = self._call_tool(
                    state,
                    "model_doc_parse_ast",
                    {
                        "file_path": file_path,
                        "file_content": content,
                    },
                )
                
                ast_structures[file_path] = result
                file_info["ast_structure"] = result
                
            except Exception as e:
                self._log(state, "parse_code_structure", f"Failed to parse {file_path}: {e}", "warning")
                ast_structures[file_path] = {
                    "classes": [],
                    "functions": [],
                    "imports": [],
                    "ast_errors": [str(e)],
                }
                file_info["ast_structure"] = ast_structures[file_path]
        
        self._log(state, "parse_code_structure", f"parsed {len(ast_structures)} files")
        return state

    def _node_build_file_hierarchy(self, state: AgentState) -> AgentState:
        """Build directory hierarchy from file list."""
        file_list = state.get("file_list", [])
        codebase_path = state.get("codebase_path")
        
        if not file_list:
            self._log(state, "build_file_hierarchy", "no files to build hierarchy", "warning")
            return state
        
        result = self._call_tool(
            state,
            "model_doc_build_hierarchy",
            {
                "file_list": file_list,
                "codebase_path": codebase_path,
            },
        )
        
        state["file_hierarchy"] = result.get("hierarchy", {})
        state["modules"] = result.get("modules", [])
        state["packages"] = result.get("packages", [])
        
        # Build codebase metadata
        codebase_metadata = state.get("codebase_metadata", {}) or {}
        codebase_metadata.update({
            "codebase_path": codebase_path,
            "codebase_id": state.get("codebase_id"),
            "file_count": len(file_list),
            "directory_structure": result.get("hierarchy", {}),
        })
        state["codebase_metadata"] = codebase_metadata
        
        self._log(state, "build_file_hierarchy", f"modules={len(state['modules'])} packages={len(state['packages'])}")
        return state

    def _node_compute_code_stats(self, state: AgentState) -> AgentState:
        """Compute statistics for all files."""
        file_list = state.get("file_list", [])
        file_contents = state.get("file_contents", {})
        
        total_lines = 0
        total_classes = 0
        total_functions = 0
        total_methods = 0
        total_imports = 0
        
        for file_info in file_list:
            file_path = file_info.get("file_path")
            content = file_contents.get(file_path, "")
            ast_structure = file_info.get("ast_structure", {})
            
            if not content or not ast_structure:
                continue
            
            try:
                result = self._call_tool(
                    state,
                    "model_doc_compute_stats",
                    {
                        "file_content": content,
                        "ast_structure": ast_structure,
                    },
                )
                
                total_lines += result.get("code_lines", 0)
                total_classes += result.get("class_count", 0)
                total_functions += result.get("function_count", 0)
                total_methods += result.get("method_count", 0)
                total_imports += result.get("import_count", 0)
                
                file_info["stats"] = result
                
            except Exception as e:
                self._log(state, "compute_code_stats", f"Failed to compute stats for {file_path}: {e}", "warning")
        
        # Update codebase metadata with stats
        codebase_metadata = state.get("codebase_metadata", {}) or {}
        codebase_metadata.update({
            "total_lines": total_lines,
            "total_classes": total_classes,
            "total_functions": total_functions,
            "total_methods": total_methods,
            "total_imports": total_imports,
        })
        state["codebase_metadata"] = codebase_metadata
        state["code_stats"] = {
            "total_lines": total_lines,
            "total_classes": total_classes,
            "total_functions": total_functions,
            "total_methods": total_methods,
            "total_imports": total_imports,
        }
        
        self._log(state, "compute_code_stats", f"LOC={total_lines} classes={total_classes} functions={total_functions}")
        return state

    # ------------------------------------------------------------------
    # Phase 2: Summarization (LLM Nodes)
    # ------------------------------------------------------------------
    def _node_summarize_file_llm(self, state: AgentState) -> AgentState:
        """Generate summaries for all files using LLM."""
        file_list = state.get("file_list", [])
        file_contents = state.get("file_contents", {})
        file_summaries = {}
        
        for file_info in file_list:
            file_path = file_info.get("file_path")
            content = file_contents.get(file_path, "")
            ast_structure = file_info.get("ast_structure", {})
            
            if not content:
                continue
            
            try:
                summary = summarize_file(content, ast_structure)
                file_summaries[file_path] = summary
                file_info["summary"] = summary
                
                self._emit_event(
                    "progress",
                    {
                        "phase": "summarization",
                        "file": file_path,
                        "progress": len(file_summaries),
                        "total": len([f for f in file_list if file_contents.get(f.get("file_path"))]),
                    },
                )
                
            except LLMNotAvailableError:
                self._log(state, "summarize_file_llm", f"LLM unavailable, skipping {file_path}", "warning")
                file_summaries[file_path] = "Summary unavailable - LLM not configured."
            except Exception as e:
                self._log(state, "summarize_file_llm", f"Failed to summarize {file_path}: {e}", "warning")
                file_summaries[file_path] = f"Summary unavailable - error: {str(e)}"
        
        state["file_summaries"] = file_summaries
        self._log(state, "summarize_file_llm", f"summarized {len(file_summaries)} files")
        return state

    def _node_generate_hierarchical_summary_llm(self, state: AgentState) -> AgentState:
        """Generate hierarchical summary of the entire codebase."""
        file_summaries = state.get("file_summaries", {})
        file_hierarchy = state.get("file_hierarchy", {})
        
        if not file_summaries:
            self._log(state, "generate_hierarchical_summary_llm", "no file summaries available", "warning")
            return state
        
        try:
            hierarchical_summary = generate_hierarchical_summary(file_summaries, file_hierarchy)
            state["hierarchical_summary"] = hierarchical_summary
            self._log(state, "generate_hierarchical_summary_llm", "hierarchical summary generated")
        except LLMNotAvailableError:
            self._log(state, "generate_hierarchical_summary_llm", "LLM unavailable", "warning")
            state["hierarchical_summary"] = {
                "component_summary": "Hierarchical summary unavailable - LLM not configured.",
                "module_hierarchy": file_hierarchy,
                "key_modules": [],
                "key_classes": [],
                "dependencies": [],
            }
        except Exception as e:
            self._log(state, "generate_hierarchical_summary_llm", f"Error: {e}", "error")
            state["hierarchical_summary"] = {
                "component_summary": f"Error generating hierarchical summary: {str(e)}",
                "module_hierarchy": file_hierarchy,
                "key_modules": [],
                "key_classes": [],
                "dependencies": [],
            }
        
        return state

    # ------------------------------------------------------------------
    # Phase 3: Outline Generation (LLM Node)
    # ------------------------------------------------------------------
    def _node_create_documentation_outline_llm(self, state: AgentState) -> AgentState:
        """Create documentation outline based on template and codebase."""
        template = state.get("template")
        hierarchical_summary = state.get("hierarchical_summary", {})
        file_summaries = state.get("file_summaries", {})
        
        if not template:
            self._log(state, "create_documentation_outline_llm", "no template loaded yet", "warning")
            return state
        
        try:
            outline = create_documentation_outline(template, hierarchical_summary, file_summaries)
            state["documentation_outline"] = outline
            self._log(state, "create_documentation_outline_llm", f"outline created with {len(outline.get('section_mappings', []))} sections")
        except LLMNotAvailableError:
            self._log(state, "create_documentation_outline_llm", "LLM unavailable", "warning")
            state["documentation_outline"] = {
                "template_id": template.get("id", "unknown"),
                "template_structure": template,
                "section_mappings": [],
                "unmapped_sections": [],
                "outline_metadata": {},
            }
        except Exception as e:
            self._log(state, "create_documentation_outline_llm", f"Error: {e}", "error")
            state["documentation_outline"] = {
                "template_id": template.get("id", "unknown"),
                "template_structure": template,
                "section_mappings": [],
                "unmapped_sections": [],
                "outline_metadata": {},
            }
        
        return state

    # ------------------------------------------------------------------
    # Phase 4: Content Generation (LLM Node)
    # ------------------------------------------------------------------
    def _node_draft_section_llm(self, state: AgentState) -> AgentState:
        """Draft content for all sections in the outline."""
        outline = state.get("documentation_outline", {})
        template = state.get("template", {})
        file_summaries = state.get("file_summaries", {})
        section_mappings = outline.get("section_mappings", [])
        
        if not section_mappings:
            self._log(state, "draft_section_llm", "no sections to draft", "warning")
            return state
        
        section_drafts = []
        
        for idx, mapping in enumerate(section_mappings):
            section_id = mapping.get("section_id", f"section_{idx}")
            template_section_id = mapping.get("template_section_id")
            
            # Find relevant file summaries
            codebase_sections = mapping.get("codebase_sections", [])
            relevant_summaries = []
            for section_ref in codebase_sections:
                if section_ref in file_summaries:
                    relevant_summaries.append(file_summaries[section_ref])
            
            # Get template section guidance if available
            template_guidance = ""
            if template_section_id:
                template_sections = template.get("sections", [])
                template_section = next(
                    (s for s in template_sections if s.get("id") == template_section_id), None
                )
                if template_section:
                    template_guidance = template_section.get("description", "")
            
            outline_section = {
                "section_id": section_id,
                "heading": mapping.get("heading", section_id),
                "template_guidance": template_guidance,
            }
            
            try:
                content = draft_section(outline_section, relevant_summaries, template)
                section_drafts.append({
                    "section_id": section_id,
                    "template_section_id": template_section_id,
                    "heading": mapping.get("heading", section_id),
                    "content": content,
                    "order": idx,
                    "status": "drafted",
                })
                
                self._emit_event(
                    "progress",
                    {
                        "phase": "content_generation",
                        "section": section_id,
                        "progress": len(section_drafts),
                        "total": len(section_mappings),
                    },
                )
                
            except LLMNotAvailableError:
                self._log(state, "draft_section_llm", f"LLM unavailable for {section_id}", "warning")
                section_drafts.append({
                    "section_id": section_id,
                    "template_section_id": template_section_id,
                    "heading": mapping.get("heading", section_id),
                    "content": "Content unavailable - LLM not configured.",
                    "order": idx,
                    "status": "error",
                })
            except Exception as e:
                self._log(state, "draft_section_llm", f"Failed to draft {section_id}: {e}", "warning")
                section_drafts.append({
                    "section_id": section_id,
                    "template_section_id": template_section_id,
                    "heading": mapping.get("heading", section_id),
                    "content": f"Content unavailable - error: {str(e)}",
                    "order": idx,
                    "status": "error",
                })
        
        state["section_drafts"] = section_drafts
        self._log(state, "draft_section_llm", f"drafted {len(section_drafts)} sections")
        return state

    # ------------------------------------------------------------------
    # Phase 5: Assembly (MCP Tools)
    # ------------------------------------------------------------------
    def _node_load_documentation_template(self, state: AgentState) -> AgentState:
        """Load the documentation template."""
        config = state.get("config", {})
        template_config = config.get("template", {})
        template_id = template_config.get("template_id", "bmo_model_documentation")
        default_path = template_config.get("default_path", "config/doc_review/outline_templates/")
        
        result = self._call_tool(
            state,
            "model_doc_load_template",
            {
                "template_id": template_id,
                "default_path": default_path,
            },
        )
        
        state["template"] = result.get("template", {})
        self._log(state, "load_documentation_template", f"template_id={template_id}")
        return state

    def _node_assemble_documentation(self, state: AgentState) -> AgentState:
        """Assemble final markdown from sections."""
        template = state.get("template", {})
        section_drafts = state.get("section_drafts", [])
        codebase_metadata = state.get("codebase_metadata", {}) or {}
        
        metadata = {
            "title": f"{codebase_metadata.get('codebase_id', 'Model')} Documentation",
            "codebase_path": codebase_metadata.get("codebase_path"),
            "file_count": codebase_metadata.get("file_count", 0),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
        
        result = self._call_tool(
            state,
            "model_doc_assemble_doc",
            {
                "template": template,
                "section_drafts": section_drafts,
                "metadata": metadata,
            },
        )
        
        state["final_documentation"] = result.get("markdown", "")
        self._log(state, "assemble_documentation", f"sections={result.get('section_count', 0)}")
        return state

    def _node_save_documentation(self, state: AgentState) -> AgentState:
        """Save the final documentation to disk."""
        final_doc = state.get("final_documentation")
        if not final_doc:
            self._log(state, "save_documentation", "no documentation to save", "warning")
            return state
        
        config = state.get("config", {})
        output_config = config.get("output", {})
        codebase_id = state.get("codebase_id")
        
        result = self._call_tool(
            state,
            "model_doc_save_doc",
            {
                "markdown_content": final_doc,
                "output_dir": output_config.get("base_dir", "external/data/model_doc/output/"),
                "filename": "final_documentation.md",
                "create_timestamped_dir": output_config.get("create_timestamped_dir", True),
                "codebase_id": codebase_id,
            },
        )
        
        state["final_documentation_path"] = result.get("file_path")
        state["output_directory"] = result.get("output_directory")
        self._log(state, "save_documentation", f"saved to {result.get('file_path')}")
        return state

