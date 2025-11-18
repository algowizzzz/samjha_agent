from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from external.doc_review.llm import LLMNotAvailableError, call_llm_json
from external.doc_review.mcp_client import call_mcp
from external.doc_review.types import (
    AgentState,
    ChangeSelectionPlan,
    ChangesData,
    DocMeta,
    Phase1Data,
    Phase2Data,
    StructureData,
    TemplateMeta,
    UserInteractionState,
    VfsArtifact,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_AGENT_CONFIG_PATH = Path("external/config/agent/doc_review_agent.json")
TEMPLATE_DIR = Path("external/doc_review/templates")

NODE_LABELS: Dict[str, str] = {
    "phase0_ingestion": "Phase 0 – Ingestion",
    "phase1_doc_summary": "Phase 1 – Document Summary",
    "phase1_toc_review": "Phase 1 – TOC Review",
    "phase1_template_fitness": "Phase 1 – Template Fitness",
    "phase1_section_strategy": "Phase 1 – Section Strategy",
    "phase2_extract_sections": "Phase 2 – Extract Sections",
    "phase2_section_reviews": "Phase 2 – Section Reviews",
    "phase2_summary": "Phase 2 – Summary",
    "change_selection_intent": "Phase 3 – Interpret Instruction",
    "apply_changes": "Phase 3 – Apply Changes",
    "verify_changes": "Phase 3 – Verify Changes",
}

NODE_KINDS: Dict[str, str] = {
    "phase0_ingestion": "tool",
    "phase1_doc_summary": "llm",
    "phase1_toc_review": "llm",
    "phase1_template_fitness": "llm",
    "phase1_section_strategy": "llm",
    "phase2_extract_sections": "orchestrator",
    "phase2_section_reviews": "llm",
    "phase2_summary": "llm",
    "change_selection_intent": "llm",
    "apply_changes": "tool",
    "verify_changes": "llm",
}

NODE_TRANSITIONS: Dict[str, str] = {
    "phase0_ingestion": "phase1_doc_summary",
    "phase1_doc_summary": "phase1_toc_review",
    "phase1_toc_review": "phase1_template_fitness",
    "phase1_template_fitness": "phase1_section_strategy",
    "phase1_section_strategy": "await_section_strategy_confirmation",
    "phase2_extract_sections": "phase2_section_reviews",
    "phase2_section_reviews": "phase2_summary",
    "phase2_summary": "await_change_instruction",
    "change_selection_intent": "apply_changes",
    "apply_changes": "verify_changes",
    "verify_changes": "completed",
}

ORCHESTRATOR_WAIT_STATES: Set[str] = {
    "await_section_strategy_confirmation",
    "await_change_instruction",
    "completed",
    "failed",
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class DocReviewAgent:
    """
    Orchestrates deterministic ingestion (Phase 0) and Phase 1 scaffolding for the
    document review workflow. Subsequent phases (LLM nodes, section reviews, agent
    planner) build on the state initialised here.
    """

    def __init__(self, agent_config_path: Optional[Path] = None) -> None:
        self.agent_config_path = agent_config_path or DEFAULT_AGENT_CONFIG_PATH
        self.config = self._load_config(self.agent_config_path)
        self.logger = LOGGER
        self._prompt_cache: Dict[str, str] = {}
        self.node_registry = self._build_node_registry()
        self.state_lock_owner: Optional[str] = None
        self.event_emitter: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self.agent_log_root = Path("logs")

    def _build_node_registry(self) -> Dict[str, Callable[[AgentState], AgentState]]:
        return {
            "phase0_ingestion": self._run_phase0_ingestion,
            "phase1_doc_summary": self._node_phase1_doc_summary_llm,
            "phase1_toc_review": self._node_phase1_toc_review_llm,
            "phase1_template_fitness": self._node_phase1_template_fitness_llm,
            "phase1_section_strategy": self._node_phase1_section_strategy_llm,
            "phase2_extract_sections": self._node_phase2_extract_sections,
            "phase2_section_reviews": self._node_phase2_section_reviews,
            "phase2_summary": self._node_phase2_summary_llm,
            "change_selection_intent": self._node_change_selection_intent_orchestrator,
            "apply_changes": self._node_apply_changes_orchestrator,
            "verify_changes": self._node_verify_changes_orchestrator,
        }

    def set_event_emitter(self, emitter: Optional[Callable[[str, Dict[str, Any]], None]]) -> None:
        self.event_emitter = emitter

    def _acquire_run_lock(self, state: AgentState, session_id: str) -> None:
        owner = state.get("locked_by")
        if owner and owner != session_id:
            raise RuntimeError(f"Run locked by {owner}")
        if owner == session_id:
            return
        state["locked_by"] = session_id
        state["lock_timestamp"] = _utcnow_iso()

    def _release_run_lock(self, state: AgentState, session_id: str) -> None:
        if state.get("locked_by") == session_id:
            state["locked_by"] = None
            state["lock_timestamp"] = None

    def _append_agent_transcript_log(self, state: AgentState, entry: Dict[str, Any]) -> None:
        log_dir = self.agent_log_root / state["run_id"]
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "agent_transcript.jsonl"
        entry_with_ts = {"timestamp": _utcnow_iso(), **entry}
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry_with_ts, ensure_ascii=False) + "\n")

    def _sync_vfs_artifacts(self, state: AgentState, entries: List[Dict[str, str]]) -> None:
        timestamp = _utcnow_iso()
        for entry in entries:
            path = entry.get("path")
            if not path:
                continue
            artifact = VfsArtifact(
                path=path,
                label=entry.get("label", Path(path).name),
                last_updated=timestamp,
            )
            self._register_vfs_artifact(state, artifact)
            self._emit_event(
                state,
                "vfs_file_updated",
                {"path": path, "label": artifact["label"], "timestamp": timestamp},
            )

    def _run_node(
        self,
        node_id: str,
        func: Callable[[AgentState], AgentState],
        state: AgentState,
    ) -> AgentState:
        node_kind = NODE_KINDS.get(node_id, "orchestrator")
        label = NODE_LABELS.get(node_id, node_id)
        start_ts = time.time()
        self._emit_event(
            state,
            "node_started",
            {
                "node": node_id,
                "node_kind": node_kind,
                "label": label,
                "timestamp": _utcnow_iso(),
            },
        )

        new_state = state
        try:
            new_state = func(state)
            status = "success"
            error_msg = None
        except Exception as exc:  # pragma: no cover - safety
            status = "failed"
            error_msg = str(exc)
            state["errors"].append(f"{node_id} failed: {exc}")
            self.logger.exception("Node %s failed", node_id)
            raise
        finally:
            duration_ms = int((time.time() - start_ts) * 1000)
            summary = self._summarize_node_result(node_id, state if status == "failed" else new_state)
            self._emit_event(
                state,
                "node_completed",
                {
                    "node": node_id,
                    "node_kind": node_kind,
                    "label": label,
                    "status": status,
                    "duration_ms": duration_ms,
                    "summary": summary,
                    "error": error_msg,
                },
            )

        new_state["last_node"] = node_id
        return new_state

    def _summarize_node_result(self, node_id: str, state: AgentState) -> str:
        if node_id == "phase1_doc_summary" and state["phase1"].get("doc_summary"):
            return state["phase1"]["doc_summary"].get("summary", "")[:140]
        if node_id == "phase2_summary" and state["phase2"].get("summary_report"):
            return state["phase2"]["summary_report"].get("overall_posture", "")
        if node_id == "phase2_section_reviews":
            return f"{len(state['changes'].get('suggested_changes', []))} issues logged"
        if node_id == "apply_changes":
            applied = len(state["changes"].get("applied_change_ids", []))
            failed = len(state["changes"].get("failed_changes", []))
            return f"Applied {applied} changes ({failed} failed)"
        return ""

    def orchestrate(
        self,
        state: AgentState,
        stop_controls: Optional[Set[str]] = None,
    ) -> AgentState:
        stop_controls = stop_controls or ORCHESTRATOR_WAIT_STATES

        while True:
            control = state.get("control")
            if not control or control in stop_controls:
                break

            node_callable = self.node_registry.get(control)
            if not node_callable:
                self.logger.warning("No node registered for control '%s'", control)
                break

            state = self._run_node(control, node_callable, state)
            state = self._advance_control(control, state)

        return state

    def _advance_control(self, current_control: str, state: AgentState) -> AgentState:
        if state.get("control") != current_control:
            # Node already mutated control (e.g., failure/waiting state)
            return state
        next_control = NODE_TRANSITIONS.get(current_control)
        if next_control:
            state["control"] = next_control
        else:
            state["control"] = "completed"
        return state

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def _node_publish_raw_markdown(self, state: AgentState) -> AgentState:
        """
        Publish the raw markdown produced during ingestion so the UI can render it
        immediately after Phase 0 completes. Mirrors content to top-level and VFS.
        Also publishes block metadata and verification suggestions.
        """
        try:
            # Prefer the structured ingestion text; fall back to any prior raw_markdown
            raw_md = (
                state.get("structure", {}).get("raw_text")
                or state.get("raw_markdown")
                or ""
            )
            if not raw_md:
                return state

            # Mirror to top-level for easy access by routes/UI
            state["raw_markdown"] = raw_md
            
            # Also expose block metadata and verification suggestions
            state["block_metadata"] = state.get("structure", {}).get("block_metadata", [])
            state["verification_suggestions"] = state.get("structure", {}).get("verification_suggestions", [])

            # Ensure VFS exists and write a file the UI can fetch
            vfs = state.setdefault("vfs", {})
            files = vfs.setdefault("files", {})
            files["/raw.md"] = {
                "type": "file",
                "content": raw_md,
                "mime": "text/markdown",
            }
            files["/blocks.json"] = {
                "type": "file",
                "content": json.dumps(state["block_metadata"], indent=2),
                "mime": "application/json",
            }
            files["/suggestions.json"] = {
                "type": "file",
                "content": json.dumps(state["verification_suggestions"], indent=2),
                "mime": "application/json",
            }

            # Notify any websocket listeners that markdown is ready
            self._emit_event(
                state,
                "markdown_ready",
                {"path": "/raw.md", "bytes": len(raw_md)},
            )
        except Exception:  # pragma: no cover - non-critical publication step
            self.logger.exception("Failed to publish raw markdown to VFS/UI")
        return state

    def run_phase1(
        self,
        document_path: str,
        run_id: Optional[str] = None,
        template_id: Optional[str] = None,
    ) -> AgentState:
        """
        Execute Phase 0 ingestion only (converts document to markdown with block metadata).
        Note: React editor uses this for ingestion, then applies templates via TemplateProcessor.
        Phase 1/2/3 LLM workflows are deprecated for React editor.
        """

        state = self._initialise_state(
            source=Path(document_path).expanduser().resolve(),
            run_id=run_id,
            template_id=template_id,
        )

        self.logger.info(
            "DocReviewAgent.run_phase1 (Phase 0 only): run_id=%s doc_id=%s", state["run_id"], state["doc_id"]
        )

        state["phase1_status"] = "running"
        
        # Run Phase 0 ingestion only (no LLM orchestration)
        state = self._run_phase0_ingestion(state)
        
        # Make raw markdown available to the UI
        state = self._node_publish_raw_markdown(state)

        state["phase1_status"] = "success"
        state["control"] = "completed"

        return state

    def run_phase2(
        self,
        state: AgentState,
        section_scope: Optional[List[str]] = None,
    ) -> AgentState:
        self.logger.info("DocReviewAgent.run_phase2: run_id=%s", state["run_id"])
        state["phase2_status"] = "running"

        previous_scope = state["user_interaction"].get("selected_section_scope")
        if section_scope:
            state["user_interaction"]["selected_section_scope"] = section_scope

        state["control"] = "phase2_extract_sections"
        self.orchestrate(state, stop_controls={"await_change_instruction", "failed"})

        if section_scope is not None:
            state["user_interaction"]["selected_section_scope"] = previous_scope

        if state["phase2_status"] != "failed":
            state["phase2_status"] = "success"
        return state

    def run_phase3(
        self,
        state: AgentState,
        change_ids: Optional[List[str]] = None,
        severity_filter: Optional[str] = None,
    ) -> AgentState:
        self.logger.info("DocReviewAgent.run_phase3: run_id=%s", state["run_id"])
        state["phase3_status"] = "running"

        selected_changes = self._select_changes_for_application(
            state,
            change_ids=change_ids,
            severity_filter=severity_filter,
            plan=state["changes"].get("change_selection_plan"),
        )
        if not selected_changes:
            state["phase3_status"] = "failed"
            state["errors"].append("Phase 3: no changes selected for application")
            return state

        original_text = state["structure"]["raw_text"]

        state["control"] = "apply_changes"
        state = self._run_node(
            "apply_changes",
            lambda s: self._apply_changes_core(s, selected_changes, original_text),
            state,
        )
        state = self._advance_control("apply_changes", state)
        state = self._run_node("verify_changes", self._node_verify_changes_orchestrator, state)
        state = self._advance_control("verify_changes", state)
        return state

    def _apply_changes_core(
        self,
        state: AgentState,
        selected_changes: List[Dict[str, Any]],
        original_text: Optional[str] = None,
    ) -> AgentState:
        if not selected_changes:
            state["phase3_status"] = "failed"
            return state

        snapshot = original_text or state["structure"]["raw_text"]
        payload = {
            "raw_markdown": state["structure"]["raw_text"],
            "changes": selected_changes,
        }
        result = self._call_tool("apply_changes_deterministic", payload)
        state["changes"]["applied_change_ids"] = result.get("applied_change_ids", [])
        state["changes"]["failed_changes"] = result.get("failed_changes", [])
        state["changes"]["new_raw_text"] = result.get("new_raw_markdown")

        if state["changes"]["applied_change_ids"]:
            state["structure"]["raw_text"] = result["new_raw_markdown"]
            state["doc_meta"]["version"] = state["doc_meta"].get("version", 1) + 1
            state["phase3_status"] = "success"
            state["changes"]["_pre_apply_text"] = snapshot
            version_label = f"v{state['doc_meta']['version']}_document.md"
            self._sync_vfs_artifacts(
                state,
                [
                    {"path": f"/versions/{version_label}", "label": "Improved Document"},
                    {"path": "/changes/applied_changes.json", "label": "Applied Changes"},
                ],
            )
        else:
            state["phase3_status"] = "failed"
        return state

    def interpret_change_instruction(
        self, state: AgentState, user_instruction: str
    ) -> Optional[ChangeSelectionPlan]:
        """
        Use the change selection intent LLM to convert a natural language instruction
        (e.g., "apply 1,2,3,4" or "apply all high severity changes") into a structured plan.
        """

        if not user_instruction or not user_instruction.strip():
            self.logger.warning("Change instruction missing or empty")
            return None

        instruction = user_instruction.strip()
        state["user_interaction"]["user_change_instruction"] = instruction
        plan = self._node_change_selection_intent_llm(state, instruction)
        if plan:
            state["changes"]["change_selection_plan"] = plan
        return plan

    def handle_user_message(
        self,
        state: AgentState,
        user_message: str,
        auto_execute: bool = True,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entrypoint for autonomous agent mode.
        Interprets user's natural language command and executes the appropriate workflow.

        Args:
            state: Current agent state
            user_message: User's natural language command
            auto_execute: If True, automatically execute the plan. If False, return plan for confirmation.

        Returns:
            Dict with plan, execution results, and any errors
        """
        self.logger.info("Handling user message: %s", user_message)
        session = session_id or f"session-{uuid4().hex[:8]}"
        self._acquire_run_lock(state, session)
        start_time = time.time()
        response: Dict[str, Any] = {
            "status": "failed",
            "plan": None,
            "execution_results": None,
        }
        try:
            plan = self._node_agent_planner_llm(state, user_message)
            if not plan:
                response["error"] = "Failed to generate plan from user message"
                return response

            self._emit_event(
                state,
                "agent_plan_generated",
                {
                    "run_id": state["run_id"],
                    "summary": plan.get("summary"),
                    "requires_confirmation": plan.get("requires_confirmation", False),
                    "step_count": len(plan.get("plan_steps", [])),
                },
            )

            requires_confirmation = plan.get("requires_confirmation", False)
            if requires_confirmation or not auto_execute:
                response["status"] = "pending_confirmation"
                response["plan"] = plan
                return response

            execution_results = self._execute_agent_plan(state, plan)
            response.update(
                {
                    "status": "success",
                    "plan": plan,
                    "execution_results": execution_results,
                }
            )
            return response
        finally:
            duration_ms = int((time.time() - start_time) * 1000)
            log_entry = {
                "user_message": user_message,
                "status": response.get("status"),
                "plan_summary": (response.get("plan") or {}).get("summary"),
                "auto_execute": auto_execute,
                "duration_ms": duration_ms,
            }
            self._append_agent_transcript_log(state, log_entry)
            self._release_run_lock(state, session)

    # ------------------------------------------------------------------ #
    # Phase 0 helpers
    # ------------------------------------------------------------------ #
    def _run_phase0_ingestion(self, state: AgentState) -> AgentState:
        state = self._node_detect_file_type(state)
        state = self._node_convert_to_markdown(state)
        state = self._node_extract_images(state)
        state = self._node_compute_file_stats(state)
        state = self._node_analyze_heading_structure(state)
        state = self._node_extract_toc_llm(state)
        state = self._node_build_file_metadata(state)
        state = self._node_collect_phase1_stats(state)
        return state

    def _node_detect_file_type(self, state: AgentState) -> AgentState:
        payload = {
            "file_id": state["doc_id"],
            "source_path": state["doc_meta"]["source_path"],
        }
        result = self._call_tool("detect_file_type", payload)
        state["doc_meta"]["file_type"] = result.get("file_type", "unknown")
        self._emit_event(state, "node_completed", {"node": "detect_file_type", "result": result})
        return state

    def _node_convert_to_markdown(self, state: AgentState) -> AgentState:
        payload = {
            "file_id": state["doc_id"],
            "source_path": state["doc_meta"]["source_path"],
            "file_type": state["doc_meta"].get("file_type") or "unknown",
        }
        result = self._call_tool("convert_to_markdown", payload)
        state["doc_meta"]["md_file_id"] = result["md_file_id"]
        state["doc_meta"]["md_path"] = result["md_path"]
        state["structure"]["raw_text"] = result["raw_markdown"]
        
        # Store block metadata and verification suggestions
        state["structure"]["block_metadata"] = result.get("block_metadata", [])
        state["structure"]["verification_suggestions"] = result.get("verification_suggestions", [])
        self._register_vfs_artifact(
            state,
            VfsArtifact(
                path=f"/original/{Path(state['doc_meta']['source_path']).name}",
                label="Original upload",
                last_updated=_utcnow_iso(),
            ),
        )
        self._register_vfs_artifact(
            state,
            VfsArtifact(
                path="/original/document.md",
                label="Normalised Markdown",
                last_updated=_utcnow_iso(),
            ),
        )
        self._emit_event(state, "node_completed", {"node": "convert_to_markdown"})
        return state

    def _node_extract_images(self, state: AgentState) -> AgentState:
        payload = {
            "file_id": state["doc_id"],
            "source_path": state["doc_meta"]["source_path"],
            "file_type": state["doc_meta"].get("file_type"),
        }
        result = self._call_tool("extract_images", payload)
        state["structure"]["images"] = result.get("images", [])
        self._emit_event(
            state,
            "node_completed",
            {"node": "extract_images", "image_count": len(state["structure"]["images"])},
        )
        return state

    def _node_compute_file_stats(self, state: AgentState) -> AgentState:
        payload = {
            "file_id": state["doc_id"],
            "md_file_id": state["doc_meta"].get("md_file_id"),
            "raw_markdown": state["structure"]["raw_text"],
        }
        result = self._call_tool("compute_file_stats", payload)
        state["doc_meta"]["word_count"] = result.get("word_count")
        state["doc_meta"]["page_count"] = result.get("page_count", 0)
        self._emit_event(
            state,
            "node_completed",
            {"node": "compute_file_stats", "word_count": result.get("word_count")},
        )
        return state

    def _node_analyze_heading_structure(self, state: AgentState) -> AgentState:
        payload = {
            "file_id": state["doc_id"],
            "md_file_id": state["doc_meta"].get("md_file_id"),
            "raw_markdown": state["structure"]["raw_text"],
        }
        result = self._call_tool("analyze_heading_structure", payload)
        headings = result.get("headings", [])
        heading_entries = []
        for heading in headings:
            heading_entries.append(
                {
                    "level": heading.get("heading_level", ""),
                    "title": heading.get("heading_text", ""),
                    "page": None,
                    "numbering": None,
                    "char_start": None,
                    "char_end": None,
                    "line_number": heading.get("line_number"),
                }
            )
        state["structure"]["headings"] = heading_entries
        state["structure"]["toc_detected"] = bool(heading_entries)
        state["structure"]["toc_entries"] = []
        self._emit_event(
            state,
            "node_completed",
            {"node": "analyze_heading_structure", "heading_count": len(heading_entries)},
        )
        return state

    def _node_extract_toc_llm(self, state: AgentState) -> AgentState:
        excerpt = self._get_phase1_excerpt(state, preferred_pages=5)
        if not excerpt:
            return state
        payload = {
            "doc_title": state["doc_meta"]["doc_title"],
            "page_count": state["doc_meta"].get("page_count", 0),
            "document_excerpt": excerpt,
        }
        result = self._invoke_llm_prompt(state, "phase1_toc_extraction.md", payload)
        if result and result.get("entries"):
            state["structure"]["toc_entries"] = result["entries"]
            state["structure"]["toc_detected"] = True
            self._emit_event(
            state,
                "node_completed",
            {
                    "node": "phase1_toc_extraction",
                    "entry_count": len(result["entries"]),
            },
        )
        return state

    def _node_build_file_metadata(self, state: AgentState) -> AgentState:
        payload = {
            "file_id": state["doc_id"],
            "source_path": state["doc_meta"]["source_path"],
            "file_type": state["doc_meta"].get("file_type"),
            "md_file_id": state["doc_meta"].get("md_file_id"),
            "md_path": state["doc_meta"].get("md_path"),
            "images": state["structure"]["images"],
            "page_count": state["doc_meta"].get("page_count", 0),
            "word_count": state["doc_meta"].get("word_count", 0),
            "heading_levels": self._derive_heading_levels(state),
        }
        result = self._call_tool("build_file_metadata", payload)
        state["file_metadata"] = result.get("file_metadata")
        self._emit_event(state, "node_completed", {"node": "build_file_metadata"})
        return state

    def _node_collect_phase1_stats(self, state: AgentState) -> AgentState:
        stats = {
            "word_count": state["doc_meta"].get("word_count", 0),
            "page_count": state["doc_meta"].get("page_count", 0),
            "heading_count": len(state["structure"]["headings"]),
            "image_count": len(state["structure"]["images"]),
        }
        state["phase1"]["stats"] = stats
        self._emit_event(state, "node_completed", {"node": "collect_phase1_stats", "stats": stats})
        return state

    def _node_phase1_doc_summary_llm(self, state: AgentState) -> AgentState:
        excerpt = self._get_phase1_excerpt(state, preferred_pages=5)
        doc_meta = state.get("doc_meta") or {}
        payload = {
            "doc_title": doc_meta.get("doc_title") or state["doc_id"],
            "page_count": doc_meta.get("page_count", 0),
            "word_count": doc_meta.get("word_count", 0),
            "document_excerpt": excerpt,
        }
        result = self._invoke_llm_prompt(state, "phase1_doc_summary.md", payload)
        if result:
            state["phase1"]["doc_summary"] = result
            self._emit_event(state, "node_completed", {"node": "phase1_doc_summary", "summary": result})
            self._sync_vfs_artifacts(
                state,
                [{"path": "/phase1/doc_summary.json", "label": "Phase 1 Doc Summary"}],
            )
        return state

    def _node_phase1_toc_review_llm(self, state: AgentState) -> AgentState:
        doc_meta = state.get("doc_meta") or {}
        payload = {
            "doc_title": doc_meta.get("doc_title") or state["doc_id"],
            "page_count": doc_meta.get("page_count", 0),
            "toc_entries": state["structure"].get("toc_entries", []),
            "headings": state["structure"].get("headings", []),
            "document_excerpt": self._get_markdown_excerpt(state, max_chars=8000),
        }
        result = self._invoke_llm_prompt(state, "phase1_toc_review.md", payload)
        if result:
            state["phase1"]["toc_review"] = result
            self._emit_event(state, "node_completed", {"node": "phase1_toc_review", "score": result.get("structure_score")})
            self._sync_vfs_artifacts(
                state,
                [{"path": "/phase1/toc_review.json", "label": "Phase 1 TOC Review"}],
            )
        return state

    def _node_phase1_template_fitness_llm(self, state: AgentState) -> AgentState:
        doc_meta = state.get("doc_meta") or {}
        payload = {
            "doc_title": doc_meta.get("doc_title") or state["doc_id"],
            "document_excerpt": self._get_markdown_excerpt(state, max_chars=12000),
            "template": {
                "template_id": state["template_meta"]["template_id"],
                "template_label": state["template_meta"].get("template_label"),
                "template_text": state["template_meta"].get("template_text"),
                "template_categories": state["template_meta"].get("template_categories"),
            },
        }
        result = self._invoke_llm_prompt(state, "phase1_template_fitness.md", payload)
        if result:
            state["phase1"]["template_fitness_report"] = result
            self._emit_event(
            state,
                "node_completed",
                {
                    "node": "phase1_template_fitness",
                    "overall_alignment": result.get("overall_alignment"),
                },
            )
            self._sync_vfs_artifacts(
                state,
                [{"path": "/phase1/template_fitness.json", "label": "Template Fitness"}],
            )
        return state

    def _node_phase1_section_strategy_llm(self, state: AgentState) -> AgentState:
        doc_summary = state["phase1"].get("doc_summary")
        toc_review = state["phase1"].get("toc_review")
        template_report = state["phase1"].get("template_fitness_report")
        if not (doc_summary and toc_review and template_report):
            self.logger.warning("Section strategy skipped: missing upstream reports")
            return state

        payload = {
            "doc_summary": doc_summary,
            "toc_review": toc_review,
            "template_fitness": template_report,
            "stats": state["phase1"].get("stats"),
        }
        result = self._invoke_llm_prompt(state, "phase1_section_strategy.md", payload)
        if result:
            state["phase1"]["section_strategy"] = result
            self._emit_event(
            state,
                "node_completed",
            {
                    "node": "phase1_section_strategy",
                    "verdict": result.get("verdict"),
            },
        )
            self._sync_vfs_artifacts(
                state,
                [{"path": "/phase1/section_strategy.json", "label": "Section Strategy"}],
            )
        return state

    def _determine_section_targets(
        self, state: AgentState, section_scope: Optional[List[str]]
    ) -> List[str]:
        if section_scope:
            return section_scope
        template_sections = self._get_template_sections(state)
        titles = [section.get("title") for section in template_sections if section.get("title")]
        if titles:
            return titles
        categories = state["template_meta"].get("template_categories") or []
        if categories:
            return categories
        return list(state["phase2"]["chunks"].keys())

    def _extract_section_chunk(self, state: AgentState, section_title: str) -> Optional[Dict[str, Any]]:
        raw_markdown = state["structure"]["raw_text"]
        headings = state["structure"].get("headings", [])
        toc_entries = state["structure"].get("toc_entries", [])

        heading_candidate = self._call_tool(
            "extract_section_by_headings",
            {
                "section_title": section_title,
                "raw_markdown": raw_markdown,
                "headings": headings,
            },
        )
        toc_candidate = self._call_tool(
            "extract_section_by_toc",
            {
                "section_title": section_title,
                "raw_markdown": raw_markdown,
                "toc_entries": toc_entries,
            },
        )

        # Check if both extraction methods failed (no text extracted)
        heading_has_text = heading_candidate.get("text", "").strip()
        toc_has_text = toc_candidate.get("text", "").strip()

        # Fallback: if both methods fail, use whole document as single chunk
        if not heading_has_text and not toc_has_text:
            self.logger.warning(
                "Both heading and TOC extraction failed for section '%s', using whole-doc fallback",
                section_title,
            )
            return {
                "section_title": section_title,
                "method": "whole_doc_fallback",
                "page_range": [1, state["doc_meta"].get("page_count", None)],
                "char_range": [0, len(raw_markdown)],
                "boundary_check": "whole_document",
                "issues": [
                    "No markdown headings found",
                    "TOC entries not found in text",
                    "Using entire document as single chunk",
                ],
                "text": raw_markdown,
            }

        # Normal flow: use LLM validator to pick best candidate
        template_section = self._lookup_template_description(state, section_title)
        validator_result = self._node_phase2_section_validator_llm(
            state,
            section_title,
            heading_candidate,
            toc_candidate,
            template_section,
        )

        if not validator_result or not validator_result.get("final_section_text"):
            return None

        return {
            "section_title": section_title,
            "method": validator_result.get("chosen_method", "headings"),
            "page_range": validator_result.get("page_range") or [None, None],
            "char_range": validator_result.get("char_range"),
            "boundary_check": validator_result.get("boundary_check", "unknown"),
            "issues": validator_result.get("issues", []),
            "text": validator_result.get("final_section_text", ""),
        }

    def _node_phase2_section_validator_llm(
        self,
        state: AgentState,
        section_title: str,
        heading_candidate: Dict[str, Any],
        toc_candidate: Dict[str, Any],
        template_section: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        payload = {
            "section_title": section_title,
            "template_expectation": template_section,
            "heading_candidate": heading_candidate,
            "toc_candidate": toc_candidate,
        }
        result = self._invoke_llm_prompt(state, "phase2_section_validator.md", payload)
        if result:
            self._emit_event(
                state,
                "node_completed",
                {
                    "node": "phase2_section_validator",
                    "section_title": section_title,
                    "method": result.get("chosen_method"),
                },
            )
        return result

    def _node_phase2_section_reviews(self, state: AgentState) -> AgentState:
        reviews: Dict[str, Any] = {}
        template_sections = self._get_template_sections(state)

        for section_title, chunk in state["phase2"]["chunks"].items():
            if not chunk.get("text"):
                continue
            expectation = self._lookup_template_description_from_list(template_sections, section_title)
            payload = {
                "section_title": section_title,
                "section_text": chunk.get("text"),
                "template_expectation": expectation,
                "doc_summary": state["phase1"].get("doc_summary"),
                "stats": state["phase1"].get("stats"),
            }
            review = self._invoke_llm_prompt(state, "phase2_section_review.md", payload)
            if not review:
                continue
            reviews[section_title] = review
            issues: List[Dict[str, Any]] = review.get("issues") or []

            # Enrich issues with char_range from chunk for precise replacements
            chunk_char_range = chunk.get("char_range")

            for issue in issues:
                if not issue.get("id"):
                    issue["id"] = f"{section_title[:3].upper()}-{len(state['changes']['suggested_changes']) + 1:03d}"

                # Add char_range if available and issue has original_text
                # This enables precise replacements in Phase 3
                if chunk_char_range and issue.get("original_text"):
                    issue["char_range"] = chunk_char_range
                    issue["source_section"] = section_title

                state["changes"]["suggested_changes"].append(issue)

        state["phase2"]["reviews"] = reviews
        if state["changes"]["suggested_changes"]:
            self._sync_vfs_artifacts(
                state,
                [{"path": "/changes/suggested_changes.json", "label": "Suggested Changes"}],
            )
        return state

    def _node_phase2_summary_llm(self, state: AgentState) -> AgentState:
        reviews = state["phase2"].get("reviews") or {}
        if not reviews:
            return state
        total_issues = len(state["changes"]["suggested_changes"])
        high_severity = [
            change for change in state["changes"]["suggested_changes"] if change.get("severity") == "high"
        ]
        doc_meta = state.get("doc_meta") or {}
        payload = {
            "doc_title": doc_meta.get("doc_title") or state["doc_id"],
            "section_reviews": list(reviews.values()),
            "total_issues": total_issues,
            "high_severity_count": len(high_severity),
        }
        summary = self._invoke_llm_prompt(state, "phase2_summary.md", payload)
        if summary:
            state["phase2"]["summary_report"] = summary
            self._emit_event(
                state,
                "node_completed",
                {"node": "phase2_summary", "overall_posture": summary.get("overall_posture")},
            )
            self._sync_vfs_artifacts(
                state,
                [{"path": "/phase2/summary_report.json", "label": "Phase 2 Summary"}],
            )
            return state

    def _node_change_selection_intent_orchestrator(self, state: AgentState) -> AgentState:
        instruction = state["user_interaction"].get("user_change_instruction")
        if not instruction:
            self.logger.warning("No user change instruction provided; remaining in await state")
            state["control"] = "await_change_instruction"
            return state

        plan = self.interpret_change_instruction(state, instruction)
        if not plan or not plan.get("change_ids_to_apply"):
            self.logger.warning("Change selection intent produced no actionable IDs")
            state["control"] = "await_change_instruction"
        return state

    def _node_apply_changes_orchestrator(self, state: AgentState) -> AgentState:
        plan = state["changes"].get("change_selection_plan")
        selected_changes = self._select_changes_for_application(state, plan=plan)

        if not selected_changes:
            state["phase3_status"] = "failed"
            state["errors"].append("Phase 3: no applicable changes in plan")
            state["control"] = "failed"
            return state

        return self._apply_changes_core(state, selected_changes, state["structure"]["raw_text"])

    def _node_verify_changes_orchestrator(self, state: AgentState) -> AgentState:
        snapshot = state["changes"].pop("_pre_apply_text", None)
        applied_ids = state["changes"].get("applied_change_ids", [])
        if snapshot and applied_ids:
            self._node_apply_changes_verifier_llm(
                state,
                snapshot,
                state["structure"]["raw_text"],
                applied_ids,
            )
        return state

    def _node_change_selection_intent_llm(
        self, state: AgentState, user_instruction: str
    ) -> Optional[ChangeSelectionPlan]:
        suggested_changes = state["changes"].get("suggested_changes", [])
        if not suggested_changes:
            self.logger.warning("No suggested changes available for selection intent")
            return None

        pending_changes = [change for change in suggested_changes if change.get("status") != "applied"]
        change_catalog = [
            {
                "id": change.get("id"),
                "index": change.get("index"),
                "section_title": change.get("section_title"),
                "severity": change.get("severity"),
                "type": change.get("type"),
                "status": change.get("status", "pending"),
            }
            for change in suggested_changes
        ]

        payload = {
            "doc_title": state["doc_meta"]["doc_title"],
            "user_instruction": user_instruction,
            "total_changes": len(suggested_changes),
            "pending_changes": len(pending_changes),
            "high_severity_changes": len([c for c in suggested_changes if c.get("severity") == "high"]),
            "change_catalog": change_catalog,
        }
        result = self._invoke_llm_prompt(state, "change_selection_intent.md", payload)
        if not result:
            return None

        apply_mode = result.get("apply_mode") or "by_ids"
        requested_ids = result.get("change_ids_to_apply") or []
        valid_ids = {change.get("id") for change in suggested_changes if change.get("id")}

        if apply_mode == "all":
            filtered_ids = [cid for cid in valid_ids]
        else:
            filtered_ids = [cid for cid in requested_ids if cid in valid_ids]
            if requested_ids and not filtered_ids:
                self.logger.warning("LLM requested change IDs that do not exist: {0}".format(requested_ids))

        plan: ChangeSelectionPlan = {
            "apply_mode": apply_mode,
            "change_ids_to_apply": filtered_ids,
            "rationale": result.get("rationale", ""),
        }

        state["changes"]["change_selection_plan"] = plan
        self._emit_event(
            state,
            "node_completed",
            {
                "node": "change_selection_intent",
                "apply_mode": plan["apply_mode"],
                "selected_change_count": len(plan["change_ids_to_apply"]),
            },
        )
        return plan

    def _node_apply_changes_verifier_llm(
        self,
        state: AgentState,
        original_text: str,
        updated_text: str,
        applied_change_ids: List[str],
    ) -> Optional[Dict[str, Any]]:
        if not applied_change_ids:
            return None

        applied_changes = [
            change
            for change in state["changes"].get("suggested_changes", [])
            if change.get("id") in applied_change_ids
        ]
        if not applied_changes:
            return None

        payload = {
            "doc_title": state["doc_meta"]["doc_title"],
            "applied_changes": [
                {
                    "id": change.get("id"),
                    "section_title": change.get("section_title"),
                    "severity": change.get("severity"),
                    "original_text": change.get("original_text"),
                    "suggested_text": change.get("suggested_text"),
                }
                for change in applied_changes
            ],
            "original_excerpt": self._get_markdown_excerpt_from_text(original_text, max_chars=6000),
            "updated_excerpt": self._get_markdown_excerpt_from_text(updated_text, max_chars=6000),
        }

        result = self._invoke_llm_prompt(state, "phase3_apply_verifier.md", payload)
        if not result:
            return None

        issues = result.get("issues") or []
        for issue in issues:
            message = issue.get("message") or "Change verification issue detected"
            change_id = issue.get("change_id") or "unknown"
            state["errors"].append(f"Phase 3 verification ({change_id}): {message}")

        self._emit_event(
            state,
            "node_completed",
            {
                "node": "apply_changes_verifier",
                "issues_found": len(issues),
            },
        )
        return result

    def _node_phase2_extract_sections(self, state: AgentState) -> AgentState:
        section_scope = state["user_interaction"].get("selected_section_scope")
        targets = self._determine_section_targets(state, section_scope)
        extracted = 0
        fallback_count = 0
        state["phase2"]["chunks"] = {}

        for title in targets:
            chunk = self._extract_section_chunk(state, title)
            if chunk and chunk.get("text"):
                state["phase2"]["chunks"][title] = chunk
                extracted += 1
                if chunk.get("method") == "whole_doc_fallback":
                    fallback_count += 1

        if extracted > 0 and fallback_count == extracted:
            self.logger.warning(
                "All %d sections used whole-doc fallback. Consolidating to single 'Full Document' chunk.",
                extracted,
            )
            raw_markdown = state["structure"]["raw_text"]
            state["phase2"]["chunks"] = {
                "Full Document": {
                    "section_title": "Full Document",
                    "method": "whole_doc_fallback",
                    "page_range": [1, state["doc_meta"].get("page_count", None)],
                    "char_range": [0, len(raw_markdown)],
                    "boundary_check": "whole_document",
                    "issues": [
                        "Document lacks markdown headings and TOC entries",
                        "Reviewing entire document as single section",
                    ],
                    "text": raw_markdown,
                }
            }
            extracted = 1

        if extracted == 0:
            state["phase2_status"] = "failed"
            state["control"] = "failed"
            state["errors"].append("Phase 2: no sections extracted")
        return state

    def _node_agent_planner_llm(self, state: AgentState, user_message: str) -> Optional[Dict[str, Any]]:
        """
        LLM node that interprets user commands and generates execution plans.

        Args:
            state: Current agent state
            user_message: User's natural language command

        Returns:
            Plan dict with plan_steps, summary, requires_confirmation, or None if error
        """
        # Build state context for the LLM
        total_changes = len(state["changes"].get("suggested_changes", []))
        applied_changes = len(state["changes"].get("applied_change_ids", []))
        skipped_changes = len(state["changes"].get("skipped_changes", []))

        payload = {
            "run_id": state["run_id"],
            "doc_id": state["doc_id"],
            "user_message": user_message,
            "phase1_status": state.get("phase1_status", "pending"),
            "phase2_status": state.get("phase2_status", "pending"),
            "phase3_status": state.get("phase3_status", "pending"),
            "total_changes": total_changes,
            "applied_changes": applied_changes,
            "skipped_changes": skipped_changes,
            "errors": state.get("errors", []),
        }

        result = self._invoke_llm_prompt(state, "agent_planner.md", payload)
        if not result:
            return None

        # Validate plan structure
        if not isinstance(result.get("plan_steps"), list):
            self.logger.error("Agent planner returned invalid plan_steps")
            return None

        # Emit event with plan summary
        self._emit_event(
            state,
            "node_completed",
            {
                "node": "agent_planner",
                "summary": result.get("summary", ""),
                "requires_confirmation": result.get("requires_confirmation", False),
                "step_count": len(result.get("plan_steps", [])),
            },
        )

        return result

    def _execute_agent_plan(
        self, state: AgentState, plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a plan generated by the agent planner.

        Args:
            state: Current agent state
            plan: Plan dict with plan_steps, summary, requires_confirmation

        Returns:
            Execution results with status, outputs, and any errors
        """
        plan_steps = plan.get("plan_steps", [])
        results = []
        execution_errors = []

        for step in plan_steps:
            tool = step.get("tool")
            parameters = step.get("parameters", {})
            reasoning = step.get("reasoning", "")

            self.logger.info("Executing plan step: %s (reason: %s)", tool, reasoning)

            try:
                result = self._execute_tool(state, tool, parameters)
                results.append({
                    "tool": tool,
                    "status": "success",
                    "result": result,
                    "reasoning": reasoning,
                })
            except Exception as exc:
                error_msg = f"Tool {tool} failed: {exc}"
                self.logger.error(error_msg)
                execution_errors.append(error_msg)
                results.append({
                    "tool": tool,
                    "status": "failed",
                    "error": str(exc),
                    "reasoning": reasoning,
                })

        return {
            "plan_summary": plan.get("summary", ""),
            "requires_confirmation": plan.get("requires_confirmation", False),
            "executed_steps": results,
            "errors": execution_errors,
        }

    def _execute_tool(
        self, state: AgentState, tool_name: str, parameters: Dict[str, Any]
    ) -> Any:
        """
        Execute a single tool by name with parameters.

        Args:
            state: Current agent state
            tool_name: Name of the tool to execute
            parameters: Tool parameters

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool name is unknown
        """
        # Phase 1: Holistic Assessment
        if tool_name == "run_phase1":
            template_id = parameters.get("template_id")
            # Phase 1 expects document_path, which should already be in state
            doc_path = state["doc_meta"]["source_path"]
            return self.run_phase1(doc_path, state["run_id"], template_id)

        # Phase 2: Section-Level Reviews
        elif tool_name == "run_phase2":
            section_scope = parameters.get("section_scope")
            return self.run_phase2(state, section_scope=section_scope)

        # Phase 3: Change Application
        elif tool_name == "run_phase3_all":
            return self.run_phase3(state, change_ids=None, severity_filter=None)

        elif tool_name == "run_phase3_severity":
            severity_filter = parameters.get("severity_filter")
            return self.run_phase3(state, change_ids=None, severity_filter=severity_filter)

        elif tool_name == "run_phase3_ids":
            change_ids = parameters.get("change_ids", [])
            return self.run_phase3(state, change_ids=change_ids, severity_filter=None)

        # Information Retrieval
        elif tool_name == "get_summary":
            return self._get_summary(state)

        elif tool_name == "get_review":
            section_title = parameters.get("section_title")
            return self._get_review(state, section_title)

        elif tool_name == "list_changes":
            severity_filter = parameters.get("severity_filter")
            return self._list_changes(state, severity_filter)

        elif tool_name == "download_artifact":
            artifact_type = parameters.get("artifact_type")
            return self._download_artifact(state, artifact_type)

        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def _get_summary(self, state: AgentState) -> Dict[str, Any]:
        """Return current state summary."""
        return {
            "run_id": state["run_id"],
            "doc_id": state["doc_id"],
            "phase1_status": state.get("phase1_status", "pending"),
            "phase2_status": state.get("phase2_status", "pending"),
            "phase3_status": state.get("phase3_status", "pending"),
            "total_changes": len(state["changes"].get("suggested_changes", [])),
            "applied_changes": len(state["changes"].get("applied_change_ids", [])),
            "skipped_changes": len(state["changes"].get("skipped_changes", [])),
            "phase1_summary": state["phase1"].get("doc_summary"),
            "phase2_summary": state["phase2"].get("summary_report"),
        }

    def _get_review(self, state: AgentState, section_title: str) -> Optional[Dict[str, Any]]:
        """Return Phase 2 review for specific section."""
        reviews = state["phase2"].get("reviews", {})
        return reviews.get(section_title)

    def _list_changes(
        self, state: AgentState, severity_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return all suggested changes, optionally filtered by severity."""
        changes = state["changes"].get("suggested_changes", [])
        if severity_filter:
            changes = [c for c in changes if c.get("severity") == severity_filter]
        return changes

    def _download_artifact(self, state: AgentState, artifact_type: str) -> Dict[str, Any]:
        """Prepare artifact for download."""
        if artifact_type == "improved_markdown":
            return {
                "artifact_type": "improved_markdown",
                "content": state["changes"].get("new_raw_text") or state["structure"]["raw_text"],
                "filename": f"{state['doc_id']}_improved.md",
            }
        elif artifact_type == "phase1_report":
            return {
                "artifact_type": "phase1_report",
                "content": json.dumps(state["phase1"], indent=2),
                "filename": f"{state['doc_id']}_phase1_report.json",
            }
        elif artifact_type == "phase2_reviews":
            return {
                "artifact_type": "phase2_reviews",
                "content": json.dumps(state["phase2"]["reviews"], indent=2),
                "filename": f"{state['doc_id']}_phase2_reviews.json",
            }
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")

    def _select_changes_for_application(
        self,
        state: AgentState,
        change_ids: Optional[List[str]] = None,
        severity_filter: Optional[str] = None,
        plan: Optional[ChangeSelectionPlan] = None,
    ) -> List[Dict[str, Any]]:
        suggestions = state["changes"].get("suggested_changes", [])

        # Filter applicable changes (exclude missing_content without original_text)
        applicable = []
        skipped = []

        for change in suggestions:
            change_type = change.get("type", "")
            original_text = change.get("original_text", "").strip()

            # Skip missing_content changes that have no original text (can't be applied deterministically)
            if change_type == "missing_content" and not original_text:
                skipped.append({
                    "change": change,
                    "reason": "missing_content requires manual insertion (no original_text anchor)",
                })
                continue

            applicable.append(change)

        # Store skipped changes for user review
        state["changes"]["skipped_changes"] = skipped

        selected = list(applicable)

        # Apply explicit filters first
        if change_ids:
            selected = [change for change in applicable if change.get("id") in change_ids]
            return selected
        if severity_filter:
            severity = severity_filter.lower()
            selected = [change for change in applicable if change.get("severity") == severity]
            return selected

        if plan:
            apply_mode = plan.get("apply_mode")
            plan_ids = plan.get("change_ids_to_apply", [])
            if apply_mode == "all":
                selected = list(applicable)
            elif plan_ids:
                selected_ids = set(plan_ids)
                selected = [change for change in applicable if change.get("id") in selected_ids]
                missing_ids = [cid for cid in plan_ids if cid not in selected_ids]
                if missing_ids:
                    self.logger.warning(
                        "Change selection plan referenced unavailable IDs: %s", missing_ids
                    )
            else:
                self.logger.warning("Change selection plan did not specify any IDs to apply.")

        self.logger.info(
            "Selected %d changes for application (%d skipped as missing_content)",
            len(selected),
            len(skipped),
        )

        return selected

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _initialise_state(
        self,
        source: Path,
        run_id: Optional[str],
        template_id: Optional[str],
    ) -> AgentState:
        doc_id = source.stem
        run_identifier = run_id or f"docrev-{uuid4().hex}"
        template_meta = self._load_template_definition(template_id)

        doc_meta: DocMeta = {
            "doc_title": doc_id.replace("_", " ").strip() or doc_id,
            "doc_source": "upload",
            "source_path": str(source),
            "file_type": None,
            "page_count": 0,
            "word_count": None,
            "md_file_id": None,
            "md_path": None,
            "version": 1,
        }

        structure: StructureData = {
            "raw_text": "",
            "pages": [],
            "headings": [],
            "toc_detected": False,
            "toc_entries": [],
            "images": [],
        }

        phase1: Phase1Data = {
            "stats": {},
            "doc_summary": None,
            "toc_review": None,
            "template_fitness_report": None,
            "section_strategy": None,
        }
        phase2: Phase2Data = {
            "chunks": {},
            "reviews": {},
            "summary_report": None,
        }
        changes: ChangesData = {
            "suggested_changes": [],
            "applied_change_ids": [],
            "failed_changes": [],
            "change_selection_plan": None,
            "skipped_changes": [],
            "new_raw_text": None,
        }
        user_interaction: UserInteractionState = {
            "user_selected_section_strategy": False,
            "selected_section_scope": None,
            "user_change_instruction": None,
        }

        state: AgentState = {
            "run_id": run_identifier,
            "doc_id": doc_id,
            "control": "phase0_ingestion",
            "last_node": None,
            "errors": [],
            "phase1_status": "pending",
            "phase2_status": "pending",
            "phase3_status": "pending",
            "locked_by": None,
            "lock_timestamp": None,
            "doc_meta": doc_meta,
            "structure": structure,
            "template_meta": template_meta,
            "phase1": phase1,
            "phase2": phase2,
            "changes": changes,
            "user_interaction": user_interaction,
            "file_metadata": None,
            "vfs_artifacts": [],
            "logs": [],
            "agent_transcript": [],
        }
        return state

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        if not config_path.exists():
            raise FileNotFoundError(f"Doc review agent config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_template_definition(self, template_id: Optional[str]) -> TemplateMeta:
        template_key = template_id or self.config.get("template", {}).get("template_id") or "policy_template"
        template_path = TEMPLATE_DIR / f"{template_key}.json"
        if not template_path.exists():
            raise FileNotFoundError(f"Template definition not found: {template_path}")
        with template_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        categories = [section.get("title", "") for section in data.get("sections", []) if section.get("title")]
        return TemplateMeta(
            template_id=template_key,
            template_label=data.get("title"),
            template_text=json.dumps(data, ensure_ascii=False),
            template_categories=categories,
            max_section_words=self.config.get("template", {}).get("max_section_words", 500),
        )

    def _call_tool(self, tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return call_mcp(tool_name, payload)
        except Exception as exc:  # pragma: no cover - relies on external tools
            self.logger.exception("MCP tool %s failed: %s", tool_name, exc)
            raise

    def _derive_heading_levels(self, state: AgentState) -> List[str]:
        headings = state["structure"].get("headings", [])
        levels = []
        for heading in headings:
            level = heading.get("level")
            if level and level not in levels:
                levels.append(level)
        return levels

    def _register_vfs_artifact(self, state: AgentState, artifact: VfsArtifact) -> None:
        existing_paths = {entry["path"] for entry in state["vfs_artifacts"]}
        if artifact["path"] in existing_paths:
            return
        state["vfs_artifacts"].append(artifact)

    def _emit_event(self, state: AgentState, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """
        Placeholder event emitter. Later phases will push these onto WebSocket streams.
        """
        payload = payload or {}
        if self.event_emitter:
            try:
                self.event_emitter(event_type, payload)
            except Exception:  # pragma: no cover
                self.logger.exception("Failed to emit event %s", event_type)
        else:
            self.logger.debug(
                "[%s] event=%s doc_id=%s payload=%s",
                state["run_id"],
                event_type,
                state["doc_id"],
                payload,
            )

    def _get_markdown_excerpt(self, state: AgentState, max_chars: int = 10000) -> str:
        raw = state["structure"].get("raw_text", "") or ""
        if not raw:
            return ""
        if len(raw) <= max_chars:
            return raw
        snippet = raw[:max_chars]
        cutoff = snippet.rfind("\n")
        if cutoff > max_chars * 0.7:
            snippet = snippet[:cutoff]
        return snippet.strip() + "\n\n..."

    @staticmethod
    def _get_markdown_excerpt_from_text(text: str, max_chars: int = 10000) -> str:
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        snippet = text[:max_chars]
        cutoff = snippet.rfind("\n")
        if cutoff > max_chars * 0.7:
            snippet = snippet[:cutoff]
        return snippet.strip() + "\n\n..."

    def _get_phase1_excerpt(self, state: AgentState, preferred_pages: int = 5) -> str:
        raw = state["structure"].get("raw_text", "") or ""
        if not raw:
            return ""
        page_count = state["doc_meta"].get("page_count", 0) or 0
        if page_count and page_count <= 10:
            return raw
        words = raw.split()
        if not words:
            return raw
        words_per_page = max(200, len(words) / max(page_count or 1, 1))
        limit_words = int(words_per_page * preferred_pages)
        excerpt_words = words[: max(limit_words, 800)]
        return " ".join(excerpt_words)

    def _load_prompt_template(self, prompt_name: str) -> str:
        if prompt_name in self._prompt_cache:
            return self._prompt_cache[prompt_name]
        prompt_path = Path("external/doc_review/prompts") / prompt_name
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
        content = prompt_path.read_text(encoding="utf-8").strip()
        self._prompt_cache[prompt_name] = content
        return content

    def _invoke_llm_prompt(
        self, state: AgentState, prompt_name: str, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        try:
            system_prompt = self._load_prompt_template(prompt_name)
            responses = call_llm_json(system_prompt, payload)
        except LLMNotAvailableError as exc:
            msg = f"{prompt_name} skipped: {exc}"
            state["errors"].append(msg)
            self.logger.warning(msg)
            return None
        except FileNotFoundError as exc:
            state["errors"].append(str(exc))
            self.logger.error(str(exc))
            return None
        except Exception as exc:  # pragma: no cover
            msg = f"LLM prompt {prompt_name} failed: {exc}"
            state["errors"].append(msg)
            self.logger.exception(msg)
            return None

        if not responses:
            self.logger.warning("LLM prompt %s returned empty result", prompt_name)
            return None
        result = responses[0]
        state["agent_transcript"].append(
            {
                "timestamp": _utcnow_iso(),
                "prompt": prompt_name,
                "payload_preview": {k: payload.get(k) for k in list(payload.keys())[:5]},
                "response_preview": {k: result.get(k) for k in list(result.keys())[:5]},
            }
        )
        return result

    def _get_template_sections(self, state: AgentState) -> List[Dict[str, Any]]:
        text = state["template_meta"].get("template_text")
        if not text:
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []
        return data.get("sections", [])

    def _lookup_template_description(self, state: AgentState, section_title: str) -> Optional[Dict[str, Any]]:
        sections = self._get_template_sections(state)
        return self._lookup_template_description_from_list(sections, section_title)

    @staticmethod
    def _lookup_template_description_from_list(
        sections: List[Dict[str, Any]], section_title: str
    ) -> Optional[Dict[str, Any]]:
        target = section_title.lower().strip()
        for section in sections:
            title = (section.get("title") or "").lower().strip()
            if title == target:
                return section
        return None


