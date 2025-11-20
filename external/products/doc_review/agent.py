from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from external.platform.llm import get_llm_client, is_llm_available
from external.platform.mcp import call_mcp
from external.products.doc_review.models import (
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


class LLMNotAvailableError(RuntimeError):
    """LLM not available error."""
    pass


def call_llm_json(system_prompt: str, payload: dict) -> list:
    """Call LLM with JSON payload and return parsed JSON response."""
    if not is_llm_available():
        raise LLMNotAvailableError("LLM not configured")
    client = get_llm_client()
    import json
    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
    response = client.invoke_with_prompt(system_prompt, user_prompt, response_format="json")
    try:
        cleaned_response = response.strip()
        if cleaned_response.startswith("```"):
            lines = cleaned_response.split("\n")
            if len(lines) > 2:
                cleaned_response = "\n".join(lines[1:-1])
        parsed = json.loads(cleaned_response)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    return [parsed]


def call_llm_markdown(system_prompt: str, payload: dict) -> str:
    """Call LLM with payload and return raw markdown response."""
    if not is_llm_available():
        raise LLMNotAvailableError("LLM not configured")
    client = get_llm_client()
    import json
    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
    response = client.invoke_with_prompt(system_prompt, user_prompt)
    return response.strip()

LOGGER = logging.getLogger(__name__)
DEFAULT_AGENT_CONFIG_PATH = Path("external/config/agent/doc_review_agent.json")
TEMPLATE_DIR = Path("external/products/doc_review/templates")

NODE_LABELS: Dict[str, str] = {
    "phase0_ingestion": "Phase 0 – Ingestion",
    "phase1_toc_review": "Phase 1 – TOC Review",
    "phase2_holistic_checks": "Phase 2 – Holistic Checks",
    "phase2_synthesis": "Phase 2 – Synthesis Summary",
}

NODE_KINDS: Dict[str, str] = {
    "phase0_ingestion": "tool",
    "phase1_toc_review": "llm",
    "phase2_holistic_checks": "llm",
    "phase2_synthesis": "llm",
}

NODE_TRANSITIONS: Dict[str, str] = {
    "phase0_ingestion": "phase1_toc_review",
    "phase1_toc_review": "phase2_holistic_checks",
    "phase2_holistic_checks": "phase2_synthesis",
    "phase2_synthesis": "completed",
}

ORCHESTRATOR_WAIT_STATES: Set[str] = {
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

    def __init__(self, agent_config_path: Optional[Path] = None, tools_registry=None) -> None:
        self.agent_config_path = agent_config_path or DEFAULT_AGENT_CONFIG_PATH
        self.config = self._load_config(self.agent_config_path)
        self.logger = LOGGER
        self._prompt_cache: Dict[str, str] = {}
        self.node_registry = self._build_node_registry()
        self.state_lock_owner: Optional[str] = None
        self.event_emitter: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self.agent_log_root = Path("logs")
        self.tools_registry = tools_registry

    def _build_node_registry(self) -> Dict[str, Callable[[AgentState], AgentState]]:
        return {
            "phase0_ingestion": self._run_phase0_ingestion,
            "phase1_toc_review": self._node_phase1_toc_review_llm,
            "phase2_holistic_checks": self._node_phase2_holistic_checks,
            "phase2_synthesis": self._node_phase2_synthesis,
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
        Execute ingestion + deterministic stats for the supplied document. LLM-driven
        analysis will be layered in subsequent phases of the implementation plan.
        """

        state = self._initialise_state(
            source=Path(document_path).expanduser().resolve(),
            run_id=run_id,
            template_id=template_id,
        )

        self.logger.info(
            "DocReviewAgent.run_phase1: run_id=%s doc_id=%s", state["run_id"], state["doc_id"]
        )

        state["phase1_status"] = "running"
        state["control"] = "phase0_ingestion"
        self.orchestrate(state, stop_controls={"completed", "failed"})

        # Make raw markdown available to the UI immediately after Phase 0+initial Phase 1 nodes
        state = self._node_publish_raw_markdown(state)

        if state.get("control") == "completed":
            state["phase1_status"] = "success"
        else:
            state["phase1_status"] = "failed"

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

        state["control"] = "phase2_holistic_checks"
        self.orchestrate(state, stop_controls={"completed", "failed"})

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
            self._emit_event(state, "node_completed", {"node": "phase1_toc_review", "markdown_length": len(result) if isinstance(result, str) else 0})
            self._sync_vfs_artifacts(
                state,
                [{"path": "/phase1/toc_review.json", "label": "Phase 1 TOC Review"}],
            )
        return state

    def _node_phase2_holistic_checks(self, state: AgentState) -> AgentState:
        """
        Phase 2: Run 4 holistic document checks in parallel.
        Stores results in state['phase2_data'].
        """
        doc_meta = state.get("doc_meta") or {}
        raw_markdown = state["structure"]["raw_text"]
        toc_review = state.get("phase1", {}).get("toc_review", {})
        
        # Prepare common payload
        common_payload = {
            "doc_title": doc_meta.get("doc_title") or state["doc_id"],
            "page_count": doc_meta.get("page_count", 0),
            "word_count": doc_meta.get("word_count", 0),
            "document_text": self._get_markdown_excerpt(state, max_chars=20000),
            "toc_review": toc_review,
            "headings": state["structure"].get("headings", []),
        }
        
        # Initialize phase2_data if not exists
        if "phase2_data" not in state:
            state["phase2_data"] = {}
        
        # Run 4 checks
        checks = [
            ("phase2_check_conceptual_coverage.md", "conceptual_coverage"),
            ("phase2_check_compliance_governance.md", "compliance_governance"),
            ("phase2_check_language_clarity.md", "language_clarity"),
            ("phase2_check_structural_presentation.md", "structural_presentation"),
        ]
        
        for prompt_file, key in checks:
            result = self._invoke_llm_prompt(state, prompt_file, common_payload)
            if result:
                state["phase2_data"][key] = result
                self._emit_event(
                    state,
                    "node_completed",
                    {"node": f"phase2_{key}", "check": key}
                )
        
        # Sync artifacts
        self._sync_vfs_artifacts(
            state,
            [{"path": "/phase2/holistic_checks.json", "label": "Phase 2 Holistic Checks"}],
        )
        
        return state

    def _node_phase2_synthesis(self, state: AgentState) -> AgentState:
        """
        Phase 2: Generate synthesis summary from all 4 checks.
        Stores result in state['phase2_data']['synthesis'].
        """
        phase2_data = state.get("phase2_data", {})
        toc_review = state.get("phase1", {}).get("toc_review", {})
        
        payload = {
            "doc_title": state.get("doc_meta", {}).get("doc_title") or state["doc_id"],
            "toc_review": toc_review,
            "conceptual_coverage": phase2_data.get("conceptual_coverage", {}),
            "compliance_governance": phase2_data.get("compliance_governance", {}),
            "language_clarity": phase2_data.get("language_clarity", {}),
            "structural_presentation": phase2_data.get("structural_presentation", {}),
        }
        
        result = self._invoke_llm_prompt(state, "phase2_synthesis_summary.md", payload)
        if result:
            state["phase2_data"]["synthesis"] = result
            self._emit_event(
                state,
                "node_completed",
                {"node": "phase2_synthesis", "markdown_length": len(result) if isinstance(result, str) else 0}
            )
            self._sync_vfs_artifacts(
                state,
                [{"path": "/phase2/synthesis.json", "label": "Phase 2 Synthesis"}],
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
            if self.tools_registry:
                # Use provided registry
                from external.platform.mcp import MCPClient
                client = MCPClient(tools_registry=self.tools_registry)
                return client.call(tool_name, payload)
            else:
                # Fall back to default global client
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
        prompt_path = Path("external/products/doc_review/prompts") / prompt_name
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
        content = prompt_path.read_text(encoding="utf-8").strip()
        self._prompt_cache[prompt_name] = content
        return content

    def _invoke_llm_prompt(
        self, state: AgentState, prompt_name: str, payload: Dict[str, Any]
    ) -> Optional[Any]:
        # Markdown prompts return raw strings, not JSON
        markdown_prompts = {
            "phase1_toc_review.md",
            "phase2_check_conceptual_coverage.md",
            "phase2_check_compliance_governance.md",
            "phase2_check_language_clarity.md",
            "phase2_check_structural_presentation.md",
            "phase2_synthesis_summary.md",
        }
        
        is_markdown = prompt_name in markdown_prompts
        
        try:
            system_prompt = self._load_prompt_template(prompt_name)
            if is_markdown:
                result = call_llm_markdown(system_prompt, payload)
            else:
                responses = call_llm_json(system_prompt, payload)
                if not responses:
                    self.logger.warning("LLM prompt %s returned empty result", prompt_name)
                    return None
                result = responses[0]
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

        # Log transcript
        if is_markdown:
            preview = result[:200] + "..." if len(result) > 200 else result
            state["agent_transcript"].append(
                {
                    "timestamp": _utcnow_iso(),
                    "prompt": prompt_name,
                    "payload_preview": {k: payload.get(k) for k in list(payload.keys())[:5]},
                    "response_preview": preview,
                }
            )
        else:
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


