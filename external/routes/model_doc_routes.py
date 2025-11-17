from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from flask import jsonify, request, render_template

from routes.base_routes import BaseRoutes
from external.model_doc.model_doc_agent import ModelDocAgent
from external.model_doc.store import ModelDocStore
from external.model_doc.llm import LLMNotAvailableError, generate_chat_reply

logger = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip())
    value = re.sub(r"-+", "-", value)
    return value.strip("-").lower() or "codebase"


class ModelDocRoutes(BaseRoutes):
    """Routes backing the model documentation workflow UI and APIs."""

    def __init__(self, auth_manager, tools_registry=None, mcp_handler=None, socketio=None):
        super().__init__(auth_manager, tools_registry, mcp_handler)
        self.agent = ModelDocAgent()
        self.store = ModelDocStore()
        self.socketio = socketio

    def _make_event_emitter(self, codebase_id: str) -> Optional[Callable[[str, Dict[str, Any]], None]]:
        if not self.socketio:
            return None

        room = f"model_doc:{codebase_id}"

        def emitter(event_type: str, payload: Dict[str, Any]) -> None:
            data = dict(payload)
            data.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
            data["codebase_id"] = codebase_id
            event_name = f"model_doc:{event_type}"
            try:
                self.socketio.emit(event_name, data, room=room)
            except Exception:  # pragma: no cover - socket failures should not break workflow
                logger.exception("Failed to emit %s event for %s", event_name, codebase_id)

        return emitter

    def register_routes(self, app):  # noqa: D401
        """Register model documentation routes."""

        @app.route("/model-doc")
        @self.login_required
        def model_doc_dashboard():
            """Render the model documentation cockpit UI."""
            try:
                return render_template("model_doc_cockpit.html")
            except Exception:
                return jsonify({"error": "UI template not found"}), 500

        @app.route("/api/model_doc/templates", methods=["GET"])
        @self.login_required
        def list_model_doc_templates():
            """List available documentation templates."""
            templates_dir = Path("config/doc_review/outline_templates")
            templates = []
            
            if templates_dir.exists():
                for path in sorted(templates_dir.glob("*.json")):
                    templates.append({
                        "template_id": path.stem,
                        "path": str(path),
                        "size": path.stat().st_size,
                        "location": "config/doc_review/outline_templates",
                    })
            else:
                templates_dir = Path("external/doc_review/templates")
                if templates_dir.exists():
                    for path in sorted(templates_dir.glob("*.json")):
                        templates.append({
                            "template_id": path.stem,
                            "path": str(path),
                            "size": path.stat().st_size,
                            "location": "external/doc_review/templates",
                        })
            return jsonify({"templates": templates})

        @app.route("/api/model_doc/documents", methods=["GET", "POST"])
        @self.login_required
        def list_model_doc_documents():
            """List or register codebases."""
            if request.method == "GET":
                return jsonify({"codebases": self.store.list_codebases()})

            payload = request.get_json(force=True, silent=True) or {}
            codebase_path = payload.get("codebase_path", "").strip()
            if not codebase_path:
                return jsonify({"error": "codebase_path is required"}), 400

            path = Path(codebase_path).expanduser()
            if not path.exists():
                return jsonify({"error": f"Codebase directory not found at {path}"}), 400
            if not path.is_dir():
                return jsonify({"error": f"Path is not a directory: {path}"}), 400

            codebase_id = payload.get("codebase_id") or _slugify(path.name)
            if self.store.load(codebase_id):
                return jsonify({"error": "Codebase with this id already exists"}), 409

            overrides = payload.get("config") or {}
            config = self.agent.build_config(overrides)
            record = self.store.save(
                codebase_id,
                str(path.resolve()),
                state={"config": config},
                status="ready",
            )

            emitter = self._make_event_emitter(codebase_id)
            if emitter:
                emitter(
                    "status",
                    {
                        "status": "ready",
                        "message": "Codebase registered. Run the workflow to generate documentation.",
                    },
                )

            return jsonify({"codebase": record}), 201

        @app.route("/api/model_doc/documents/<codebase_id>", methods=["GET"])
        @self.login_required
        def get_codebase(codebase_id: str):
            """Get codebase details."""
            record = self.store.load(codebase_id)
            if not record:
                return jsonify({"error": "Codebase not found"}), 404
            return jsonify({"codebase": record})

        @app.route("/api/model_doc/documents/<codebase_id>/config", methods=["PATCH"])
        @self.login_required
        def update_codebase_config(codebase_id: str):
            """Update codebase configuration."""
            record = self.store.load(codebase_id)
            if not record:
                return jsonify({"error": "Codebase not found"}), 404

            payload = request.get_json(force=True, silent=True) or {}
            overrides = payload.get("config") if isinstance(payload, dict) else {}
            
            state = record.get("state", {}) or {}
            current_config = state.get("config", self.agent.build_config())
            updated_config = self.agent.build_config(overrides)
            
            state["config"] = updated_config
            record["state"] = state
            record["updated_at"] = datetime.utcnow().isoformat() + "Z"
            
            self.store.save(
                codebase_id,
                record.get("codebase_path", ""),
                state,
                record.get("status", "ready"),
            )
            
            return jsonify({"codebase": record})

        @app.route("/api/model_doc/documents/<codebase_id>/run", methods=["POST"])
        @self.login_required
        def run_workflow(codebase_id: str):
            """Execute the full model documentation workflow."""
            record = self.store.load(codebase_id)
            if not record:
                return jsonify({"error": "Codebase not found"}), 404

            body = request.get_json(force=True, silent=True) or {}
            overrides = body.get("config") if isinstance(body, dict) else {}
            state_payload = record.get("state", {})
            config_seed = state_payload.get("config", self.agent.build_config())
            if overrides:
                config_seed = self.agent.build_config(overrides)

            codebase_path = record.get("codebase_path")
            if not codebase_path:
                return jsonify({"error": "Stored codebase missing codebase_path"}), 500

            state_input = {
                "codebase_id": codebase_id,
                "codebase_path": codebase_path,
                "config": config_seed,
            }

            emitter = self._make_event_emitter(codebase_id)

            try:
                self.store.update_status(codebase_id, "running")
                final_state = self.agent.run(state_input, event_emitter=emitter)
                
                updated = self.store.save(codebase_id, codebase_path, final_state, status="completed")
                
                if emitter:
                    emitter(
                        "status",
                        {
                            "status": "completed",
                            "message": "Model documentation workflow completed",
                        },
                    )
                return jsonify({"codebase": updated})
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Workflow run failed for %s", codebase_id)
                self.store.update_status(codebase_id, "error")
                if emitter:
                    emitter(
                        "status",
                        {
                            "status": "error",
                            "message": str(exc),
                        },
                    )
                return jsonify({"error": str(exc)}), 500

        @app.route("/api/model_doc/documents/<codebase_id>/run_phase1", methods=["POST"])
        @self.login_required
        def model_doc_run_phase1_only(codebase_id: str):
            """Run only Phase 1 (Codebase Discovery) workflow."""
            record = self.store.load(codebase_id)
            if not record:
                return jsonify({"error": "Codebase not found"}), 404

            body = request.get_json(force=True, silent=True) or {}
            overrides = body.get("config") if isinstance(body, dict) else {}
            state_payload = record.get("state", {})
            config_seed = state_payload.get("config", self.agent.build_config())
            if overrides:
                config_seed = self.agent.build_config(overrides)

            codebase_path = record.get("codebase_path")
            if not codebase_path:
                return jsonify({"error": "Stored codebase missing codebase_path"}), 500

            state_input = {
                "codebase_id": codebase_id,
                "codebase_path": codebase_path,
                "config": config_seed,
            }

            emitter = self._make_event_emitter(codebase_id)

            try:
                self.store.update_status(codebase_id, "running")

                state = self.agent._initialise_state(state_input)
                if emitter:
                    emitter(
                        "status",
                        {
                            "status": "running",
                            "message": "Phase 1 workflow started",
                        },
                    )

                # Phase 1: Codebase Discovery (MCP Tools)
                state = self.agent._node_list_codebase_files(state)
                state = self.agent._node_read_code_files(state)
                state = self.agent._node_parse_code_structure(state)
                state = self.agent._node_build_file_hierarchy(state)
                state = self.agent._node_compute_code_stats(state)

                updated = self.store.save(codebase_id, codebase_path, state, status="ready")
                if emitter:
                    emitter(
                        "status",
                        {
                            "status": "ready",
                            "message": "Phase 1 completed successfully",
                        },
                    )
                return jsonify({"codebase": updated})
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Phase 1 run failed for %s", codebase_id)
                self.store.update_status(codebase_id, "error")
                if emitter:
                    emitter(
                        "status",
                        {
                            "status": "error",
                            "message": str(exc),
                        },
                    )
                return jsonify({"error": str(exc)}), 500

        @app.route("/api/model_doc/documents/<codebase_id>/status", methods=["GET"])
        @self.login_required
        def get_status(codebase_id: str):
            """Get codebase workflow status."""
            record = self.store.load(codebase_id)
            if not record:
                return jsonify({"error": "Codebase not found"}), 404
            
            state = record.get("state", {}) or {}
            return jsonify({
                "codebase_id": codebase_id,
                "status": record.get("status", "unknown"),
                "updated_at": record.get("updated_at"),
                "last_node": state.get("last_node"),
                "phase_stats": state.get("phase_stats", {}),
            })

        @app.route("/api/model_doc/chat/<codebase_id>", methods=["POST"])
        @self.login_required
        def chat_with_codebase(codebase_id: str):
            """Chat with codebase using LLM."""
            record = self.store.load(codebase_id)
            if not record:
                return jsonify({"error": "Codebase not found"}), 404

            payload = request.get_json(force=True, silent=True) or {}
            message = payload.get("message", "").strip()
            selection = payload.get("selection")

            if not message:
                return jsonify({"error": "message is required"}), 400

            state = record.get("state", {}) or {}

            try:
                response = generate_chat_reply(message, state, selection=selection)
                
                # Append to chat history
                self.store.append_chat_message(codebase_id, message, response, selection=selection)
                
                return jsonify({
                    "response": response,
                    "codebase_id": codebase_id,
                })
            except LLMNotAvailableError:
                return jsonify({"error": "LLM not configured"}), 503
            except Exception as exc:
                logger.exception("Chat failed for %s", codebase_id)
                return jsonify({"error": str(exc)}), 500

