from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from flask import Blueprint, jsonify, redirect, render_template, request, send_file, session, url_for

from .services import BulkDocService

logger = logging.getLogger(__name__)

# Global service instance (in production, this could be DB-backed, injected via DI, etc.)
_service = None


def get_service() -> BulkDocService:
    """Get or create the service instance."""
    global _service
    if _service is None:
        storage_base = Path("data/ai_bulk_doc_analysis")
        _service = BulkDocService(storage_base=storage_base)
    return _service


def create_bulk_doc_blueprint(auth_manager) -> Blueprint:
    """
    Feature-isolated blueprint for the AI Bulk Doc Analysis UI.

    Notes:
    - We intentionally keep all new UI code under `external/ai_bulk_doc_analysis/`.
    - Core app only registers this blueprint (minimal wiring).
    """
    bp = Blueprint(
        "bulk_doc",
        __name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/bulk-doc-static",
    )

    def _current_user_session() -> Optional[dict]:
        token = session.get("token")
        if not token:
            return None
        try:
            return auth_manager.validate_session(token)
        except Exception:
            return None

    def _ensure_session(user_id: str) -> str:
        """Ensure a Bulk Doc session exists for the user."""
        svc = get_service()
        return svc.ensure_session(user_id)

    @bp.route("/bulk-doc-analysis", methods=["GET"])
    def bulk_doc_home():
        user_session = _current_user_session()
        if not user_session:
            return redirect(url_for("login", next=url_for("bulk_doc.bulk_doc_home")))

        user_id = user_session.get("user_id") or "anonymous"
        bulk_session_id = _ensure_session(user_id)

        return render_template(
            "bulk_doc_analysis.html",
            user=user_session,
            bulk_doc_ui_session_id=bulk_session_id,
        )

    # ==================== API Endpoints ====================

    @bp.route("/api/bulk-doc-analysis/documents/upload", methods=["POST"])
    def api_upload_documents():
        """Upload PDF documents."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401

        user_id = user_session.get("user_id") or "anonymous"
        bulk_session_id = _ensure_session(user_id)

        if "files" not in request.files:
            return jsonify({"error": "No files provided"}), 400

        files = request.files.getlist("files")
        if not files or all(f.filename == "" for f in files):
            return jsonify({"error": "No files selected"}), 400

        # Validate PDFs only (per spec)
        pdfs = []
        for f in files:
            if not f.filename:
                continue
            if not f.filename.lower().endswith(".pdf"):
                return jsonify({"error": f"Only PDF files allowed: {f.filename}"}), 400
            pdfs.append(f)

        if not pdfs:
            return jsonify({"error": "No valid PDF files"}), 400

        try:
            svc = get_service()
            docs = svc.create_documents(bulk_session_id, pdfs, user_id)
            return jsonify({
                "success": True,
                "documents": [d.to_dict() for d in docs],
            })
        except Exception as e:
            logger.error(f"Upload error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/documents", methods=["GET"])
    def api_list_documents():
        """List documents in the current session."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401

        user_id = user_session.get("user_id") or "anonymous"
        bulk_session_id = _ensure_session(user_id)

        try:
            svc = get_service()
            docs = svc.list_documents(bulk_session_id)
            return jsonify({
                "documents": [d.to_dict() for d in docs],
            })
        except Exception as e:
            logger.error(f"List docs error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/documents/<doc_id>", methods=["DELETE"])
    def api_delete_document(doc_id: str):
        """Delete an errored document."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401

        try:
            svc = get_service()
            success = svc.delete_document(doc_id)
            if not success:
                return jsonify({"error": "Document not found"}), 404
            return jsonify({"success": True})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Delete doc error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/chains", methods=["GET"])
    def api_list_chains():
        """List available prompt chains."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401

        try:
            svc = get_service()
            chains = svc.list_chains()
            return jsonify({
                "chains": [c.to_dict() for c in chains],
            })
        except Exception as e:
            logger.error(f"List chains error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/chains", methods=["POST"])
    def api_create_chain():
        """Create a new prompt chain."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401

        user_id = user_session.get("user_id") or "anonymous"
        data = request.get_json() or {}

        name = data.get("name", "").strip()
        description = data.get("description", "").strip()
        steps = data.get("steps", [])

        if not name:
            return jsonify({"error": "Chain name is required"}), 400

        if not steps or len(steps) == 0:
            return jsonify({"error": "At least one step is required"}), 400

        try:
            svc = get_service()
            chain = svc.create_chain(user_id, name, description, steps)
            return jsonify({
                "success": True,
                "chain": chain.to_dict(),
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Create chain error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/chains/<chain_id>", methods=["PUT"])
    def api_update_chain(chain_id: str):
        """Update an existing prompt chain."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json() or {}

        name = data.get("name", "").strip()
        description = data.get("description", "").strip()
        steps = data.get("steps", [])

        if not name:
            return jsonify({"error": "Chain name is required"}), 400

        if not steps or len(steps) == 0:
            return jsonify({"error": "At least one step is required"}), 400

        try:
            svc = get_service()
            chain = svc.update_chain(chain_id, name, description, steps)
            return jsonify({
                "success": True,
                "chain": chain.to_dict(),
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Update chain error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/runs", methods=["POST"])
    def api_create_run():
        """Create a new run."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401

        user_id = user_session.get("user_id") or "anonymous"
        bulk_session_id = _ensure_session(user_id)

        data = request.get_json() or {}
        chain_version_id = data.get("chain_version_id")
        if not chain_version_id:
            return jsonify({"error": "chain_version_id required"}), 400

        try:
            svc = get_service()
            run = svc.create_run(bulk_session_id, chain_version_id)
            return jsonify({
                "success": True,
                "run": run.to_dict(),
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Create run error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/runs/<run_id>/progress", methods=["GET"])
    def api_get_run_progress(run_id: str):
        """Get run progress."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401

        try:
            svc = get_service()
            progress = svc.get_run_progress(run_id)
            if not progress:
                return jsonify({"error": "Run not found"}), 404
            return jsonify(progress)
        except Exception as e:
            logger.error(f"Get run progress error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/runs/<run_id>/download/<doc_id>", methods=["GET"])
    def api_download_document_output(run_id: str, doc_id: str):
        """Download final output for a document."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401

        try:
            svc = get_service()
            doc = svc.get_document(doc_id)
            if not doc or not doc.converted_md_path:
                return jsonify({"error": "Output not found"}), 404

            # For now, return the converted markdown file
            # Later: return final R(N) output from object storage
            md_path = Path(doc.converted_md_path)
            if not md_path.exists():
                return jsonify({"error": "Output file not found"}), 404

            return send_file(
                str(md_path),
                mimetype="text/markdown",
                as_attachment=True,
                download_name=f"{doc.original_filename}.md",
            )
        except Exception as e:
            logger.error(f"Download error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    return bp


