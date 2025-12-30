from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from flask import Blueprint, jsonify, redirect, render_template, request, send_file, session, url_for

from .services import BulkDocService
from .queue_config import get_conversion_queue, get_execution_queue, init_queues
from .workers.rq_worker import convert_doc_job, execute_step_job

logger = logging.getLogger(__name__)

# Try to initialize queues (will fail gracefully if Redis not available)
USE_QUEUES = os.getenv("REDIS_URL") is not None
if USE_QUEUES:
    try:
        init_queues()
        logger.info("Queue system enabled (Redis available)")
    except Exception as e:
        logger.warning(f"Queue system disabled (Redis not available): {e}")
        USE_QUEUES = False
else:
    logger.info("Queue system disabled (REDIS_URL not set)")

# Global service instance (in production, this could be DB-backed, injected via DI, etc.)
_service = None


def get_service():
    """Get or create the service instance (DB-backed if DATABASE_URL available, else in-memory)."""
    global _service
    if _service is None:
        # Get project root (go up from web/ to project root)
        project_root = Path(__file__).parent.parent.parent
        storage_base = project_root / "data" / "ai_bulk_doc_analysis"
        
        # Try to use DB service if DATABASE_URL is available
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            try:
                from .db_service import BulkDocDBService, init_db
                # Force initialization and table creation
                init_db()
                _service = BulkDocDBService(storage_base=storage_base)
                logger.info("Using DB-backed service (PostgreSQL)")
            except Exception as e:
                logger.warning(f"DB service unavailable, falling back to in-memory: {e}", exc_info=True)
                _service = BulkDocService(storage_base=storage_base)
        else:
            _service = BulkDocService(storage_base=storage_base)
            logger.info("Using in-memory service (DATABASE_URL not set)")
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

    def _ensure_session(user_id: str, session_id: Optional[str] = None) -> str:
        """Get or create a Bulk Doc session for the user."""
        svc = get_service()
        if session_id:
            return svc.ensure_session(user_id, session_id)
        # Check if there's a current session in Flask session
        current_session_id = session.get("bulk_doc_session_id")
        if current_session_id:
            try:
                return svc.ensure_session(user_id, current_session_id)
            except ValueError:
                # Session doesn't exist, create new one
                pass
        # Create new session and store in Flask session
        new_session_id = svc.create_session(user_id)
        session["bulk_doc_session_id"] = new_session_id
        return new_session_id

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
        """Upload documents (multiple file types supported)."""
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

        # Get workflow_version_id to validate file types
        workflow_version_id = request.form.get("workflow_version_id")
        ingestion_profile_id = None
        
        if workflow_version_id:
            # Get workflow version to check accepted input types
            try:
                from .workflow_service import WorkflowService
                workflow_service = WorkflowService()
                workflow_version = workflow_service.get_workflow_version(workflow_version_id)
                
                if not workflow_version:
                    return jsonify({"error": "Workflow version not found"}), 404
                
                ingestion_profile_id = workflow_version.ingestion_profile_id
                
                # Get ingestion profile
                from .ingestion_service import IngestionService
                ingestion_service = IngestionService()
                ingestion_profile = ingestion_service.get_ingestion_profile(ingestion_profile_id)
                
                if not ingestion_profile:
                    return jsonify({"error": "Ingestion profile not found"}), 404
                
                accepted_types = ingestion_profile.accepted_input_types
            except Exception as e:
                logger.error(f"Error getting workflow version: {e}", exc_info=True)
                return jsonify({"error": "Invalid workflow_version_id"}), 400
        else:
            # Fallback: accept PDF only (backward compatibility)
            accepted_types = ["PDF"]

        # Validate file types
        valid_extensions = {
            "PDF": [".pdf"],
            "DOCX": [".docx"],
            "TXT": [".txt"],
            "MD": [".md"],
            "CSV": [".csv"]
        }
        
        accepted_extensions = []
        for file_type in accepted_types:
            accepted_extensions.extend(valid_extensions.get(file_type, []))
        
        valid_files = []
        for f in files:
            if not f.filename:
                continue
            ext = Path(f.filename).suffix.lower()
            if ext not in accepted_extensions:
                return jsonify({"error": f"File type not accepted: {f.filename}. Accepted types: {accepted_types}"}), 400
            valid_files.append(f)

        if not valid_files:
            return jsonify({"error": "No valid files"}), 400

        try:
            svc = get_service()
            docs = svc.create_documents(bulk_session_id, valid_files, user_id)
            
            # Enqueue conversion jobs if queues are enabled
            if USE_QUEUES:
                try:
                    conversion_queue = get_conversion_queue()
                    for doc in docs:
                        # Get file path - check if we have it stored
                        doc_dir = svc.storage_base / "docs" / doc.doc_id
                        file_path = None
                        for ext in accepted_extensions:
                            for f in doc_dir.glob(f"*{ext}"):
                                file_path = f
                                break
                            if file_path:
                                break
                        
                        if not file_path:
                            logger.warning(f"File not found for doc {doc.doc_id}, skipping queue")
                            continue
                        
                        # Use relative path from storage_base
                        object_storage_key = str(file_path.relative_to(svc.storage_base))
                        
                        job_data = {
                            "doc_id": doc.doc_id,
                            "session_id": bulk_session_id,
                            "object_storage_key": object_storage_key,
                            "ingestion_profile_id": ingestion_profile_id,  # NEW
                            "idempotency_key": f"convert:{doc.doc_id}",
                        }
                        from external.ai_bulk_doc_analysis.workers.rq_worker import convert_doc_job
                        conversion_queue.enqueue(convert_doc_job, job_data, job_id=f"convert_{doc.doc_id}")
                        logger.info(f"Enqueued conversion job for doc {doc.doc_id}")
                except Exception as e:
                    logger.warning(f"Failed to enqueue conversion jobs, will use synchronous execution: {e}", exc_info=True)
            
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

    @bp.route("/api/bulk-doc-analysis/sessions", methods=["GET"])
    def api_list_sessions():
        """List all sessions for the current user."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401

        user_id = user_session.get("user_id") or "anonymous"
        try:
            svc = get_service()
            sessions = svc.list_sessions(user_id)
            return jsonify({"sessions": sessions})
        except Exception as e:
            logger.error(f"List sessions error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/sessions", methods=["POST"])
    def api_create_session():
        """Create a new session for the user."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401

        user_id = user_session.get("user_id") or "anonymous"
        data = request.get_json() or {}
        session_name = data.get("name")

        try:
            svc = get_service()
            new_session_id = svc.create_session(user_id)
            
            # Optionally set name in metadata
            if session_name:
                from .db_service import get_db_session, init_db
                from .models import Session as DBSession
                init_db()  # Ensure DB is initialized
                with get_db_session() as db:
                    db_session = db.query(DBSession).filter(DBSession.session_id == new_session_id).first()
                    if db_session:
                        if not isinstance(db_session.metadata_json, dict):
                            db_session.metadata_json = {}
                        db_session.metadata_json["name"] = session_name
                        db.commit()
            
            # Set as current session
            session["bulk_doc_session_id"] = new_session_id
            
            return jsonify({
                "success": True,
                "session_id": new_session_id,
            })
        except Exception as e:
            logger.error(f"Create session error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/sessions/<session_id>/select", methods=["POST"])
    def api_select_session(session_id: str):
        """Select/switch to a different session."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401

        user_id = user_session.get("user_id") or "anonymous"
        try:
            svc = get_service()
            # Validate session belongs to user
            svc.ensure_session(user_id, session_id)
            # Set as current session
            session["bulk_doc_session_id"] = session_id
            return jsonify({"success": True, "session_id": session_id})
        except ValueError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            logger.error(f"Select session error: {e}", exc_info=True)
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

    # ==================== Workflow APIs (New) ====================
    
    @bp.route("/api/bulk-doc-analysis/workflows", methods=["GET"])
    def api_list_workflows():
        """List workflows (domain-filtered)."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401

        user_id = user_session.get("user_id") or "anonymous"
        
        # Get user domains from session
        user_domains = user_session.get("domains", [])
        is_super_admin = auth_manager.is_super_admin(user_session)
        
        try:
            from .workflow_service import WorkflowService
            workflow_service = WorkflowService()
            workflows = workflow_service.list_workflows(user_id, user_domains, is_super_admin)
            return jsonify({"workflows": workflows})
        except Exception as e:
            logger.error(f"List workflows error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/workflows", methods=["POST"])
    def api_create_workflow():
        """Create a new workflow."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401
        
        user_id = user_session.get("user_id") or "anonymous"
        data = request.get_json() or {}
        
        name = data.get("name", "").strip()
        description = data.get("description", "").strip()
        domains = data.get("domains", [])
        ingestion_profile_id = data.get("ingestion_profile_id")
        chain_version_id = data.get("chain_version_id")
        export_profile_id = data.get("export_profile_id")
        
        # Validation
        if not name:
            return jsonify({"error": "Name is required"}), 400
        if not description:
            return jsonify({"error": "Description is required"}), 400
        if len(description) < 20 or len(description) > 240:
            return jsonify({"error": "Description must be 20-240 characters"}), 400
        if not domains or len(domains) == 0:
            return jsonify({"error": "At least one domain is required"}), 400
        if not ingestion_profile_id:
            return jsonify({"error": "ingestion_profile_id is required"}), 400
        if not chain_version_id:
            return jsonify({"error": "chain_version_id is required"}), 400
        if not export_profile_id:
            return jsonify({"error": "export_profile_id is required"}), 400
        
        try:
            from .workflow_service import WorkflowService
            workflow_service = WorkflowService()
            workflow = workflow_service.create_workflow(
                user_id=user_id,
                name=name,
                description=description,
                domains=domains,
                ingestion_profile_id=ingestion_profile_id,
                chain_version_id=chain_version_id,
                export_profile_id=export_profile_id
            )
            return jsonify({
                "success": True,
                "workflow": {
                    "workflow_id": workflow.workflow_id,
                    "name": workflow.name,
                    "description": workflow.description,
                }
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Create workflow error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/workflows/<workflow_id>", methods=["GET"])
    def api_get_workflow(workflow_id: str):
        """Get workflow by ID."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401

        try:
            from .workflow_service import WorkflowService
            from .db_service import get_db_session
            from .models import WorkflowDomain
            
            workflow_service = WorkflowService()
            workflow = workflow_service.get_workflow(workflow_id)
            
            if not workflow:
                return jsonify({"error": "Workflow not found"}), 404
            
            # Get domains
            with get_db_session() as db:
                domains = [wd.domain for wd in db.query(WorkflowDomain).filter(
                    WorkflowDomain.workflow_id == workflow_id
                ).all()]
            
            return jsonify({
                "workflow_id": workflow.workflow_id,
                "name": workflow.name,
                "description": workflow.description,
                "domains": domains,
                "created_at": workflow.created_at.isoformat() if workflow.created_at else None,
                "updated_at": workflow.updated_at.isoformat() if workflow.updated_at else None,
            })
        except Exception as e:
            logger.error(f"Get workflow error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/workflows/<workflow_id>", methods=["PUT"])
    def api_update_workflow(workflow_id: str):
        """Update workflow (creates new version)."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401

        user_id = user_session.get("user_id") or "anonymous"
        data = request.get_json() or {}

        name = data.get("name", "").strip()
        description = data.get("description", "").strip()
        domains = data.get("domains", [])
        ingestion_profile_id = data.get("ingestion_profile_id")
        chain_version_id = data.get("chain_version_id")
        export_profile_id = data.get("export_profile_id")

        # Validation
        if not name:
            return jsonify({"error": "Name is required"}), 400
        if not description:
            return jsonify({"error": "Description is required"}), 400
        if len(description) < 20 or len(description) > 240:
            return jsonify({"error": "Description must be 20-240 characters"}), 400
        if not domains or len(domains) == 0:
            return jsonify({"error": "At least one domain is required"}), 400
        if not ingestion_profile_id:
            return jsonify({"error": "ingestion_profile_id is required"}), 400
        if not chain_version_id:
            return jsonify({"error": "chain_version_id is required"}), 400
        if not export_profile_id:
            return jsonify({"error": "export_profile_id is required"}), 400
        
        try:
            from .workflow_service import WorkflowService
            from .domain_service import DomainService
            
            workflow_service = WorkflowService()
            workflow = workflow_service.get_workflow(workflow_id)
            
            if not workflow:
                return jsonify({"error": "Workflow not found"}), 404
            
            # Check domain access
            if not DomainService.can_edit_workflow(user_session, workflow):
                return jsonify({"error": "Access denied"}), 403
            
            workflow_version = workflow_service.update_workflow(
                workflow_id=workflow_id,
                name=name,
                description=description,
                domains=domains,
                ingestion_profile_id=ingestion_profile_id,
                chain_version_id=chain_version_id,
                export_profile_id=export_profile_id
            )
            return jsonify({
                "success": True,
                "workflow_version": {
                    "workflow_version_id": workflow_version.workflow_version_id,
                    "workflow_id": workflow_version.workflow_id,
                    "version_number": workflow_version.version_number,
                }
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Update workflow error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/workflows/<workflow_id>", methods=["DELETE"])
    def api_delete_workflow(workflow_id: str):
        """Delete workflow."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401
        
        try:
            from .workflow_service import WorkflowService
            from .domain_service import DomainService
            
            workflow_service = WorkflowService()
            workflow = workflow_service.get_workflow(workflow_id)
            
            if not workflow:
                return jsonify({"error": "Workflow not found"}), 404
            
            # Check domain access
            if not DomainService.can_edit_workflow(user_session, workflow):
                return jsonify({"error": "Access denied"}), 403
            
            success = workflow_service.delete_workflow(workflow_id)
            
            if not success:
                return jsonify({"error": "Workflow not found"}), 404
            
            return jsonify({"success": True})
        except Exception as e:
            logger.error(f"Delete workflow error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    # ==================== Ingestion Profile APIs ====================
    
    @bp.route("/api/bulk-doc-analysis/ingestion-profiles", methods=["GET"])
    def api_list_ingestion_profiles():
        """List all ingestion profiles."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401
        
        try:
            from .ingestion_service import IngestionService
            ingestion_service = IngestionService()
            profiles = ingestion_service.list_ingestion_profiles()
            return jsonify({
                "profiles": [{
                    "ingestion_profile_id": p.ingestion_profile_id,
                    "name": p.name,
                    "accepted_input_types": p.accepted_input_types,
                    "mode": p.mode,
                    "has_vision_prompt": p.vision_prompt is not None,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                } for p in profiles]
            })
        except Exception as e:
            logger.error(f"List ingestion profiles error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/ingestion-profiles", methods=["POST"])
    def api_create_ingestion_profile():
        """Create a new ingestion profile."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json() or {}

        name = data.get("name", "").strip()
        accepted_input_types = data.get("accepted_input_types", [])
        mode = data.get("mode", "programmatic")
        vision_prompt = data.get("vision_prompt")

        # Validation
        if not name:
            return jsonify({"error": "Name is required"}), 400
        if not accepted_input_types or len(accepted_input_types) == 0:
            return jsonify({"error": "At least one accepted_input_type is required"}), 400
        if mode == 'vision' and not vision_prompt:
            return jsonify({"error": "vision_prompt is required when mode='vision'"}), 400
        
        try:
            from .ingestion_service import IngestionService
            ingestion_service = IngestionService()
            profile = ingestion_service.create_ingestion_profile(
                name=name,
                accepted_input_types=accepted_input_types,
                mode=mode,
                vision_prompt=vision_prompt
            )
            return jsonify({
                "success": True,
                "profile": {
                    "ingestion_profile_id": profile.ingestion_profile_id,
                    "name": profile.name,
                    "accepted_input_types": profile.accepted_input_types,
                    "mode": profile.mode,
                }
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Create ingestion profile error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/ingestion-profiles/<profile_id>", methods=["GET"])
    def api_get_ingestion_profile(profile_id: str):
        """Get ingestion profile by ID."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401
        
        try:
            from .ingestion_service import IngestionService
            ingestion_service = IngestionService()
            profile = ingestion_service.get_ingestion_profile(profile_id)
            
            if not profile:
                return jsonify({"error": "Ingestion profile not found"}), 404
            
            return jsonify({
                "ingestion_profile_id": profile.ingestion_profile_id,
                "name": profile.name,
                "accepted_input_types": profile.accepted_input_types,
                "mode": profile.mode,
                "vision_prompt": profile.vision_prompt,  # Include prompt in response
                "created_at": profile.created_at.isoformat() if profile.created_at else None,
            })
        except Exception as e:
            logger.error(f"Get ingestion profile error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/ingestion-profiles/<profile_id>", methods=["PUT"])
    def api_update_ingestion_profile(profile_id: str):
        """Update ingestion profile."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401
        
        data = request.get_json() or {}
        
        name = data.get("name", "").strip()
        accepted_input_types = data.get("accepted_input_types", [])
        mode = data.get("mode")
        vision_prompt = data.get("vision_prompt")
        
        try:
            from .ingestion_service import IngestionService
            from .db_service import get_db_session
            from .models import IngestionProfile
            
            ingestion_service = IngestionService()
            profile = ingestion_service.get_ingestion_profile(profile_id)
            
            if not profile:
                return jsonify({"error": "Ingestion profile not found"}), 404
            
            # Update fields
            with get_db_session() as db:
                db_profile = db.query(IngestionProfile).filter(
                    IngestionProfile.ingestion_profile_id == profile_id
                ).first()
                
                if name:
                    db_profile.name = name
                if accepted_input_types:
                    db_profile.accepted_input_types = accepted_input_types
                if mode:
                    if mode == 'vision' and not vision_prompt and not db_profile.vision_prompt:
                        return jsonify({"error": "vision_prompt is required when mode='vision'"}), 400
                    db_profile.mode = mode
                if vision_prompt is not None:
                    db_profile.vision_prompt = vision_prompt
                
                db.commit()
            
            return jsonify({"success": True})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Update ingestion profile error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/ingestion-profiles/<profile_id>", methods=["DELETE"])
    def api_delete_ingestion_profile(profile_id: str):
        """Delete ingestion profile."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401
        
        try:
            from .db_service import get_db_session
            from .models import IngestionProfile
            
            with get_db_session() as db:
                profile = db.query(IngestionProfile).filter(
                    IngestionProfile.ingestion_profile_id == profile_id
                ).first()
                
                if not profile:
                    return jsonify({"error": "Ingestion profile not found"}), 404
                
                db.delete(profile)
                db.commit()
            
            return jsonify({"success": True})
        except Exception as e:
            logger.error(f"Delete ingestion profile error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    # ==================== Export Profile APIs ====================
    
    @bp.route("/api/bulk-doc-analysis/export-profiles", methods=["GET"])
    def api_list_export_profiles():
        """List all export profiles."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401
        
        try:
            from .export_service import ExportService
            export_service = ExportService()
            profiles = export_service.list_export_profiles()
            return jsonify({
                "profiles": [{
                    "export_profile_id": p.export_profile_id,
                    "name": p.name,
                    "format": p.format,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                } for p in profiles]
            })
        except Exception as e:
            logger.error(f"List export profiles error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/export-profiles", methods=["POST"])
    def api_create_export_profile():
        """Create a new export profile."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401
        
        data = request.get_json() or {}
        
        name = data.get("name", "").strip()
        format = data.get("format", "MD")
        config_json = data.get("config_json", {})
        
        # Validation
        if not name:
            return jsonify({"error": "Name is required"}), 400
        valid_formats = ['CSV', 'JSON', 'MD', 'DOCX', 'PDF']
        if format not in valid_formats:
            return jsonify({"error": f"Format must be one of: {valid_formats}"}), 400
        
        try:
            from .export_service import ExportService
            export_service = ExportService()
            profile = export_service.create_export_profile(
                name=name,
                format=format,
                config_json=config_json
            )
            return jsonify({
                "success": True,
                "profile": {
                    "export_profile_id": profile.export_profile_id,
                    "name": profile.name,
                    "format": profile.format,
                }
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Create export profile error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/export-profiles/<profile_id>", methods=["GET"])
    def api_get_export_profile(profile_id: str):
        """Get export profile by ID."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401
        
        try:
            from .export_service import ExportService
            export_service = ExportService()
            profile = export_service.get_export_profile(profile_id)
            
            if not profile:
                return jsonify({"error": "Export profile not found"}), 404
            
            return jsonify({
                "export_profile_id": profile.export_profile_id,
                "name": profile.name,
                "format": profile.format,
                "config_json": profile.config_json,
                "created_at": profile.created_at.isoformat() if profile.created_at else None,
            })
        except Exception as e:
            logger.error(f"Get export profile error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/export-profiles/<profile_id>", methods=["PUT"])
    def api_update_export_profile(profile_id: str):
        """Update export profile."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401
        
        data = request.get_json() or {}
        
        name = data.get("name", "").strip()
        format = data.get("format")
        config_json = data.get("config_json")
        
        try:
            from .export_service import ExportService
            from .db_service import get_db_session
            from .models import ExportProfile
            
            export_service = ExportService()
            profile = export_service.get_export_profile(profile_id)
            
            if not profile:
                return jsonify({"error": "Export profile not found"}), 404
            
            # Update fields
            with get_db_session() as db:
                db_profile = db.query(ExportProfile).filter(
                    ExportProfile.export_profile_id == profile_id
                ).first()
                
                if name:
                    db_profile.name = name
                if format:
                    valid_formats = ['CSV', 'JSON', 'MD', 'DOCX', 'PDF']
                    if format not in valid_formats:
                        return jsonify({"error": f"Format must be one of: {valid_formats}"}), 400
                    db_profile.format = format
                if config_json is not None:
                    db_profile.config_json = config_json
                
                db.commit()
            
            return jsonify({"success": True})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Update export profile error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/export-profiles/<profile_id>", methods=["DELETE"])
    def api_delete_export_profile(profile_id: str):
        """Delete export profile."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401
        
        try:
            from .db_service import get_db_session
            from .models import ExportProfile
            
            with get_db_session() as db:
                profile = db.query(ExportProfile).filter(
                    ExportProfile.export_profile_id == profile_id
                ).first()
                
                if not profile:
                    return jsonify({"error": "Export profile not found"}), 404
                
                db.delete(profile)
                db.commit()
            
            return jsonify({"success": True})
        except Exception as e:
            logger.error(f"Delete export profile error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    # ==================== Chain APIs (DEPRECATED - Remove in Phase 1) ====================
    # Note: These endpoints are kept temporarily for backward compatibility but will be removed
    # Users should migrate to workflow system

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
        workflow_version_id = data.get("workflow_version_id")  # NEW
        
        if not chain_version_id:
            return jsonify({"error": "chain_version_id required"}), 400

        try:
            from .db_service import BulkDocDBService
            svc = BulkDocDBService()
            run = svc.create_run(bulk_session_id, chain_version_id, workflow_version_id=workflow_version_id)
            
            # Check if CSV workflow
            is_csv_workflow = False
            if workflow_version_id:
                from .workflow_service import WorkflowService
                from .ingestion_service import IngestionService
                workflow_service = WorkflowService()
                workflow_version = workflow_service.get_workflow_version(workflow_version_id)
                if workflow_version:
                    ingestion_service = IngestionService()
                    ingestion_profile = ingestion_service.get_ingestion_profile(workflow_version.ingestion_profile_id)
                    if ingestion_profile and 'CSV' in ingestion_profile.accepted_input_types:
                        is_csv_workflow = True
            
            # Enqueue execution jobs if queues are enabled
            if USE_QUEUES:
                try:
                    execution_queue = get_execution_queue()
                    chain = svc.get_chain(chain_version_id)
                    if not chain:
                        return jsonify({"error": "Chain not found"}), 404
                    
                    if is_csv_workflow:
                        # CSV: Enqueue jobs for each task/step
                        from .db_service import get_db_session
                        from .models import ExecutionTask
                        
                        with get_db_session() as db:
                            tasks = db.query(ExecutionTask).filter(ExecutionTask.run_id == run.run_id).all()
                            
                            for task in tasks:
                                for step_idx in range(1, chain.step_count + 1):
                                    step_def = next((s for s in chain.steps if s.get("index") == step_idx), None)
                                    if not step_def:
                                        continue
                                    
                                    model_config = step_def.get("model_config", {
                                        "model": "claude-3-haiku-20240307",
                                        "max_tokens": 4096,
                                        "temperature": 0.2
                                    })
                                    
                                    job_data = {
                                        "run_id": run.run_id,
                                        "doc_id": task.doc_id,
                                        "task_id": task.task_id,  # NEW
                                        "step_index": step_idx,
                                        "chain_version_id": chain_version_id,
                                        "required_inputs": step_def.get("required_inputs", []),
                                        "prompt": step_def.get("prompt", ""),
                                        "model_config": model_config,
                                        "idempotency_key": f"step:{run.run_id}:{task.doc_id}:{task.task_id}:{step_idx}",
                                    }
                                    execution_queue.enqueue(
                                        execute_step_job,
                                        job_data,
                                        job_id=f"execute_{run.run_id}_{task.doc_id}_{task.task_id}_step{step_idx}"
                                    )
                    else:
                        # Non-CSV: Enqueue EXECUTE_STEP job for each doc/step
                        for doc_id in run.document_ids:
                            for step_idx in range(1, chain.step_count + 1):
                                step_def = next((s for s in chain.steps if s.get("index") == step_idx), None)
                                if not step_def:
                                    continue
                                
                                # Get model_config from step (with defaults)
                                model_config = step_def.get("model_config", {
                                    "model": "claude-3-haiku-20240307",
                                    "max_tokens": 4096,
                                    "temperature": 0.2
                                })
                                
                                job_data = {
                                    "run_id": run.run_id,
                                    "doc_id": doc_id,
                                    "step_index": step_idx,
                                    "chain_version_id": chain_version_id,
                                    "required_inputs": step_def.get("required_inputs", []),
                                    "prompt": step_def.get("prompt", ""),
                                    "model_config": model_config,
                                    "idempotency_key": f"step:{run.run_id}:{doc_id}:{step_idx}",
                                }
                                execution_queue.enqueue(
                                    execute_step_job,
                                    job_data,
                                    job_id=f"exec_{run.run_id}_{doc_id}_step{step_idx}"
                                )
                                logger.info(f"Enqueued execution job for {run.run_id}/{doc_id}/step{step_idx}")
                except Exception as e:
                    logger.warning(f"Failed to enqueue execution jobs, will use synchronous execution: {e}")
            
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
        """Download final output (R(N)) for a document."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401

        try:
            svc = get_service()
            run = svc.runs.get(run_id)
            if not run:
                return jsonify({"error": "Run not found"}), 404

            doc = svc.get_document(doc_id)
            if not doc:
                return jsonify({"error": "Document not found"}), 404

            chain = svc.get_chain(run.chain_version_id)
            if not chain:
                return jsonify({"error": "Chain not found"}), 404

            # Get final step output (R(N))
            output_path = svc.get_final_output_path(run_id, doc_id, chain)
            if not output_path or not output_path.exists():
                # Fallback to converted markdown if no step output exists
                if doc.converted_md_path:
                    md_path = Path(doc.converted_md_path)
                    if md_path.exists():
                        output_path = md_path
                    else:
                        return jsonify({"error": "Output file not found"}), 404
                else:
                    return jsonify({"error": "Output not found"}), 404

            # Determine filename
            base_name = Path(doc.original_filename).stem
            download_name = f"{base_name}_output.md"

            return send_file(
                str(output_path),
                mimetype="text/markdown",
                as_attachment=True,
                download_name=download_name,
            )
        except Exception as e:
            logger.error(f"Download error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/runs/<run_id>/download-all", methods=["GET"])
    def api_download_all_outputs(run_id: str):
        """Download all document outputs for a run (format based on workflow export profile)."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401

        try:
            import zipfile
            import io
            
            svc = get_service()
            # Get run - handle both DB and in-memory services
            run = None
            doc_ids = []
            
            # Try DB service method first
            if hasattr(svc, 'get_run'):
                try:
                    run = svc.get_run(run_id)
                    if run:
                        doc_ids = getattr(run, 'document_ids', [])
                except:
                    pass
            
            # Fallback to in-memory service
            if not run and hasattr(svc, 'runs'):
                run = svc.runs.get(run_id)
                if run:
                    doc_ids = getattr(run, 'document_ids', [])
            
            if not run:
                return jsonify({"error": "Run not found"}), 404
            
            # Get chain
            chain_version_id = getattr(run, 'chain_version_id', None)
            if not chain_version_id:
                return jsonify({"error": "Run missing chain_version_id"}), 400
            
            chain = svc.get_chain(chain_version_id)
            if not chain:
                return jsonify({"error": "Chain not found"}), 404
            
            if not doc_ids:
                return jsonify({"error": "No documents found for this run"}), 404

            # Create ZIP in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for doc_id in doc_ids:
                    doc = svc.get_document(doc_id)
                    if not doc:
                        continue
                    
                    # Get final output path
                    output_path = svc.get_final_output_path(run_id, doc_id, chain)
                    
                    if output_path and output_path.exists():
                        # Add to ZIP with readable filename
                        base_name = Path(doc.original_filename).stem
                        zip_filename = f"{base_name}_output.md"
                        zip_file.write(str(output_path), zip_filename)

            zip_buffer.seek(0)

            # Return ZIP file
            return send_file(
                zip_buffer,
                mimetype="application/zip",
                as_attachment=True,
                download_name=f"run_{run_id}_outputs.zip"
            )

        except Exception as e:
            logger.error(f"Bulk download error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    return bp


