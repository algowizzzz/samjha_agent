from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from flask import Blueprint, jsonify, redirect, render_template, request, send_file, session, url_for

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
    """Get or create the DB-backed service instance (DATABASE_URL required)."""
    global _service
    if _service is None:
        # Get project root (go up from web/ to project root)
        project_root = Path(__file__).parent.parent.parent
        storage_base = project_root / "data" / "ai_bulk_doc_analysis"
        
        # Require DATABASE_URL - fail fast if not set
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL environment variable is required. "
                "Please set DATABASE_URL to use the bulk document analysis feature."
            )
        
        try:
            from .db_service import BulkDocDBService, init_db
            # Force initialization and table creation
            init_db()
            _service = BulkDocDBService(storage_base=storage_base)
            logger.info("Using DB-backed service (PostgreSQL)")
        except Exception as e:
            logger.error(f"Failed to initialize DB service: {e}", exc_info=True)
            raise RuntimeError(
                f"Database service unavailable: {e}. "
                "Please ensure DATABASE_URL is correct and database is accessible."
            ) from e
    return _service


def create_bulk_doc_blueprint(auth_manager) -> Blueprint:
    # CRITICAL: Initialize database before blueprint routes
    # This ensures DATABASE_URL is loaded and tables exist
    try:
        from .db_service import init_db, get_db_session
        from .models import Base
        from sqlalchemy import inspect
        
        init_db()
        
        # Verify tables exist, create if missing
        with get_db_session() as db:
            inspector = inspect(db.bind)
            existing_tables = inspector.get_table_names()
            if 'chains' not in existing_tables:
                logger.warning("Tables missing, creating them...")
                Base.metadata.create_all(bind=db.bind, checkfirst=True)
            logger.info(f"Database initialized with {len(existing_tables)} tables")
    except Exception as e:
        logger.error(f"Database initialization error: {e}", exc_info=True)
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

    # ==================== Test Files API (Automation-Friendly) ====================
    
    @bp.route("/api/bulk-doc-analysis/test-files", methods=["GET"])
    def api_list_test_files():
        """List available test files for quick upload (automation-friendly)."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401
        
        try:
            # Define project root
            project_root = Path(__file__).parent.parent.parent
            
            # Look for test files in a designated test_files directory
            test_files_dir = project_root / "test_files"
            if not test_files_dir.exists():
                test_files_dir.mkdir(parents=True, exist_ok=True)
            
            # Also include car.pdf from project root if it exists
            root_test_files = []
            for ext in ['.pdf', '.md', '.txt', '.docx', '.csv']:
                for f in project_root.glob(f'*{ext}'):
                    if f.is_file() and not f.name.startswith('.'):
                        root_test_files.append({
                            'filename': f.name,
                            'path': str(f.relative_to(project_root)),
                            'size': f.stat().st_size,
                            'type': ext.upper().replace('.', '')
                        })
            
            # Get files from test_files directory
            test_files = []
            for f in test_files_dir.glob('*'):
                if f.is_file() and not f.name.startswith('.'):
                    ext = f.suffix.upper().replace('.', '')
                    test_files.append({
                        'filename': f.name,
                        'path': str(f.relative_to(project_root)),
                        'size': f.stat().st_size,
                        'type': ext
                    })
            
            # Combine and sort by name
            all_files = root_test_files + test_files
            all_files.sort(key=lambda x: x['filename'])
            
            return jsonify({"test_files": all_files})
        except Exception as e:
            logger.error(f"List test files error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    
    @bp.route("/api/bulk-doc-analysis/test-files/upload", methods=["POST"])
    def api_upload_test_file():
        """Upload a test file from the server to the current session (automation-friendly)."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401
        
        # Define project root
        project_root = Path(__file__).parent.parent.parent
        
        user_id = user_session.get("user_id") or "anonymous"
        bulk_session_id = _ensure_session(user_id)
        
        data = request.get_json() or {}
        file_path = data.get("file_path")
        workflow_version_id = data.get("workflow_version_id")
        
        if not file_path:
            return jsonify({"error": "file_path is required"}), 400
        
        try:
            # Resolve the file path
            full_path = project_root / file_path
            if not full_path.exists():
                return jsonify({"error": f"File not found: {file_path}"}), 404
            
            if not full_path.is_file():
                return jsonify({"error": f"Not a file: {file_path}"}), 400
            
            # Get accepted types and ingestion profile from workflow if provided
            accepted_types = ["PDF", "DOCX", "TXT", "MD", "CSV"]  # Default
            ingestion_profile_id = None
            if workflow_version_id:
                try:
                    from .workflow_service import WorkflowService
                    workflow_service = WorkflowService()
                    workflow_version = workflow_service.get_workflow_version(workflow_version_id)
                    if workflow_version:
                        ingestion_profile_id = workflow_version.ingestion_profile_id
                        from .ingestion_service import IngestionService
                        ingestion_service = IngestionService()
                        ingestion_profile = ingestion_service.get_ingestion_profile(workflow_version.ingestion_profile_id)
                        if ingestion_profile:
                            accepted_types = ingestion_profile.accepted_input_types
                except Exception as e:
                    logger.warning(f"Could not get workflow accepted types: {e}")
            
            # Check file type
            file_ext = full_path.suffix.upper().replace('.', '')
            if file_ext not in accepted_types:
                return jsonify({
                    "error": f"File type {file_ext} not accepted. Allowed: {accepted_types}"
                }), 400
            
            # Create a file-like object for the create_documents API
            from io import BytesIO
            from werkzeug.datastructures import FileStorage
            
            with open(full_path, 'rb') as f:
                file_content = f.read()
            
            # Create FileStorage object (mimics uploaded file)
            file_obj = FileStorage(
                stream=BytesIO(file_content),
                filename=full_path.name,
                content_type='application/octet-stream'
            )
            
            # Create document entry
            svc = get_service()
            docs = svc.create_documents(bulk_session_id, [file_obj], user_id)
            doc = docs[0]
            
            # Queue conversion job (if Redis available)
            try:
                import redis
                redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
                redis_client = redis.from_url(redis_url)
                from rq import Queue
                conversion_queue = Queue("conversion", connection=redis_client)
                
                # Get the file path relative to storage base for the worker
                doc_dir = svc.storage_base / "docs" / doc.doc_id
                file_path = doc_dir / full_path.name
                object_storage_key = str(file_path.relative_to(svc.storage_base))
                
                job_data = {
                    "doc_id": doc.doc_id,
                    "session_id": bulk_session_id,
                    "object_storage_key": object_storage_key,
                    "ingestion_profile_id": ingestion_profile_id,
                    "idempotency_key": f"convert:{doc.doc_id}",
                }
                from external.ai_bulk_doc_analysis.workers.rq_worker import convert_doc_job
                conversion_queue.enqueue(convert_doc_job, job_data, job_id=f"convert_{doc.doc_id}")
                logger.info(f"Enqueued conversion job for test file {doc.doc_id}")
            except Exception as e:
                logger.warning(f"Could not queue conversion job: {e}")
            
            return jsonify({
                "document": doc.to_dict(),
                "message": f"Test file '{full_path.name}' added to session"
            })
            
        except Exception as e:
            logger.error(f"Upload test file error: {e}", exc_info=True)
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
                    "workflow_version_id": workflow.workflow_version_id,
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
                "workflow_version": workflow_version
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
        valid_formats = ['CSV', 'JSON', 'MD', 'DOCX', 'PDF', 'XLSX']
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
                    valid_formats = ['CSV', 'JSON', 'MD', 'DOCX', 'PDF', 'XLSX']
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

    # ==================== Chain APIs ====================
    
    @bp.route("/api/bulk-doc-analysis/chains", methods=["GET"])
    def api_list_chains():
        """List all chains."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401
        
        try:
            from .db_service import BulkDocDBService
            svc = BulkDocDBService()
            chains = svc.list_chains()
            return jsonify({
                "chains": [chain.to_dict() for chain in chains]
            })
        except Exception as e:
            logger.error(f"List chains error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    
    @bp.route("/api/bulk-doc-analysis/chains", methods=["POST"])
    def api_create_chain():
        """Create a new chain."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401
        
        user_id = user_session.get("user_id") or "anonymous"
        data = request.get_json() or {}
        
        name = data.get("name", "").strip()
        description = data.get("description", "").strip()
        steps = data.get("steps", [])
        
        # Validation
        if not name:
            return jsonify({"error": "Name is required"}), 400
        if not steps or len(steps) == 0:
            return jsonify({"error": "At least one step is required"}), 400
        
        try:
            from .db_service import BulkDocDBService
            svc = BulkDocDBService()
            chain = svc.create_chain(user_id, name, description, steps)
            return jsonify({
                "success": True,
                "chain": chain.to_dict()
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Create chain error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    
    @bp.route("/api/bulk-doc-analysis/chains/<chain_id>", methods=["GET"])
    def api_get_chain(chain_id: str):
        """Get chain by ID (returns latest version)."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401
        
        try:
            from .db_service import BulkDocDBService
            svc = BulkDocDBService()
            chain = svc.get_chain_by_id(chain_id)
            
            if not chain:
                return jsonify({"error": "Chain not found"}), 404
            
            return jsonify(chain.to_dict())
        except Exception as e:
            logger.error(f"Get chain error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    
    @bp.route("/api/bulk-doc-analysis/chains/<chain_id>", methods=["PUT"])
    def api_update_chain(chain_id: str):
        """Update chain (creates new version)."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401
        
        data = request.get_json() or {}
        
        name = data.get("name", "").strip()
        description = data.get("description", "").strip()
        steps = data.get("steps", [])
        
        # Validation
        if not name:
            return jsonify({"error": "Name is required"}), 400
        if not steps or len(steps) == 0:
            return jsonify({"error": "At least one step is required"}), 400
        
        try:
            from .db_service import BulkDocDBService
            svc = BulkDocDBService()
            chain = svc.update_chain(chain_id, name, description, steps)
            return jsonify({
                "success": True,
                "chain": chain.to_dict()
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Update chain error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    
    @bp.route("/api/bulk-doc-analysis/chains/<chain_id>", methods=["DELETE"])
    def api_delete_chain(chain_id: str):
        """Delete chain (validates no workflows reference it)."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401
        
        try:
            from .db_service import get_db_session
            from .models import Chain, ChainVersion, WorkflowVersion
            
            with get_db_session() as db:
                # Check if any workflow versions reference this chain
                chain_versions = db.query(ChainVersion).filter(
                    ChainVersion.chain_id == chain_id
                ).all()
                
                chain_version_ids = [cv.chain_version_id for cv in chain_versions]
                
                if chain_version_ids:
                    workflow_refs = db.query(WorkflowVersion).filter(
                        WorkflowVersion.chain_version_id.in_(chain_version_ids)
                    ).count()
                    
                    if workflow_refs > 0:
                        return jsonify({
                            "error": f"Cannot delete chain: {workflow_refs} workflow version(s) reference this chain"
                        }), 400
                
                # Delete chain (cascades to versions and steps)
                chain = db.query(Chain).filter(Chain.chain_id == chain_id).first()
                if not chain:
                    return jsonify({"error": "Chain not found"}), 404
                
                db.delete(chain)
                db.commit()
            
            return jsonify({"success": True})
        except Exception as e:
            logger.error(f"Delete chain error: {e}", exc_info=True)
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
        workflow_version_id = data.get("workflow_version_id")
        
        # Scalable solution: Accept either chain_version_id OR workflow_version_id
        # If only workflow_version_id is provided, resolve chain_version_id from it
        if not chain_version_id and workflow_version_id:
            try:
                from .workflow_service import WorkflowService
                workflow_service = WorkflowService()
                workflow_version = workflow_service.get_workflow_version(workflow_version_id)
                if not workflow_version:
                    return jsonify({"error": f"Workflow version {workflow_version_id} not found"}), 404
                chain_version_id = workflow_version.chain_version_id
            except Exception as e:
                logger.error(f"Error resolving chain_version_id from workflow_version_id: {e}", exc_info=True)
                return jsonify({"error": f"Failed to resolve chain_version_id: {str(e)}"}), 400
        
        if not chain_version_id:
            return jsonify({"error": "Either chain_version_id or workflow_version_id is required"}), 400

        try:
            from .db_service import BulkDocDBService, get_db_session
            from .models import Run as DBRun
            from sqlalchemy import desc
            
            svc = BulkDocDBService()
            
            # Check if there's an existing PAUSED run for this session/workflow that we can resume
            existing_run = None
            with get_db_session() as db:
                db_run = db.query(DBRun).filter(
                    DBRun.session_id == bulk_session_id,
                    DBRun.workflow_version_id == workflow_version_id,
                    DBRun.status == 'PAUSED'
                ).order_by(desc(DBRun.created_at)).first()
                
                if db_run:
                    existing_run = svc.get_run(db_run.run_id)
            
            # If we have a paused run, resume it instead of creating new
            if existing_run:
                # Call the resume endpoint (defined below)
                # Import here to avoid forward reference issues
                from .blueprint import api_resume_run
                return api_resume_run(existing_run.run_id)
                # Fall through to create resume logic inline here
                # OR redirect to resume endpoint
                # For now, we'll handle it inline to avoid circular imports
                from .models import StepResult, ExecutionTask
                
                with get_db_session() as db:
                    db_run = db.query(DBRun).filter(DBRun.run_id == run_id_to_resume).first()
                    if db_run:
                        chain = svc.get_chain(db_run.chain_version_id)
                        if chain:
                            # Check if CSV workflow
                            is_csv_workflow = False
                            if db_run.workflow_version_id:
                                from .workflow_service import WorkflowService
                                from .ingestion_service import IngestionService
                                workflow_service = WorkflowService()
                                workflow_version = workflow_service.get_workflow_version(db_run.workflow_version_id)
                                if workflow_version:
                                    ingestion_service = IngestionService()
                                    ingestion_profile = ingestion_service.get_ingestion_profile(workflow_version.ingestion_profile_id)
                                    if ingestion_profile and 'CSV' in ingestion_profile.accepted_input_types:
                                        is_csv_workflow = True
                            
                            db_run.status = 'RUNNING'
                            db.commit()
                            
                            # Enqueue only incomplete steps
                            if USE_QUEUES:
                                try:
                                    execution_queue = get_execution_queue()
                                    
                                    if is_csv_workflow:
                                        tasks = db.query(ExecutionTask).filter(ExecutionTask.run_id == run_id_to_resume).all()
                                        for task in tasks:
                                            for step_idx in range(1, chain.step_count + 1):
                                                step_result = db.query(StepResult).filter(
                                                    StepResult.run_id == run_id_to_resume,
                                                    StepResult.doc_id == task.doc_id,
                                                    StepResult.task_id == task.task_id,
                                                    StepResult.step_index == step_idx
                                                ).first()
                                                if step_result and step_result.status == 'SUCCESS':
                                                    continue
                                                step_def = next((s for s in chain.steps if s.get("index") == step_idx), None)
                                                if not step_def:
                                                    continue
                                                model_config = step_def.get("model_config", {
                                                    "model": "claude-3-haiku-20240307",
                                                    "max_tokens": 4096,
                                                    "temperature": 0.2
                                                })
                                                job_data = {
                                                    "run_id": run_id_to_resume,
                                                    "doc_id": task.doc_id,
                                                    "session_id": db_run.session_id,
                                                    "task_id": task.task_id,
                                                    "step_index": step_idx,
                                                    "chain_version_id": db_run.chain_version_id,
                                                    "required_inputs": step_def.get("required_inputs", []),
                                                    "prompt": step_def.get("prompt", ""),
                                                    "model_config": model_config,
                                                    "idempotency_key": f"step:{run_id_to_resume}:{task.doc_id}:{task.task_id}:{step_idx}",
                                                }
                                                execution_queue.enqueue(
                                                    execute_step_job,
                                                    job_data,
                                                    job_id=f"execute_{run_id_to_resume}_{task.doc_id}_{task.task_id}_step{step_idx}"
                                                )
                                    else:
                                        step_results = db.query(StepResult).filter(StepResult.run_id == run_id_to_resume).all()
                                        doc_ids = list(set(sr.doc_id for sr in step_results))
                                        for doc_id in doc_ids:
                                            for step_idx in range(1, chain.step_count + 1):
                                                step_result = db.query(StepResult).filter(
                                                    StepResult.run_id == run_id_to_resume,
                                                    StepResult.doc_id == doc_id,
                                                    StepResult.step_index == step_idx
                                                ).first()
                                                if step_result and step_result.status == 'SUCCESS':
                                                    continue
                                                step_def = next((s for s in chain.steps if s.get("index") == step_idx), None)
                                                if not step_def:
                                                    continue
                                                model_config = step_def.get("model_config", {
                                                    "model": "claude-3-haiku-20240307",
                                                    "max_tokens": 4096,
                                                    "temperature": 0.2
                                                })
                                                job_data = {
                                                    "run_id": run_id_to_resume,
                                                    "doc_id": doc_id,
                                                    "session_id": db_run.session_id,
                                                    "step_index": step_idx,
                                                    "chain_version_id": db_run.chain_version_id,
                                                    "required_inputs": step_def.get("required_inputs", []),
                                                    "prompt": step_def.get("prompt", ""),
                                                    "model_config": model_config,
                                                    "idempotency_key": f"step:{run_id_to_resume}:{doc_id}:{step_idx}",
                                                }
                                                execution_queue.enqueue(
                                                    execute_step_job,
                                                    job_data,
                                                    job_id=f"exec_{run_id_to_resume}_{doc_id}_step{step_idx}"
                                                )
                                except Exception as e:
                                    logger.error(f"Failed to enqueue resume jobs: {e}", exc_info=True)
                
                run = svc.get_run(run_id_to_resume)
                return jsonify({
                    "success": True,
                    "run": run.to_dict() if run else None,
                })
            
            # Otherwise create a new run
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
                                        "session_id": run.session_id,  # Needed to find R0.md
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
                                    "session_id": run.session_id,  # Needed to find R0.md
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

    @bp.route("/api/bulk-doc-analysis/runs", methods=["GET"])
    def api_list_runs():
        """List all runs for the current user."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401

        try:
            from .db_service import get_db_session
            from .models import (
                Run as DBRun, Session as DBSession,
                WorkflowVersion, Workflow, ExecutionTask, StepResult
            )
            from sqlalchemy import func, distinct

            user_id = user_session.get("user_id") or "anonymous"
            
            with get_db_session() as db:
                # Get all sessions for the user
                sessions = db.query(DBSession).filter(
                    DBSession.user_id == user_id
                ).all()
                session_ids = [s.session_id for s in sessions]
                
                if not session_ids:
                    return jsonify({"runs": []})
                
                # Get runs for those sessions, ordered by created_at DESC
                runs = db.query(DBRun).filter(
                    DBRun.session_id.in_(session_ids)
                ).order_by(DBRun.created_at.desc()).limit(50).all()
                
                results = []
                for run in runs:
                    # Get workflow name
                    workflow_name = "Unknown Workflow"
                    if run.workflow_version_id:
                        workflow_version = db.query(WorkflowVersion).filter(
                            WorkflowVersion.workflow_version_id == run.workflow_version_id
                        ).first()
                        if workflow_version:
                            workflow = db.query(Workflow).filter(
                                Workflow.workflow_id == workflow_version.workflow_id
                            ).first()
                            if workflow:
                                workflow_name = workflow.name
                    
                    # Count tasks to determine if CSV workflow
                    task_count = db.query(ExecutionTask).filter(
                        ExecutionTask.run_id == run.run_id
                    ).count()
                    is_csv_workflow = task_count > 0
                    
                    # Count documents for non-CSV workflows
                    document_count = 1  # Default
                    if not is_csv_workflow:
                        # Count unique doc_ids from step_results
                        unique_docs = db.query(distinct(StepResult.doc_id)).filter(
                            StepResult.run_id == run.run_id
                        ).count()
                        document_count = unique_docs if unique_docs > 0 else 1
                    else:
                        document_count = task_count
                    
                    results.append({
                        "run_id": run.run_id,
                        "workflow_name": workflow_name,
                        "workflow_version_id": run.workflow_version_id,
                        "status": run.status,
                        "created_at": run.created_at.isoformat() if run.created_at else None,
                        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                        "total_input_tokens": run.total_input_tokens or 0,
                        "total_output_tokens": run.total_output_tokens or 0,
                        "document_count": document_count,
                        "task_count": task_count if is_csv_workflow else 0,
                        "is_csv_workflow": is_csv_workflow,
                    })
                
                return jsonify({"runs": results})
        except Exception as e:
            logger.error(f"List runs error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/runs/<run_id>/pause", methods=["POST"])
    def api_pause_run(run_id: str):
        """Pause a running run."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401
        
        try:
            from .db_service import get_db_session
            from .models import Run as DBRun
            
            with get_db_session() as db:
                db_run = db.query(DBRun).filter(DBRun.run_id == run_id).first()
                if not db_run:
                    return jsonify({"error": "Run not found"}), 404
                
                # Only allow pausing QUEUED or RUNNING runs
                if db_run.status not in ['QUEUED', 'RUNNING']:
                    return jsonify({"error": f"Cannot pause run with status: {db_run.status}"}), 400
                
                db_run.status = 'PAUSED'
                db.commit()
        
            return jsonify({"success": True, "run_id": run_id, "status": "PAUSED"})
        except Exception as e:
            logger.error(f"Pause run error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/runs/<run_id>/resume", methods=["POST"])
    def api_resume_run(run_id: str):
        """Resume a paused run - only enqueues incomplete steps."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401
        
        try:
            from .db_service import BulkDocDBService, get_db_session
            from .models import Run as DBRun, StepResult, ExecutionTask
            
            svc = BulkDocDBService()
            
            with get_db_session() as db:
                db_run = db.query(DBRun).filter(DBRun.run_id == run_id).first()
                if not db_run:
                    return jsonify({"error": "Run not found"}), 404
                
                # Only allow resuming PAUSED runs
                if db_run.status != 'PAUSED':
                    return jsonify({"error": f"Cannot resume run with status: {db_run.status}"}), 400
                
                # Get chain to determine steps
                chain = svc.get_chain(db_run.chain_version_id)
                if not chain:
                    return jsonify({"error": "Chain not found"}), 404
                
                # Check if CSV workflow
                is_csv_workflow = False
                if db_run.workflow_version_id:
                    from .workflow_service import WorkflowService
                    from .ingestion_service import IngestionService
                    workflow_service = WorkflowService()
                    workflow_version = workflow_service.get_workflow_version(db_run.workflow_version_id)
                    if workflow_version:
                        ingestion_service = IngestionService()
                        ingestion_profile = ingestion_service.get_ingestion_profile(workflow_version.ingestion_profile_id)
                        if ingestion_profile and 'CSV' in ingestion_profile.accepted_input_types:
                            is_csv_workflow = True
                
                # Update run status to RUNNING
                db_run.status = 'RUNNING'
                db.commit()
                
                # Enqueue only incomplete steps
                if USE_QUEUES:
                    try:
                        execution_queue = get_execution_queue()
                        
                        if is_csv_workflow:
                            # CSV: Resume incomplete task/step combinations
                            tasks = db.query(ExecutionTask).filter(ExecutionTask.run_id == run_id).all()
                            
                            for task in tasks:
                                for step_idx in range(1, chain.step_count + 1):
                                    # Check if this step is already complete
                                    step_result = db.query(StepResult).filter(
                                        StepResult.run_id == run_id,
                                        StepResult.doc_id == task.doc_id,
                                        StepResult.task_id == task.task_id,
                                        StepResult.step_index == step_idx
                                    ).first()
                                    
                                    # Skip if already complete
                                    if step_result and step_result.status == 'SUCCESS':
                                        continue
                                    
                                    step_def = next((s for s in chain.steps if s.get("index") == step_idx), None)
                                    if not step_def:
                                        continue
                                    
                                    model_config = step_def.get("model_config", {
                                        "model": "claude-3-haiku-20240307",
                                        "max_tokens": 4096,
                                        "temperature": 0.2
                                    })
                                    
                                    job_data = {
                                        "run_id": run_id,
                                        "doc_id": task.doc_id,
                                        "session_id": db_run.session_id,
                                        "task_id": task.task_id,
                                        "step_index": step_idx,
                                        "chain_version_id": db_run.chain_version_id,
                                        "required_inputs": step_def.get("required_inputs", []),
                                        "prompt": step_def.get("prompt", ""),
                                        "model_config": model_config,
                                        "idempotency_key": f"step:{run_id}:{task.doc_id}:{task.task_id}:{step_idx}",
                                    }
                                    execution_queue.enqueue(
                                        execute_step_job,
                                        job_data,
                                        job_id=f"execute_{run_id}_{task.doc_id}_{task.task_id}_step{step_idx}"
                                    )
                                    logger.info(f"Resumed execution job for {run_id}/{task.doc_id}/{task.task_id}/step{step_idx}")
                        else:
                            # Non-CSV: Resume incomplete doc/step combinations
                            step_results = db.query(StepResult).filter(StepResult.run_id == run_id).all()
                            doc_ids = list(set(sr.doc_id for sr in step_results))
                            
                            for doc_id in doc_ids:
                                for step_idx in range(1, chain.step_count + 1):
                                    # Check if this step is already complete
                                    step_result = db.query(StepResult).filter(
                                        StepResult.run_id == run_id,
                                        StepResult.doc_id == doc_id,
                                        StepResult.step_index == step_idx
                                    ).first()
                                    
                                    # Skip if already complete
                                    if step_result and step_result.status == 'SUCCESS':
                                        continue
                                    
                                    step_def = next((s for s in chain.steps if s.get("index") == step_idx), None)
                                    if not step_def:
                                        continue
                                    
                                    model_config = step_def.get("model_config", {
                                        "model": "claude-3-haiku-20240307",
                                        "max_tokens": 4096,
                                        "temperature": 0.2
                                    })
                                    
                                    job_data = {
                                        "run_id": run_id,
                                        "doc_id": doc_id,
                                        "session_id": db_run.session_id,
                                        "step_index": step_idx,
                                        "chain_version_id": db_run.chain_version_id,
                                        "required_inputs": step_def.get("required_inputs", []),
                                        "prompt": step_def.get("prompt", ""),
                                        "model_config": model_config,
                                        "idempotency_key": f"step:{run_id}:{doc_id}:{step_idx}",
                                    }
                                    execution_queue.enqueue(
                                        execute_step_job,
                                        job_data,
                                        job_id=f"exec_{run_id}_{doc_id}_step{step_idx}"
                                    )
                                    logger.info(f"Resumed execution job for {run_id}/{doc_id}/step{step_idx}")
                    except Exception as e:
                        logger.error(f"Failed to enqueue resume jobs: {e}", exc_info=True)
                        return jsonify({"error": f"Failed to resume: {str(e)}"}), 500
            
            run = svc.get_run(run_id)
            return jsonify({
                "success": True,
                "run": run.to_dict() if run else None,
            })
        except Exception as e:
            logger.error(f"Resume run error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    @bp.route("/api/bulk-doc-analysis/runs/<run_id>", methods=["DELETE"])
    def api_delete_run(run_id):
        """Delete a run and all associated data."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401

        try:
            svc = get_service()
            svc.delete_run(run_id)
            return jsonify({"message": "Run deleted successfully"})
        except ValueError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            logger.error(f"Delete run error: {e}", exc_info=True)
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
        """Download final output (R(N)) for a document or task."""
        user_session = _current_user_session()
        if not user_session:
            return jsonify({"error": "Unauthorized"}), 401

        try:
            from .db_service import get_db_session
            from .models import Run as DBRun, WorkflowVersion, ExportProfile, ExecutionTask
            
            # Check if this is a CSV workflow (task_id in query params)
            task_id = request.args.get('task_id')
            
            svc = get_service()
            run = svc.get_run(run_id)
            if not run:
                return jsonify({"error": "Run not found"}), 404

            doc = svc.get_document(doc_id)
            if not doc:
                return jsonify({"error": "Document not found"}), 404

            chain = svc.get_chain(run.chain_version_id)
            if not chain:
                return jsonify({"error": "Chain not found"}), 404

            # Get export profile format from workflow first (needed for get_final_output_path)
            export_format = "MD"  # Default
            with get_db_session() as db:
                db_run = db.query(DBRun).filter(DBRun.run_id == run_id).first()
                if db_run and db_run.workflow_version_id:
                    workflow_version = db.query(WorkflowVersion).filter(
                        WorkflowVersion.workflow_version_id == db_run.workflow_version_id
                    ).first()
                    if workflow_version:
                        export_profile = db.query(ExportProfile).filter(
                            ExportProfile.export_profile_id == workflow_version.export_profile_id
                        ).first()
                        if export_profile:
                            export_format = export_profile.format
            
            # Get final step output (R(N)) - this will check for converted files if export_format requires it
            output_path = svc.get_final_output_path(run_id, doc_id, chain, export_format=export_format, task_id=task_id)
            if not output_path or not output_path.exists():
                # Fallback to converted markdown if no step output exists
                if doc.converted_md_path:
                    md_path = Path(doc.converted_md_path)
                    # Handle relative paths - resolve from project root
                    if not md_path.is_absolute():
                        project_root = Path(__file__).parent.parent.parent
                        md_path = project_root / md_path
                    if md_path.exists():
                        output_path = md_path
                    else:
                        return jsonify({"error": f"Output file not found: {md_path}"}), 404
                else:
                    return jsonify({"error": "Output not found"}), 404

            # Determine file extension and mimetype based on actual file (may be converted or markdown)
            file_ext = output_path.suffix
            format_map = {
                ".json": ("application/json", "JSON"),
                ".md": ("text/markdown", "MD"),
                ".csv": ("text/csv", "CSV"),
                ".xlsx": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "XLSX"),
                ".docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "DOCX"),
                ".pdf": ("application/pdf", "PDF"),
            }
            mimetype, detected_format = format_map.get(file_ext, ("text/markdown", "MD"))

            # Determine filename - use export format extension if file is converted, otherwise use detected
            base_name = Path(doc.original_filename).stem
            if task_id:
                # For CSV workflows, include row number in filename
                with get_db_session() as db:
                    from .models import ExecutionTask
                    task = db.query(ExecutionTask).filter(ExecutionTask.task_id == task_id).first()
                    if task:
                        download_name = f"{base_name}_row{task.row_index + 1}_output{file_ext}"
                    else:
                        download_name = f"{base_name}_output{file_ext}"
            else:
                download_name = f"{base_name}_output{file_ext}"
            
            logger.info(f"Download: run_id={run_id}, doc_id={doc_id}, task_id={task_id}, export_format={export_format}, file={output_path.name}, download_name={download_name}, mimetype={mimetype}")
            
            # Use send_file with download_name (Flask 2.0+)
            # The download_name parameter sets Content-Disposition header correctly
            return send_file(
                str(output_path),
                mimetype=mimetype,
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
            from .db_service import get_db_session
            from .models import Run as DBRun, WorkflowVersion, ExportProfile
            
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
                run = svc.get_run(run_id)
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

            # Get export profile format from workflow and check if CSV
            export_format = "MD"  # Default
            is_csv_workflow = False
            with get_db_session() as db:
                db_run = db.query(DBRun).filter(DBRun.run_id == run_id).first()
                if db_run and db_run.workflow_version_id:
                    workflow_version = db.query(WorkflowVersion).filter(
                        WorkflowVersion.workflow_version_id == db_run.workflow_version_id
                    ).first()
                    if workflow_version:
                        export_profile = db.query(ExportProfile).filter(
                            ExportProfile.export_profile_id == workflow_version.export_profile_id
                        ).first()
                        if export_profile:
                            export_format = export_profile.format
                        
                        # Check if CSV workflow
                        from .models import IngestionProfile, ExecutionTask
                        ingestion_profile = db.query(IngestionProfile).filter(
                            IngestionProfile.ingestion_profile_id == workflow_version.ingestion_profile_id
                        ).first()
                        if ingestion_profile and 'CSV' in ingestion_profile.accepted_input_types:
                            is_csv_workflow = True
                            
                            logger.info(f"Bulk download: run_id={run_id}, export_format={export_format}, CSV workflow detected")

            # Create ZIP in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                if is_csv_workflow:
                    # CSV workflow: iterate over tasks
                    with get_db_session() as db:
                        from .models import ExecutionTask
                        tasks = db.query(ExecutionTask).filter(ExecutionTask.run_id == run_id).order_by(ExecutionTask.row_index).all()
                        for task in tasks:
                            doc = svc.get_document(task.doc_id)
                            if not doc:
                                continue
                            
                            # Get final output path for this task
                            output_path = svc.get_final_output_path(run_id, task.doc_id, chain, export_format=export_format, task_id=task.task_id)
                            
                            if output_path and output_path.exists():
                                file_ext = output_path.suffix
                                base_name = Path(doc.original_filename).stem
                                zip_filename = f"{base_name}_row{task.row_index + 1}_output{file_ext}"
                                zip_file.write(str(output_path), zip_filename)
                else:
                    # Non-CSV workflow: iterate over documents
                    if not doc_ids:
                        return jsonify({"error": "No documents found for this run"}), 404
                    
                    for doc_id in doc_ids:
                        doc = svc.get_document(doc_id)
                        if not doc:
                            continue
                        
                        # Get final output path (checks for converted files if export_format requires it)
                        output_path = svc.get_final_output_path(run_id, doc_id, chain, export_format=export_format)
                        
                        if output_path and output_path.exists():
                            # Use actual file extension from the file (may be converted or markdown)
                            file_ext = output_path.suffix
                            base_name = Path(doc.original_filename).stem
                            zip_filename = f"{base_name}_output{file_ext}"
                            zip_file.write(str(output_path), zip_filename)

            zip_buffer.seek(0)

            # Return ZIP file
            # Note: send_file with BytesIO requires attachment_filename in older Flask versions
            # Using Response for better compatibility
            from flask import Response
            response = Response(
                zip_buffer.getvalue(),
                mimetype="application/zip",
                headers={
                    "Content-Disposition": f'attachment; filename="run_{run_id}_outputs.zip"'
                }
            )
            return response

        except Exception as e:
            logger.error(f"Bulk download error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    return bp


