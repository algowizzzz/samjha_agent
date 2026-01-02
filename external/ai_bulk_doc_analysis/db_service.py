"""
Database service layer for AI Bulk Doc Analysis.
Replaces in-memory storage with PostgreSQL via SQLAlchemy.
"""
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# CRITICAL: Load DATABASE_URL BEFORE importing models
# This ensures _is_postgresql is correctly set in models.py
database_url = os.getenv("DATABASE_URL")
if not database_url:
    try:
        from dotenv import load_dotenv
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / ".env.local"
        if env_file.exists():
            load_dotenv(env_file)
            database_url = os.getenv("DATABASE_URL")
    except (ImportError, Exception):
        pass

# Set DATABASE_URL in environment so models.py can detect PostgreSQL
if database_url:
    os.environ["DATABASE_URL"] = database_url

from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker, Session as SQLASession
from sqlalchemy.exc import IntegrityError

# Import all models to ensure they're registered before FK resolution
from .models import (
    Base,
    Session as DBSession,
    Document as DBDocument,
    Chain as DBChain,
    ChainStep as DBChainStep,
    ChainVersion as DBChainVersion,
    Run as DBRun,
    StepResult as DBStepResult,
    Workflow,
    WorkflowVersion,
    WorkflowDomain,
    IngestionProfile,
    ExportProfile,
    ExecutionTask,
    JobQueueLog,
)
# Force mapper configuration after all imports
try:
    from sqlalchemy.orm import configure_mappers
    configure_mappers()
except Exception:
    pass  # Ignore mapper config errors - they might resolve at runtime
from .services import Document, Chain, StepResult, Run

logger = logging.getLogger(__name__)

# Global engine and session factory
_engine = None
_SessionLocal = None


def init_db(database_url: Optional[str] = None):
    """Initialize database connection and create tables if needed."""
    global _engine, _SessionLocal

    if _engine is not None:
        return

    database_url = database_url or os.getenv("DATABASE_URL")
    if not database_url:
        # Try loading from .env.local if available
        try:
            from dotenv import load_dotenv
            load_dotenv(".env.local")
            database_url = os.getenv("DATABASE_URL")
        except ImportError:
            pass
    
    if not database_url:
        # Default to SQLite for local development
        project_root = Path(__file__).parent.parent.parent
        db_path = project_root / "data" / "ai_bulk_doc_analysis" / "db" / "bulk_doc.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        database_url = f"sqlite:///{db_path}"
        logger.info(f"No DATABASE_URL set, using SQLite: {database_url}")

    # Create engine with connection pooling
    # For PostgreSQL, ensure we're using the public schema
    if database_url.startswith('postgresql'):
        # Ensure schema is specified
        if '?' not in database_url:
            database_url = f"{database_url}?options=-csearch_path%3Dpublic"
        else:
            database_url = f"{database_url}&options=-csearch_path%3Dpublic"
    
    _engine = create_engine(
        database_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,  # Verify connections before using
        echo=False,  # Set to True for SQL debugging
        connect_args={"options": "-c search_path=public"} if database_url.startswith('postgresql') else {}
    )
    
    # For PostgreSQL, set search_path on every connection via event listener
    if database_url.startswith('postgresql'):
        from sqlalchemy import event
        
        @event.listens_for(_engine, "connect")
        def set_search_path(dbapi_conn, connection_record):
            """Set search_path to public on every new connection."""
            try:
                with dbapi_conn.cursor() as cur:
                    cur.execute("SET search_path TO public")
                    dbapi_conn.commit()
            except Exception as e:
                logger.warning(f"Failed to set search_path: {e}")
        
        # Also set search_path on checkout from pool (for reused connections)
        @event.listens_for(_engine, "checkout")
        def set_search_path_checkout(dbapi_conn, connection_record, connection_proxy):
            """Set search_path on connection checkout from pool."""
            try:
                with dbapi_conn.cursor() as cur:
                    cur.execute("SET search_path TO public")
                    dbapi_conn.commit()
            except Exception:
                pass  # Ignore errors

    # Create session factory
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

    # Create tables - ensure they exist
    # Use checkfirst=True to avoid errors if tables already exist
    try:
        Base.metadata.create_all(bind=_engine, checkfirst=True)
        logger.info("Database tables created/verified successfully")
    except Exception as e:
        logger.error(f"Error creating tables: {e}", exc_info=True)
        # Don't fail completely - tables might already exist
    
    # Fix step_results unique constraint for CSV workflows (PostgreSQL)
    if database_url and database_url.startswith('postgresql'):
        try:
            with _engine.connect() as conn:
                from sqlalchemy import text
                # Drop old constraint if exists
                conn.execute(text("""
                    DO $$ 
                    BEGIN
                        IF EXISTS (
                            SELECT 1 FROM pg_constraint 
                            WHERE conname = 'step_results_run_id_doc_id_step_index_key'
                        ) THEN
                            ALTER TABLE step_results DROP CONSTRAINT step_results_run_id_doc_id_step_index_key;
                        END IF;
                        IF EXISTS (
                            SELECT 1 FROM pg_constraint 
                            WHERE conname = 'uq_step_result'
                        ) THEN
                            ALTER TABLE step_results DROP CONSTRAINT uq_step_result;
                        END IF;
                    END $$;
                """))
                conn.commit()
                
                # Create new unique index with COALESCE for NULL handling
                conn.execute(text("""
                    DROP INDEX IF EXISTS uq_step_result_with_task;
                    CREATE UNIQUE INDEX IF NOT EXISTS uq_step_result_with_task 
                    ON step_results (run_id, doc_id, step_index, COALESCE(task_id, ''));
                """))
                conn.commit()
                logger.info("Updated step_results constraint for CSV workflows")
        except Exception as e:
            logger.warning(f"Could not update step_results constraint: {e}")
    
    logger.info("Database initialized and tables created if needed")


def get_db_session() -> SQLASession:
    """Get a database session (use in context manager)."""
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()


class BulkDocDBService:
    """Database-backed service for Bulk Doc Analysis."""

    def __init__(self, storage_base: Optional[Path] = None):
        self.storage_base = storage_base or Path("data/ai_bulk_doc_analysis")
        self.storage_base.mkdir(parents=True, exist_ok=True)
        # Initialize DB if not already done
        init_db()

    # -------- Session Management --------
    def ensure_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        """
        Get or create a session for the user.
        If session_id provided, returns that session (if exists).
        Otherwise creates a new session.
        """
        from sqlalchemy import text
        db = get_db_session()
        try:
            # Explicitly set search_path for this connection (in case event listener didn't fire)
            if os.getenv("DATABASE_URL", "").startswith('postgresql'):
                db.execute(text("SET search_path TO public"))
                db.commit()
            
            if session_id:
                # Return existing session if provided
                db_session = db.query(DBSession).filter(
                    DBSession.session_id == session_id,
                    DBSession.user_id == user_id
                ).first()
                if db_session:
                    return session_id
                # Session doesn't exist or doesn't belong to user
                raise ValueError(f"Session {session_id} not found or access denied")
            
            # Create new session
            session_id = f"session_{user_id}_{uuid.uuid4().hex[:12]}"
            db_session = DBSession(session_id=session_id, user_id=user_id)
            db.add(db_session)
            db.commit()
            return session_id
        finally:
            db.close()
    
    def create_session(self, user_id: str, name: Optional[str] = None) -> str:
        """Create a new session for the user."""
        return self.ensure_session(user_id, session_id=None)
    
    def list_sessions(self, user_id: str) -> List[Dict]:
        """List all sessions for a user, sorted by newest first."""
        with get_db_session() as db:
            db_sessions = (
                db.query(DBSession)
                .filter(DBSession.user_id == user_id)
                .order_by(DBSession.created_at.desc())
                .all()
            )
            sessions = []
            for s in db_sessions:
                # Count documents in this session
                doc_count = db.query(DBDocument).filter(DBDocument.session_id == s.session_id).count()
                sessions.append({
                    "session_id": s.session_id,
                    "user_id": s.user_id,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                    "updated_at": s.updated_at.isoformat() if s.updated_at else None,
                    "document_count": doc_count,
                    "name": s.metadata_json.get("name") if isinstance(s.metadata_json, dict) else None,
                })
            return sessions

    # -------- Document Management --------
    def create_documents(self, session_id: str, files: List, user_id: str) -> List[Document]:
        """Create document records from uploaded files."""
        docs = []
        now = datetime.utcnow()

        with get_db_session() as db:
            for file in files:
                doc_id = str(uuid.uuid4())
                filename = file.filename or "unknown.pdf"
                size = len(file.read())
                file.seek(0)

                # Store file
                doc_dir = self.storage_base / "docs" / doc_id
                doc_dir.mkdir(parents=True, exist_ok=True)
                file_path = doc_dir / filename
                if hasattr(file, 'save'):
                    file.save(str(file_path))
                else:
                    file.seek(0)
                    with open(file_path, "wb") as f:
                        f.write(file.read())

                # Create DB record
                db_doc = DBDocument(
                    doc_id=doc_id,
                    session_id=session_id,
                    original_filename=filename,
                    file_type="PDF",
                    size_bytes=size,
                    status="QUEUED",
                    object_storage_key=str(file_path.relative_to(self.storage_base)),
                )
                db.add(db_doc)
                db.commit()

                # Convert to dataclass
                doc = self._db_doc_to_dataclass(db_doc)
                docs.append(doc)

        return docs

    def update_document_status(
        self,
        doc_id: str,
        status: str,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        converted_md_path: Optional[str] = None,
    ):
        """Update document status."""
        with get_db_session() as db:
            db_doc = db.query(DBDocument).filter(DBDocument.doc_id == doc_id).first()
            if not db_doc:
                raise ValueError(f"Document {doc_id} not found")

            db_doc.status = status
            if error_code:
                db_doc.error_code = error_code
            if error_message:
                db_doc.error_message = error_message
            if converted_md_path:
                db_doc.converted_md_path = converted_md_path
            db_doc.updated_at = datetime.utcnow()
            db.commit()

    def list_documents(self, session_id: str) -> List[Document]:
        """List all documents in a session, sorted by newest first."""
        with get_db_session() as db:
            db_docs = (
                db.query(DBDocument)
                .filter(DBDocument.session_id == session_id)
                .order_by(DBDocument.created_at.desc())
                .all()
            )
            return [self._db_doc_to_dataclass(d) for d in db_docs]

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        with get_db_session() as db:
            db_doc = db.query(DBDocument).filter(DBDocument.doc_id == doc_id).first()
            if not db_doc:
                return None
            return self._db_doc_to_dataclass(db_doc)

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document (allowed for ERROR and QUEUED status, or if explicitly requested)."""
        with get_db_session() as db:
            db_doc = db.query(DBDocument).filter(DBDocument.doc_id == doc_id).first()
            if not db_doc:
                return False
            # Allow deletion of ERROR, QUEUED, or CONVERTED documents
            # CONVERTED can be deleted if user wants to re-upload
            allowed_statuses = ["ERROR", "QUEUED", "CONVERTED"]
            if db_doc.status not in allowed_statuses:
                raise ValueError(f"Can only delete documents with status: {', '.join(allowed_statuses)}")
            db.delete(db_doc)
            db.commit()
            return True

    # -------- Chain Management --------
    def list_chains(self) -> List[Chain]:
        """List all available chains."""
        with get_db_session() as db:
            # Get latest version of each chain
            chains = []
            for db_chain in db.query(DBChain).all():
                # Get latest version
                latest_version = (
                    db.query(DBChainVersion)
                    .filter(DBChainVersion.chain_id == db_chain.chain_id)
                    .order_by(DBChainVersion.version_number.desc())
                    .first()
                )
                if latest_version:
                    chain = self._build_chain_from_version(latest_version, db)
                    chains.append(chain)
            return chains

    def get_chain(self, chain_version_id: str) -> Optional[Chain]:
        """Get a chain by version ID."""
        with get_db_session() as db:
            db_version = db.query(DBChainVersion).filter(
                DBChainVersion.chain_version_id == chain_version_id
            ).first()
            if not db_version:
                return None
            return self._build_chain_from_version(db_version, db)

    def get_chain_by_id(self, chain_id: str) -> Optional[Chain]:
        """Get latest version of a chain by chain_id."""
        with get_db_session() as db:
            latest_version = (
                db.query(DBChainVersion)
                .filter(DBChainVersion.chain_id == chain_id)
                .order_by(DBChainVersion.version_number.desc())
                .first()
            )
            if not latest_version:
                return None
            return self._build_chain_from_version(latest_version, db)

    def create_chain(self, user_id: str, name: str, description: str, steps: List[Dict]) -> Chain:
        """Create a new chain."""
        from .services import BulkDocService
        # Use in-memory service for validation (temporary)
        temp_service = BulkDocService()
        is_valid, error_msg = temp_service._validate_chain_structure(steps)
        if not is_valid:
            raise ValueError(f"Invalid chain structure: {error_msg}")

        chain_id = f"chain_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow()

        with get_db_session() as db:
            # Create chain
            db_chain = DBChain(
                chain_id=chain_id,
                name=name,
                description=description,
                user_id=user_id,
            )
            db.add(db_chain)
            db.flush()

            # Create chain steps
            for step_data in steps:
                model_config = step_data.get("model_config", {
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 4096,
                    "temperature": 0.2
                })
                db_step = DBChainStep(
                    chain_id=chain_id,
                    step_index=step_data["index"],
                    title=step_data.get("title", ""),
                    prompt=step_data["prompt"],
                    description=step_data.get("description", ""),
                    required_inputs=step_data["required_inputs"],
                    model_config=model_config,
                )
                db.add(db_step)

            # Create chain version
            snapshot = {
                "chain_id": chain_id,
                "name": name,
                "description": description,
                "steps": steps,
            }
            version_number = 1
            chain_version_id = f"cv_{chain_id}-v{version_number}"

            db_version = DBChainVersion(
                chain_version_id=chain_version_id,
                chain_id=chain_id,
                version_number=version_number,
                step_count=len(steps),
                valid=is_valid,
                snapshot=snapshot,
            )
            db.add(db_version)
            db.commit()

            # Build and return Chain dataclass
            return self._build_chain_from_version(db_version, db)

    def update_chain(self, chain_id: str, name: str, description: str, steps: List[Dict]) -> Chain:
        """Update an existing chain (creates new version)."""
        from .services import BulkDocService
        temp_service = BulkDocService()
        is_valid, error_msg = temp_service._validate_chain_structure(steps)
        if not is_valid:
            raise ValueError(f"Invalid chain structure: {error_msg}")

        with get_db_session() as db:
            # Get existing chain
            db_chain = db.query(DBChain).filter(DBChain.chain_id == chain_id).first()
            if not db_chain:
                raise ValueError(f"Chain {chain_id} not found")

            # Update chain metadata
            db_chain.name = name
            db_chain.description = description
            db_chain.updated_at = datetime.utcnow()

            # Delete old steps
            db.query(DBChainStep).filter(DBChainStep.chain_id == chain_id).delete()

            # Create new steps
            for step_data in steps:
                model_config = step_data.get("model_config", {
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 4096,
                    "temperature": 0.2
                })
                db_step = DBChainStep(
                    chain_id=chain_id,
                    step_index=step_data["index"],
                    title=step_data.get("title", ""),
                    prompt=step_data["prompt"],
                    description=step_data.get("description", ""),
                    required_inputs=step_data["required_inputs"],
                    model_config=model_config,
                )
                db.add(db_step)

            # Get next version number
            latest_version = (
                db.query(DBChainVersion)
                .filter(DBChainVersion.chain_id == chain_id)
                .order_by(DBChainVersion.version_number.desc())
                .first()
            )
            version_number = (latest_version.version_number + 1) if latest_version else 1
            chain_version_id = f"cv_{chain_id}-v{version_number}"

            # Create new version
            snapshot = {
                "chain_id": chain_id,
                "name": name,
                "description": description,
                "steps": steps,
            }
            db_version = DBChainVersion(
                chain_version_id=chain_version_id,
                chain_id=chain_id,
                version_number=version_number,
                step_count=len(steps),
                valid=is_valid,
                snapshot=snapshot,
            )
            db.add(db_version)
            db.commit()

            return self._build_chain_from_version(db_version, db)

    # -------- Run Management --------
    def create_run(self, session_id: str, chain_version_id: str, workflow_version_id: Optional[str] = None) -> Run:
        """Create a new run."""
        from .models import WorkflowVersion, IngestionProfile, ExecutionTask as DBExecutionTask
        import pandas as pd
        import json
        
        run_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat() + "Z"

        with get_db_session() as db:
            # Validate chain
            chain_version = db.query(DBChainVersion).filter(
                DBChainVersion.chain_version_id == chain_version_id
            ).first()
            if not chain_version:
                raise ValueError(f"Chain version {chain_version_id} not found")

            # Get workflow version if provided
            ingestion_profile = None
            is_csv_workflow = False
            if workflow_version_id:
                workflow_version = db.query(WorkflowVersion).filter(
                    WorkflowVersion.workflow_version_id == workflow_version_id
                ).first()
                if workflow_version:
                    ingestion_profile = db.query(IngestionProfile).filter(
                        IngestionProfile.ingestion_profile_id == workflow_version.ingestion_profile_id
                    ).first()
                    if ingestion_profile and 'CSV' in ingestion_profile.accepted_input_types:
                        is_csv_workflow = True

            # Get all documents in session
            db_docs = db.query(DBDocument).filter(DBDocument.session_id == session_id).all()
            if not db_docs:
                raise ValueError(f"No documents found in session {session_id}")

            # Check all docs are converted
            unconverted = [d for d in db_docs if d.status != "CONVERTED"]
            if unconverted:
                raise ValueError(f"Some documents not converted: {[d.doc_id for d in unconverted]}")

            # Create run
            db_run = DBRun(
                run_id=run_id,
                session_id=session_id,
                chain_version_id=chain_version_id,
                workflow_version_id=workflow_version_id,
                status="QUEUED",
            )
            db.add(db_run)
            db.flush()

            # Handle CSV workflows: create execution_tasks per row
            if is_csv_workflow:
                for db_doc in db_docs:
                    if db_doc.file_type != 'CSV':
                        continue
                    
                    # Load CSV file
                    storage_base = Path("data/ai_bulk_doc_analysis")
                    csv_path = storage_base / db_doc.object_storage_key if db_doc.object_storage_key else None
                    if not csv_path or not csv_path.exists():
                        logger.warning(f"CSV file not found for doc {db_doc.doc_id}")
                        continue
                    
                    try:
                        df = pd.read_csv(csv_path)
                        
                        # Create execution_task for each row
                        for row_index, row in df.iterrows():
                            task_id = f"task_{run_id}_{db_doc.doc_id}_{row_index}"
                            row_data = row.to_dict()
                            
                            db_task = DBExecutionTask(
                                task_id=task_id,
                                run_id=run_id,
                                doc_id=db_doc.doc_id,
                                row_index=int(row_index),
                                row_data=row_data,  # JSONB
                                status="QUEUED"
                            )
                            db.add(db_task)
                            
                            # Create step result records for each task/step
                            for step_idx in range(1, chain_version.step_count + 1):
                                db_step_result = DBStepResult(
                                    id=str(uuid.uuid4()),
                                    run_id=run_id,
                                    doc_id=db_doc.doc_id,
                                    task_id=task_id,  # Link to task
                                    step_index=step_idx,
                                    status="QUEUED",
                                )
                                db.add(db_step_result)
                    except Exception as e:
                        logger.error(f"Error processing CSV {db_doc.doc_id}: {e}", exc_info=True)
                        raise ValueError(f"CSV processing error: {str(e)}")
            else:
                # Non-CSV: create step result records for each doc/step
                for db_doc in db_docs:
                    for step_idx in range(1, chain_version.step_count + 1):
                        db_step_result = DBStepResult(
                            id=str(uuid.uuid4()),
                            run_id=run_id,
                            doc_id=db_doc.doc_id,
                            step_index=step_idx,
                            status="QUEUED",
                        )
                        db.add(db_step_result)

            db.commit()

            # Build Run dataclass
            return Run(
                run_id=run_id,
                session_id=session_id,
                chain_version_id=chain_version_id,
                status="QUEUED",
                created_at=now,
                document_ids=[d.doc_id for d in db_docs],
            )

    def get_run(self, run_id: str) -> Optional[Run]:
        """Get a run by ID."""
        with get_db_session() as db:
            db_run = db.query(DBRun).filter(DBRun.run_id == run_id).first()
            if not db_run:
                return None

            # Get document IDs from step results
            step_results = db.query(DBStepResult).filter(DBStepResult.run_id == run_id).all()
            doc_ids = list(set(sr.doc_id for sr in step_results))

            return Run(
                run_id=db_run.run_id,
                session_id=db_run.session_id,
                chain_version_id=db_run.chain_version_id,
                status=db_run.status,
                created_at=db_run.created_at.isoformat() + "Z",
                total_input_tokens=db_run.total_input_tokens or 0,
                total_output_tokens=db_run.total_output_tokens or 0,
                document_ids=doc_ids,
            )

    def update_step_result(
        self,
        run_id: str,
        doc_id: str,
        step_index: int,
        status: str,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        latency_ms: Optional[int] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        output_object_key: Optional[str] = None,
    ):
        """Update a step result."""
        with get_db_session() as db:
            db_step_result = db.query(DBStepResult).filter(
                and_(
                    DBStepResult.run_id == run_id,
                    DBStepResult.doc_id == doc_id,
                    DBStepResult.step_index == step_index,
                )
            ).first()

            if not db_step_result:
                raise ValueError(f"Step result not found: {run_id}/{doc_id}/{step_index}")

            db_step_result.status = status
            if input_tokens is not None:
                db_step_result.input_tokens = input_tokens
            if output_tokens is not None:
                db_step_result.output_tokens = output_tokens
            if model:
                db_step_result.model = model
            if max_tokens is not None:
                db_step_result.max_tokens = max_tokens
            if temperature is not None:
                db_step_result.temperature = temperature
            if latency_ms is not None:
                db_step_result.latency_ms = latency_ms
            if error_code:
                db_step_result.error_code = error_code
            if error_message:
                db_step_result.error_message = error_message
            if output_object_key:
                db_step_result.output_object_key = output_object_key
            db_step_result.updated_at = datetime.utcnow()

            # Update run token totals
            db_run = db.query(DBRun).filter(DBRun.run_id == run_id).first()
            if db_run:
                # Recalculate totals from all step results
                all_results = db.query(DBStepResult).filter(DBStepResult.run_id == run_id).all()
                db_run.total_input_tokens = sum(sr.input_tokens or 0 for sr in all_results)
                db_run.total_output_tokens = sum(sr.output_tokens or 0 for sr in all_results)

            db.commit()

    def get_run_progress(self, run_id: str) -> Dict:
        """Get run progress with per-document or per-task status."""
        from .models import ExecutionTask as DBExecutionTask
        
        with get_db_session() as db:
            db_run = db.query(DBRun).filter(DBRun.run_id == run_id).first()
            if not db_run:
                raise ValueError(f"Run {run_id} not found")

            chain_version = db.query(DBChainVersion).filter(
                DBChainVersion.chain_version_id == db_run.chain_version_id
            ).first()
            if not chain_version:
                raise ValueError(f"Chain version {db_run.chain_version_id} not found")

            # Check if CSV workflow (has execution_tasks)
            tasks = db.query(DBExecutionTask).filter(DBExecutionTask.run_id == run_id).all()
            is_csv_workflow = len(tasks) > 0

            if is_csv_workflow:
                # CSV workflow: show per-task progress
                rows = []
                for task in tasks:
                    # Get step results for this task
                    task_results = db.query(DBStepResult).filter(
                        DBStepResult.run_id == run_id,
                        DBStepResult.task_id == task.task_id
                    ).all()
                    
                    completed_steps = sum(1 for sr in task_results if sr.status == 'SUCCESS')
                    failed_steps = sum(1 for sr in task_results if sr.status == 'ERROR')
                    total_steps = chain_version.step_count
                    
                    rows.append({
                        "doc_id": task.doc_id,
                        "task_id": task.task_id,
                        "row_index": task.row_index,
                        "status": task.status,
                        "completed_steps": completed_steps,
                        "failed_steps": failed_steps,
                        "total_steps": total_steps,
                        "progress_pct": (completed_steps / total_steps * 100) if total_steps > 0 else 0,
                    })
                
                # Aggregate stats
                total_tasks = len(tasks)
                completed_tasks = sum(1 for t in tasks if t.status == 'SUCCESS')
                failed_tasks = sum(1 for t in tasks if t.status == 'ERROR')
                
                return {
                    "run_id": run_id,
                    "status": db_run.status,
                    "is_csv_workflow": True,
                    "total_tasks": total_tasks,
                    "completed_tasks": completed_tasks,
                    "failed_tasks": failed_tasks,
                    "in_progress_tasks": total_tasks - completed_tasks - failed_tasks,
                    "tasks": rows,
                    "total_input_tokens": db_run.total_input_tokens or 0,
                    "total_output_tokens": db_run.total_output_tokens or 0,
                }
            else:
                # Non-CSV workflow: show per-document progress
                doc_ids = set()
                step_results = db.query(DBStepResult).filter(DBStepResult.run_id == run_id).all()
                for sr in step_results:
                    doc_ids.add(sr.doc_id)

                rows = []
                for doc_id in doc_ids:
                    # Get document
                    db_doc = db.query(DBDocument).filter(DBDocument.doc_id == doc_id).first()
                    if not db_doc:
                        continue

                    # Get step results for this doc
                    doc_step_results = [sr for sr in step_results if sr.doc_id == doc_id]
                    doc_step_results.sort(key=lambda x: x.step_index)

                    # Determine status
                    step_statuses = [sr.status for sr in doc_step_results]
                    max_step = max([sr.step_index for sr in doc_step_results], default=0)
                    step_label = f"R{max_step}" if max_step > 0 else "R0"

                    # Aggregate tokens
                    total_input_tokens = sum(sr.input_tokens or 0 for sr in doc_step_results)
                    total_output_tokens = sum(sr.output_tokens or 0 for sr in doc_step_results)

                    # Determine doc status
                    if all(s == "SUCCESS" for s in step_statuses) and len(step_statuses) == chain_version.step_count:
                        doc_status = "SUCCESS"
                        can_download = True
                    elif any(s == "ERROR" for s in step_statuses):
                        doc_status = "ERROR"
                        can_download = False
                    elif any(s == "RUNNING" for s in step_statuses):
                        doc_status = "RUNNING"
                        can_download = False
                    else:
                        doc_status = step_statuses[-1] if step_statuses else "QUEUED"
                        can_download = False

                    rows.append({
                    "doc_id": doc_id,
                    "filename": db_doc.original_filename,
                    "step_label": step_label,
                    "status": doc_status,
                    "input_tokens": total_input_tokens if total_input_tokens > 0 else None,
                    "output_tokens": total_output_tokens if total_output_tokens > 0 else None,
                    "can_download": can_download,
                })

            # Determine run status
            all_statuses = [sr.status for sr in step_results]
            if all(s == "SUCCESS" for s in all_statuses):
                run_status = "COMPLETE"
            elif any(s == "ERROR" for s in all_statuses):
                run_status = "ERROR"
            elif any(s == "RUNNING" for s in all_statuses):
                run_status = "RUNNING"
            else:
                run_status = db_run.status

            return {
                "run_id": run_id,
                "status": run_status,
                "rows": rows,
            }

    def get_final_output_path(self, run_id: str, doc_id: str, chain: Chain) -> Optional[Path]:
        """Get the path to the final step output."""
        if chain.step_count == 0:
            return None

        final_r_key = f"R{chain.step_count}"
        run_dir = self.storage_base / "runs" / run_id / "docs" / doc_id
        r_path = run_dir / f"{final_r_key}.md"

        if r_path.exists():
            return r_path
        return None

    # -------- Helper Methods --------
    def _db_doc_to_dataclass(self, db_doc: DBDocument) -> Document:
        """Convert DB document to dataclass."""
        return Document(
            doc_id=db_doc.doc_id,
            session_id=db_doc.session_id,
            original_filename=db_doc.original_filename,
            file_type=db_doc.file_type,
            size_bytes=db_doc.size_bytes,
            status=db_doc.status,
            error_code=db_doc.error_code,
            error_message=db_doc.error_message,
            created_at=db_doc.created_at.isoformat() + "Z",
            updated_at=db_doc.updated_at.isoformat() + "Z",
            converted_md_path=db_doc.converted_md_path,
        )

    def _build_chain_from_version(self, db_version: DBChainVersion, db: SQLASession) -> Chain:
        """Build Chain dataclass from ChainVersion."""
        # Get chain
        db_chain = db.query(DBChain).filter(DBChain.chain_id == db_version.chain_id).first()
        if not db_chain:
            raise ValueError(f"Chain {db_version.chain_id} not found")

        # Get steps
        db_steps = (
            db.query(DBChainStep)
            .filter(DBChainStep.chain_id == db_version.chain_id)
            .order_by(DBChainStep.step_index)
            .all()
        )

        steps = []
        for db_step in db_steps:
            step_dict = {
                "index": db_step.step_index,
                "title": db_step.title or "",
                "required_inputs": db_step.required_inputs,
                "prompt": db_step.prompt,
                "description": db_step.description or "",
                "model_config": db_step.model_config,
            }
            steps.append(step_dict)

        return Chain(
            chain_id=db_version.chain_id,
            name=db_chain.name,
            description=db_chain.description or "",
            updated_at=db_version.created_at.isoformat() + "Z",
            chain_version_id=db_version.chain_version_id,
            step_count=db_version.step_count,
            valid=db_version.valid,
            steps=steps,
        )

