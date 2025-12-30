"""
SQLAlchemy models for AI Bulk Doc Analysis database.
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, Integer, BigInteger, Boolean, Text, DECIMAL, TIMESTAMP, JSON, ForeignKey, ARRAY, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()

# Detect if using PostgreSQL
_is_postgresql = False
try:
    import os
    database_url = os.getenv("DATABASE_URL", "")
    if database_url and "postgresql" in database_url.lower():
        _is_postgresql = True
except Exception:
    pass

# Use JSONB for PostgreSQL, JSON for SQLite
def get_json_type():
    """Return JSONB for PostgreSQL, JSON for SQLite."""
    # This will be resolved at runtime based on the database dialect
    # For now, we'll use a TypeDecorator that adapts
    return JSONB().with_variant(JSON(), 'sqlite')


class Session(Base):
    __tablename__ = "sessions"
    __table_args__ = {"schema": "public"} if _is_postgresql else {}

    session_id = Column(String(255), primary_key=True)
    user_id = Column(String(255), nullable=False, index=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    metadata_json = Column("metadata", JSONB().with_variant(JSON(), 'sqlite'), default={})

    # Relationships
    documents = relationship("Document", back_populates="session", cascade="all, delete-orphan")
    runs = relationship("Run", back_populates="session", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = {"schema": "public"} if _is_postgresql else {}

    doc_id = Column(String(255), primary_key=True)
    session_id = Column(String(255), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    original_filename = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)  # 'PDF', 'DOCX', 'TXT', 'MD', 'CSV'
    size_bytes = Column(BigInteger, nullable=False)
    status = Column(String(50), nullable=False, index=True)  # 'QUEUED', 'PROCESSING', 'CONVERTED', 'ERROR'
    error_code = Column(String(100), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    converted_md_path = Column(Text, nullable=True)
    object_storage_key = Column(Text, nullable=True)

    # Relationships
    session = relationship("Session", back_populates="documents")
    step_results = relationship("StepResult", back_populates="document", cascade="all, delete-orphan")
    tasks = relationship("ExecutionTask", back_populates="document", cascade="all, delete-orphan")


class Chain(Base):
    __tablename__ = "chains"
    __table_args__ = {"schema": "public"} if _is_postgresql else {}

    chain_id = Column(String(255), primary_key=True)
    name = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), index=True)
    user_id = Column(String(255), nullable=True, index=True)  # For user/team scoping
    metadata_json = Column("metadata", JSONB().with_variant(JSON(), 'sqlite'), default={})

    # Relationships
    steps = relationship("ChainStep", back_populates="chain", cascade="all, delete-orphan", order_by="ChainStep.step_index")
    versions = relationship("ChainVersion", back_populates="chain", cascade="all, delete-orphan")


class ChainStep(Base):
    __tablename__ = "chain_steps"
    __table_args__ = {"schema": "public"} if _is_postgresql else {}

    id = Column(Integer, primary_key=True, autoincrement=True)
    chain_id = Column(String(255), ForeignKey("chains.chain_id", ondelete="CASCADE"), nullable=False, index=True)
    step_index = Column(Integer, nullable=False)
    title = Column(String(500), nullable=True)  # Step title (required in UI, nullable in DB for migration)
    prompt = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    required_inputs = Column(ARRAY(String).with_variant(Text(), 'sqlite'), nullable=False)  # Array: ['R0', 'R1', ...]
    model_config = Column(JSONB().with_variant(JSON(), 'sqlite'), nullable=False, server_default='{"model": "claude-3-haiku-20240307", "max_tokens": 4096, "temperature": 0.2}')
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    chain = relationship("Chain", back_populates="steps")

    __table_args__ = (UniqueConstraint("chain_id", "step_index", name="uq_chain_step"),)


class ChainVersion(Base):
    __tablename__ = "chain_versions"
    __table_args__ = {"schema": "public"} if _is_postgresql else {}

    chain_version_id = Column(String(255), primary_key=True)
    chain_id = Column(String(255), ForeignKey("chains.chain_id", ondelete="CASCADE"), nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    step_count = Column(Integer, nullable=False)
    valid = Column(Boolean, nullable=False, default=True)
    snapshot = Column(JSONB().with_variant(JSON(), 'sqlite'), nullable=False)  # Full chain + steps JSON snapshot
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    chain = relationship("Chain", back_populates="versions")
    runs = relationship("Run", back_populates="chain_version")
    workflow_versions = relationship("WorkflowVersion", back_populates="chain_version")

    __table_args__ = (UniqueConstraint("chain_id", "version_number", name="uq_chain_version"),)


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = {"schema": "public"} if _is_postgresql else {}

    run_id = Column(String(255), primary_key=True)
    session_id = Column(String(255), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    chain_version_id = Column(String(255), ForeignKey("chain_versions.chain_version_id"), nullable=False, index=True)
    workflow_version_id = Column(String(255), ForeignKey("workflow_versions.workflow_version_id"), nullable=True, index=True)
    status = Column(String(50), nullable=False, index=True)  # 'QUEUED', 'RUNNING', 'COMPLETE', 'ERROR'
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    total_input_tokens = Column(BigInteger, default=0)
    total_output_tokens = Column(BigInteger, default=0)
    estimated_cost_usd = Column(DECIMAL(10, 6), nullable=True)
    metadata_json = Column("metadata", JSONB().with_variant(JSON(), 'sqlite'), default={})

    # Relationships
    session = relationship("Session", back_populates="runs")
    chain_version = relationship("ChainVersion", back_populates="runs")
    workflow_version = relationship("WorkflowVersion", back_populates="runs")
    step_results = relationship("StepResult", back_populates="run", cascade="all, delete-orphan")
    tasks = relationship("ExecutionTask", back_populates="run", cascade="all, delete-orphan")


class StepResult(Base):
    __tablename__ = "step_results"
    __table_args__ = {"schema": "public"} if _is_postgresql else {}

    id = Column(String(255), primary_key=True)
    run_id = Column(String(255), ForeignKey("runs.run_id", ondelete="CASCADE"), nullable=False, index=True)
    doc_id = Column(String(255), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False, index=True)
    task_id = Column(String(255), ForeignKey("execution_tasks.task_id"), nullable=True, index=True)  # For CSV task-based execution
    step_index = Column(Integer, nullable=False)
    status = Column(String(50), nullable=False, index=True)  # 'QUEUED', 'RUNNING', 'SUCCESS', 'ERROR'
    error_code = Column(String(100), nullable=True)
    error_message = Column(Text, nullable=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    model = Column(String(100), nullable=True)  # e.g., 'claude-3-haiku-20240307'
    max_tokens = Column(Integer, nullable=True)
    temperature = Column(DECIMAL(3, 2), nullable=True)
    latency_ms = Column(Integer, nullable=True)
    claude_request_id = Column(String(255), nullable=True)  # For support/debugging
    output_object_key = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    run = relationship("Run", back_populates="step_results")
    document = relationship("Document", back_populates="step_results")
    task = relationship("ExecutionTask", back_populates="step_results")

    # Unique constraint: for CSV workflows, task_id is part of uniqueness
    # For non-CSV, task_id is NULL, so (run_id, doc_id, step_index) must be unique
    # For CSV, (run_id, doc_id, step_index, task_id) must be unique
    # Using COALESCE to handle NULL task_id: use empty string for NULL in constraint
    __table_args__ = (
        UniqueConstraint("run_id", "doc_id", "step_index", "task_id", name="uq_step_result"),
    )


class JobQueueLog(Base):
    __tablename__ = "job_queue_log"
    __table_args__ = {"schema": "public"} if _is_postgresql else {}

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(255), nullable=False, unique=True)
    job_type = Column(String(100), nullable=False, index=True)  # 'CONVERT_DOC', 'EXECUTE_STEP'
    status = Column(String(50), nullable=False, index=True)  # 'QUEUED', 'RUNNING', 'SUCCESS', 'FAILED'
    payload = Column(JSONB().with_variant(JSON(), 'sqlite'), nullable=False)
    idempotency_key = Column(String(500), nullable=True, index=True)
    run_id = Column(String(255), nullable=True, index=True)
    doc_id = Column(String(255), nullable=True)
    step_index = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at = Column(TIMESTAMP(timezone=True), nullable=True)


# ============================================================================
# Workflow System Models
# ============================================================================

class Workflow(Base):
    __tablename__ = "workflows"
    __table_args__ = {"schema": "public"} if _is_postgresql else {}

    workflow_id = Column(String(255), primary_key=True)
    name = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    visibility_scope = Column(String(50), nullable=False)  # 'super' | 'domain'
    created_by = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    metadata_json = Column("metadata", JSONB().with_variant(JSON(), 'sqlite'), default={})

    # Relationships
    domains = relationship("WorkflowDomain", back_populates="workflow", cascade="all, delete-orphan")
    versions = relationship("WorkflowVersion", back_populates="workflow", cascade="all, delete-orphan")


class WorkflowVersion(Base):
    __tablename__ = "workflow_versions"
    __table_args__ = {"schema": "public"} if _is_postgresql else {}

    workflow_version_id = Column(String(255), primary_key=True)
    workflow_id = Column(String(255), ForeignKey("workflows.workflow_id", ondelete="CASCADE"), nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    ingestion_profile_id = Column(String(255), nullable=False)
    chain_version_id = Column(String(255), ForeignKey("chain_versions.chain_version_id"), nullable=False, index=True)
    export_profile_id = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    workflow = relationship("Workflow", back_populates="versions")
    chain_version = relationship("ChainVersion", back_populates="workflow_versions")
    runs = relationship("Run", back_populates="workflow_version")

    __table_args__ = (UniqueConstraint("workflow_id", "version_number", name="uq_workflow_version"),)


class WorkflowDomain(Base):
    __tablename__ = "workflow_domains"
    __table_args__ = {"schema": "public"} if _is_postgresql else {}

    id = Column(Integer, primary_key=True, autoincrement=True)
    workflow_id = Column(String(255), ForeignKey("workflows.workflow_id", ondelete="CASCADE"), nullable=False, index=True)
    domain = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    workflow = relationship("Workflow", back_populates="domains")

    __table_args__ = (UniqueConstraint("workflow_id", "domain", name="uq_workflow_domain"),)


class IngestionProfile(Base):
    __tablename__ = "ingestion_profiles"
    __table_args__ = {"schema": "public"} if _is_postgresql else {}

    ingestion_profile_id = Column(String(255), primary_key=True)
    name = Column(String(500), nullable=False)
    accepted_input_types = Column(ARRAY(String).with_variant(Text(), 'sqlite'), nullable=False)  # ['PDF', 'DOCX', 'TXT', 'MD', 'CSV']
    mode = Column(String(50), nullable=False)  # 'programmatic' | 'vision'
    vision_prompt = Column(Text, nullable=True)  # Stored in DB, not file path
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    metadata_json = Column("metadata", JSONB().with_variant(JSON(), 'sqlite'), default={})


class ExportProfile(Base):
    __tablename__ = "export_profiles"
    __table_args__ = {"schema": "public"} if _is_postgresql else {}

    export_profile_id = Column(String(255), primary_key=True)
    name = Column(String(500), nullable=False)
    format = Column(String(50), nullable=False)  # 'CSV' | 'JSON' | 'MD' | 'DOCX' | 'PDF'
    config_json = Column(JSONB().with_variant(JSON(), 'sqlite'), default={})
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())


class ExecutionTask(Base):
    __tablename__ = "execution_tasks"
    __table_args__ = {"schema": "public"} if _is_postgresql else {}

    task_id = Column(String(255), primary_key=True)
    run_id = Column(String(255), ForeignKey("runs.run_id", ondelete="CASCADE"), nullable=False, index=True)
    doc_id = Column(String(255), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False, index=True)
    row_index = Column(Integer, nullable=False)
    row_data = Column(JSONB().with_variant(JSON(), 'sqlite'), nullable=False)  # Serialized row data
    status = Column(String(50), nullable=False, index=True)  # 'QUEUED' | 'RUNNING' | 'SUCCESS' | 'ERROR'
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    run = relationship("Run", back_populates="tasks")
    document = relationship("Document", back_populates="tasks")
    step_results = relationship("StepResult", back_populates="task", cascade="all, delete-orphan")

    __table_args__ = (UniqueConstraint("run_id", "doc_id", "row_index", name="uq_execution_task"),)

