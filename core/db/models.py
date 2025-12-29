import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    String,
    DateTime,
    Boolean,
    Integer,
    Text,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


def _utcnow() -> datetime:
    return datetime.utcnow()


class Base(DeclarativeBase):
    pass


class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    agent_type: Mapped[str] = mapped_column(String(32), nullable=False)  # structured|unstructured|external
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    domain_file: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)  # Filename only (for reference)
    domain_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Actual domain file content
    data_folder: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)  # LLM model (e.g., claude-3-5-sonnet-20241022)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)


class Prompt(Base):
    __tablename__ = "prompts"

    name: Mapped[str] = mapped_column(String(128), primary_key=True)
    category: Mapped[str] = mapped_column(String(32), nullable=False, default="structured")
    current_content: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    revisions: Mapped[list["PromptRevision"]] = relationship("PromptRevision", back_populates="prompt")


class PromptRevision(Base):
    __tablename__ = "prompt_revisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    prompt_name: Mapped[str] = mapped_column(String(128), ForeignKey("prompts.name"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    editor_user_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    prompt: Mapped["Prompt"] = relationship("Prompt", back_populates="revisions")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    agent_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow, nullable=False)

    messages: Mapped[list["Message"]] = relationship("Message", back_populates="conversation")
    runs: Mapped[list["Run"]] = relationship("Run", back_populates="conversation")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id: Mapped[str] = mapped_column(String(64), ForeignKey("conversations.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user|agent
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("conversations.id"), nullable=True)
    agent_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    show_thinking: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    error_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    decider_output_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    final_sql: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    conversation: Mapped[Optional["Conversation"]] = relationship("Conversation", back_populates="runs")
    events: Mapped[list["RunEvent"]] = relationship("RunEvent", back_populates="run")
    tool_traces: Mapped[list["ToolTrace"]] = relationship("ToolTrace", back_populates="run")
    results: Mapped[Optional["RunResult"]] = relationship("RunResult", back_populates="run")


class RunEvent(Base):
    __tablename__ = "run_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("runs.id"), nullable=False)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    run: Mapped["Run"] = relationship("Run", back_populates="events")

    __table_args__ = (UniqueConstraint("run_id", "seq", name="uq_run_events_run_id_seq"),)


class RunResult(Base):
    __tablename__ = "run_results"

    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("runs.id"), primary_key=True)
    schema_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    rows_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    run: Mapped["Run"] = relationship("Run", back_populates="results")


class ToolTrace(Base):
    __tablename__ = "tool_traces"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("runs.id"), nullable=False)
    step_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    tool_args_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    tool_output_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)

    run: Mapped["Run"] = relationship("Run", back_populates="tool_traces")



