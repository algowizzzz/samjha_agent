"""
Backend service layer for AI Bulk Doc Analysis (isolated feature).

This module handles:
- Session/document management
- Chain management
- Run execution tracking
- PDF → Markdown conversion (placeholder; real conversion will be worker-based)

All data is stored in-memory for now (can be replaced with DB later).
"""

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Document:
    doc_id: str
    session_id: str
    original_filename: str
    file_type: str
    size_bytes: int
    status: str  # QUEUED | PROCESSING | CONVERTED | ERROR
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    converted_md_path: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove internal path, expose preview flag instead
        d.pop('converted_md_path', None)
        d['has_converted_artifact'] = self.converted_md_path is not None
        return d


@dataclass
class Chain:
    chain_id: str
    name: str
    description: str
    updated_at: str
    chain_version_id: str
    step_count: int
    valid: bool
    steps: List[Dict]  # List of step definitions

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StepResult:
    id: str
    run_id: str
    doc_id: str
    step_index: int
    status: str  # QUEUED | RUNNING | SUCCESS | ERROR
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    model: Optional[str] = None
    latency_ms: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    output_object_key: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Run:
    run_id: str
    session_id: str
    chain_version_id: str
    status: str  # QUEUED | RUNNING | COMPLETE | ERROR
    created_at: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    document_ids: List[str] = None

    def __post_init__(self):
        if self.document_ids is None:
            self.document_ids = []

    def to_dict(self) -> dict:
        return asdict(self)


class BulkDocService:
    """In-memory service for Bulk Doc Analysis (replace with DB-backed service later)."""

    def __init__(self, storage_base: Optional[Path] = None):
        self.storage_base = storage_base or Path("data/ai_bulk_doc_analysis")
        self.storage_base.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, List[str]] = {}  # session_id -> [doc_ids]
        self.documents: Dict[str, Document] = {}
        self.chains: Dict[str, Chain] = {}
        self.runs: Dict[str, Run] = {}
        self.step_results: Dict[str, StepResult] = {}  # key: f"{run_id}:{doc_id}:{step_index}"
        self._init_mock_chains()

    def _init_mock_chains(self):
        """Initialize with a few example chains for testing."""
        chain1 = Chain(
            chain_id="chain-example-1",
            name="Example 4-Step Analysis",
            description="Demonstrates a 4-step prompt chain with R-series dependencies",
            updated_at=datetime.utcnow().isoformat() + "Z",
            chain_version_id="cv-example-1-v1",
            step_count=4,
            valid=True,
            steps=[
                {"index": 1, "required_inputs": ["R0"], "prompt": "Summarize the document into structured sections.", "description": "Extract initial structured summary"},
                {"index": 2, "required_inputs": ["R0", "R1"], "prompt": "Expand the summary with risk implications.", "description": "Enrich analysis using prior output"},
                {"index": 3, "required_inputs": ["R0", "R1"], "prompt": "Identify gaps and missing controls.", "description": "Parallel refinement step"},
                {"index": 4, "required_inputs": ["R1", "R2", "R3"], "prompt": "Produce a final consolidated assessment.", "description": "Final synthesis"},
            ],
        )
        chain2 = Chain(
            chain_id="chain-incomplete",
            name="Incomplete Chain",
            description="Chain with missing steps (for testing invalid state)",
            updated_at=datetime.utcnow().isoformat() + "Z",
            chain_version_id="cv-incomplete-v1",
            step_count=2,
            valid=False,
            steps=[],
        )
        self.chains[chain1.chain_version_id] = chain1
        self.chains[chain2.chain_version_id] = chain2

    # -------- Session Management --------
    def ensure_session(self, user_id: str) -> str:
        """Get or create a session for the user."""
        session_id = f"session_{user_id}"
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return session_id

    # -------- Document Management --------
    def create_documents(self, session_id: str, files: List, user_id: str) -> List[Document]:
        """Create document records from uploaded files."""
        docs = []
        now = datetime.utcnow().isoformat() + "Z"

        for file in files:
            doc_id = str(uuid.uuid4())
            filename = file.filename or "unknown.pdf"
            size = len(file.read())
            file.seek(0)  # Reset for later processing

            doc = Document(
                doc_id=doc_id,
                session_id=session_id,
                original_filename=filename,
                file_type="PDF",
                size_bytes=size,
                status="QUEUED",
                created_at=now,
                updated_at=now,
            )
            self.documents[doc_id] = doc
            self.sessions[session_id].append(doc_id)
            docs.append(doc)

            # Store file temporarily (later: move to object storage)
            doc_dir = self.storage_base / "docs" / doc_id
            doc_dir.mkdir(parents=True, exist_ok=True)
            file_path = doc_dir / filename
            # Flask file objects support .save() method
            if hasattr(file, 'save'):
                file.save(str(file_path))
            else:
                # Fallback: write bytes directly
                file.seek(0)
                with open(file_path, "wb") as f:
                    f.write(file.read())

            # Trigger conversion (for now, synchronous; later: async worker)
            self._trigger_conversion(doc_id, file_path)

        return docs

    def _trigger_conversion(self, doc_id: str, file_path: Path):
        """Trigger PDF → Markdown conversion (placeholder; real conversion will be async worker)."""
        doc = self.documents.get(doc_id)
        if not doc:
            return

        # For now: mark as PROCESSING, then simulate CONVERTED after delay
        # Real implementation will:
        # 1. Enqueue a conversion job
        # 2. Worker processes PDF → MD
        # 3. Store artifact in object storage
        # 4. Update status via polling/events
        doc.status = "PROCESSING"
        doc.updated_at = datetime.utcnow().isoformat() + "Z"

        # Simulate conversion success (replace with real conversion later)
        try:
            # Placeholder: just mark as converted (real conversion would use PyPDF2/pdfplumber/etc)
            md_path = file_path.with_suffix(".md")
            md_path.write_text(f"# Converted from {doc.original_filename}\n\n(Conversion placeholder)\n")
            doc.converted_md_path = str(md_path)
            doc.status = "CONVERTED"
            doc.updated_at = datetime.utcnow().isoformat() + "Z"
            logger.info(f"Document {doc_id} marked as CONVERTED (placeholder)")
        except Exception as e:
            doc.status = "ERROR"
            doc.error_code = "CONVERSION_FAILED"
            doc.error_message = str(e)
            doc.updated_at = datetime.utcnow().isoformat() + "Z"
            logger.error(f"Conversion failed for {doc_id}: {e}")

    def list_documents(self, session_id: str) -> List[Document]:
        """List all documents in a session."""
        doc_ids = self.sessions.get(session_id, [])
        return [self.documents[did] for did in doc_ids if did in self.documents]

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.documents.get(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """Delete an errored document (only allowed for ERROR status)."""
        doc = self.documents.get(doc_id)
        if not doc:
            return False
        if doc.status != "ERROR":
            raise ValueError("Can only delete documents with ERROR status")
        # Remove from session
        if doc.session_id in self.sessions:
            self.sessions[doc.session_id] = [d for d in self.sessions[doc.session_id] if d != doc_id]
        # Remove document
        del self.documents[doc_id]
        return True

    # -------- Chain Management --------
    def list_chains(self) -> List[Chain]:
        """List all available chains."""
        return list(self.chains.values())

    def get_chain(self, chain_version_id: str) -> Optional[Chain]:
        """Get a chain by version ID."""
        return self.chains.get(chain_version_id)

    def get_chain_by_id(self, chain_id: str) -> Optional[Chain]:
        """Get a chain by chain_id (not version_id)."""
        for chain in self.chains.values():
            if chain.chain_id == chain_id:
                return chain
        return None

    def create_chain(self, user_id: str, name: str, description: str, steps: List[Dict]) -> Chain:
        """Create a new chain."""
        # Validate chain structure
        is_valid, error_msg = self._validate_chain_structure(steps)
        if not is_valid:
            raise ValueError(f"Invalid chain structure: {error_msg}")

        chain_id = f"chain_{uuid.uuid4().hex[:12]}"
        chain_version_id = self._generate_chain_version_id(chain_id)

        now = datetime.utcnow().isoformat() + "Z"

        chain = Chain(
            chain_id=chain_id,
            name=name,
            description=description,
            updated_at=now,
            chain_version_id=chain_version_id,
            step_count=len(steps),
            valid=is_valid,
            steps=steps,
        )

        self.chains[chain_version_id] = chain
        return chain

    def update_chain(self, chain_id: str, name: str, description: str, steps: List[Dict]) -> Chain:
        """Update an existing chain (mutates in place)."""
        # Find chain by chain_id
        chain = self.get_chain_by_id(chain_id)
        if not chain:
            raise ValueError(f"Chain not found: {chain_id}")

        # Validate new chain structure
        is_valid, error_msg = self._validate_chain_structure(steps)
        if not is_valid:
            raise ValueError(f"Invalid chain structure: {error_msg}")

        # Update chain metadata and steps
        chain.name = name
        chain.description = description
        chain.step_count = len(steps)
        chain.steps = steps
        chain.valid = is_valid
        chain.updated_at = datetime.utcnow().isoformat() + "Z"

        return chain

    def _validate_chain_structure(self, steps: List[Dict]) -> tuple[bool, Optional[str]]:
        """
        Validate chain structure:
        - At least 1 step required
        - Step indices must be sequential (1, 2, 3...)
        - Each step must have required_inputs (array) and prompt (non-empty)
        - Inputs must be valid: R0 always valid, R1 only if step 1 exists, etc.
        """
        if not steps or len(steps) == 0:
            return False, "At least one step is required"

        # Check step indices are sequential starting at 1
        indices = [s.get("index") for s in steps]
        expected_indices = list(range(1, len(steps) + 1))
        if indices != expected_indices:
            return False, f"Step indices must be sequential starting at 1, got {indices}"

        # Validate each step
        for step in steps:
            index = step.get("index")
            required_inputs = step.get("required_inputs", [])
            prompt = step.get("prompt", "")

            # Check required_inputs is an array
            if not isinstance(required_inputs, list):
                return False, f"Step {index}: required_inputs must be an array"

            # Check at least one input
            if len(required_inputs) == 0:
                return False, f"Step {index}: At least one required_input is required"

            # Check prompt is non-empty
            if not prompt or not prompt.strip():
                return False, f"Step {index}: Prompt is required and cannot be empty"

            # Validate inputs: R0 always valid, R1 only if step 1 exists, etc.
            available_inputs = ["R0"]  # R0 always available
            for i in range(1, index):  # R1..R(N-1) available
                available_inputs.append(f"R{i}")

            invalid_inputs = [inp for inp in required_inputs if inp not in available_inputs]
            if invalid_inputs:
                return False, f"Step {index}: Invalid inputs {invalid_inputs}. Available: {available_inputs}"

        return True, None

    def _generate_chain_version_id(self, chain_id: str) -> str:
        """Generate a chain version ID."""
        # For now, simple format. Can enhance with versioning later.
        return f"cv_{chain_id}-v1"

    # -------- Run Management --------
    def create_run(self, session_id: str, chain_version_id: str) -> Run:
        """Create a new run for a session and chain version."""
        # Validate chain exists and is valid
        chain = self.chains.get(chain_version_id)
        if not chain:
            raise ValueError(f"Chain version not found: {chain_version_id}")
        if not chain.valid:
            raise ValueError(f"Chain is invalid: {chain_version_id}")

        # Validate all documents are CONVERTED
        docs = self.list_documents(session_id)
        if not docs:
            raise ValueError("No documents in session")
        unconverted = [d for d in docs if d.status != "CONVERTED"]
        if unconverted:
            raise ValueError(f"Some documents not converted: {[d.doc_id for d in unconverted]}")

        run_id = str(uuid.uuid4())
        run = Run(
            run_id=run_id,
            session_id=session_id,
            chain_version_id=chain_version_id,
            status="QUEUED",
            created_at=datetime.utcnow().isoformat() + "Z",
            document_ids=[d.doc_id for d in docs],
        )
        self.runs[run_id] = run

        # Trigger execution (for now: synchronous placeholder; later: async worker)
        self._trigger_run_execution(run_id)

        return run

    def _trigger_run_execution(self, run_id: str):
        """Trigger run execution (placeholder; real execution will be async worker)."""
        run = self.runs.get(run_id)
        if not run:
            return

        run.status = "RUNNING"
        chain = self.chains.get(run.chain_version_id)
        if not chain:
            return

        # For each document, create step results (placeholder; real execution calls Claude)
        for doc_id in run.document_ids:
            for step_idx in range(1, chain.step_count + 1):
                step_key = f"{run_id}:{doc_id}:{step_idx}"
                step_result = StepResult(
                    id=str(uuid.uuid4()),
                    run_id=run_id,
                    doc_id=doc_id,
                    step_index=step_idx,
                    status="QUEUED",
                )
                self.step_results[step_key] = step_result

        # Simulate execution (replace with real Claude calls later)
        # For now, mark all as SUCCESS with mock tokens
        for doc_id in run.document_ids:
            for step_idx in range(1, chain.step_count + 1):
                step_key = f"{run_id}:{doc_id}:{step_idx}"
                step_result = self.step_results.get(step_key)
                if step_result:
                    step_result.status = "SUCCESS"
                    step_result.input_tokens = 1200 + step_idx * 100
                    step_result.output_tokens = 900 + step_idx * 80
                    step_result.model = "claude-3-5-sonnet"
                    run.total_input_tokens += step_result.input_tokens
                    run.total_output_tokens += step_result.output_tokens

        run.status = "COMPLETE"

    def get_run_progress(self, run_id: str) -> Dict:
        """Get run progress with per-document, per-step status."""
        run = self.runs.get(run_id)
        if not run:
            return None

        # Build progress rows per document
        rows = []
        for doc_id in run.document_ids:
            doc = self.documents.get(doc_id)
            if not doc:
                continue

            # Find highest completed step and aggregate status/tokens
            max_step = 0
            latest_status = "QUEUED"
            total_input_tokens = 0
            total_output_tokens = 0
            step_statuses = []

            chain = self.chains.get(run.chain_version_id)
            chain_step_count = chain.step_count if chain else 1

            for step_idx in range(1, chain_step_count + 1):
                step_key = f"{run_id}:{doc_id}:{step_idx}"
                step_result = self.step_results.get(step_key)
                if not step_result:
                    break
                max_step = step_idx
                step_statuses.append(step_result.status)
                latest_status = step_result.status
                if step_result.input_tokens:
                    total_input_tokens += step_result.input_tokens
                if step_result.output_tokens:
                    total_output_tokens += step_result.output_tokens

            step_label = f"R{max_step}" if max_step > 0 else "R0"

            # Determine doc status: SUCCESS if all steps are SUCCESS, ERROR if any ERROR, else RUNNING/QUEUED
            if all(s == "SUCCESS" for s in step_statuses) and len(step_statuses) == chain_step_count:
                doc_status = "SUCCESS"
                can_download = True
            elif any(s == "ERROR" for s in step_statuses):
                doc_status = "ERROR"
                can_download = False
            elif any(s == "RUNNING" for s in step_statuses):
                doc_status = "RUNNING"
                can_download = False
            else:
                doc_status = latest_status
                can_download = False

            rows.append({
                "doc_id": doc_id,
                "filename": doc.original_filename,
                "step_label": step_label,
                "status": doc_status,
                "input_tokens": total_input_tokens if total_input_tokens > 0 else None,
                "output_tokens": total_output_tokens if total_output_tokens > 0 else None,
                "can_download": can_download,
            })

        return {
            "run_id": run_id,
            "status": run.status,
            "rows": rows,
        }

    def get_download_url(self, run_id: str, doc_id: str) -> Optional[str]:
        """Get download URL for a document's final output (placeholder)."""
        # Real implementation will generate signed URLs from object storage
        return f"/api/bulk-doc-analysis/runs/{run_id}/download/{doc_id}"

