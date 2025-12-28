-- AI Bulk Doc Analysis Database Schema
-- Postgres-compatible schema for production deployment

-- ============================================================================
-- 1. SESSIONS
-- ============================================================================
-- Groups documents uploaded in a single batch
CREATE TABLE sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_created_at ON sessions(created_at);

-- ============================================================================
-- 2. DOCUMENTS
-- ============================================================================
-- Document metadata and conversion status
CREATE TABLE documents (
    doc_id VARCHAR(255) PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    original_filename VARCHAR(500) NOT NULL,
    file_type VARCHAR(50) NOT NULL,  -- 'PDF', 'DOCX', 'TXT', 'MD'
    size_bytes BIGINT NOT NULL,
    status VARCHAR(50) NOT NULL,  -- 'QUEUED', 'PROCESSING', 'CONVERTED', 'ERROR'
    error_code VARCHAR(100),
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    converted_md_path TEXT,  -- Path to R0.md artifact
    object_storage_key TEXT  -- S3/object storage key for raw file
);

CREATE INDEX idx_documents_session_id ON documents(session_id);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_created_at ON documents(created_at);

-- ============================================================================
-- 3. CHAINS
-- ============================================================================
-- Prompt chain definitions
CREATE TABLE chains (
    chain_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(255),  -- For user/team scoping
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_chains_user_id ON chains(user_id);
CREATE INDEX idx_chains_updated_at ON chains(updated_at);

-- ============================================================================
-- 4. CHAIN_STEPS
-- ============================================================================
-- Individual steps within a chain
CREATE TABLE chain_steps (
    id SERIAL PRIMARY KEY,
    chain_id VARCHAR(255) NOT NULL REFERENCES chains(chain_id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    prompt TEXT NOT NULL,
    description TEXT,
    required_inputs TEXT[] NOT NULL,  -- Array: ['R0', 'R1', ...]
    model_config JSONB DEFAULT '{"model": "claude-3-haiku-20240307", "max_tokens": 4096, "temperature": 0.2}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(chain_id, step_index)
);

CREATE INDEX idx_chain_steps_chain_id ON chain_steps(chain_id);

-- ============================================================================
-- 5. CHAIN_VERSIONS
-- ============================================================================
-- Immutable snapshots of chains at run-time
CREATE TABLE chain_versions (
    chain_version_id VARCHAR(255) PRIMARY KEY,
    chain_id VARCHAR(255) NOT NULL REFERENCES chains(chain_id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    step_count INTEGER NOT NULL,
    valid BOOLEAN NOT NULL DEFAULT true,
    snapshot JSONB NOT NULL,  -- Full chain + steps JSON snapshot
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(chain_id, version_number)
);

CREATE INDEX idx_chain_versions_chain_id ON chain_versions(chain_id);

-- ============================================================================
-- 6. RUNS
-- ============================================================================
-- Execution runs (session + chain version)
CREATE TABLE runs (
    run_id VARCHAR(255) PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    chain_version_id VARCHAR(255) NOT NULL REFERENCES chain_versions(chain_version_id),
    status VARCHAR(50) NOT NULL,  -- 'QUEUED', 'RUNNING', 'COMPLETE', 'ERROR'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    total_input_tokens BIGINT DEFAULT 0,
    total_output_tokens BIGINT DEFAULT 0,
    estimated_cost_usd DECIMAL(10, 6),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_runs_session_id ON runs(session_id);
CREATE INDEX idx_runs_chain_version_id ON runs(chain_version_id);
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_runs_created_at ON runs(created_at);

-- ============================================================================
-- 7. STEP_RESULTS
-- ============================================================================
-- Per-document, per-step execution results
CREATE TABLE step_results (
    id VARCHAR(255) PRIMARY KEY,
    run_id VARCHAR(255) NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    doc_id VARCHAR(255) NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    status VARCHAR(50) NOT NULL,  -- 'QUEUED', 'RUNNING', 'SUCCESS', 'ERROR'
    error_code VARCHAR(100),
    error_message TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    model VARCHAR(100),  -- e.g., 'claude-3-haiku-20240307'
    max_tokens INTEGER,
    temperature DECIMAL(3, 2),
    latency_ms INTEGER,
    claude_request_id VARCHAR(255),  -- For support/debugging
    output_object_key TEXT,  -- Path to R{n}.md artifact
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, doc_id, step_index)  -- Idempotency constraint
);

CREATE INDEX idx_step_results_run_id ON step_results(run_id);
CREATE INDEX idx_step_results_doc_id ON step_results(doc_id);
CREATE INDEX idx_step_results_status ON step_results(status);
CREATE INDEX idx_step_results_run_doc ON step_results(run_id, doc_id);

-- ============================================================================
-- 8. JOB_QUEUE (Optional: for tracking job state)
-- ============================================================================
-- Track Redis job state in DB for observability
CREATE TABLE job_queue_log (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(255) NOT NULL UNIQUE,
    job_type VARCHAR(100) NOT NULL,  -- 'CONVERT_DOC', 'EXECUTE_STEP'
    status VARCHAR(50) NOT NULL,  -- 'QUEUED', 'RUNNING', 'SUCCESS', 'FAILED'
    payload JSONB NOT NULL,
    idempotency_key VARCHAR(500),
    run_id VARCHAR(255),
    doc_id VARCHAR(255),
    step_index INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_job_queue_log_job_type ON job_queue_log(job_type);
CREATE INDEX idx_job_queue_log_status ON job_queue_log(status);
CREATE INDEX idx_job_queue_log_idempotency ON job_queue_log(idempotency_key);
CREATE INDEX idx_job_queue_log_run_id ON job_queue_log(run_id);

-- ============================================================================
-- VIEWS (for common queries)
-- ============================================================================

-- Run progress summary
CREATE VIEW run_progress_summary AS
SELECT 
    r.run_id,
    r.status as run_status,
    r.total_input_tokens,
    r.total_output_tokens,
    COUNT(DISTINCT sr.doc_id) as total_docs,
    COUNT(DISTINCT CASE WHEN sr.status = 'SUCCESS' THEN sr.doc_id END) as completed_docs,
    COUNT(DISTINCT CASE WHEN sr.status = 'ERROR' THEN sr.doc_id END) as failed_docs,
    COUNT(DISTINCT CASE WHEN sr.status IN ('QUEUED', 'RUNNING') THEN sr.doc_id END) as in_progress_docs
FROM runs r
LEFT JOIN step_results sr ON r.run_id = sr.run_id
GROUP BY r.run_id, r.status, r.total_input_tokens, r.total_output_tokens;

-- Document conversion status
CREATE VIEW document_conversion_status AS
SELECT 
    s.session_id,
    COUNT(*) as total_docs,
    COUNT(CASE WHEN d.status = 'CONVERTED' THEN 1 END) as converted_docs,
    COUNT(CASE WHEN d.status = 'ERROR' THEN 1 END) as error_docs,
    COUNT(CASE WHEN d.status IN ('QUEUED', 'PROCESSING') THEN 1 END) as pending_docs
FROM sessions s
LEFT JOIN documents d ON s.session_id = d.session_id
GROUP BY s.session_id;

