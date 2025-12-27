# Engineering Handoff Pack (Claude API–Based)

This document updates the **System Architecture, API Contracts, Data Models, State Machines, and Job Design** to explicitly use **Anthropic Claude APIs** instead of OpenAI.

This is an **authoritative execution reference** for epics, stories, and junior‑developer handoff.

---

## 5) System Architecture & Execution Flow (Claude‑First)

### 5.1 High‑Level Architecture

#### Frontend (Web App)
- Three‑panel UI (Ingestion, Run, Results)
- Polling‑based job status (SSE later)
- Typed API client (generated from OpenAPI)

#### Backend API
- Session Service
- Document & Conversion Service
- Chain & Prompt Service
- Run & Step Orchestration Service
- Artifact & Download Service

#### Worker Layer (LLM + Conversion)
- Document conversion workers
- Claude execution workers
- Artifact build workers

#### Storage
- Object Storage: raw files, R outputs, artifacts
- Database (Postgres): metadata, runs, steps, jobs

#### Queue
- Redis / SQS / RabbitMQ
- Supports retries, DLQ, idempotency

---

### 5.2 Claude Execution Boundary

Claude is **never called directly from the API layer**.

All Claude calls:
- Occur inside **workers**
- Are wrapped with retry + timeout logic
- Log token usage and latency

API → Queue → Worker → Claude → Storage → DB

---

### 5.3 Claude‑Based Step Execution Sequence

1. FE requests step execution
2. API validates inputs and enqueues EXECUTE_STEP job
3. Worker loads R inputs from storage
4. Worker constructs Claude request
5. Claude returns completion
6. Worker writes R(n) output
7. Worker updates StepResult + token usage

---

## 6) API Contract (Claude‑Compatible)

### 6.1 Claude Model Configuration Strategy

Claude configuration is **server‑side only**.

Stored per step execution:
- `model` (e.g. `claude-3-5-sonnet`)
- `max_tokens`
- `temperature`
- `thinking_enabled` (future)

Frontend never passes raw prompts to Claude directly.

---

### 6.2 Step Execution Request Schema (Updated)

```json
{
  "run_id": "uuid",
  "step_index": 1,
  "selected_inputs": ["R0"],
  "document_ids": ["doc1", "doc2"],
  "model_config": {
    "model": "claude-3-5-sonnet",
    "max_tokens": 4096,
    "temperature": 0.2
  }
}
```

---

### 6.3 StepResult Schema (Claude‑Specific Fields)

```json
{
  "id": "uuid",
  "run_id": "uuid",
  "doc_id": "uuid",
  "step_index": 1,
  "status": "SUCCESS",
  "input_tokens": 1240,
  "output_tokens": 860,
  "model": "claude-3-5-sonnet",
  "latency_ms": 3120,
  "output_object_key": "runs/{run_id}/docs/{doc_id}/R1.md"
}
```

---

## 7) Data Model Updates (Claude‑Aware)

### 7.1 step_results (extended)

Additional columns:
- `model`
- `max_tokens`
- `temperature`
- `latency_ms`
- `claude_request_id` (for support/debugging)

Unique constraint:
- `(run_id, doc_id, step_index)`

---

### 7.2 runs (token aggregation)

Aggregate across Claude calls:
- `total_input_tokens`
- `total_output_tokens`
- `estimated_cost_usd` (optional)

---

## 8) State Machines (Unchanged, Claude‑Compatible)

Claude does not change state semantics.

### 8.1 StepResult States

- `QUEUED`
- `RUNNING`
- `SUCCESS`
- `ERROR`

Claude failures map to:
- `ERROR` with `error_code = CLAUDE_TIMEOUT | CLAUDE_RATE_LIMIT | CLAUDE_INVALID_REQUEST`

---

### 8.2 Retry Rules (Claude‑Aware)

Retry allowed:
- Timeouts
- 429 rate limits
- transient network errors

No retry:
- Invalid prompt format
- Exceeded hard token limits
- Deleted document or missing R input

---

## 9) Job & Queue Design (Claude Execution)

### 9.1 EXECUTE_STEP Job Payload (Final)

```json
{
  "job_type": "EXECUTE_STEP",
  "run_id": "uuid",
  "doc_id": "uuid",
  "step_index": 1,
  "chain_version_id": "uuid",
  "selected_inputs": ["R0"],
  "input_object_keys": ["..."],
  "prompt_object_key": "chains/{chain_id}/v1.md",
  "model_config": {
    "model": "claude-3-5-sonnet",
    "max_tokens": 4096,
    "temperature": 0.2
  },
  "idempotency_key": "step:{run_id}:{doc_id}:{step_index}"
}
```

---

### 9.2 Claude Call Wrapper (Worker Responsibility)

Workers must:
- Enforce max input token size
- Chunk inputs if needed (future)
- Capture:
  - request id
  - tokens in/out
  - latency

Claude API errors must be normalized before writing to DB.

---

### 9.3 Observability (Claude‑Specific)

Every Claude call log:
- `claude_model`
- `claude_request_id`
- `input_tokens`
- `output_tokens`
- `latency_ms`
- `job_id`, `run_id`, `doc_id`, `step_index`

---

## 10) Guardrails for Junior Developers

### Do
- Pin `chain_version_id` at run creation
- Always check existing StepResult before calling Claude
- Store prompts and outputs in object storage, not DB

### Don’t
- Call Claude from API routes
- Retry non‑transient Claude errors
- Mutate chain definitions mid‑run

---

## 11) Definition of Done (Claude Execution)

A step is considered DONE only if:
- StepResult status = SUCCESS
- R output stored and retrievable
- Token usage recorded
- Latency recorded
- RunDocument status updated

---

## 12) Future‑Proofing (Claude Roadmap)

Planned additions:
- Claude thinking mode capture
- Streaming outputs
- Multi‑model routing (Haiku vs Sonnet)
- Cost‑based throttling

---

**This document is the execution contract for Claude‑based runs.**

