# Prompt Chain Execution – Developer Documentation

This document explains **how prompt chains work end-to-end** in the system: how they are defined, stored, selected, and executed step-by-step against documents using Claude APIs.

This is an **authoritative reference** for backend, frontend, and junior developers.

---

## 1. What Is a Prompt Chain?

A **prompt chain** is a structured, ordered sequence of steps where:

- Each step has its **own prompt**
- Each step declares which **inputs** it requires (Document / R0, R1, R2, …)
- Each step produces **one new output** (R1, R2, R3, …)
- Outputs from earlier steps can be reused in later steps

A chain is **deterministic and auditable**: every step execution is logged, stored, and reproducible.

---

## 2. Conceptual Example (4-Step Chain)

### Chain Logic

| Step | Inputs Used | Output Produced | Description |
|----|----|----|----|
| 1 | Document (R0) | R1 | Extract initial structured summary |
| 2 | Document (R0) + R1 | R2 | Enrich analysis using prior output |
| 3 | Document (R0) + R1 | R3 | Parallel refinement step |
| 4 | R1 + R2 + R3 | R4 | Final synthesis |

Each step uses **its own prompt**.

---

## 3. How Chains Are Authored (Markdown)

Chains are authored by users as **Markdown files** and uploaded via the UI.

Example (simplified):

```md
# Chain: 4-Step Analysis

## Step 1
Inputs: R0

Prompt:
Summarize the document into structured sections.

---

## Step 2
Inputs: R0, R1

Prompt:
Expand the summary with risk implications.

---

## Step 3
Inputs: R0, R1

Prompt:
Identify gaps and missing controls.

---

## Step 4
Inputs: R1, R2, R3

Prompt:
Produce a final consolidated assessment.
```

---

## 4. How Chains Are Stored (Backend Representation)

When a chain Markdown file is uploaded:

1. The raw `.md` file is stored in **object storage**
2. The backend **parses the chain once**
3. A structured JSON representation is saved with the chain version

### Stored Chain JSON (Example)

```json
{
  "chain_version_id": "cv_123",
  "name": "4-Step Analysis",
  "steps": [
    { "index": 1, "required_inputs": ["R0"], "prompt_object_key": "p1" },
    { "index": 2, "required_inputs": ["R0", "R1"], "prompt_object_key": "p2" },
    { "index": 3, "required_inputs": ["R0", "R1"], "prompt_object_key": "p3" },
    { "index": 4, "required_inputs": ["R1", "R2", "R3"], "prompt_object_key": "p4" }
  ]
}
```

This parsed structure becomes the **source of truth** during execution.

---

## 5. Chain Versioning & Reuse

Chains are **versioned**.

### Chain Objects

- **Chain**: logical container (name + description)
- **Chain Version**: immutable snapshot of a specific prompt file

Rules:
- A run always pins to a **specific chain version**
- Editing a chain creates a **new version**
- Old runs remain reproducible

### Stored Fields (Chain Version)

- `chain_id`
- `version_tag` (e.g. v1.0.0)
- `prompt_md_object_key`
- `parsed_steps_json`
- `step_count`
- `status` (ACTIVE / DEPRECATED)

---

## 6. Selecting a Chain (UI + Backend)

### From the UI

Users can:
- Select from **previously saved chains**
- View:
  - chain name
  - description
  - number of steps
  - version

### Backend Behavior

- `GET /chains` → returns chain metadata
- `GET /chains/{id}/versions` → returns versions
- User selects a `chain_version_id`
- That version is attached to the run

---

## 7. How a Run Uses a Chain

A **run** binds together:

- One session (documents)
- One chain version

Once created:
- The chain **cannot change** for that run
- All step executions reference the same chain JSON

---

## 8. Step Execution (Per Document)

Each step is executed **independently per document**.

### Execution Flow (Single Document)

1. Load chain JSON
2. Identify step N
3. Validate required inputs exist (R0, R1, …)
4. Load required inputs from storage
5. Build Claude request
6. Call Claude
7. Store output as R(N)
8. Record StepResult metadata

Each step = **one Claude API call**.

---

## 9. What Is Sent to Claude

Each step call includes:

- **System message** (stable runtime rules)
- **User message** containing:
  - step-specific prompt
  - labeled inputs (R0, R1, R2…)

Claude never receives the entire chain at once.

---

## 10. Storage of Outputs (R-Series)

Outputs are stored as immutable artifacts:

- R0 → normalized document
- R1..Rn → step outputs

Example object keys:

```text
sessions/{session_id}/r/{doc_id}/R0.md
runs/{run_id}/docs/{doc_id}/R1.md
runs/{run_id}/docs/{doc_id}/R2.md
```

---

## 11. Step Results & Audit Trail

For each step execution the backend stores:

- inputs used (R types + object keys)
- prompt version
- Claude model
- token usage (input/output)
- latency
- status (SUCCESS / ERROR)

This enables:
- debugging
- partial retries
- compliance and auditability

---

## 12. Failure & Retry Rules

- Steps only run if required inputs exist
- Errors are isolated per document and per step
- Failed steps can be retried without rerunning earlier steps
- A run may be PARTIAL if some documents fail

---

## 13. Why This Design

This design ensures:

- Deterministic execution
- Clear separation of concerns
- Reproducibility
- Fine-grained retries
- Safe scaling across documents

---

## 14. Key Takeaways for Developers

- Chains are **parsed once, executed many times**
- One step = one Claude call
- Inputs and outputs are explicit and versioned
- Never mutate chains mid-run
- Always store intermediate R outputs

---

**This document defines how prompt chains work in production.**
