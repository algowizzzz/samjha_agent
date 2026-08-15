# DeepSeek Harness (dsh) × SAJHA Agent Platform — Research & Code Review

**Date:** 2026-08-15
**Repos reviewed (both cloned and read at code level):**
- `dmurugan1208/agent_clean` @ branch `main-mcponprem-complete` (the SAJHA / "Durga" agent platform)
- `deepseek-ai/deepseek-harness` @ master (dsh `0.1.0-rc.x`, MIT, released 2026-08-13)

**Question under review:** Can dsh be added to the SAJHA platform as a second agent runtime
(`agent_runtime: "dsh"` per worker), what do we get for it, and what does the code actually
support today?

---

## Part 1 — SAJHA agent platform: verified architecture map

### 1.1 Topology

Two independent FastAPI services plus a static frontend:

| Service | Port | Entry | Role |
|---|---|---|---|
| Agent runtime | 8000 | `agent_clean_repo/agent_server.py` (3,937 lines) | FastAPI, ~90 routes, JWT/RBAC, worker/user CRUD, SSE agent endpoint, serves the frontend |
| SAJHA MCP server | 3002 | `sajhamcpserver/run_server.py` | FastAPI (not Flask — `CLAUDE.md` is stale), real MCP protocol `2025-11-25`, 517 tool configs |

### 1.2 Agent runtime — the facts that matter for a second runtime

- **Framework:** LangChain 1.0 `create_agent` (not LangGraph's `create_react_agent` directly), with
  LangChain 1.0 `AgentMiddleware` hooks. `agent/agent.py:8`.
- **Factory:** `create_agent_for_worker(system_prompt, tools, extra_middleware, checkpointer_override)`
  at `agent/agent.py:101` is the **only** place an agent is constructed, and it is rebuilt fresh on
  every HTTP request (`agent_server.py:2694/2696`). This is the natural seam.
- **Model:** `ChatAnthropic` only (`agent/llm_factory.py` — docstring: "All other LLM providers …
  have been disabled"). Direct Anthropic API, Bedrock commented out of requirements. Default model
  `claude-opus-4-1-20250805`, hot-swappable via `PUT /api/super/llm-config`.
- **Checkpointing:** `AsyncSqliteSaver` (dev) or `AsyncPostgresSaver` when `DATABASE_URL` is set
  (`agent_server.py:729-772`). Injected as a **module-level global** into `agent.agent` — a coupling
  a second runtime must not inherit.
- **Threading:** `thread_id = str(uuid4())`, ownership enforced by a separate `_thread_registry`
  (JSONL + Postgres), validated on resume (`agent_server.py:2702-2706`).

### 1.3 Worker config (the extension point)

Live file: `agent_clean_repo/agent/config/workers.json` (51 workers).
⚠️ `agent_clean_repo/config/workers.json` is a **dead, already-divergent duplicate** — nothing reads
it (50 vs 51 workers; `w-market-risk.enabled_tools` has 81 entries there vs 1 in the live file).
Delete it before it misleads someone.

Fields read by code (with read sites): `worker_id`, `system_prompt` (`prompt.py:135`),
`agent_mode` `"single"|"multi"` (`agent_server.py:2655,2673,2685`), `enabled_tools`
(`agent_server.py:2657` → `tools.py:578`), `domain_data_path` / `my_data_path` /
`common_data_path` / `workflows_path` / `my_workflows_path` (→ `X-Worker-*` headers,
`tools.py:28-53`), `max_concurrent_subagents`, `enable_memory` + memory tuning fields,
`hitl_triggers`, `connector_scope`.

`agent_mode: "multi"` is **not a different orchestrator** — it is single mode plus a `task()` tool
(`sub_agent_tool.py`), a `SubagentLimitMiddleware`, and ~30 prompt lines. Same graph, same loop,
same model. Only 3 of 51 workers use it. This confirms harness choice (`agent_runtime`) and
orchestration mode (`agent_mode`) are orthogonal — a new runtime should be a **new field, not a
third `agent_mode` value**.

### 1.4 Tool layer — the best news in the whole review

The agent talks to SAJHA over **plain MCP JSON-RPC 2.0 over HTTP POST** (`agent/tools.py:198-273`,
`510-516`): `POST {SAJHA_BASE}/api/mcp` with `tools/list` (cursor-paginated per MCP 2025-11-25) and
`tools/call`, authenticated by a `sja_…` API key in the `Authorization` header. SAJHA also exposes
streamable-HTTP SSE (`GET /mcp`, `/mcp/sse` with `Last-Event-ID` resumption) and WebSocket
transports (`sajha/routes/mcp_routes.py`).

**SAJHA is a real, standards-compliant MCP server.** Any MCP client — dsh included — can consume it
without an adapter. This was the make-or-break question and the answer is yes.

Two critical caveats any second runtime must reproduce:

1. **The per-worker `enabled_tools` allowlist is enforced entirely client-side** in the agent
   process (`tools.py:578-587`). SAJHA sees one omnipotent `sja_full_access_admin` key
   (`{"mode": "all"}` in `apikeys.json`). A runtime that skips the filter gets all 500+ tools with
   no server-side objection. (SAJHA *does* support per-key tool allowlists in `apikeys.json` —
   the right long-term fix is per-worker API keys so enforcement moves server-side.)
2. **Worker scoping travels as custom headers** built by `_service_headers()` (`tools.py:28-53`):
   `X-Worker-Id`, `X-User-Id`, `X-Worker-Data-Root`, `X-Worker-Common-Root`,
   `X-Worker-My-Data-Root`, `X-Worker-Verified-Workflows`, `X-Worker-My-Workflows` — sourced from
   a ContextVar set per request (`agent_server.py:2724`). A second runtime must send the same
   headers or every file/data tool resolves paths wrong.

Other tool-layer behavior to preserve: per-tool timeouts (`tools.py:71-87`), 400K-char output caps,
the `_chart_ready` contract that drives canvas chart events (`tools.py:108` →
`agent_server.py:2835`), and auto-injection of `user_id`/`worker_id` into tool args when the
schema declares them.

⚠️ Discovery runs as **import-time network I/O** — `DYNAMIC_TOOLS = discover_sajha_tools()` at
`tools.py:571` blocks up to 120 s at module import and never refreshes (new SAJHA tools require an
agent-server restart).

### 1.5 Middleware stack (what dsh would and would not replace)

Fixed stack hard-coded in `agent/agent.py:123-133`:
`DanglingToolCallMiddleware` → `SummarisationMiddleware` (180K-token trigger, compress to ~36K) →
`MessageTrimmer` (800K-char fallback) → *worker-conditional slot* → `LoopDetectionMiddleware`
(MD5 fingerprint of tool calls, warn at 3 repeats, strip tool_calls at 5) →
`ToolErrorHandlingMiddleware`. Worker-conditional: `MemoryMiddleware` (SQLite,
keyword-Jaccard — no embeddings), `TokenBudgetMiddleware`, `SubagentLimitMiddleware`,
`HumanInTheLoopMiddleware`, always-on `AuditMiddleware` (JSONL + redaction).

### 1.6 Pre-existing bugs found during this review (independent of dsh)

These are worth fixing regardless of any harness decision:

1. **HITL SSE is silently broken.** `middlewares/hitl.py:98` imports `get_stream_writer` from
   `agent_server` — that symbol does not exist there. A blanket `except Exception: pass` swallows
   the ImportError, so the `hitl_required` event is never emitted, while the frontend
   (`mcp-agent.html:5163`) sits waiting for it. (It would also pass a JSON string where the queue
   writer expects a dict.)
2. **`SummarisationMiddleware` fetches the wrong prompt.** `summariser.py:318` calls
   `get_system_prompt(worker_id)` with a string; the function expects a worker dict. The
   `AttributeError` is caught and compression silently falls back to the generic prompt — token
   accounting is off for every worker.
3. **`SubagentLimitMiddleware` docs contradict code** — `__init__.py` says clamp `[2,4]`, code
   clamps `[2,8]` (`subagent_limit.py:27-28`). `w-finance-agent` sets 5, which only works under
   the real bounds.
4. **`RetryMiddleware` is dead code** — exported and documented, never instantiated.
5. **`TokenBudgetMiddleware` and `HumanInTheLoopMiddleware` never activate** — gated on
   `max_tokens_per_query` (absent from every worker) and `hitl_triggers` (empty everywhere).
6. **Duplicate, divergent config trees** (`config/` vs `agent/config/`) — see §1.3.
7. **`CLAUDE.md` is stale** — claims Flask (it's FastAPI), 121 tools (517 configs), multi-provider
   LLM factory (Anthropic-only). Don't treat it as a spec.

### 1.7 Seams: what makes a second runtime easy vs hard

**Favorable:**
- One clean factory (`create_agent_for_worker`) already parameterized and reused recursively.
- `get_system_prompt(worker, agent_mode)` and `get_tools_for_worker(enabled_tools)` are pure
  functions callable by any runtime.
- The tool layer is transport-level `httpx` + JSON-RPC; only the thin `StructuredTool` wrapper is
  LangChain-specific.
- Worker context is ambient (ContextVar), so any runtime that sets it gets header injection and
  audit correlation for free.
- The SSE envelope is a plain-text protocol (`data: {json}` with a `type` discriminator) — any
  runtime that can yield `(text_delta | tool_start | tool_end)` drives the existing UI unchanged.

**Obstructive:**
- `llm` and `checkpointer` are module globals in `agent.agent`; `agent_server.py:2687-2691` even
  reaches around the abstraction (`import agent.agent as _ag; llm=_ag.llm`).
- **The SSE loop is the single biggest lift**: ~280 lines (`agent_server.py:2750-3011`) hard-bound
  to LangChain `astream_events` v2 event names and Anthropic content-block shapes. It needs to be
  split into (a) a runtime-specific event normalizer and (b) the reusable canvas/envelope/chart
  logic downstream.
- Two more LangGraph couplings: `aget_state` for the context gauge (`agent_server.py:3001-3006`)
  and raw checkpoint reads in thread replay (`agent_server.py:3574-3577`).
- `run_agent()` is a 410-line function; only ~lines 2654-2696 are runtime-specific, but the rest
  must be extracted before a second runtime can share it.

---

## Part 2 — DeepSeek Harness (dsh): verified code review

Reviewed from source at `/workspace/deepseek-ai/deepseek-harness` — version `0.1.0-rc.5`, MIT,
Node `^22.19 || >=24`, single squashed initial commit (2026-08-13). 1,258 source TS files,
**763 test files** — an unusually high test ratio — with generated, CI-verified docs
(`docs/config-catalog.md`, `docs/tool-catalog.md`), so the docs quoted below are provably in
sync with code. Explicit status: *"developer preview … THERE WILL BE COMPATIBILITY-BREAKING
CHANGES."*

### 2.1 What it is

An agent runtime built on **Cordis** (vendored plugin/event-bus framework): every capability —
model adapters, tools, session persistence, the agent loop, sandbox, UIs — is a plugin row in a
YAML composition (`cordis.patch.yml` layers). A row is addressed by `id`, added with `insert:`,
disabled with `disabled: true`; a patch **replaces** the targeted row's config (no deep-merge).

### 2.2 Built-in tools (verified from `docs/tool-catalog.md` + package sources)

| Family | Tools | Package |
|---|---|---|
| Shell | `bash` (one-shot + persistent PTY variant), `pwsh` | `packages/shell/tool-bash{,-persistent}` |
| Filesystem | `read`, `write`, `edit`, `read_image`, `glob`, `grep` (bundled ripgrep), `str_replace_editor` | `packages/fs/*` |
| Web | `web_search`, `web_fetch` | `packages/web/tool-web` |
| Planning | `exit_plan_mode`, `todo_write`, `create_goal`/`get_goal`/`update_goal` | `packages/plan`, `todo`, `goal` |
| Orchestration | `subagent`, `subagent_fork`, `send_message`, `interrupt_agent`, `list_agents`, `report` | `packages/subagent/*` |
| Jobs/terminals | `job_output`/`job_list`/`job_kill`, `terminal_open/send/read/signal/close/list` | `packages/jobs`, `terminal` |
| Skills/workflows | `skill`, `workflow`, `ralph` | `packages/skill`, `workflow` |
| Interaction | `ask_user_question` | `packages/interaction/tool-ask-user` |
| Code Mode | `run_code` (reserved transport) | `packages/core/tools/src/code-mode.ts` |
| Opt-in | `lsp`, `session_search`/`session_trace`, `schedule_*`, `cordis_*` (runtime introspection) | various |
| MCP | `mcp__<server>__<tool>` | `packages/mcp/mcp-client` |

**Runtime modes are agent presets** (per-session compositions in `apps/cli/config/agent-presets/`):
`standard` (full set), `code` (= standard + `tool-presentation: {mode: code}` → PTC),
`minimal` (only persistent `bash` + `str_replace_editor`, for benchmark isolation),
`cordis` (creative — standard + runtime introspection/preset-authoring skills).

### 2.3 MCP support — first-class, and compatible with SAJHA

Package `@deepseek-ai/dsh-mcp-client` uses the **official `@modelcontextprotocol/sdk`**, one
plugin instance per server. Transports: **`stdio`** and **`streamable-http`** (no legacy
HTTP+SSE arm). Config (from `packages/mcp/mcp-client/src/index.ts:98`):

```yaml
- id: mcp-sajha
  name: '@deepseek-ai/dsh-mcp-client'
  config:
    serverName: sajha
    transport: streamable-http
    url: http://127.0.0.1:3002/mcp
    headers:
      Authorization: !!js process.env.SAJHA_API_KEY
      X-Worker-Id: w-market-risk
      X-Worker-Data-Root: ./data/workers/w-market-risk/domain_data
      # … remaining X-Worker-* headers
    toolCallTimeoutMs: 120000
    failOnStartupError: true
    reconnect: { enabled: true, maxAttempts: 10 }
```

This matters twice for us:
- **`headers` is an arbitrary map** → SAJHA's non-Bearer `Authorization: sja_…` key *and* the
  per-worker `X-Worker-*` scoping headers can be baked into a per-worker MCP row. The worker
  scoping problem from §1.4 is solved by configuration, not code.
- Tools register as **native tools** (`mcp__sajha__duckdb_query` etc.), indistinguishable from
  built-ins, and are **automatically included in Code Mode's generated typed SDK** — so PTC works
  over SAJHA tools with zero extra work.

Discovery awaits `tools/list` before the first turn, re-syncs atomically on
`notifications/tools/list_changed` (fixes SAJHA's restart-to-refresh problem from §1.4), and
reconnects with exponential backoff. Known limits: MCP **tools only** (Resources/Prompts
deferred); image/audio result blocks become placeholders; stdio MCP servers are spawned
*outside* the sandbox (why none ship enabled).

### 2.4 Model providers

Two adapters on the `ctx.llm` seam: `dsh-llm-deepseek` (default: `deepseek-v4-flash`) and
**`dsh-llm-pi-ai`** — a generic multi-provider adapter, mounted dormant, configured via YAML:

```yaml
- id: llm
  name: '@deepseek-ai/dsh-llm-pi-ai'
  config:
    providers:
      anthropic:
        apiKeyEnv: ANTHROPIC_API_KEY
        models:
          - id: claude-sonnet-4-5
            contextWindow: 200000
```

**Anthropic direct API works with an API key** — matching agent_clean's current provider exactly.
⚠️ **Bedrock does not**: `docs/user/guide/providers.md` states Bedrock/Vertex/Azure "use native
credentials … filling only the API-key field does not configure them." If a future deployment
requires Bedrock, that's an open item. Also: the Python SDK server's only auto-mount fallback is
the DeepSeek adapter — non-DeepSeek providers must be composed in your own `cordis.yml`.

Secrets never live in YAML (`apiKeyEnv` references; literals in `$DSH_HOME/.credentials.yaml`).

### 2.5 Programmatic control — the real integration surface

**There is no HTTP API for embedding.** The options are:

1. **Python SDK** (`deepseek-harness-sdk` on PyPI, source `python/sdk/`) — bundles a single-file
   `dsh-jsonrpc-agent` Node executable (no system Node needed) and speaks **newline-delimited
   JSON-RPC 2.0 over stdio**. `DeepSeekHarness(provider, model, cwd, session_root, cordis=…)` →
   `Session.run(input, on_notification=…)` → `RunResult(final_response, finish_reason, events)`.
   Reusing a `session_id` continues the durable conversation **and** its persistent bash PTY.
   Protocol methods: `initialize`, `session/prompt` (enqueue receipt only), `shutdown`; server
   pushes `session.event` (every session event, full log envelope), `session.status`,
   `subagent.started/finished`.
   **Gaps: no cancel, no per-session close, no protocol version negotiation.** Sessions live
   until process shutdown.
2. **ACP server** (`packages/acp/acp`) — JSON-RPC stdio with `session/new`, `session/prompt`,
   **`session/cancel`** (the one surface with cancellation), `session/request_permission`. But it
   rejects `mcpServers` in `session/new` — MCP must come from the cordis composition, which is
   fine for us.
3. **Headless one-shot** (`dsh --profile headless "task"`) — no port, prints final text, exit
   code from `turn/end` kind.

The web UI (`dsh web`, port 3080) has **no auth at all** — loopback-only by design, `--host
0.0.0.0` deliberately blocked ("unsupported until remote access has an authentication layer").
It is a developer console, not a deployable service.

### 2.6 Sessions & trajectory log (the eval-framework prize)

`packages/session/session-persistence-jsonl`: append-only zstd-compressed JSONL under
`$DSH_HOME/sessions/`, immutable header + one `SessionEvent` per line, contiguous `seq`, every
batch fsynced, crash recovery truncates incomplete tails and synthesizes tool-call closers.
**Hard runtime invariant: anything the model sees must be reconstructable from the log**
(model-visible ⟺ logged). Fork (`ctx.sessions.fork`) and seeded resume are real APIs.
Limits: format is v0 with **no migration path**, **nothing ever deletes session files**, one
live writer per session.

### 2.7 PTC / Code Mode — real and well-specified

`tools: {mode: native | code | both}`. In `code` mode the model gets exactly one tool,
`run_code({code, description})`, where `code` is the body of an async TypeScript function; a
generated, deterministic **typed SDK** (`await tools.name(args)`) covers every visible tool —
**including MCP tools** — and any direct tool call resolves to `UNKNOWN_TOOL`. Each binding call
re-enters the full guarded tool pipeline (pre-execute → guards → execute → post-execute), so
approval/audit hooks still fire. Only what the program prints/returns re-enters model context —
this is the token-tax fix for chained SAJHA workflows.

Sandbox for the generated program: fresh `worker_threads.Worker` per run, `env: {}` (zero
ambient credentials), hostile-peer message validation, compute + wall-clock budgets → hard
`worker.terminate()`. Explicitly labelled *"containment, not a security boundary"*; known gap:
OS processes spawned by the program survive termination.

### 2.8 Restricting a worker to MCP-only tools — the intended path

In the web profile every model-facing tool row is already `disabled: true` at the base layer;
tools are mounted per-preset. **An agent preset whose `agent.cordis.yml` contains only a persona
row + `dsh-mcp-client` rows yields an agent whose entire catalog is `mcp__*`** — confirmed by
docs ("a child that joins nothing reaches the model with no tools at all"). No shell, no web
search, no file tools, by construction rather than by deny-list. Secondary mechanisms:
`ctx.tools.restrict({allow,deny})` (visibility mask, agent-scoped, explicitly *not* an authority
boundary), `ctx.tools.guard()` (monotonic deny-only), and the `tools/pre-execute`
allow/deny/ask waterfall.

### 2.9 Sandboxing

`SandboxMode = read-only | workspace-write | danger-full-access`, backends: Linux
bwrap/Landlock (native runner in-tree), macOS Seatbelt, Windows ACL restricted-token
(reports `partial`). **Scope is filesystem writes only** — "reads, network access, and process
visibility are not confined." Enforcement level is a reported fact (`full`/`partial`);
fail-closed when confinement is unavailable. An experimental `packages/e2b/` swaps fs+subprocess
for a remote sandbox (POC only).

### 2.10 Maturity scorecard

**For:** 763 test files / 1,258 source files; fixture-based replay testing (recorded session
logs as CI fixtures — no API keys, no LLM-judge flake); ~15 `verify-*` CI gates; generated docs
checked for freshness; every package README carries mandatory "Model Experience" and "Known
Limitations" sections; only 58 TODOs; telemetry **off by default** with a hard opt-out env.

**Against:** `0.1.0-rc.5` with explicit breaking-change warnings; single squashed commit (no
history/velocity signal); SDK protocol has no cancel/close/version-negotiation; session format
v0, no migration, no pruning; zero auth on the web server; if telemetry *is* opted in, there is
**no redaction** (message text, tool args, paths export to DeepSeek's OTLP endpoint —
`DSH_TELEMETRY_DISABLED` must be pinned in any bank deployment); `DSH_TOOLS_MODE` env seam
marked "TEMPORARY workaround"; Bedrock not API-key-configurable.

---

## Part 3 — Integration design (against the actual code)

### 3.1 Config: a new orthogonal field

`agent/config/workers.json` (the **live** file — not `config/workers.json`):

```json
{
  "worker_id": "w-market-risk",
  "agent_mode": "multi",
  "agent_runtime": "langchain",   // default; or "dsh"
  "dsh": {                         // only read when agent_runtime == "dsh"
    "preset": "sajha-mcp-only",   // generated agent preset id
    "tools_mode": "code",         // native | code  (PTC toggle per worker)
    "sandbox_mode": "read-only"
  }
}
```

`agent_mode` stays untouched — harness and orchestration are orthogonal, and dsh has its own
`subagent` tool if a dsh worker needs multi-agent behavior.

### 3.2 Runtime port

Introduce `agent/runtimes/` with a two-method protocol, registry keyed on
`worker.get("agent_runtime", "langchain")`:

```python
class AgentRuntime(Protocol):
    def build(self, worker: dict, system_prompt: str, agent_mode: str) -> "Runner": ...

class Runner(Protocol):
    def astream(self, query: str, thread_id: str) -> AsyncIterator[NormalizedEvent]: ...
    # NormalizedEvent kinds: text_delta | tool_start | tool_end | usage | error | done
```

- `runtimes/langchain_runtime.py` = today's `create_agent_for_worker` + the `astream_events`
  normalization moved out of `agent_server.py` (lines 2750–3011). The `llm` and `checkpointer`
  module globals move into this runtime instance.
- `agent_server.run_agent()` keeps everything downstream of normalization unchanged: the
  `[CANVAS]` marker parser, envelope fallback, `_chart_ready` chart events, audit, thread
  registry. Those already operate on plain text/dicts.

### 3.3 `agent/runtimes/dsh_runtime.py`

Per-worker embedding via the Python SDK (`deepseek-harness-sdk`):

1. **Generated composition.** On worker save (or first use), render
   `data/dsh/<worker_id>/cordis.yml` from a template:
   - persona row ← `worker["system_prompt"]` (+ the canvas addendum — the `[CANVAS]` marker
     protocol is prompt-driven, so it ports for free);
   - `dsh-llm-pi-ai` row ← existing `llm_config.json` (Anthropic + `ANTHROPIC_API_KEY` env ref);
   - one `dsh-mcp-client` row → `streamable-http` at `{SAJHA_BASE}/mcp` with `Authorization`
     and all `X-Worker-*` headers baked in (values from the same fields `_service_headers()`
     reads today);
   - `tool-presentation: {mode: code}` when `dsh.tools_mode == "code"`;
   - sandbox-policy row: `mode: read-only` (or `workspace-write` with
     `workspaceRoot: <worker my_data_path>`), `approval: never` for headless operation;
   - **no shell/fs/web tool rows** for regulated workers → MCP-only catalog by construction
     (§2.8).
2. **Process lifecycle.** One `DeepSeekHarness` subprocess per (worker, generation), pooled and
   recycled after N runs / M minutes — this is forced by the protocol's missing
   `session/close` (§2.5): a long-lived process accumulates sessions forever. Recycling also
   picks up worker-config changes (headers are baked at process start).
3. **Session mapping.** `thread_id` (uuid4) → dsh `session_id` (sanitized). dsh persists the
   conversation in its own JSONL log — **the LangGraph checkpointer is bypassed on this path**,
   which means `GET /api/agent/threads/{id}/messages` (which reads raw checkpoint internals,
   `agent_server.py:3574`) must learn to replay from the dsh session log for dsh threads.
   The `_thread_registry` ownership layer is runtime-agnostic and works unchanged.
4. **Streaming.** `session.run(query, on_notification=cb)`: map `session.event` payloads →
   `NormalizedEvent` (`assistant/chunk` → `text_delta`; tool call start/end events →
   `tool_start`/`tool_end` with the `mcp__sajha__` prefix stripped so the frontend and audit
   see familiar tool names).
5. **Enforcement of `enabled_tools`.** dsh has no per-root-session tool filter exposed in
   config (only subagent `toolFilter`), so do it **server-side, which is where it always
   belonged**: mint one SAJHA API key per worker in `sajhamcpserver/config/apikeys.json` with
   that worker's tool allowlist (SAJHA already supports per-key allowlists —
   `_handle_tools_list` filters by permission). This closes the §1.4 client-side-only
   enforcement hole for *both* runtimes.
6. **Interrupts.** The stdio SDK protocol cannot cancel a running turn. Options, in order:
   run via the **ACP server** instead (has `session/cancel`, same embedding effort), or accept
   kill-the-subprocess as the stop mechanism for the spike (crash-safe by design — the session
   log recovers with synthesized tool closers).

### 3.4 What dsh replaces vs what stays

| SAJHA piece | dsh equivalent | Verdict |
|---|---|---|
| `DanglingToolCallMiddleware` | crash recovery synthesizes tool closers in the log | replaced |
| `SummarisationMiddleware` / `MessageTrimmer` | built-in compaction (disabled in minimal preset, on elsewhere) | replaced |
| `LoopDetectionMiddleware` | loop/termination logic in agent-loop package | mostly replaced |
| `RetryMiddleware` (dead) | `dsh-llm-retry` | replaced (and actually wired) |
| `AuditMiddleware` JSONL | append-only session log, model-visible ⟺ logged invariant | superset |
| `MemoryMiddleware`, HITL, token budget | no direct equivalent (guards/approval exist but differ) | stays ours |
| Canvas protocol, thread ownership, JWT/RBAC, worker CRUD | — | stays ours |

### 3.5 Deployment deltas

- Node ^22.19/24 requirement — but the Python SDK **bundles its own runtime**, so the container
  image needs no system Node; pin the wheel version.
- Pin `DSH_TELEMETRY_DISABLED=1` in every environment (belt-and-braces on top of the
  off-by-default).
- `$DSH_HOME` per deployment → session logs on the persistent volume; add a retention cron
  (dsh will never prune them itself).
- SBOM/dependency scan of `@deepseek-ai/dsh` + the Python wheel for the vendor-risk file; the
  "harness is MIT local code, model stays Claude via our existing Anthropic account, telemetry
  hard-disabled" line is now backed by specific code citations (§2.10).

---

## Part 4 — Recommendation

**Do the two-week spike, but with three revisions to the earlier sketch, forced by what the
code actually says:**

1. **Enforce `enabled_tools` server-side with per-worker SAJHA API keys** — dsh can't do
   per-session allowlists from config, and SAJHA already supports per-key allowlists. This
   fixes a real security gap in the current platform (any client with the shared
   `sja_full_access_admin` key gets all 517 tools) regardless of the harness decision.
2. **Embed via stdio (Python SDK or ACP), not HTTP** — there is no HTTP embedding surface, and
   the web server has no auth. Plan for subprocess pooling/recycling because sessions can't be
   closed, and use ACP if turn cancellation is a launch requirement.
3. **Keep Anthropic-direct as the model path** — works today via `llm-pi-ai` + API key.
   Bedrock is *not* just-an-API-key in dsh; if Bedrock ever becomes mandatory, that's a
   blocking item to re-verify.

**Sequencing:** the §3.2 runtime port refactor (extract event normalization from
`run_agent()`, de-globalize `llm`/`checkpointer`) is pure win even if dsh is rejected — it's
also the seam for a Claude Agent SDK runtime later. Do it first, land it, then add
`dsh_runtime.py` behind `agent_runtime: "dsh"` on one non-production worker, point the eval
framework's golden set at both runtimes, and compare PTC vs ReAct on workflow-execution
fidelity and token cost.

**Independent of dsh, fix now:** the seven §1.6 bugs — especially the silently-broken HITL
stream and the summariser prompt-type bug — and delete the divergent duplicate config tree.

**What would change the answer:** if the rc-series breaks compatibility during the spike
(explicitly promised), the port abstraction contains the blast radius to `dsh_runtime.py` +
the generated YAML. That containment is the real insurance policy, and it's why the runtime
port comes first.
