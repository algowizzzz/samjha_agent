# Dual-Runtime Workers — Product Plan & Sprint Schedule

**Date:** 2026-08-15 · **Audience:** product / delivery
**Grounded in:** code review of `agent_clean @ main-mcponprem-complete` and
`deepseek-harness 0.1.0-rc.5` — full technical detail in
`docs/deepseek-harness-integration-review.md`.

## The concept in one paragraph

A worker today is a system prompt + domain data + verified workflows + a tool allowlist, with an
**agent mode** (single / multi). We add one more attribute: **runtime** — *LangChain (current)*
or *DeepSeek Harness*. It is a separate dropdown, not a third agent mode, because dsh brings its
own multi-agent tooling. Everything the worker owns plugs into dsh through the front door: SAJHA
is already a standards-compliant MCP server, and dsh is an MCP client that accepts our auth key
and per-worker scoping headers in plain config. The rollout is two experiences: **Phase 1 —
engine swap** (users keep the exact chat UI they know; dsh runs underneath), then **Phase 2 —
Harness Workbench** (the dsh UI, embedded behind our login, for pilot workers). Phase 2 is gated
because the dsh web UI ships with *no authentication* — we must put our gateway in front of it,
one harness process per user.

## User flow — if we go forward

### Admin panel
1. Create a worker exactly as today: name, system prompt, upload domain data, add verified
   workflows, tick MCP tools from the checklist.
2. **New:** a **Runtime** selector beside Agent Mode — "LangChain (default)" or
   "DeepSeek Harness (beta)".
3. **New:** picking Harness reveals a config card: **PTC toggle** (model writes one program that
   chains tools — fewer round-trips on long workflows), **sandbox level**, and built-in dsh
   tools (shell, web search, file edit) shown **off by default** for regulated workers.
4. **New:** Save auto-mints a **per-worker SAJHA API key** carrying that worker's tool allowlist
   (enforced server-side — closes a gap that exists today), and generates the harness config file.
5. **New:** worker card shows a **runtime health chip** (harness process up / restarting) and a
   link to the trajectory log of any session for audit.

### User (analyst)
1. Log in as today; see the workers you're mapped to. A harness worker carries a small "Harness"
   badge — otherwise nothing changes.
2. **Phase 1:** chat looks and behaves identically — same threads, canvas charts, file uploads.
   The engine underneath is dsh; answers on multi-step workflows arrive in fewer, larger steps.
3. Conversations resume across visits (dsh sessions are durable, including the worker's
   Python/analysis state within a thread).
4. **Phase 2:** an **"Open Workbench"** button appears on harness workers: the embedded dsh UI
   behind our JWT login — inspect every step the agent took (trajectory view), fork a session to
   try a what-if, watch live token/cache stats.
5. In the Workbench, domain data and workflows appear through the same tools
   (`document_search`, `workflow_get`, …) — the worker's scope travels with every call
   automatically.

## What plugs in today vs. what we build

| Capability | Status | Why |
|---|---|---|
| Domain data, my-data, verified workflows | **Ships today** | Reached via MCP tools; worker paths ride as headers baked into dsh's MCP config — zero code. |
| All 500+ SAJHA MCP tools (incl. in PTC mode) | **Ships today** | dsh registers MCP tools as natives and auto-includes them in PTC's typed SDK; live tool refresh included. |
| Per-worker tool allowlist | **Build** | Move enforcement into SAJHA via per-worker API keys (SAJHA already supports this; today one admin key sees everything). |
| Runtime selector + harness config card in admin panel | **Build** | New UI on the existing worker-config screen; config file generated on save. |
| Chat streaming + canvas from the dsh engine | **Build** | Adapter maps dsh's event stream to our SSE envelope; canvas protocol is prompt-driven, so it ports. |
| Thread history replay for harness threads | **Build** | Today's replay reads LangGraph checkpoints; harness threads replay from the dsh session log instead. |
| Embedded Workbench behind our login | **Decide** | dsh web UI has no auth and is single-operator; needs our gateway + one process per user. Scope for Phase 2, pilot workers only. |
| Stop button mid-run · HITL approvals · memory | **Decide** | dsh's embedding protocol has no cancel (a sibling protocol does); HITL/memory stay LangChain-only until dsh equivalents are designed. |

## Sprint plan — three sprints, decision gate at the end

**Sprint 1 · The seam** *(weeks 1–2, no user-visible change)*
Refactor the runtime boundary so a second engine can exist: extract the streaming adapter,
un-hardwire the model/checkpointer, add the `agent_runtime` field. Mint per-worker SAJHA API
keys (server-side allowlists). Fix the four platform bugs found in review (broken HITL stream,
summariser prompt bug, dead retry middleware, duplicate config tree).
**Demo:** existing workers run unchanged through the new seam; a worker's key can no longer see
other workers' tools. **Value even if dsh is rejected.**

**Sprint 2 · Engine swap** *(weeks 3–4, one pilot worker)*
Build the dsh runtime adapter (Python SDK subprocess, pooled/recycled), per-worker harness
config generation, SSE + canvas mapping, harness-thread replay. Admin panel: runtime selector,
harness config card, health chip. Flip **one non-production worker** to dsh; run the eval golden
set against both runtimes (PTC vs. current loop: workflow fidelity, token cost, latency).
**Demo:** same chat UI, side-by-side eval scorecard LangChain vs. Harness.

**Sprint 3 · Workbench + hardening** *(weeks 5–6, gate)*
Authenticated gateway proxying per-user harness processes; "Open Workbench" embed for pilot
workers. Ops hardening: session-log retention job, telemetry hard-disabled, SBOM/dependency
scan and the vendor-risk one-pager ("MIT local code; model stays Claude on our account; no data
to DeepSeek"). Stop-button decision (adopt the cancellable protocol or accept process-kill).
**Gate:** eval scorecard + Workbench pilot feedback → expand to more workers, hold at
engine-swap only, or revert (blast radius = one adapter file + generated config).

## Risks the plan is shaped around

- **Developer preview.** dsh is rc-stage and promises breaking changes — hence the seam-first
  sequencing: all dsh-specific code sits behind one adapter.
- **No auth on the dsh UI.** Never exposed directly; Workbench ships only behind our gateway,
  loopback-bound, one process per user.
- **No cancel in the embedding protocol.** Stop-generation needs the sibling ACP protocol or a
  process restart; decided in Sprint 3 before any wide rollout.
- **Vendor optics.** The brand will trigger third-party-risk review; the Sprint 3 evidence pack
  (MIT license, no telemetry, Claude stays the model) pre-empts it.
