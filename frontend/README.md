# Dual-Runtime Workers — Frontend (Sprint 2, UI-first slice)

UI for selecting **DeepSeek Harness** as a worker runtime, built directly into the real
`agent_clean` frontend files. +122 lines across two files, no framework changes.

## What's here

| Path | What |
|---|---|
| `public/admin.html` | Modified admin panel (full file, drop-in replacement for `agent_clean_repo/public/admin.html`) |
| `public/mcp-agent.html` | Modified user chat (drop-in for `agent_clean_repo/public/mcp-agent.html`) |
| `dsh-ui.patch` | The same changes as a unified diff against `agent_clean @ main-mcponprem-complete` — apply with `git apply dsh-ui.patch` from the agent_clean repo root |
| `demo/demo-shim.js` | Mock-API shim: seeds a super-admin session and fakes the agent-server endpoints so both pages run standalone |
| `demo/shot*.png` | Screenshots of every new state |

## Run the demo locally

```sh
mkdir demo-run && cp public/*.html demo/demo-shim.js demo-run/
cp -r <agent_clean>/agent_clean_repo/public/js demo-run/   # file-tree.js is required
# add <script src="demo-shim.js"></script> right after <head> in both HTML files
cd demo-run && python3 -m http.server 8777
# open http://127.0.0.1:8777/admin.html and /mcp-agent.html
```

## What was added

**Admin panel (`admin.html`)**
- **Runtime** select (`wc-agent-runtime`) under Agent Mode: `langchain` (default) / `dsh` (beta).
- **DeepSeek Harness settings** card, shown only for `dsh`: Tool Presentation
  (native / PTC), Sandbox (read-only / workspace-write), built-in harness tool toggles
  (Shell / Web search / File editing — all off by default), read-only per-worker API key
  field, runtime health chip (Not provisioned / Provisions on save / Runtime up / error).
- Worker cards: indigo **Harness** badge next to Online/Offline.
- Create Worker modal: Runtime select (`cwnew-runtime`), sent as `agent_runtime`.
- `saveWorkerConfig()` now sends `agent_runtime` and a `dsh{tools_mode, sandbox_mode,
  enable_shell, enable_web, enable_fs}` object; `loadWorkerConfig()` populates them.

**User chat (`mcp-agent.html`)**
- **Harness** badge + **Open Workbench** button in the header, visible only when the active
  worker's `agent_runtime === 'dsh'`; updates on worker switch.
- Super-admin worker switcher labels harness workers "`<name> · Harness`".
- `openWorkbench()` opens `{AGENT_BASE}/workbench/{worker_id}` in a new tab.

## Backend contract this UI expects (Sprint 2 server work)

1. `workers.json` / worker CRUD accepts + returns: `agent_runtime` ("langchain" | "dsh",
   default "langchain") and the `dsh` object above. Unknown fields already pass through
   `_save_workers()` untouched, so persistence is free; validation goes in the PUT handler.
2. Worker responses include `dsh_status` ("unprovisioned" | "running" | "error") and
   `sajha_api_key_masked` (server mints a per-worker SAJHA key on save when runtime is dsh —
   never returns the full key after creation).
3. The login/session payload (`rg_user`) includes `agent_runtime` for the user's mapped
   worker, and `GET /api/super/workers` includes it per worker (both flow from workers.json
   automatically once stored).
4. `GET /workbench/{worker_id}` — JWT-checked reverse proxy to the per-user dsh web process
   (Phase 2; the button 404s harmlessly until it ships).
