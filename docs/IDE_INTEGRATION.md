## OpenVSCode Server Integration Guide (Doc Review)

### Prereqs
- Backend running at `http://localhost:8000`
- OpenVSCode Server running at `http://localhost:3000` (Docker or tarball)

### 1) Launch OpenVSCode Server
Docker:
```bash
docker run -it --init -p 3000:3000 -v "$(pwd):/home/workspace:cached" \
  -e DOCKER_USER="$(id -u):$(id -g)" \
  gitpod/openvscode-server
```

### 2) Open IDE via our app
- Visit `http://localhost:8000/doc-review/ide`
- Use “Open IDE in new tab” or inline iframe

### 3) Install the Doc Review VFS extension
Build:
```bash
cd web/vscode_extensions/doc-review-vfs
npm install
npm run compile
```
Install in OpenVSCode:
- Use “Install from VSIX” or load unpacked folder (depending on your deployment)

### 4) Mount a document
- Command palette → “Doc Review: Select Document”
- Choose a `file_id` → mounts `docrev://<file_id>/`
- Explorer shows `/original`, `/phase1`, `/phase2`, `/changes`, `/versions`

### 5) Open Agent Console
- Command palette → “Doc Review: Open Agent Console”
- Send messages (e.g., “Run Phase 2”)
- Watch plan/status update; artifacts refresh automatically

### Notes
- VFS endpoints:
  - GET `/api/doc_review/vfs/{tree,stat,file}`; PATCH `/api/doc_review/vfs/file`
- Agent chat:
  - POST `/api/doc_review/handle_user_message`
- Live events:
  - WebSocket streams `doc_review:*` (room-scoped per file_id)


