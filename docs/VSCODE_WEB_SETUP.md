## VS Code Web Setup Guide

The document review cockpit now exposes `/doc-review/ide`, which expects a local copy of the VS Code Web OSS bundle. Follow these steps to install and serve it through Flask.

### 1. Download VS Code Web assets

From the repo root:

```bash
chmod +x scripts/setup_vscode_web.sh
./scripts/setup_vscode_web.sh
```

The script downloads the latest `linux-x64-web` build from `update.code.visualstudio.com`, extracts it to `web/static/vscode-web/`, and removes the archive. You can override defaults:

```bash
VS_CODE_CHANNEL=insider VS_CODE_PLATFORM=linux-arm64-web ./scripts/setup_vscode_web.sh
```

### 2. Restart the Flask server

After the assets are in place, restart the web app (`./start_server.sh` or `flask run`). Visit:

- Dashboard: `http://localhost:5000/doc-review/ide`
- Direct IDE entry: `http://localhost:5000/doc-review/ide/launch`

If the bundle is missing, the page shows setup instructions with the exact script to run.

### 3. Reverse proxy / nginx tips

When hosting behind nginx, forward the `/doc-review/ide` and `/doc-review/ide/assets/` paths to Flask. Example snippet:

```
location /doc-review/ide/ {
    proxy_pass http://127.0.0.1:5000/doc-review/ide/;
    proxy_set_header Host $host;
}
```

Ensure static assets under `web/static/vscode-web` are readable by the web user.

### 4. Updating VS Code Web

Re-run `scripts/setup_vscode_web.sh` whenever you want to pick up a new release. The script wipes the previous contents before extracting the latest build. If you maintain a specific version, pin `VS_CODE_CHANNEL` or download a tagged `.tar.gz` from the official VS Code releases page.

### 5. Optional: Doc Review VFS Extension

The repo ships a VS Code extension at `web/vscode_extensions/doc-review-vfs/` that mounts the Document Review backend via the new VFS APIs. To build it:

```bash
cd web/vscode_extensions/doc-review-vfs
npm install
npm run compile
```

Load the resulting extension (from `dist/`) into VS Code Web using `vsce`, `code-server`, or `--extensionDevelopmentPath`. Once installed, run the command “Doc Review: Select Document” to mount `docrev://<file_id>/` as a workspace folder, and browse/edit the backend state directly inside the IDE.

