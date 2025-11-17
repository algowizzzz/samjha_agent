# Doc Review VFS Extension

This VS Code extension registers a `docrev://` file system provider backed by the Document Review Agent REST APIs. It lets you browse and edit run artifacts inside the VS Code Web shell.

## Build

```bash
cd web/vscode_extensions/doc-review-vfs
npm install
npm run compile
```

The compiled extension lives in `dist/extension.js`.

## Usage

1. Install/serve the VS Code Web bundle (see `docs/VSCODE_WEB_SETUP.md`).
2. Load this extension into the IDE (e.g., with `vsce package` or `--extensionDevelopmentPath`).
3. Run **Doc Review: Select Document** to mount a run (`docrev://<file_id>/`).
4. Use the Explorer to open `/phase1`, `/phase2`, `/changes`, and `/versions` files. Writes to supported files (original markdown, suggested changes JSON) are sent back to the backend via `/api/doc_review/vfs/file`.

