import * as vscode from 'vscode';

interface VfsEntry {
    name: string;
    type: 'file' | 'directory';
}

interface StatResponse {
    path: string;
    type: 'file' | 'directory';
    size?: number;
}

interface DocumentRecord {
    file_id: string;
    status?: string;
    source_path?: string;
}

function ensureLeadingSlash(path: string): string {
    if (!path.startsWith('/')) {
        return `/${path}`;
    }
    return path;
}

function parseDocUri(uri: vscode.Uri): { fileId: string; path: string } {
    const fileId = uri.authority || '';
    if (!fileId) {
        throw vscode.FileSystemError.FileNotFound(`Missing file id in ${uri.toString()}`);
    }
    const path = ensureLeadingSlash(uri.path || '/');
    return { fileId, path };
}

class DocReviewApi {
    constructor(private readonly baseUrl: string) {}

    private buildUrl(path: string, params?: Record<string, string>): string {
        const url = new URL(path, this.baseUrl || (globalThis as any).location?.origin || '');
        if (params) {
            Object.entries(params).forEach(([key, value]) => {
                url.searchParams.set(key, value);
            });
        }
        return url.toString();
    }

    async getJSON<T>(path: string, params?: Record<string, string>): Promise<T> {
        const response = await fetch(this.buildUrl(path, params), {
            method: 'GET',
            credentials: 'include',
            headers: {
                'X-API-Key': 'docreview_dev_key_12345',
            },
        });
        if (!response.ok) {
            throw new Error(`Request failed: ${response.statusText}`);
        }
        return (await response.json()) as T;
    }

    async patchJSON<T>(path: string, body: unknown): Promise<T> {
        const response = await fetch(this.buildUrl(path), {
            method: 'PATCH',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': 'docreview_dev_key_12345',
            },
            body: JSON.stringify(body),
        });
        if (!response.ok) {
            throw new Error(`Request failed: ${response.statusText}`);
        }
        return (await response.json()) as T;
    }
}

class DocReviewFileSystemProvider implements vscode.FileSystemProvider {
    private readonly emitter = new vscode.EventEmitter<vscode.FileChangeEvent[]>();
    readonly onDidChangeFile: vscode.Event<vscode.FileChangeEvent[]> = this.emitter.event;

    constructor(private readonly api: DocReviewApi) {}

    watch(): vscode.Disposable {
        return new vscode.Disposable(() => {
            /* no-op */
        });
    }

    async stat(uri: vscode.Uri): Promise<vscode.FileStat> {
        const { fileId, path } = parseDocUri(uri);
        const data = await this.api.getJSON<{ stat: StatResponse }>('/api/doc_review/vfs/stat', {
            file_id: fileId,
            path,
        });
        const type =
            data.stat.type === 'directory' ? vscode.FileType.Directory : vscode.FileType.File;
        const size = data.stat.size ?? 0;
        return {
            type,
            ctime: Date.now(),
            mtime: Date.now(),
            size,
        };
    }

    async readDirectory(uri: vscode.Uri): Promise<[string, vscode.FileType][]> {
        const { fileId, path } = parseDocUri(uri);
        const data = await this.api.getJSON<{ entries: VfsEntry[] }>(
            '/api/doc_review/vfs/tree',
            { file_id: fileId, path },
        );
        return data.entries.map((entry) => [
            entry.name,
            entry.type === 'directory' ? vscode.FileType.Directory : vscode.FileType.File,
        ]);
    }

    async readFile(uri: vscode.Uri): Promise<Uint8Array> {
        const { fileId, path } = parseDocUri(uri);
        const data = await this.api.getJSON<{ content: string }>('/api/doc_review/vfs/file', {
            file_id: fileId,
            path,
        });
        return new TextEncoder().encode(data.content ?? '');
    }

    async writeFile(
        uri: vscode.Uri,
        content: Uint8Array,
        options: { create: boolean; overwrite: boolean },
    ): Promise<void> {
        const { fileId, path } = parseDocUri(uri);
        const text = new TextDecoder().decode(content);
        await this.api.patchJSON('/api/doc_review/vfs/file', {
            file_id: fileId,
            path,
            data: text,
            create: options.create,
            overwrite: options.overwrite,
        });
        this.emitter.fire([{ type: vscode.FileChangeType.Changed, uri }]);
    }

    // Unsupported operations (handled gracefully)
    createDirectory(): void | Thenable<void> {
        throw vscode.FileSystemError.NoPermissions('Directories are managed by the backend');
    }
    delete(): void | Thenable<void> {
        throw vscode.FileSystemError.NoPermissions('Deletions are not supported');
    }
    rename(): void | Thenable<void> {
        throw vscode.FileSystemError.NoPermissions('Renames are not supported');
    }
}

async function pickDocument(api: DocReviewApi): Promise<DocumentRecord | undefined> {
    const result = await api.getJSON<{ documents: DocumentRecord[] }>(
        '/api/doc_review/documents',
    );
    if (!result.documents.length) {
        vscode.window.showWarningMessage('No documents registered yet.');
        return;
    }
    const item = await vscode.window.showQuickPick(
        result.documents.map((doc) => ({
            label: doc.file_id,
            description: doc.status ?? '',
            doc,
        })),
        { placeHolder: 'Select a document to mount into VS Code' },
    );
    return item?.doc;
}

async function ensureWorkspaceFolder(uri: vscode.Uri, label: string): Promise<void> {
    const folders = vscode.workspace.workspaceFolders || [];
    const existingIndex = folders.findIndex((folder) => folder.uri.toString() === uri.toString());
    if (existingIndex >= 0) {
        vscode.workspace.updateWorkspaceFolders(existingIndex, 1, { uri, name: label });
    } else {
        vscode.workspace.updateWorkspaceFolders(
            folders.length,
            0,
            { uri, name: label },
        );
    }
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
    const config = vscode.workspace.getConfiguration('docReview');
    const apiBase = config.get<string>('apiBaseUrl') || '';
    const api = new DocReviewApi(apiBase);
    const provider = new DocReviewFileSystemProvider(api);

    context.subscriptions.push(
        vscode.workspace.registerFileSystemProvider('docrev', provider, {
            isCaseSensitive: true,
        }),
    );

    // Helper to get current selected docrev file_id from workspace
    const getActiveDocId = (): string | undefined => {
        const docrevFolder = (vscode.workspace.workspaceFolders || []).find(
            (f) => f.uri.scheme === 'docrev',
        );
        return docrevFolder?.uri.authority || undefined;
    };

    context.subscriptions.push(
        vscode.commands.registerCommand('docReview.selectDocument', async () => {
            const doc = await pickDocument(api);
            if (!doc) {
                return;
            }
            const uri = vscode.Uri.parse(`docrev://${doc.file_id}/`);
            await ensureWorkspaceFolder(uri, `Doc Review (${doc.file_id})`);
            vscode.window.showInformationMessage(`Mounted doc review run ${doc.file_id}`);
        }),
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('docReview.refresh', () => {
            const folders = vscode.workspace.workspaceFolders || [];
            folders
                .filter((folder) => folder.uri.scheme === 'docrev')
                .forEach((folder) => {
                    provider['emitter'].fire([{ type: vscode.FileChangeType.Changed, uri: folder.uri }]);
                });
            vscode.window.showInformationMessage('Doc Review workspace refreshed');
        }),
    );

    // Agent Console WebView (minimal chat + timeline placeholder)
    context.subscriptions.push(
        vscode.commands.registerCommand('docReview.openAgentConsole', async () => {
            const activeDocId = getActiveDocId();
            if (!activeDocId) {
                vscode.window.showWarningMessage('Mount a document (Doc Review: Select Document) first.');
                return;
            }
            const panel = vscode.window.createWebviewPanel(
                'docReviewAgentConsole',
                `Doc Review – Agent Console (${activeDocId})`,
                vscode.ViewColumn.Beside,
                {
                    enableScripts: true,
                    retainContextWhenHidden: true,
                },
            );
            const webviewBase = apiBase || panel.webview.asWebviewUri(vscode.Uri.parse('/')).toString();
    const html = `<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${panel.webview.cspSource} https: data:; style-src ${panel.webview.cspSource} 'unsafe-inline'; script-src ${panel.webview.cspSource}; connect-src ${webviewBase} http: https: ws: wss:;">
  <style>
    body { font: 13px/1.4 -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin: 0; padding: 0; }
    .wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 12px; height: 100vh; box-sizing: border-box; }
    .panel { border: 1px solid #ddd; border-radius: 6px; display: flex; flex-direction: column; }
    .panel h3 { margin: 0; padding: 8px 10px; border-bottom: 1px solid #eee; background: #fafafa; }
    .panel .body { padding: 8px 10px; overflow: auto; flex: 1; }
    .row { display: flex; gap: 8px; }
    textarea { width: 100%; height: 72px; }
    button { padding: 6px 10px; }
    .msg { margin: 6px 0; }
    .msg .from { font-weight: 600; margin-right: 6px;}
    .muted { color: #666; font-size: 12px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h3>Chat</h3>
      <div class="body" id="chat"></div>
      <div style="padding:8px 10px;">
        <div class="row">
          <textarea id="input" placeholder="Ask the agent, e.g. 'Run a full review'"></textarea>
        </div>
        <div class="row" style="justify-content: space-between; align-items: center;">
          <label class="muted"><input type="checkbox" id="auto" checked /> Auto-execute plan steps</label>
          <button id="send">Send</button>
        </div>
      </div>
    </div>
    <div class="panel">
      <h3>Execution Timeline</h3>
      <div class="body" id="timeline"><div class="muted">Live node events will appear here.</div></div>
      <div style="padding:8px 10px;" class="muted">WebSocket wiring TBD; this panel updates via events.</div>
    </div>
  </div>
  <script>
  const vscode = acquireVsCodeApi?.() || null;
  let socket = null; // placeholder for socket.io – future enhancement
  const chatEl = document.getElementById('chat');
  const inputEl = document.getElementById('input');
  const autoEl = document.getElementById('auto');
  const sendBtn = document.getElementById('send');
  const docId = ${JSON.stringify(activeDocId)};
  const base = ${JSON.stringify(webviewBase)};
  const post = async (path, body) => {
    const url = new URL(path, base).toString();
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify(body || {})
    });
    if (!res.ok) throw new Error('Request failed: ' + res.statusText);
    return await res.json();
  };
  const addMsg = (from, html) => {
    const div = document.createElement('div');
    div.className = 'msg';
    div.innerHTML = '<span class="from">' + from + ':</span>' + html;
    chatEl.appendChild(div);
    chatEl.scrollTop = chatEl.scrollHeight;
  };
  sendBtn.addEventListener('click', async () => {
    const text = (inputEl.value || '').trim();
    if (!text) return;
    inputEl.value = '';
    addMsg('You', text);
    try {
      const resp = await post('/api/doc_review/handle_user_message', {
        file_id: docId,
        message: text,
        auto_execute: !!autoEl.checked
      });
      const plan = resp?.result?.plan;
      const exec = resp?.result?.execution_results;
      const status = resp?.result?.status || 'pending';
      let html = '';
      if (plan?.summary) html += '<div>' + plan.summary + '</div>';
      html += '<div class="muted">Status: ' + status + '</div>';
      if (Array.isArray(plan?.plan_steps) && plan.plan_steps.length) {
        html += '<ol>' + plan.plan_steps.map(s => '<li><code>' + (s.tool||'step') + '</code> – ' + (s.reasoning||'') + '</li>').join('') + '</ol>';
      }
      if (Array.isArray(exec?.errors) && exec.errors.length) {
        html += '<div class="muted">Errors: ' + exec.errors.join(', ') + '</div>';
      }
      addMsg('Agent', html || '<em>No plan returned</em>');
    } catch (e) {
      addMsg('Agent', '<span class="muted">Error: ' + (e && e.message || e) + '</span>');
    }
  });
  </script>
</body>
</html>`;
            panel.webview.html = html;
        }),
    );

    if (!(vscode.workspace.workspaceFolders || []).some((f) => f.uri.scheme === 'docrev')) {
        vscode.window
            .showInformationMessage('Select a document to browse via Doc Review VFS', 'Select')
            .then((choice) => {
                if (choice === 'Select') {
                    vscode.commands.executeCommand('docReview.selectDocument');
                }
            });
    }
}

export function deactivate(): void {
    // no-op
}

