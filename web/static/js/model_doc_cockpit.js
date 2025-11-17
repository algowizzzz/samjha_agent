/*
 * Model Documentation Cockpit - JavaScript
 * Adapted from doc_review_cockpit_new.html pattern
 */

console.info('[ModelDoc][cockpit] Initializing...', new Date().toISOString());

// Global variables
let codebasesListEl = null;
let chatHistory = null;
let templatesSelect = null;
let btnRunPhase1 = null;
let btnRunPhase2 = null;
let btnRunPhase3 = null;
let btnRunPhase4 = null;
let btnRegisterCodebase = null;
let documentationEditor = null;

let activeCodebaseId = null;
let activeCodebase = null;
let socket = null;
let currentRoom = null;

// Initialize DOM references
function initDOMElements() {
    codebasesListEl = document.getElementById('codebasesList');
    chatHistory = document.getElementById('chatHistory');
    templatesSelect = document.getElementById('selectTemplate');
    btnRunPhase1 = document.getElementById('btnRunPhase1');
    btnRunPhase2 = document.getElementById('btnRunPhase2');
    btnRunPhase3 = document.getElementById('btnRunPhase3');
    btnRunPhase4 = document.getElementById('btnRunPhase4');
    btnRegisterCodebase = document.getElementById('btnRegisterCodebase');
    documentationEditor = document.getElementById('documentation-editor');
}

// Status badge helper
function statusBadge(status) {
    const normalized = (status || 'new').toLowerCase();
    return `<span class="status-badge ${normalized}">${normalized}</span>`;
}

// Show phase view
function showPhaseView(phaseNumber) {
    ['phase-welcome', 'phase-1-view', 'phase-2-view', 'phase-3-view', 'phase-4-view', 'phase-5-view'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = 'none';
    });
    
    if (phaseNumber === 0) {
        const el = document.getElementById('phase-welcome');
        if (el) el.style.display = 'block';
        updateWorkspaceHeader('Select a codebase to begin', 'Choose a codebase from the left sidebar');
    } else {
        const el = document.getElementById(`phase-${phaseNumber}-view`);
        if (el) el.style.display = 'block';
        const titles = {
            1: 'Phase 1: Codebase Discovery',
            2: 'Phase 2: Summarization',
            3: 'Phase 3: Outline Generation',
            4: 'Phase 4: Content Generation',
            5: 'Phase 5: Final Documentation'
        };
        updateWorkspaceHeader(titles[phaseNumber] || '', '');
    }
}

function updateWorkspaceHeader(title, subtitle) {
    const titleEl = document.getElementById('workspaceTitle');
    const subtitleEl = document.getElementById('workspaceSubtitle');
    if (titleEl) titleEl.textContent = title;
    if (subtitleEl) subtitleEl.textContent = subtitle;
}

// Load codebases list
async function loadCodebases() {
    try {
        const res = await fetch('/api/model_doc/documents');
        const data = await res.json();
        if (!codebasesListEl) return;
        
        codebasesListEl.innerHTML = '';
        if (!Array.isArray(data.codebases) || !data.codebases.length) {
            codebasesListEl.innerHTML = '<div class="text-center text-muted small p-3">No codebases yet. Register a codebase to begin.</div>';
            return;
        }
        
        data.codebases.forEach(codebase => {
            const item = document.createElement('div');
            item.className = 'codebase-list-item';
            if (codebase.codebase_id === activeCodebaseId) {
                item.classList.add('active');
            }
            item.dataset.codebaseId = codebase.codebase_id;
            const metadata = codebase.codebase_metadata || {};
            item.innerHTML = `
                <div class="codebase-id">${codebase.codebase_id}</div>
                <div class="codebase-meta">${metadata.file_count || 0} files</div>
                ${statusBadge(codebase.status)}
            `;
            item.addEventListener('click', () => selectCodebase(codebase.codebase_id));
            codebasesListEl.appendChild(item);
        });
    } catch (err) {
        console.error('Failed to load codebases', err);
    }
}

// Select codebase
async function selectCodebase(codebaseId) {
    activeCodebaseId = codebaseId;
    highlightActiveCodebase();
    
    try {
        const res = await fetch(`/api/model_doc/documents/${codebaseId}`);
        const data = await res.json();
        activeCodebase = data.codebase;
        
        // Join Socket.IO room
        joinSocketRoom(codebaseId);
        
        // Show Phase 1 view
        showPhaseView(1);
        
        // Render phase 1 dashboard if data exists
        const state = activeCodebase.state || {};
        if (state.file_list && state.file_list.length > 0) {
            renderPhase1Dashboard(state);
        }
    } catch (err) {
        console.error('Failed to select codebase', err);
        alert(`Failed to load codebase: ${err.message}`);
    }
}

function highlightActiveCodebase() {
    if (!codebasesListEl) return;
    codebasesListEl.querySelectorAll('.codebase-list-item').forEach(item => {
        item.classList.toggle('active', item.dataset.codebaseId === activeCodebaseId);
    });
}

// Render Phase 1 Dashboard
function renderPhase1Dashboard(state) {
    const metadata = state.codebase_metadata || {};
    const stats = state.code_stats || {};
    
    // Update metrics
    const fileCountEl = document.getElementById('metric-file-count');
    const classCountEl = document.getElementById('metric-class-count');
    const functionCountEl = document.getElementById('metric-function-count');
    const locEl = document.getElementById('metric-loc');
    
    if (fileCountEl) fileCountEl.textContent = metadata.file_count || 0;
    if (classCountEl) classCountEl.textContent = stats.total_classes || 0;
    if (functionCountEl) functionCountEl.textContent = stats.total_functions || 0;
    if (locEl) locEl.textContent = (stats.total_lines || 0).toLocaleString();
    
    // Render code structure
    const structureViewer = document.getElementById('code-structure-viewer');
    if (structureViewer) {
        const fileList = state.file_list || [];
        let structureText = 'Codebase Structure:\n\n';
        fileList.slice(0, 50).forEach(file => {
            structureText += `${file.relative_path || file.file_path}\n`;
            const ast = file.ast_structure || {};
            if (ast.classes && ast.classes.length > 0) {
                ast.classes.forEach(cls => {
                    structureText += `  └─ class ${cls.name} (${cls.methods?.length || 0} methods)\n`;
                });
            }
            if (ast.functions && ast.functions.length > 0) {
                ast.functions.forEach(fn => {
                    structureText += `  └─ def ${fn.name}()\n`;
                });
            }
        });
        structureViewer.textContent = structureText;
    }
    
    // Show dashboard
    const dashboardEl = document.getElementById('phase1-dashboard');
    if (dashboardEl) dashboardEl.style.display = 'block';
}

// Run Phase 1
async function runPhase1Workflow() {
    if (!activeCodebaseId) {
        alert('Please select a codebase first');
        return;
    }
    
    if (btnRunPhase1) {
        btnRunPhase1.disabled = true;
        btnRunPhase1.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Running...';
    }
    
    try {
        const res = await fetch(`/api/model_doc/documents/${activeCodebaseId}/run_phase1`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || res.statusText);
        }
        
        const data = await res.json();
        activeCodebase = data.codebase;
        renderPhase1Dashboard(activeCodebase.state || {});
        await loadCodebases();
        
        if (btnRunPhase1) {
            btnRunPhase1.innerHTML = '<i class="bi bi-check-circle"></i> Phase 1 Complete';
            setTimeout(() => {
                btnRunPhase1.innerHTML = '<i class="bi bi-play-circle"></i> Start Phase 1: Codebase Discovery';
                btnRunPhase1.disabled = false;
            }, 3000);
        }
    } catch (err) {
        alert(`Phase 1 failed: ${err.message}`);
        if (btnRunPhase1) {
            btnRunPhase1.innerHTML = '<i class="bi bi-play-circle"></i> Start Phase 1: Codebase Discovery';
            btnRunPhase1.disabled = false;
        }
    }
}

// Register codebase
async function registerCodebase() {
    const pathInput = document.getElementById('codebasePathInput');
    const idInput = document.getElementById('codebaseIdInput');
    const statusEl = document.getElementById('registerStatus');
    
    const codebasePath = pathInput?.value.trim();
    if (!codebasePath) {
        alert('Please enter a codebase directory path');
        return;
    }
    
    if (btnRegisterCodebase) {
        btnRegisterCodebase.disabled = true;
        btnRegisterCodebase.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Registering...';
    }
    
    try {
        const payload = {
            codebase_path: codebasePath,
        };
        
        if (idInput?.value.trim()) {
            payload.codebase_id = idInput.value.trim();
        }
        
        const res = await fetch('/api/model_doc/documents', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || res.statusText);
        }
        
        const data = await res.json();
        if (statusEl) {
            statusEl.textContent = `✅ Registered: ${data.codebase.codebase_id}`;
        }
        
        await loadCodebases();
        
        if (data.codebase && data.codebase.codebase_id) {
            await selectCodebase(data.codebase.codebase_id);
        }
        
        // Clear inputs
        if (pathInput) pathInput.value = '';
        if (idInput) idInput.value = '';
        
        if (btnRegisterCodebase) {
            btnRegisterCodebase.innerHTML = '<i class="bi bi-plus-circle"></i> Register Codebase';
            btnRegisterCodebase.disabled = false;
        }
    } catch (err) {
        alert(`Registration failed: ${err.message}`);
        if (statusEl) {
            statusEl.textContent = `❌ Failed: ${err.message}`;
        }
        if (btnRegisterCodebase) {
            btnRegisterCodebase.innerHTML = '<i class="bi bi-plus-circle"></i> Register Codebase';
            btnRegisterCodebase.disabled = false;
        }
    }
}

// Load templates
async function loadTemplates() {
    try {
        const res = await fetch('/api/model_doc/templates');
        const data = await res.json();
        if (templatesSelect) {
            templatesSelect.innerHTML = '';
            data.templates.forEach(t => {
                const option = document.createElement('option');
                option.value = t.template_id;
                option.textContent = `${t.template_id} (${Math.round(t.size/1024)} KB)`;
                templatesSelect.appendChild(option);
            });
        }
    } catch (err) {
        console.error('Failed to load templates', err);
    }
}

// Socket.IO connection
function initSocketIO() {
    socket = io();
    
    socket.on('connect', () => {
        console.log('Socket.IO connected');
    });
    
    socket.on('model_doc:joined', (data) => {
        console.log('Joined room:', data.codebase_id);
    });
    
    socket.on('model_doc:status', (data) => {
        console.log('Status update:', data);
    });
    
    socket.on('model_doc:log', (data) => {
        console.log('Log:', data);
    });
    
    socket.on('model_doc:progress', (data) => {
        console.log('Progress:', data);
        updateProgress(data);
    });
}

function joinSocketRoom(codebaseId) {
    if (!socket) return;
    
    if (currentRoom) {
        socket.emit('model_doc:leave', { codebase_id: currentRoom });
    }
    
    socket.emit('model_doc:join', { 
        codebase_id: codebaseId,
        token: getAuthToken() 
    });
    
    currentRoom = codebaseId;
}

function getAuthToken() {
    // Try to get token from cookies or localStorage
    const cookies = document.cookie.split(';');
    for (let cookie of cookies) {
        const [name, value] = cookie.trim().split('=');
        if (name === 'token' || name === 'session_token') {
            return value;
        }
    }
    return null;
}

function updateProgress(data) {
    if (data.phase === 'content_generation') {
        const progressEl = document.getElementById('section-progress');
        const statusEl = document.getElementById('section-status');
        
        if (progressEl && data.total > 0) {
            const percent = (data.progress / data.total) * 100;
            progressEl.style.width = `${percent}%`;
            progressEl.textContent = `${Math.round(percent)}%`;
        }
        
        if (statusEl) {
            statusEl.textContent = `Generated ${data.progress} of ${data.total} sections`;
        }
    }
}

// Submit chat
async function submitChat(evt) {
    evt.preventDefault();
    if (!activeCodebaseId) {
        alert('Please select a codebase first');
        return;
    }
    
    const chatInput = document.getElementById('inputChat');
    if (!chatInput) return;
    
    const message = chatInput.value.trim();
    if (!message) return;
    
    // Add user message
    const userMessage = {
        role: 'user',
        content: message,
        timestamp: new Date().toISOString()
    };
    
    if (!activeCodebase) activeCodebase = {};
    if (!activeCodebase.state) activeCodebase.state = {};
    if (!activeCodebase.state.chat_history) activeCodebase.state.chat_history = [];
    activeCodebase.state.chat_history.push(userMessage);
    renderChatHistory(activeCodebase.state.chat_history);
    
    chatInput.value = '';
    
    try {
        const res = await fetch(`/api/model_doc/chat/${activeCodebaseId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        });
        
        if (!res.ok) {
            throw new Error(res.statusText);
        }
        
        const data = await res.json();
        
        if (data.response) {
            activeCodebase.state.chat_history.push({
                role: 'assistant',
                content: data.response,
                timestamp: new Date().toISOString()
            });
            renderChatHistory(activeCodebase.state.chat_history);
        }
    } catch (err) {
        console.error('Chat error:', err);
        alert(err.message || 'Chat service unavailable');
    }
}

function renderChatHistory(history) {
    if (!chatHistory) return;
    
    chatHistory.innerHTML = '';
    if (!Array.isArray(history) || !history.length) {
        chatHistory.innerHTML = '<em class="text-muted">No interactions yet.</em>';
        return;
    }
    
    history.forEach(entry => {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${entry.role}`;
        
        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble';
        bubble.textContent = entry.content;
        
        messageDiv.appendChild(bubble);
        chatHistory.appendChild(messageDiv);
    });
    
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// Event handlers
function setupEventHandlers() {
    initDOMElements();
    
    if (btnRunPhase1) {
        btnRunPhase1.addEventListener('click', runPhase1Workflow);
    }
    
    if (btnRegisterCodebase) {
        btnRegisterCodebase.addEventListener('click', registerCodebase);
    }
    
    const chatForm = document.getElementById('chat-form');
    if (chatForm) {
        chatForm.addEventListener('submit', submitChat);
    }
    
    const refreshBtn = document.getElementById('btn-refresh');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', loadCodebases);
    }
    
    // Proceed buttons
    document.addEventListener('click', function(e) {
        if (e.target && (e.target.id === 'btnProceedPhase2' || e.target.closest('#btnProceedPhase2'))) {
            e.preventDefault();
            showPhaseView(2);
        }
        if (e.target && (e.target.id === 'btnProceedPhase3' || e.target.closest('#btnProceedPhase3'))) {
            e.preventDefault();
            showPhaseView(3);
        }
        if (e.target && (e.target.id === 'btnProceedPhase4' || e.target.closest('#btnProceedPhase4'))) {
            e.preventDefault();
            showPhaseView(4);
        }
        if (e.target && (e.target.id === 'btnProceedPhase5' || e.target.closest('#btnProceedPhase5'))) {
            e.preventDefault();
            showPhaseView(5);
            renderPhase5View();
        }
    });
}

function renderPhase5View() {
    const state = activeCodebase?.state || {};
    const finalDoc = state.final_documentation || '';
    
    const previewEl = document.getElementById('final-documentation-preview');
    if (previewEl) {
        if (finalDoc) {
            previewEl.innerHTML = marked.parse(finalDoc);
        } else {
            previewEl.textContent = 'No documentation generated yet. Run the full workflow to generate documentation.';
        }
    }
    
    if (documentationEditor) {
        documentationEditor.value = finalDoc;
        documentationEditor.readOnly = false;
    }
}

// Initialize app
function initializeApp() {
    setupEventHandlers();
    initSocketIO();
    
    (async () => {
        try {
            await loadTemplates();
            await loadCodebases();
            showPhaseView(0);
        } catch (err) {
            console.error('Initialization error:', err);
        }
    })();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

