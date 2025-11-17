(() => {
    const api = {
        get: (url) => fetch(url, { credentials: 'same-origin' }).then(handleResponse),
        post: (url, body) =>
            fetch(url, {
                method: 'POST',
                credentials: 'same-origin',
                headers: { 'Content-Type': 'application/json' },
                body: body ? JSON.stringify(body) : '{}',
            }).then(handleResponse),
    };

    let documents = [];
    let filteredDocuments = [];
    let selectedDocumentId = null;
    let currentDocumentRecord = null;

    const dom = {
        documentList: document.getElementById('documentList'),
        documentSearch: document.getElementById('documentSearch'),
        docCount: document.getElementById('docCount'),
        btnRefreshDocs: document.getElementById('btnRefreshDocs'),
        btnRegisterDocument: document.getElementById('btnRegisterDocument'),
        btnRunPhase1: document.getElementById('btnRunPhase1'),
        btnDownloadMarkdown: document.getElementById('btnDownloadMarkdown'),
        btnOpenSource: document.getElementById('btnOpenSource'),
        btnOpenUploadDir: document.getElementById('btnOpenUploadDir'),
        inputSourcePath: document.getElementById('inputSourcePath'),
        phase1Spinner: document.getElementById('phase1Spinner'),
        selectedDocTitle: document.getElementById('selectedDocTitle'),
        statusChips: document.querySelectorAll('.status-chip[data-phase]'),
        metricPages: document.getElementById('metricPages'),
        metricWords: document.getElementById('metricWords'),
        metricHeadings: document.getElementById('metricHeadings'),
        metricImages: document.getElementById('metricImages'),
        summaryConfidence: document.getElementById('summaryConfidence'),
        execSummaryBody: document.getElementById('execSummaryBody'),
        tocScore: document.getElementById('tocScore'),
        tocEntries: document.getElementById('tocEntries'),
        tocObservations: document.getElementById('tocObservations'),
        tocGaps: document.getElementById('tocGaps'),
        templateAlignmentScore: document.getElementById('templateAlignmentScore'),
        templateCategoryList: document.getElementById('templateCategoryList'),
        strategyVerdict: document.getElementById('strategyVerdict'),
        strategyRationale: document.getElementById('strategyRationale'),
        strategyLevel: document.getElementById('strategyLevel'),
        strategySections: document.getElementById('strategySections'),
        strategyActions: document.getElementById('strategyActions'),
        welcomeMessage: document.getElementById('welcomeMessage'),
        btnRefreshWelcome: document.getElementById('btnRefreshWelcome'),
        activityFeed: document.getElementById('activityFeed'),
        uploadDirDisplay: document.getElementById('uploadDirDisplay'),
        chatHistory: document.getElementById('chatHistory'),
        chatInput: document.getElementById('chatInput'),
        btnSendChat: document.getElementById('btnSendChat'),
        chatAutoExecute: document.getElementById('chatAutoExecute'),
        chatSpinner: document.getElementById('chatSpinner'),
        chatStatusBadge: document.getElementById('chatStatusBadge'),
        timelineFeed: document.getElementById('timelineFeed'),
        btnClearTimeline: document.getElementById('btnClearTimeline'),
    };

    const appEl = document.getElementById('docReviewApp');
    const sessionToken = appEl?.dataset?.sessionToken || '';
    let socket = null;
    let joinedRoomId = null;
    let timelineEvents = [];
    let chatMessages = [];

    async function init() {
        bindEvents();
        initSocket();
        setChatStatus('idle');
        await Promise.all([loadDocuments(), loadUploadDir(), loadWelcome()]);
    }

    function bindEvents() {
        dom.btnRefreshDocs?.addEventListener('click', loadDocuments);
        dom.btnRegisterDocument?.addEventListener('click', registerDocument);
        dom.btnRunPhase1?.addEventListener('click', runPhase1);
        dom.btnDownloadMarkdown?.addEventListener('click', downloadMarkdown);
        dom.btnOpenSource?.addEventListener('click', openSourceFile);
        dom.documentSearch?.addEventListener('input', debounce(handleSearch, 150));
        dom.btnRefreshWelcome?.addEventListener('click', loadWelcome);
        dom.btnOpenUploadDir?.addEventListener('click', () => window.open('/api/doc_review/upload_dir/files', '_blank'));
        dom.btnSendChat?.addEventListener('click', sendChatMessage);
        dom.chatInput?.addEventListener('keydown', handleChatKeydown);
        dom.btnClearTimeline?.addEventListener('click', clearTimeline);
    }

    async function loadDocuments() {
        toggleButton(dom.btnRefreshDocs, true);
        try {
            const { documents: records = [] } = await api.get('/api/doc_review/documents');
            documents = records;
            filteredDocuments = records;
            dom.docCount.textContent = records.length;
            renderDocumentList();
            if (!selectedDocumentId && records.length) {
                selectDocument(records[0].file_id);
            }
        } catch (err) {
            showToast('Unable to load documents', err.message || err);
        } finally {
            toggleButton(dom.btnRefreshDocs, false);
        }
    }

    async function loadUploadDir() {
        try {
            const data = await api.get('/api/doc_review/upload_dir/files');
            if (data.upload_dir) {
                dom.uploadDirDisplay.textContent = data.upload_dir;
            }
        } catch (err) {
            console.warn('Upload dir fetch failed', err);
        }
    }

    async function loadWelcome() {
        dom.welcomeMessage.innerHTML = '<div class="skeleton skeleton-paragraph"></div><div class="skeleton skeleton-paragraph"></div>';
        try {
            const data = await api.get('/api/doc_review/welcome');
            dom.welcomeMessage.innerHTML = `<div class="welcome-copy">${markdownToHtml(data.content)}</div>`;
        } catch (err) {
            dom.welcomeMessage.innerHTML = `<p class="text-muted">Unable to load welcome message: ${err.message || err}</p>`;
        }
    }

    function renderDocumentList() {
        if (!filteredDocuments.length) {
            dom.documentList.innerHTML = '<div class="empty-state">No matching documents.</div>';
            return;
        }

        dom.documentList.innerHTML = filteredDocuments
            .map(
                (doc) => `
                    <div class="doc-list-item ${doc.file_id === selectedDocumentId ? 'active' : ''}" data-id="${doc.file_id}">
                        <h4>${doc.file_id}</h4>
                        <div class="doc-meta">
                            <span>${summarizeDocStats(doc)}</span>
                            <span class="status-pill ${doc.status}">${doc.status || 'unknown'}</span>
                        </div>
                    </div>
                `,
            )
            .join('');

        dom.documentList.querySelectorAll('.doc-list-item').forEach((el) => {
            el.addEventListener('click', () => selectDocument(el.dataset.id));
        });
    }

    async function selectDocument(fileId) {
        if (!fileId) return;
        selectedDocumentId = fileId;
        renderDocumentList();
        await loadDocumentDetails(fileId);
        joinDocRoom(fileId);
    }

    async function loadDocumentDetails(fileId) {
        resetPhase1Cards();
        setPrimaryButtonsState(true);
        try {
            const { document: record } = await api.get(`/api/doc_review/documents/${fileId}`);
            if (!record) return;
            currentDocumentRecord = record;
            setPrimaryButtonsState(false, record);
            renderDocumentHeader(record);
            renderMetrics(record);
            renderSummaryCard(record);
            renderTocCard(record);
            renderTemplateFitness(record);
            renderSectionStrategy(record);
            renderActivityFeed(record);
        } catch (err) {
            showToast('Unable to load document details', err.message || err);
        }
    }

    function renderDocumentHeader(record) {
        dom.selectedDocTitle.textContent = record.file_id || 'Unnamed document';
        const statuses = record.state?.phase_statuses || {
            phase1: record.state?.phase1_status,
            phase2: record.state?.phase2_status,
            phase3: record.state?.phase3_status,
        };
        dom.statusChips.forEach((chip) => {
            const phase = chip.dataset.phase;
            const status = statuses?.[phase] || 'pending';
            chip.textContent = `${phase.replace('phase', 'Phase ')} • ${status}`;
            chip.classList.remove('success', 'running');
            if (status === 'success') chip.classList.add('success');
            if (status === 'running') chip.classList.add('running');
        });
    }

    function renderMetrics(record) {
        const stats = record.state?.phase1?.stats || {};
        dom.metricPages.textContent = stats.page_count ?? record.page_count ?? '—';
        dom.metricWords.textContent = stats.word_count ?? record.word_count ?? '—';
        dom.metricHeadings.textContent = stats.heading_count ?? '—';
        dom.metricImages.textContent = stats.image_count ?? '—';
    }

    function renderSummaryCard(record) {
        const summary = record.state?.phase1?.doc_summary;
        if (!summary) {
            dom.summaryConfidence.textContent = 'awaiting run';
            dom.summaryConfidence.classList.remove('high');
            dom.execSummaryBody.innerHTML = '<p class="text-muted mb-0">Run Phase 1 to generate the executive summary.</p>';
            return;
        }

        dom.summaryConfidence.textContent = `${summary.confidence || 'unknown'} confidence`;
        dom.summaryConfidence.classList.toggle('high', summary.confidence === 'high');
        dom.execSummaryBody.innerHTML = `
            <p>${summary.summary}</p>
            <div class="summary-meta">
                <p><strong>Document type:</strong> ${summary.document_type || '—'}</p>
                <p><strong>Purpose:</strong> ${summary.purpose || '—'}</p>
                <p><strong>Audience:</strong> ${summary.audience || '—'}</p>
                <p><strong>Themes:</strong> ${(summary.themes || []).join(', ') || '—'}</p>
            </div>
        `;
    }

    function renderTocCard(record) {
        const toc = record.state?.phase1?.toc_review;
        if (!toc) {
            dom.tocScore.textContent = '—';
            dom.tocEntries.innerHTML = '<div class="empty-state">No TOC detected yet.</div>';
            dom.tocObservations.innerHTML = '';
            dom.tocGaps.innerHTML = '';
            return;
        }

        dom.tocScore.textContent = toc.structure_score || 'unknown';
        dom.tocEntries.innerHTML = (toc.entries || [])
            .map(
                (entry) => `
                    <div class="toc-entry">
                        <h4>${'#'.repeat(entry.level || 1)} ${entry.title}</h4>
                        <p class="text-muted mb-0">${entry.notes || ''}</p>
                    </div>
                `,
            )
            .join('');

        dom.tocObservations.innerHTML = renderListItems(toc.observations);
        dom.tocGaps.innerHTML = renderListItems(toc.gaps || [], 'No gaps detected');
    }

    function renderTemplateFitness(record) {
        const report = record.state?.phase1?.template_fitness_report;
        if (!report) {
            dom.templateAlignmentScore.textContent = '—';
            dom.templateCategoryList.innerHTML = '<div class="empty-state">Run Phase 1 to evaluate template coverage.</div>';
            return;
        }
        dom.templateAlignmentScore.textContent = report.overall_alignment || 'unknown';
        dom.templateCategoryList.innerHTML = (report.categories || [])
            .map(
                (cat) => `
                    <div class="fitness-item">
                        <h4>${cat.name}</h4>
                        <p class="mb-1"><strong>Coverage:</strong> ${cat.coverage}</p>
                        <p class="mb-1"><strong>Effort:</strong> ${cat.effort}</p>
                        <p class="mb-0 text-muted">${cat.gaps?.join('; ') || 'No gaps noted.'}</p>
                    </div>
                `,
            )
            .join('');
    }

    function renderSectionStrategy(record) {
        const strategy = record.state?.phase1?.section_strategy;
        if (!strategy) {
            dom.strategyVerdict.textContent = 'Awaiting analysis';
            dom.strategyVerdict.classList.remove('success');
            dom.strategyRationale.textContent = 'Run Phase 1 to receive actionable recommendations.';
            dom.strategyLevel.textContent = '—';
            dom.strategySections.textContent = '—';
            dom.strategyActions.innerHTML = '';
            return;
        }

        dom.strategyVerdict.textContent = strategy.verdict || 'unknown';
        dom.strategyVerdict.classList.toggle('success', strategy.verdict === 'ready');
        dom.strategyRationale.textContent = strategy.rationale || '';
        dom.strategyLevel.textContent = strategy.recommended_section_level || '—';
        dom.strategySections.textContent = strategy.estimated_sections ?? '—';
        dom.strategyActions.innerHTML = renderListItems(strategy.next_steps, 'No actions proposed');
    }

    function renderActivityFeed(record) {
        const logs = record.state?.logs || [];
        if (!logs.length) {
            dom.activityFeed.innerHTML = '<div class="empty-state">No activity yet.</div>';
            return;
        }
        dom.activityFeed.innerHTML = logs
            .slice(-10)
            .reverse()
            .map(
                (log) => `
                    <div class="activity-item">
                        <time>${formatTimestamp(log.timestamp || log.time || Date.now())}</time>
                        <p class="mb-0">${log.message || log.detail || ''}</p>
                    </div>
                `,
            )
            .join('');
    }

    function initSocket() {
        if (typeof io === 'undefined') {
            console.warn('Socket.IO not available');
            return;
        }
        socket = io();
        socket.on('connect', () => setChatStatus('connected'));
        socket.on('disconnect', () => setChatStatus('offline'));
        socket.on('doc_review:error', (payload) => {
            showToast('Stream error', payload?.error || 'Unknown', 'danger');
        });
        ['status', 'log', 'node_started', 'node_completed', 'vfs_file_updated', 'agent_plan_generated'].forEach(
            (eventType) => {
                socket.on(`doc_review:${eventType}`, (payload) => handleDocEvent(eventType, payload));
            },
        );
    }

    function handleDocEvent(eventType, payload = {}) {
        if (payload.file_id && joinedRoomId && payload.file_id !== joinedRoomId) {
            return;
        }
        addTimelineEvent(eventType, payload);
    }

    function joinDocRoom(fileId) {
        if (!socket || !sessionToken || !fileId) {
            return;
        }
        if (joinedRoomId) {
            socket.emit('doc_review:leave', { file_id: joinedRoomId });
        }
        socket.emit('doc_review:join', { token: sessionToken, file_id: fileId });
        joinedRoomId = fileId;
        setChatStatus('connected');
        timelineEvents = [];
        renderTimeline();
    }

    function addTimelineEvent(eventType, payload) {
        const timestamp = payload.timestamp || new Date().toISOString();
        timelineEvents.unshift({ eventType, payload, timestamp });
        if (timelineEvents.length > 40) {
            timelineEvents.pop();
        }
        renderTimeline();
    }

    function renderTimeline() {
        if (!timelineEvents.length) {
            dom.timelineFeed.innerHTML = '<div class="empty-state">Live workflow events will appear here.</div>';
            return;
        }
        dom.timelineFeed.innerHTML = timelineEvents
            .map((event) => {
                const label = formatTimelineLabel(event.eventType, event.payload);
                return `
                    <div class="timeline-event">
                        <time>${formatTimestamp(event.timestamp)}</time>
                        <strong>${label.title}</strong>
                        <p class="mb-0 text-muted">${label.detail}</p>
                    </div>
                `;
            })
            .join('');
    }

    function formatTimelineLabel(eventType, payload = {}) {
        switch (eventType) {
            case 'node_started':
                return {
                    title: `Node started: ${payload.node || payload.label || 'unknown'}`,
                    detail: payload.summary || '',
                };
            case 'node_completed':
                return {
                    title: `Node completed: ${payload.node || payload.label || 'unknown'} (${payload.status || 'success'})`,
                    detail: payload.summary || '',
                };
            case 'agent_plan_generated':
                return {
                    title: 'Agent plan generated',
                    detail: payload.summary || 'Plan ready for confirmation',
                };
            case 'vfs_file_updated':
                return {
                    title: 'VFS updated',
                    detail: payload.path || '',
                };
            default:
                return {
                    title: `Event: ${eventType}`,
                    detail: payload.message || '',
                };
        }
    }

    function clearTimeline() {
        timelineEvents = [];
        renderTimeline();
    }

    function handleChatKeydown(event) {
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
            event.preventDefault();
            sendChatMessage();
        }
    }

    async function sendChatMessage() {
        if (!selectedDocumentId) {
            showToast('Select a document before chatting', '', 'warning');
            return;
        }
        const message = dom.chatInput.value.trim();
        if (!message) {
            return;
        }
        appendChatEntry('user', message);
        dom.chatInput.value = '';
        toggleChatSending(true);
        setChatStatus('thinking');
        try {
            const payload = {
                file_id: selectedDocumentId,
                message,
                auto_execute: dom.chatAutoExecute?.checked ?? true,
            };
            const response = await api.post('/api/doc_review/handle_user_message', payload);
            appendChatEntry('agent', formatPlanResponse(response.result));
        } catch (err) {
            const errorMsg = err.message || err;
            appendChatEntry('agent', `<p class="text-danger mb-0">Error: ${errorMsg}</p>`);
            showToast('Agent command failed', errorMsg, 'danger');
        } finally {
            toggleChatSending(false);
            setChatStatus('idle');
        }
    }

    function appendChatEntry(role, content) {
        chatMessages.push({
            role,
            content,
            timestamp: new Date().toISOString(),
        });
        if (chatMessages.length > 30) {
            chatMessages.shift();
        }
        renderChatHistory();
    }

    function renderChatHistory() {
        if (!chatMessages.length) {
            dom.chatHistory.innerHTML = '<div class="empty-state">Select a document and send a command to the agent.</div>';
            return;
        }
        dom.chatHistory.innerHTML = chatMessages
            .map(
                (entry) => `
                <div class="chat-entry ${entry.role}">
                    <h6>${entry.role === 'user' ? 'You' : 'Agent'} • ${formatTimestamp(entry.timestamp)}</h6>
                    <div>${entry.content}</div>
                </div>
            `,
            )
            .join('');
        dom.chatHistory.scrollTop = dom.chatHistory.scrollHeight;
    }

    function formatPlanResponse(result = {}) {
        if (!result || !result.plan) {
            return '<p class="mb-0">No plan returned.</p>';
        }
        const plan = result.plan || {};
        const steps = plan.plan_steps || [];
        const exec = result.execution_results || {};
        const status = result.status || 'pending';
        const summary = plan.summary ? `<p class="mb-2">${plan.summary}</p>` : '';
        const stepList = steps.length
            ? `<ol class="mb-2">${steps.map((step) => `<li><code>${step.tool}</code> – ${step.reasoning || ''}</li>`).join('')}</ol>`
            : '';
        const execErrors = (exec.errors || []).length
            ? `<p class="text-warning mb-0">Errors: ${exec.errors.join(', ')}</p>`
            : '';
        return `
            ${summary}
            <p class="mb-1"><strong>Status:</strong> ${status}</p>
            ${stepList || '<p class="text-muted mb-2">No plan steps returned.</p>'}
            ${execErrors}
        `;
    }

    function toggleChatSending(isLoading) {
        if (!dom.btnSendChat) return;
        dom.btnSendChat.disabled = isLoading;
        dom.chatSpinner?.classList.toggle('d-none', !isLoading);
    }

    function setChatStatus(state) {
        if (!dom.chatStatusBadge) return;
        dom.chatStatusBadge.textContent = state;
        dom.chatStatusBadge.classList.remove('bg-light', 'bg-success', 'bg-danger', 'bg-warning');
        if (state === 'connected') {
            dom.chatStatusBadge.classList.add('bg-success');
        } else if (state === 'thinking') {
            dom.chatStatusBadge.classList.add('bg-warning');
        } else if (state === 'offline') {
            dom.chatStatusBadge.classList.add('bg-danger');
        } else {
            dom.chatStatusBadge.classList.add('bg-light');
        }
    }

    async function registerDocument() {
        const sourcePath = dom.inputSourcePath.value.trim();
        if (!sourcePath) {
            showToast('Please provide an absolute path', '', 'warning');
            return;
        }
        toggleButton(dom.btnRegisterDocument, true);
        try {
            await api.post('/api/doc_review/documents', { source_path: sourcePath });
            dom.inputSourcePath.value = '';
            await loadDocuments();
            showToast('Document registered', sourcePath, 'success');
        } catch (err) {
            showToast('Unable to register document', err.message || err, 'danger');
        } finally {
            toggleButton(dom.btnRegisterDocument, false);
        }
    }

    async function runPhase1() {
        if (!selectedDocumentId) return;
        toggleButton(dom.btnRunPhase1, true);
        dom.phase1Spinner?.classList.remove('d-none');
        try {
            await api.post(`/api/doc_review/documents/${selectedDocumentId}/run_phase1`);
            await loadDocuments();
            await loadDocumentDetails(selectedDocumentId);
            showToast('Phase 1 completed', selectedDocumentId, 'success');
        } catch (err) {
            showToast('Phase 1 failed', err.message || err, 'danger');
        } finally {
            toggleButton(dom.btnRunPhase1, false);
            dom.phase1Spinner?.classList.add('d-none');
        }
    }

    function setPrimaryButtonsState(disabled, record) {
        dom.btnRunPhase1.disabled = disabled || !record;
        dom.btnDownloadMarkdown.disabled = !record?.state?.structure?.raw_text;
        dom.btnOpenSource.disabled = !record?.source_path;
    }

    function downloadMarkdown() {
        if (!currentDocumentRecord?.state?.structure?.raw_text) {
            showToast('Markdown not available yet', '', 'warning');
            return;
        }
        const blob = new Blob([currentDocumentRecord.state.structure.raw_text], { type: 'text/markdown;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${currentDocumentRecord.file_id || 'document'}.md`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function openSourceFile() {
        if (!currentDocumentRecord?.source_path) {
            showToast('Source path unavailable', '', 'warning');
            return;
        }
        navigator.clipboard.writeText(currentDocumentRecord.source_path).then(
            () => showToast('Source path copied', currentDocumentRecord.source_path, 'success'),
            () => showToast('Unable to copy path', currentDocumentRecord.source_path, 'warning'),
        );
    }

    function resetPhase1Cards() {
        dom.execSummaryBody.innerHTML = '<p class="text-muted mb-0">Loading...</p>';
        dom.tocEntries.innerHTML = '<div class="skeleton skeleton-paragraph"></div>';
        dom.templateCategoryList.innerHTML = '<div class="skeleton skeleton-paragraph"></div>';
        dom.strategyActions.innerHTML = '';
    }

    function handleSearch() {
        const term = (dom.documentSearch.value || '').toLowerCase();
        filteredDocuments = documents.filter((doc) => doc.file_id.toLowerCase().includes(term));
        renderDocumentList();
    }

    function renderListItems(list, emptyMessage = 'No items') {
        if (!list || !list.length) {
            return `<li>${emptyMessage}</li>`;
        }
        return list.map((item) => `<li>${item}</li>`).join('');
    }

    function summarizeDocStats(doc) {
        const stats = doc.state?.phase1?.stats || {};
        const pages = stats.page_count ?? '—';
        const words = stats.word_count ?? '—';
        return `${pages} pages • ${words} words`;
    }

    function toggleButton(button, isLoading) {
        if (!button) return;
        button.disabled = isLoading;
        button.classList.toggle('disabled', isLoading);
    }

    function handleResponse(response) {
        if (!response.ok) {
            return response.json().catch(() => ({})).then((data) => {
                const message = data.error || data.message || response.statusText;
                throw new Error(message);
            });
        }
        return response.json();
    }

    function showToast(title, message, type = 'info') {
        const event = new CustomEvent('docreview-toast', { detail: { title, message, type } });
        window.dispatchEvent(event);
        console.log(`[DocReview] ${title}: ${message}`);
    }

    function markdownToHtml(markdown = '') {
        return markdown
            .replace(/^### (.*$)/gim, '<h4>$1</h4>')
            .replace(/^## (.*$)/gim, '<h3>$1</h3>')
            .replace(/^# (.*$)/gim, '<h2>$1</h2>')
            .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/gim, '<em>$1</em>')
            .replace(/\n$/gim, '<br />');
    }

    function formatTimestamp(ts) {
        if (!ts) return '';
        const date = new Date(ts);
        return date.toLocaleString();
    }

    function debounce(fn, delay) {
        let timeout;
        return (...args) => {
            clearTimeout(timeout);
            timeout = setTimeout(() => fn.apply(null, args), delay);
        };
    }

    document.addEventListener('DOMContentLoaded', init);
})();

