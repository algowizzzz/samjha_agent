/* AI Bulk Doc Analysis - isolated feature JS (no core changes) */

(function () {
  const appEl = document.getElementById('bulkDocApp');
  if (!appEl) return;

  // Bootstrap init (tooltips etc) comes from global main.js, but we still need to
  // re-init tooltips for elements rendered/updated later.
  function initTooltips() {
    if (typeof bootstrap === 'undefined') return;
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.forEach((el) => {
      try {
        bootstrap.Tooltip.getOrCreateInstance(el);
      } catch (_) {
        // ignore
      }
    });
  }

  function toast(type, title, message) {
    // Reuse global helper if present (defined in web/static/js/main.js)
    if (typeof showNotification === 'function') {
      const mapped = type === 'error' ? 'danger' : type;
      showNotification(title, message, mapped);
      return;
    }
    // Fallback
    console.log(`[${type}] ${title}: ${message}`);
  }

  const Status = {
    QUEUED: 'QUEUED',
    PROCESSING: 'PROCESSING',
    CONVERTED: 'CONVERTED',
    ERROR: 'ERROR',
  };

  const state = {
    sessionId: appEl.dataset.sessionId || null,
    sessions: /** @type {Array<{session_id:string, name?:string, created_at:string, document_count:number}>} */ ([]),
    currentSessionId: appEl.dataset.sessionId || null,
    workflows: /** @type {Array<{workflow_id:string, name:string, description:string, latest_version_id:string, accepted_input_types:Array<string>, export_type:string, step_count:number}>} */ ([]),
    selectedWorkflow: null,
    selectedWorkflowVersionId: null,
    docs: /** @type {Array<{localId:string, filename:string, status:string, errorMessage?:string, doc_id?:string}>} */ ([]),
    selectedExportFormat: null,
    currentRun: null,
    recentRuns: /** @type {Array<{run_id:string, workflow_name:string, created_at:string, status:string, document_count:number}>} */ ([]),
    wizardStep: 1, // 1=upload, 2=format, 3=execute (selection happens on initial page)
    workflowWizardVisible: false, // Whether the wizard is visible (hidden initially)
    run: {
      runId: null,
      rows: /** @type {Array<{localId:string, filename:string, stepLabel:string, status:string, inputTokens?:number, outputTokens?:number, canDownload:boolean}>} */ ([]),
    },
    polling: {
      docsActive: false,
      runActive: false,
      intervalId: null,
      runIntervalId: null,
    },
    ui: {
      deleteTargetLocalId: null,
    },
  };

  // -------- DOM refs --------
  const refs = {
    // Session elements (automation-friendly buttons instead of dropdown)
    sessionButtons: document.getElementById('bulkDocSessionButtons'),
    newSessionBtn: document.getElementById('bulkDocNewSessionBtn'),
    currentSessionName: document.getElementById('bulkDocCurrentSessionName'),
    
    // Workflow elements (automation-friendly cards instead of dropdown)
    workflowCards: document.getElementById('workflowCards'),
    workflowPreview: document.getElementById('workflowPreview'),
    
    // Upload elements
    uploadInput: document.getElementById('bulkDocUploadInput'),
    uploadArea: document.getElementById('uploadArea'),
    acceptedTypesHint: document.getElementById('acceptedTypesHint'),
    uploadedFilesList: document.getElementById('uploadedFilesList'),
    testFilesGrid: document.getElementById('testFilesGrid'),
    
    // Output format (automation-friendly cards)
    outputFormatCards: document.getElementById('outputFormatCards'),
    
    // Processing elements
    startProcessingBtn: document.getElementById('startProcessingBtn'),
    pauseProcessingBtn: document.getElementById('pauseProcessingBtn'),
    pauseProcessingBtn: document.getElementById('pauseProcessingBtn'),
    docsWrap: document.getElementById('bulkDocDocsTableWrap'),
    docsTbody: document.getElementById('bulkDocDocsTbody'),
    runWrap: document.getElementById('bulkDocRunTableWrap'),
    runTbody: document.getElementById('bulkDocRunTbody'),
    runProgressSection: document.getElementById('runProgressSection'),
    runSummary: document.getElementById('bulkDocRunSummary'),
    summaryCounts: document.getElementById('bulkDocSummaryCounts'),
    downloadAllBtn: document.getElementById('bulkDocDownloadAllBtn'),
    recentRunsList: document.getElementById('recentRunsList'),

    drawerEl: document.getElementById('bulkDocDocDrawer'),
    drawerFilename: document.getElementById('bulkDocDrawerFilename'),
    drawerStatus: document.getElementById('bulkDocDrawerStatus'),
    drawerDetails: document.getElementById('bulkDocDrawerDetails'),

    deleteModalEl: document.getElementById('bulkDocDeleteModal'),
    deleteTargetEl: document.getElementById('bulkDocDeleteTarget'),
    confirmDeleteBtn: document.getElementById('bulkDocConfirmDeleteBtn'),
  };

  // -------- helpers --------
  const mockMode = new URLSearchParams(window.location.search).get('mock') === '1';

  function makeLocalId(prefix) {
    return `${prefix}_${Math.random().toString(36).slice(2, 10)}_${Date.now()}`;
  }

  function statusBadge(status) {
    const map = {
      [Status.QUEUED]: 'badge-queued',
      [Status.PROCESSING]: 'info',
      [Status.CONVERTED]: 'success',
      [Status.ERROR]: 'danger',
      SUCCESS: 'success',
      COMPLETE: 'badge-complete',
      RUNNING: 'info',
      PAUSED: 'warning',
    };
    const cls = map[status] || 'secondary';
    // Use custom classes for COMPLETE and QUEUED, Bootstrap classes for others
    if (cls === 'badge-complete' || cls === 'badge-queued') {
      return `<span class="badge ${cls}">${status}</span>`;
    }
    return `<span class="badge bg-${cls}">${status}</span>`;
  }

  // Wizard navigation
  // Step numbers: 1=upload, 2=format, 3=execute
  function goToStep(stepNumber, skipValidation = false) {
    // Validate current step before allowing next step (skip if explicitly requested)
    if (!skipValidation && stepNumber > state.wizardStep) {
      if (state.wizardStep === 1 && state.docs.length === 0) {
        toast('error', 'Validation', 'Please upload at least one document');
        return;
      }
      if (state.wizardStep === 2 && !state.selectedExportFormat) {
        toast('error', 'Validation', 'Please select an output format');
        return;
      }
    }
    
    state.wizardStep = stepNumber;
    
    // Update active step styling (all steps visible, just mark active one)
    const wizard = document.getElementById('workflowWizard');
    if (wizard) {
      wizard.querySelectorAll('.wizard-step').forEach((step) => {
        const stepNum = parseInt(step.getAttribute('data-step') || '0');
        if (stepNum === stepNumber) {
          step.classList.add('active');
        } else {
          step.classList.remove('active');
        }
        // Always show all steps (no display: none)
      });
    }
    
    // Scroll to step
    const stepEl = wizard ? wizard.querySelector(`[data-step="${stepNumber}"]`) : null;
    if (stepEl) {
      stepEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  function updateStartButton() {
    const hasWorkflow = !!state.selectedWorkflow;
    const hasDocs = state.docs.length > 0;
    const allConverted = hasDocs && state.docs.every((d) => d.status === Status.CONVERTED);
    const hasFormat = !!state.selectedExportFormat;
    
    const canStart = hasWorkflow && allConverted && hasFormat;
    if (refs.startProcessingBtn) {
      refs.startProcessingBtn.disabled = !canStart;
      
      let reason = 'Ready to process';
      if (!hasWorkflow) reason = 'Select an agentic workflow template';
      else if (!hasDocs) reason = 'Upload documents';
      else if (!allConverted) reason = 'Wait for all documents to be converted';
      else if (!hasFormat) reason = 'Select an output format';
      
      refs.startProcessingBtn.setAttribute('title', reason);
    }
  }

  // -------- rendering --------
  function renderDocs() {
    const hasDocs = state.docs.length > 0;
    if (refs.docsWrap) {
      refs.docsWrap.classList.toggle('d-none', !hasDocs);
    }

    if (!hasDocs) {
      if (refs.docsTbody) refs.docsTbody.innerHTML = '';
      updateStartButton();
      return;
    }

    refs.docsTbody.innerHTML = state.docs.map((d) => {
      // Allow deletion for ERROR, QUEUED, or CONVERTED documents
      const canDelete = d.status === Status.ERROR || d.status === Status.QUEUED || d.status === Status.CONVERTED;
      const deleteBtn = canDelete
        ? `<button class="btn btn-sm btn-outline-danger" data-action="delete" data-id="${d.localId}" title="Delete document"><i class="bi bi-trash"></i></button>`
        : '';
      const statusDetail = (d.status === Status.ERROR && d.errorMessage)
        ? `<div class="bulk-doc-hint text-danger text-truncate" style="max-width: 220px;">${escapeHtml(d.errorMessage)}</div>`
        : '';
      // Format timestamp if available
      let timeInfo = '';
      if (d.created_at) {
        try {
          const date = new Date(d.created_at);
          const now = new Date();
          const diffMs = now - date;
          const diffMins = Math.floor(diffMs / 60000);
          const diffHours = Math.floor(diffMs / 3600000);
          const diffDays = Math.floor(diffMs / 86400000);
          
          if (diffMins < 1) timeInfo = 'just now';
          else if (diffMins < 60) timeInfo = `${diffMins}m ago`;
          else if (diffHours < 24) timeInfo = `${diffHours}h ago`;
          else if (diffDays < 7) timeInfo = `${diffDays}d ago`;
          else timeInfo = date.toLocaleDateString();
        } catch (e) {
          // Ignore date parsing errors
        }
      }
      const timeLabel = timeInfo ? `<div class="bulk-doc-hint" style="font-size: 11px; margin-top: 2px;">${escapeHtml(timeInfo)}</div>` : '';
      
      return `
        <tr role="button" tabindex="0" data-action="open" data-id="${d.localId}">
          <td class="text-truncate" style="max-width: 260px;">
            ${escapeHtml(d.filename)}
            ${timeLabel}
          </td>
          <td>${statusBadge(d.status)}${statusDetail}</td>
          <td class="text-end">${deleteBtn}</td>
        </tr>
      `;
    }).join('');

    updateStartButton();
  }

  function renderChains() {
    const hasChains = state.chains.length > 0;
    if (refs.chainSelectHint) {
      refs.chainSelectHint.classList.toggle('d-none', !!state.selectedChainVersionId);
    }
    refs.chainsWrap.classList.toggle('d-none', !hasChains);

    if (!hasChains) {
      refs.chainsList.innerHTML = '';
      refs.chainDetail.innerHTML = '<div class="bulk-doc-hint">—</div>';
      updateStartButton();
      return;
    }

    refs.chainsList.innerHTML = state.chains.map((c) => {
      const selected = c.chain_version_id === state.selectedChainVersionId;
      const disabled = !c.valid;
      const cls = [
        'bulk-doc-card',
        selected ? 'bulk-doc-card--selected' : '',
        disabled ? 'bulk-doc-card--disabled' : '',
      ].filter(Boolean).join(' ');

      const invalidLabel = disabled ? `<div class="bulk-doc-hint text-danger mt-1"><i class="bi bi-exclamation-triangle"></i> Incomplete chain</div>` : '';
      const chainId = c.chain_id || c.chain_version_id;

      return `
        <div class="${cls} bulk-doc-chain-card" role="button" tabindex="0"
             data-action="select-chain" data-id="${escapeHtml(c.chain_version_id)}"
             aria-disabled="${disabled ? 'true' : 'false'}"
             style="position: relative;">
          <button type="button" class="btn btn-sm btn-link bulk-doc-chain-edit-btn" 
                  data-action="edit-chain" data-chain-id="${escapeHtml(chainId)}"
                  data-chain-version-id="${escapeHtml(c.chain_version_id)}"
                  title="Edit chain"
                  onclick="event.stopPropagation()">
            <i class="bi bi-pencil"></i>
          </button>
          <div style="font-weight:600;">${escapeHtml(c.name)}</div>
          <div class="bulk-doc-hint">${escapeHtml(c.description || '')}</div>
          <div class="bulk-doc-hint mt-1">Steps: ${c.step_count}</div>
          ${invalidLabel}
        </div>
      `;
    }).join('');

    const selected = state.chains.find((c) => c.chain_version_id === state.selectedChainVersionId);
    if (selected) {
      refs.chainDetail.innerHTML = renderChainDetail(selected);
    } else {
      refs.chainDetail.innerHTML = '<div class="bulk-doc-hint">Select a chain to view steps.</div>';
    }

    updateStartButton();
  }

  function renderChainDetail(chain) {
    const steps = Array.isArray(chain.steps) ? chain.steps : [];
    if (steps.length === 0) {
      return `<div class="bulk-doc-hint">No step details available yet.</div>`;
    }
    const rows = steps.map((s) => {
      const idx = s.index ?? s.step_index ?? '';
      const inputs = Array.isArray(s.required_inputs) ? s.required_inputs.join(', ') : (s.inputs || '');
      const desc = s.description || '';
      return `
        <div class="bulk-doc-card" style="margin-bottom:8px;">
          <div style="font-weight:600;">Step ${escapeHtml(String(idx))}</div>
          <div class="bulk-doc-hint">Inputs: ${escapeHtml(String(inputs))}</div>
          <div class="bulk-doc-hint">${escapeHtml(String(desc))}</div>
        </div>
      `;
    }).join('');
    return rows;
  }

  function renderRun() {
    const hasRows = state.run.rows.length > 0;
    
    // Show/hide run progress section
    if (refs.runProgressSection) {
      refs.runProgressSection.style.display = hasRows ? 'block' : 'none';
    }
    
    if (refs.runWrap) {
      refs.runWrap.classList.toggle('d-none', !hasRows);
    }
    if (refs.runSummary) {
      refs.runSummary.classList.toggle('d-none', !hasRows);
    }

    // Check if run is complete to hide pause button
    const allComplete = hasRows && state.run.rows.every(r => r.status === 'SUCCESS' || r.status === 'ERROR' || r.status === 'COMPLETE');
    if (refs.pauseProcessingBtn) {
      refs.pauseProcessingBtn.style.display = (hasRows && !allComplete) ? 'inline-block' : 'none';
    }

    if (!hasRows) {
      if (refs.runTbody) refs.runTbody.innerHTML = '';
      if (refs.downloadAllBtn) refs.downloadAllBtn.disabled = true;
      if (refs.summaryCounts) refs.summaryCounts.textContent = '0 processed';
      return;
    }

    // Get output format for display
    const outputFormat = state.selectedWorkflow?.export_format || state.selectedExportFormat;
    const formatLabelMap = {
      'MD': 'Markdown',
      'JSON': 'JSON',
      'CSV': 'CSV',
      'XLSX': 'Excel',
      'DOCX': 'Word Doc',
      'PDF': 'PDF'
    };
    const outputFormatLabel = outputFormat ? (formatLabelMap[outputFormat] || outputFormat) : '';
    
    refs.runTbody.innerHTML = state.run.rows.map((r) => {
      const tokensText = (typeof r.inputTokens === 'number' && typeof r.outputTokens === 'number')
        ? `${r.inputTokens} / ${r.outputTokens}`
        : '—';
      const downloadBtn = r.canDownload
        ? `<button class="btn btn-sm btn-outline-primary" data-action="download-one" data-id="${r.localId}"><i class="bi bi-download"></i></button>`
        : `<button class="btn btn-sm btn-outline-secondary" disabled title="Available on SUCCESS"><i class="bi bi-download"></i></button>`;
      
      // Show filename with output format indicator
      const filenameDisplay = outputFormatLabel 
        ? `${escapeHtml(r.filename)} <span class="text-muted small">→ ${escapeHtml(outputFormatLabel)}</span>`
        : escapeHtml(r.filename);
      
      return `
        <tr>
          <td class="text-truncate" style="max-width: 260px;">${filenameDisplay}</td>
          <td>${escapeHtml(r.stepLabel || '—')}</td>
          <td>${statusBadge(r.status)}</td>
          <td class="bulk-doc-mono">${escapeHtml(tokensText)}</td>
          <td class="text-end">${downloadBtn}</td>
        </tr>
      `;
    }).join('');

    const total = state.run.rows.length;
    const success = state.run.rows.filter((r) => r.status === 'SUCCESS').length;
    const failed = state.run.rows.filter((r) => r.status === 'ERROR').length;
    refs.summaryCounts.textContent = `${total} processed · ${success} success · ${failed} failed`;
    refs.downloadAllBtn.disabled = success === 0;
  }

  function openDocDrawer(doc) {
    refs.drawerFilename.textContent = doc.filename;
    refs.drawerStatus.innerHTML = statusBadge(doc.status);
    refs.drawerDetails.textContent = doc.errorMessage ? doc.errorMessage : '—';
    if (typeof bootstrap !== 'undefined' && refs.drawerEl) {
      bootstrap.Offcanvas.getOrCreateInstance(refs.drawerEl).show();
    }
  }

  function openDeleteModal(doc) {
    state.ui.deleteTargetLocalId = doc.localId;
    refs.deleteTargetEl.textContent = doc.filename;
    if (typeof bootstrap !== 'undefined' && refs.deleteModalEl) {
      bootstrap.Modal.getOrCreateInstance(refs.deleteModalEl).show();
    }
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function formatRunDate(isoString) {
    if (!isoString) return '—';
    const date = new Date(isoString);
    if (isNaN(date.getTime())) return '—';
    
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    const runDate = new Date(date.getFullYear(), date.getMonth(), date.getDate());
    
    // Format time in 12-hour format
    const hours = date.getHours();
    const minutes = date.getMinutes();
    const ampm = hours >= 12 ? 'PM' : 'AM';
    const displayHours = hours % 12 || 12;
    const displayMinutes = minutes.toString().padStart(2, '0');
    const timeStr = `${displayHours}:${displayMinutes} ${ampm}`;
    
    if (runDate.getTime() === today.getTime()) {
      return `Today ${timeStr}`;
    } else if (runDate.getTime() === yesterday.getTime()) {
      return `Yesterday ${timeStr}`;
    } else {
      // Format full date: "Jan 2, 2026 1:51 PM"
      const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
      const month = months[date.getMonth()];
      const day = date.getDate();
      const year = date.getFullYear();
      return `${month} ${day}, ${year} ${timeStr}`;
    }
  }

  // -------- API layer (intentionally minimal; backend will define exact endpoints) --------
  const api = {
    async listChains() {
      const res = await fetch('/api/bulk-doc-analysis/chains', { credentials: 'same-origin' });
      if (!res.ok) throw new Error(`chains failed: ${res.status}`);
      return await res.json();
    },
    async listDocs() {
      const res = await fetch('/api/bulk-doc-analysis/documents', { credentials: 'same-origin' });
      if (!res.ok) throw new Error(`docs failed: ${res.status}`);
      return await res.json();
    },
    async listSessions() {
      const res = await fetch('/api/bulk-doc-analysis/sessions', { credentials: 'same-origin' });
      if (!res.ok) throw new Error(`sessions failed: ${res.status}`);
      return await res.json();
    },
    async createSession(name) {
      const res = await fetch('/api/bulk-doc-analysis/sessions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
        credentials: 'same-origin',
      });
      if (!res.ok) throw new Error(`create session failed: ${res.status}`);
      return await res.json();
    },
    async selectSession(sessionId) {
      const res = await fetch(`/api/bulk-doc-analysis/sessions/${encodeURIComponent(sessionId)}/select`, {
        method: 'POST',
        credentials: 'same-origin',
      });
      if (!res.ok) throw new Error(`select session failed: ${res.status}`);
      return await res.json();
    },
    async deleteDoc(docId) {
      const res = await fetch(`/api/bulk-doc-analysis/documents/${encodeURIComponent(docId)}`, { method: 'DELETE', credentials: 'same-origin' });
      if (!res.ok) throw new Error(`delete failed: ${res.status}`);
      return await res.json();
    },
    async createRun(requestBody) {
      // Accepts: { workflow_version_id?, chain_version_id?, session_id? }
      // Backend resolves chain_version_id from workflow_version_id if needed
      const res = await fetch('/api/bulk-doc-analysis/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
        credentials: 'same-origin',
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: `Failed: ${res.status}` }));
        throw new Error(err.error || `create run failed: ${res.status}`);
      }
      return await res.json();
    },
    async getRunProgress(runId) {
      const res = await fetch(`/api/bulk-doc-analysis/runs/${encodeURIComponent(runId)}/progress`, { credentials: 'same-origin' });
      if (!res.ok) throw new Error(`run progress failed: ${res.status}`);
      return await res.json();
    },
    async downloadDocOutput(runId, docId) {
      const res = await fetch(`/api/bulk-doc-analysis/runs/${encodeURIComponent(runId)}/download/${encodeURIComponent(docId)}`, { credentials: 'same-origin' });
      if (!res.ok) throw new Error(`download failed: ${res.status}`);
      return res.blob();
    },
    async downloadAllOutputs(runId) {
      const res = await fetch(`/api/bulk-doc-analysis/runs/${encodeURIComponent(runId)}/download-all`, { credentials: 'same-origin' });
      if (!res.ok) throw new Error(`bulk download failed: ${res.status}`);
      return res.blob();
    },
    async listRuns() {
      const res = await fetch('/api/bulk-doc-analysis/runs', { credentials: 'same-origin' });
      if (!res.ok) throw new Error(`list runs failed: ${res.status}`);
      return await res.json();
    },
    async deleteRun(runId) {
      const res = await fetch(`/api/bulk-doc-analysis/runs/${encodeURIComponent(runId)}`, {
        method: 'DELETE',
        credentials: 'same-origin'
      });
      if (!res.ok) throw new Error(`delete run failed: ${res.status}`);
      return await res.json();
    },
    async createChain(name, description, steps) {
      const res = await fetch('/api/bulk-doc-analysis/chains', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, description, steps }),
        credentials: 'same-origin',
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: `Failed: ${res.status}` }));
        throw new Error(err.error || `create chain failed: ${res.status}`);
      }
      return await res.json();
    },
    async updateChain(chainId, name, description, steps) {
      const res = await fetch(`/api/bulk-doc-analysis/chains/${encodeURIComponent(chainId)}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, description, steps }),
        credentials: 'same-origin',
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: `Failed: ${res.status}` }));
        throw new Error(err.error || `update chain failed: ${res.status}`);
      }
      return await res.json();
    },
  };

  // -------- chain editor functions --------
  function openChainEditor(mode, chainId, chainVersionId) {
    state.chainEditor.mode = mode;
    
    if (mode === 'edit' && (chainId || chainVersionId)) {
      // Find chain by chain_id or chain_version_id
      const chain = chainId 
        ? state.chains.find((c) => c.chain_id === chainId)
        : state.chains.find((c) => c.chain_version_id === chainVersionId);
      
      if (chain) {
        state.chainEditor.chainId = chain.chain_id || chain.chain_version_id;
        state.chainEditor.name = chain.name || '';
        state.chainEditor.description = chain.description || '';
        state.chainEditor.steps = (chain.steps || []).map((s) => ({
          index: s.index || 0,
          required_inputs: Array.isArray(s.required_inputs) ? [...s.required_inputs] : [],
          prompt: s.prompt || '',
          description: s.description || '',
          model_config: s.model_config || {
            model: 'claude-3-haiku-20240307',
            max_tokens: 4096,
            temperature: 0.2
          }
        }));
      } else {
        toast('error', 'Chain not found', 'Could not find chain to edit.');
        return;
      }
    } else {
      // Create mode - reset state
      state.chainEditor.chainId = null;
      state.chainEditor.name = '';
      state.chainEditor.description = '';
      state.chainEditor.steps = [];
    }

    renderChainEditor();
    if (typeof bootstrap !== 'undefined' && refs.chainEditorDrawer) {
      const drawer = bootstrap.Offcanvas.getOrCreateInstance(refs.chainEditorDrawer);
      drawer.show();
    }
  }

  function closeChainEditor() {
    state.chainEditor.mode = null;
    state.chainEditor.chainId = null;
    state.chainEditor.name = '';
    state.chainEditor.description = '';
    state.chainEditor.steps = [];
    if (typeof bootstrap !== 'undefined' && refs.chainEditorDrawer) {
      bootstrap.Offcanvas.getInstance(refs.chainEditorDrawer)?.hide();
    }
  }

  function addStep() {
    const nextIndex = state.chainEditor.steps.length + 1;
    state.chainEditor.steps.push({
      index: nextIndex,
      required_inputs: ['R0'], // Default to R0
      prompt: '',
      description: '',
      model_config: {
        model: 'claude-3-haiku-20240307',
        max_tokens: 4096,
        temperature: 0.2
      }
    });
    renderChainEditor();
  }

  function removeStep(stepIndex) {
    state.chainEditor.steps = state.chainEditor.steps.filter((s) => s.index !== stepIndex);
    // Re-index steps sequentially
    state.chainEditor.steps.forEach((s, idx) => {
      s.index = idx + 1;
    });
    renderChainEditor();
  }

  function updateStep(stepIndex, field, value) {
    const step = state.chainEditor.steps.find((s) => s.index === stepIndex);
    if (step) {
      step[field] = value;
      renderChainEditor();
    }
  }

  function updateStepInputs(stepIndex, inputs) {
    const step = state.chainEditor.steps.find((s) => s.index === stepIndex);
    if (step) {
      step.required_inputs = Array.isArray(inputs) ? inputs : [];
      renderChainEditor();
    }
  }

  function updateStepModelConfig(stepIndex, field, value) {
    const step = state.chainEditor.steps.find((s) => s.index === stepIndex);
    if (step) {
      if (!step.model_config) {
        step.model_config = { model: 'claude-3-haiku-20240307', max_tokens: 4096, temperature: 0.2 };
      }
      step.model_config[field] = value;
    }
  }

  function getAvailableInputs(stepIndex) {
    // R0 is always available
    const available = ['R0'];
    // R1..RN are available if previous steps exist
    for (let i = 1; i < stepIndex; i++) {
      available.push(`R${i}`);
    }
    return available;
  }

  function renderChainEditor() {
    if (!refs.chainEditorDrawerLabel || !refs.chainEditorName || !refs.chainEditorDescription) return;

    // Update drawer title
    const title = state.chainEditor.mode === 'edit' 
      ? `Edit Chain: ${escapeHtml(state.chainEditor.name || '')}`
      : 'Create New Chain';
    refs.chainEditorDrawerLabel.textContent = title;

    // Update save button text
    if (refs.chainEditorSaveBtn) {
      refs.chainEditorSaveBtn.textContent = state.chainEditor.mode === 'edit' ? 'Save Changes' : 'Create Chain';
    }

    // Populate form fields
    refs.chainEditorName.value = state.chainEditor.name;
    refs.chainEditorDescription.value = state.chainEditor.description;

    // Render steps
    if (refs.chainEditorStepsContainer) {
      refs.chainEditorStepsContainer.innerHTML = state.chainEditor.steps.map((step) => renderStepCard(step)).join('');

      // Attach event handlers for step cards
      state.chainEditor.steps.forEach((step) => {
        const stepEl = refs.chainEditorStepsContainer.querySelector(`[data-step-index="${step.index}"]`);
        if (!stepEl) return;

        // Prompt textarea - update state directly to avoid re-render (prevents cursor loss)
        const promptTextarea = stepEl.querySelector(`[data-field="prompt"]`);
        if (promptTextarea) {
          const stepIndex = step.index; // Capture for closure
          promptTextarea.addEventListener('input', (e) => {
            // Update state directly without re-rendering (this prevents cursor loss)
            const stepObj = state.chainEditor.steps.find(s => s.index === stepIndex);
            if (stepObj) {
              stepObj.prompt = e.target.value;
            }
          });
        }

        // Description input - update state directly to avoid re-render
        const descInput = stepEl.querySelector(`[data-field="description"]`);
        if (descInput) {
          const stepIndex = step.index; // Capture for closure
          descInput.addEventListener('input', (e) => {
            // Update state directly without re-rendering
            const stepObj = state.chainEditor.steps.find(s => s.index === stepIndex);
            if (stepObj) {
              stepObj.description = e.target.value;
            }
          });
        }

        // Input checkboxes
        const inputCheckboxes = stepEl.querySelectorAll(`[data-input]`);
        inputCheckboxes.forEach((cb) => {
          cb.addEventListener('change', () => {
            const checked = Array.from(stepEl.querySelectorAll(`[data-input]:checked`)).map((c) => c.getAttribute('data-input'));
            updateStepInputs(step.index, checked);
          });
        });

        // Remove button
        const removeBtn = stepEl.querySelector(`[data-action="remove-step"]`);
        if (removeBtn) {
          removeBtn.addEventListener('click', () => {
            removeStep(step.index);
          });
        }

        // Model config fields
        const modelSelect = stepEl.querySelector(`[data-field="model"]`);
        if (modelSelect) {
          modelSelect.addEventListener('change', (e) => {
            updateStepModelConfig(step.index, 'model', e.target.value);
          });
        }

        const maxTokensInput = stepEl.querySelector(`[data-field="max_tokens"]`);
        if (maxTokensInput) {
          maxTokensInput.addEventListener('input', (e) => {
            updateStepModelConfig(step.index, 'max_tokens', parseInt(e.target.value) || 4096);
          });
        }

        const temperatureSlider = stepEl.querySelector(`[data-field="temperature"]`);
        if (temperatureSlider) {
          temperatureSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            updateStepModelConfig(step.index, 'temperature', value);
            // Update display
            const displayEl = document.getElementById(`temp_${step.index}_value`);
            if (displayEl) displayEl.textContent = value.toFixed(1);
          });
        }
      });
    }

    // Clear errors
    if (refs.chainEditorNameError) refs.chainEditorNameError.textContent = '';
    if (refs.chainEditorStepsError) refs.chainEditorStepsError.textContent = '';
  }

  function renderStepCard(step) {
    const availableInputs = getAvailableInputs(step.index);
    const inputsHtml = availableInputs.map((input) => {
      const checked = step.required_inputs.includes(input) ? 'checked' : '';
      return `
        <div class="form-check form-check-inline">
          <input class="form-check-input" type="checkbox" data-input="${input}" id="step${step.index}_input_${input}" ${checked}>
          <label class="form-check-label" for="step${step.index}_input_${input}">${escapeHtml(input)}</label>
        </div>
      `;
    }).join('');

    const modelConfig = step.model_config || { model: 'claude-3-haiku-20240307', max_tokens: 4096, temperature: 0.2 };
    const models = [
      { value: 'claude-3-haiku-20240307', label: 'Haiku' },
      { value: 'claude-3-5-sonnet-20241022', label: 'Sonnet' },
      { value: 'claude-3-opus-20240229', label: 'Opus' }
    ];
    const modelSelectHtml = models.map(m => 
      `<option value="${m.value}" ${modelConfig.model === m.value ? 'selected' : ''}>${m.label}</option>`
    ).join('');

    return `
      <div class="bulk-doc-chain-step-card" data-step-index="${step.index}">
        <div class="d-flex justify-content-between align-items-center mb-2">
          <div class="bulk-doc-label">Step ${step.index}</div>
          <button type="button" class="btn btn-sm btn-link text-danger" data-action="remove-step" title="Remove step">
            <i class="bi bi-trash"></i>
          </button>
        </div>
        <div class="mb-2">
          <label class="form-label bulk-doc-hint" style="font-size: 12px;">Required Inputs</label>
          <div class="d-flex flex-wrap gap-2">
            ${inputsHtml}
          </div>
        </div>
        <div class="row g-2 mb-2">
          <div class="col-md-4">
            <label class="form-label bulk-doc-hint" style="font-size: 12px;">Model</label>
            <select class="form-select form-select-sm" data-field="model" data-subfield="model">
              ${modelSelectHtml}
            </select>
          </div>
          <div class="col-md-4">
            <label class="form-label bulk-doc-hint" style="font-size: 12px;">Max Tokens</label>
            <input type="number" class="form-control form-control-sm" 
                   data-field="max_tokens" data-subfield="max_tokens"
                   value="${modelConfig.max_tokens || 4096}" min="1" max="8192" step="1">
          </div>
          <div class="col-md-4">
            <label class="form-label bulk-doc-hint" style="font-size: 12px;">
              Temperature: <span id="temp_${step.index}_value">${(modelConfig.temperature || 0.2).toFixed(1)}</span>
            </label>
            <input type="range" class="form-range" 
                   data-field="temperature" data-subfield="temperature"
                   value="${modelConfig.temperature || 0.2}" min="0" max="1" step="0.1">
          </div>
        </div>
        <div class="mb-2">
          <label class="form-label bulk-doc-label" style="font-size: 12px;">
            Prompt <span class="text-danger">*</span>
          </label>
          <textarea class="form-control form-control-sm" rows="4" 
                    data-field="prompt" 
                    placeholder="Enter the prompt for this step..."
                    required>${escapeHtml(step.prompt || '')}</textarea>
        </div>
        <div class="mb-2">
          <label class="form-label bulk-doc-hint" style="font-size: 12px;">Description (optional)</label>
          <input type="text" class="form-control form-control-sm" 
                 data-field="description" 
                 placeholder="Brief description of this step"
                 value="${escapeHtml(step.description || '')}">
        </div>
      </div>
    `;
  }

  function validateChainForm() {
    let isValid = true;
    let errors = [];

    // Validate name
    if (!state.chainEditor.name || state.chainEditor.name.trim() === '') {
      isValid = false;
      errors.push({ field: 'name', message: 'Chain name is required' });
    }

    // Validate at least one step
    if (state.chainEditor.steps.length === 0) {
      isValid = false;
      errors.push({ field: 'steps', message: 'At least one step is required' });
    }

    // Validate each step
    state.chainEditor.steps.forEach((step) => {
      if (!step.prompt || step.prompt.trim() === '') {
        isValid = false;
        errors.push({ field: `step${step.index}`, message: `Step ${step.index}: Prompt is required` });
      }
      if (!step.required_inputs || step.required_inputs.length === 0) {
        isValid = false;
        errors.push({ field: `step${step.index}`, message: `Step ${step.index}: At least one input is required` });
      }
      // Validate inputs are valid (can't use R2 if step 2 doesn't exist)
      const availableInputs = getAvailableInputs(step.index);
      const invalidInputs = step.required_inputs.filter((input) => !availableInputs.includes(input));
      if (invalidInputs.length > 0) {
        isValid = false;
        errors.push({ field: `step${step.index}`, message: `Step ${step.index}: Invalid inputs: ${invalidInputs.join(', ')}` });
      }
    });

    // Display errors
    if (refs.chainEditorNameError) {
      const nameError = errors.find((e) => e.field === 'name');
      refs.chainEditorNameError.textContent = nameError ? nameError.message : '';
    }
    if (refs.chainEditorStepsError) {
      const stepErrors = errors.filter((e) => e.field.startsWith('step'));
      refs.chainEditorStepsError.textContent = stepErrors.length > 0 
        ? stepErrors.map((e) => e.message).join('; ')
        : errors.find((e) => e.field === 'steps')?.message || '';
    }

    return isValid;
  }

  async function saveChain() {
    if (!validateChainForm()) {
      toast('error', 'Validation failed', 'Please fix the errors before saving.');
      return;
    }

    try {
      // Prepare steps with sequential indices
      const steps = state.chainEditor.steps.map((step, idx) => ({
        index: idx + 1,
        required_inputs: step.required_inputs,
        prompt: step.prompt.trim(),
        description: step.description ? step.description.trim() : '',
        model_config: step.model_config || {
          model: 'claude-3-haiku-20240307',
          max_tokens: 4096,
          temperature: 0.2
        }
      }));

      let result;
      if (state.chainEditor.mode === 'edit' && state.chainEditor.chainId) {
        result = await api.updateChain(
          state.chainEditor.chainId,
          state.chainEditor.name.trim(),
          state.chainEditor.description.trim(),
          steps
        );
        toast('success', 'Chain updated', 'Chain has been updated successfully.');
      } else {
        result = await api.createChain(
          state.chainEditor.name.trim(),
          state.chainEditor.description.trim(),
          steps
        );
        toast('success', 'Chain created', 'Chain has been created successfully.');
      }

      // Refresh chains list
      await refreshChains();

      // If this was the created/updated chain, select it
      if (result && result.chain) {
        const chainVersionId = result.chain.chain_version_id;
        if (chainVersionId) {
          state.selectedChainVersionId = chainVersionId;
          renderChains();
        }
      }

      closeChainEditor();
    } catch (err) {
      console.error('Save chain error:', err);
      toast('error', 'Save failed', err.message || 'Failed to save chain. Please try again.');
    }
  }

  // -------- chain editor event handlers --------
  if (refs.createChainBtn) {
    refs.createChainBtn.addEventListener('click', () => {
      openChainEditor('create');
    });
  }

  if (refs.chainEditorCancelBtn) {
    refs.chainEditorCancelBtn.addEventListener('click', () => {
      closeChainEditor();
    });
  }

  if (refs.chainEditorSaveBtn) {
    refs.chainEditorSaveBtn.addEventListener('click', () => {
      saveChain();
    });
  }

  if (refs.chainEditorAddStepBtn) {
    refs.chainEditorAddStepBtn.addEventListener('click', () => {
      addStep();
    });
  }

  if (refs.chainEditorName) {
    refs.chainEditorName.addEventListener('input', (e) => {
      state.chainEditor.name = e.target.value;
    });
  }

  if (refs.chainEditorDescription) {
    refs.chainEditorDescription.addEventListener('input', (e) => {
      state.chainEditor.description = e.target.value;
    });
  }

  // Close drawer on Bootstrap hide event
  if (refs.chainEditorDrawer) {
    refs.chainEditorDrawer.addEventListener('hidden.bs.offcanvas', () => {
      closeChainEditor();
    });
  }

  // -------- session management --------
  // Note: Session buttons have click handlers set in renderSessionSelector()
  
  async function switchSession(sessionId) {
    if (!sessionId || sessionId === state.currentSessionId) return;
    
    try {
      await api.selectSession(sessionId);
      state.currentSessionId = sessionId;
      updateCurrentSessionDisplay();
      renderSessionSelector(); // Update button states
      // Refresh documents for new session
      await refreshDocs();
      toast('success', 'Session switched', 'Switched to selected session.');
    } catch (err) {
      console.error('Switch session error:', err);
      toast('error', 'Switch failed', err.message || 'Failed to switch session.');
      renderSessionSelector();
    }
  }

  if (refs.newSessionBtn) {
    refs.newSessionBtn.addEventListener('click', async () => {
      const name = prompt('Enter a name for the new session (optional):');
      if (name === null) return; // User cancelled

      try {
        const result = await api.createSession(name || null);
        if (result && result.session_id) {
          state.currentSessionId = result.session_id;
          // Refresh sessions list and update UI
          await refreshSessions();
          // Clear current documents (new session has none)
          state.docs = [];
          renderDocs();
          toast('success', 'Session created', 'New session created and selected.');
        }
      } catch (err) {
        console.error('Create session error:', err);
        toast('error', 'Create failed', err.message || 'Failed to create new session.');
      }
    });
  }

  // -------- events --------
  // File input change event - use unified upload handler
  if (refs.uploadInput) {
    refs.uploadInput.addEventListener('change', async (e) => {
      const files = Array.from(e.target.files || []);
      if (files.length === 0) return;
      
      // Use the unified upload handler (handles workflow validation, file type validation, etc.)
      await handleDocumentUpload(files);
      
      // Clear input
      refs.uploadInput.value = '';
    });
  }

  if (refs.docsTbody) {
    refs.docsTbody.addEventListener('click', (e) => {
      const btn = e.target.closest('button');
      const row = e.target.closest('tr');
      if (!row) return;

      const docId = (btn || row).getAttribute('data-id');
      const action = (btn || row).getAttribute('data-action');
      const doc = state.docs.find((d) => d.localId === docId);
      if (!doc) return;

      if (action === 'delete') {
        openDeleteModal(doc);
        e.stopPropagation();
        return;
      }
      if (action === 'open') {
        openDocDrawer(doc);
      }
    });

    refs.docsTbody.addEventListener('keydown', (e) => {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      const row = e.target.closest('tr[data-id]');
      if (!row) return;
      const docId = row.getAttribute('data-id');
      const doc = state.docs.find((d) => d.localId === docId);
      if (!doc) return;
      openDocDrawer(doc);
      e.preventDefault();
    });
  }

  if (refs.confirmDeleteBtn) {
    refs.confirmDeleteBtn.addEventListener('click', async () => {
      const id = state.ui.deleteTargetLocalId;
      if (!id) return;
      const doc = state.docs.find((d) => d.localId === id);
      if (!doc) return;

      // Use doc_id from backend if available, fallback to localId
      const docIdToDelete = doc.doc_id || doc.docId || id;

      // Temporarily stop polling to prevent refresh during delete
      const wasPolling = !!state.polling.intervalId;
      if (wasPolling) {
        stopPolling();
      }

      try {
        await api.deleteDoc(docIdToDelete);
        // Remove from UI state after successful deletion
        state.docs = state.docs.filter((d) => d.localId !== id);
        renderDocs();
        toast('success', 'Deleted', 'Document removed successfully.');
        
        // Restart polling after a brief delay to allow backend to update
        if (wasPolling) {
          window.setTimeout(() => {
            startPolling();
          }, 1000);
        }
      } catch (err) {
        // If backend deletion fails, show error and restart polling
        console.error('Delete failed:', err);
        toast('error', 'Delete failed', err.message || 'Failed to delete document. It may already be in use.');
        
        // Restart polling
        if (wasPolling) {
          window.setTimeout(() => {
            startPolling();
          }, 1000);
        }
      }

      if (typeof bootstrap !== 'undefined' && refs.deleteModalEl) {
        bootstrap.Modal.getOrCreateInstance(refs.deleteModalEl).hide();
      }
    });
  }

  if (refs.chainsList) {
    refs.chainsList.addEventListener('click', (e) => {
      // Check if edit button was clicked
      const editBtn = e.target.closest('[data-action="edit-chain"]');
      if (editBtn) {
        const chainId = editBtn.getAttribute('data-chain-id');
        const chainVersionId = editBtn.getAttribute('data-chain-version-id');
        openChainEditor('edit', chainId, chainVersionId);
        return;
      }

      const card = e.target.closest('[data-action="select-chain"]');
      if (!card) return;
      const chainId = card.getAttribute('data-id');
      const chain = state.chains.find((c) => c.chain_version_id === chainId);
      if (!chain || !chain.valid) return;
      state.selectedChainVersionId = chainId;
      renderChains();
    });

    refs.chainsList.addEventListener('keydown', (e) => {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      const card = e.target.closest('[data-action="select-chain"]');
      if (!card) return;
      const chainId = card.getAttribute('data-id');
      const chain = state.chains.find((c) => c.chain_version_id === chainId);
      if (!chain || !chain.valid) return;
      state.selectedChainVersionId = chainId;
      renderChains();
      e.preventDefault();
    });
  }

  if (refs.runBtn) {
    refs.runBtn.addEventListener('click', async () => {
      const gate = canRun();
      if (!gate.ok) return;

      const selected = state.chains.find((c) => c.chain_version_id === state.selectedChainVersionId);
      if (!selected) {
        toast('error', 'No chain selected', 'Please select a valid chain.');
        return;
      }

      try {
        // Create run via backend
        const runData = await api.createRun(state.sessionId, state.selectedChainVersionId);
        state.run.runId = runData.run.run_id;

        // Initial progress fetch
        await refreshRunProgress();

        // Start polling run progress
        startRunPolling();

        toast('info', 'Run started', 'Execution in progress. Progress will update automatically.');
      } catch (err) {
        console.error(err);
        toast('error', 'Run failed', err.message || 'Failed to start run. Check backend availability.');
      }
    });
  }

  if (refs.downloadAllBtn) {
    refs.downloadAllBtn.addEventListener('click', async () => {
      if (!state.run.runId) {
        toast('error', 'No run', 'No active run to download.');
        return;
      }

      const successRows = state.run.rows.filter((r) => r.canDownload);
      if (successRows.length === 0) {
        toast('info', 'Nothing to download', 'No successful outputs available.');
        return;
      }

      try {
        toast('info', 'Preparing download', 'Creating ZIP archive...');
        const blob = await api.downloadAllOutputs(state.run.runId);
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `run_${state.run.runId}_outputs.zip`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        toast('success', 'Download started', 'ZIP archive download initiated.');
      } catch (err) {
        console.error(err);
        toast('error', 'Download failed', err.message || 'Failed to download outputs.');
      }
      toast('success', 'Download complete', `${successRows.length} file(s) downloaded.`);
    });
  }

  if (refs.runTbody) {
    refs.runTbody.addEventListener('click', async (e) => {
      const btn = e.target.closest('button[data-action="download-one"]');
      if (!btn) return;
      const id = btn.getAttribute('data-id');
      const row = state.run.rows.find((r) => r.localId === id);
      if (!row || !row.canDownload || !state.run.runId) return;

      try {
        // For CSV workflows, include task_id as query parameter
        let downloadUrl = `/api/bulk-doc-analysis/runs/${encodeURIComponent(state.run.runId)}/download/${encodeURIComponent(row.docId || row.localId)}`;
        if (row.taskId) {
          downloadUrl += `?task_id=${encodeURIComponent(row.taskId)}`;
        }
        const res = await fetch(downloadUrl, { credentials: 'same-origin' });
        if (!res.ok) throw new Error(`download failed: ${res.status}`);
        
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        
        // Get filename from Content-Disposition header, fallback to default
        const contentDisposition = res.headers.get('Content-Disposition');
        let downloadFilename = `${row.filename}_output.md`; // Default fallback
        if (contentDisposition) {
          const filenameMatch = contentDisposition.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/);
          if (filenameMatch && filenameMatch[1]) {
            downloadFilename = filenameMatch[1].replace(/['"]/g, '');
          }
        }
        
        a.download = downloadFilename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        toast('success', 'Downloaded', `Downloaded ${downloadFilename}`);
      } catch (err) {
        toast('error', 'Download failed', err.message || 'Failed to download output.');
      }
    });
  }

  // Event handler for recent runs table
  if (refs.recentRunsList) {
    refs.recentRunsList.addEventListener('click', async (e) => {
      const btn = e.target.closest('button[data-action]');
      if (!btn) return;
      const action = btn.getAttribute('data-action');
      const runId = btn.getAttribute('data-run-id');
      
      if (action === 'view-run' && runId) {
        await openRunDetails(runId);
      } else if (action === 'delete-run' && runId) {
        if (confirm(`Are you sure you want to delete this run? This action cannot be undone.`)) {
          await deleteRun(runId);
        }
      }
    });
  }

  // -------- polling (docs + chains + run) --------
  function startPolling() {
    if (state.polling.intervalId) return;
    state.polling.intervalId = window.setInterval(async () => {
      await refreshDocs();
      await refreshChains();
    }, 2000);
  }

  function startRunPolling() {
    if (state.polling.runIntervalId) return;
    state.polling.runActive = true;
    state.polling.runIntervalId = window.setInterval(async () => {
      await refreshRunProgress();
      // Stop polling if run is complete
      const isComplete = state.run.rows.every((r) => r.status === 'SUCCESS' || r.status === 'ERROR');
      if (isComplete && state.polling.runIntervalId) {
        stopRunPolling();
      }
    }, 2000);
  }

  function stopRunPolling() {
    if (state.polling.runIntervalId) {
      window.clearInterval(state.polling.runIntervalId);
      state.polling.runIntervalId = null;
      state.polling.runActive = false;
    }
  }

  async function refreshRunProgress() {
    if (!state.run.runId) return;
    try {
      const progress = await api.getRunProgress(state.run.runId);
      if (!progress || !progress.rows) return;

      // Set export format from run if available (for display purposes)
      if (progress.export_format) {
        state.selectedExportFormat = progress.export_format;
      }

      // Map backend progress rows to UI state
      state.run.rows = progress.rows.map((r) => ({
        localId: makeLocalId('runrow'), // Could preserve mapping if needed
        docId: r.doc_id,
        taskId: r.task_id || null,  // Include task_id for CSV workflows
        filename: r.filename,
        stepLabel: r.step_label || 'R0',
        status: r.status || 'QUEUED',
        inputTokens: r.input_tokens || null,
        outputTokens: r.output_tokens || null,
        canDownload: r.can_download || false,
      }));

      // Map filenames back to existing doc localIds for consistency
      const filenameMap = new Map(state.docs.map((d) => [d.filename, d.localId]));
      state.run.rows.forEach((row) => {
        const docLocalId = filenameMap.get(row.filename);
        if (docLocalId) row.localId = docLocalId;
      });

      renderRun();
    } catch (err) {
      console.warn('Run progress refresh failed:', err);
      if (String(err).includes('404')) {
        stopRunPolling();
      }
    }
  }

  function renderRecentRuns() {
    if (!refs.recentRunsList) return;
    
    if (!state.recentRuns || state.recentRuns.length === 0) {
      refs.recentRunsList.innerHTML = '<tr><td colspan="6" class="text-muted text-center">No previous runs</td></tr>';
      return;
    }
    
    const rows = state.recentRuns.map((run) => {
      const workflowName = escapeHtml(run.workflow_name || 'Unknown Workflow');
      const dateTime = formatRunDate(run.created_at);
      const statusBadgeHtml = statusBadge(run.status);
      
      // Documents/Tasks count
      let countText = '';
      if (run.is_csv_workflow) {
        countText = `${run.task_count || 0} task${run.task_count !== 1 ? 's' : ''}`;
      } else {
        countText = `${run.document_count || 1} document${run.document_count !== 1 ? 's' : ''}`;
      }
      
      return `
        <tr>
          <td>${workflowName}</td>
          <td>${dateTime}</td>
          <td>${statusBadgeHtml}</td>
          <td>${escapeHtml(countText)}</td>
          <td class="text-end">
            <button class="btn btn-sm btn-outline-primary" data-action="view-run" data-run-id="${escapeHtml(run.run_id)}">
              View
            </button>
          </td>
          <td class="text-end">
            <button class="btn btn-sm btn-outline-danger" data-action="delete-run" data-run-id="${escapeHtml(run.run_id)}">
              <i class="bi bi-trash"></i> Delete
            </button>
          </td>
        </tr>
      `;
    }).join('');
    
    refs.recentRunsList.innerHTML = rows;
  }

  async function openRunDetails(runId) {
    if (!runId) return;
    
    try {
      // Ensure workflows are loaded first
      if (!state.workflows || state.workflows.length === 0) {
        await loadWorkflows();
      }
      
      // Get run details to find workflow_version_id
      const runs = await api.listRuns();
      const run = runs.runs?.find(r => r.run_id === runId);
      
      if (!run) {
        toast('error', 'Run not found', 'Could not find the selected run');
        return;
      }
      
      // Find and select the workflow associated with this run
      if (run.workflow_version_id) {
        let workflow = state.workflows.find(w => 
          w.workflow_version_id === run.workflow_version_id || 
          w.latest_version_id === run.workflow_version_id
        );
        
        // If workflow not found, try reloading workflows
        if (!workflow && state.workflows.length > 0) {
          await loadWorkflows();
          workflow = state.workflows.find(w => 
            w.workflow_version_id === run.workflow_version_id || 
            w.latest_version_id === run.workflow_version_id
          );
        }
        
        if (workflow) {
          // Select the workflow first
          await onWorkflowSelect(workflow.workflow_id);
          
          // Wait a bit for workflow selection to complete
          await new Promise(resolve => setTimeout(resolve, 300));
        } else {
          toast('warning', 'Workflow not found', 'The workflow for this run may have been deleted. Showing run details anyway.');
        }
      } else {
        toast('warning', 'No workflow version', 'This run does not have an associated workflow version.');
      }
      
      // Set the run ID
      state.run.runId = runId;
      
      // Navigate to step 3 (run progress) - skip validation since we're opening an existing run
      goToStep(3, true);
      
      // Show the run progress section
      if (refs.runProgressSection) {
        refs.runProgressSection.style.display = 'block';
      }
      
      // Load run progress (this will populate the table)
      await refreshRunProgress();
      
      // Check status and start polling if needed
      const progress = await api.getRunProgress(runId);
      if (progress) {
        const status = progress.status;
        if (status === 'QUEUED' || status === 'RUNNING') {
          startRunPolling();
          toast('info', 'Resuming run', 'Polling for updates...');
        } else {
          toast('success', 'Run loaded', 'Run details displayed');
        }
      } else {
        toast('success', 'Run loaded', 'Run details displayed');
      }
    } catch (err) {
      console.error('Failed to open run details:', err);
      toast('error', 'Failed to load run', err.message || 'Could not load run details');
    }
  }

  async function loadRecentRuns() {
    if (!refs.recentRunsList) return;
    
    try {
      const data = await api.listRuns();
      state.recentRuns = data.runs || [];
      renderRecentRuns();
    } catch (err) {
      console.error('Failed to load recent runs:', err);
      if (refs.recentRunsList) {
        refs.recentRunsList.innerHTML = '<tr><td colspan="6" class="text-muted text-danger text-center">Failed to load recent runs</td></tr>';
      }
    }
  }

  async function deleteRun(runId) {
    try {
      await api.deleteRun(runId);
      toast('success', 'Run Deleted', 'The run has been deleted successfully.');
      // Reload recent runs
      await loadRecentRuns();
    } catch (err) {
      console.error('Failed to delete run:', err);
      toast('error', 'Delete Failed', err.message || 'Failed to delete run');
    }
  }

  async function refreshDocs() {
    try {
      const data = await api.listDocs();
      if (data && Array.isArray(data.documents)) {
        // Use doc_id for matching to avoid duplicates when same filename exists
        const existingMap = new Map(state.docs.map((d) => {
          if (d.doc_id) return [d.doc_id, d];
          // Fallback to filename for old entries without doc_id
          return [d.filename, d];
        }));
        
        // Merge backend docs with existing state, preserving localId
        // Backend already returns sorted by newest first, so we preserve that order
        state.docs = data.documents.map((d) => {
          const existing = existingMap.get(d.doc_id) || existingMap.get(d.original_filename);
          return {
            localId: existing ? existing.localId : makeLocalId('doc'),
            doc_id: d.doc_id, // Store backend doc_id for deletion
            filename: d.original_filename,
            status: d.status,
            errorMessage: d.error_message || null,
            created_at: d.created_at || null, // Store timestamp for display
          };
        });
        renderDocs();
      }
    } catch (err) {
      // stop noisy polling if endpoint not present
      if (String(err).includes('docs failed: 404')) {
        stopPolling();
      }
    }
  }

  async function refreshSessions() {
    try {
      const data = await api.listSessions();
      if (data && Array.isArray(data.sessions)) {
        state.sessions = data.sessions;
        // Set current session if not set or if it doesn't exist
        if (!state.currentSessionId && state.sessions.length > 0) {
          state.currentSessionId = state.sessions[0].session_id;
        }
        renderSessionSelector();
        updateCurrentSessionDisplay();
      }
    } catch (err) {
      console.warn('Sessions refresh failed:', err);
      // Show error in buttons area
      if (refs.sessionButtons) {
        refs.sessionButtons.innerHTML = '<span class="text-danger">Error loading sessions</span>';
      }
    }
  }

  function renderSessionSelector() {
    // Render as clickable buttons (automation-friendly)
    if (!refs.sessionButtons) return;
    
    if (state.sessions.length === 0) {
      refs.sessionButtons.innerHTML = '<span class="text-muted">No sessions</span>';
      return;
    }
    
    // Show up to 5 recent sessions as buttons
    const recentSessions = state.sessions.slice(0, 5);
    const buttons = recentSessions.map((s) => {
      const shortId = s.session_id.split('_').pop()?.slice(0, 8) || 'new';
      const docCount = s.document_count || 0;
      const isActive = s.session_id === state.currentSessionId ? 'active' : '';
      return `
        <button type="button" 
                class="session-btn ${isActive}" 
                data-session-id="${escapeHtml(s.session_id)}"
                title="Session ${shortId} - ${docCount} document(s)">
          ${escapeHtml(shortId)} <span class="doc-count">(${docCount})</span>
        </button>
      `;
    }).join('');
    
    refs.sessionButtons.innerHTML = buttons;
    
    // Add click handlers to session buttons
    refs.sessionButtons.querySelectorAll('.session-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const sessionId = btn.dataset.sessionId;
        if (sessionId && sessionId !== state.currentSessionId) {
          switchSession(sessionId);
        }
      });
    });
  }

  function updateCurrentSessionDisplay() {
    if (!refs.currentSessionName) return;
    
    const current = state.sessions.find(s => s.session_id === state.currentSessionId);
    if (current) {
      const name = current.name || `Session ${current.session_id.split('_').pop()?.slice(0, 8) || 'new'}`;
      refs.currentSessionName.textContent = name;
    } else {
      refs.currentSessionName.textContent = state.currentSessionId || 'None';
    }
  }

  async function refreshChains() {
    try {
      if (refs.chainsSkeleton) refs.chainsSkeleton.classList.remove('d-none');
      const data = await api.listChains();
      if (data && Array.isArray(data.chains)) {
        state.chains = data.chains.map((c) => ({
          chain_id: c.chain_id || null,
          chain_version_id: c.chain_version_id || c.id || makeLocalId('chain'),
          name: c.name || 'Unnamed chain',
          description: c.description || '',
          step_count: Number(c.step_count || 0),
          valid: c.valid !== false,
          steps: c.steps || [],
        }));
        renderChains();
      }
    } catch (err) {
      console.warn('Chains refresh failed:', err);
      // Don't reset chains if we already have some - only on initial load
      if (state.chains.length === 0) {
        // Provide a minimal placeholder so Panel 2 is buildable now without backend.
        state.chains = [
          {
            chain_id: 'placeholder-valid',
            chain_version_id: 'placeholder-valid-v1',
            name: 'Example 4-step chain',
            description: 'Placeholder (backend chains not wired yet)',
            step_count: 4,
            valid: true,
            steps: [
              { index: 1, required_inputs: ['R0'], description: 'Step 1' },
              { index: 2, required_inputs: ['R0', 'R1'], description: 'Step 2' },
              { index: 3, required_inputs: ['R0', 'R1'], description: 'Step 3' },
              { index: 4, required_inputs: ['R1', 'R2', 'R3'], description: 'Step 4' },
            ],
          },
          {
            chain_id: 'placeholder-invalid',
            chain_version_id: 'placeholder-invalid-v1',
            name: 'Incomplete chain',
            description: 'Disabled until complete',
            step_count: 2,
            valid: false,
            steps: [],
          },
        ];
        renderChains();
      }
    } finally {
      if (refs.chainsSkeleton) refs.chainsSkeleton.classList.add('d-none');
    }
  }

  function simulateRun(stepCount) {
    // Deterministic: any filename containing 'fail' triggers an ERROR on step 1.
    const shouldFail = (filename) => String(filename || '').toLowerCase().includes('fail');

    let tick = 0;
    const interval = window.setInterval(() => {
      tick += 1;
      state.run.rows = state.run.rows.map((r) => {
        if (r.status === 'SUCCESS' || r.status === 'ERROR') return r;

        const stepIndex = Math.min(tick, Math.max(stepCount, 1));
        const stepLabel = `R${stepIndex}`;

        if (shouldFail(r.filename) && stepIndex >= 1) {
          return {
            ...r,
            stepLabel,
            status: 'ERROR',
            inputTokens: 800,
            outputTokens: 0,
            canDownload: false,
          };
        }

        if (stepIndex >= stepCount) {
          return {
            ...r,
            stepLabel,
            status: 'SUCCESS',
            inputTokens: 1200 + stepCount * 100,
            outputTokens: 900 + stepCount * 80,
            canDownload: true,
          };
        }

        return {
          ...r,
          stepLabel,
          status: 'RUNNING',
          inputTokens: 300 + stepIndex * 200,
          outputTokens: 200 + stepIndex * 150,
          canDownload: false,
        };
      });

      renderRun();

      const done = state.run.rows.every((r) => r.status === 'SUCCESS' || r.status === 'ERROR');
      if (done) {
        window.clearInterval(interval);
        const failed = state.run.rows.some((r) => r.status === 'ERROR');
        toast(failed ? 'error' : 'success', 'Run complete', failed ? 'Partial failure (mock).' : 'All documents succeeded (mock).');
      }
    }, 900);
  }

  function stopPolling() {
    if (!state.polling.intervalId) return;
    window.clearInterval(state.polling.intervalId);
    state.polling.intervalId = null;
  }

  // -------- Workflow Functions --------
  async function loadWorkflows() {
    try {
      const response = await fetch('/api/bulk-doc-analysis/workflows', {
        credentials: 'same-origin'
      });
      if (!response.ok) throw new Error('Failed to load workflows');
      
      const data = await response.json();
      state.workflows = data.workflows || [];
      
      // Render workflow cards (automation-friendly - no dropdown)
      renderWorkflowCards();
    } catch (error) {
      console.error('Failed to load workflows:', error);
      toast('error', 'Error', 'Failed to load workflows: ' + error.message);
      if (refs.workflowCards) {
        refs.workflowCards.innerHTML = '<span class="text-danger">Error loading workflows</span>';
      }
    }
  }
  
  function renderWorkflowCards() {
    if (!refs.workflowCards) return;
    
    if (state.workflows.length === 0) {
      refs.workflowCards.innerHTML = '<span class="text-muted">No workflows available. Create one in the Admin panel.</span>';
      return;
    }
    
    // Format labels for export formats (consistent with renderOutputFormatOptions)
    const formatLabelMap = {
      'MD': 'Markdown',
      'JSON': 'JSON',
      'CSV': 'CSV',
      'XLSX': 'Excel',
      'DOCX': 'Word Doc',
      'PDF': 'PDF'
    };
    
    const cards = state.workflows.map(workflow => {
      const isSelected = state.selectedWorkflow?.workflow_id === workflow.workflow_id;
      const inputTypes = (workflow.accepted_input_types || []);
      const outputFormatRaw = workflow.export_format || workflow.export_type;
      const stepCount = workflow.step_count || 0;
      const stepTitles = workflow.step_titles || [];
      
      // Build input types badges
      const inputTypesHtml = inputTypes.map(t => 
        `<span class="workflow-card__tag input-type">${escapeHtml(t)}</span>`
      ).join('');
      
      // Build output type badge
      const outputTypeHtml = outputFormatRaw ? 
        `<span class="workflow-card__tag output-type">${escapeHtml(formatLabelMap[outputFormatRaw] || outputFormatRaw)}</span>` : '';
      
      // Build step titles list
      const stepTitlesHtml = stepTitles.length > 0 
        ? `<div class="workflow-card__steps">
             <div class="workflow-card__steps-label">
               <i class="bi bi-list-ol"></i> ${stepCount} step${stepCount !== 1 ? 's' : ''}:
             </div>
             <div class="workflow-card__steps-list">
               ${stepTitles.map((title, idx) => 
                 `<span class="workflow-card__step-title">${idx + 1}. ${escapeHtml(title)}</span>`
               ).join('')}
             </div>
           </div>`
        : stepCount > 0
        ? `<div class="workflow-card__steps">
             <div class="workflow-card__steps-label">
               <i class="bi bi-list-ol"></i> ${stepCount} step${stepCount !== 1 ? 's' : ''}
             </div>
           </div>`
        : '';
      
      return `
        <div class="workflow-card ${isSelected ? 'selected' : ''}" 
             data-workflow-id="${escapeHtml(workflow.workflow_id)}"
             role="button"
             tabindex="0">
          <div class="workflow-card__header">
            <div class="workflow-card__name">${escapeHtml(workflow.name)}</div>
          </div>
          <div class="workflow-card__body">
            <div class="workflow-card__desc">${escapeHtml(workflow.description || 'No description available')}</div>
            ${stepTitlesHtml}
            <div class="workflow-card__meta">
              <div class="workflow-card__meta-row">
                <span class="workflow-card__meta-label"><i class="bi bi-file-earmark-arrow-up"></i> Input:</span>
                <div class="workflow-card__tags">${inputTypesHtml || '<span class="text-muted">Any</span>'}</div>
              </div>
              <div class="workflow-card__meta-row">
                <span class="workflow-card__meta-label"><i class="bi bi-file-earmark-arrow-down"></i> Output:</span>
                <div class="workflow-card__tags">${outputTypeHtml || '<span class="text-muted">Markdown</span>'}</div>
              </div>
            </div>
          </div>
        </div>
      `;
    }).join('');
    
    refs.workflowCards.innerHTML = cards;
    
    // Add click handlers
    refs.workflowCards.querySelectorAll('.workflow-card').forEach(card => {
      card.addEventListener('click', () => {
        const workflowId = card.dataset.workflowId;
        onWorkflowSelect(workflowId);
      });
      // Also handle keyboard
      card.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          const workflowId = card.dataset.workflowId;
          onWorkflowSelect(workflowId);
        }
      });
    });
  }

  // Show/hide wizard and initial sections
  function showWorkflowWizard() {
    const wizard = document.getElementById('workflowWizard');
    const startSection = document.getElementById('startNewWorkflowSection');
    const recentRuns = document.querySelector('.recent-runs-section');
    
    if (wizard) wizard.style.display = 'block';
    if (startSection) startSection.style.display = 'none';
    if (recentRuns) recentRuns.style.display = 'none';
    
    state.workflowWizardVisible = true;
  }

  function hideWorkflowWizard() {
    const wizard = document.getElementById('workflowWizard');
    const startSection = document.getElementById('startNewWorkflowSection');
    const recentRuns = document.querySelector('.recent-runs-section');
    
    if (wizard) wizard.style.display = 'none';
    if (startSection) startSection.style.display = 'block';
    if (recentRuns) recentRuns.style.display = 'block';
    
    state.workflowWizardVisible = false;
    state.selectedWorkflow = null;
    state.selectedWorkflowVersionId = null;
    state.wizardStep = 1;
    
    // Clear card selection
    if (refs.workflowCards) {
      refs.workflowCards.querySelectorAll('.workflow-card').forEach(c => c.classList.remove('selected'));
    }
  }

  async function onWorkflowSelect(workflowId) {
    if (!workflowId) {
      hideWorkflowWizard();
      return;
    }
    
    const workflow = state.workflows.find(w => w.workflow_id === workflowId);
    if (!workflow) {
      toast('error', 'Error', 'Workflow not found');
      return;
    }
    
    state.selectedWorkflow = workflow;
    state.selectedWorkflowVersionId = workflow.workflow_version_id || workflow.latest_version_id;
    
    // Auto-create new session when starting workflow
    try {
      const result = await api.createSession(`Workflow: ${workflow.name}`);
      if (result && result.session_id) {
        state.currentSessionId = result.session_id;
        state.sessionId = result.session_id;
        await refreshSessions();
      }
    } catch (err) {
      console.warn('Failed to create session, using existing:', err);
    }
    
    // Update card selection state (automation-friendly visual feedback)
    if (refs.workflowCards) {
      refs.workflowCards.querySelectorAll('.workflow-card').forEach(card => {
        if (card.dataset.workflowId === workflowId) {
          card.classList.add('selected');
        } else {
          card.classList.remove('selected');
        }
      });
    }
    
    // Store chain_version_id if available (for efficient run creation)
    if (workflow.chain_version_id) {
      state.selectedWorkflow.chain_version_id = workflow.chain_version_id;
    }
    
    // Update workflow preview
    if (refs.workflowPreview) {
      const inputTypes = (workflow.accepted_input_types || []).join(', ');
      const outputType = workflow.export_type || 'N/A';
      const stepCount = workflow.step_count || 0;
      refs.workflowPreview.innerHTML = `
        <strong>Preview:</strong> Accepts ${inputTypes} → ${stepCount} step(s) → Outputs ${outputType}
      `;
      refs.workflowPreview.style.display = 'block';
    }
    
    // Update accepted types hint
    if (refs.acceptedTypesHint) {
      const inputTypes = (workflow.accepted_input_types || []).join(', ');
      refs.acceptedTypesHint.textContent = `Accepted file types: ${inputTypes}`;
    }
    
    // Update upload input accept attribute
    if (refs.uploadInput) {
      const extensions = [];
      workflow.accepted_input_types.forEach(type => {
        switch(type) {
          case 'PDF': extensions.push('.pdf'); break;
          case 'DOCX': extensions.push('.docx'); break;
          case 'TXT': extensions.push('.txt'); break;
          case 'MD': extensions.push('.md'); break;
          case 'CSV': extensions.push('.csv'); break;
        }
      });
      refs.uploadInput.setAttribute('accept', extensions.join(','));
    }
    
    // Render output format options
    renderOutputFormatOptions();
    
    // Reload test files filtered by accepted types
    await loadTestFiles();
    
    // Show wizard and hide initial sections
    showWorkflowWizard();
    
    // Go to step 1 (upload documents)
    goToStep(1);
  }

  function renderOutputFormatOptions() {
    // Render as clickable cards (automation-friendly - no radio buttons)
    if (!refs.outputFormatCards || !state.selectedWorkflow) return;
    
    const outputFormat = state.selectedWorkflow.export_format || state.selectedWorkflow.export_type;
    if (!outputFormat) {
      refs.outputFormatCards.innerHTML = '<span class="text-muted">No output formats available</span>';
      return;
    }
    
    // Format labels and icons
    const formatConfig = {
      'MD': { label: 'Markdown', icon: '📝' },
      'JSON': { label: 'JSON', icon: '{ }' },
      'CSV': { label: 'CSV', icon: '📊' },
      'XLSX': { label: 'Excel', icon: '📊' },
      'DOCX': { label: 'Word Doc', icon: '📄' },
      'PDF': { label: 'PDF', icon: '📕' }
    };
    
    // For now, single format from workflow (can be enhanced for multiple)
    const formats = [outputFormat];
    
    const cards = formats.map(fmt => {
      const config = formatConfig[fmt] || { label: fmt, icon: '📁' };
      const isSelected = state.selectedExportFormat === fmt;
      return `
        <div class="output-format-card ${isSelected ? 'selected' : ''}" 
             data-format="${escapeHtml(fmt)}"
             role="button"
             tabindex="0">
          <span class="format-icon">${config.icon}</span>
          <span class="format-name">${escapeHtml(config.label)}</span>
        </div>
      `;
    }).join('');
    
    refs.outputFormatCards.innerHTML = cards;
    
    // Add click handlers
    refs.outputFormatCards.querySelectorAll('.output-format-card').forEach(card => {
      card.addEventListener('click', () => {
        const format = card.dataset.format;
        selectOutputFormat(format);
      });
      card.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          selectOutputFormat(card.dataset.format);
        }
      });
    });
    
    // Auto-select the first format
    selectOutputFormat(outputFormat);
  }
  
  function selectOutputFormat(format) {
    state.selectedExportFormat = format;
    
    // Update card selection state
    if (refs.outputFormatCards) {
      refs.outputFormatCards.querySelectorAll('.output-format-card').forEach(card => {
        if (card.dataset.format === format) {
          card.classList.add('selected');
        } else {
          card.classList.remove('selected');
        }
      });
    }
    
    // Enable step 3 (processing)
    updateStartButton();
    goToStep(3);
  }

  // -------- Test Files Functions (Automation-Friendly) --------
  async function loadTestFiles() {
    if (!refs.testFilesGrid) return;
    
    try {
      const response = await fetch('/api/bulk-doc-analysis/test-files', {
        credentials: 'same-origin'
      });
      if (!response.ok) throw new Error('Failed to load test files');
      
      const data = await response.json();
      const testFiles = data.test_files || [];
      
      renderTestFiles(testFiles);
    } catch (error) {
      console.error('Failed to load test files:', error);
      refs.testFilesGrid.innerHTML = '<span class="text-muted">No test files available</span>';
    }
  }
  
  function renderTestFiles(files) {
    if (!refs.testFilesGrid) return;
    
    if (files.length === 0) {
      refs.testFilesGrid.innerHTML = '<span class="text-muted">No test files available. Add files to the project root or test_files/ folder.</span>';
      return;
    }
    
    // Filter based on selected workflow accepted types
    let filteredFiles = files;
    if (state.selectedWorkflow && state.selectedWorkflow.accepted_input_types) {
      const acceptedTypes = state.selectedWorkflow.accepted_input_types;
      filteredFiles = files.filter(f => acceptedTypes.includes(f.type));
    }
    
    // Deduplicate: keep only one file per type (first occurrence)
    const seenTypes = new Set();
    const uniqueFiles = filteredFiles.filter(file => {
      if (seenTypes.has(file.type)) {
        return false;
      }
      seenTypes.add(file.type);
      return true;
    });
    
    if (uniqueFiles.length === 0) {
      refs.testFilesGrid.innerHTML = '<span class="text-muted">No compatible test files for this workflow.</span>';
      return;
    }
    
    const fileIcons = {
      'PDF': '📕',
      'MD': '📝',
      'TXT': '📄',
      'DOCX': '📃',
      'CSV': '📊'
    };
    
    const formatSize = (bytes) => {
      if (bytes < 1024) return bytes + ' B';
      if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
      return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    };
    
    const buttons = uniqueFiles.map(file => {
      const icon = fileIcons[file.type] || '📁';
      return `
        <button type="button" 
                class="test-file-btn" 
                data-file-path="${escapeHtml(file.path)}"
                data-filename="${escapeHtml(file.filename)}">
          <span class="file-icon">${icon}</span>
          <span class="file-name">${escapeHtml(file.filename)}</span>
          <span class="file-size">(${formatSize(file.size)})</span>
        </button>
      `;
    }).join('');
    
    refs.testFilesGrid.innerHTML = buttons;
    
    // Add click handlers
    refs.testFilesGrid.querySelectorAll('.test-file-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const filePath = btn.dataset.filePath;
        const filename = btn.dataset.filename;
        uploadTestFile(filePath, filename);
      });
    });
  }
  
  async function uploadTestFile(filePath, filename) {
    try {
      toast('info', 'Uploading', `Adding ${filename} to session...`);
      
      const requestBody = {
        file_path: filePath
      };
      
      // Include workflow_version_id if selected
      if (state.selectedWorkflowVersionId) {
        requestBody.workflow_version_id = state.selectedWorkflowVersionId;
      }
      
      const response = await fetch('/api/bulk-doc-analysis/test-files/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
        credentials: 'same-origin'
      });
      
      if (!response.ok) {
        const error = await response.json().catch(() => ({ error: 'Upload failed' }));
        throw new Error(error.error || 'Failed to upload test file');
      }
      
      const data = await response.json();
      toast('success', 'Uploaded', data.message || `${filename} added to session`);
      
      // Refresh documents list
      await refreshDocs();
      
      // Start polling for conversion status
      startPollingDocs();
      
    } catch (error) {
      console.error('Upload test file error:', error);
      toast('error', 'Error', error.message || 'Failed to upload test file');
    }
  }

  async function startProcessing() {
    if (!state.selectedWorkflow || !state.selectedWorkflowVersionId) {
      toast('error', 'Validation', 'Please select a workflow');
      return;
    }
    
    if (state.docs.length === 0) {
      toast('error', 'Validation', 'Please upload at least one document');
      return;
    }
    
    const allConverted = state.docs.every(d => d.status === Status.CONVERTED);
    if (!allConverted) {
      toast('error', 'Validation', 'Please wait for all documents to be converted');
      return;
    }
    
    if (!state.selectedExportFormat) {
      toast('error', 'Validation', 'Please select an output format');
      return;
    }
    
    try {
      // Scalable solution: Backend automatically resolves chain_version_id from workflow_version_id
      // We can also pass chain_version_id if available for efficiency, but it's optional
      const requestBody = {
        workflow_version_id: state.selectedWorkflowVersionId
      };
      
      // If chain_version_id is available in workflow object, include it for efficiency
      if (state.selectedWorkflow.chain_version_id) {
        requestBody.chain_version_id = state.selectedWorkflow.chain_version_id;
      }
      
      // Check if we have an existing paused run to resume
      let runResponse;
      let runData;
      
      if (state.run.runId) {
        // Try to resume existing run
        runResponse = await fetch(`/api/bulk-doc-analysis/runs/${state.run.runId}/resume`, {
          method: 'POST',
          credentials: 'same-origin'
        });
        
        if (runResponse.ok) {
          runData = await runResponse.json();
          toast('info', 'Resumed', 'Resuming from where you left off...');
        } else {
          // If resume fails, create new run
          runResponse = await fetch('/api/bulk-doc-analysis/runs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody),
            credentials: 'same-origin'
          });
          
          if (!runResponse.ok) {
            const error = await runResponse.json().catch(() => ({ error: 'Failed to create run' }));
            throw new Error(error.error || 'Failed to start processing');
          }
          
          runData = await runResponse.json();
        }
      } else {
        // No existing run, create new one
        // Backend will check for paused runs and resume automatically
        runResponse = await fetch('/api/bulk-doc-analysis/runs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
          credentials: 'same-origin'
        });
        
        if (!runResponse.ok) {
          const error = await runResponse.json().catch(() => ({ error: 'Failed to create run' }));
          throw new Error(error.error || 'Failed to start processing');
        }
        
        runData = await runResponse.json();
      }
      state.currentRun = runData.run;
      state.run.runId = runData.run.run_id;
      
      // Initialize run rows from documents
      state.run.rows = state.docs.map(doc => ({
        localId: makeLocalId('runrow'),
        docId: doc.doc_id,
        filename: doc.filename,
        stepLabel: 'R0',
        status: 'QUEUED',
        inputTokens: null,
        outputTokens: null,
        canDownload: false,
      }));
      
      renderRun();
      
      // Show progress section
      if (refs.runProgressSection) {
        refs.runProgressSection.style.display = 'block';
      }
      
      goToStep(3);
      
      // Start polling for progress
      startPollingRunProgress();
      
      // Show pause button, hide start button
      if (refs.pauseProcessingBtn) {
        refs.pauseProcessingBtn.style.display = 'inline-block';
      }
      if (refs.startProcessingBtn) {
        refs.startProcessingBtn.style.display = 'none';
      }
      
      toast('success', 'Processing Started', 'Workflow execution has started');
      
    } catch (error) {
      console.error('Failed to start processing:', error);
      toast('error', 'Error', 'Failed to start processing: ' + error.message);
    }
  }

  async function pauseProcessing() {
    if (!state.run.runId) {
      toast('error', 'No Run', 'No active run to pause');
      return;
    }
    
    try {
      const response = await fetch(`/api/bulk-doc-analysis/runs/${state.run.runId}/pause`, {
        method: 'POST',
        credentials: 'same-origin'
      });
      
      if (!response.ok) {
        const error = await response.json().catch(() => ({ error: 'Failed to pause' }));
        throw new Error(error.error || 'Failed to pause processing');
      }
      
      // Stop polling
      stopPollingRunProgress();
      
      // Show pause button as hidden, show start button
      if (refs.pauseProcessingBtn) {
        refs.pauseProcessingBtn.style.display = 'none';
      }
      if (refs.startProcessingBtn) {
        refs.startProcessingBtn.style.display = 'inline-block';
        refs.startProcessingBtn.disabled = false;
      }
      
      toast('info', 'Paused', 'Run processing paused. Click "Start Processing" to resume.');
    } catch (error) {
      console.error('Failed to pause processing:', error);
      toast('error', 'Error', 'Failed to pause processing: ' + error.message);
    }
  }

  async function handleDocumentUpload(files) {
    if (!files || files.length === 0) return;
    
    if (!state.selectedWorkflow) {
      toast('error', 'Validation', 'Please select an agentic workflow template first');
      goToStep(1);
      return;
    }
    
    // Validate file types against workflow's accepted types
    const acceptedTypes = state.selectedWorkflow.accepted_input_types || [];
    const validFiles = [];
    const invalidFiles = [];
    
    files.forEach(file => {
      const ext = '.' + file.name.split('.').pop().toLowerCase();
      const typeMap = {
        '.pdf': 'PDF',
        '.docx': 'DOCX',
        '.txt': 'TXT',
        '.md': 'MD',
        '.csv': 'CSV'
      };
      const fileType = typeMap[ext];
      if (fileType && acceptedTypes.includes(fileType)) {
        validFiles.push(file);
      } else {
        invalidFiles.push(file.name);
      }
    });
    
    if (invalidFiles.length > 0) {
      toast('error', 'Invalid Files', `The following files are not accepted: ${invalidFiles.join(', ')}`);
    }
    
    if (validFiles.length === 0) {
      return;
    }
    
    // Optimistically add to state
    validFiles.forEach((f) => {
      state.docs.push({ localId: makeLocalId('doc'), filename: f.name, status: Status.QUEUED });
    });
    renderDocs();
    
    // Update status to processing
    state.docs.forEach((d) => {
      if (d.status === Status.QUEUED) d.status = Status.PROCESSING;
    });
    renderDocs();
    
    try {
      // Upload with workflow_version_id
      const form = new FormData();
      validFiles.forEach((f) => form.append('files', f));
      form.append('workflow_version_id', state.selectedWorkflowVersionId);
      
      const response = await fetch('/api/bulk-doc-analysis/documents/upload', {
        method: 'POST',
        body: form,
        credentials: 'same-origin'
      });
      
      if (!response.ok) {
        const error = await response.json().catch(() => ({ error: 'Upload failed' }));
        throw new Error(error.error || 'Upload failed');
      }
      
      const result = await response.json();
      
      // Update local doc entries with backend doc_ids
      if (result && result.documents) {
        const filenameToDoc = new Map(state.docs.map(d => [d.filename, d]));
        result.documents.forEach(backendDoc => {
          const localDoc = filenameToDoc.get(backendDoc.original_filename);
          if (localDoc) {
            localDoc.doc_id = backendDoc.doc_id;
            localDoc.status = backendDoc.status || Status.QUEUED;
          }
        });
      }
      
      renderDocs();
      updateStartButton();
      
      // Start polling for document conversion status
      startPollingDocs();
      
      toast('success', 'Upload Successful', `${validFiles.length} file(s) uploaded successfully`);
      
    } catch (error) {
      console.error('Upload error:', error);
      toast('error', 'Upload Failed', error.message);
      
      // Remove failed uploads from state
      state.docs = state.docs.filter(d => !validFiles.find(f => f.name === d.filename));
      renderDocs();
    }
  }

  function startPollingDocs() {
    if (state.polling.docsActive) return;
    state.polling.docsActive = true;
    if (state.polling.intervalId) {
      window.clearInterval(state.polling.intervalId);
    }
    state.polling.intervalId = window.setInterval(async () => {
      await refreshDocs();
      // Check if all docs are converted, then enable step 3
      const allConverted = state.docs.length > 0 && state.docs.every((d) => d.status === Status.CONVERTED);
      if (allConverted) {
        updateStartButton();
        // Auto-advance to step 2 (format selection) if we're on step 1
        if (state.wizardStep === 1) {
          goToStep(2);
        }
        // Stop polling docs once all converted
        stopPollingDocs();
      }
    }, 2000);
  }

  function stopPollingDocs() {
    if (state.polling.intervalId) {
      window.clearInterval(state.polling.intervalId);
      state.polling.intervalId = null;
      state.polling.docsActive = false;
    }
  }

  function startPollingRunProgress() {
    if (state.polling.runActive) return;
    state.polling.runActive = true;
    if (state.polling.runIntervalId) {
      window.clearInterval(state.polling.runIntervalId);
    }
    state.polling.runIntervalId = window.setInterval(async () => {
      if (!state.run.runId) return;
      try {
        const progress = await api.getRunProgress(state.run.runId);
        if (!progress || !progress.rows) return;
        
        state.run.rows = progress.rows.map((r) => ({
          localId: makeLocalId('runrow'),
          docId: r.doc_id,
          taskId: r.task_id || null,  // Include task_id for CSV workflows
          filename: r.filename,
          stepLabel: r.step_label || 'R0',
          status: r.status || 'QUEUED',
          inputTokens: r.input_tokens,
          outputTokens: r.output_tokens,
          canDownload: r.can_download || false,
        }));
        
        renderRun();
        
        // Stop polling if run is complete
        const isComplete = state.run.rows.every((r) => r.status === 'SUCCESS' || r.status === 'ERROR');
        if (isComplete) {
          stopPollingRunProgress();
          toast('success', 'Processing Complete', 'All documents have been processed');
        }
      } catch (error) {
        console.error('Error polling run progress:', error);
      }
    }, 2000);
  }

  function stopPollingRunProgress() {
    if (state.polling.runIntervalId) {
      window.clearInterval(state.polling.runIntervalId);
      state.polling.runIntervalId = null;
      state.polling.runActive = false;
    }
  }


  // -------- init --------
  (async function() {
    initTooltips();
    
    // Start with initial page (wizard hidden)
    hideWorkflowWizard();
    
    // Load workflows (renders as clickable cards)
    await loadWorkflows();
    
    // Load test files (automation-friendly)
    await loadTestFiles();
    
    // Load sessions
    await refreshSessions();
    
    // Setup event listeners
    // Note: Workflow cards have click handlers set in renderWorkflowCards()
    // Note: Session buttons have click handlers set in renderSessionSelector()
    
    if (refs.startProcessingBtn) {
      refs.startProcessingBtn.addEventListener('click', startProcessing);
    }
    
    if (refs.pauseProcessingBtn) {
      refs.pauseProcessingBtn.addEventListener('click', pauseProcessing);
    }
    
    // Setup upload area drag & drop
    if (refs.uploadArea && refs.uploadInput) {
      refs.uploadArea.addEventListener('click', (e) => {
        // Don't trigger if click is on the label/button inside upload area
        // The label already handles the click and opens the file picker
        if (e.target.tagName === 'LABEL' || e.target.closest('label') || 
            e.target.tagName === 'INPUT' || e.target.closest('button')) {
          return;
        }
        refs.uploadInput.click();
      });
      
      refs.uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        refs.uploadArea.classList.add('dragover');
      });
      
      refs.uploadArea.addEventListener('dragleave', () => {
        refs.uploadArea.classList.remove('dragover');
      });
      
      refs.uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        refs.uploadArea.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files);
        handleDocumentUpload(files);
      });
      
      // NOTE: Don't add 'change' handler here - it already exists earlier in the code (line ~988)
      // Adding it again would cause duplicate uploads
    }
    
    // Load recent runs
    await loadRecentRuns();
  })();
})();


