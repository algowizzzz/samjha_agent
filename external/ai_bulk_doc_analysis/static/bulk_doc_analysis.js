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
    docs: /** @type {Array<{localId:string, filename:string, status:string, errorMessage?:string}>} */ ([]),
    chains: /** @type {Array<{chain_version_id:string, name:string, description?:string, step_count:number, valid:boolean, steps?:Array<any>}>} */ ([]),
    selectedChainVersionId: null,
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
    chainEditor: {
      mode: null, // 'create' | 'edit'
      chainId: null,
      name: '',
      description: '',
      steps: [], // [{index: 1, required_inputs: ['R0'], prompt: '', description: ''}]
    },
  };

  // -------- DOM refs --------
  const refs = {
    uploadBtnLabel: document.getElementById('bulkDocUploadBtnLabel'),
    uploadInput: document.getElementById('bulkDocUploadInput'),
    docsEmpty: document.getElementById('bulkDocDocsEmpty'),
    docsSkeleton: document.getElementById('bulkDocDocsSkeleton'),
    docsWrap: document.getElementById('bulkDocDocsTableWrap'),
    docsTbody: document.getElementById('bulkDocDocsTbody'),

    chainSelectHint: document.getElementById('bulkDocChainSelectHint'),
    chainsWrap: document.getElementById('bulkDocChainsWrap'),
    chainsSkeleton: document.getElementById('bulkDocChainsSkeleton'),
    chainsList: document.getElementById('bulkDocChainsList'),
    chainDetail: document.getElementById('bulkDocChainDetail'),

    runBtn: document.getElementById('bulkDocRunBtn'),
    runEmpty: document.getElementById('bulkDocRunEmpty'),
    runSkeleton: document.getElementById('bulkDocRunSkeleton'),
    runWrap: document.getElementById('bulkDocRunTableWrap'),
    runTbody: document.getElementById('bulkDocRunTbody'),
    runSummary: document.getElementById('bulkDocRunSummary'),
    summaryCounts: document.getElementById('bulkDocSummaryCounts'),
    downloadAllBtn: document.getElementById('bulkDocDownloadAllBtn'),

    drawerEl: document.getElementById('bulkDocDocDrawer'),
    drawerFilename: document.getElementById('bulkDocDrawerFilename'),
    drawerStatus: document.getElementById('bulkDocDrawerStatus'),
    drawerDetails: document.getElementById('bulkDocDrawerDetails'),

    deleteModalEl: document.getElementById('bulkDocDeleteModal'),
    deleteTargetEl: document.getElementById('bulkDocDeleteTarget'),
    confirmDeleteBtn: document.getElementById('bulkDocConfirmDeleteBtn'),

    createChainBtn: document.getElementById('bulkDocCreateChainBtn'),
    chainEditorDrawer: document.getElementById('bulkDocChainEditorDrawer'),
    chainEditorDrawerLabel: document.getElementById('bulkDocChainEditorDrawerLabel'),
    chainEditorForm: document.getElementById('bulkDocChainEditorForm'),
    chainEditorName: document.getElementById('bulkDocChainName'),
    chainEditorNameError: document.getElementById('bulkDocChainNameError'),
    chainEditorDescription: document.getElementById('bulkDocChainDescription'),
    chainEditorStepsContainer: document.getElementById('bulkDocChainStepsContainer'),
    chainEditorStepsError: document.getElementById('bulkDocChainStepsError'),
    chainEditorAddStepBtn: document.getElementById('bulkDocAddStepBtn'),
    chainEditorCancelBtn: document.getElementById('bulkDocChainEditorCancelBtn'),
    chainEditorSaveBtn: document.getElementById('bulkDocChainEditorSaveBtn'),
  };

  // -------- helpers --------
  const mockMode = new URLSearchParams(window.location.search).get('mock') === '1';

  function makeLocalId(prefix) {
    return `${prefix}_${Math.random().toString(36).slice(2, 10)}_${Date.now()}`;
  }

  function statusBadge(status) {
    const map = {
      [Status.QUEUED]: 'secondary',
      [Status.PROCESSING]: 'info',
      [Status.CONVERTED]: 'success',
      [Status.ERROR]: 'danger',
      SUCCESS: 'success',
      RUNNING: 'info',
    };
    const cls = map[status] || 'secondary';
    return `<span class="badge bg-${cls}">${status}</span>`;
  }

  function canRun() {
    const hasDocs = state.docs.length > 0;
    const allConverted = hasDocs && state.docs.every((d) => d.status === Status.CONVERTED);
    const hasValidChain = !!state.selectedChainVersionId && state.chains.some((c) => c.chain_version_id === state.selectedChainVersionId && c.valid);
    return { ok: allConverted && hasValidChain, allConverted, hasValidChain, hasDocs };
  }

  function updateRunButton() {
    const gate = canRun();
    refs.runBtn.disabled = !gate.ok;

    let reason = 'Ready';
    if (!gate.hasDocs) reason = 'Upload and convert documents, then select a valid chain';
    else if (!gate.allConverted) reason = 'Wait for all documents to reach CONVERTED';
    else if (!gate.hasValidChain) reason = 'Select a valid chain to run';

    refs.runBtn.setAttribute('title', reason);
    initTooltips();
  }

  // -------- rendering --------
  function renderDocs() {
    const hasDocs = state.docs.length > 0;
    refs.docsEmpty.classList.toggle('d-none', hasDocs);
    refs.docsWrap.classList.toggle('d-none', !hasDocs);

    if (!hasDocs) {
      refs.docsTbody.innerHTML = '';
      updateRunButton();
      return;
    }

    refs.docsTbody.innerHTML = state.docs.map((d) => {
      const canDelete = d.status === Status.ERROR;
      const deleteBtn = canDelete
        ? `<button class="btn btn-sm btn-outline-danger" data-action="delete" data-id="${d.localId}" title="Delete errored document"><i class="bi bi-trash"></i></button>`
        : '';
      const statusDetail = (d.status === Status.ERROR && d.errorMessage)
        ? `<div class="bulk-doc-hint text-danger text-truncate" style="max-width: 220px;">${escapeHtml(d.errorMessage)}</div>`
        : '';
      return `
        <tr role="button" tabindex="0" data-action="open" data-id="${d.localId}">
          <td class="text-truncate" style="max-width: 260px;">${escapeHtml(d.filename)}</td>
          <td>${statusBadge(d.status)}${statusDetail}</td>
          <td class="text-end">${deleteBtn}</td>
        </tr>
      `;
    }).join('');

    updateRunButton();
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
      updateRunButton();
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

    updateRunButton();
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
    refs.runEmpty.classList.toggle('d-none', hasRows);
    refs.runWrap.classList.toggle('d-none', !hasRows);
    refs.runSummary.classList.toggle('d-none', !hasRows);

    if (!hasRows) {
      refs.runTbody.innerHTML = '';
      refs.downloadAllBtn.disabled = true;
      refs.summaryCounts.textContent = '0 processed';
      return;
    }

    refs.runTbody.innerHTML = state.run.rows.map((r) => {
      const tokensText = (typeof r.inputTokens === 'number' && typeof r.outputTokens === 'number')
        ? `${r.inputTokens} / ${r.outputTokens}`
        : '—';
      const downloadBtn = r.canDownload
        ? `<button class="btn btn-sm btn-outline-primary" data-action="download-one" data-id="${r.localId}"><i class="bi bi-download"></i></button>`
        : `<button class="btn btn-sm btn-outline-secondary" disabled title="Available on SUCCESS"><i class="bi bi-download"></i></button>`;
      return `
        <tr>
          <td class="text-truncate" style="max-width: 260px;">${escapeHtml(r.filename)}</td>
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

  // -------- API layer (intentionally minimal; backend will define exact endpoints) --------
  const api = {
    async uploadPdfs(files) {
      // Placeholder endpoint; will be implemented server-side later.
      // Keep a stable interface so UI stays deterministic.
      const form = new FormData();
      files.forEach((f) => form.append('files', f));
      const res = await fetch('/api/bulk-doc-analysis/documents/upload', { method: 'POST', body: form, credentials: 'same-origin' });
      if (!res.ok) throw new Error(`upload failed: ${res.status}`);
      return await res.json();
    },
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
    async deleteDoc(docId) {
      const res = await fetch(`/api/bulk-doc-analysis/documents/${encodeURIComponent(docId)}`, { method: 'DELETE', credentials: 'same-origin' });
      if (!res.ok) throw new Error(`delete failed: ${res.status}`);
      return await res.json();
    },
    async createRun(sessionId, chainVersionId) {
      const res = await fetch('/api/bulk-doc-analysis/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, chain_version_id: chainVersionId }),
        credentials: 'same-origin',
      });
      if (!res.ok) throw new Error(`create run failed: ${res.status}`);
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

        // Prompt textarea
        const promptTextarea = stepEl.querySelector(`[data-field="prompt"]`);
        if (promptTextarea) {
          promptTextarea.addEventListener('input', (e) => {
            updateStep(step.index, 'prompt', e.target.value);
          });
        }

        // Description input
        const descInput = stepEl.querySelector(`[data-field="description"]`);
        if (descInput) {
          descInput.addEventListener('input', (e) => {
            updateStep(step.index, 'description', e.target.value);
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

  // -------- events --------
  if (refs.uploadInput) {
    refs.uploadInput.addEventListener('change', async (e) => {
      const files = Array.from(e.target.files || []);
      if (files.length === 0) return;

      if (refs.uploadBtnLabel) {
        refs.uploadBtnLabel.innerHTML = `<i class="bi bi-upload"></i> Upload PDFs (${files.length})`;
      }
      if (refs.docsSkeleton) {
        refs.docsSkeleton.classList.remove('d-none');
      }

      // Optimistically add to table immediately (acceptance criteria)
      files.forEach((f) => {
        state.docs.push({ localId: makeLocalId('doc'), filename: f.name, status: Status.QUEUED });
      });
      renderDocs();

      // Show queued -> processing immediately
      state.docs.forEach((d) => {
        if (d.status === Status.QUEUED) d.status = Status.PROCESSING;
      });
      renderDocs();

      try {
        await api.uploadPdfs(files);
        toast('info', 'Upload started', `${files.length} file(s) uploaded. Conversion in progress.`);
        startPolling();
      } catch (err) {
        console.warn(err);
        if (mockMode) {
          toast('info', 'Mock mode', 'Backend missing; simulating conversion success locally.');
          window.setTimeout(() => {
            state.docs.forEach((d) => {
              if (d.status === Status.PROCESSING) d.status = Status.CONVERTED;
            });
            renderDocs();
            toast('success', 'Converted', 'All documents marked CONVERTED (mock).');
          }, 1200);
        } else {
          // Make failures actionable per spec: show ERROR and allow delete
          state.docs.forEach((d) => {
            if (d.status === Status.PROCESSING) {
              d.status = Status.ERROR;
              d.errorMessage = 'Conversion service unavailable (API not wired)';
            }
          });
          renderDocs();
          toast('error', 'Backend unavailable', 'Conversion API not available yet; documents marked ERROR so you can delete and proceed once backend is ready.');
        }
      } finally {
        refs.uploadInput.value = '';
        if (refs.uploadBtnLabel) {
          refs.uploadBtnLabel.innerHTML = `<i class="bi bi-upload"></i> Upload PDFs`;
        }
        if (refs.docsSkeleton) {
          window.setTimeout(() => refs.docsSkeleton.classList.add('d-none'), 400);
        }
      }
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

      // UI-only deletion (backend delete will come later); keep safe confirm UX
      try {
        await api.deleteDoc(id);
      } catch (err) {
        // If backend not present, still allow removal of errored docs from UI per acceptance
        console.warn(err);
      }

      state.docs = state.docs.filter((d) => d.localId !== id);
      renderDocs();

      if (typeof bootstrap !== 'undefined' && refs.deleteModalEl) {
        bootstrap.Modal.getOrCreateInstance(refs.deleteModalEl).hide();
      }
      toast('success', 'Deleted', 'Errored document removed.');
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

      toast('info', 'Downloading', `Downloading ${successRows.length} file(s)...`);
      for (const row of successRows) {
        try {
          const blob = await api.downloadDocOutput(state.run.runId, row.docId || row.localId);
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `${row.filename}.md`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          window.URL.revokeObjectURL(url);
        } catch (err) {
          console.error(`Download failed for ${row.filename}:`, err);
        }
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
        const blob = await api.downloadDocOutput(state.run.runId, row.docId || row.localId);
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${row.filename}.md`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        toast('success', 'Downloaded', `Downloaded ${row.filename}`);
      } catch (err) {
        toast('error', 'Download failed', err.message || 'Failed to download output.');
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

      // Map backend progress rows to UI state
      state.run.rows = progress.rows.map((r) => ({
        localId: makeLocalId('runrow'), // Could preserve mapping if needed
        docId: r.doc_id,
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

  async function refreshDocs() {
    try {
      const data = await api.listDocs();
      if (data && Array.isArray(data.documents)) {
        // Map backend docs to UI state (preserve localId if exists, else create new mapping)
        const existingMap = new Map(state.docs.map((d) => [d.filename, d.localId]));
        state.docs = data.documents.map((d) => ({
          localId: existingMap.get(d.original_filename) || makeLocalId('doc'),
          filename: d.original_filename,
          status: d.status,
          errorMessage: d.error_message || null,
        }));
        renderDocs();
      }
    } catch (err) {
      // stop noisy polling if endpoint not present
      if (String(err).includes('docs failed: 404')) {
        stopPolling();
      }
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

  // -------- init --------
  initTooltips();
  renderDocs();
  renderChains();
  renderRun();
  updateRunButton();
})();


