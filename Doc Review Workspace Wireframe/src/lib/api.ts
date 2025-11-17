export type BlockType = 'paragraph' | 'heading1' | 'heading2' | 'heading3' | 'bullet' | 'numbered' | 'table' | 'quote' | 'empty';

export type BlockMetadata = {
  id: string;           // Stable ID: "p1_b3_a8f2c9" (page_block_hash)
  page: number;
  block_num: number;    // Block number within page
  start_line: number;   // Start line in original markdown
  end_line: number;     // End line in original markdown
  content: string;      // Full block content (can be multi-line)
  type: BlockType;
};

export type VerificationSuggestion = {
  block_id: string;     // Matches BlockMetadata.id
  original: string;
  suggested: string;
  reason: string;
  confidence: 'high' | 'medium' | 'low';
};

export type ApiDocument = {
  file_id: string;
  source_path: string;
  status: string;
  updated_at?: string;
  file_metadata?: Record<string, unknown>;
  state?: {
    raw_markdown?: string;
    block_metadata?: BlockMetadata[];
    verification_suggestions?: VerificationSuggestion[];
    [key: string]: unknown;
  };
};

export type ApiTemplate = {
  template_id: string;
  path: string;
  size: number;
  location: string;
};

// Use port 8000 as Flask server runs on 8000 (port 5000 is used by macOS ControlCenter)
const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://localhost:8000/api'
  : '/api';
const DEV_API_KEY = 'docreview_dev_key_12345';

function buildHeaders(isJson: boolean = true): HeadersInit {
  const headers: HeadersInit = {
    'X-API-Key': DEV_API_KEY,
  };
  if (isJson) {
    headers['Content-Type'] = 'application/json';
  }
  return headers;
}

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.error) message = body.error;
    } catch {
      // ignore
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

// Debug helper
async function fetchWithDebug(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  try {
    // eslint-disable-next-line no-console
    console.debug('[api] request', typeof input === 'string' ? input : (input as URL).toString(), init?.method || 'GET', init);
  } catch {
    // ignore
  }
  
  const res = await fetch(input, init);
  
  try {
    // eslint-disable-next-line no-console
    console.debug('[api] response', typeof input === 'string' ? input : (input as URL).toString(), res.status, res.statusText);
  } catch {
    // ignore
  }
  
  // Check if we got redirected (302/301 would have been followed by fetch)
  // If the response URL is different from request URL, it was redirected
  const requestUrl = typeof input === 'string' ? input : (input as URL).toString();
  if (res.url && res.url !== requestUrl && !res.url.includes(requestUrl)) {
    console.error('[api] Unexpected redirect from', requestUrl, 'to', res.url);
    throw new Error('Authentication failed - session expired or invalid API key');
  }
  
  return res;
}

// Documents
export async function listDocuments(): Promise<{ documents: ApiDocument[] }> {
  const res = await fetchWithDebug(`${API_BASE}/doc_review/documents`, {
    headers: buildHeaders(),
  });
  return handleResponse(res);
}

export async function getDocument(fileId: string): Promise<{ document: ApiDocument }> {
  const res = await fetchWithDebug(`${API_BASE}/doc_review/documents/${encodeURIComponent(fileId)}`, {
    headers: buildHeaders(),
  });
  return handleResponse(res);
}

export async function deleteDocument(fileId: string): Promise<{ message: string }> {
  const res = await fetchWithDebug(`${API_BASE}/doc_review/documents/${encodeURIComponent(fileId)}`, {
    method: 'DELETE',
    headers: buildHeaders(),
  });
  return handleResponse(res);
}

export async function registerDocument(params: {
  source_path: string;
  file_id?: string;
  config?: Record<string, unknown>;
}): Promise<{ document: ApiDocument }> {
  const res = await fetchWithDebug(`${API_BASE}/doc_review/documents`, {
    method: 'POST',
    headers: buildHeaders(true),
    body: JSON.stringify(params),
  });
  return handleResponse(res);
}

export async function uploadFile(file: File): Promise<{ file_id: string; saved_path: string; original_filename: string }> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetchWithDebug(`${API_BASE}/doc_review/upload`, {
    method: 'POST',
    headers: {
      'X-API-Key': DEV_API_KEY,
    },
    body: form,
  });
  return handleResponse(res);
}

// Runs
export async function runFull(fileId: string): Promise<{ document: ApiDocument }> {
  const res = await fetchWithDebug(`${API_BASE}/doc_review/documents/${encodeURIComponent(fileId)}/run`, {
    method: 'POST',
    headers: buildHeaders(),
  });
  return handleResponse(res);
}

export async function runPhase1(fileId: string, templateId?: string): Promise<{ document: ApiDocument }> {
  const res = await fetchWithDebug(`${API_BASE}/doc_review/documents/${encodeURIComponent(fileId)}/run_phase1`, {
    method: 'POST',
    headers: buildHeaders(),
    body: JSON.stringify({ template_id: templateId }),
  });
  return handleResponse(res);
}

export async function runPhase2(fileId: string, sectionScope?: unknown): Promise<{ document: ApiDocument }> {
  const res = await fetchWithDebug(`${API_BASE}/doc_review/documents/${encodeURIComponent(fileId)}/run_phase2`, {
    method: 'POST',
    headers: buildHeaders(),
    body: JSON.stringify({ section_scope: sectionScope }),
  });
  return handleResponse(res);
}

export async function runPhase4(fileId: string): Promise<{ document: ApiDocument }> {
  const res = await fetchWithDebug(`${API_BASE}/doc_review/documents/${encodeURIComponent(fileId)}/run_phase4`, {
    method: 'POST',
    headers: buildHeaders(),
  });
  return handleResponse(res);
}

// get_template endpoint requires login; handle gracefully if blocked
export async function getTemplate(templateId: string): Promise<{ template_id: string; content: unknown }> {
  const res = await fetchWithDebug(`${API_BASE}/doc_review/templates/${encodeURIComponent(templateId)}`, {
    headers: buildHeaders(),
  });
  return handleResponse(res);
}

// VFS
export async function vfsTree(fileId: string, path: string = '/'): Promise<{ file_id: string; path: string; entries: unknown[] }> {
  const url = new URL(`${API_BASE}/doc_review/vfs/tree`, window.location.origin);
  url.searchParams.set('file_id', fileId);
  url.searchParams.set('path', path);
  const res = await fetchWithDebug(url.toString().replace(window.location.origin, ''), {
    headers: buildHeaders(),
  });
  return handleResponse(res);
}

export async function vfsReadFile(fileId: string, path: string): Promise<{ file_id: string; path: string; content: string }> {
  const url = new URL(`${API_BASE}/doc_review/vfs/file`, window.location.origin);
  url.searchParams.set('file_id', fileId);
  url.searchParams.set('path', path);
  const res = await fetchWithDebug(url.toString().replace(window.location.origin, ''), {
    headers: buildHeaders(),
  });
  return handleResponse(res);
}

// Markdown update
export async function updateDocumentMarkdown(
  fileId: string, 
  markdown: string, 
  toc_markdown?: string,
  block_metadata?: any[],
  accepted_suggestions?: string[],
  rejected_suggestions?: string[]
): Promise<{ document: ApiDocument }> {
  const url = `${API_BASE}/doc_review/documents/${encodeURIComponent(fileId)}/markdown`;
  console.log('[updateDocumentMarkdown] URL:', url);
  console.log('[updateDocumentMarkdown] fileId:', fileId);
  console.log('[updateDocumentMarkdown] payload:', {
    markdownLength: markdown.length,
    blockMetadataCount: block_metadata?.length,
    acceptedCount: accepted_suggestions?.length,
    rejectedCount: rejected_suggestions?.length
  });
  
  const res = await fetchWithDebug(url, {
    method: 'PUT',
    headers: buildHeaders(true),
    credentials: 'include', // Include cookies for session-based auth
    body: JSON.stringify({ 
      markdown, 
      toc_markdown,
      block_metadata,
      accepted_suggestions,
      rejected_suggestions
    }),
  });
  
  console.log('[updateDocumentMarkdown] Response status:', res.status);
  return handleResponse(res);
}

// Dev token for sockets
export async function getDevToken(): Promise<{ token: string; dev?: boolean }> {
  const res = await fetchWithDebug(`${API_BASE}/doc_review/dev_token`, {
    headers: buildHeaders(),
  });
  return handleResponse(res);
}

// Ask RiskGPT for block improvements
export type RiskGPTSuggestion = {
  block_id: string;
  original: string;
  suggested: string;
  reason: string;
  confidence: 'high' | 'medium' | 'low';
};

export async function askRiskGPT(
  fileId: string,
  selectedBlockIds: string[],
  userPrompt: string,
  conversationHistory?: Array<{ role: string; content: string }>
): Promise<{ file_id: string; analysis: string; suggestions: RiskGPTSuggestion[]; selected_block_ids: string[]; user_prompt: string }> {
  const res = await fetchWithDebug(`${API_BASE}/doc_review/ask_riskgpt`, {
    method: 'POST',
    headers: buildHeaders(true),
    credentials: 'include',
    body: JSON.stringify({
      file_id: fileId,
      selected_block_ids: selectedBlockIds,
      user_prompt: userPrompt,
      conversation_history: conversationHistory || [],
    }),
  });
  return handleResponse(res);
}

// Template management
export async function listTemplates(): Promise<{ templates: string[] }> {
  const res = await fetchWithDebug(`${API_BASE}/doc_review/templates`, {
    headers: buildHeaders(),
  });
  return handleResponse(res);
}

export async function uploadTemplate(file: File): Promise<{ template_name: string; message: string }> {
  const formData = new FormData();
  formData.append('file', file);
  
  // For FormData, only include API key, let browser set Content-Type with boundary
  const res = await fetchWithDebug(`${API_BASE}/doc_review/templates/upload`, {
    method: 'POST',
    headers: {
      'X-API-Key': DEV_API_KEY,
    },
    body: formData,
  });
  return handleResponse(res);
}

export async function deleteTemplate(templateName: string): Promise<{ message: string }> {
  const res = await fetchWithDebug(`${API_BASE}/doc_review/templates/${encodeURIComponent(templateName)}`, {
    method: 'DELETE',
    headers: buildHeaders(),
  });
  return handleResponse(res);
}

export type TemplateGapAnalysis = {
  block_id: string;
  gaps: string[];
  severity: 'high' | 'medium' | 'low';
  template_section: string;
  reasoning: string;
};

export type TemplateImprovement = {
  block_id: string;
  original: string;
  improved: string;
  changes_made: string[];
  reasoning: string;
  confidence: 'high' | 'medium' | 'low';
};

export interface TemplateSynthesis {
  overall_assessment: {
    compliance_level: 'full' | 'partial' | 'minimal' | 'none' | 'unknown';
    compliance_percentage: number;
    summary: string;
  };
  critical_gaps: Array<{
    title: string;
    impact: string;
    affected_pages: number[];
    count: number;
  }>;
  improvement_areas: Array<{
    category: string;
    title: string;
    issue_count: number;
    pages_affected: number[];
  }>;
  strengths: string[];
  priority_recommendations: string[];
  statistics: {
    total_issues: number;
    high_severity: number;
    medium_severity: number;
    low_severity: number;
    sections_analyzed: number;
    sections_missing: number;
  };
}

export async function applyTemplate(
  fileId: string,
  templateName: string
): Promise<{
  file_id: string;
  template_name: string;
  gap_analysis: TemplateGapAnalysis[];
  improvements: TemplateImprovement[];
  synthesis?: TemplateSynthesis;
  document: ApiDocument;
}> {
  const res = await fetchWithDebug(`${API_BASE}/doc_review/documents/${encodeURIComponent(fileId)}/apply_template`, {
    method: 'POST',
    headers: buildHeaders(true),
    body: JSON.stringify({
      template_name: templateName,
    }),
  });
  return handleResponse(res);
}

// ========================
// Prompts API
// ========================

export interface ApiPrompt {
  name: string;
  filename: string;
  size: number;
}

export interface ApiPromptContent {
  name: string;
  content: string;
}

export async function listPrompts(): Promise<{ prompts: ApiPrompt[] }> {
  const res = await fetchWithDebug(`${API_BASE}/doc_review/prompts`, {
    method: 'GET',
    headers: buildHeaders(false),
  });
  return handleResponse(res);
}

export async function getPrompt(name: string): Promise<ApiPromptContent> {
  const res = await fetchWithDebug(`${API_BASE}/doc_review/prompts/${encodeURIComponent(name)}`, {
    method: 'GET',
    headers: buildHeaders(false),
  });
  return handleResponse(res);
}

export async function updatePrompt(name: string, content: string): Promise<{ name: string; message: string }> {
  const res = await fetchWithDebug(`${API_BASE}/doc_review/prompts/${encodeURIComponent(name)}`, {
    method: 'PUT',
    headers: buildHeaders(true),
    body: JSON.stringify({ content }),
  });
  return handleResponse(res);
}

export async function getTemplateContent(templateName: string): Promise<{ template_name: string; content: string }> {
  const res = await fetchWithDebug(`${API_BASE}/doc_review/templates/${encodeURIComponent(templateName)}/content`, {
    method: 'GET',
    headers: buildHeaders(false),
  });
  return handleResponse(res);
}

