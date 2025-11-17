import { useEffect, useMemo, useRef, useState } from 'react';
import { Play, FileText, Upload } from 'lucide-react';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { BlockEditor } from './BlockEditor';
import { getDocument, runFull, runPhase1, runPhase2, runPhase4, type ApiDocument, updateDocumentMarkdown, type BlockMetadata, listTemplates, applyTemplate, type TemplateImprovement } from '@/lib/api';
import { MarkdownViewer } from './MarkdownViewer';

interface CenterPaneProps {
  mode: 'editing' | 'diff';
  onModeChange: (mode: 'editing' | 'diff') => void;
  onTextSelect: (text: string) => void;
  selectedIssueId: string | null;
  onCommentClick: (blockId: string) => void;
  fileId?: string;
  onSelectedBlocksChange?: (selectedBlocks: BlockMetadata[]) => void; // NEW: Pass selected blocks to parent
  aiSuggestions?: Array<{ block_id: string; original: string; suggested: string; reason: string }>; // NEW: AI suggestions from chat
  onSuggestionsListChange?: (suggestions: Array<{ block_id: string; original: string; suggested: string; reason: string; block_content: string }>) => void; // NEW: Pass all suggestions to parent
  selectedSuggestionId?: string | null; // NEW: Highlight block when suggestion clicked
  onBlockWithSuggestionClick?: (blockId: string) => void; // NEW: Notify parent when block with suggestion is clicked
  onAcceptSuggestion?: (blockId: string) => void; // NEW: Accept suggestion from left panel
  onRejectSuggestion?: (blockId: string) => void; // NEW: Reject suggestion from left panel
}

type PhaseStatus = 'idle' | 'running' | 'done';

export function CenterPane({ mode, onModeChange, onTextSelect, selectedIssueId, onCommentClick, fileId, onSelectedBlocksChange, aiSuggestions = [], onSuggestionsListChange, selectedSuggestionId, onBlockWithSuggestionClick, onAcceptSuggestion, onRejectSuggestion }: CenterPaneProps) {
  const [phaseStatuses, setPhaseStatuses] = useState<Record<string, PhaseStatus>>({
    phase1: 'idle',
    phase2: 'idle',
    phase4: 'idle',
  });
  const [doc, setDoc] = useState<ApiDocument | null>(null);
  const [loading, setLoading] = useState(false);
  const pollTimer = useRef<number | null>(null);
  const [trackChangesEnabled, setTrackChangesEnabled] = useState(false);
  const [templates, setTemplates] = useState<string[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<string>('');
  const [showTemplateDropdown, setShowTemplateDropdown] = useState(false);
  const [applyingTemplate, setApplyingTemplate] = useState(false);
  const [templateSuggestions, setTemplateSuggestions] = useState<Array<{ block_id: string; original: string; suggested: string; reason: string }>>([]);
  // sockets disabled for now to avoid connection issues

  function clearPoll() {
    if (pollTimer.current) {
      window.clearInterval(pollTimer.current);
      pollTimer.current = null;
    }
  }

  async function refreshDocument() {
    if (!fileId) return;
    try {
      const res = await getDocument(fileId);
      // eslint-disable-next-line no-console
      console.debug('[CenterPane] getDocument ->', res);
      setDoc(res.document);
    } catch (e) {
      // ignore for now
    }
  }

  useEffect(() => {
    clearPoll();
    setDoc(null);
    setPhaseStatuses({ phase1: 'idle', phase2: 'idle', phase4: 'idle' });
    if (fileId) {
      refreshDocument();
    }
    return () => clearPoll();
  }, [fileId]);

  // Load templates on mount
  useEffect(() => {
    async function loadTemplates() {
      try {
        const res = await listTemplates();
        setTemplates(res.templates);
        if (res.templates.length > 0) {
          setSelectedTemplate(res.templates[0]);
        }
      } catch (e) {
        console.error('[CenterPane] Failed to load templates', e);
      }
    }
    loadTemplates();
  }, []);

  // Load saved template suggestions when document changes, filtering out accepted/rejected
  useEffect(() => {
    if (doc?.state?.template_improvements) {
      const improvements = doc.state.template_improvements as any[];
      const acceptedIds = new Set(doc.state.accepted_suggestions || []);
      const rejectedIds = new Set(doc.state.rejected_suggestions || []);
      
      // Filter out accepted and rejected suggestions
      const pendingSuggestions = improvements
        .filter((imp: any) => !acceptedIds.has(imp.block_id) && !rejectedIds.has(imp.block_id))
        .map((imp: any) => ({
          block_id: imp.block_id,
          original: imp.original,
          suggested: imp.improved,
          reason: `${imp.reasoning}\n\nChanges: ${imp.changes_made.join(', ')}`
        }));
      
      console.log('[CenterPane] Loaded template suggestions:', improvements.length, 'pending:', pendingSuggestions.length, 'accepted:', acceptedIds.size, 'rejected:', rejectedIds.size);
      console.log('[CenterPane] Accepted IDs:', Array.from(acceptedIds));
      console.log('[CenterPane] Rejected IDs:', Array.from(rejectedIds));
      setTemplateSuggestions(pendingSuggestions);
    } else {
      setTemplateSuggestions([]);
    }
  }, [doc]);

  // Auto-run Phase 1 if raw_markdown is missing (Phase 0 not done yet)
  useEffect(() => {
    if (!doc || !fileId) return;
    const rawMd = (doc.state as any)?.raw_markdown as string | undefined;
    const status = (doc.status || '').toLowerCase();
    if (!rawMd && status !== 'running' && status !== 'completed') {
      // eslint-disable-next-line no-console
      console.info('[CenterPane] Auto-running Phase 0 ingestion for', fileId);
      setLoading(true);
      handleRun('phase1');
    }
  }, [doc, fileId]);

  useEffect(() => {
    if (!doc) return;
    const status = (doc.status || '').toLowerCase();
    if (status === 'running') {
      if (!pollTimer.current) {
        pollTimer.current = window.setInterval(() => {
          refreshDocument();
    }, 2000);
      }
    } else {
      clearPoll();
    }
  }, [doc]);

  const title = useMemo(() => {
    if (!doc) return 'No document selected';
    const name = (doc.file_metadata as any)?.name as string | undefined;
    return name || doc.file_id;
  }, [doc]);

  const rawMarkdown = (doc?.state as any)?.raw_markdown as string | undefined;
  const improvedMarkdown = (doc?.state as any)?.improved_markdown as string | undefined;
  const leftContent = rawMarkdown || '';
  const rightContent = improvedMarkdown || rawMarkdown || '';

  const docStatus = (doc?.status || 'idle').toLowerCase();
  useEffect(() => {
    setPhaseStatuses({
      phase1: docStatus === 'running' ? 'running' : (docStatus === 'ready' || docStatus === 'completed') ? 'done' : 'idle',
      phase2: docStatus === 'running' ? 'running' : (docStatus === 'ready' || docStatus === 'completed') ? 'done' : 'idle',
      phase4: (doc?.state && (docStatus === 'ready' || docStatus === 'completed')) ? 'done' : (docStatus === 'running' ? 'running' : 'idle'),
    });
  }, [docStatus]);

  const handleRun = async (which: 'full' | 'phase1' | 'phase2' | 'phase4') => {
    if (!fileId) return;
    // eslint-disable-next-line no-console
    console.debug('[UI] Click Run', which, { fileId });
    setLoading(true);
    try {
      if (which === 'full') await runFull(fileId);
      if (which === 'phase1') await runPhase1(fileId);
      if (which === 'phase2') await runPhase2(fileId);
      if (which === 'phase4') await runPhase4(fileId);
      await refreshDocument();
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error('[CenterPane] run error', e);
      // swallow for MVP; future: show toast
    } finally {
      setLoading(false);
    }
  };

  const handleApplyTemplate = async () => {
    if (!fileId || !selectedTemplate) return;
    console.log('[CenterPane] Applying template:', selectedTemplate);
    setApplyingTemplate(true);
    setShowTemplateDropdown(false);
    try {
      const result = await applyTemplate(fileId, selectedTemplate);
      console.log('[CenterPane] Template applied:', result);
      
      // Convert improvements to suggestions format for BlockEditor
      const suggestions = result.improvements.map(imp => ({
        block_id: imp.block_id,
        original: imp.original,
        suggested: imp.improved,
        reason: `${imp.reasoning}\n\nChanges: ${imp.changes_made.join(', ')}`
      }));
      
      setTemplateSuggestions(suggestions);
      await refreshDocument();
    } catch (e) {
      console.error('[CenterPane] Template application error', e);
      alert(`Failed to apply template: ${e}`);
    } finally {
      setApplyingTemplate(false);
    }
  };

  const getStatusBadgeClass = (status: PhaseStatus) => {
    if (status === 'done') return 'bg-emerald-100 text-emerald-800';
    if (status === 'running') return 'bg-blue-100 text-blue-800';
    return 'bg-neutral-100 text-neutral-600';
  };

  return (
    <div className="flex flex-col h-full">
      {/* Top Bar */}
      <div className="border-b border-neutral-200 px-6 py-3">
        <div className="flex items-start justify-between mb-2">
          <div>
            <h1 className="text-neutral-900 mb-1">{title}</h1>
            <p className="text-neutral-500 text-xs">{doc?.file_id || ''}</p>
          </div>
          <div className="flex gap-2 items-center relative">
            {/* Template Dropdown */}
            <div className="relative">
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => setShowTemplateDropdown(!showTemplateDropdown)}
                disabled={!fileId || applyingTemplate || templates.length === 0}
                className="min-w-[200px] justify-between"
              >
                <span className="flex items-center">
                  <FileText className="w-3 h-3 mr-2" />
                  {selectedTemplate || 'Select Template'}
                </span>
                <svg className="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </Button>
              
              {showTemplateDropdown && (
                <div className="absolute top-full left-0 mt-1 w-full bg-white border border-neutral-200 rounded-lg shadow-lg z-50 max-h-60 overflow-y-auto">
                  {templates.map(template => (
                    <button
                      key={template}
                      onClick={() => {
                        setSelectedTemplate(template);
                        setShowTemplateDropdown(false);
                      }}
                      className={`w-full px-3 py-2 text-left text-sm hover:bg-neutral-100 transition-colors ${
                        selectedTemplate === template ? 'bg-blue-50 text-blue-900 font-medium' : 'text-neutral-700'
                      }`}
                    >
                      {template}
                    </button>
                  ))}
                </div>
              )}
            </div>
            
            {/* Apply Template Button */}
            <Button 
              variant="default" 
              size="sm"
              onClick={handleApplyTemplate}
              disabled={!fileId || !selectedTemplate || applyingTemplate}
              className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700"
            >
              {applyingTemplate ? (
                <>
                  <div className="w-3 h-3 mr-2 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  Applying...
                </>
              ) : (
                <>
                  <Play className="w-3 h-3 mr-2" />
                  Apply Template
                </>
              )}
            </Button>
          </div>
        </div>

        {/* Phase badges removed - not needed */}
      </div>

      {/* Mode Toggle & Track Changes */}
      <div className="border-b border-neutral-200 px-6 py-2 flex items-center justify-between">
        <div className="inline-flex bg-neutral-100 rounded p-1">
          <button
            onClick={() => onModeChange('editing')}
                onMouseDown={() => console.debug('[UI] Switch mode -> editing')}
            className={`px-4 py-1.5 rounded transition-colors text-sm ${
              mode === 'editing' 
                ? 'bg-white text-neutral-900 shadow-sm' 
                : 'text-neutral-600 hover:text-neutral-900'
            }`}
          >
            Editing
          </button>
          <button
            onClick={() => onModeChange('diff')}
                onMouseDown={() => console.debug('[UI] Switch mode -> diff')}
            className={`px-4 py-1.5 rounded transition-colors text-sm ${
              mode === 'diff' 
                ? 'bg-white text-neutral-900 shadow-sm' 
                : 'text-neutral-600 hover:text-neutral-900'
            }`}
          >
            Diff
          </button>
        </div>

        {mode === 'editing' && (
          <div className="flex items-center gap-3">
            <span className="text-xs text-neutral-600">Track Changes:</span>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={trackChangesEnabled}
                onChange={(e) => setTrackChangesEnabled(e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-9 h-5 bg-neutral-300 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-neutral-900 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-neutral-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-neutral-900"></div>
            </label>
            <span className={`text-xs ${trackChangesEnabled ? 'text-neutral-900 font-medium' : 'text-neutral-500'}`}>
              {trackChangesEnabled ? 'On' : 'Off'}
            </span>
          </div>
        )}
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-hidden">
        {loading || docStatus === 'running' ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-neutral-900 mx-auto mb-4"></div>
              <p className="text-neutral-700 font-medium">Running Phase 0 Ingestion...</p>
              <p className="text-neutral-500 text-sm mt-2">Converting document to markdown and extracting structure</p>
            </div>
          </div>
        ) : mode === 'editing' ? (
          <div className="flex h-full">
            <div className="flex-1 overflow-y-auto">
              <div className="bg-neutral-100 px-6 py-2 sticky top-0 border-b border-neutral-200">
                <span className="text-neutral-700 text-sm font-medium">
                  {improvedMarkdown ? 'Improved Markdown' : rawMarkdown ? 'Original Markdown' : 'No content yet'}
                </span>
              </div>
              <div className="px-10 py-6">
                {improvedMarkdown || rawMarkdown ? (
          <BlockEditor 
            trackChangesEnabled={trackChangesEnabled}
            onCommentClick={onCommentClick}
            selectedIssueId={selectedIssueId}
                    initialMarkdown={improvedMarkdown || rawMarkdown || ''}
                    blockMetadata={doc?.state?.block_metadata}
                    verificationSuggestions={doc?.state?.verification_suggestions}
                    fileId={fileId}
                    onSelectedBlocksChange={onSelectedBlocksChange}
                    aiSuggestions={[...aiSuggestions, ...templateSuggestions]}
                    onSuggestionsListChange={onSuggestionsListChange}
                    selectedSuggestionId={selectedSuggestionId}
                    onBlockWithSuggestionClick={onBlockWithSuggestionClick}
                    onAcceptSuggestion={onAcceptSuggestion}
                    onRejectSuggestion={onRejectSuggestion}
                    onSave={async (data: { 
                      markdown: string; 
                      blockMetadata: any[]; 
                      acceptedSuggestions: string[]; 
                      rejectedSuggestions: string[] 
                    }) => {
                      console.log('[CenterPane] onSave called with data:', {
                        markdownLength: data.markdown.length,
                        blockMetadataCount: data.blockMetadata.length,
                        acceptedCount: data.acceptedSuggestions.length,
                        rejectedCount: data.rejectedSuggestions.length
                      });
                      
                      if (!fileId) {
                        console.error('[CenterPane] Cannot save: fileId missing');
                        return;
                      }
                      
                      try {
                        const res = await updateDocumentMarkdown(
                          fileId, 
                          data.markdown,
                          undefined, // toc_markdown
                          data.blockMetadata,
                          data.acceptedSuggestions,
                          data.rejectedSuggestions
                        );
                        console.log('[CenterPane] ✅ Save successful!');
                        // Refresh document after a short delay to avoid loops
                        setTimeout(() => refreshDocument(), 500);
                      } catch (e) {
                        console.error('[CenterPane] ❌ Save failed:', e);
                        alert(`Failed to save: ${e}`);
                      }
                    }}
                  />
                ) : (
                  <div className="text-sm text-neutral-600">
                    Run Phase 1 to ingest and normalize the document, then content will appear here.
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="flex h-full">
            {/* Original - Left */}
            <div className="flex-1 border-right border-neutral-200 overflow-y-auto">
              <div className="bg-neutral-100 px-6 py-2 sticky top-0 border-b border-neutral-200">
                <span className="text-neutral-700 text-sm font-medium">Original</span>
              </div>
              <div className="px-10 py-6">
                {leftContent ? (
                  <MarkdownViewer content={leftContent} title={`${title} (original)`} onCommentClick={onCommentClick} />
                ) : (
                  <div className="text-sm text-neutral-600">No original markdown available.</div>
                )}
              </div>
            </div>
            {/* Updated - Right */}
            <div className="flex-1 overflow-y-auto">
              <div className="bg-neutral-100 px-6 py-2 sticky top-0 border-b border-neutral-200">
                <span className="text-neutral-700 text-sm font-medium">Improved</span>
              </div>
              <div className="px-10 py-6">
                {rightContent ? (
                  <MarkdownViewer content={rightContent} title={`${title} (improved)`} onCommentClick={onCommentClick} />
                ) : (
                  <div className="text-sm text-neutral-600">No improved markdown available. Run Assemble.</div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}