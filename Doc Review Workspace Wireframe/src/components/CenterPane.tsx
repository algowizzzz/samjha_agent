import { useEffect, useMemo, useRef, useState } from 'react';
import { Play, FileText, Upload, ChevronDown, ChevronUp } from 'lucide-react';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { BlockEditor } from './BlockEditor';
import { getDocument, runFull, runPhase1, runPhase2, runPhase4, type ApiDocument, updateDocumentMarkdown, type BlockMetadata, listTemplates, applyTemplate, type TemplateImprovement } from '@/lib/api';
import { MarkdownViewer } from './MarkdownViewer';
import { activityLogger } from '@/utils/activityLogger';

interface CenterPaneProps {
  mode: 'editing' | 'original' | 'diff';
  onModeChange: (mode: 'editing' | 'original' | 'diff') => void;
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
  onSynthesisReceived?: (synthesis: any) => void; // NEW: Pass synthesis summary to parent
}

type PhaseStatus = 'idle' | 'running' | 'done';

export function CenterPane({ mode, onModeChange, onTextSelect, selectedIssueId, onCommentClick, fileId, onSelectedBlocksChange, aiSuggestions = [], onSuggestionsListChange, selectedSuggestionId, onBlockWithSuggestionClick, onAcceptSuggestion, onRejectSuggestion, onSynthesisReceived }: CenterPaneProps) {
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
  const [showLogs, setShowLogs] = useState(false);
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

  // Expose refresh function to window for external refresh button
  useEffect(() => {
    (window as any).__centerPaneRefreshDocument = refreshDocument;
    return () => {
      delete (window as any).__centerPaneRefreshDocument;
    };
  }, [fileId]);

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

  // Get display status for badge
  const getDisplayStatus = (): { label: string; variant: 'default' | 'secondary' | 'destructive' | 'outline'; className?: string } => {
    if (!doc) return { label: 'No Document', variant: 'default' };
    
    const state = doc.state as any;
    const status = doc.status?.toLowerCase() || 'pending';
    
    // Check for errors
    if (status === 'error' || status === 'failed') {
      return { label: 'Error', variant: 'destructive' };
    }
    
    // Check if running
    if (status === 'running') {
      return { label: 'Processing', variant: 'default', className: 'bg-amber-100 text-amber-800 border-amber-200' };
    }
    
    // Check which phase is completed
    const hasBlocks = state?.block_metadata && state.block_metadata.length > 0;
    const hasSuggestions = state?.verification_suggestions && state.verification_suggestions.length > 0;
    const hasImprovedMarkdown = !!state?.improved_markdown;
    
    if (hasImprovedMarkdown) {
      return { label: 'Improved', variant: 'default', className: 'bg-emerald-100 text-emerald-800 border-emerald-200' };
    } else if (hasSuggestions && hasSuggestions.length > 0) {
      return { label: 'Reviewed', variant: 'default', className: 'bg-emerald-100 text-emerald-800 border-emerald-200' };
    } else if (hasBlocks) {
      return { label: 'Analyzed', variant: 'default', className: 'bg-emerald-100 text-emerald-800 border-emerald-200' };
    } else if (status === 'ready' || status === 'completed') {
      return { label: 'Ready', variant: 'default', className: 'bg-emerald-100 text-emerald-800 border-emerald-200' };
    } else if (status === 'pending') {
      return { label: 'Uploaded', variant: 'secondary' };
    }
    
    return { label: 'Pending', variant: 'default' };
  };

  const displayStatus = getDisplayStatus();

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
      
      // Pass synthesis to parent (RightPane via App)
      if (result.synthesis && onSynthesisReceived) {
        onSynthesisReceived(result.synthesis);
      }
      
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
          <div className="flex items-center gap-3">
            <div>
              <h1 className="text-neutral-900 mb-1">{title}</h1>
              <p className="text-neutral-500 text-xs">{doc?.file_id || ''}</p>
            </div>
            {doc && (
              <Badge variant={displayStatus.variant} className={`mt-1 ${displayStatus.className || ''}`}>
                {displayStatus.label}
              </Badge>
            )}
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
            onClick={() => onModeChange('original')}
            className={`px-4 py-1.5 rounded transition-colors text-sm ${
              mode === 'original' 
                ? 'bg-white text-neutral-900 shadow-sm' 
                : 'text-neutral-600 hover:text-neutral-900'
            }`}
          >
            Original
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
                      activityLogger.info('Saving document...');
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
                        console.log('[CenterPane] ✅ Save successful! Current editor state persisted.');
                        activityLogger.changesSaved();
                        
                        // Update local state with new accepted/rejected counts without reloading editor
                        if (doc && doc.state) {
                          setDoc({
                            ...doc,
                            state: {
                              ...doc.state,
                              accepted_suggestions: data.acceptedSuggestions,
                              rejected_suggestions: data.rejectedSuggestions,
                              block_metadata: data.blockMetadata,
                              improved_markdown: data.markdown
                            }
                          });
                        }
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
        ) : mode === 'original' ? (
          <div className="flex-1 overflow-y-auto">
            <div className="bg-neutral-100 px-6 py-2 sticky top-0 border-b border-neutral-200">
              <span className="text-neutral-700 text-sm font-medium">Original Document</span>
            </div>
            <div className="px-10 py-6">
              {doc?.state?.original_markdown || doc?.state?.raw_markdown ? (
                <MarkdownViewer 
                  content={doc?.state?.original_markdown || doc?.state?.raw_markdown || ''} 
                  title={`${title} (original)`} 
                  onCommentClick={onCommentClick} 
                />
              ) : (
                <div className="text-sm text-neutral-600">No original markdown available.</div>
              )}
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

      {/* Activity Section - Collapsible */}
      {doc && (
        <div className="border-t border-neutral-200 bg-neutral-50">
          <button
            onClick={() => setShowLogs(!showLogs)}
            className="w-full px-6 py-2 flex items-center justify-between hover:bg-neutral-100 transition-colors"
          >
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-neutral-700">Activity & Logs</span>
              {((doc.state?.logs?.length || 0) + (doc.state?.errors?.length || 0)) > 0 && (
                <Badge variant="secondary" className="text-xs">
                  {(doc.state?.logs?.length || 0) + (doc.state?.errors?.length || 0)} logs
                </Badge>
              )}
            </div>
            {showLogs ? (
              <ChevronUp className="w-4 h-4 text-neutral-500" />
            ) : (
              <ChevronDown className="w-4 h-4 text-neutral-500" />
            )}
          </button>
          
          {showLogs && (
            <div className="px-6 py-3 max-h-64 overflow-y-auto bg-white border-t border-neutral-200">
              {/* Current Processing Status */}
              {doc.status === 'running' && (
                <div className="mb-4 pb-3 border-b border-neutral-200">
                  <div className="flex items-center gap-2 text-amber-700 mb-2">
                    <div className="animate-spin h-4 w-4 border-2 border-amber-300 border-t-amber-700 rounded-full"></div>
                    <span className="text-xs font-semibold">Processing in Progress...</span>
                  </div>
                  <div className="text-xs text-neutral-600 bg-amber-50 px-3 py-2 rounded">
                    {doc.state?.last_node || 'Initializing'} 
                    {doc.state?.control && ` → ${doc.state.control}`}
                  </div>
                </div>
              )}

              {/* Phase Status Overview */}
              {(doc.state?.block_metadata?.length > 0 || doc.state?.verification_suggestions?.length > 0 || doc.state?.improved_markdown) && (
                <div className="mb-4 pb-3 border-b border-neutral-200">
                  <div className="text-xs font-semibold text-neutral-700 mb-2">Processing Stages:</div>
                  <div className="space-y-1">
                    <div className={`flex items-center gap-2 text-xs px-2 py-1 rounded ${doc.state?.block_metadata?.length > 0 ? 'bg-emerald-50 text-emerald-700' : 'bg-neutral-50 text-neutral-400'}`}>
                      <div className={`w-2 h-2 rounded-full ${doc.state?.block_metadata?.length > 0 ? 'bg-emerald-500' : 'bg-neutral-300'}`}></div>
                      <span>Phase 1: Analyzed ({doc.state?.block_metadata?.length || 0} blocks)</span>
                    </div>
                    <div className={`flex items-center gap-2 text-xs px-2 py-1 rounded ${doc.state?.verification_suggestions?.length > 0 ? 'bg-emerald-50 text-emerald-700' : 'bg-neutral-50 text-neutral-400'}`}>
                      <div className={`w-2 h-2 rounded-full ${doc.state?.verification_suggestions?.length > 0 ? 'bg-emerald-500' : 'bg-neutral-300'}`}></div>
                      <span>Phase 2: Reviewed ({doc.state?.verification_suggestions?.length || 0} suggestions)</span>
                    </div>
                    <div className={`flex items-center gap-2 text-xs px-2 py-1 rounded ${doc.state?.improved_markdown ? 'bg-emerald-50 text-emerald-700' : 'bg-neutral-50 text-neutral-400'}`}>
                      <div className={`w-2 h-2 rounded-full ${doc.state?.improved_markdown ? 'bg-emerald-500' : 'bg-neutral-300'}`}></div>
                      <span>Phase 3: {doc.state?.improved_markdown ? 'Improved document ready' : 'Not started'}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Errors */}
              {doc.state?.errors && doc.state.errors.length > 0 && (
                <div className="mb-3">
                  <div className="text-xs font-semibold text-red-700 mb-1">⚠️ Errors:</div>
                  {doc.state.errors.map((error: string, idx: number) => (
                    <div key={`error-${idx}`} className="text-xs text-red-600 py-1 px-2 bg-red-50 rounded mb-1 font-mono">
                      {error}
                    </div>
                  ))}
                </div>
              )}
              
              {/* Processing Logs */}
              {doc.state?.logs && doc.state.logs.length > 0 && (
                <div className="mb-3">
                  <div className="text-xs font-semibold text-neutral-700 mb-1">🔍 Processing Details:</div>
                  <div className="space-y-0.5 max-h-32 overflow-y-auto">
                    {doc.state.logs.slice(-20).map((log: string | { node?: string; msg?: string; timestamp?: string }, idx: number) => {
                      const logText = typeof log === 'string' ? log : (log.msg || JSON.stringify(log));
                      return (
                        <div key={`log-${idx}`} className="text-xs text-neutral-600 py-1 px-2 bg-neutral-50 rounded font-mono">
                          {logText}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Document Stats */}
              {(doc.state?.raw_markdown || doc.updated_at) && (
                <div className="text-xs text-neutral-500 pt-2 border-t border-neutral-100">
                  <div className="flex justify-between">
                    <span>Last updated: {doc.updated_at ? new Date(doc.updated_at).toLocaleString() : 'Never'}</span>
                    {doc.state?.raw_markdown && (
                      <span>{Math.round(doc.state.raw_markdown.length / 1024)}KB</span>
                    )}
                  </div>
                </div>
              )}
              
              {!doc.state?.logs?.length && !doc.state?.errors?.length && !doc.state?.accepted_suggestions?.length && !doc.state?.rejected_suggestions?.length && !doc.state?.block_metadata?.length && (
                <div className="text-xs text-neutral-500 italic text-center py-4">No activity yet. Upload a document to get started.</div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}