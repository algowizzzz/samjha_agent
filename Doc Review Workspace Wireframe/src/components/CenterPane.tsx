import { useEffect, useMemo, useRef, useState } from 'react';
import { Play, FileText, Upload } from 'lucide-react';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { BlockEditor } from './BlockEditor';
import { getDocument, runIngestion, type ApiDocument, updateDocumentMarkdown, type BlockMetadata, listTemplates, applyTemplate, type TemplateImprovement } from '@/lib/api';
import { MarkdownViewer } from './MarkdownViewer';
import { DiffView } from './DiffView';
import { activityLogger } from '@/utils/activityLogger';
import { isFeatureEnabled } from '@/lib/featureFlags';
import { SingleDocumentEditor } from './singleEditor/SingleDocumentEditor';
import { convertBlockMetadataToDocState } from './singleEditor/utils/converters';
import type { DocState } from '@/model/docTypes';

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

export function CenterPane({ mode, onModeChange, onTextSelect, selectedIssueId, onCommentClick, fileId, onSelectedBlocksChange, aiSuggestions = [], onSuggestionsListChange, selectedSuggestionId, onBlockWithSuggestionClick, onAcceptSuggestion, onRejectSuggestion, onSynthesisReceived }: CenterPaneProps) {
  const [doc, setDoc] = useState<ApiDocument | null>(null);
  const [loading, setLoading] = useState(false);
  const pollTimer = useRef<number | null>(null);
  const [templates, setTemplates] = useState<string[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<string>('');
  const [showTemplateDropdown, setShowTemplateDropdown] = useState(false);
  const [applyingTemplate, setApplyingTemplate] = useState(false);
  const [templateSuggestions, setTemplateSuggestions] = useState<Array<{ block_id: string; original: string; suggested: string; reason: string }>>([]);
  const [useSingleEditor] = useState(() => isFeatureEnabled('singleEditor'));
  
  // Memoize DocState conversion - only recalculate when file changes, not on state updates
  const initialDocState = useMemo(() => {
    const blockMetadata = doc?.state?.block_metadata || [];
    console.log('[CenterPane] Creating initialDocState for file:', fileId);
    return convertBlockMetadataToDocState(blockMetadata);
  }, [fileId]); // Only depend on fileId, not entire doc object
  
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

  // Auto-run ingestion if raw_markdown is missing
  useEffect(() => {
    if (!doc || !fileId) return;
    const rawMd = (doc.state as any)?.raw_markdown as string | undefined;
    const status = (doc.status || '').toLowerCase();
    if (!rawMd && status !== 'running' && status !== 'completed') {
      activityLogger.info(`[CenterPane] Auto-running ingestion for ${fileId}`);
      setLoading(true);
      runIngestionHandler();
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
  const docStatus = (doc?.status || 'idle').toLowerCase();

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

  const runIngestionHandler = async () => {
    if (!fileId) return;
    console.debug('[CenterPane] Running ingestion', { fileId });
    setLoading(true);
    try {
      await runIngestion(fileId);
      await refreshDocument();
    } catch (e) {
      console.error('[CenterPane] Ingestion error', e);
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
              className="!bg-blue-600 hover:!bg-blue-700 text-white"
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

        {/* Track Changes toggle removed per user request */}
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
              {improvedMarkdown || rawMarkdown ? (
                useSingleEditor ? (
                  // NEW: Single Document Editor (Option 3) - Clean Notion-like experience
                  <div className="h-full bg-white">
                    <SingleDocumentEditor
                      key={fileId}
                      initialDoc={initialDocState}
                      onDocChange={(docState: DocState) => {
                        console.log('[CenterPane] SingleDocumentEditor changed:', docState);
                        // TODO: Implement auto-save or debounced save
                      }}
                      readOnly={false}
                    />
                  </div>
                ) : (
                  // LEGACY: Per-block editor with header
                  <>
                    <div className="bg-neutral-100 px-6 py-2 sticky top-0 border-b border-neutral-200">
                      <span className="text-neutral-700 text-sm font-medium">
                        {improvedMarkdown ? 'Improved Markdown' : rawMarkdown ? 'Original Markdown' : 'No content yet'}
                      </span>
                    </div>
                    <div className="px-10 py-6">
                    // LEGACY: Per-block editor
                    <BlockEditor 
                        trackChangesEnabled={false}
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
                          
                          try {
                            const fileId = doc?.file_id;
                            if (!fileId) throw new Error('No file ID available');
                            
                            // FIX: Pass parameters in correct order (markdown, toc_markdown, block_metadata, ...)
                            await updateDocumentMarkdown(
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
                    </div>
                  </>
                )
              ) : (
                <div className="px-10 py-6">
                  <div className="text-sm text-neutral-600">
                    Run Phase 1 to ingest and normalize the document, then content will appear here.
                  </div>
                </div>
              )}
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
          // Diff mode - Show line-by-line comparison
          (() => {
            // Prepare blocks data for DiffView
            const blockMetadata = doc?.state?.block_metadata as BlockMetadata[] | undefined;
            const verificationSuggestions = doc?.state?.verification_suggestions || [];
            const templateImprovements = doc?.state?.template_improvements || [];
            
            // Convert metadata to blocks with change history
            const blocks = (blockMetadata || []).map((meta) => {
              const changeHistory: Array<{
                timestamp: string;
                type: string;
                original: string;
                modified: string;
                reason?: string;
                user?: string;
              }> = [];
              
              // Check for verification suggestions
              const verifySuggestion = verificationSuggestions.find((s: any) => s.block_id === meta.id);
              if (verifySuggestion) {
                changeHistory.push({
                  timestamp: new Date().toISOString(),
                  type: 'verified',
                  original: verifySuggestion.original,
                  modified: verifySuggestion.suggested,
                  reason: verifySuggestion.reason,
                  user: 'system'
                });
              }
              
              // Check for template improvements
              const templateImprovement = templateImprovements.find((imp: any) => imp.block_id === meta.id);
              if (templateImprovement) {
                changeHistory.push({
                  timestamp: new Date().toISOString(),
                  type: 'ai_suggested',
                  original: templateImprovement.original,
                  modified: templateImprovement.improved,
                  reason: `${templateImprovement.reasoning}\n\nChanges: ${templateImprovement.changes_made?.join(', ') || 'N/A'}`,
                  user: 'riskgpt'
                });
              }
              
              return {
                id: meta.id,
                type: meta.type || 'paragraph',
                content: meta.content,
                changeHistory
              };
            });
            
            return blocks.length > 0 ? (
              <DiffView blocks={blocks} blockMetadata={blockMetadata} />
            ) : (
              <div className="flex items-center justify-center h-full">
                <div className="text-center text-neutral-500">
                  <p className="text-sm">No content available for diff comparison.</p>
                  <p className="text-xs mt-2">Please ensure the document has been processed.</p>
                </div>
              </div>
            );
          })()
        )}
      </div>
    </div>
  );
}