import { useCallback, useEffect, useMemo, useState, memo } from 'react';
import { Resizable } from 're-resizable';
import { ChevronLeft, ChevronRight, PanelLeftClose, PanelRightClose } from 'lucide-react';
import { LeftPane } from './components/LeftPane';
import { CenterPane } from './components/CenterPane';
import { RightPane } from './components/RightPane';
import { DocumentsList } from './components/DocumentsList';
import { PromptsPage } from './components/PromptsPage';
import { SettingsPage } from './components/SettingsPage';
import { MainNav } from './components/MainNav';
import { MarkdownViewer } from './components/MarkdownViewer';
import { SingleEditorDemo } from './components/SingleEditorDemo';
import { type BlockMetadata } from './lib/api';
import { activityLogger } from './utils/activityLogger';
import { enableFeature } from './lib/featureFlags';

type Page = 'documents' | 'workspace' | 'prompts' | 'settings' | 'demo';

export default function App() {
  // Initialize currentPage based on whether we have a stored document
  const [currentPage, setCurrentPage] = useState<Page>(() => {
    const storedDocId = localStorage.getItem('lastSelectedDocumentId');
    return storedDocId ? 'workspace' : 'documents';
  });
  
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(() => {
    // Try to restore from localStorage
    return localStorage.getItem('lastSelectedDocumentId') || null;
  });
  
  const [selectedText, setSelectedText] = useState<string>('');
  const [centerMode, setCenterMode] = useState<'editing' | 'original' | 'diff'>('editing');
  const [selectedIssueId, setSelectedIssueId] = useState<string | null>(null);
  const [selectedBlockId, setSelectedBlockId] = useState<string | null>(null);
  const [selectedBlocks, setSelectedBlocks] = useState<BlockMetadata[]>([]); // NEW: Selected blocks for RiskGPT
  const [aiSuggestions, setAiSuggestions] = useState<Array<{ block_id: string; original: string; suggested: string; reason: string }>>([]); // NEW: AI suggestions from chat
  const [suggestions, setSuggestions] = useState<Array<{ block_id: string; original: string; suggested: string; reason: string; block_content: string }>>([]); // NEW: All suggestions for left panel
  const [selectedSuggestionId, setSelectedSuggestionId] = useState<string | null>(null); // NEW: Selected suggestion in left panel
  const [synthesisData, setSynthesisData] = useState<any>(null); // NEW: Template synthesis summary
  const [textSuggestion, setTextSuggestion] = useState<{ original: string; suggested: string } | null>(null); // NEW: AI text improvement suggestion
  const [selectedArtifact, setSelectedArtifact] = useState<{
    fileName: string;
    filePath: string;
    content: string;
    fileType: string;
  } | null>(null);
  
  // Text suggestion handlers (received from CenterPane)
  const [acceptTextSuggestion, setAcceptTextSuggestion] = useState<(() => void) | null>(null);
  const [rejectTextSuggestion, setRejectTextSuggestion] = useState<(() => void) | null>(null);
  
  // Analysis state (received from CenterPane)
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzeHandler, setAnalyzeHandler] = useState<(() => Promise<void>) | null>(null);
  
  const [leftPaneWidth, setLeftPaneWidth] = useState(280);
  const [rightPaneWidth, setRightPaneWidth] = useState(360);
  const [leftPaneCollapsed, setLeftPaneCollapsed] = useState(false);
  const [rightPaneCollapsed, setRightPaneCollapsed] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const file = params.get('file');
    const demo = params.get('demo');
    
    // Auto-enable singleEditor feature flag for demo
    if (demo === 'true') {
      enableFeature('singleEditor');
      setCurrentPage('demo');
    } else if (file) {
      setSelectedDocumentId(file);
      localStorage.setItem('lastSelectedDocumentId', file); // Persist for refresh
      setCurrentPage('workspace');
    } else if (selectedDocumentId) {
      // If no URL param but we have a stored documentId, go to workspace
      setCurrentPage('workspace');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Only run once on mount

  const handleOpenDocument = (fileId: string) => {
    setSelectedDocumentId(fileId);
    localStorage.setItem('lastSelectedDocumentId', fileId); // Persist for refresh
    setSelectedArtifact(null);
    setSelectedBlocks([]); // Clear selected blocks when switching documents
    setAiSuggestions([]); // Clear AI suggestions when switching documents
    setSuggestions([]); // Clear suggestions when switching documents
    setSelectedSuggestionId(null); // Clear selected suggestion
    setSynthesisData(null); // Clear synthesis data when switching documents
    setCurrentPage('workspace');
    const params = new URLSearchParams(window.location.search);
    params.set('file', fileId);
    const newUrl = `${window.location.pathname}?${params.toString()}`;
    window.history.pushState({}, '', newUrl);
  };

  const handleCommentClick = (blockId: string) => {
    setSelectedBlockId(blockId);
  };

  const handleArtifactSelect = (artifact: { fileName: string; filePath: string; content: string; fileType: string }) => {
    setSelectedArtifact(artifact);
  };

  const handleBackToDocument = () => {
    setSelectedArtifact(null);
  };

  // Handler for accepting a suggestion from the left panel
  const handleAcceptSuggestion = (blockId: string) => {
    console.log('[App] Accept suggestion:', blockId);
    activityLogger.suggestionAccepted(blockId);
    // Call the BlockEditor's accept function via window global
    if ((window as any).__blockEditorAcceptSuggestion) {
      (window as any).__blockEditorAcceptSuggestion(blockId);
    }
    // BlockEditor will update blocks state, which triggers onSuggestionsListChange
    // to automatically update the suggestions list
  };

  // Handler for rejecting a suggestion from the left panel
  const handleRejectSuggestion = (blockId: string) => {
    console.log('[App] Reject suggestion:', blockId);
    activityLogger.suggestionRejected(blockId);
    // Call the BlockEditor's reject function via window global
    if ((window as any).__blockEditorRejectSuggestion) {
      (window as any).__blockEditorRejectSuggestion(blockId);
    }
    // BlockEditor will update blocks state, which triggers onSuggestionsListChange
    // to automatically update the suggestions list
  };

  // Handler for AI text improvement suggestions
  const handleAISuggestion = (original: string, suggested: string) => {
    console.log('[App] AI text suggestion:', { original, suggested });
    setTextSuggestion({ original, suggested });
  };

  // Receive text suggestion handlers from CenterPane
  const handleTextSuggestionHandlers = (accept: () => void, reject: () => void) => {
    setAcceptTextSuggestion(() => accept);
    setRejectTextSuggestion(() => reject);
  };

  // Handler for analysis state changes from CenterPane
  const handleAnalysisStateChange = useCallback((analyzing: boolean, handler: (() => Promise<void>) | null) => {
    setIsAnalyzing(analyzing);
    setAnalyzeHandler(() => handler);
  }, []);

  // Handler for accepting text suggestion
  const handleAcceptTextSuggestion = useCallback(() => {
    console.log('[App] Accept text suggestion');
    if (acceptTextSuggestion) {
      acceptTextSuggestion();
    }
    setTextSuggestion(null);
  }, [acceptTextSuggestion]);

  // Handler for rejecting text suggestion
  const handleRejectTextSuggestion = useCallback(() => {
    console.log('[App] Reject text suggestion');
    if (rejectTextSuggestion) {
      rejectTextSuggestion();
    }
    setTextSuggestion(null);
  }, [rejectTextSuggestion]);

  // Handler for deselecting a block
  const handleDeselectBlock = useCallback((blockId: string) => {
    setSelectedBlocks(prev => prev.filter(b => b.id !== blockId));
    // Also deselect in BlockEditor
    if ((window as any).__blockEditorDeselectBlock) {
      (window as any).__blockEditorDeselectBlock(blockId);
    }
  }, []);

  // Handler for clearing all blocks
  const handleClearAllBlocks = useCallback(() => {
    // Clear the state in App
    setSelectedBlocks([]);
    // Also clear the selection in BlockEditor
    if ((window as any).__blockEditorClearSelection) {
      (window as any).__blockEditorClearSelection();
    }
  }, []);

  // Handler for commenting on a suggestion (opens RiskGPT chat)
  const handleCommentSuggestion = (blockId: string) => {
    console.log('[App] Comment on suggestion:', blockId);
    const suggestion = suggestions.find(s => s.block_id === blockId);
    if (suggestion) {
      // Set the selected suggestion to scroll to it in the editor
      setSelectedSuggestionId(blockId);
      
      // Select the block in the editor (like clicking sparkle button)
      // This will trigger the block to be selected and ready for RiskGPT
      if ((window as any).__blockEditorSelectBlock) {
        (window as any).__blockEditorSelectBlock(blockId);
      }
    }
  };

  
  // Workspace view
  if (currentPage === 'workspace') {
    return (
      <div className="flex flex-col h-screen w-screen bg-neutral-50 overflow-hidden">
        <MainNav currentPage={currentPage} onNavigate={setCurrentPage} />
        <div className="flex flex-1 overflow-hidden">
          {/* Left Pane - Resizable & Collapsible */}
          {!leftPaneCollapsed && (
            <Resizable
              size={{ width: leftPaneWidth, height: '100%' }}
              onResizeStop={(e, direction, ref, d) => {
                setLeftPaneWidth(leftPaneWidth + d.width);
              }}
              minWidth={200}
              maxWidth={500}
              enable={{ right: true }}
              handleStyles={{
                right: {
                  width: '4px',
                  right: '0',
                  cursor: 'col-resize',
                },
              }}
              handleClasses={{
                right: 'hover:bg-blue-500 transition-colors',
              }}
              className="border-r border-neutral-200 bg-white flex-shrink-0 relative"
            >
              <button
                onClick={() => setLeftPaneCollapsed(true)}
                className="absolute top-4 right-2 z-10 p-1 rounded hover:bg-neutral-100 transition-colors"
                title="Collapse panel"
              >
                <PanelLeftClose className="w-4 h-4 text-neutral-500" />
              </button>
              <LeftPane 
                fileId={selectedDocumentId || undefined}
                onIssueSelect={setSelectedIssueId}
                selectedIssueId={selectedIssueId}
                onAnalyzeDocument={analyzeHandler ? () => analyzeHandler() : undefined}
                isAnalyzing={isAnalyzing}
                onArtifactSelect={handleArtifactSelect}
                suggestions={suggestions}
                onSuggestionSelect={setSelectedSuggestionId}
                selectedSuggestionId={selectedSuggestionId}
                onAcceptSuggestion={handleAcceptSuggestion}
                onRejectSuggestion={handleRejectSuggestion}
                onCommentSuggestion={handleCommentSuggestion}
              />
            </Resizable>
          )}

          {/* Left Pane Collapsed Toggle */}
          {leftPaneCollapsed && (
            <div className="w-10 border-r border-neutral-200 bg-white flex-shrink-0 flex items-start justify-center pt-4">
              <button
                onClick={() => setLeftPaneCollapsed(false)}
                className="p-1 rounded hover:bg-neutral-100 transition-colors"
                title="Expand panel"
              >
                <ChevronRight className="w-4 h-4 text-neutral-500" />
              </button>
            </div>
          )}

          {/* Center Pane - Flexible */}
        <div className="flex-1 flex flex-col bg-white overflow-hidden">
            {selectedArtifact && selectedArtifact.fileType === 'markdown' ? (
              <MarkdownViewer
                content={selectedArtifact.content}
                title={selectedArtifact.fileName}
                onCommentClick={handleCommentClick}
              />
            ) : (
              <CenterPane 
                mode={centerMode}
                onModeChange={setCenterMode}
                onTextSelect={setSelectedText}
                selectedIssueId={selectedIssueId}
                onCommentClick={handleCommentClick}
                onSelectedBlocksChange={setSelectedBlocks}
                aiSuggestions={aiSuggestions}
                onSuggestionsListChange={setSuggestions}
                selectedSuggestionId={selectedSuggestionId}
                onBlockWithSuggestionClick={setSelectedSuggestionId}
                onAcceptSuggestion={handleAcceptSuggestion}
                onRejectSuggestion={handleRejectSuggestion}
                onSynthesisReceived={setSynthesisData}
                onAISuggestion={handleAISuggestion}
                onTextSuggestionHandlers={handleTextSuggestionHandlers}
                onAnalysisStateChange={handleAnalysisStateChange}
              // @ts-ignore
              fileId={selectedDocumentId || undefined}
              />
            )}
          </div>

          {/* Right Pane Collapsed Toggle */}
          {rightPaneCollapsed && (
            <div className="w-10 border-l border-neutral-200 bg-white flex-shrink-0 flex items-start justify-center pt-4">
              <button
                onClick={() => setRightPaneCollapsed(false)}
                className="p-1 rounded hover:bg-neutral-100 transition-colors"
                title="Expand panel"
              >
                <ChevronLeft className="w-4 h-4 text-neutral-500" />
              </button>
            </div>
          )}

          {/* Right Pane - Resizable & Collapsible */}
          {!rightPaneCollapsed && (
            <Resizable
              key="right-pane-resizable"
              size={{ width: rightPaneWidth, height: '100%' }}
              onResizeStop={(e, direction, ref, d) => {
                setRightPaneWidth(rightPaneWidth + d.width);
              }}
              minWidth={300}
              maxWidth={600}
              enable={{ left: true }}
              handleStyles={{
                left: {
                  width: '4px',
                  left: '0',
                  cursor: 'col-resize',
                },
              }}
              handleClasses={{
                left: 'hover:bg-blue-500 transition-colors',
              }}
              className="border-l border-neutral-200 bg-white flex-shrink-0 relative"
            >
              <button
                onClick={() => setRightPaneCollapsed(true)}
                className="absolute top-4 left-2 z-10 p-1 rounded hover:bg-neutral-100 transition-colors"
                title="Collapse panel"
              >
                <PanelRightClose className="w-4 h-4 text-neutral-500" />
              </button>
              <RightPane 
                key="right-pane-stable"
                selectedText={selectedText}
                selectedBlockId={selectedBlockId}
                onCommentClick={handleCommentClick}
                fileId={selectedDocumentId}
                selectedBlocks={selectedBlocks}
                onSuggestionsReceived={setAiSuggestions}
                synthesisData={synthesisData}
                textSuggestion={textSuggestion}
                onAcceptTextSuggestion={handleAcceptTextSuggestion}
                onRejectTextSuggestion={handleRejectTextSuggestion}
                onDeselectBlock={handleDeselectBlock}
                onClearAllBlocks={handleClearAllBlocks}
              />
            </Resizable>
          )}
        </div>
      </div>
    );
  }

  // Demo page - full screen
  if (currentPage === 'demo') {
    return <SingleEditorDemo />;
  }

  // Other pages
  return (
    <div className="flex flex-col h-screen w-screen bg-neutral-50 overflow-hidden">
      <MainNav currentPage={currentPage} onNavigate={setCurrentPage} />
      <div className="flex-1 overflow-hidden">
        {currentPage === 'documents' && (
          <DocumentsList onOpenDocument={handleOpenDocument} />
        )}
        {currentPage === 'prompts' && <PromptsPage />}
        {currentPage === 'settings' && <SettingsPage />}
      </div>
    </div>
  );
}