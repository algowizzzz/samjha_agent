import React, { useEffect, useMemo, useState, useCallback } from 'react';
import { FileText, AlertCircle, FolderTree, ChevronRight, ChevronDown, MessageSquare, Sparkles, Check, X } from 'lucide-react';
import { Badge } from './ui/badge';
import { ArtifactPreviewModal } from './ArtifactPreviewModal';
import { vfsReadFile, vfsTree } from '@/lib/api';
import { getDocument } from '@/lib/api';
import { activityLogger } from '@/utils/activityLogger';

type LeftPaneTab = 'suggestions' | 'outline' | 'issues' | 'artifacts';

// Separate component for suggestions list to avoid hooks in loops
function SuggestionsList({ 
  suggestions, 
  selectedSuggestionId, 
  onAcceptSuggestion, 
  onRejectSuggestion,
  onCommentSuggestion,
  onSuggestionSelect
}: {
  suggestions: Array<{ block_id: string; original: string; suggested: string; reason: string; block_content: string }>;
  selectedSuggestionId: string | null;
  onAcceptSuggestion?: (blockId: string) => void;
  onRejectSuggestion?: (blockId: string) => void;
  onCommentSuggestion?: (blockId: string) => void;
  onSuggestionSelect?: (blockId: string) => void;
}) {
  // Refs for scrolling to suggestions
  const suggestionRefs = React.useRef<Map<string, HTMLDivElement>>(new Map());

  // State for each suggestion box (collapsed/expanded)
  const [expandedBoxes, setExpandedBoxes] = useState<Record<string, boolean>>(() => {
    const initial: Record<string, boolean> = {};
    suggestions.forEach(s => {
      initial[s.block_id] = false; // All collapsed by default
    });
    return initial;
  });

  // State for each suggestion's collapsible sections
  const [expandedSections, setExpandedSections] = useState<Record<string, { original: boolean; reasoning: boolean; improved: boolean }>>(() => {
    const initial: Record<string, { original: boolean; reasoning: boolean; improved: boolean }> = {};
    suggestions.forEach(s => {
      initial[s.block_id] = { original: true, reasoning: true, improved: true };
    });
    return initial;
  });

  const toggleBox = (blockId: string) => {
    setExpandedBoxes(prev => ({
      ...prev,
      [blockId]: !prev[blockId]
    }));
  };

  const toggleSection = (blockId: string, section: 'original' | 'reasoning' | 'improved') => {
    setExpandedSections(prev => ({
      ...prev,
      [blockId]: {
        ...prev[blockId],
        [section]: !prev[blockId]?.[section]
      }
    }));
  };

  // Scroll to suggestion when selectedSuggestionId changes
  useEffect(() => {
    if (selectedSuggestionId) {
      const suggestionElement = suggestionRefs.current.get(selectedSuggestionId);
      if (suggestionElement) {
        // Expand the box if collapsed
        setExpandedBoxes(prev => ({ ...prev, [selectedSuggestionId]: true }));
        
        // Scroll to the suggestion
        setTimeout(() => {
          suggestionElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
          // Briefly highlight with animation
          suggestionElement.classList.add('highlight-flash');
          setTimeout(() => {
            suggestionElement.classList.remove('highlight-flash');
          }, 1000);
        }, 100);
      }
    }
  }, [selectedSuggestionId]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle if not typing in an input
      if ((e.target as HTMLElement).tagName === 'INPUT' || (e.target as HTMLElement).tagName === 'TEXTAREA') {
        return;
      }

      const currentIndex = selectedSuggestionId 
        ? suggestions.findIndex(s => s.block_id === selectedSuggestionId)
        : -1;

      // j or ArrowDown - next suggestion
      if (e.key === 'j' || e.key === 'ArrowDown') {
        e.preventDefault();
        const nextIndex = currentIndex < suggestions.length - 1 ? currentIndex + 1 : 0;
        onSuggestionSelect?.(suggestions[nextIndex].block_id);
      }
      // k or ArrowUp - previous suggestion
      else if (e.key === 'k' || e.key === 'ArrowUp') {
        e.preventDefault();
        const prevIndex = currentIndex > 0 ? currentIndex - 1 : suggestions.length - 1;
        onSuggestionSelect?.(suggestions[prevIndex].block_id);
      }
      // a - accept suggestion
      else if (e.key === 'a' && selectedSuggestionId) {
        e.preventDefault();
        onAcceptSuggestion?.(selectedSuggestionId);
      }
      // r - reject suggestion
      else if (e.key === 'r' && selectedSuggestionId) {
        e.preventDefault();
        onRejectSuggestion?.(selectedSuggestionId);
      }
      // c - comment/ask RiskGPT
      else if (e.key === 'c' && selectedSuggestionId) {
        e.preventDefault();
        onCommentSuggestion?.(selectedSuggestionId);
      }
      // Enter - toggle expand/collapse
      else if (e.key === 'Enter' && selectedSuggestionId) {
        e.preventDefault();
        setExpandedBoxes(prev => ({
          ...prev,
          [selectedSuggestionId]: !prev[selectedSuggestionId]
        }));
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [suggestions, selectedSuggestionId, onSuggestionSelect, onAcceptSuggestion, onRejectSuggestion, onCommentSuggestion]);

  return (
    <>
      {suggestions.map((suggestion, index) => {
        const expanded = expandedSections[suggestion.block_id] || { original: true, reasoning: true, improved: true };
        const isBoxExpanded = expandedBoxes[suggestion.block_id] !== false;
        
        // Generate meaningful title from reasoning
        const suggestionTitle = suggestion.reason 
          ? suggestion.reason.substring(0, 60).trim() + (suggestion.reason.length > 60 ? '...' : '')
          : `Suggestion ${index + 1}`;
        
        return (
          <div
            key={suggestion.block_id}
            ref={(el) => {
              if (el) suggestionRefs.current.set(suggestion.block_id, el);
              else suggestionRefs.current.delete(suggestion.block_id);
            }}
            className={`group p-4 rounded-xl border transition-all duration-300 ${
              selectedSuggestionId === suggestion.block_id
                ? 'bg-gradient-to-br from-amber-50 to-yellow-50 border-amber-300 shadow-lg ring-2 ring-amber-200'
                : 'bg-white border-neutral-200 hover:border-amber-200 hover:shadow-md hover:-translate-y-0.5'
            }`}
            style={{
              animation: selectedSuggestionId === suggestion.block_id ? 'none' : undefined,
            }}
          >
            {/* Header with number badge, title, and metadata */}
            <div className="flex items-start gap-3 mb-3">
              {/* Number badge */}
              <div className="flex-shrink-0 w-8 h-8 bg-gradient-to-br from-amber-400 to-yellow-500 text-white rounded-lg flex items-center justify-center text-sm font-bold shadow-sm">
                {index + 1}
              </div>

              {/* Title and metadata */}
              <button
                onClick={() => {
                  if (!isBoxExpanded) {
                    setExpandedBoxes(prev => ({ ...prev, [suggestion.block_id]: true }));
                  }
                  onSuggestionSelect?.(suggestion.block_id);
                }}
                className="flex-1 min-w-0 text-left group-hover:bg-amber-50/50 rounded-lg px-2 py-1 transition-all"
                title="Jump to block in editor"
              >
                <p className="text-sm font-semibold text-neutral-800 mb-1">{suggestionTitle}</p>
                <p className="text-xs text-neutral-500 truncate leading-relaxed">
                  {suggestion.block_content?.substring(0, 60) || 'Click to view in editor'}...
                </p>
              </button>
              
              {/* Action Icons: Comment, Accept, Reject */}
              <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    console.log('[LeftPane] Comment/AskRiskGPT clicked:', suggestion.block_id);
                    onCommentSuggestion?.(suggestion.block_id);
                  }}
                  className="p-2 hover:bg-blue-100 rounded-lg transition-all hover:scale-110 active:scale-95"
                  title="Ask RiskGPT (C)"
                >
                  <MessageSquare className="w-4 h-4 text-blue-600" />
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    console.log('[LeftPane] Accept clicked:', suggestion.block_id);
                    activityLogger.suggestionAccepted(suggestion.block_id);
                    onAcceptSuggestion?.(suggestion.block_id);
                  }}
                  className="p-2 hover:bg-green-100 rounded-lg transition-all hover:scale-110 active:scale-95"
                  title="Accept (A)"
                >
                  <Check className="w-4 h-4 text-green-600" />
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    console.log('[LeftPane] Reject clicked:', suggestion.block_id);
                    activityLogger.suggestionRejected(suggestion.block_id);
                    onRejectSuggestion?.(suggestion.block_id);
                  }}
                  className="p-2 hover:bg-red-100 rounded-lg transition-all hover:scale-110 active:scale-95"
                  title="Reject (R)"
                >
                  <X className="w-4 h-4 text-red-600" />
                </button>
              </div>
              
              {/* Expand/Collapse toggle */}
              <button
                onClick={() => toggleBox(suggestion.block_id)}
                className="flex-shrink-0 p-1 text-neutral-400 hover:text-neutral-600 hover:bg-neutral-100 rounded transition-all"
                title={isBoxExpanded ? 'Collapse (Enter)' : 'Expand (Enter)'}
              >
                {isBoxExpanded ? (
                  <ChevronDown className="w-4 h-4" />
                ) : (
                  <ChevronRight className="w-4 h-4" />
                )}
              </button>
            </div>

            {/* Collapsible content with smooth animation */}
            <div
              className="overflow-hidden transition-all duration-300 ease-in-out"
              style={{
                maxHeight: isBoxExpanded ? '1000px' : '0px',
                opacity: isBoxExpanded ? 1 : 0,
              }}
            >
              <div className="space-y-3 pt-2">
                {/* Original Content Section */}
                <div className="border border-neutral-200 rounded-lg overflow-hidden bg-neutral-50/50">
                  <button
                    onClick={() => toggleSection(suggestion.block_id, 'original')}
                    className="w-full flex items-center justify-between text-left py-2 px-3 hover:bg-neutral-100 transition-colors"
                  >
                    <span className="text-xs font-semibold text-neutral-700">
                      Original Content
                    </span>
                    {expanded.original ? (
                      <ChevronDown className="w-3 h-3 text-neutral-400" />
                    ) : (
                      <ChevronRight className="w-3 h-3 text-neutral-400" />
                    )}
                  </button>
                  <div
                    className="overflow-hidden transition-all duration-200"
                    style={{
                      maxHeight: expanded.original ? '200px' : '0px',
                    }}
                  >
                    <div className="p-3 bg-white text-xs text-neutral-600 leading-relaxed overflow-y-auto border-t border-neutral-200">
                      {suggestion.block_content || suggestion.original || 'N/A'}
                    </div>
                  </div>
                </div>

                {/* Reasoning Section */}
                <div className="border border-blue-200 rounded-lg overflow-hidden bg-blue-50/30">
                  <button
                    onClick={() => toggleSection(suggestion.block_id, 'reasoning')}
                    className="w-full flex items-center justify-between text-left py-2 px-3 hover:bg-blue-100/50 transition-colors"
                  >
                    <span className="text-xs font-semibold text-blue-700">
                      Why Change This
                    </span>
                    {expanded.reasoning ? (
                      <ChevronDown className="w-3 h-3 text-blue-400" />
                    ) : (
                      <ChevronRight className="w-3 h-3 text-blue-400" />
                    )}
                  </button>
                  <div
                    className="overflow-hidden transition-all duration-200"
                    style={{
                      maxHeight: expanded.reasoning ? '200px' : '0px',
                    }}
                  >
                    <div className="p-3 bg-white text-xs text-neutral-700 leading-relaxed overflow-y-auto border-t border-blue-200">
                      {suggestion.reason || 'No reasoning provided'}
                    </div>
                  </div>
                </div>

                {/* Improved Content Section */}
                <div className="border border-green-200 rounded-lg overflow-hidden bg-green-50/30">
                  <button
                    onClick={() => toggleSection(suggestion.block_id, 'improved')}
                    className="w-full flex items-center justify-between text-left py-2 px-3 hover:bg-green-100/50 transition-colors"
                  >
                    <span className="text-xs font-semibold text-green-700">
                      Improved Version
                    </span>
                    {expanded.improved ? (
                      <ChevronDown className="w-3 h-3 text-green-400" />
                    ) : (
                      <ChevronRight className="w-3 h-3 text-green-400" />
                    )}
                  </button>
                  <div
                    className="overflow-hidden transition-all duration-200"
                    style={{
                      maxHeight: expanded.improved ? '200px' : '0px',
                    }}
                  >
                    <div className="p-3 bg-white text-xs text-neutral-800 leading-relaxed overflow-y-auto border-t border-green-200 font-medium">
                      {suggestion.suggested || 'No improved content'}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </>
  );
}

interface Issue {
  id: string;
  severity: 'High' | 'Medium' | 'Low';
  section: string;
  description: string;
}

interface OutlineItem {
  id: string;
  level: number;
  title: string;
}

interface ArtifactNode {
  name: string;
  type: 'folder' | 'file';
  children?: ArtifactNode[];
}

const mockIssues: Issue[] = [];

const mockOutline: OutlineItem[] = [];

const mockArtifacts: ArtifactNode = {
  name: 'root',
  type: 'folder',
  children: [
    {
      name: 'phase1',
      type: 'folder',
      children: [
        { name: 'initial_analysis.json', type: 'file' },
        { name: 'risk_matrix.json', type: 'file' },
        { name: 'section_breakdown.json', type: 'file' },
      ],
    },
    {
      name: 'phase2',
      type: 'folder',
      children: [
        { name: 'detailed_review.json', type: 'file' },
        { name: 'issue_summary.json', type: 'file' },
        { name: 'recommendations.json', type: 'file' },
      ],
    },
    {
      name: 'phase4',
      type: 'folder',
      children: [
        { name: 'final_report.md', type: 'file' },
        { name: 'executive_summary.md', type: 'file' },
      ],
    },
  ],
};

interface Suggestion {
  block_id: string;
  original: string;
  suggested: string;
  reason: string;
  block_content: string;
}

interface LeftPaneProps {
  onIssueSelect: (id: string | null) => void;
  selectedIssueId: string | null;
  onArtifactSelect?: (artifact: { fileName: string; filePath: string; content: string; fileType: string }) => void;
  fileId?: string;
  suggestions?: Suggestion[];
  onSuggestionSelect?: (blockId: string) => void;
  selectedSuggestionId?: string | null;
  onAcceptSuggestion?: (blockId: string) => void;
  onRejectSuggestion?: (blockId: string) => void;
  onCommentSuggestion?: (blockId: string) => void;
}

export function LeftPane({ onIssueSelect, selectedIssueId, onArtifactSelect, fileId, suggestions = [], onSuggestionSelect, selectedSuggestionId, onAcceptSuggestion, onRejectSuggestion, onCommentSuggestion }: LeftPaneProps) {
  const [activeTab, setActiveTab] = useState<LeftPaneTab>('suggestions');
  const [severityFilter, setSeverityFilter] = useState<string>('All');
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(['phase1', 'phase2', 'phase4']));
  const [artifactPreview, setArtifactPreview] = useState<{
    fileName: string;
    filePath: string;
    content: string;
    fileType: 'json' | 'markdown' | 'text';
  } | null>(null);
  const [tree, setTree] = useState<any[] | null>(null);
  const [treeError, setTreeError] = useState<string | null>(null);
  const [treeLoading, setTreeLoading] = useState(false);
  const [outline, setOutline] = useState<OutlineItem[]>([]);
  const [issues, setIssues] = useState<Issue[]>([]);

  useEffect(() => {
    async function loadTree() {
      if (!fileId) {
        setTree(null);
        return;
      }
      setTreeLoading(true);
      setTreeError(null);
      try {
        const res = await vfsTree(fileId, '/');
        // eslint-disable-next-line no-console
        console.debug('[LeftPane] vfsTree ->', res);
        setTree(res.entries || []);
      } catch (e: any) {
        // eslint-disable-next-line no-console
        console.error('[LeftPane] vfsTree error', e);
        setTreeError(e?.message || 'Failed to load artifacts');
      } finally {
        setTreeLoading(false);
      }
    }
    loadTree();
  }, [fileId]);

  useEffect(() => {
    async function loadOutlineAndIssues() {
      if (!fileId) {
        setOutline([]);
        setIssues([]);
        return;
      }
      try {
        const res = await getDocument(fileId);
        const state: any = res.document?.state || {};
        // Outline: prefer heading_structure; fallback parse from raw_markdown
        const hs: any[] = state.heading_structure || [];
        let ol: OutlineItem[] = [];
        if (hs.length > 0) {
          ol = hs.map((h: any, idx: number) => ({
            id: `h${idx}`,
            level: Math.min(3, (h.level || 1)),
            title: String(h.title || h.text || '').trim(),
          })).filter(o => o.title);
        } else if (state.raw_markdown) {
          const lines = String(state.raw_markdown).split('\\n');
          ol = lines.map((line, idx) => {
            if (line.startsWith('### ')) return { id: `h${idx}`, level: 3, title: line.replace(/^###\\s+/, '') };
            if (line.startsWith('## ')) return { id: `h${idx}`, level: 2, title: line.replace(/^##\\s+/, '') };
            if (line.startsWith('# ')) return { id: `h${idx}`, level: 1, title: line.replace(/^#\\s+/, '') };
            return null as any;
          }).filter(Boolean);
        }
        setOutline(ol);
        // Issues: derive from phase2 or phase1 reports if present
        const phaseIssues = (state.phase2_report?.issues || state.phase1_report?.issues || []) as any[];
        const mappedIssues: Issue[] = (phaseIssues || []).slice(0, 50).map((it: any, i: number) => ({
          id: `i${i}`,
          severity: (String(it.severity || 'Medium').charAt(0).toUpperCase() + String(it.severity || 'Medium').slice(1)) as any,
          section: String(it.section || it.location || 'Document'),
          description: String(it.description || it.message || it.text || 'Issue'),
        }));
        setIssues(mappedIssues);
      } catch {
        setOutline([]);
        setIssues([]);
      }
    }
    loadOutlineAndIssues();
  }, [fileId]);

  const toggleFolder = (folderName: string) => {
    // eslint-disable-next-line no-console
    console.debug('[UI] Toggle folder', folderName);
    const newExpanded = new Set(expandedFolders);
    if (newExpanded.has(folderName)) {
      newExpanded.delete(folderName);
    } else {
      newExpanded.add(folderName);
    }
    setExpandedFolders(newExpanded);
  };

  const handleArtifactClick = async (fileName: string, path: string) => {
    if (!fileId) return;
    // eslint-disable-next-line no-console
    console.debug('[UI] Click artifact', { fileName, path });
    try {
      const res = await vfsReadFile(fileId, path);
      // eslint-disable-next-line no-console
      console.debug('[LeftPane] vfsReadFile ->', { path, len: (res.content || '').length });
      const content = res.content || '';
      let fileType: 'json' | 'markdown' | 'text' = 'text';
      if (fileName.endsWith('.json')) fileType = 'json';
      else if (fileName.endsWith('.md')) fileType = 'markdown';
      const artifact = { fileName, filePath: path, content, fileType };
      if (fileType === 'markdown' && onArtifactSelect) {
        onArtifactSelect(artifact);
      } else {
        setArtifactPreview(artifact);
      }
    } catch (e) {
      // Ignore for MVP
    }
  };

  const filteredIssues = severityFilter === 'All' 
    ? issues 
    : issues.filter(issue => issue.severity === severityFilter);

  const renderArtifactNode = (node: any, depth: number = 0, parentPath: string = ''): JSX.Element | null => {
    if (node.name === 'root') {
      return <>{node.children?.map((child, i) => renderArtifactNode(child, depth, '/'))}</>;
    }

    const isFolder = node.type === 'directory' || node.type === 'folder';
    const isExpanded = expandedFolders.has(node.name);
    const paddingLeft = depth * 16 + 12;
    const currentPath = `${parentPath}${node.name}${isFolder ? '/' : ''}`;

    if (isFolder) {
      return (
        <div key={node.name}>
          <button
            onClick={() => toggleFolder(node.name)}
            className="w-full flex items-center gap-2 px-3 py-1.5 hover:bg-neutral-100 transition-colors"
            style={{ paddingLeft }}
          >
            {isExpanded ? <ChevronDown className="w-4 h-4 text-neutral-500" /> : <ChevronRight className="w-4 h-4 text-neutral-500" />}
            <FolderTree className="w-4 h-4 text-neutral-600" />
            <span className="text-neutral-700">{node.name}</span>
          </button>
          {isExpanded && node.children?.map((child: any) => renderArtifactNode(child, depth + 1, currentPath))}
        </div>
      );
    }

    return (
      <button
        key={node.name}
        onClick={() => handleArtifactClick(node.name, currentPath)}
        className="w-full flex items-center gap-2 px-3 py-1.5 hover:bg-neutral-100 transition-colors"
        style={{ paddingLeft: paddingLeft + 20 }}
      >
        <FileText className="w-4 h-4 text-neutral-500" />
        <span className="text-neutral-600">{node.name}</span>
      </button>
    );
  };

  // Bulk actions
  const handleAcceptAll = () => {
    suggestions.forEach(suggestion => {
      onAcceptSuggestion?.(suggestion.block_id);
    });
  };

  const handleRejectAll = () => {
    suggestions.forEach(suggestion => {
      onRejectSuggestion?.(suggestion.block_id);
    });
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header - Suggestions Only */}
      <div className="border-b border-neutral-200 px-4 py-3 bg-gradient-to-r from-amber-50 to-yellow-50">
        <div className="flex items-center justify-between gap-2 mb-2">
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-semibold text-neutral-900">Template Suggestions</h2>
            {suggestions.length > 0 && (
              <span className="px-2 py-0.5 bg-gradient-to-r from-amber-400 to-yellow-500 text-white rounded-full text-xs font-bold shadow-sm">
                {suggestions.length}
              </span>
            )}
          </div>
          {/* Bulk action buttons */}
          {suggestions.length > 0 && (
            <div className="flex items-center gap-1">
              <button
                onClick={handleAcceptAll}
                className="px-2 py-1 text-xs font-medium text-green-700 bg-green-100 hover:bg-green-200 rounded-lg transition-colors flex items-center gap-1"
                title="Accept all suggestions"
              >
                <Check className="w-3 h-3" />
                Accept All
              </button>
              <button
                onClick={handleRejectAll}
                className="px-2 py-1 text-xs font-medium text-red-700 bg-red-100 hover:bg-red-200 rounded-lg transition-colors flex items-center gap-1"
                title="Reject all suggestions"
              >
                <X className="w-3 h-3" />
                Reject All
              </button>
            </div>
          )}
        </div>
        {/* Keyboard shortcuts hint */}
        {suggestions.length > 0 && (
          <p className="text-xs text-neutral-500">
            j/k to navigate • a to accept • r to reject • c to ask RiskGPT
          </p>
        )}
      </div>

      {/* Suggestions Content */}
      <div className="flex-1 overflow-y-auto" style={{ overflowY: 'auto', overflowX: 'visible' }}>
        {/* Always show suggestions - no tabs */}
        {(
          <div className="p-4 space-y-3" style={{ minHeight: '100%' }}>
            {suggestions.length === 0 ? (
              <div className="text-center py-12 px-4">
                <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-amber-100 to-yellow-100 rounded-full flex items-center justify-center">
                  <Sparkles className="w-8 h-8 text-amber-500" />
                </div>
                <h3 className="text-sm font-semibold text-neutral-800 mb-2">No suggestions yet</h3>
                <p className="text-xs text-neutral-500 leading-relaxed max-w-xs mx-auto">
                  Apply a template to your document to receive AI-powered improvement suggestions
                </p>
              </div>
            ) : (
              <SuggestionsList 
                suggestions={suggestions}
                selectedSuggestionId={selectedSuggestionId || null}
                onAcceptSuggestion={onAcceptSuggestion}
                onRejectSuggestion={onRejectSuggestion}
                onCommentSuggestion={onCommentSuggestion}
                onSuggestionSelect={onSuggestionSelect}
              />
            )}
          </div>
        )}
      </div>

      {artifactPreview && (
        <ArtifactPreviewModal
          fileName={artifactPreview.fileName}
          filePath={artifactPreview.filePath}
          content={artifactPreview.content}
          fileType={artifactPreview.fileType}
          onClose={() => setArtifactPreview(null)}
        />
      )}
    </div>
  );
}