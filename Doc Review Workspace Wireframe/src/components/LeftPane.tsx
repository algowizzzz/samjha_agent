import React, { useEffect, useMemo, useState } from 'react';
import { FileText, AlertCircle, FolderTree, ChevronRight, ChevronDown, MessageSquare, Sparkles, Check, X } from 'lucide-react';
import { Badge } from './ui/badge';
import { ArtifactPreviewModal } from './ArtifactPreviewModal';
import { vfsReadFile, vfsTree } from '@/lib/api';
import { getDocument } from '@/lib/api';

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
      initial[s.block_id] = true; // All expanded by default
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
          // Briefly highlight
          suggestionElement.style.backgroundColor = '#fef3c7';
          setTimeout(() => {
            suggestionElement.style.backgroundColor = '';
          }, 2000);
        }, 100);
      }
    }
  }, [selectedSuggestionId]);

  return (
    <>
      {suggestions.map((suggestion, index) => {
        const expanded = expandedSections[suggestion.block_id] || { original: true, reasoning: true, improved: true };
        const isBoxExpanded = expandedBoxes[suggestion.block_id] !== false;
        
        return (
          <div
            key={suggestion.block_id}
            ref={(el) => {
              if (el) suggestionRefs.current.set(suggestion.block_id, el);
              else suggestionRefs.current.delete(suggestion.block_id);
            }}
            className={`p-3 rounded-lg border-2 transition-all ${
              selectedSuggestionId === suggestion.block_id
                ? 'bg-yellow-50 border-yellow-400 shadow-md'
                : 'bg-yellow-50/50 border-yellow-200 hover:border-yellow-300 hover:shadow-sm'
            }`}
          >
            {/* Header with number badge, action icons, and collapse button */}
            <div className="flex items-start gap-2 mb-3">
              <button
                onClick={() => toggleBox(suggestion.block_id)}
                className="flex-shrink-0 w-6 h-6 bg-yellow-400 text-white rounded-full flex items-center justify-center text-xs font-bold hover:bg-yellow-500 transition-colors"
                title="Collapse/Expand"
              >
                {index + 1}
              </button>
              <button
                onClick={() => {
                  // Expand the box and scroll to the block in the editor
                  if (!isBoxExpanded) {
                    setExpandedBoxes(prev => ({ ...prev, [suggestion.block_id]: true }));
                  }
                  onSuggestionSelect?.(suggestion.block_id);
                }}
                className="flex-1 min-w-0 text-left hover:bg-yellow-100 rounded px-2 py-1 transition-colors"
                title="Jump to block in editor"
              >
                <p className="text-xs text-neutral-600 font-medium">Suggestion {index + 1}</p>
                <p className="text-xs text-neutral-500 truncate mt-0.5">
                  {suggestion.block_content?.substring(0, 50) || 'Click to view in editor'}...
                </p>
              </button>
              
              {/* Action Icons: Comment, Accept, Reject */}
              <div className="flex items-center gap-1">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    console.log('[LeftPane] Comment/AskRiskGPT clicked:', suggestion.block_id);
                    onCommentSuggestion?.(suggestion.block_id);
                  }}
                  className="p-1.5 hover:bg-blue-100 rounded transition-colors"
                  title="Comment / Ask RiskGPT"
                >
                  <MessageSquare className="w-4 h-4 text-blue-600" />
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    console.log('[LeftPane] Accept clicked:', suggestion.block_id);
                    onAcceptSuggestion?.(suggestion.block_id);
                  }}
                  className="p-1.5 hover:bg-green-100 rounded transition-colors"
                  title="Accept Suggestion"
                >
                  <Check className="w-4 h-4 text-green-600" />
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    console.log('[LeftPane] Reject clicked:', suggestion.block_id);
                    onRejectSuggestion?.(suggestion.block_id);
                  }}
                  className="p-1.5 hover:bg-red-100 rounded transition-colors"
                  title="Reject Suggestion"
                >
                  <X className="w-4 h-4 text-red-600" />
                </button>
              </div>
              
              <button
                onClick={() => toggleBox(suggestion.block_id)}
                className="flex-shrink-0 text-neutral-500 text-sm hover:text-neutral-700 px-2 transition-colors"
                title="Collapse/Expand"
              >
                {isBoxExpanded ? '▼' : '▶'}
              </button>
            </div>

            {/* Collapsible content */}
            {isBoxExpanded && (
              <div>
                {/* Collapsible: Original Content */}
                <div className="mb-2">
                  <button
                    onClick={() => toggleSection(suggestion.block_id, 'original')}
                    className="w-full flex items-center justify-between text-left py-1.5 px-2 bg-neutral-100 hover:bg-neutral-200 rounded transition-colors"
                  >
                    <span className="text-xs font-semibold text-neutral-700">📄 Original</span>
                    <span className="text-xs text-neutral-500">{expanded.original ? '▼' : '▶'}</span>
                  </button>
                  {expanded.original && (
                    <div className="mt-1 p-2 bg-white border border-neutral-200 rounded text-xs text-neutral-600 leading-relaxed max-h-32 overflow-y-auto">
                      {suggestion.block_content || suggestion.original || 'N/A'}
                    </div>
                  )}
                </div>

                {/* Collapsible: Reasoning / Gap Analysis */}
                <div className="mb-2">
                  <button
                    onClick={() => toggleSection(suggestion.block_id, 'reasoning')}
                    className="w-full flex items-center justify-between text-left py-1.5 px-2 bg-blue-100 hover:bg-blue-200 rounded transition-colors"
                  >
                    <span className="text-xs font-semibold text-blue-700">💡 Reasoning</span>
                    <span className="text-xs text-blue-500">{expanded.reasoning ? '▼' : '▶'}</span>
                  </button>
                  {expanded.reasoning && (
                    <div className="mt-1 p-2 bg-white border border-blue-200 rounded text-xs text-neutral-600 leading-relaxed max-h-32 overflow-y-auto">
                      {suggestion.reason || 'No reasoning provided'}
                    </div>
                  )}
                </div>

                {/* Collapsible: Improved Content */}
                <div className="mb-3">
                  <button
                    onClick={() => toggleSection(suggestion.block_id, 'improved')}
                    className="w-full flex items-center justify-between text-left py-1.5 px-2 bg-green-100 hover:bg-green-200 rounded transition-colors"
                  >
                    <span className="text-xs font-semibold text-green-700">✨ Improved</span>
                    <span className="text-xs text-green-500">{expanded.improved ? '▼' : '▶'}</span>
                  </button>
                  {expanded.improved && (
                    <div className="mt-1 p-2 bg-white border border-green-200 rounded text-xs text-neutral-700 leading-relaxed max-h-32 overflow-y-auto font-medium">
                      {suggestion.suggested || 'No improved content'}
                    </div>
                  )}
                </div>
              </div>
            )}
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

  return (
    <div className="flex flex-col h-full">
      {/* Header - Suggestions Only */}
      <div className="border-b border-neutral-200 px-4 py-3 bg-yellow-50">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-neutral-900">Template Suggestions</h2>
          {suggestions.length > 0 && (
            <span className="px-2 py-1 bg-yellow-400 text-yellow-900 rounded-full text-xs font-bold">
              {suggestions.length}
            </span>
          )}
        </div>
      </div>

      {/* Suggestions Content */}
      <div className="flex-1 overflow-y-auto" style={{ overflowY: 'auto', overflowX: 'visible' }}>
        {/* Always show suggestions - no tabs */}
        {(
          <div className="p-3 space-y-3" style={{ minHeight: '100%' }}>
            {suggestions.length === 0 ? (
              <div className="text-center py-8 text-neutral-500 text-sm">
                <AlertCircle className="w-8 h-8 mx-auto mb-2 text-neutral-300" />
                <p>No suggestions yet</p>
                <p className="text-xs mt-1">Apply a template to see suggestions</p>
              </div>
            ) : (
              <SuggestionsList 
                suggestions={suggestions}
                selectedSuggestionId={selectedSuggestionId}
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