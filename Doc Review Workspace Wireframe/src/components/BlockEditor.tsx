import { useState, useRef, useEffect } from 'react';
import { 
  GripVertical, 
  MessageSquare, 
  MoreVertical, 
  Bold, 
  Italic, 
  Highlighter, 
  Link as LinkIcon, 
  MessageSquarePlus,
  Sparkles,
  Plus,
  Check,
  X as XIcon,
  AlertCircle
} from 'lucide-react';
import { BlockMetadata, VerificationSuggestion, RiskGPTSuggestion, askRiskGPT } from '@/lib/api';
import { activityLogger } from '@/utils/activityLogger';

type BlockType = 'paragraph' | 'heading1' | 'heading2' | 'heading3' | 'bullet' | 'numbered' | 'table' | 'callout' | 'quote' | 'empty';
type ChangeType = 'verified' | 'modified' | 'ai_suggested' | 'ai_applied' | 'rejected' | 'none';

interface ChangeRecord {
  timestamp: string;
  type: ChangeType;
  original: string;
  modified: string;
  reason?: string;
  user?: string;
}

interface Block {
  id: string;
  type: BlockType;
  content: string;
  changeType: ChangeType;
  commentCount: number;
  suggestion?: VerificationSuggestion;
  aiSuggestion?: RiskGPTSuggestion;  // NEW: AI suggestions from RiskGPT
  changeHistory: ChangeRecord[];  // Track all changes
}

interface BlockEditorProps {
  trackChangesEnabled: boolean;
  onCommentClick: (blockId: string) => void;
  selectedIssueId: string | null;
  initialMarkdown?: string;
  blockMetadata?: BlockMetadata[];  // NEW: Stable block IDs from backend
  verificationSuggestions?: VerificationSuggestion[];  // NEW: Suggestions
  onSave?: (data: { 
    markdown: string; 
    blockMetadata: BlockMetadata[]; 
    acceptedSuggestions: string[]; 
    rejectedSuggestions: string[] 
  }) => void;
  fileId?: string;  // NEW: For RiskGPT API calls
  onSelectedBlocksChange?: (selectedBlocks: BlockMetadata[]) => void;  // NEW: Callback when selection changes
  aiSuggestions?: Array<{ block_id: string; original: string; suggested: string; reason: string }>;  // NEW: AI suggestions from chat
  onSuggestionsListChange?: (suggestions: Array<{ block_id: string; original: string; suggested: string; reason: string; block_content: string }>) => void;  // NEW: Pass all suggestions to parent for left panel
  selectedSuggestionId?: string | null;  // NEW: Highlight block when suggestion clicked in left panel
  onBlockWithSuggestionClick?: (blockId: string) => void;  // NEW: Notify parent when block with suggestion is clicked
  onAcceptSuggestion?: (blockId: string) => void;  // NEW: Accept suggestion from left panel
  onRejectSuggestion?: (blockId: string) => void;  // NEW: Reject suggestion from left panel
}

const mockBlocks: Block[] = [];

function parseMarkdownToBlocks(markdown: string): Block[] {
  const lines = (markdown || '').split('\n');
  const blocks: Block[] = [];
  for (const line of lines) {
    const trimmed = line.trimEnd();
    if (trimmed.startsWith('### ')) {
      blocks.push({ id: `b${blocks.length + 1}`, type: 'heading3', content: trimmed.replace(/^###\\s+/, ''), changeType: 'none', commentCount: 0, changeHistory: [] });
    } else if (trimmed.startsWith('## ')) {
      blocks.push({ id: `b${blocks.length + 1}`, type: 'heading2', content: trimmed.replace(/^##\\s+/, ''), changeType: 'none', commentCount: 0, changeHistory: [] });
    } else if (trimmed.startsWith('# ')) {
      blocks.push({ id: `b${blocks.length + 1}`, type: 'heading1', content: trimmed.replace(/^#\\s+/, ''), changeType: 'none', commentCount: 0, changeHistory: [] });
    } else if (trimmed.startsWith('- ')) {
      blocks.push({ id: `b${blocks.length + 1}`, type: 'bullet', content: trimmed.replace(/^-\\s+/, ''), changeType: 'none', commentCount: 0, changeHistory: [] });
    } else if (trimmed.match(/^\\d+\\.\\s+/)) {
      blocks.push({ id: `b${blocks.length + 1}`, type: 'numbered', content: trimmed.replace(/^\\d+\\.\\s+/, ''), changeType: 'none', commentCount: 0, changeHistory: [] });
    } else if (trimmed.startsWith('> ')) {
      blocks.push({ id: `b${blocks.length + 1}`, type: 'quote', content: trimmed.replace(/^>\\s+/, ''), changeType: 'none', commentCount: 0, changeHistory: [] });
    } else if (trimmed.length === 0) {
      blocks.push({ id: `b${blocks.length + 1}`, type: 'paragraph', content: '', changeType: 'none', commentCount: 0, changeHistory: [] });
    } else {
      blocks.push({ id: `b${blocks.length + 1}`, type: 'paragraph', content: trimmed, changeType: 'none', commentCount: 0, changeHistory: [] });
    }
  }
  return blocks;
}

function parseMarkdownWithMetadata(
  markdown: string,
  blockMetadata: BlockMetadata[],
  suggestions: VerificationSuggestion[]
): Block[] {
  const blocks: Block[] = [];
  
  // Use semantic blocks from backend
  blockMetadata.forEach((meta) => {
    const blockId = meta.id;
    const blockType = (meta.type as BlockType) || 'paragraph';
    
    // Check if this block has a suggestion
    const suggestion = suggestions.find(s => s.block_id === blockId);
    
    // Determine change type based on suggestion
    const changeType: ChangeType = suggestion ? 'verified' : 'none';
    
    // Initialize change history with verification suggestion if present
    const changeHistory: ChangeRecord[] = suggestion ? [{
      timestamp: new Date().toISOString(),
      type: 'verified',
      original: suggestion.original,
      modified: suggestion.suggested,
      reason: suggestion.reason,
      user: 'system'
    }] : [];
    
    blocks.push({
      id: blockId,
      type: blockType,
      content: meta.content,  // Use full block content (can be multi-line)
      changeType,
      commentCount: 0,
      suggestion,
      changeHistory
    });
  });
  
  return blocks;
}

function blocksToMarkdown(blocks: Block[]): string {
  const lines: string[] = [];
  for (const b of blocks) {
    // For semantic blocks, content may already include markdown formatting
    // Just add the content as-is
    lines.push(b.content);
  }
  return lines.join('\n');
}

export function BlockEditor({ 
  trackChangesEnabled, 
  onCommentClick, 
  selectedIssueId, 
  initialMarkdown, 
  blockMetadata,
  verificationSuggestions,
  onSave,
  fileId,
  onSelectedBlocksChange,
  aiSuggestions,
  onSuggestionsListChange,
  selectedSuggestionId,
  onBlockWithSuggestionClick,
  onAcceptSuggestion,
  onRejectSuggestion
}: BlockEditorProps) {
  const [blocks, setBlocks] = useState<Block[]>(() => {
    if (initialMarkdown && initialMarkdown.trim().length > 0) {
      if (blockMetadata && verificationSuggestions) {
        return parseMarkdownWithMetadata(initialMarkdown, blockMetadata, verificationSuggestions);
      }
      return parseMarkdownToBlocks(initialMarkdown);
    }
    return mockBlocks;
  });
  const [hoveredBlock, setHoveredBlock] = useState<string | null>(null);
  const [selectedText, setSelectedText] = useState(false);
  const [selectionPosition, setSelectionPosition] = useState({ x: 0, y: 0 });
  const [showSlashMenu, setShowSlashMenu] = useState(false);
  const [slashMenuPosition, setSlashMenuPosition] = useState({ x: 0, y: 0 });
  const [editingBlockId, setEditingBlockId] = useState<string | null>(null);
  const editorRef = useRef<HTMLDivElement>(null);
  
  // Track accepted/rejected suggestions for persistence
  const [acceptedSuggestions, setAcceptedSuggestions] = useState<string[]>([]);
  const [rejectedSuggestions, setRejectedSuggestions] = useState<string[]>([]);
  
  // NEW: Block selection for RiskGPT
  const [selectedBlockIds, setSelectedBlockIds] = useState<Set<string>>(new Set());
  const [riskGPTPrompt, setRiskGPTPrompt] = useState('');
  const [isAskingRiskGPT, setIsAskingRiskGPT] = useState(false);
  const blockRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  useEffect(() => {
    if (initialMarkdown !== undefined) {
      if (blockMetadata && verificationSuggestions) {
        // Parse blocks with metadata and suggestions
        const blocksWithSuggestions = parseMarkdownWithMetadata(initialMarkdown, blockMetadata, verificationSuggestions);
        
        // AUTO-ACCEPT all verification suggestions silently
        const blocksWithAutoAccept = blocksWithSuggestions.map(block => {
          if (block.suggestion) {
            return {
              ...block,
              content: block.suggestion.suggested, // Apply the suggestion
              suggestion: undefined, // Remove the suggestion
              changeType: 'none' as const, // No visual indicator
              changeHistory: [
                ...block.changeHistory,
                {
                  timestamp: new Date().toISOString(),
                  type: 'verified' as const,
                  original: block.content,
                  modified: block.suggestion.suggested,
                  reason: `Auto-accepted verification: ${block.suggestion.reason}`,
                  user: 'system'
                }
              ]
            };
          }
          return block;
        });
        
        setBlocks(blocksWithAutoAccept);
      } else {
      setBlocks(parseMarkdownToBlocks(initialMarkdown || ''));
      }
    }
  }, [initialMarkdown, blockMetadata, verificationSuggestions]);

  // Auto-resize all textareas on mount and when blocks change
  useEffect(() => {
    const textareas = editorRef.current?.querySelectorAll('textarea');
    textareas?.forEach((textarea) => {
      textarea.style.height = 'auto';
      textarea.style.height = textarea.scrollHeight + 'px';
    });
  }, [blocks]);

  // Notify parent when selected blocks change
  useEffect(() => {
    if (onSelectedBlocksChange && blockMetadata) {
      const selectedBlocks = blockMetadata.filter(b => selectedBlockIds.has(b.id));
      onSelectedBlocksChange(selectedBlocks);
    }
  }, [selectedBlockIds, blockMetadata, onSelectedBlocksChange]);

  // Apply AI suggestions from chat to blocks (only once per suggestion set)
  const appliedSuggestionsRef = useRef<string>('');
  useEffect(() => {
    if (aiSuggestions && aiSuggestions.length > 0) {
      // Create a unique key for this set of suggestions
      const suggestionsKey = aiSuggestions.map(s => s.block_id).sort().join(',');
      
      // Only apply if we haven't seen this exact set before
      if (suggestionsKey !== appliedSuggestionsRef.current) {
        appliedSuggestionsRef.current = suggestionsKey;
        
        setBlocks(prevBlocks => {
          const updated = prevBlocks.map(block => {
            const suggestion = aiSuggestions.find(s => s.block_id === block.id);
            if (suggestion && !block.aiSuggestion) { // Only apply if block doesn't already have a suggestion
              return {
                ...block,
                aiSuggestion: {
                  block_id: suggestion.block_id,
                  original: suggestion.original,
                  suggested: suggestion.suggested,
                  reason: suggestion.reason,
                  confidence: 'high' as const
                },
                changeType: 'ai_suggested' as const,
                changeHistory: [
                  ...block.changeHistory,
                  {
                    timestamp: new Date().toISOString(),
                    type: 'ai_suggested' as const,
                    original: block.content,
                    modified: suggestion.suggested,
                    reason: suggestion.reason,
                    user: 'riskgpt'
                  }
                ]
              };
            }
            return block;
          });
          return updated;
        });
      }
    }
  }, [aiSuggestions]);

  // Notify parent of all suggestions for left panel (excluding accepted/rejected)
  useEffect(() => {
    if (onSuggestionsListChange) {
      const allSuggestions = blocks
        .filter(b => b.aiSuggestion && !acceptedSuggestions.includes(b.id) && !rejectedSuggestions.includes(b.id))
        .map(b => ({
          block_id: b.id,
          original: b.aiSuggestion!.original,
          suggested: b.aiSuggestion!.suggested,
          reason: b.aiSuggestion!.reason,
          block_content: b.content.substring(0, 100) // First 100 chars for preview
        }));
      onSuggestionsListChange(allSuggestions);
    }
  }, [blocks, onSuggestionsListChange, acceptedSuggestions, rejectedSuggestions]);

  // Scroll to block when suggestion is selected in left panel
  useEffect(() => {
    if (selectedSuggestionId) {
      const blockElement = blockRefs.current.get(selectedSuggestionId);
      if (blockElement) {
        blockElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
        // Briefly highlight the block
        blockElement.style.backgroundColor = '#fef3c7';
        setTimeout(() => {
          blockElement.style.backgroundColor = '';
        }, 2000);
      }
    }
  }, [selectedSuggestionId]);

  // Listen for accept/reject/select from left panel
  useEffect(() => {
    if (onAcceptSuggestion) {
      // Store the handler reference so we can call it from the left panel
      (window as any).__blockEditorAcceptSuggestion = (blockId: string) => {
        console.log('[BlockEditor] Accepting suggestion from left panel:', blockId);
        activityLogger.suggestionAccepted(blockId);
        acceptAISuggestion(blockId);
      };
    }
    if (onRejectSuggestion) {
      (window as any).__blockEditorRejectSuggestion = (blockId: string) => {
        console.log('[BlockEditor] Rejecting suggestion from left panel:', blockId);
        activityLogger.suggestionRejected(blockId);
        rejectAISuggestion(blockId);
      };
    }
    
    // Expose block selection for comment button
    (window as any).__blockEditorSelectBlock = (blockId: string) => {
      console.log('[BlockEditor] Selecting block from left panel:', blockId);
      const block = blocks.find(b => b.id === blockId);
      if (block && blockMetadata) {
        const metadata = blockMetadata.find(m => m.id === blockId);
        if (metadata) {
          setSelectedBlockIds(new Set([blockId]));
          if (onSelectedBlocksChange) {
            onSelectedBlocksChange([metadata]);
          }
        }
      }
    };
    
    return () => {
      delete (window as any).__blockEditorAcceptSuggestion;
      delete (window as any).__blockEditorRejectSuggestion;
      delete (window as any).__blockEditorSelectBlock;
    };
  }, [onAcceptSuggestion, onRejectSuggestion, blocks, blockMetadata, onSelectedBlocksChange]);

  useEffect(() => {
    const handleSelectionChange = () => {
      const selection = window.getSelection();
      if (selection && selection.toString().length > 0) {
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        setSelectionPosition({ x: rect.left, y: rect.top - 50 });
        setSelectedText(true);
      } else {
        setSelectedText(false);
      }
    };

    document.addEventListener('selectionchange', handleSelectionChange);
    return () => document.removeEventListener('selectionchange', handleSelectionChange);
  }, []);

  const handleAcceptChange = (blockId: string) => {
    setBlocks(blocks.map(b => 
      b.id === blockId ? { ...b, changeType: 'none' } : b
    ));
  };

  const handleRejectChange = (blockId: string) => {
    setBlocks(blocks.filter(b => b.id !== blockId));
  };

  const handleInputChange = (blockId: string, value: string) => {
    setBlocks(prev => prev.map(b => b.id === blockId ? { ...b, content: value } : b));
  };

  const acceptSuggestion = (blockId: string) => {
    setBlocks(prev => prev.map(b => {
      if (b.id === blockId && b.suggestion) {
        const newChangeRecord: ChangeRecord = {
          timestamp: new Date().toISOString(),
          type: 'verified',
          original: b.content,
          modified: b.suggestion.suggested,
          reason: `Accepted verification: ${b.suggestion.reason}`,
          user: 'user'
        };
        return {
          ...b,
          content: b.suggestion.suggested,
          changeType: 'none',
          suggestion: undefined,
          changeHistory: [...b.changeHistory, newChangeRecord]
        };
      }
      return b;
    }));
  };

  const rejectSuggestion = (blockId: string) => {
    setBlocks(prev => prev.map(b => {
      if (b.id === blockId && b.suggestion) {
        const newChangeRecord: ChangeRecord = {
          timestamp: new Date().toISOString(),
          type: 'rejected',
          original: b.content,
          modified: b.content,
          reason: `Rejected verification: ${b.suggestion.reason}`,
          user: 'user'
        };
        return {
          ...b,
          changeType: 'rejected',
          suggestion: undefined,
          changeHistory: [...b.changeHistory, newChangeRecord]
        };
      }
      return b;
    }));
  };

  // NEW: RiskGPT handlers
  const handleBlockClick = (blockId: string, event: React.MouseEvent) => {
    // Don't select if clicking on input/textarea (user is editing)
    const target = event.target as HTMLElement;
    if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
      return;
    }
    
    console.log('[BlockEditor] Block clicked:', blockId);
    activityLogger.blockSelected(blockId);
    
    // If this block has a suggestion, notify parent to highlight it in left panel
    const clickedBlock = blocks.find(b => b.id === blockId);
    if (clickedBlock?.aiSuggestion && onBlockWithSuggestionClick) {
      console.log('[BlockEditor] Block with suggestion clicked, notifying parent:', blockId);
      onBlockWithSuggestionClick(blockId);
    }
    
    if (event.shiftKey || event.metaKey || event.ctrlKey) {
      // Multi-select
      setSelectedBlockIds(prev => {
        const newSet = new Set(prev);
        if (newSet.has(blockId)) {
          newSet.delete(blockId);
        } else {
          newSet.add(blockId);
        }
        console.log('[BlockEditor] Multi-select, new selection:', Array.from(newSet));
        return newSet;
      });
    } else {
      // Single select
      const newSet = new Set([blockId]);
      console.log('[BlockEditor] Single select, new selection:', Array.from(newSet));
      setSelectedBlockIds(newSet);
    }
  };

  const handleAskRiskGPT = async () => {
    if (!riskGPTPrompt.trim() || selectedBlockIds.size === 0 || !fileId) return;
    
    setIsAskingRiskGPT(true);
    try {
      const result = await askRiskGPT(
        fileId,
        Array.from(selectedBlockIds),
        riskGPTPrompt
      );
      
      // Apply AI suggestions to blocks
      setBlocks(prev => prev.map(b => {
        const suggestion = result.suggestions.find(s => s.block_id === b.id);
        if (suggestion) {
          const newChangeRecord: ChangeRecord = {
            timestamp: new Date().toISOString(),
            type: 'ai_suggested',
            original: b.content,
            modified: suggestion.suggested,
            reason: suggestion.reason,
            user: 'system'
          };
          return {
            ...b,
            changeType: 'ai_suggested',
            aiSuggestion: suggestion,
            changeHistory: [...b.changeHistory, newChangeRecord]
          };
        }
        return b;
      }));
      
      setRiskGPTPrompt('');
      setSelectedBlockIds(new Set());
    } catch (error) {
      console.error('RiskGPT failed:', error);
      alert(`RiskGPT failed: ${error}`);
    } finally {
      setIsAskingRiskGPT(false);
    }
  };

  const acceptAISuggestion = (blockId: string) => {
    // Track accepted suggestion
    setAcceptedSuggestions(prev => [...prev, blockId]);
    
    setBlocks(prev => prev.map(b => {
      if (b.id === blockId && b.aiSuggestion) {
        const newChangeRecord: ChangeRecord = {
          timestamp: new Date().toISOString(),
          type: 'ai_applied',
          original: b.content,
          modified: b.aiSuggestion.suggested,
          reason: `Accepted RiskGPT: ${b.aiSuggestion.reason}`,
          user: 'user'
        };
        return {
          ...b,
          content: b.aiSuggestion.suggested,
          changeType: 'none',
          aiSuggestion: undefined,
          changeHistory: [...b.changeHistory, newChangeRecord]
        };
      }
      return b;
    }));
  };

  const rejectAISuggestion = (blockId: string) => {
    // Track rejected suggestion
    setRejectedSuggestions(prev => [...prev, blockId]);
    
    setBlocks(prev => prev.map(b => {
      if (b.id === blockId && b.aiSuggestion) {
        const newChangeRecord: ChangeRecord = {
          timestamp: new Date().toISOString(),
          type: 'rejected',
          original: b.content,
          modified: b.content,
          reason: `Rejected RiskGPT: ${b.aiSuggestion.reason}`,
          user: 'user'
        };
        return {
          ...b,
          changeType: 'rejected',
          aiSuggestion: undefined,
          changeHistory: [...b.changeHistory, newChangeRecord]
        };
      }
      return b;
    }));
  };

  const handleAddParagraph = () => {
    setBlocks(prev => [
      ...prev,
      { 
        id: `b${prev.length + 1}`, 
        type: 'paragraph', 
        content: '', 
        changeType: 'modified', 
        commentCount: 0,
        changeHistory: [{
          timestamp: new Date().toISOString(),
          type: 'modified',
          original: '',
          modified: '',
          reason: 'New paragraph added by user',
          user: 'user'
        }]
      },
    ]);
  };

  const handleSave = () => {
    if (!onSave || !blockMetadata) {
      console.error('[BlockEditor] ❌ Cannot save: missing onSave or blockMetadata');
      return;
    }
    
    // Convert blocks back to markdown
    const md = blocksToMarkdown(blocks);
    
    // Update block metadata with current content
    const updatedBlockMetadata = blockMetadata.map(meta => {
      const block = blocks.find(b => b.id === meta.id);
      if (block) {
        return {
          ...meta,
          content: block.content
        };
      }
      return meta;
    });
    
    activityLogger.info('Saving changes...');
    console.log('[BlockEditor] 💾 Saving...', {
      blocks: updatedBlockMetadata.length,
      accepted: acceptedSuggestions.length,
      rejected: rejectedSuggestions.length
    });
    
    // Pass all data for persistence
    onSave({
      markdown: md,
      blockMetadata: updatedBlockMetadata,
      acceptedSuggestions,
      rejectedSuggestions
    });
  };

  const handleAcceptAllChanges = () => {
    setBlocks(prevBlocks => prevBlocks.map(block => {
      // Accept verification suggestions
      if (block.suggestion) {
        return {
          ...block,
          content: block.suggestion.suggested,
          suggestion: undefined,
          changeType: 'none',
          changeHistory: [
            ...block.changeHistory,
            {
              timestamp: new Date().toISOString(),
              type: 'verified',
              original: block.content,
              modified: block.suggestion.suggested,
              reason: `Accepted verification: ${block.suggestion.reason}`,
              user: 'current_user'
            }
          ]
        };
      }
      // Accept AI suggestions
      if (block.aiSuggestion) {
        return {
          ...block,
          content: block.aiSuggestion.suggested,
          aiSuggestion: undefined,
          changeType: 'none',
          changeHistory: [
            ...block.changeHistory,
            {
              timestamp: new Date().toISOString(),
              type: 'ai_applied',
              original: block.content,
              modified: block.aiSuggestion.suggested,
              reason: `Accepted AI suggestion: ${block.aiSuggestion.reason}`,
              user: 'current_user'
            }
          ]
        };
      }
      return block;
    }));
  };

  const getBlockClassName = (block: Block) => {
    const baseClasses = 'relative group px-16 py-0.5 rounded transition-all cursor-pointer bg-white';
    
    // Check if block is selected
    const isSelected = selectedBlockIds.has(block.id);
    if (isSelected) {
      return `${baseClasses} border-2 border-blue-400`;
    }
    
    // Apply colored left borders based on change type
    switch (block.changeType) {
      case 'verified':
        return `${baseClasses} border-l-4 border-yellow-500 hover:bg-yellow-50/30`;
      case 'ai_suggested':
        return `${baseClasses} border-l-4 border-blue-500 hover:bg-blue-50/30`;
      case 'ai_applied':
        return `${baseClasses} border-l-4 border-purple-500 hover:bg-purple-50/30`;
      case 'modified':
        return `${baseClasses} border-l-4 border-green-500 hover:bg-green-50/30`;
      case 'rejected':
        return `${baseClasses} border-l-4 border-red-500 hover:bg-red-50/30`;
      default:
        return `${baseClasses} hover:bg-neutral-50`;
    }
  };

  const getBlockStyles = (type: BlockType) => {
    switch (type) {
      case 'heading1':
        return 'text-2xl font-semibold text-neutral-900 leading-tight';
      case 'heading2':
        return 'text-xl font-semibold text-neutral-900 leading-tight';
      case 'heading3':
        return 'text-lg font-semibold text-neutral-900 leading-tight';
      case 'bullet':
        return 'text-sm text-neutral-700 ml-6 list-disc leading-snug';
      case 'numbered':
        return 'text-sm text-neutral-700 ml-6 list-decimal leading-snug';
      case 'callout':
        return 'text-sm text-neutral-700 bg-blue-50 border-l-4 border-blue-400 p-3 leading-snug';
      case 'quote':
        return 'text-sm text-neutral-600 italic border-l-4 border-neutral-300 pl-4 leading-snug';
      default:
        return 'text-sm text-neutral-700 leading-snug';
    }
  };

  const renderBlock = (block: Block) => {
    const isHovered = hoveredBlock === block.id;
    // Show Accept/Reject buttons only for AI suggestions (not verification - those are auto-accepted)
    const showAISuggestionButtons = !!block.aiSuggestion;

    return (
      <div
        key={block.id}
        ref={(el) => {
          if (el) blockRefs.current.set(block.id, el);
          else blockRefs.current.delete(block.id);
        }}
        className={getBlockClassName(block)}
        onMouseEnter={() => setHoveredBlock(block.id)}
        onMouseLeave={() => setHoveredBlock(null)}
        onClick={(e) => handleBlockClick(block.id, e)}
      >
        {/* Yellow Flag for Suggestions */}
        {block.aiSuggestion && (
          <div className="absolute left-0 top-0 bottom-0 w-1 bg-yellow-400"></div>
        )}

        {/* Left Gutter - Drag Handle / Selection */}
        <div className={`absolute left-4 top-1/2 -translate-y-1/2 transition-opacity ${
          selectedBlockIds.has(block.id) ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
        }`}>
          {selectedBlockIds.has(block.id) ? (
            <div className="p-1 bg-blue-500 rounded">
              <Check className="w-4 h-4 text-white" />
            </div>
          ) : (
            <button 
              className="p-1 hover:bg-neutral-200 rounded cursor-pointer"
              onClick={(e) => {
                e.stopPropagation();
                handleBlockClick(block.id, e);
              }}
              title="Click to select block (Shift/Cmd for multi-select)"
            >
            <GripVertical className="w-4 h-4 text-neutral-400" />
          </button>
          )}
        </div>

        {/* Block Content */}
        <div className={block.changeType === 'removed' ? 'line-through opacity-50' : ''}>
          {block.type === 'bullet' || block.type === 'numbered' ? (
            <li className={getBlockStyles(block.type)}>
              <input
                value={block.content}
                onChange={(e) => handleInputChange(block.id, e.target.value)}
                className="w-full bg-transparent outline-none"
              />
            </li>
          ) : (
            <div className={getBlockStyles(block.type)}>
              <textarea
                value={block.content}
                onChange={(e) => {
                  handleInputChange(block.id, e.target.value);
                  // Auto-resize
                  e.target.style.height = 'auto';
                  e.target.style.height = e.target.scrollHeight + 'px';
                }}
                onFocus={(e) => {
                  // Set initial height on focus
                  e.target.style.height = 'auto';
                  e.target.style.height = e.target.scrollHeight + 'px';
                }}
                className="w-full bg-transparent outline-none resize-none overflow-hidden"
                rows={1}
                style={{ minHeight: '1.5rem' }}
              />
            </div>
          )}
        </div>

        {/* Suggestion card removed - details shown in right panel instead */}

        {/* Right Gutter - Comment & Menu */}
        <div className="absolute right-4 top-1/2 -translate-y-1/2 flex items-center gap-2">
          {block.commentCount > 0 && (
            <button
              onClick={() => onCommentClick(block.id)}
              className="relative p-1 hover:bg-neutral-200 rounded"
            >
              <MessageSquare className="w-4 h-4 text-blue-600" />
              <span className="absolute -top-1 -right-1 bg-blue-600 text-white text-xs rounded-full w-4 h-4 flex items-center justify-center">
                {block.commentCount}
              </span>
            </button>
          )}

          {isHovered && (
            <>
              <button
                onClick={() => onCommentClick(block.id)}
                className="p-1 hover:bg-neutral-200 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                title="Add comment"
              >
                <MessageSquarePlus className="w-4 h-4 text-neutral-500" />
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleBlockClick(block.id, e);
                }}
                className={`p-1 hover:bg-blue-100 rounded opacity-0 group-hover:opacity-100 transition-opacity ${
                  selectedBlockIds.has(block.id) ? 'bg-blue-100' : ''
                }`}
                title="Ask RiskGPT to improve this block"
              >
                <Sparkles className={`w-4 h-4 ${selectedBlockIds.has(block.id) ? 'text-blue-600' : 'text-neutral-500'}`} />
              </button>
              <button className="p-1 hover:bg-neutral-200 rounded opacity-0 group-hover:opacity-100 transition-opacity">
                <MoreVertical className="w-4 h-4 text-neutral-500" />
              </button>
            </>
          )}
        </div>



        {/* Add Block Button */}
        {isHovered && (
          <div className="absolute left-1/2 -translate-x-1/2 -bottom-3 opacity-0 group-hover:opacity-100 transition-opacity">
            <button className="p-1 bg-white border border-neutral-300 rounded-full hover:bg-neutral-100 shadow-sm">
              <Plus className="w-3 h-3 text-neutral-500" />
            </button>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="relative h-full overflow-y-auto bg-white" ref={editorRef}>
      {/* Floating Toolbar */}
      {selectedText && (
        <div
          className="fixed z-50 flex items-center gap-1 bg-neutral-900 text-white rounded-lg shadow-lg p-1"
          style={{ left: selectionPosition.x, top: selectionPosition.y }}
        >
          <button className="p-2 hover:bg-neutral-700 rounded">
            <Bold className="w-4 h-4" />
          </button>
          <button className="p-2 hover:bg-neutral-700 rounded">
            <Italic className="w-4 h-4" />
          </button>
          <button className="p-2 hover:bg-neutral-700 rounded">
            <Highlighter className="w-4 h-4" />
          </button>
          <button className="p-2 hover:bg-neutral-700 rounded">
            <LinkIcon className="w-4 h-4" />
          </button>
          <div className="w-px h-5 bg-neutral-600 mx-1" />
          <button className="p-2 hover:bg-neutral-700 rounded">
            <MessageSquarePlus className="w-4 h-4" />
          </button>
          <button className="p-2 hover:bg-neutral-700 rounded">
            <Sparkles className="w-4 h-4" />
          </button>
        </div>
      )}


      {/* Track Changes Legend */}
      <div className="sticky top-0 z-20 bg-white border-b border-neutral-200 px-4 py-3 shadow-sm">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4 text-xs">
            <span className="font-semibold text-neutral-700">Track Changes:</span>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 border-l-4 border-yellow-500 bg-white"></div>
              <span className="text-neutral-600">Verification</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 border-l-4 border-blue-500 bg-white"></div>
              <span className="text-neutral-600">AI Suggestion</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 border-l-4 border-purple-500 bg-white"></div>
              <span className="text-neutral-600">AI Applied</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 border-l-4 border-green-500 bg-white"></div>
              <span className="text-neutral-600">User Edit</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 border-l-4 border-red-500 bg-white"></div>
              <span className="text-neutral-600">Rejected</span>
            </div>
          </div>
          {onSave && (
            <button className="px-3 py-1 text-xs bg-neutral-900 text-white rounded hover:bg-neutral-800" onClick={handleSave}>
              Save changes
            </button>
          )}
        </div>
      </div>

      {/* Editor Content */}
      <div className="w-full py-8">
        <div className="flex justify-between items-center mb-4 gap-2 px-2">
          <div>
            {blocks.filter(b => b.aiSuggestion || b.suggestion).length > 0 && (
              <div className="px-3 py-2 bg-blue-100 border border-blue-300 rounded text-sm text-blue-900">
                <Sparkles className="w-4 h-4 inline mr-2" />
                {blocks.filter(b => b.aiSuggestion || b.suggestion).length} block{blocks.filter(b => b.aiSuggestion || b.suggestion).length > 1 ? 's' : ''} with suggestions - Review below
              </div>
            )}
          </div>
        </div>
        <div className="max-w-4xl mx-auto">
        {blocks.map(renderBlock)}
        </div>
      </div>

      {/* Slash Menu */}
      {showSlashMenu && (
        <div
          className="fixed z-50 bg-white border border-neutral-200 rounded-lg shadow-lg py-2 w-64"
          style={{ left: slashMenuPosition.x, top: slashMenuPosition.y }}
        >
          <button className="w-full px-4 py-2 text-left text-sm hover:bg-neutral-100 text-neutral-900">
            Heading 1
          </button>
          <button className="w-full px-4 py-2 text-left text-sm hover:bg-neutral-100 text-neutral-900">
            Heading 2
          </button>
          <button className="w-full px-4 py-2 text-left text-sm hover:bg-neutral-100 text-neutral-900">
            Heading 3
          </button>
          <div className="border-t border-neutral-200 my-1" />
          <button className="w-full px-4 py-2 text-left text-sm hover:bg-neutral-100 text-neutral-900">
            Bulleted List
          </button>
          <button className="w-full px-4 py-2 text-left text-sm hover:bg-neutral-100 text-neutral-900">
            Numbered List
          </button>
          <button className="w-full px-4 py-2 text-left text-sm hover:bg-neutral-100 text-neutral-900">
            Callout
          </button>
          <button className="w-full px-4 py-2 text-left text-sm hover:bg-neutral-100 text-neutral-900">
            Quote
          </button>
          <div className="border-t border-neutral-200 my-1" />
          <button className="w-full px-4 py-2 text-left text-sm hover:bg-neutral-100 text-neutral-700 flex items-center gap-2">
            <Sparkles className="w-4 h-4" />
            Ask AI to rewrite
          </button>
        </div>
      )}
    </div>
  );
}
