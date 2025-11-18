import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  GripVertical, 
  MessageSquare, 
  Bold, 
  Italic, 
  Highlighter, 
  Link as LinkIcon, 
  Plus,
  Check,
  X as XIcon,
  AlertCircle,
  Undo,
  Redo,
  Save
} from 'lucide-react';
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  verticalListSortingStrategy,
  useSortable,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { motion, AnimatePresence } from 'framer-motion';
import { BlockMetadata, VerificationSuggestion, RiskGPTSuggestion, askRiskGPT } from '@/lib/api';
import { activityLogger } from '@/utils/activityLogger';
import { SlashCommandMenu } from './editor/SlashCommandMenu';
import { ContextMenu } from './editor/ContextMenu';
import { useUndoRedo } from '@/hooks/useUndoRedo';
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts';
import { LexicalBlock } from './editor/LexicalBlock';
import { useComments } from '@/hooks/useComments';

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
  formatting?: {
    bold?: boolean;
    italic?: boolean;
    has_bold?: boolean;
    has_italic?: boolean;
    has_highlight?: boolean;
    alignment?: 'left' | 'center' | 'right';
    size?: 'small' | 'normal' | 'large';
  };
  indent_level?: number;
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
      const content = trimmed.replace(/^###\s+/, '');
      blocks.push({ id: `b${blocks.length + 1}`, type: 'heading3', content: markdownToHtml(content), changeType: 'none', commentCount: 0, changeHistory: [] });
    } else if (trimmed.startsWith('## ')) {
      const content = trimmed.replace(/^##\s+/, '');
      blocks.push({ id: `b${blocks.length + 1}`, type: 'heading2', content: markdownToHtml(content), changeType: 'none', commentCount: 0, changeHistory: [] });
    } else if (trimmed.startsWith('# ')) {
      const content = trimmed.replace(/^#\s+/, '');
      blocks.push({ id: `b${blocks.length + 1}`, type: 'heading1', content: markdownToHtml(content), changeType: 'none', commentCount: 0, changeHistory: [] });
    } else if (trimmed.startsWith('- ')) {
      const content = trimmed.replace(/^-\s+/, '');
      blocks.push({ id: `b${blocks.length + 1}`, type: 'bullet', content: markdownToHtml(content), changeType: 'none', commentCount: 0, changeHistory: [] });
    } else if (trimmed.match(/^\d+\.\s+/)) {
      const content = trimmed.replace(/^\d+\.\s+/, '');
      blocks.push({ id: `b${blocks.length + 1}`, type: 'numbered', content: markdownToHtml(content), changeType: 'none', commentCount: 0, changeHistory: [] });
    } else if (trimmed.startsWith('> ')) {
      const content = trimmed.replace(/^>\s+/, '');
      blocks.push({ id: `b${blocks.length + 1}`, type: 'quote', content: markdownToHtml(content), changeType: 'none', commentCount: 0, changeHistory: [] });
    } else if (trimmed.length === 0) {
      blocks.push({ id: `b${blocks.length + 1}`, type: 'paragraph', content: '', changeType: 'none', commentCount: 0, changeHistory: [] });
    } else {
      blocks.push({ id: `b${blocks.length + 1}`, type: 'paragraph', content: markdownToHtml(trimmed), changeType: 'none', commentCount: 0, changeHistory: [] });
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
    
    // Map heading type with level to specific heading type
    let blockType: BlockType;
    if (meta.type === 'heading' && meta.level) {
      blockType = `heading${meta.level}` as BlockType; // heading1, heading2, heading3
    } else {
      blockType = (meta.type as BlockType) || 'paragraph';
    }
    
    // Check if this block has a suggestion
    const suggestion = suggestions.find(s => s.block_id === blockId);
    
    // Convert markdown content to HTML for contentEditable
    const htmlContent = markdownToHtml(meta.content);
    
    // Determine change type based on suggestion
    const changeType: ChangeType = suggestion ? 'verified' : 'none';
    
    // Initialize change history with verification suggestion if present
    const changeHistory: ChangeRecord[] = suggestion ? [{
      timestamp: new Date().toISOString(),
      type: 'verified',
      original: markdownToHtml(suggestion.original),
      modified: markdownToHtml(suggestion.suggested),
      reason: suggestion.reason,
      user: 'system'
    }] : [];
    
    // Convert suggestion content to HTML as well
    const htmlSuggestion = suggestion ? {
      ...suggestion,
      original: markdownToHtml(suggestion.original),
      suggested: markdownToHtml(suggestion.suggested)
    } : undefined;
    
    blocks.push({
      id: blockId,
      type: blockType,
      content: htmlContent,  // Use HTML content for contentEditable
      changeType,
      commentCount: 0,
      suggestion: htmlSuggestion,
      changeHistory,
      formatting: (meta as any).formatting,  // Pass formatting metadata
      indent_level: (meta as any).indent_level  // Pass indent level
    });
  });
  
  return blocks;
}

// Helper: Convert HTML to plain text (strip tags but preserve formatting markers)
function htmlToPlainText(html: string): string {
  console.log('[htmlToPlainText] Input HTML:', html);
  
  // Create a temporary div to parse HTML
  const temp = document.createElement('div');
  temp.innerHTML = html;
  
  // Convert common HTML tags to markdown/plain text
  temp.querySelectorAll('strong, b').forEach(el => {
    const boldText = `**${el.textContent}**`;
    console.log('[htmlToPlainText] Converting bold:', el.textContent, '→', boldText);
    el.replaceWith(boldText);
  });
  temp.querySelectorAll('em, i').forEach(el => {
    const italicText = `*${el.textContent}*`;
    console.log('[htmlToPlainText] Converting italic:', el.textContent, '→', italicText);
    el.replaceWith(italicText);
  });
  temp.querySelectorAll('u').forEach(el => {
    el.replaceWith(el.textContent || '');
  });
  temp.querySelectorAll('br').forEach(el => {
    el.replaceWith('\n');
  });
  
  const result = temp.textContent || '';
  console.log('[htmlToPlainText] Output:', result);
  return result;
}

function markdownToHtml(markdown: string): string {
  console.log('[markdownToHtml] Input markdown:', markdown);
  let html = markdown;
  
  // IMPORTANT: Process bold BEFORE italic to avoid conflicts
  // Bold: **text** or __text__ -> <strong>text</strong>
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');
  
  // Italic: *text* or _text_ -> <em>text</em>
  // Match single * or _ that are NOT part of bold markers
  html = html.replace(/(?<![*_])\*([^*]+?)\*(?![*_])/g, '<em>$1</em>');
  html = html.replace(/(?<![*_])_([^_]+?)_(?![*_])/g, '<em>$1</em>');
  
  console.log('[markdownToHtml] Output HTML:', html);
  return html;
}

function blocksToMarkdown(blocks: Block[]): string {
  const lines: string[] = [];
  for (const b of blocks) {
    // Convert HTML content to plain text
    const plainContent = htmlToPlainText(b.content);
    
    // Add markdown syntax based on block type
    let line = plainContent;
    if (b.type === 'heading1') {
      line = `# ${plainContent}`;
    } else if (b.type === 'heading2') {
      line = `## ${plainContent}`;
    } else if (b.type === 'heading3') {
      line = `### ${plainContent}`;
    } else if (b.type === 'bullet_list') {
      line = `- ${plainContent}`;
    } else if (b.type === 'numbered_list') {
      line = `1. ${plainContent}`;
    } else if (b.type === 'quote') {
      line = `> ${plainContent}`;
    }
    
    lines.push(line);
  }
  return lines.join('\n');
}

// Sortable Block Item Component (defined outside to prevent re-renders)
interface SortableBlockItemProps {
  block: Block;
  index: number;
  hoveredBlock: string | null;
  setHoveredBlock: (id: string | null) => void;
  getBlockClassName: (block: Block) => string;
  setContextMenu: (menu: { blockId: string; position: { x: number; y: number } } | null) => void;
  handleInputChange: (blockId: string, value: string, e?: React.FormEvent<HTMLTextAreaElement>) => void;
  blocks: Block[];
  blockRefs: React.MutableRefObject<Map<string, HTMLDivElement>>;
  setBlocks: (blocks: Block[]) => void;
  onCommentClick: (blockId: string) => void;
  selectedBlockIds: Set<string>;
  handleBlockSelect: (blockId: string, event: React.MouseEvent) => void;
  getBlockStyles: (type: BlockType, block?: Block) => string;
}

const SortableBlockItem = React.memo(({ 
  block, 
  index, 
  hoveredBlock, 
  setHoveredBlock, 
  getBlockClassName, 
  setContextMenu,
  handleInputChange,
  blocks,
  blockRefs,
  setBlocks,
  onCommentClick,
  selectedBlockIds,
  handleBlockSelect,
  getBlockStyles
}: SortableBlockItemProps) => {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: block.id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };

  const isHovered = hoveredBlock === block.id;
  const showAISuggestionButtons = !!block.aiSuggestion;

  return (
    <div
      ref={setNodeRef}
      style={style}
      key={block.id}
      className={getBlockClassName(block)}
      onMouseEnter={() => setHoveredBlock(block.id)}
      onMouseLeave={() => setHoveredBlock(null)}
      onContextMenu={(e) => {
        e.preventDefault();
        setContextMenu({ blockId: block.id, position: { x: e.clientX, y: e.clientY } });
      }}
    >
      {/* Yellow Flag for Suggestions */}
      {block.aiSuggestion && (
        <div className="absolute left-0 top-0 bottom-0 w-1 bg-yellow-400"></div>
      )}

      {/* Left Gutter with Drag Handle (like Notion) */}
      {isHovered && (
        <div className="absolute left-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
          {/* Drag Handle - 6 dots for reordering */}
          <div {...attributes} {...listeners} className="cursor-grab active:cursor-grabbing p-1 hover:bg-neutral-200 rounded transition-colors">
            <GripVertical className="w-4 h-4 text-neutral-400" />
          </div>
        </div>
      )}

      {/* Block Content */}
      <div className={`${block.changeType === 'removed' ? 'line-through opacity-50' : ''} select-text`}>
        {block.type === 'bullet' || block.type === 'numbered' ? (
          <LexicalBlock
            block={block}
            onChange={(textContent, htmlContent) => {
              handleInputChange(block.id, htmlContent, null as any);
            }}
            onKeyDown={(e) => {
              // Handle Enter - create new list item
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                const index = blocks.findIndex(b => b.id === block.id);
                const newBlock: Block = {
                  id: `b${Date.now()}`,
                  type: block.type, // Keep same list type
                  content: '',
                  changeType: 'none',
                  commentCount: 0,
                  changeHistory: [],
                };
                setBlocks([...blocks.slice(0, index + 1), newBlock, ...blocks.slice(index + 1)]);
                // Focus new block after render
                setTimeout(() => {
                  const newEl = blockRefs.current.get(newBlock.id);
                  if (newEl) {
                    const contentDiv = newEl.querySelector('.lexical-content-editable') as HTMLElement;
                    contentDiv?.focus();
                  }
                }, 0);
              }
              // Handle Backspace on empty - delete block
              else if (e.key === 'Backspace') {
                const target = e.target as HTMLElement;
                if (target.textContent === '' && blocks.length > 1) {
                  e.preventDefault();
                  setBlocks(blocks.filter(b => b.id !== block.id));
                }
              }
            }}
            className={getBlockStyles(block.type, block)}
          />
        ) : (
          <LexicalBlock
            block={block}
            onChange={(textContent, htmlContent) => {
              handleInputChange(block.id, htmlContent, null as any);
            }}
            onKeyDown={(e) => {
              // Handle Enter - create new block
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                const index = blocks.findIndex(b => b.id === block.id);
                const newBlock: Block = {
                  id: `b${Date.now()}`,
                  type: 'paragraph',
                  content: '',
                  changeType: 'none',
                  commentCount: 0,
                  changeHistory: [],
                };
                setBlocks([...blocks.slice(0, index + 1), newBlock, ...blocks.slice(index + 1)]);
                // Focus new block after render
                setTimeout(() => {
                  const newEl = blockRefs.current.get(newBlock.id);
                  if (newEl) {
                    const contentDiv = newEl.querySelector('.lexical-content-editable') as HTMLElement;
                    contentDiv?.focus();
                  }
                }, 0);
              }
              // Handle Backspace on empty - delete block
              else if (e.key === 'Backspace') {
                const target = e.target as HTMLElement;
                if (target.textContent === '' && blocks.length > 1) {
                  e.preventDefault();
                  setBlocks(blocks.filter(b => b.id !== block.id));
                }
              }
              // Handle Tab to indent
              else if (e.key === 'Tab') {
                e.preventDefault();
                const indent = e.shiftKey ? -1 : 1;
                setBlocks(prev => prev.map(b => 
                  b.id === block.id 
                    ? { ...b, indent_level: Math.max(0, (b.indent_level || 0) + indent) }
                    : b
                ));
              }
            }}
            className={getBlockStyles(block.type, block)}
          />
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

        {(isHovered || selectedBlockIds.has(block.id)) && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleBlockSelect(block.id, e);
            }}
            className={`px-2 py-0.5 rounded text-xs transition-all ${
              selectedBlockIds.has(block.id) 
                ? 'bg-blue-600 text-white font-bold opacity-100' 
                : 'bg-neutral-200 text-neutral-700 hover:bg-neutral-300 opacity-0 group-hover:opacity-100'
            }`}
            title={selectedBlockIds.has(block.id) ? "Block selected (click to deselect)" : "Ask RiskGPT about this block (Cmd/Shift to multi-select)"}
          >
            {selectedBlockIds.has(block.id) ? '✓ Selected' : 'Ask RiskGPT'}
          </button>
        )}
      </div>

      {/* ENHANCED: Add Block Button */}
      {isHovered && (
        <div className="absolute left-1/2 -translate-x-1/2 -bottom-3 opacity-0 group-hover:opacity-100 transition-opacity">
          <button 
            onClick={(e) => {
              e.stopPropagation();
              const index = blocks.findIndex(b => b.id === block.id);
              const newBlock: Block = {
                id: `b${Date.now()}`,
                type: 'paragraph',
                content: '',
                changeType: 'none',
                commentCount: 0,
                changeHistory: [],
              };
              setBlocks([...blocks.slice(0, index + 1), newBlock, ...blocks.slice(index + 1)]);
            }}
            className="p-1 bg-white border border-neutral-300 rounded-full hover:bg-neutral-100 shadow-sm"
          >
            <Plus className="w-3 h-3 text-neutral-500" />
          </button>
        </div>
      )}
    </div>
  );
});

SortableBlockItem.displayName = 'SortableBlockItem';

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
  // Enhanced: Undo/Redo state
  const initialBlocks = (() => {
    if (initialMarkdown && initialMarkdown.trim().length > 0) {
      if (blockMetadata && verificationSuggestions) {
        return parseMarkdownWithMetadata(initialMarkdown, blockMetadata, verificationSuggestions);
      }
      return parseMarkdownToBlocks(initialMarkdown);
    }
    return mockBlocks;
  })();
  
  // ENHANCED: Undo/Redo with history
  const { state: blocks, setState: setBlocks, undo, redo, canUndo, canRedo } = useUndoRedo<Block[]>(initialBlocks);
  
  // Debug logging
  console.log('[BlockEditor] Blocks count:', blocks.length);
  console.log('[BlockEditor] First 3 blocks:', blocks.slice(0, 3));
  
  const [hoveredBlock, setHoveredBlock] = useState<string | null>(null);
  const [showSlashMenu, setShowSlashMenu] = useState(false);
  const [slashMenuPosition, setSlashMenuPosition] = useState({ x: 0, y: 0 });
  const [slashSearchQuery, setSlashSearchQuery] = useState('');
  const [editingBlockId, setEditingBlockId] = useState<string | null>(null);
  const [contextMenu, setContextMenu] = useState<{ blockId: string; position: { x: number; y: number } } | null>(null);
  const editorRef = useRef<HTMLDivElement>(null);
  
  // Track accepted/rejected suggestions for persistence
  const [acceptedSuggestions, setAcceptedSuggestions] = useState<string[]>([]);
  const [rejectedSuggestions, setRejectedSuggestions] = useState<string[]>([]);
  
  // NEW: Fetch comment counts for blocks
  const { commentCounts } = useComments(fileId || null);
  
  // NEW: Block selection for RiskGPT
  const [selectedBlockIds, setSelectedBlockIds] = useState<Set<string>>(new Set());
  const [riskGPTPrompt, setRiskGPTPrompt] = useState('');
  const [isAskingRiskGPT, setIsAskingRiskGPT] = useState(false);
  const blockRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const [lastClickedBlockId, setLastClickedBlockId] = useState<string | null>(null);
  
  // ENHANCED: Drag & drop sensors - only activate on drag handle
  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 8, // Only activate after dragging 8px to avoid interfering with text selection
      },
    }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );
  
  // Auto-save disabled per user request
  // Manual save only via "Save" button
  const isSaving = false;
  const lastSaved = null;
  const saveNow = () => {
    // Manual save via handleSave() called by button
  };

  // ENHANCED: Drag & drop handler
  const handleDragEnd = useCallback((event: DragEndEvent) => {
    const { active, over } = event;
    if (over && active.id !== over.id) {
      setBlocks((items) => {
        const oldIndex = items.findIndex((item) => item.id === active.id);
        const newIndex = items.findIndex((item) => item.id === over.id);
        return arrayMove(items, oldIndex, newIndex);
      });
    }
  }, [setBlocks]);

  // ENHANCED: Slash command handler
  const handleSlashCommand = useCallback((position: { x: number; y: number }) => {
    setSlashMenuPosition(position);
    setShowSlashMenu(true);
  }, []);

  const handleSlashSelect = useCallback((blockType: any) => {
    if (blockType === 'ai') {
      activityLogger.info('AI assistant requested');
    } else {
      const selectedId = Array.from(selectedBlockIds)[0];
      if (selectedId) {
        setBlocks((prevBlocks) =>
          prevBlocks.map((block) =>
            block.id === selectedId ? { ...block, type: blockType } : block
          )
        );
      }
    }
    setShowSlashMenu(false);
    setSlashSearchQuery('');
  }, [selectedBlockIds, setBlocks]);

  // ENHANCED: Keyboard shortcuts
  useKeyboardShortcuts({
    'cmd+z': undo,
    'ctrl+z': undo,
    'cmd+shift+z': redo,
    'ctrl+shift+z': redo,
    'cmd+s': (e) => {
      e.preventDefault();
      handleSave();
    },
    'ctrl+s': (e) => {
      e.preventDefault();
      handleSave();
    },
  });

  useEffect(() => {
    console.log('[BlockEditor] useEffect triggered:', {
      hasInitialMarkdown: !!initialMarkdown,
      hasBlockMetadata: !!blockMetadata,
      hasSuggestions: !!verificationSuggestions,
      markdownLength: initialMarkdown?.length,
      metadataCount: blockMetadata?.length
    });
    
    if (initialMarkdown !== undefined && initialMarkdown.trim().length > 0) {
      if (blockMetadata && blockMetadata.length > 0 && verificationSuggestions) {
        // Parse blocks with metadata and suggestions
        const blocksWithSuggestions = parseMarkdownWithMetadata(initialMarkdown, blockMetadata, verificationSuggestions);
        console.log('[BlockEditor] Parsed blocks with metadata:', blocksWithSuggestions.length);
        
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
        
        console.log('[BlockEditor] Setting blocks with auto-accept:', blocksWithAutoAccept.length);
        setBlocks(blocksWithAutoAccept);
      } else {
        const parsed = parseMarkdownToBlocks(initialMarkdown || '');
        console.log('[BlockEditor] Setting blocks from markdown:', parsed.length);
        setBlocks(parsed);
      }
    }
  }, [initialMarkdown, blockMetadata, verificationSuggestions, setBlocks]);

  // No longer needed - using contentEditable divs instead of textareas

  // Notify parent when selected blocks change
  useEffect(() => {
    if (onSelectedBlocksChange && blockMetadata) {
      // NEW FIX: Send current edited content instead of old metadata
      const selectedBlocks = blockMetadata
        .filter(b => selectedBlockIds.has(b.id))
        .map(meta => {
          // Find the current block content from editor state
          const currentBlock = blocks.find(b => b.id === meta.id);
          if (currentBlock) {
            // Use the current edited content from the editor
            return {
              ...meta,
              content: htmlToPlainText(currentBlock.content)
            };
          }
          return meta;
        });
      onSelectedBlocksChange(selectedBlocks);
    }
  }, [selectedBlockIds, blockMetadata, onSelectedBlocksChange, blocks]);

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

  // Update blocks with comment counts
  useEffect(() => {
    if (commentCounts && Object.keys(commentCounts).length > 0) {
      setBlocks(prevBlocks => 
        prevBlocks.map(block => ({
          ...block,
          commentCount: commentCounts[block.id] || 0
        }))
      );
    }
  }, [commentCounts]);

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
            // NEW FIX: Send current edited content instead of old metadata
            const updatedMetadata = {
              ...metadata,
              content: htmlToPlainText(block.content)
            };
            onSelectedBlocksChange([updatedMetadata]);
          }
        }
      }
    };
    
    // NEW: Expose clear selection for Clear All button
    (window as any).__blockEditorClearSelection = () => {
      console.log('[BlockEditor] Clearing all block selections');
      setSelectedBlockIds(new Set());
      if (onSelectedBlocksChange) {
        onSelectedBlocksChange([]);
      }
    };
    
    // NEW: Expose deselect single block
    (window as any).__blockEditorDeselectBlock = (blockId: string) => {
      console.log('[BlockEditor] Deselecting block:', blockId);
      setSelectedBlockIds(prev => {
        const newSet = new Set(prev);
        newSet.delete(blockId);
        
        // Update parent with new selection
        if (onSelectedBlocksChange && blockMetadata) {
          const selectedBlocks = blockMetadata
            .filter(b => newSet.has(b.id))
            .map(meta => {
              const currentBlock = blocks.find(b => b.id === meta.id);
              if (currentBlock) {
                return {
                  ...meta,
                  content: htmlToPlainText(currentBlock.content)
                };
              }
              return meta;
            });
          onSelectedBlocksChange(selectedBlocks);
        }
        
        return newSet;
      });
    };
    
    return () => {
      delete (window as any).__blockEditorAcceptSuggestion;
      delete (window as any).__blockEditorRejectSuggestion;
      delete (window as any).__blockEditorSelectBlock;
      delete (window as any).__blockEditorClearSelection;
      delete (window as any).__blockEditorDeselectBlock;
    };
  }, [onAcceptSuggestion, onRejectSuggestion, blocks, blockMetadata, onSelectedBlocksChange]);

  const handleAcceptChange = (blockId: string) => {
    setBlocks(blocks.map(b => 
      b.id === blockId ? { ...b, changeType: 'none' } : b
    ));
  };

  const handleRejectChange = (blockId: string) => {
    setBlocks(blocks.filter(b => b.id !== blockId));
  };

  const handleInputChange = (blockId: string, value: string, e?: React.FormEvent<HTMLTextAreaElement>) => {
    // ENHANCED: Detect slash command
    if (value.startsWith('/') && e) {
      const target = e.target as HTMLTextAreaElement;
      const rect = target.getBoundingClientRect();
      setSlashMenuPosition({ x: rect.left, y: rect.bottom + 5 });
      setSlashSearchQuery(value.slice(1)); // Remove the /
      setShowSlashMenu(true);
      setSelectedBlockIds(new Set([blockId])); // Select this block for slash command
    } else {
      setShowSlashMenu(false);
    }
    
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

  // NEW: RiskGPT handlers (Notion-style range selection)
  const handleBlockSelect = (blockId: string, event: React.MouseEvent) => {
    console.log('[BlockEditor] Block selected for RiskGPT:', blockId);
    activityLogger.blockSelected(blockId);
    
    // If this block has a suggestion, notify parent to highlight it in left panel
    const clickedBlock = blocks.find(b => b.id === blockId);
    if (clickedBlock?.aiSuggestion && onBlockWithSuggestionClick) {
      console.log('[BlockEditor] Block with suggestion clicked, notifying parent:', blockId);
      onBlockWithSuggestionClick(blockId);
    }
    
    if (event.shiftKey && lastClickedBlockId) {
      // SHIFT+CLICK: Range selection (like Notion)
      const currentIndex = blocks.findIndex(b => b.id === blockId);
      const lastIndex = blocks.findIndex(b => b.id === lastClickedBlockId);
      
      if (currentIndex !== -1 && lastIndex !== -1) {
        const start = Math.min(currentIndex, lastIndex);
        const end = Math.max(currentIndex, lastIndex);
        const rangeIds = blocks.slice(start, end + 1).map(b => b.id);
        
        setSelectedBlockIds(new Set(rangeIds));
        console.log('[BlockEditor] Range select:', rangeIds.length, 'blocks');
      }
    } else if (event.metaKey || event.ctrlKey) {
      // CMD/CTRL+CLICK: Add/remove from selection
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
      setLastClickedBlockId(blockId);
    } else {
      // REGULAR CLICK: Toggle single selection
      setSelectedBlockIds(prev => {
        const newSet = new Set(prev);
        if (newSet.has(blockId)) {
          newSet.delete(blockId); // Deselect if already selected
        } else {
          newSet.add(blockId); // Select if not selected
        }
        console.log('[BlockEditor] Toggle select, new selection:', Array.from(newSet));
        return newSet;
      });
      setLastClickedBlockId(blockId);
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
    
    // Update block metadata with current content and type (convert HTML to plain text)
    const updatedBlockMetadata = blockMetadata.map(meta => {
      const block = blocks.find(b => b.id === meta.id);
      if (block) {
        // Map block type to metadata format
        let metaType = meta.type;
        let metaLevel = meta.level;
        
        if (block.type === 'heading1') {
          metaType = 'heading';
          metaLevel = 1;
        } else if (block.type === 'heading2') {
          metaType = 'heading';
          metaLevel = 2;
        } else if (block.type === 'heading3') {
          metaType = 'heading';
          metaLevel = 3;
        } else if (block.type === 'bullet' || block.type === 'bullet_list') {
          metaType = 'list_item';
          metaLevel = undefined;
        } else if (block.type === 'numbered' || block.type === 'numbered_list') {
          metaType = 'list_item';
          metaLevel = undefined;
        } else if (block.type === 'quote') {
          metaType = 'quote';
          metaLevel = undefined;
        } else {
          metaType = 'paragraph';
          metaLevel = undefined;
        }
        
        return {
          ...meta,
          type: metaType,
          level: metaLevel,
          content: htmlToPlainText(block.content)
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
    const baseClasses = 'relative group px-16 py-0.5 rounded transition-all select-text';
    
    // Check if block is selected for RiskGPT
    const isSelected = selectedBlockIds.has(block.id);
    const selectedClass = isSelected ? 'bg-blue-50' : 'bg-white';
    
    // Apply colored left borders based on change type
    switch (block.changeType) {
      case 'verified':
        return `${baseClasses} ${selectedClass} border-l-4 border-yellow-500 hover:bg-yellow-50/30`;
      case 'ai_suggested':
        return `${baseClasses} ${selectedClass} border-l-4 border-blue-500 hover:bg-blue-50/30`;
      case 'ai_applied':
        return `${baseClasses} ${selectedClass} border-l-4 border-purple-500 hover:bg-purple-50/30`;
      case 'modified':
        return `${baseClasses} ${selectedClass} border-l-4 border-green-500 hover:bg-green-50/30`;
      case 'rejected':
        return `${baseClasses} ${selectedClass} border-l-4 border-red-500 hover:bg-red-50/30`;
      default:
        return `${baseClasses} ${selectedClass} hover:bg-neutral-50`;
    }
  };

  const getFormattingStyles = (block: Block): string => {
    const classes: string[] = [];
    const fmt = block.formatting;
    
    if (fmt) {
      // Full block formatting
      if (fmt.bold) classes.push('font-bold');
      if (fmt.italic) classes.push('italic');
      
      // Alignment
      if (fmt.alignment === 'center') classes.push('text-center');
      else if (fmt.alignment === 'right') classes.push('text-right');
      
      // Size
      if (fmt.size === 'small') classes.push('text-xs');
      else if (fmt.size === 'large') classes.push('text-lg');
      
      // Highlighting (visual cue for has_highlight)
      if (fmt.has_highlight) classes.push('bg-yellow-50 border-l-2 border-yellow-400 pl-2');
    }
    
    // Indentation
    if (block.indent_level && block.indent_level > 0) {
      classes.push(`ml-${block.indent_level * 4}`);
    }
    
    return classes.join(' ');
  };

  const getBlockStyles = (type: BlockType, block?: Block) => {
    let baseStyles = '';
    
    switch (type) {
      case 'heading1':
        baseStyles = 'text-2xl font-semibold text-neutral-900 leading-tight';
        break;
      case 'heading2':
        baseStyles = 'text-xl font-semibold text-neutral-900 leading-tight';
        break;
      case 'heading3':
        baseStyles = 'text-lg font-semibold text-neutral-900 leading-tight';
        break;
      case 'bullet':
        baseStyles = 'text-sm text-neutral-700 ml-6 list-disc leading-snug';
        break;
      case 'numbered':
        baseStyles = 'text-sm text-neutral-700 ml-6 list-decimal leading-snug';
        break;
      case 'callout':
        baseStyles = 'text-sm text-neutral-700 bg-blue-50 border-l-4 border-blue-400 p-3 leading-snug';
        break;
      case 'quote':
        baseStyles = 'text-sm text-neutral-600 italic border-l-4 border-neutral-300 pl-4 leading-snug';
        break;
      default:
        baseStyles = 'text-sm text-neutral-700 leading-snug';
    }
    
    // Apply formatting metadata if block provided
    if (block) {
      const formattingStyles = getFormattingStyles(block);
      return `${baseStyles} ${formattingStyles}`.trim();
    }
    
    return baseStyles;
  };


  return (
    <div className="relative h-full overflow-y-auto bg-white" ref={editorRef}>

      {/* Track Changes Legend with Undo/Redo */}
      <div className="sticky top-0 z-20 bg-white border-b border-neutral-200 px-4 py-2 shadow-sm">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            {/* Undo/Redo */}
            <div className="flex items-center gap-1">
              <button
                onClick={undo}
                disabled={!canUndo}
                className="p-1 hover:bg-neutral-100 rounded disabled:opacity-30 disabled:cursor-not-allowed"
                title="Undo (⌘Z)"
              >
                <Undo className="w-4 h-4" />
          </button>
              <button
                onClick={redo}
                disabled={!canRedo}
                className="p-1 hover:bg-neutral-100 rounded disabled:opacity-30 disabled:cursor-not-allowed"
                title="Redo (⌘⇧Z)"
              >
                <Redo className="w-4 h-4" />
          </button>
        </div>

      {/* Track Changes Legend */}
            <div className="flex items-center gap-3 text-xs">
              <span className="font-semibold text-neutral-700">Track:</span>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 border-l-4 border-blue-500 bg-white"></div>
              <span className="text-neutral-600">AI Suggestion</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 border-l-4 border-purple-500 bg-white"></div>
              <span className="text-neutral-600">AI Applied</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 border-l-4 border-red-500 bg-white"></div>
              <span className="text-neutral-600">Rejected</span>
            </div>
          </div>
          </div>

          {/* Save Button */}
          <button
            onClick={handleSave}
            className="px-3 py-1 text-xs bg-neutral-900 text-white rounded hover:bg-neutral-800 flex items-center gap-2"
          >
            <Save className="w-3 h-3" />
            Save (Cmd+S)
            </button>
        </div>
      </div>

      {/* Editor Content */}
      <div className="w-full py-8">
        <div className="max-w-4xl mx-auto">
          {blocks.length === 0 ? (
            <div className="text-center py-12 text-neutral-500">
              <p className="text-lg mb-2">No content to display</p>
              <p className="text-sm">Document may be loading or empty</p>
            </div>
          ) : (
            /* ENHANCED: Drag & Drop Context */
            <DndContext
              sensors={sensors}
              collisionDetection={closestCenter}
              onDragEnd={handleDragEnd}
            >
              <SortableContext items={blocks.map((b) => b.id)} strategy={verticalListSortingStrategy}>
                {blocks.map((block, index) => (
                  <SortableBlockItem 
                    key={block.id} 
                    block={block} 
                    index={index}
                    hoveredBlock={hoveredBlock}
                    setHoveredBlock={setHoveredBlock}
                    getBlockClassName={getBlockClassName}
                    setContextMenu={setContextMenu}
                    handleInputChange={handleInputChange}
                    blocks={blocks}
                    blockRefs={blockRefs}
                    setBlocks={setBlocks}
                    onCommentClick={onCommentClick}
                    selectedBlockIds={selectedBlockIds}
                    handleBlockSelect={handleBlockSelect}
                    getBlockStyles={getBlockStyles}
                  />
                ))}
              </SortableContext>
            </DndContext>
          )}
        </div>
      </div>

      {/* ENHANCED: Slash Command Menu */}
      {showSlashMenu && (
        <SlashCommandMenu
          position={slashMenuPosition}
          searchQuery={slashSearchQuery}
          onSelect={handleSlashSelect}
          onClose={() => setShowSlashMenu(false)}
        />
      )}

      {/* ENHANCED: Context Menu */}
      {contextMenu && (
        <ContextMenu
          position={contextMenu.position}
          onClose={() => setContextMenu(null)}
          onCopy={() => {
            const block = blocks.find((b) => b.id === contextMenu.blockId);
            if (block) navigator.clipboard.writeText(block.content);
          }}
          onDuplicate={() => {
            const index = blocks.findIndex((b) => b.id === contextMenu.blockId);
            const block = blocks[index];
            const newBlock = { ...block, id: `b${Date.now()}` };
            setBlocks([...blocks.slice(0, index + 1), newBlock, ...blocks.slice(index + 1)]);
          }}
          onDelete={() => {
            setBlocks(blocks.filter((b) => b.id !== contextMenu.blockId));
          }}
          onMoveUp={() => {
            const index = blocks.findIndex((b) => b.id === contextMenu.blockId);
            if (index > 0) setBlocks(arrayMove(blocks, index, index - 1));
          }}
          onMoveDown={() => {
            const index = blocks.findIndex((b) => b.id === contextMenu.blockId);
            if (index < blocks.length - 1) setBlocks(arrayMove(blocks, index, index + 1));
          }}
          onTurnInto={() => {
            setShowSlashMenu(true);
            setSelectedBlockIds(new Set([contextMenu.blockId]));
          }}
          onComment={() => onCommentClick(contextMenu.blockId)}
          onAskAI={async () => {
            // ENHANCED: Connect Ask AI to RiskGPT
            if (!fileId) {
              alert('No file ID available');
              return;
            }
            
            const block = blocks.find(b => b.id === contextMenu.blockId);
            if (!block) return;
            
            try {
              activityLogger.info(`Asking RiskGPT to improve block: ${contextMenu.blockId}`);
              const result = await askRiskGPT(
                fileId,
                [contextMenu.blockId],
                `Improve this content: "${block.content.substring(0, 100)}..."`
              );
              
              // Apply suggestions
              if (result.suggestions && result.suggestions.length > 0) {
                setBlocks(prev => prev.map(b => {
                  const suggestion = result.suggestions.find(s => s.block_id === b.id);
                  if (suggestion) {
                    return {
                      ...b,
                      aiSuggestion: suggestion,
                      changeType: 'ai_suggested' as const,
                      changeHistory: [
                        ...b.changeHistory,
                        {
                          timestamp: new Date().toISOString(),
                          type: 'ai_suggested' as const,
                          original: b.content,
                          modified: suggestion.suggested,
                          reason: suggestion.reason,
                          user: 'riskgpt'
                        }
                      ]
                    };
                  }
                  return b;
                }));
              }
            } catch (error) {
              console.error('[BlockEditor] RiskGPT error:', error);
              alert(`Failed to get AI suggestions: ${error}`);
            }
          }}
        />
      )}
    </div>
  );
}
