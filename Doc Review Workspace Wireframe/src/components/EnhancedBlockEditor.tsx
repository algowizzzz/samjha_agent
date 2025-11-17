import { useState, useRef, useCallback, useEffect } from 'react';
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
import {
  GripVertical,
  MessageSquare,
  MessageSquarePlus,
  Sparkles,
  Plus,
  Check,
  X as XIcon,
  Undo,
  Redo,
  Save,
  Clock,
} from 'lucide-react';

import type { Block, BlockType } from './editor/types';
import { SlashCommandMenu } from './editor/SlashCommandMenu';
import { FloatingToolbar } from './editor/FloatingToolbar';
import { ContextMenu } from './editor/ContextMenu';
import { RichTextBlock } from './editor/RichTextBlock';
import { useUndoRedo } from '@/hooks/useUndoRedo';
import { useAutoSave } from '@/hooks/useAutoSave';
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts';
import type { BlockMetadata, VerificationSuggestion, RiskGPTSuggestion } from '@/lib/api';
import { activityLogger } from '@/utils/activityLogger';

interface EnhancedBlockEditorProps {
  trackChangesEnabled: boolean;
  onCommentClick: (blockId: string) => void;
  selectedIssueId: string | null;
  initialMarkdown?: string;
  blockMetadata?: BlockMetadata[];
  verificationSuggestions?: VerificationSuggestion[];
  onSave?: (data: {
    markdown: string;
    blockMetadata: BlockMetadata[];
    acceptedSuggestions: string[];
    rejectedSuggestions: string[];
  }) => void;
  fileId?: string;
  onSelectedBlocksChange?: (selectedBlocks: BlockMetadata[]) => void;
  aiSuggestions?: Array<{ block_id: string; original: string; suggested: string; reason: string }>;
  onSuggestionsListChange?: (suggestions: Array<{ block_id: string; original: string; suggested: string; reason: string; block_content: string }>) => void;
  selectedSuggestionId?: string | null;
  onBlockWithSuggestionClick?: (blockId: string) => void;
  onAcceptSuggestion?: (blockId: string) => void;
  onRejectSuggestion?: (blockId: string) => void;
}

// Sortable Block Item Component
function SortableBlockItem({
  block,
  blocks,
  setBlocks,
  isSelected,
  isHovered,
  onHover,
  onSelect,
  onChange,
  onKeyDown,
  onSlashCommand,
  onComment,
  onContextMenu,
}: {
  block: Block;
  blocks: Block[];
  setBlocks: (blocks: Block[]) => void;
  isSelected: boolean;
  isHovered: boolean;
  onHover: (id: string | null) => void;
  onSelect: (id: string, event: React.MouseEvent) => void;
  onChange: (id: string, content: string, formatting?: any) => void;
  onKeyDown: (id: string, e: React.KeyboardEvent) => void;
  onSlashCommand: (position: { x: number; y: number }) => void;
  onComment: (id: string) => void;
  onContextMenu: (id: string, position: { x: number; y: number }) => void;
}) {
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

  const getBlockClassName = () => {
    const baseClasses = 'relative group px-16 py-2 rounded transition-all cursor-pointer bg-white';
    
    if (isSelected) {
      return `${baseClasses} ring-2 ring-blue-400 ring-inset`;
    }
    
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

  return (
    <motion.div
      ref={setNodeRef}
      style={style}
      layout
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, x: -100 }}
      className={getBlockClassName()}
      onMouseEnter={() => onHover(block.id)}
      onMouseLeave={() => onHover(null)}
      onClick={(e) => {
        // Don't select if clicking on editable content
        const target = e.target as HTMLElement;
        if (!target.isContentEditable) {
          onSelect(block.id, e);
        }
      }}
      onContextMenu={(e) => {
        e.preventDefault();
        onContextMenu(block.id, { x: e.clientX, y: e.clientY });
      }}
    >
      {/* AI Suggestion Flag */}
      {block.aiSuggestion && (
        <div className="absolute left-0 top-0 bottom-0 w-1 bg-yellow-400" />
      )}

      {/* Left Gutter - Drag Handle */}
      <div
        className={`absolute left-4 top-1/2 -translate-y-1/2 transition-opacity ${
          isSelected || isHovered ? 'opacity-100' : 'opacity-0'
        }`}
      >
        {isSelected ? (
          <div className="p-1 bg-blue-500 rounded">
            <Check className="w-4 h-4 text-white" />
          </div>
        ) : (
          <button
            {...attributes}
            {...listeners}
            className="p-1 hover:bg-neutral-200 rounded cursor-grab active:cursor-grabbing"
            title="Drag to reorder"
          >
            <GripVertical className="w-4 h-4 text-neutral-400" />
          </button>
        )}
      </div>

      {/* Block Content */}
      <RichTextBlock
        block={block}
        onChange={(content, formatting) => onChange(block.id, content, formatting)}
        onKeyDown={(e) => onKeyDown(block.id, e)}
        onSlashCommand={onSlashCommand}
      />

      {/* Right Gutter - Actions */}
      <div className="absolute right-4 top-1/2 -translate-y-1/2 flex items-center gap-2">
        {block.commentCount > 0 && (
          <button
            onClick={() => onComment(block.id)}
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
              onClick={() => onComment(block.id)}
              className="p-1 hover:bg-neutral-200 rounded transition-opacity"
              title="Add comment"
            >
              <MessageSquarePlus className="w-4 h-4 text-neutral-500" />
            </button>
            <button
              className={`p-1 hover:bg-blue-100 rounded transition-opacity ${
                isSelected ? 'bg-blue-100' : ''
              }`}
              title="Ask RiskGPT to improve this block"
            >
              <Sparkles
                className={`w-4 h-4 ${isSelected ? 'text-blue-600' : 'text-neutral-500'}`}
              />
            </button>
          </>
        )}
      </div>

      {/* Add Block Button */}
      {isHovered && (
        <div className="absolute left-1/2 -translate-x-1/2 -bottom-3 transition-opacity">
          <button 
            onClick={(e) => {
              e.stopPropagation();
              // Create new block below this one
              const currentIndex = blocks.findIndex(b => b.id === block.id);
              const newBlock: Block = {
                id: `b${Date.now()}`,
                type: 'paragraph',
                content: '',
                changeType: 'none',
                commentCount: 0,
                changeHistory: [],
              };
              setBlocks([
                ...blocks.slice(0, currentIndex + 1),
                newBlock,
                ...blocks.slice(currentIndex + 1),
              ]);
            }}
            className="p-1 bg-white border border-neutral-300 rounded-full hover:bg-neutral-100 shadow-sm"
          >
            <Plus className="w-3 h-3 text-neutral-500" />
          </button>
        </div>
      )}
    </motion.div>
  );
}

export function EnhancedBlockEditor(props: EnhancedBlockEditorProps) {
  const {
    onCommentClick,
    blockMetadata,
    verificationSuggestions = [],
    onSave,
    fileId,
    onSelectedBlocksChange,
    aiSuggestions,
  } = props;

  // Initialize blocks from props
  const initialBlocks = blockMetadata?.map((meta, index) => ({
    id: meta.id,
    type: (meta.type === 'heading' ? `heading${meta.level || 1}` : meta.type) as BlockType,
    content: meta.content,
    changeType: 'none' as const,
    commentCount: 0,
    changeHistory: [],
    formatting: (meta as any).formatting,
    indent_level: (meta as any).indent_level,
  })) || [];

  // Ensure at least one block exists
  if (initialBlocks.length === 0) {
    initialBlocks.push({
      id: 'b1',
      type: 'paragraph',
      content: 'Start typing or press / for commands...',
      changeType: 'none',
      commentCount: 0,
      changeHistory: [],
    });
  }

  console.log('[EnhancedBlockEditor] Initialized with', initialBlocks.length, 'blocks');

  // State management
  const { state: blocks, setState: setBlocks, undo, redo, canUndo, canRedo } = useUndoRedo<Block[]>(initialBlocks);
  const [selectedBlockIds, setSelectedBlockIds] = useState<Set<string>>(new Set());
  const [hoveredBlock, setHoveredBlock] = useState<string | null>(null);
  const [acceptedSuggestions, setAcceptedSuggestions] = useState<string[]>([]);
  const [rejectedSuggestions, setRejectedSuggestions] = useState<string[]>([]);

  // UI State
  const [showSlashMenu, setShowSlashMenu] = useState(false);
  const [slashMenuPosition, setSlashMenuPosition] = useState({ x: 0, y: 0 });
  const [slashSearchQuery, setSlashSearchQuery] = useState('');
  const [showFloatingToolbar, setShowFloatingToolbar] = useState(false);
  const [toolbarPosition, setToolbarPosition] = useState({ x: 0, y: 0 });
  const [contextMenu, setContextMenu] = useState<{ blockId: string; position: { x: number; y: number } } | null>(null);

  const editorRef = useRef<HTMLDivElement>(null);

  // Auto-save
  const { isSaving, lastSaved, saveNow } = useAutoSave({
    data: blocks,
    onSave: async (data) => {
      if (onSave && blockMetadata) {
        const markdown = data.map(b => b.content).join('\n');
        const updatedMetadata = blockMetadata.map(meta => {
          const block = data.find(b => b.id === meta.id);
          return block ? { ...meta, content: block.content } : meta;
        });
        onSave({
          markdown,
          blockMetadata: updatedMetadata,
          acceptedSuggestions,
          rejectedSuggestions,
        });
      }
    },
    delay: 2000,
  });

  // Drag and drop sensors
  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;

    if (over && active.id !== over.id) {
      setBlocks((items) => {
        const oldIndex = items.findIndex((item) => item.id === active.id);
        const newIndex = items.findIndex((item) => item.id === over.id);
        return arrayMove(items, oldIndex, newIndex);
      });
    }
  };

  // Block operations
  const handleBlockChange = useCallback((id: string, content: string, formatting?: any) => {
    setBlocks((prevBlocks) =>
      prevBlocks.map((block) =>
        block.id === id
          ? { ...block, content, formatting: { ...block.formatting, ...formatting } }
          : block
      )
    );
  }, [setBlocks]);

  const handleBlockSelect = useCallback((id: string, event: React.MouseEvent) => {
    // Allow editing - don't interfere with contentEditable
    const target = event.target as HTMLElement;
    if (target.isContentEditable) {
      return;
    }

    if (event.shiftKey || event.metaKey || event.ctrlKey) {
      setSelectedBlockIds((prev) => {
        const newSet = new Set(prev);
        if (newSet.has(id)) {
          newSet.delete(id);
        } else {
          newSet.add(id);
        }
        return newSet;
      });
    } else {
      setSelectedBlockIds(new Set([id]));
    }
  }, []);

  const handleBlockKeyDown = useCallback((id: string, e: React.KeyboardEvent) => {
    const currentIndex = blocks.findIndex((b) => b.id === id);
    const currentBlock = blocks[currentIndex];

    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      // Create new block below
      const newBlock: Block = {
        id: `b${Date.now()}`,
        type: 'paragraph',
        content: '',
        changeType: 'none',
        commentCount: 0,
        changeHistory: [],
      };
      setBlocks([
        ...blocks.slice(0, currentIndex + 1),
        newBlock,
        ...blocks.slice(currentIndex + 1),
      ]);
    } else if (e.key === 'Backspace' && currentBlock.content === '') {
      e.preventDefault();
      if (blocks.length > 1) {
        setBlocks(blocks.filter((b) => b.id !== id));
        // Focus previous block
        if (currentIndex > 0) {
          const prevBlock = blocks[currentIndex - 1];
          setTimeout(() => {
            const prevElement = document.querySelector(`[data-block-id="${prevBlock.id}"]`);
            if (prevElement) {
              (prevElement as HTMLElement).focus();
            }
          }, 0);
        }
      }
    } else if (e.key === 'Tab') {
      e.preventDefault();
      // Indent/outdent
      const indent = e.shiftKey ? -1 : 1;
      setBlocks((prevBlocks) =>
        prevBlocks.map((block) =>
          block.id === id
            ? {
                ...block,
                indent_level: Math.max(0, (block.indent_level || 0) + indent),
              }
            : block
        )
      );
    }
  }, [blocks, setBlocks]);

  const handleSlashCommand = useCallback((position: { x: number; y: number }) => {
    setSlashMenuPosition(position);
    setShowSlashMenu(true);
  }, []);

  const handleSlashSelect = useCallback((blockType: BlockType | 'ai') => {
    if (blockType === 'ai') {
      // Trigger AI for selected blocks
      activityLogger.info('AI assistant requested');
    } else {
      // Convert current block to selected type
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

  // Text selection for floating toolbar
  useEffect(() => {
    const handleSelectionChange = () => {
      const selection = window.getSelection();
      if (selection && selection.toString().length > 0) {
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        setToolbarPosition({ x: rect.left + rect.width / 2, y: rect.top - 50 });
        setShowFloatingToolbar(true);
      } else {
        setShowFloatingToolbar(false);
      }
    };

    document.addEventListener('selectionchange', handleSelectionChange);
    return () => document.removeEventListener('selectionchange', handleSelectionChange);
  }, []);

  // Keyboard shortcuts
  useKeyboardShortcuts({
    'cmd+z': undo,
    'ctrl+z': undo,
    'cmd+shift+z': redo,
    'ctrl+shift+z': redo,
    'cmd+s': (e) => {
      e.preventDefault();
      saveNow();
    },
    'ctrl+s': (e) => {
      e.preventDefault();
      saveNow();
    },
  });

  return (
    <div className="relative h-full overflow-y-auto bg-white" ref={editorRef} data-enhanced-editor="true">
      {/* Top Bar with Controls */}
      <div className="sticky top-0 z-20 bg-white border-b border-neutral-200 px-4 py-3 shadow-sm">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <button
                onClick={undo}
                disabled={!canUndo}
                className="p-1.5 hover:bg-neutral-100 rounded disabled:opacity-30 disabled:cursor-not-allowed"
                title="Undo (⌘Z)"
              >
                <Undo className="w-4 h-4" />
              </button>
              <button
                onClick={redo}
                disabled={!canRedo}
                className="p-1.5 hover:bg-neutral-100 rounded disabled:opacity-30 disabled:cursor-not-allowed"
                title="Redo (⌘⇧Z)"
              >
                <Redo className="w-4 h-4" />
              </button>
            </div>

            {/* Save Status */}
            <div className="flex items-center gap-2 text-xs text-neutral-600">
              {isSaving ? (
                <>
                  <Clock className="w-3 h-3 animate-spin" />
                  <span>Saving...</span>
                </>
              ) : lastSaved ? (
                <>
                  <Check className="w-3 h-3 text-green-600" />
                  <span>Saved {new Date(lastSaved).toLocaleTimeString()}</span>
                </>
              ) : null}
            </div>
          </div>

          <button
            onClick={saveNow}
            disabled={isSaving}
            className="px-3 py-1.5 text-xs bg-neutral-900 text-white rounded hover:bg-neutral-800 disabled:opacity-50 flex items-center gap-2"
          >
            <Save className="w-3 h-3" />
            Save
          </button>
        </div>
      </div>

      {/* Editor Content */}
      <div className="w-full py-8">
        <div className="max-w-4xl mx-auto">
          {blocks.length === 0 && <div className="text-neutral-500 text-center py-8">No blocks to display</div>}
          {console.log('[EnhancedBlockEditor] Rendering', blocks.length, 'blocks')}
          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            onDragEnd={handleDragEnd}
          >
            <SortableContext items={blocks.map((b) => b.id)} strategy={verticalListSortingStrategy}>
              <AnimatePresence>
                {blocks.map((block) => (
                  <SortableBlockItem
                    key={block.id}
                    block={block}
                    blocks={blocks}
                    setBlocks={setBlocks}
                    isSelected={selectedBlockIds.has(block.id)}
                    isHovered={hoveredBlock === block.id}
                    onHover={setHoveredBlock}
                    onSelect={handleBlockSelect}
                    onChange={handleBlockChange}
                    onKeyDown={handleBlockKeyDown}
                    onSlashCommand={handleSlashCommand}
                    onComment={onCommentClick}
                    onContextMenu={(id, position) => setContextMenu({ blockId: id, position })}
                  />
                ))}
              </AnimatePresence>
            </SortableContext>
          </DndContext>
        </div>
      </div>

      {/* Slash Command Menu */}
      {showSlashMenu && (
        <SlashCommandMenu
          position={slashMenuPosition}
          searchQuery={slashSearchQuery}
          onSelect={handleSlashSelect}
          onClose={() => setShowSlashMenu(false)}
        />
      )}

      {/* Floating Toolbar */}
      {showFloatingToolbar && (
        <FloatingToolbar
          position={toolbarPosition}
          onFormat={(format) => {
            document.execCommand(format);
          }}
          onLink={() => {
            const url = prompt('Enter URL:');
            if (url) document.execCommand('createLink', false, url);
          }}
          onComment={() => {
            const selectedId = Array.from(selectedBlockIds)[0];
            if (selectedId) onCommentClick(selectedId);
          }}
          onAI={() => activityLogger.info('AI requested from toolbar')}
        />
      )}

      {/* Context Menu */}
      {contextMenu && (
        <ContextMenu
          position={contextMenu.position}
          onClose={() => setContextMenu(null)}
          onCopy={() => {
            const block = blocks.find((b) => b.id === contextMenu.blockId);
            if (block) {
              navigator.clipboard.writeText(block.content);
            }
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
            if (index > 0) {
              setBlocks(arrayMove(blocks, index, index - 1));
            }
          }}
          onMoveDown={() => {
            const index = blocks.findIndex((b) => b.id === contextMenu.blockId);
            if (index < blocks.length - 1) {
              setBlocks(arrayMove(blocks, index, index + 1));
            }
          }}
          onTurnInto={() => setShowSlashMenu(true)}
          onComment={() => onCommentClick(contextMenu.blockId)}
          onAskAI={() => activityLogger.info('AI requested from context menu')}
        />
      )}
    </div>
  );
}

