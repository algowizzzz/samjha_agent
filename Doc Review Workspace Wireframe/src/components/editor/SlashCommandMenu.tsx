import { useState, useEffect, useRef } from 'react';
import { 
  Heading1, Heading2, Heading3, List, ListOrdered, 
  Quote, Code, CheckSquare, Image, Table, 
  Sparkles, Type
} from 'lucide-react';
import type { BlockType } from './types';

interface SlashCommand {
  id: string;
  title: string;
  icon: React.ReactNode;
  description: string;
  blockType: BlockType | 'ai';
  keywords: string[];
}

const commands: SlashCommand[] = [
  {
    id: 'text',
    title: 'Text',
    icon: <Type className="w-4 h-4" />,
    description: 'Plain text paragraph',
    blockType: 'paragraph',
    keywords: ['text', 'paragraph', 'p'],
  },
  {
    id: 'h1',
    title: 'Heading 1',
    icon: <Heading1 className="w-4 h-4" />,
    description: 'Large section heading',
    blockType: 'heading1',
    keywords: ['heading', 'h1', 'title', 'large'],
  },
  {
    id: 'h2',
    title: 'Heading 2',
    icon: <Heading2 className="w-4 h-4" />,
    description: 'Medium section heading',
    blockType: 'heading2',
    keywords: ['heading', 'h2', 'subtitle', 'medium'],
  },
  {
    id: 'h3',
    title: 'Heading 3',
    icon: <Heading3 className="w-4 h-4" />,
    description: 'Small section heading',
    blockType: 'heading3',
    keywords: ['heading', 'h3', 'small'],
  },
  {
    id: 'bullet',
    title: 'Bullet List',
    icon: <List className="w-4 h-4" />,
    description: 'Unordered list',
    blockType: 'bullet',
    keywords: ['bullet', 'list', 'ul', 'unordered'],
  },
  {
    id: 'numbered',
    title: 'Numbered List',
    icon: <ListOrdered className="w-4 h-4" />,
    description: 'Ordered list',
    blockType: 'numbered',
    keywords: ['numbered', 'list', 'ol', 'ordered'],
  },
  {
    id: 'quote',
    title: 'Quote',
    icon: <Quote className="w-4 h-4" />,
    description: 'Blockquote',
    blockType: 'quote',
    keywords: ['quote', 'blockquote', 'citation'],
  },
  {
    id: 'code',
    title: 'Code Block',
    icon: <Code className="w-4 h-4" />,
    description: 'Code with syntax highlighting',
    blockType: 'code',
    keywords: ['code', 'programming', 'snippet'],
  },
  {
    id: 'checkbox',
    title: 'Checkbox',
    icon: <CheckSquare className="w-4 h-4" />,
    description: 'To-do item with checkbox',
    blockType: 'checkbox',
    keywords: ['todo', 'task', 'checkbox', 'check'],
  },
  {
    id: 'table',
    title: 'Table',
    icon: <Table className="w-4 h-4" />,
    description: 'Table with rows and columns',
    blockType: 'table',
    keywords: ['table', 'grid', 'data'],
  },
  {
    id: 'callout',
    title: 'Callout',
    icon: <Image className="w-4 h-4" />,
    description: 'Highlighted callout box',
    blockType: 'callout',
    keywords: ['callout', 'highlight', 'note', 'info'],
  },
  {
    id: 'ai',
    title: 'Ask AI',
    icon: <Sparkles className="w-4 h-4" />,
    description: 'Get AI assistance',
    blockType: 'ai',
    keywords: ['ai', 'assistant', 'help', 'riskgpt'],
  },
];

interface SlashCommandMenuProps {
  position: { x: number; y: number };
  searchQuery: string;
  onSelect: (blockType: BlockType | 'ai') => void;
  onClose: () => void;
}

export function SlashCommandMenu({ position, searchQuery, onSelect, onClose }: SlashCommandMenuProps) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const menuRef = useRef<HTMLDivElement>(null);

  // Filter commands based on search query
  const filteredCommands = searchQuery
    ? commands.filter((cmd) =>
        cmd.keywords.some((keyword) =>
          keyword.toLowerCase().includes(searchQuery.toLowerCase())
        ) ||
        cmd.title.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : commands;

  // Reset selection when filtered commands change
  useEffect(() => {
    setSelectedIndex(0);
  }, [searchQuery]);

  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex((prev) => (prev + 1) % filteredCommands.length);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex((prev) => (prev - 1 + filteredCommands.length) % filteredCommands.length);
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (filteredCommands[selectedIndex]) {
          onSelect(filteredCommands[selectedIndex].blockType);
        }
      } else if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [filteredCommands, selectedIndex, onSelect, onClose]);

  // Auto-scroll selected item into view
  useEffect(() => {
    const selectedElement = menuRef.current?.querySelector(`[data-index="${selectedIndex}"]`);
    if (selectedElement) {
      selectedElement.scrollIntoView({ block: 'nearest' });
    }
  }, [selectedIndex]);

  if (filteredCommands.length === 0) {
    return (
      <>
        {/* Backdrop to close menu */}
        <div
          className="fixed inset-0 z-40"
          onClick={onClose}
        />
        
        <div
          ref={menuRef}
          className="fixed z-50 bg-white border border-neutral-200 rounded-lg shadow-xl py-2 w-72"
          style={{ left: position.x, top: position.y }}
        >
          <div className="px-4 py-2 text-sm text-neutral-500">
            No commands found for "{searchQuery}"
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      {/* Backdrop to close menu */}
      <div
        className="fixed inset-0 z-40"
        onClick={onClose}
      />
      
      <div
        ref={menuRef}
        className="fixed z-50 bg-white border border-neutral-200 rounded-lg shadow-xl py-2 w-72 max-h-80 overflow-y-auto"
        style={{ left: position.x, top: position.y }}
      >
        {filteredCommands.map((command, index) => (
          <button
            key={command.id}
            data-index={index}
            className={`w-full px-4 py-2.5 text-left flex items-start gap-3 transition-colors ${
              index === selectedIndex
                ? 'bg-blue-50 text-blue-900'
                : 'hover:bg-neutral-50 text-neutral-900'
            }`}
            onClick={(e) => {
              e.stopPropagation();
              onSelect(command.blockType);
              onClose();
            }}
            onMouseEnter={() => setSelectedIndex(index)}
          >
            <div className={`mt-0.5 ${index === selectedIndex ? 'text-blue-600' : 'text-neutral-500'}`}>
              {command.icon}
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium">{command.title}</div>
              <div className={`text-xs ${index === selectedIndex ? 'text-blue-700' : 'text-neutral-500'}`}>
                {command.description}
              </div>
            </div>
          </button>
        ))}
      </div>
    </>
  );
}

