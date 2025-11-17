import { useState } from 'react';
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
  Plus
} from 'lucide-react';

interface MarkdownViewerProps {
  content: string;
  title: string;
  onCommentClick: (blockId: string) => void;
}

export function MarkdownViewer({ content, title, onCommentClick }: MarkdownViewerProps) {
  const [hoveredLineIndex, setHoveredLineIndex] = useState<number | null>(null);
  const [selectedText, setSelectedText] = useState(false);
  const [selectionPosition, setSelectionPosition] = useState({ x: 0, y: 0 });

  const lines = content.split('\n');

  const getLineClassName = (line: string) => {
    if (line.startsWith('# ')) {
      return 'text-2xl font-semibold text-neutral-900 mt-4 mb-2';
    } else if (line.startsWith('## ')) {
      return 'text-xl font-semibold text-neutral-900 mt-3 mb-2';
    } else if (line.startsWith('### ')) {
      return 'text-lg font-semibold text-neutral-900 mt-2 mb-1';
    } else if (line.startsWith('- ')) {
      return 'text-sm text-neutral-700 ml-6';
    } else if (line.startsWith('|')) {
      return 'text-sm text-neutral-600 font-mono';
    } else if (line.trim() === '') {
      return '';
    } else {
      return 'text-sm text-neutral-700 leading-relaxed';
    }
  };

  const renderLine = (line: string, index: number) => {
    const isHovered = hoveredLineIndex === index;
    const lineClass = getLineClassName(line);

    if (line.trim() === '') {
      return <div key={index} className="h-4" />;
    }

    let displayContent = line;
    if (line.startsWith('# ')) displayContent = line.replace('# ', '');
    else if (line.startsWith('## ')) displayContent = line.replace('## ', '');
    else if (line.startsWith('### ')) displayContent = line.replace('### ', '');
    else if (line.startsWith('- ')) displayContent = line.replace('- ', '');

    return (
      <div
        key={index}
        className="relative group px-16 py-2 rounded transition-all hover:bg-neutral-50"
        onMouseEnter={() => setHoveredLineIndex(index)}
        onMouseLeave={() => setHoveredLineIndex(null)}
      >
        {/* Left Gutter - Drag Handle */}
        <div className="absolute left-4 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
          <button className="p-1 hover:bg-neutral-200 rounded cursor-grab">
            <GripVertical className="w-4 h-4 text-neutral-400" />
          </button>
        </div>

        {/* Line Content */}
        {line.startsWith('- ') ? (
          <li className={lineClass}>{displayContent}</li>
        ) : (
          <div className={lineClass}>{displayContent}</div>
        )}

        {/* Right Gutter - Comment & Menu */}
        {isHovered && (
          <div className="absolute right-4 top-1/2 -translate-y-1/2 flex items-center gap-2">
            <button
              onClick={() => onCommentClick(`line-${index}`)}
              className="p-1 hover:bg-neutral-200 rounded opacity-0 group-hover:opacity-100 transition-opacity"
            >
              <MessageSquarePlus className="w-4 h-4 text-neutral-500" />
            </button>
            <button className="p-1 hover:bg-neutral-200 rounded opacity-0 group-hover:opacity-100 transition-opacity">
              <MoreVertical className="w-4 h-4 text-neutral-500" />
            </button>
          </div>
        )}

        {/* Add Line Button */}
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
    <div className="relative h-full overflow-y-auto bg-white">
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

      {/* Header */}
      <div className="border-b border-neutral-200 px-6 py-3 sticky top-0 bg-white z-10">
        <h2 className="text-neutral-900">{title}</h2>
        <p className="text-neutral-500 text-xs">Markdown Document</p>
      </div>

      {/* Content */}
      <div className="max-w-4xl mx-auto py-8">
        {lines.map((line, index) => renderLine(line, index))}
      </div>
    </div>
  );
}
