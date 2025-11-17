import { useState, useRef, useEffect } from 'react';
import type { Block, BlockType } from './types';

interface RichTextBlockProps {
  block: Block;
  onChange: (content: string, formatting?: any) => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
  onSlashCommand: (position: { x: number; y: number }) => void;
  autoFocus?: boolean;
}

export function RichTextBlock({ 
  block, 
  onChange, 
  onKeyDown, 
  onSlashCommand,
  autoFocus 
}: RichTextBlockProps) {
  const [isEditing, setIsEditing] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (autoFocus && contentRef.current) {
      contentRef.current.focus();
    }
  }, [autoFocus]);

  const handleInput = (e: React.FormEvent<HTMLDivElement>) => {
    if (contentRef.current) {
      const text = contentRef.current.textContent || '';
      
      // Check for slash command
      if (text.startsWith('/')) {
        const rect = contentRef.current.getBoundingClientRect();
        onSlashCommand({ x: rect.left, y: rect.bottom + 5 });
      }
      
      onChange(text);
    }
  };

  const handleKeyDownLocal = (e: React.KeyboardEvent<HTMLDivElement>) => {
    // Handle formatting shortcuts
    if (e.metaKey || e.ctrlKey) {
      if (e.key === 'b') {
        e.preventDefault();
        document.execCommand('bold');
      } else if (e.key === 'i') {
        e.preventDefault();
        document.execCommand('italic');
      } else if (e.key === 'u') {
        e.preventDefault();
        document.execCommand('underline');
      }
    }
    
    onKeyDown(e);
  };

  const getBlockStyles = (): string => {
    const baseStyles = 'w-full bg-transparent outline-none';
    const fmt = block.formatting;
    const styles: string[] = [baseStyles];

    // Type-specific styles
    switch (block.type) {
      case 'heading1':
        styles.push('text-2xl font-semibold text-neutral-900 leading-tight');
        break;
      case 'heading2':
        styles.push('text-xl font-semibold text-neutral-900 leading-tight');
        break;
      case 'heading3':
        styles.push('text-lg font-semibold text-neutral-900 leading-tight');
        break;
      case 'bullet':
        styles.push('text-sm text-neutral-700 ml-6 leading-snug');
        break;
      case 'numbered':
        styles.push('text-sm text-neutral-700 ml-6 leading-snug');
        break;
      case 'quote':
        styles.push('text-sm text-neutral-600 italic border-l-4 border-neutral-300 pl-4 leading-snug');
        break;
      case 'code':
        styles.push('text-sm font-mono bg-neutral-100 p-3 rounded leading-snug');
        break;
      case 'callout':
        styles.push('text-sm text-neutral-700 bg-blue-50 border-l-4 border-blue-400 p-3 leading-snug');
        break;
      default:
        styles.push('text-sm text-neutral-700 leading-snug');
    }

    // Formatting styles
    if (fmt) {
      if (fmt.bold) styles.push('font-bold');
      if (fmt.italic) styles.push('italic');
      if (fmt.underline) styles.push('underline');
      if (fmt.strikethrough) styles.push('line-through');
      if (fmt.alignment === 'center') styles.push('text-center');
      else if (fmt.alignment === 'right') styles.push('text-right');
      if (fmt.highlight) styles.push('bg-yellow-200');
    }

    // Indentation
    if (block.indent_level && block.indent_level > 0) {
      styles.push(`ml-${block.indent_level * 4}`);
    }

    return styles.join(' ');
  };

  const renderBlockContent = () => {
    if (block.type === 'checkbox') {
      return (
        <div className="flex items-start gap-2">
          <input
            type="checkbox"
            checked={block.checked || false}
            onChange={(e) => {
              onChange(block.content, { ...block.formatting, checked: e.target.checked });
            }}
            className="mt-1 w-4 h-4 rounded border-neutral-300"
          />
          <div
            ref={contentRef}
            contentEditable
            suppressContentEditableWarning
            onInput={handleInput}
            onKeyDown={handleKeyDownLocal}
            onFocus={() => setIsEditing(true)}
            onBlur={() => setIsEditing(false)}
            className={getBlockStyles()}
          >
            {block.content}
          </div>
        </div>
      );
    }

    return (
      <div
        ref={contentRef}
        contentEditable
        suppressContentEditableWarning
        onInput={handleInput}
        onKeyDown={handleKeyDownLocal}
        onFocus={() => setIsEditing(true)}
        onBlur={() => setIsEditing(false)}
        className={getBlockStyles()}
      >
        {block.content}
      </div>
    );
  };

  return renderBlockContent();
}

