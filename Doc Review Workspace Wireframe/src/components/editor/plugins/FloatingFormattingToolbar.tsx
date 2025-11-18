import { useCallback, useEffect, useRef, useState } from 'react';
import { useLexicalComposerContext } from '@lexical/react/LexicalComposerContext';
import {
  $getSelection,
  $isRangeSelection,
  SELECTION_CHANGE_COMMAND,
  COMMAND_PRIORITY_LOW,
  FORMAT_TEXT_COMMAND,
} from 'lexical';
import { Bold, Italic, Underline, Strikethrough, Code, X } from 'lucide-react';

/**
 * Floating toolbar that appears when text is selected
 * Provides quick access to text formatting options
 */
export function FloatingFormattingToolbar() {
  const [editor] = useLexicalComposerContext();
  const toolbarRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const [formats, setFormats] = useState({
    bold: false,
    italic: false,
    underline: false,
    strikethrough: false,
    code: false,
  });

  const updateToolbar = useCallback(() => {
    const selection = $getSelection();
    
    if (!$isRangeSelection(selection) || selection.isCollapsed()) {
      setIsVisible(false);
      return;
    }

    // Get current formats
    const isBold = selection.hasFormat('bold');
    const isItalic = selection.hasFormat('italic');
    const isUnderline = selection.hasFormat('underline');
    const isStrikethrough = selection.hasFormat('strikethrough');
    const isCode = selection.hasFormat('code');

    setFormats({
      bold: isBold,
      italic: isItalic,
      underline: isUnderline,
      strikethrough: isStrikethrough,
      code: isCode,
    });

    // Calculate position based on selection
    const nativeSelection = window.getSelection();
    if (!nativeSelection || nativeSelection.rangeCount === 0) {
      setIsVisible(false);
      return;
    }

    const range = nativeSelection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    
    // Position toolbar above the selection
    const top = rect.top + window.scrollY - 50;
    const left = rect.left + window.scrollX + rect.width / 2;
    
    setPosition({ top, left });
    setIsVisible(true);
  }, []);

  useEffect(() => {
    return editor.registerCommand(
      SELECTION_CHANGE_COMMAND,
      () => {
        updateToolbar();
        return false;
      },
      COMMAND_PRIORITY_LOW
    );
  }, [editor, updateToolbar]);

  const applyFormat = (format: 'bold' | 'italic' | 'underline' | 'strikethrough' | 'code') => {
    editor.dispatchCommand(FORMAT_TEXT_COMMAND, format);
  };

  const clearFormatting = () => {
    editor.update(() => {
      const selection = $getSelection();
      if ($isRangeSelection(selection)) {
        // Remove all text formats
        const formats: Array<'bold' | 'italic' | 'underline' | 'strikethrough' | 'code'> = [
          'bold',
          'italic',
          'underline',
          'strikethrough',
          'code',
        ];
        formats.forEach((format) => {
          if (selection.hasFormat(format)) {
            selection.formatText(format);
          }
        });
      }
    });
  };

  if (!isVisible) {
    return null;
  }

  return (
    <div
      ref={toolbarRef}
      className="absolute z-50 bg-white border border-neutral-300 rounded-lg shadow-lg p-1 flex items-center gap-0.5"
      style={{
        top: `${position.top}px`,
        left: `${position.left}px`,
        transform: 'translateX(-50%)',
      }}
    >
      <button
        onClick={() => applyFormat('bold')}
        className={`p-2 rounded hover:bg-neutral-100 transition-colors ${
          formats.bold ? 'bg-blue-100 text-blue-700' : 'text-neutral-700'
        }`}
        title="Bold (Cmd+B)"
        type="button"
      >
        <Bold className="w-4 h-4" />
      </button>

      <button
        onClick={() => applyFormat('italic')}
        className={`p-2 rounded hover:bg-neutral-100 transition-colors ${
          formats.italic ? 'bg-blue-100 text-blue-700' : 'text-neutral-700'
        }`}
        title="Italic (Cmd+I)"
        type="button"
      >
        <Italic className="w-4 h-4" />
      </button>

      <button
        onClick={() => applyFormat('underline')}
        className={`p-2 rounded hover:bg-neutral-100 transition-colors ${
          formats.underline ? 'bg-blue-100 text-blue-700' : 'text-neutral-700'
        }`}
        title="Underline (Cmd+U)"
        type="button"
      >
        <Underline className="w-4 h-4" />
      </button>

      <button
        onClick={() => applyFormat('strikethrough')}
        className={`p-2 rounded hover:bg-neutral-100 transition-colors ${
          formats.strikethrough ? 'bg-blue-100 text-blue-700' : 'text-neutral-700'
        }`}
        title="Strikethrough"
        type="button"
      >
        <Strikethrough className="w-4 h-4" />
      </button>

      <button
        onClick={() => applyFormat('code')}
        className={`p-2 rounded hover:bg-neutral-100 transition-colors ${
          formats.code ? 'bg-blue-100 text-blue-700' : 'text-neutral-700'
        }`}
        title="Code"
        type="button"
      >
        <Code className="w-4 h-4" />
      </button>

      <div className="w-px h-6 bg-neutral-300 mx-1" />

      <button
        onClick={clearFormatting}
        className="p-2 rounded hover:bg-neutral-100 text-neutral-700 transition-colors"
        title="Clear formatting"
        type="button"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  );
}

