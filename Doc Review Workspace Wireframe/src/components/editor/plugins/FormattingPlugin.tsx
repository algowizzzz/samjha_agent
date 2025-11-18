import { useEffect } from 'react';
import { useLexicalComposerContext } from '@lexical/react/LexicalComposerContext';
import {
  $getSelection,
  $isRangeSelection,
  COMMAND_PRIORITY_EDITOR,
  FORMAT_TEXT_COMMAND,
  KEY_DOWN_COMMAND,
} from 'lexical';

/**
 * Plugin to handle keyboard shortcuts for text formatting
 * Supports: Bold (Cmd+B), Italic (Cmd+I), Underline (Cmd+U)
 */
export function FormattingPlugin() {
  const [editor] = useLexicalComposerContext();

  useEffect(() => {
    // Register keyboard shortcuts for formatting
    const removeCommandListeners = [
      // Bold, Italic, Underline: Cmd/Ctrl + B/I/U
      editor.registerCommand(
        KEY_DOWN_COMMAND,
        (event: KeyboardEvent) => {
          const { code, ctrlKey, metaKey, key } = event;
          
          // Bold: Cmd/Ctrl + B
          if ((code === 'KeyB' || key === 'b') && (ctrlKey || metaKey)) {
            event.preventDefault();
            event.stopPropagation();
            editor.update(() => {
              const selection = $getSelection();
              if ($isRangeSelection(selection)) {
                selection.formatText('bold');
              }
            });
            return true;
          }

          // Italic: Cmd/Ctrl + I
          if ((code === 'KeyI' || key === 'i') && (ctrlKey || metaKey)) {
            event.preventDefault();
            event.stopPropagation();
            editor.update(() => {
              const selection = $getSelection();
              if ($isRangeSelection(selection)) {
                selection.formatText('italic');
              }
            });
            return true;
          }

          // Underline: Cmd/Ctrl + U
          if ((code === 'KeyU' || key === 'u') && (ctrlKey || metaKey)) {
            event.preventDefault();
            event.stopPropagation();
            editor.update(() => {
              const selection = $getSelection();
              if ($isRangeSelection(selection)) {
                selection.formatText('underline');
              }
            });
            return true;
          }

          return false;
        },
        COMMAND_PRIORITY_EDITOR
      ),
    ];

    return () => {
      removeCommandListeners.forEach((remove) => remove());
    };
  }, [editor]);

  return null;
}

/**
 * Helper function to apply formatting to selected text
 */
export function applyFormatting(
  editor: any,
  format: 'bold' | 'italic' | 'underline' | 'strikethrough' | 'code'
) {
  editor.update(() => {
    const selection = $getSelection();
    if ($isRangeSelection(selection)) {
      selection.formatText(format);
    }
  });
}

/**
 * Helper function to check if text has specific formatting
 */
export function hasFormat(
  editor: any,
  format: 'bold' | 'italic' | 'underline' | 'strikethrough' | 'code'
): boolean {
  let hasFormatting = false;
  
  editor.getEditorState().read(() => {
    const selection = $getSelection();
    if ($isRangeSelection(selection)) {
      hasFormatting = selection.hasFormat(format);
    }
  });

  return hasFormatting;
}

