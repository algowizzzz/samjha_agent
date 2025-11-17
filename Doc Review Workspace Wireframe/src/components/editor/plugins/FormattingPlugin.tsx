import { useEffect } from 'react';
import { useLexicalComposerContext } from '@lexical/react/LexicalComposerContext';
import {
  $getSelection,
  $isRangeSelection,
  COMMAND_PRIORITY_LOW,
  FORMAT_TEXT_COMMAND,
  KEY_MODIFIER_COMMAND,
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
      // Bold: Cmd/Ctrl + B
      editor.registerCommand(
        KEY_MODIFIER_COMMAND,
        (payload) => {
          const event = payload as KeyboardEvent;
          const { code, ctrlKey, metaKey } = event;

          if (code === 'KeyB' && (ctrlKey || metaKey)) {
            event.preventDefault();
            editor.dispatchCommand(FORMAT_TEXT_COMMAND, 'bold');
            return true;
          }

          // Italic: Cmd/Ctrl + I
          if (code === 'KeyI' && (ctrlKey || metaKey)) {
            event.preventDefault();
            editor.dispatchCommand(FORMAT_TEXT_COMMAND, 'italic');
            return true;
          }

          // Underline: Cmd/Ctrl + U
          if (code === 'KeyU' && (ctrlKey || metaKey)) {
            event.preventDefault();
            editor.dispatchCommand(FORMAT_TEXT_COMMAND, 'underline');
            return true;
          }

          return false;
        },
        COMMAND_PRIORITY_LOW
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

