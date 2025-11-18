import { useLexicalComposerContext } from '@lexical/react/LexicalComposerContext';
import { useEffect, useCallback } from 'react';
import { $getSelection, $isRangeSelection, COMMAND_PRIORITY_NORMAL, createCommand, LexicalCommand } from 'lexical';
import { $isDocHeadingNode } from '../nodes/DocHeadingNode';
import { $isDocParagraphNode } from '../nodes/DocParagraphNode';

export interface SelectionContext {
  text: string;
  blockIds: string[];
  sectionKeys: string[];
  blockTypes: string[];
}

export const ASK_AI_COMMAND: LexicalCommand<void> = createCommand('ASK_AI_COMMAND');

interface SelectionToAIPluginProps {
  onAskAI?: (context: SelectionContext) => void;
}

/**
 * Plugin that enables "Select text → Ask RiskGPT" workflow.
 * 
 * Listens for ASK_AI_COMMAND and extracts:
 * - Selected text
 * - Block IDs and types
 * - Section keys
 * 
 * Passes this context to the onAskAI callback for AI processing.
 */
export function SelectionToAIPlugin({ onAskAI }: SelectionToAIPluginProps) {
  const [editor] = useLexicalComposerContext();

  const handleAskAI = useCallback(() => {
    editor.getEditorState().read(() => {
      const selection = $getSelection();
      
      if (!$isRangeSelection(selection)) {
        return false;
      }

      const text = selection.getTextContent();
      const nodes = selection.getNodes();
      
      const blockIds: string[] = [];
      const sectionKeys: string[] = [];
      const blockTypes: string[] = [];

      // Extract context from selected nodes
      nodes.forEach((node) => {
        const parent = node.getParent();
        
        if ($isDocHeadingNode(parent)) {
          blockIds.push(parent.getKey());
          blockTypes.push(`heading-${parent.__level}`);
          if (parent.__sectionKey) {
            sectionKeys.push(parent.__sectionKey);
          }
        } else if ($isDocParagraphNode(parent)) {
          blockIds.push(parent.getKey());
          blockTypes.push('paragraph');
          // DocParagraphNode can also have sectionKey if needed
        }
      });

      // Remove duplicates
      const context: SelectionContext = {
        text,
        blockIds: [...new Set(blockIds)],
        sectionKeys: [...new Set(sectionKeys)],
        blockTypes: [...new Set(blockTypes)],
      };

      if (onAskAI && text) {
        onAskAI(context);
      }
    });

    return true;
  }, [editor, onAskAI]);

  useEffect(() => {
    return editor.registerCommand(
      ASK_AI_COMMAND,
      handleAskAI,
      COMMAND_PRIORITY_NORMAL
    );
  }, [editor, handleAskAI]);

  return null;
}

/**
 * Helper to programmatically trigger the AI command from outside the editor.
 * 
 * @example
 * ```tsx
 * <button onClick={() => dispatchAskAICommand(editor)}>
 *   Ask RiskGPT
 * </button>
 * ```
 */
export function dispatchAskAICommand(editor: any) {
  editor.dispatchCommand(ASK_AI_COMMAND, undefined);
}

