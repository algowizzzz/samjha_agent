import { useLexicalComposerContext } from '@lexical/react/LexicalComposerContext';
import { useEffect, useCallback } from 'react';
import {
  $getSelection,
  $isRangeSelection,
  COMMAND_PRIORITY_NORMAL,
  createCommand,
  LexicalCommand,
  $isTextNode,
} from 'lexical';
import { $isAiTextNode } from '../nodes/AiTextNode';

export interface CommentAnchor {
  commentId: string;
  text: string;
  blockId: string;
  startOffset: number;
  endOffset: number;
}

export const ADD_COMMENT_COMMAND: LexicalCommand<string> = createCommand('ADD_COMMENT_COMMAND');
export const REMOVE_COMMENT_COMMAND: LexicalCommand<string> = createCommand('REMOVE_COMMENT_COMMAND');

interface CommentsPluginProps {
  onCommentAdded?: (anchor: CommentAnchor) => void;
  onCommentRemoved?: (commentId: string) => void;
}

/**
 * Plugin that enables inline comments on selected text.
 * 
 * Flow:
 * 1. User selects text → clicks "Add Comment"
 * 2. Plugin adds commentId to AiTextNode's commentIds array
 * 3. CSS styles the text (e.g., yellow highlight)
 * 4. Sidebar displays comment thread
 * 
 * Comments are stored as metadata on TextNodes, not as separate nodes,
 * to avoid DOM complexity and ensure they survive edits.
 */
export function CommentsPlugin({ onCommentAdded, onCommentRemoved }: CommentsPluginProps) {
  const [editor] = useLexicalComposerContext();

  const handleAddComment = useCallback(
    (commentId: string) => {
      editor.update(() => {
        const selection = $getSelection();

        if (!$isRangeSelection(selection)) {
          return false;
        }

        const text = selection.getTextContent();
        const nodes = selection.getNodes();
        let blockId = '';
        let startOffset = 0;
        let endOffset = text.length;

        // Apply commentId to all selected text nodes
        nodes.forEach((node) => {
          if ($isTextNode(node)) {
            const parent = node.getParent();
            if (!blockId) {
              blockId = parent?.getKey() || '';
            }

            // If it's an AiTextNode, we can store commentIds
            if ($isAiTextNode(node)) {
              const writableNode = node.getWritable();
              const currentCommentIds = writableNode.__aiSuggestionStatus
                ? []
                : []; // We'd need to extend AiTextNode to store commentIds

              // For now, we'll add a class or data attribute via CSS
              // In a full implementation, extend AiTextNode with __commentIds: string[]
            }

            // Apply a mark/class for visual highlighting
            // This requires extending the TextNode or using a wrapper
            // For simplicity, we'll use a custom attribute (requires Lexical v0.12+)
          }
        });

        if (onCommentAdded && text && blockId) {
          onCommentAdded({
            commentId,
            text,
            blockId,
            startOffset,
            endOffset,
          });
        }
      });

      return true;
    },
    [editor, onCommentAdded]
  );

  const handleRemoveComment = useCallback(
    (commentId: string) => {
      editor.update(() => {
        const root = editor.getEditorState()._nodeMap;

        // Iterate through all text nodes and remove commentId
        root.forEach((node: any) => {
          if ($isAiTextNode(node)) {
            // Remove commentId from the node's commentIds array
            // This requires extending AiTextNode to store commentIds
          }
        });

        if (onCommentRemoved) {
          onCommentRemoved(commentId);
        }
      });

      return true;
    },
    [editor, onCommentRemoved]
  );

  useEffect(() => {
    const unregisterAdd = editor.registerCommand(
      ADD_COMMENT_COMMAND,
      handleAddComment,
      COMMAND_PRIORITY_NORMAL
    );

    const unregisterRemove = editor.registerCommand(
      REMOVE_COMMENT_COMMAND,
      handleRemoveComment,
      COMMAND_PRIORITY_NORMAL
    );

    return () => {
      unregisterAdd();
      unregisterRemove();
    };
  }, [editor, handleAddComment, handleRemoveComment]);

  return null;
}

/**
 * Helper to dispatch add comment command.
 * 
 * @param editor - Lexical editor instance
 * @param commentId - Unique ID for the comment thread
 */
export function addCommentToSelection(editor: any, commentId: string) {
  editor.dispatchCommand(ADD_COMMENT_COMMAND, commentId);
}

/**
 * Helper to dispatch remove comment command.
 * 
 * @param editor - Lexical editor instance
 * @param commentId - ID of comment to remove
 */
export function removeComment(editor: any, commentId: string) {
  editor.dispatchCommand(REMOVE_COMMENT_COMMAND, commentId);
}

/**
 * Note: For full comment functionality, extend AiTextNode to include:
 * 
 * ```ts
 * export class AiTextNode extends TextNode {
 *   __commentIds: string[];
 *   
 *   setCommentIds(ids: string[]): this {
 *     const self = this.getWritable();
 *     self.__commentIds = ids;
 *     return self;
 *   }
 *   
 *   addCommentId(id: string): this {
 *     const self = this.getWritable();
 *     if (!self.__commentIds.includes(id)) {
 *       self.__commentIds.push(id);
 *     }
 *     return self;
 *   }
 *   
 *   removeCommentId(id: string): this {
 *     const self = this.getWritable();
 *     self.__commentIds = self.__commentIds.filter(cid => cid !== id);
 *     return self;
 *   }
 * }
 * ```
 * 
 * Then add CSS class in createDOM():
 * ```ts
 * if (this.__commentIds && this.__commentIds.length > 0) {
 *   dom.classList.add('has-comments');
 * }
 * ```
 */

