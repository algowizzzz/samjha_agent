// Plugin to export Lexical editor state to DocState format

import { useLexicalComposerContext } from '@lexical/react/LexicalComposerContext';
import { useEffect, useCallback, useRef } from 'react';
import { $getRoot, $isTextNode } from 'lexical';
import type { DocState, DocBlock, TextRun } from '@/model/docTypes';
import { $isDocHeadingNode } from '../nodes/DocHeadingNode';
import { $isDocParagraphNode } from '../nodes/DocParagraphNode';
import { $isAiTextNode } from '../nodes/AiTextNode';
import { $isDocListNode } from '../nodes/DocListNode';
import { $isDocListItemNode } from '../nodes/DocListItemNode';
import { $isDocCodeNode } from '../nodes/DocCodeNode';
import { $isDocQuoteNode } from '../nodes/DocQuoteNode';
import { $isDocDividerNode } from '../nodes/DocDividerNode';
import { $isDocImageNode } from '../nodes/DocImageNode';
import { $isDocEmptyNode } from '../nodes/DocEmptyNode';

interface DocExportOnChangePluginProps {
  onDocChange: (doc: DocState) => void;
  debounceMs?: number;
}

export function DocExportOnChangePlugin({
  onDocChange,
  debounceMs = 300,
}: DocExportOnChangePluginProps) {
  const [editor] = useLexicalComposerContext();
  const timeoutRef = useRef<NodeJS.Timeout>();
  
  const exportToDocState = useCallback(() => {
    editor.getEditorState().read(() => {
      const root = $getRoot();
      const blocks: DocBlock[] = [];

      root.getChildren().forEach((child) => {
        if ($isDocHeadingNode(child)) {
          const textRuns = collectTextRuns(child);
          blocks.push({
            id: child.getKey(),
            type: 'heading',
            level: child.getLevel(),
            sectionKey: child.getSectionKey(),
            text: textRuns,
          });
        } else if ($isDocParagraphNode(child)) {
          const textRuns = collectTextRuns(child);
          blocks.push({
            id: child.getKey(),
            type: 'paragraph',
            sectionKey: child.getSectionKey(),
            text: textRuns,
          });
        } else if ($isDocListNode(child)) {
          const listStyle = child.getListStyle();
          const listItemNodes = child.getChildren().filter($isDocListItemNode);
          const items = listItemNodes.map((itemNode) => {
            const textContent = itemNode.getTextContent();
            return { content: textContent };
          });
          blocks.push({
            id: child.getBlockId(),
            type: listStyle === 'bullet' ? 'bulleted_list' : 'numbered_list',
            items: items,
          } as any);
        } else if ($isDocCodeNode(child)) {
          const codeContent = child.getTextContent(); // Extract from TextNode children
          const language = child.getLanguage();
          blocks.push({
            id: child.getBlockId(),
            type: 'code',
            content: codeContent,
            language: language,
          } as any);
        } else if ($isDocQuoteNode(child)) {
          const textRuns = collectTextRuns(child);
          blocks.push({
            id: child.getBlockId(),
            type: 'quote',
            text: textRuns,
          } as any);
        } else if ($isDocDividerNode(child)) {
          blocks.push({
            id: child.getBlockId(),
            type: 'divider',
          } as any);
        } else if ($isDocImageNode(child)) {
          blocks.push({
            id: child.getBlockId(),
            type: 'image',
            src: child.getSrc(),
            description: child.getDescription(),
          } as any);
        } else if ($isDocEmptyNode(child)) {
          blocks.push({
            id: child.getBlockId(),
            type: 'empty',
          } as any);
        }
        // TODO: Handle table, note/callout blocks
      });

      const docState: DocState = {
        id: 'current-doc', // This should come from props or context
        blocks,
      };

      onDocChange(docState);
    });
  }, [editor, onDocChange]);

  useEffect(() => {
    return editor.registerUpdateListener(({ editorState, tags }) => {
      // Don't export if this update is from history (undo/redo)
      if (tags.has('historic')) {
        return;
      }

      // Debounce exports to avoid too many updates
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      timeoutRef.current = setTimeout(() => {
        exportToDocState();
      }, debounceMs);
    });
  }, [editor, exportToDocState, debounceMs]);

  return null;
}

// Collect text runs from a node's children
function collectTextRuns(node: any): TextRun[] {
  const runs: TextRun[] = [];
  
  node.getChildren().forEach((child: any) => {
    if ($isTextNode(child)) {
      const text = child.getTextContent();
      
      if (text.length > 0) {
        const run: TextRun = {
          text,
          bold: child.hasFormat('bold'),
          italic: child.hasFormat('italic'),
          underline: child.hasFormat('underline'),
          code: child.hasFormat('code'),
        };

        // Add AI suggestion status if this is an AiTextNode
        if ($isAiTextNode(child)) {
          const status = child.getAiSuggestionStatus();
          if (status) {
            run.aiSuggestionStatus = status;
          }
          
          const commentIds = child.getCommentIds();
          if (commentIds.length > 0) {
            run.commentIds = commentIds;
          }
        }

        runs.push(run);
      }
    }
  });

  return runs;
}

