/**
 * Helper utilities for applying comment highlighting to text nodes
 */

import { LexicalEditor } from 'lexical';
import { $getRoot, $getSelection, $isRangeSelection } from 'lexical';
import { $isAiTextNode } from '../nodes/AiTextNode';
import { $isDocParagraphNode } from '../nodes/DocParagraphNode';
import { $isDocHeadingNode } from '../nodes/DocHeadingNode';
import { $isDocListNode } from '../nodes/DocListNode';
import { $isDocCodeNode } from '../nodes/DocCodeNode';
import { $isDocQuoteNode } from '../nodes/DocQuoteNode';

/**
 * Apply comment highlighting to text nodes in the current selection
 * Returns the blockId and text offsets for the comment
 */
export function applyCommentToSelection(
  editor: LexicalEditor,
  commentId: string
): { blockId: string; selectedText: string; startOffset: number; endOffset: number } | null {
  let result: { blockId: string; selectedText: string; startOffset: number; endOffset: number } | null = null;

  editor.update(() => {
    const selection = $getSelection();
    if (!$isRangeSelection(selection)) {
      return;
    }

    const selectedText = selection.getTextContent();
    if (!selectedText) {
      return;
    }

    // Get all selected nodes
    const nodes = selection.getNodes();
    
    // Find the parent block to get blockId
    let blockId = 'unknown';
    for (const node of nodes) {
      let current = node;
      while (current) {
        const nodeType = current.getType();
        if (
          nodeType === 'doc-paragraph' ||
          nodeType === 'doc-heading' ||
          nodeType === 'doc-list' ||
          nodeType === 'doc-code' ||
          nodeType === 'doc-quote'
        ) {
          // Get blockId from the block node
          blockId = (current as any).getBlockId?.() || 'unknown';
          break;
        }
        current = current.getParent();
      }
      if (blockId !== 'unknown') break;
    }

    // Mark all text nodes in selection with commentId
    for (const node of nodes) {
      if ($isAiTextNode(node)) {
        node.addCommentId(commentId);
      }
    }

    // Calculate offsets (simplified - using selection text length)
    // In a real implementation, you'd calculate actual character offsets within the block
    result = {
      blockId,
      selectedText,
      startOffset: 0, // TODO: Calculate actual offset
      endOffset: selectedText.length,
    };
  });

  return result;
}

/**
 * Remove comment highlighting from all text nodes with this commentId
 */
export function removeCommentHighlight(
  editor: LexicalEditor,
  commentId: string
): void {
  editor.update(() => {
    const root = $getRoot();
    
    // Recursively find all text nodes
    const textNodes: any[] = [];
    
    function collectTextNodes(node: any) {
      if ($isAiTextNode(node)) {
        textNodes.push(node);
      }
      
      const children = node.getChildren?.();
      if (children) {
        children.forEach(collectTextNodes);
      }
    }
    
    collectTextNodes(root);
    
    // Remove commentId from all text nodes
    for (const textNode of textNodes) {
      if (textNode.getCommentIds?.().includes(commentId)) {
        textNode.removeCommentId(commentId);
      }
    }
  });
}

/**
 * Highlight text in a specific block with specific offsets (for clicking on a comment)
 */
export function highlightCommentInEditor(
  editor: LexicalEditor,
  blockId: string,
  startOffset: number,
  endOffset: number
): void {
  editor.update(() => {
    const root = $getRoot();
    
    // Find the block node with this ID
    let targetBlock: any = null;
    
    function findBlock(node: any): boolean {
      const nodeType = node.getType?.();
      if (
        nodeType === 'doc-paragraph' ||
        nodeType === 'doc-heading' ||
        nodeType === 'doc-list' ||
        nodeType === 'doc-code' ||
        nodeType === 'doc-quote'
      ) {
        if (node.getBlockId?.() === blockId) {
          targetBlock = node;
          return true;
        }
      }
      
      const children = node.getChildren?.();
      if (children) {
        for (const child of children) {
          if (findBlock(child)) return true;
        }
      }
      return false;
    }
    
    findBlock(root);
    
    if (!targetBlock) {
      console.warn(`Block with ID ${blockId} not found`);
      return;
    }

    // TODO: Scroll to and highlight the specific text range
    // For now, just scroll to the block
    const domElement = editor.getElementByKey(targetBlock.getKey());
    if (domElement) {
      domElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  });
}

