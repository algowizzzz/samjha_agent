/**
 * Helper utilities for applying comment highlighting to text nodes
 */

import { LexicalEditor, LexicalNode } from 'lexical';
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
      let current: any = node;
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
          blockId = current.getBlockId?.() || 'unknown';
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
 * Apply comment highlighting to the current selection
 * This is used when creating a new comment
 */
export function applyCommentHighlight(
  editor: LexicalEditor,
  commentId: string
): void {
  editor.update(() => {
    const selection = $getSelection();
    if (!$isRangeSelection(selection)) {
      return;
    }

    const nodes = selection.getNodes();
    
    // Mark all text nodes in selection with commentId
    for (const node of nodes) {
      if ($isAiTextNode(node)) {
        node.addCommentId(commentId);
      }
    }
  });
}

/**
 * Apply comment highlighting based on comment data (for loading existing comments)
 * Uses precise character offsets to highlight exact selection
 */
export function applyCommentHighlightByData(
  editor: LexicalEditor,
  commentId: string,
  blockId: string,
  selectionText: string,
  startOffset?: number,
  endOffset?: number
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
      console.warn(`[applyCommentHighlightByData] Block with ID ${blockId} not found`);
      return;
    }

    // Find all text nodes in this block
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
    
    collectTextNodes(targetBlock);
    
    // If we have precise offsets, use them
    let actualStartIndex = startOffset ?? -1;
    let actualEndIndex = endOffset ?? -1;
    
    // Fallback: search for text if no offsets provided
    if (actualStartIndex === -1 || actualEndIndex === -1) {
      const blockText = targetBlock.getTextContent();
      actualStartIndex = blockText.indexOf(selectionText);
      
      if (actualStartIndex === -1) {
        console.warn(`[applyCommentHighlightByData] Selection text "${selectionText}" not found in block ${blockId}`);
        return;
      }
      
      actualEndIndex = actualStartIndex + selectionText.length;
    }
    
    // Apply commentId to text nodes that fall within the selection range
    let currentIndex = 0;
    for (const textNode of textNodes) {
      const nodeText = textNode.getTextContent();
      const nodeStart = currentIndex;
      const nodeEnd = currentIndex + nodeText.length;
      
      // Check if this node overlaps with the selection
      if (nodeStart < actualEndIndex && nodeEnd > actualStartIndex) {
        // For now, apply to entire node if it overlaps (simpler, more reliable)
        // TODO: Implement precise character-level node splitting
        textNode.addCommentId(commentId);
      }
      
      currentIndex = nodeEnd;
    }
    
    console.log(`[applyCommentHighlightByData] Applied highlight for comment ${commentId} in block ${blockId} (offsets: ${actualStartIndex}-${actualEndIndex})`);
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

