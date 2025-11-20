/**
 * Calculate character offsets for a selection within a block
 */

import { LexicalEditor } from 'lexical';
import { $getSelection, $isRangeSelection, $isTextNode } from 'lexical';

export interface SelectionOffsets {
  startOffset: number;
  endOffset: number;
  blockId: string;
  selectedText: string;
}

/**
 * Get character offsets for the current selection within its parent block
 */
export function getSelectionOffsets(editor: LexicalEditor): SelectionOffsets | null {
  let result: SelectionOffsets | null = null;

  editor.getEditorState().read(() => {
    const selection = $getSelection();
    if (!$isRangeSelection(selection)) {
      return;
    }

    // Get the anchor and focus nodes
    const anchorNode = selection.anchor.getNode();
    const focusNode = selection.focus.getNode();

    // Find the parent block node
    let blockNode: any = anchorNode;
    let blockId = 'unknown';

    while (blockNode) {
      const nodeType = blockNode.getType?.();
      if (
        nodeType === 'doc-paragraph' ||
        nodeType === 'doc-heading' ||
        nodeType === 'doc-list' ||
        nodeType === 'doc-code' ||
        nodeType === 'doc-quote'
      ) {
        blockId = blockNode.getBlockId?.() || 'unknown';
        break;
      }
      blockNode = blockNode.getParent();
    }

    if (!blockNode || blockId === 'unknown') {
      console.warn('[getSelectionOffsets] Could not find parent block');
      return;
    }

    // Get all text nodes in the block
    const textNodes: any[] = [];
    function collectTextNodes(node: any) {
      if ($isTextNode(node)) {
        textNodes.push(node);
      }
      const children = node.getChildren?.();
      if (children) {
        children.forEach(collectTextNodes);
      }
    }
    collectTextNodes(blockNode);

    // Calculate offsets
    let currentOffset = 0;
    let startOffset = -1;
    let endOffset = -1;

    const anchorOffset = selection.anchor.offset;
    const focusOffset = selection.focus.offset;
    
    // Determine if selection is forward or backward
    const isBackward = selection.isBackward();
    const actualStart = isBackward ? focusNode : anchorNode;
    const actualEnd = isBackward ? anchorNode : focusNode;
    const actualStartOffset = isBackward ? focusOffset : anchorOffset;
    const actualEndOffset = isBackward ? anchorOffset : focusOffset;

    for (const textNode of textNodes) {
      const nodeText = textNode.getTextContent();
      const nodeLength = nodeText.length;

      // Check if this is the start node
      if (textNode === actualStart) {
        startOffset = currentOffset + actualStartOffset;
      }

      // Check if this is the end node
      if (textNode === actualEnd) {
        endOffset = currentOffset + actualEndOffset;
        break;
      }

      currentOffset += nodeLength;
    }

    // Fallback: if we didn't find both offsets, use the entire selection range
    if (startOffset === -1 || endOffset === -1) {
      console.warn('[getSelectionOffsets] Could not determine offsets precisely');
      const blockText = blockNode.getTextContent();
      const selectedText = selection.getTextContent();
      startOffset = blockText.indexOf(selectedText);
      endOffset = startOffset + selectedText.length;
    }

    result = {
      startOffset: Math.min(startOffset, endOffset),
      endOffset: Math.max(startOffset, endOffset),
      blockId,
      selectedText: selection.getTextContent()
    };
  });

  return result;
}


