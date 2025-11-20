/**
 * Helper utilities for applying AI suggestion highlighting to text nodes
 */

import { LexicalEditor } from 'lexical';
import { $getRoot, $getSelection, $isRangeSelection, $createTextNode } from 'lexical';
import { $isAiTextNode, $createAiTextNode } from '../nodes/AiTextNode';
import type { AiSuggestionStatus } from '@/model/docTypes';

/**
 * Apply AI suggestion highlighting to the current selection
 * This is used when creating a new AI suggestion
 */
export function applyAISuggestionHighlight(
  editor: LexicalEditor,
  suggestionId: string
): void {
  editor.update(() => {
    const selection = $getSelection();
    if (!$isRangeSelection(selection)) {
      return;
    }

    const nodes = selection.getNodes();
    
    // Mark all text nodes in selection with suggestion ID and status
    for (const node of nodes) {
      if ($isAiTextNode(node)) {
        node.setAiSuggestionId(suggestionId);
        node.setAiSuggestionStatus('suggested');
      }
    }
  });
}

/**
 * Apply AI suggestion highlighting based on suggestion data (for loading existing suggestions)
 * Uses precise character offsets to highlight exact selection
 */
export function applyAISuggestionHighlightByData(
  editor: LexicalEditor,
  suggestionId: string,
  blockId: string,
  selectionText: string,
  status: 'pending' | 'accepted' | 'rejected' = 'pending',
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
      console.warn(`[applyAISuggestionHighlightByData] Block with ID ${blockId} not found`);
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
        console.warn(`[applyAISuggestionHighlightByData] Selection text "${selectionText}" not found in block ${blockId}`);
        return;
      }
      
      actualEndIndex = actualStartIndex + selectionText.length;
    }
    
    // Map status to AI suggestion status
    const aiStatus = status === 'accepted' ? 'applied' : status === 'rejected' ? 'rejected' : 'suggested';
    
    // Apply suggestion ID and status to text nodes that fall within the selection range
    let currentIndex = 0;
    for (const textNode of textNodes) {
      const nodeText = textNode.getTextContent();
      const nodeStart = currentIndex;
      const nodeEnd = currentIndex + nodeText.length;
      
      // Check if this node overlaps with the selection
      if (nodeStart < actualEndIndex && nodeEnd > actualStartIndex) {
        // For now, apply to entire node if it overlaps (simpler, more reliable)
        // TODO: Implement precise character-level node splitting
        textNode.setAiSuggestionId(suggestionId);
        textNode.setAiSuggestionStatus(aiStatus);
      }
      
      currentIndex = nodeEnd;
    }
    
    console.log(`[applyAISuggestionHighlightByData] Applied highlight for suggestion ${suggestionId} in block ${blockId} with status ${status} (offsets: ${actualStartIndex}-${actualEndIndex})`);
  });
}

/**
 * Remove AI suggestion highlighting from all text nodes with this suggestionId
 */
export function removeAISuggestionHighlight(
  editor: LexicalEditor,
  suggestionId: string
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
    
    // Remove suggestion ID and status from all matching text nodes
    for (const textNode of textNodes) {
      if (textNode.getAiSuggestionId?.() === suggestionId) {
        textNode.setAiSuggestionId(undefined);
        textNode.setAiSuggestionStatus(null);
      }
    }
  });
}

/**
 * Update AI suggestion status (when accepting or rejecting)
 */
export function updateAISuggestionStatus(
  editor: LexicalEditor,
  suggestionId: string,
  status: 'pending' | 'accepted' | 'rejected'
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
    
    // Map status to AI suggestion status
    const aiStatus = status === 'accepted' ? 'applied' : status === 'rejected' ? 'rejected' : 'suggested';
    
    // Update status for all matching text nodes
    for (const textNode of textNodes) {
      if (textNode.getAiSuggestionId?.() === suggestionId) {
        textNode.setAiSuggestionStatus(aiStatus);
      }
    }
  });
}

/**
 * Get the currently selected text from the editor
 */
export function getCurrentSelectionText(editor: LexicalEditor): string {
  let selectedText = '';
  
  editor.getEditorState().read(() => {
    const selection = $getSelection();
    if ($isRangeSelection(selection)) {
      selectedText = selection.getTextContent();
    }
  });
  
  return selectedText;
}

/**
 * Set AI suggestion status on the currently selected text nodes
 */
export function setAiStatusOnSelection(
  editor: LexicalEditor,
  status: AiSuggestionStatus
): void {
  editor.update(() => {
    const selection = $getSelection();
    if (!$isRangeSelection(selection)) {
      return;
    }

    const nodes = selection.getNodes();
    
    for (const node of nodes) {
      if ($isAiTextNode(node)) {
        node.setAiSuggestionStatus(status);
      }
    }
  });
}

/**
 * Insert AI-suggested text at the current selection
 * Replaces selected text with the improved version
 */
export function insertAiSuggestion(
  editor: LexicalEditor,
  improvedText: string,
  suggestionId?: string
): void {
  editor.update(() => {
    const selection = $getSelection();
    if (!$isRangeSelection(selection) || selection.isCollapsed()) {
      return;
    }

    // Remove the selected text first to collapse selection within an existing block
    const anchorNode = selection.anchor.getNode();
    selection.removeText();
    
    // Now insert the AI suggestion text as AiTextNode at collapsed selection
    const aiTextNode = $createAiTextNode(improvedText);
    aiTextNode.setAiSuggestionStatus('suggested');
    if (suggestionId) {
      aiTextNode.setAiSuggestionId(suggestionId);
    }
    
    // Insert the node at the current (now collapsed) selection
    selection.insertNodes([aiTextNode]);
  });
}
