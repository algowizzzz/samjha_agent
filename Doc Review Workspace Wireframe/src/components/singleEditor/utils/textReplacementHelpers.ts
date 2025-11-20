/**
 * Helper utilities for replacing text in the editor while maintaining highlights
 */

import { LexicalEditor } from 'lexical';
import { $getRoot, $isRangeSelection } from 'lexical';
import { $isAiTextNode, $createAiTextNode } from '../nodes/AiTextNode';

/**
 * Replace text that has a specific AI suggestion ID with new text
 * This is used when accepting AI suggestions
 */
export function replaceTextBySuggestionId(
  editor: LexicalEditor,
  suggestionId: string,
  newText: string
): void {
  editor.update(() => {
    const root = $getRoot();
    
    // Find all text nodes with this suggestion ID
    const textNodes: any[] = [];
    
    function collectTextNodes(node: any) {
      if ($isAiTextNode(node)) {
        if (node.getAiSuggestionId?.() === suggestionId) {
          textNodes.push(node);
        }
      }
      
      const children = node.getChildren?.();
      if (children) {
        children.forEach(collectTextNodes);
      }
    }
    
    collectTextNodes(root);
    
    if (textNodes.length === 0) {
      console.warn(`[replaceTextBySuggestionId] No nodes found with suggestion ID: ${suggestionId}`);
      return;
    }
    
    // Get the parent of the first node to know where to insert
    const firstNode = textNodes[0];
    const parent = firstNode.getParent();
    
    if (!parent) {
      console.error('[replaceTextBySuggestionId] No parent found for text node');
      return;
    }
    
    // Create a new AiTextNode with the improved text, keeping the suggestion ID
    const newNode = $createAiTextNode(newText);
    newNode.setAiSuggestionId(suggestionId);
    newNode.setAiSuggestionStatus('applied'); // Mark as applied/accepted
    
    // Copy formatting from the first original node
    if ($isAiTextNode(firstNode)) {
      newNode.setFormat(firstNode.getFormat());
      newNode.setStyle(firstNode.getStyle());
    }
    
    // Replace the first node and remove the rest
    firstNode.replace(newNode);
    
    // Remove remaining nodes (if the selection spanned multiple text nodes)
    for (let i = 1; i < textNodes.length; i++) {
      textNodes[i].remove();
    }
    
    console.log(`[replaceTextBySuggestionId] Replaced ${textNodes.length} node(s) with new text`);
  });
}

/**
 * Find and select text by suggestion ID
 * Useful for programmatically selecting the text that was suggested
 */
export function selectTextBySuggestionId(
  editor: LexicalEditor,
  suggestionId: string
): boolean {
  let found = false;
  
  editor.update(() => {
    const root = $getRoot();
    
    // Find all text nodes with this suggestion ID
    const textNodes: any[] = [];
    
    function collectTextNodes(node: any) {
      if ($isAiTextNode(node)) {
        if (node.getAiSuggestionId?.() === suggestionId) {
          textNodes.push(node);
        }
      }
      
      const children = node.getChildren?.();
      if (children) {
        children.forEach(collectTextNodes);
      }
    }
    
    collectTextNodes(root);
    
    if (textNodes.length > 0) {
      // Select from the start of the first node to the end of the last node
      const firstNode = textNodes[0];
      const lastNode = textNodes[textNodes.length - 1];
      
      firstNode.select(0, lastNode.getTextContentSize());
      found = true;
    }
  });
  
  return found;
}


