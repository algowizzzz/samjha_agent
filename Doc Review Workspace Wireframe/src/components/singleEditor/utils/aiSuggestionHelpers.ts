// Helper utilities for managing AI suggestion status in the editor

import type { LexicalEditor } from 'lexical';
import { $getSelection, $isRangeSelection, $isTextNode, $getRoot } from 'lexical';
import { $isAiTextNode, $createAiTextNode } from '../nodes/AiTextNode';
import { $isDocHeadingNode } from '../nodes/DocHeadingNode';
import { $isDocParagraphNode } from '../nodes/DocParagraphNode';
import type { AiSuggestionStatus } from '@/model/docTypes';

/**
 * Set AI suggestion status on currently selected text
 */
export function setAiStatusOnSelection(
  editor: LexicalEditor,
  status: AiSuggestionStatus
): void {
  editor.update(() => {
    const selection = $getSelection();
    if (!$isRangeSelection(selection)) {
      console.warn('[AI] No range selection active');
      return;
    }

    const nodes = selection.getNodes();
    nodes.forEach((node) => {
      if ($isAiTextNode(node)) {
        node.setAiSuggestionStatus(status);
      }
    });
  });
}

/**
 * Get currently selected text content
 */
export function getCurrentSelectionText(editor: LexicalEditor): string {
  let text = '';
  editor.getEditorState().read(() => {
    const selection = $getSelection();
    if ($isRangeSelection(selection)) {
      text = selection.getTextContent();
    }
  });
  return text;
}

/**
 * Insert AI-suggested text at current selection
 */
export function insertAiSuggestion(
  editor: LexicalEditor,
  suggestedText: string,
  status: AiSuggestionStatus = 'suggested'
): void {
  editor.update(() => {
    const selection = $getSelection();
    if (!$isRangeSelection(selection)) return;

    // Remove the currently selected text
    selection.removeText();
    
    // Create AI text node with suggestion status
    const aiNode = $createAiTextNode(suggestedText);
    aiNode.setAiSuggestionStatus(status);
    
    // Insert the AI node at the now-collapsed selection
    selection.insertNodes([aiNode]);
  });
}

/**
 * Accept all AI suggestions in the document (change status from 'suggested' to 'applied')
 */
export function acceptAllSuggestions(editor: LexicalEditor): void {
  editor.update(() => {
    const root = $getRoot();
    
    const walk = (node: any) => {
      if ($isAiTextNode(node)) {
        if (node.getAiSuggestionStatus() === 'suggested') {
          node.setAiSuggestionStatus('applied');
        }
      }
      
      const children = node.getChildren ? node.getChildren() : [];
      children.forEach((child: any) => walk(child));
    };
    
    walk(root);
  });
}

/**
 * Reject all AI suggestions in the document (change status to 'rejected')
 */
export function rejectAllSuggestions(editor: LexicalEditor): void {
  editor.update(() => {
    const root = $getRoot();
    
    const walk = (node: any) => {
      if ($isAiTextNode(node)) {
        if (node.getAiSuggestionStatus() === 'suggested') {
          node.setAiSuggestionStatus('rejected');
        }
      }
      
      const children = node.getChildren ? node.getChildren() : [];
      children.forEach((child: any) => walk(child));
    };
    
    walk(root);
  });
}

/**
 * Remove all rejected suggestions from the document
 */
export function removeRejectedSuggestions(editor: LexicalEditor): void {
  editor.update(() => {
    const root = $getRoot();
    
    const nodesToRemove: any[] = [];
    
    const walk = (node: any) => {
      if ($isAiTextNode(node)) {
        if (node.getAiSuggestionStatus() === 'rejected') {
          nodesToRemove.push(node);
        }
      }
      
      const children = node.getChildren ? node.getChildren() : [];
      children.forEach((child: any) => walk(child));
    };
    
    walk(root);
    
    // Remove collected nodes
    nodesToRemove.forEach(node => node.remove());
  });
}

/**
 * Clear all AI suggestion statuses (set to null)
 */
export function clearAllSuggestionStatuses(editor: LexicalEditor): void {
  editor.update(() => {
    const root = $getRoot();
    
    const walk = (node: any) => {
      if ($isAiTextNode(node)) {
        node.setAiSuggestionStatus(null);
      }
      
      const children = node.getChildren ? node.getChildren() : [];
      children.forEach((child: any) => walk(child));
    };
    
    walk(root);
  });
}

/**
 * Count AI suggestions by status
 */
export function countSuggestionsByStatus(editor: LexicalEditor): {
  suggested: number;
  applied: number;
  rejected: number;
} {
  const counts = {
    suggested: 0,
    applied: 0,
    rejected: 0,
  };

  editor.getEditorState().read(() => {
    const root = $getRoot();
    
    const walk = (node: any) => {
      if ($isAiTextNode(node)) {
        const status = node.getAiSuggestionStatus();
        if (status === 'suggested') counts.suggested++;
        else if (status === 'applied') counts.applied++;
        else if (status === 'rejected') counts.rejected++;
      }
      
      const children = node.getChildren ? node.getChildren() : [];
      children.forEach((child: any) => walk(child));
    };
    
    walk(root);
  });

  return counts;
}

/**
 * Get text content of a specific section by sectionKey
 */
export function getSectionContent(editor: LexicalEditor, sectionKey: string): string {
  let content = '';
  
  editor.getEditorState().read(() => {
    const root = $getRoot();
    
    let inSection = false;
    
    root.getChildren().forEach((child: any) => {
      if ($isDocHeadingNode(child) || $isDocParagraphNode(child)) {
        const nodeSectionKey = child.getSectionKey();
        
        if (nodeSectionKey === sectionKey) {
          inSection = true;
          content += child.getTextContent() + '\n';
        } else if (inSection && $isDocHeadingNode(child)) {
          // New section started, stop collecting
          inSection = false;
        } else if (inSection) {
          content += child.getTextContent() + '\n';
        }
      }
    });
  });
  
  return content.trim();
}

