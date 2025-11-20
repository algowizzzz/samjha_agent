// SelectionBridgePlugin: Extracts block IDs from text selection
// This bridges Lexical's text selection to our block-based backend

import { useLexicalComposerContext } from '@lexical/react/LexicalComposerContext';
import { $getSelection, $isRangeSelection } from 'lexical';
import { useEffect } from 'react';
import { $findMatchingParent } from '@lexical/utils';
import { DocParagraphNode } from '../nodes/DocParagraphNode';
import { DocHeadingNode } from '../nodes/DocHeadingNode';
import { DocListNode } from '../nodes/DocListNode';
import { DocListItemNode } from '../nodes/DocListItemNode';
import { DocCodeNode } from '../nodes/DocCodeNode';
import { DocQuoteNode } from '../nodes/DocQuoteNode';
import { DocDividerNode } from '../nodes/DocDividerNode';
import { DocImageNode } from '../nodes/DocImageNode';
import { DocEmptyNode } from '../nodes/DocEmptyNode';

export interface SelectionData {
  selectionScope: 'blocks' | 'text' | 'none';
  blockIds: string[];
  selectedText: string;
  isEmpty: boolean;
  currentBlockType?: 'paragraph' | 'heading' | 'list' | 'code' | 'quote' | 'divider' | 'image' | 'empty';
  currentBlockLevel?: 1 | 2 | 3;
  currentListStyle?: 'bullet' | 'number';
  isConvertible?: boolean; // Can this block be converted via "Turn Into"?
}

interface SelectionBridgePluginProps {
  onSelectionChange?: (data: SelectionData) => void;
}

/**
 * SelectionBridgePlugin tracks text selection and extracts:
 * - Block IDs involved in the selection
 * - Selected text content
 * - Mode: 'text' for small selections, 'blocks' for multi-block selections
 */
export function SelectionBridgePlugin({ onSelectionChange }: SelectionBridgePluginProps) {
  const [editor] = useLexicalComposerContext();
  
  useEffect(() => {
    if (!onSelectionChange) return;
    
    return editor.registerUpdateListener(({ editorState }) => {
      editorState.read(() => {
        const selection = $getSelection();
        
        // No selection
        if (!selection) {
          onSelectionChange({
            selectionScope: 'none',
            blockIds: [],
            selectedText: '',
            isEmpty: true,
          });
          return;
        }
        
        // Not range selection - might still be near a block
        if (!$isRangeSelection(selection)) {
          onSelectionChange({
            selectionScope: 'none',
            blockIds: [],
            selectedText: '',
            isEmpty: true,
          });
          return;
        }
        
        const selectedText = selection.getTextContent();
        const isEmpty = selection.isCollapsed();
        
        // Get current block type info (always, even for empty selection)
        const anchor = selection.anchor.getNode();
        
        // First try to find a list item (if we're in a list)
        const listItemNode = $findMatchingParent(anchor, (n) => n instanceof DocListItemNode);
        
        // Then find the block container
        const blockNode = $findMatchingParent(
          anchor,
          (n) => n instanceof DocParagraphNode || 
                 n instanceof DocHeadingNode || 
                 n instanceof DocListNode ||
                 n instanceof DocCodeNode ||
                 n instanceof DocQuoteNode ||
                 n instanceof DocDividerNode ||
                 n instanceof DocImageNode ||
                 n instanceof DocEmptyNode
        );
        
        let currentBlockType: SelectionData['currentBlockType'];
        let currentBlockLevel: SelectionData['currentBlockLevel'];
        let currentListStyle: SelectionData['currentListStyle'];
        let isConvertible = true; // Most blocks can be converted
        
        if (blockNode instanceof DocHeadingNode) {
          currentBlockType = 'heading';
          currentBlockLevel = blockNode.getLevel() as 1 | 2 | 3;
        } else if (blockNode instanceof DocParagraphNode) {
          currentBlockType = 'paragraph';
        } else if (blockNode instanceof DocListNode) {
          currentBlockType = 'list';
          currentListStyle = blockNode.getListStyle();
        } else if (blockNode instanceof DocCodeNode) {
          currentBlockType = 'code';
        } else if (blockNode instanceof DocQuoteNode) {
          currentBlockType = 'quote';
        } else if (blockNode instanceof DocDividerNode) {
          currentBlockType = 'divider';
          isConvertible = true; // Allow conversion to editable blocks
        } else if (blockNode instanceof DocImageNode) {
          currentBlockType = 'image';
          isConvertible = true; // Allow conversion to editable blocks
        } else if (blockNode instanceof DocEmptyNode) {
          currentBlockType = 'empty';
          isConvertible = true; // Allow conversion to editable blocks
        }
        
        // Empty selection (just cursor)
        if (isEmpty) {
          onSelectionChange({
            selectionScope: 'none',
            blockIds: [],
            selectedText: '',
            isEmpty: true,
            currentBlockType,
            currentBlockLevel,
            currentListStyle,
            isConvertible,
          });
          return;
        }
        
        // Extract block IDs from selected nodes
        const nodes = selection.getNodes();
        const blockIds = new Set<string>();
        
        nodes.forEach(node => {
          // Find parent block node
          const blockNode = $findMatchingParent(
            node,
            (n) => n instanceof DocParagraphNode || 
                   n instanceof DocHeadingNode ||
                   n instanceof DocListNode ||
                   n instanceof DocCodeNode ||
                   n instanceof DocQuoteNode
          );
          
          if (blockNode) {
            const id = (blockNode as any).getBlockId?.() || blockNode.getKey();
            blockIds.add(id);
          }
        });
        
        // Determine selection scope:
        // - 'text' scope: small selection within single block (for comments)
        // - 'blocks' scope: large/multi-block selection (for RiskGPT analysis)
        const selectionScope = blockIds.size === 1 && selectedText.length < 500 ? 'text' : 'blocks';
        
        onSelectionChange({
          selectionScope,
          blockIds: Array.from(blockIds),
          selectedText,
          isEmpty: false,
          currentBlockType,
          currentBlockLevel,
          currentListStyle,
          isConvertible,
        });
      });
    });
  }, [editor, onSelectionChange]);
  
  return null;
}

