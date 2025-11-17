import { useEffect } from 'react';
import { useLexicalComposerContext } from '@lexical/react/LexicalComposerContext';
import { 
  $createHeadingNode,
  $createQuoteNode,
  HeadingTagType
} from '@lexical/rich-text';
import { $createCodeNode } from '@lexical/code';
import { 
  $createListNode,
  $createListItemNode,
  ListType
} from '@lexical/list';
import { 
  $getRoot,
  $createParagraphNode,
  LexicalNode
} from 'lexical';
import type { BlockType } from '../types';

interface BlockTypePluginProps {
  blockType: BlockType;
  onBlockTypeChange?: (newType: BlockType) => void;
}

/**
 * Plugin to manage block types in Lexical editor
 * Converts between different block types (heading, paragraph, list, etc.)
 */
export function BlockTypePlugin({ blockType, onBlockTypeChange }: BlockTypePluginProps) {
  const [editor] = useLexicalComposerContext();

  useEffect(() => {
    // Apply block type to the root node
    editor.update(() => {
      const root = $getRoot();
      const firstChild = root.getFirstChild();
      
      if (!firstChild) return;

      let newNode: LexicalNode | null = null;

      switch (blockType) {
        case 'heading1':
          newNode = $createHeadingNode('h1' as HeadingTagType);
          break;
        case 'heading2':
          newNode = $createHeadingNode('h2' as HeadingTagType);
          break;
        case 'heading3':
          newNode = $createHeadingNode('h3' as HeadingTagType);
          break;
        case 'quote':
          newNode = $createQuoteNode();
          break;
        case 'code':
          newNode = $createCodeNode();
          break;
        case 'bullet':
          const bulletList = $createListNode('bullet' as ListType);
          const bulletItem = $createListItemNode();
          bulletList.append(bulletItem);
          newNode = bulletList;
          break;
        case 'numbered':
          const numberedList = $createListNode('number' as ListType);
          const numberedItem = $createListItemNode();
          numberedList.append(numberedItem);
          newNode = numberedList;
          break;
        case 'paragraph':
        default:
          newNode = $createParagraphNode();
          break;
      }

      if (newNode && firstChild.__type !== newNode.__type) {
        // Replace the first child with the new node type
        // Preserve text content
        const textContent = firstChild.getTextContent();
        if (newNode.__type === 'list') {
          // For lists, add content to the first list item
          const listItem = newNode.getFirstChild();
          if (listItem && textContent) {
            listItem.append(firstChild);
          }
        } else {
          // For other types, just append children
          const children = firstChild.getChildren();
          children.forEach(child => newNode?.append(child));
        }
        
        firstChild.replace(newNode);
      }
    });
  }, [editor, blockType]);

  return null;
}

/**
 * Helper function to convert block type string to Lexical node
 */
export function createNodeForBlockType(blockType: BlockType): LexicalNode {
  switch (blockType) {
    case 'heading1':
      return $createHeadingNode('h1' as HeadingTagType);
    case 'heading2':
      return $createHeadingNode('h2' as HeadingTagType);
    case 'heading3':
      return $createHeadingNode('h3' as HeadingTagType);
    case 'quote':
      return $createQuoteNode();
    case 'code':
      return $createCodeNode();
    case 'bullet': {
      const list = $createListNode('bullet' as ListType);
      const item = $createListItemNode();
      list.append(item);
      return list;
    }
    case 'numbered': {
      const list = $createListNode('number' as ListType);
      const item = $createListItemNode();
      list.append(item);
      return list;
    }
    case 'paragraph':
    default:
      return $createParagraphNode();
  }
}

