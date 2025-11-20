// Plugin to initialize Lexical editor from DocState

import { useEffect, useRef } from 'react';
import { useLexicalComposerContext } from '@lexical/react/LexicalComposerContext';
import { $getRoot } from 'lexical';
import type { DocState, DocBlock, TextRun } from '@/model/docTypes';
import { $createDocHeadingNode } from '../nodes/DocHeadingNode';
import { $createDocParagraphNode } from '../nodes/DocParagraphNode';
import { $createAiTextNode } from '../nodes/AiTextNode';
import { $createDocListNode } from '../nodes/DocListNode';
import { $createDocListItemNode } from '../nodes/DocListItemNode';
import { $createDocCodeNode } from '../nodes/DocCodeNode';
import { $createDocQuoteNode } from '../nodes/DocQuoteNode';
import { $createDocDividerNode } from '../nodes/DocDividerNode';
import { $createDocImageNode } from '../nodes/DocImageNode';
import { $createDocEmptyNode } from '../nodes/DocEmptyNode';

interface DocInitializerPluginProps {
  initialDoc: DocState;
}

export function DocInitializerPlugin({ initialDoc }: DocInitializerPluginProps) {
  const [editor] = useLexicalComposerContext();
  const initializedRef = useRef(false);

  useEffect(() => {
    // Only initialize once when component mounts
    if (initializedRef.current) return;
    initializedRef.current = true;

    console.log('[DocInitializerPlugin] Initializing editor with', initialDoc.blocks.length, 'blocks');

    editor.update(() => {
      const root = $getRoot();
      root.clear();

      for (const block of initialDoc.blocks) {
        const node = createNodeFromBlock(block);
        if (node) {
          root.append(node);
        }
      }
    });
  }, [editor, initialDoc]);

  return null;
}

// Convert DocBlock to Lexical node
function createNodeFromBlock(block: DocBlock) {
  switch (block.type) {
    case 'heading': {
      const headingNode = $createDocHeadingNode(block.level, block.id, block.sectionKey);
      appendTextRuns(headingNode, block.content);
      return headingNode;
    }
    
    case 'paragraph': {
      const pNode = $createDocParagraphNode(block.id, block.sectionKey);
      appendTextRuns(pNode, block.content);
      return pNode;
    }
    
    case 'list': {
      // Handle legacy list format with proper Lexical nodes
      const items = block.items.map(item => ({
        content: typeof item === 'string' ? item : item.content?.map((t: any) => t.text).join('') || ''
      }));
      const listNode = $createDocListNode(block.id, block.listStyle, items);
      
      // Create ListItemNode children with TextNode content
      items.forEach((item) => {
        const listItemNode = $createDocListItemNode();
        const textNode = $createAiTextNode(item.content || '');
        listItemNode.append(textNode);
        listNode.append(listItemNode);
      });
      
      return listNode;
    }
    
    case 'bulleted_list':
    case 'numbered_list': {
      // Handle BlockEditor format with proper Lexical nodes
      const style = block.type === 'bulleted_list' ? 'bullet' : 'number';
      const rawItems = (block as any).items || [];
      const items = rawItems.map((item: any) => ({
        content: item.content || item.text || (typeof item === 'string' ? item : '')
      }));
      const listNode = $createDocListNode((block as any).id || block.id, style, items);
      
      // Create ListItemNode children with TextNode content
      items.forEach((item: any) => {
        const listItemNode = $createDocListItemNode();
        const textNode = $createAiTextNode(item.content || '');
        listItemNode.append(textNode);
        listNode.append(listItemNode);
      });
      
      return listNode;
    }
    
    case 'preformatted':
    case 'code': {
      // Handle code blocks with proper Lexical TextNode children
      const codeContent = typeof (block as any).content === 'string' 
        ? (block as any).content 
        : Array.isArray((block as any).content)
          ? (block as any).content.map((t: any) => t.text).join('')
          : '';
      const language = (block as any).language;
      const codeNode = $createDocCodeNode((block as any).id || block.id, codeContent, language);
      
      // Add TextNode child with the code content
      const textNode = $createAiTextNode(codeContent);
      codeNode.append(textNode);
      
      return codeNode;
    }
    
    case 'blockquote':
    case 'quote': {
      // Handle quotes
      const quoteNode = $createDocQuoteNode((block as any).id || block.id);
      const content = (block as any).content;
      if (typeof content === 'string') {
        const textNode = $createAiTextNode(content);
        quoteNode.append(textNode);
      } else if (Array.isArray(content)) {
        appendTextRuns(quoteNode, content);
      }
      return quoteNode;
    }
    
    case 'divider': {
      // Handle dividers/horizontal rules
      return $createDocDividerNode((block as any).id || block.id);
    }
    
    case 'image': {
      // Handle images
      const src = (block as any).src || '';
      const description = (block as any).description;
      const widthPx = (block as any).widthPx;
      const heightPx = (block as any).heightPx;
      return $createDocImageNode((block as any).id || block.id, src, description, widthPx, heightPx);
    }
    
    case 'empty': {
      // Handle empty/blank lines
      return $createDocEmptyNode((block as any).id || block.id);
    }
    
    // TODO: Implement table, note/callout blocks
    default: {
      console.warn('[DocInitializerPlugin] Unsupported block type:', (block as any).type);
      // Fallback to paragraph
      const pNode = $createDocParagraphNode((block as any).id || `fallback-${Date.now()}`);
      const textNode = $createAiTextNode(`[Unsupported block: ${(block as any).type}]`);
      pNode.append(textNode);
      return pNode;
    }
  }
}

// Append text runs with formatting to a parent node
function appendTextRuns(parentNode: any, runs: TextRun[]) {
  if (!runs || runs.length === 0) {
    // Add empty text node for empty blocks
    const textNode = $createAiTextNode('');
    parentNode.append(textNode);
    return;
  }

  runs.forEach((run) => {
    const textNode = $createAiTextNode(run.text);
    
    // Apply formatting
    if (run.bold) textNode.toggleFormat('bold');
    if (run.italic) textNode.toggleFormat('italic');
    if (run.underline) textNode.toggleFormat('underline');
    if (run.code) textNode.toggleFormat('code');
    if (run.superscript) textNode.toggleFormat('superscript');
    if (run.subscript) textNode.toggleFormat('subscript');
    
    // Apply AI suggestion status
    if (run.aiSuggestionStatus) {
      textNode.setAiSuggestionStatus(run.aiSuggestionStatus);
    }
    
    // Apply comment IDs
    if (run.commentIds && run.commentIds.length > 0) {
      run.commentIds.forEach(commentId => {
        textNode.addCommentId(commentId);
      });
    }
    
    parentNode.append(textNode);
  });
}

