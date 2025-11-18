// Configuration for the single document editor

import type { InitialConfigType } from '@lexical/react/LexicalComposer';
import { HeadingNode, QuoteNode } from '@lexical/rich-text';
import { ListNode, ListItemNode } from '@lexical/list';
import { CodeNode } from '@lexical/code';
import { AiTextNode } from './nodes/AiTextNode';
import { DocHeadingNode } from './nodes/DocHeadingNode';
import { DocParagraphNode } from './nodes/DocParagraphNode';
import { DocListNode } from './nodes/DocListNode';
import { DocListItemNode } from './nodes/DocListItemNode';
import { DocCodeNode } from './nodes/DocCodeNode';
import { DocQuoteNode } from './nodes/DocQuoteNode';
import { DocDividerNode } from './nodes/DocDividerNode';
import { DocImageNode } from './nodes/DocImageNode';
import { DocEmptyNode } from './nodes/DocEmptyNode';

export const singleDocEditorConfig: InitialConfigType = {
  namespace: 'doc-review-single-editor',
  theme: {
    // Lexical theme classes
    paragraph: 'doc-paragraph',
    text: {
      bold: 'font-bold',
      italic: 'italic',
      underline: 'underline',
      strikethrough: 'line-through',
      code: 'font-mono bg-gray-100 px-1 rounded',
    },
    list: {
      ul: 'list-disc list-inside my-2',
      ol: 'list-decimal list-inside my-2',
      listitem: 'ml-4',
    },
    quote: 'border-l-4 border-gray-300 pl-4 italic text-gray-700 my-2',
    code: 'bg-gray-100 font-mono p-2 rounded block my-2',
  },
  onError(error: Error) {
    console.error('[SingleDocEditor] Lexical error:', error);
    throw error;
  },
  nodes: [
    // Custom nodes
    AiTextNode,
    DocHeadingNode,
    DocParagraphNode,
    DocListNode,
    DocListItemNode,
    DocCodeNode,
    DocQuoteNode,
    DocDividerNode,
    DocImageNode,
    DocEmptyNode,
    
    // Standard Lexical nodes
    HeadingNode,
    QuoteNode,
    ListNode,
    ListItemNode,
    CodeNode,
  ],
  editable: true,
};

