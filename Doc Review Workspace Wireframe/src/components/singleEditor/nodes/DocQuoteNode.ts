// Custom blockquote node
import {
  ElementNode,
  EditorConfig,
  LexicalNode,
  NodeKey,
  SerializedElementNode,
  Spread,
  RangeSelection,
} from 'lexical';
import { $createDocParagraphNode } from './DocParagraphNode';

export type SerializedDocQuoteNode = Spread<
  {
    blockId: string;
  },
  SerializedElementNode
>;

export class DocQuoteNode extends ElementNode {
  __blockId: string;

  static getType(): string {
    return 'doc-quote';
  }

  static clone(node: DocQuoteNode): DocQuoteNode {
    return new DocQuoteNode(node.__blockId, node.__key);
  }

  constructor(blockId?: string, key?: NodeKey) {
    super(key);
    this.__blockId = blockId || `quote-${Date.now()}`;
  }

  createDOM(config: EditorConfig): HTMLElement {
    const blockquote = document.createElement('blockquote');
    blockquote.setAttribute('data-block-id', this.__blockId);
    blockquote.className = 'doc-quote';
    // Quotes can have editable text, so don't set contenteditable=false
    return blockquote;
  }

  updateDOM(prevNode: DocQuoteNode, dom: HTMLElement): boolean {
    if (prevNode.__blockId !== this.__blockId) {
      dom.setAttribute('data-block-id', this.__blockId);
    }
    return false;
  }

  static importJSON(serializedNode: SerializedDocQuoteNode): DocQuoteNode {
    return $createDocQuoteNode(serializedNode.blockId);
  }

  exportJSON(): SerializedDocQuoteNode {
    return {
      ...super.exportJSON(),
      type: 'doc-quote',
      version: 1,
      blockId: this.__blockId,
    };
  }

  getBlockId(): string {
    return this.__blockId;
  }

  setBlockId(blockId: string): void {
    const writable = this.getWritable();
    writable.__blockId = blockId;
  }

  // Handle Enter key - exit quote if empty
  insertNewAfter(selection: RangeSelection, restoreSelection: boolean): LexicalNode | null {
    // Check if the quote is empty or we're at the end with empty content
    const children = this.getChildren();
    const lastChild = children[children.length - 1];
    
    // If quote is empty or last child is empty, exit quote
    if (children.length === 0 || (lastChild && lastChild.getTextContent().trim() === '')) {
      const newParagraph = $createDocParagraphNode();
      this.insertAfter(newParagraph, restoreSelection);
      return newParagraph;
    }
    
    // Otherwise, stay in quote (return null to use default behavior)
    return null;
  }
}

export function $createDocQuoteNode(blockId?: string): DocQuoteNode {
  return new DocQuoteNode(blockId);
}

export function $isDocQuoteNode(node: LexicalNode | null | undefined): node is DocQuoteNode {
  return node instanceof DocQuoteNode;
}

