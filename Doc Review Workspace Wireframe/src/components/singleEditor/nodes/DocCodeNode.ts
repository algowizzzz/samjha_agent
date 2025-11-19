// Custom code block node
import {
  ElementNode,
  EditorConfig,
  LexicalNode,
  NodeKey,
  SerializedElementNode,
  Spread,
  RangeSelection,
  $isTextNode,
} from 'lexical';
import { $createDocParagraphNode } from './DocParagraphNode';

export type SerializedDocCodeNode = Spread<
  {
    blockId: string;
    code: string;
    language?: string;
  },
  SerializedElementNode
>;

export class DocCodeNode extends ElementNode {
  __blockId: string;
  __code: string;
  __language?: string;

  static getType(): string {
    return 'doc-code';
  }

  static clone(node: DocCodeNode): DocCodeNode {
    return new DocCodeNode(node.__blockId, node.__code, node.__language, node.__key);
  }

  constructor(blockId?: string, code: string = '', language?: string, key?: NodeKey) {
    super(key);
    this.__blockId = blockId || `code-${Date.now()}`;
    this.__code = code;
    this.__language = language;
  }

  createDOM(config: EditorConfig): HTMLElement {
    const pre = document.createElement('pre');
    pre.setAttribute('data-block-id', this.__blockId);
    pre.className = 'doc-code';
    if (this.__language) {
      pre.setAttribute('data-language', this.__language);
    }
    // Don't render static content - Lexical TextNode children will render
    return pre;
  }

  updateDOM(prevNode: DocCodeNode, dom: HTMLElement): boolean {
    // Update block ID if changed
    if (prevNode.__blockId !== this.__blockId) {
      dom.setAttribute('data-block-id', this.__blockId);
    }
    
    // Update language if changed
    if (prevNode.__language !== this.__language) {
      if (this.__language) {
        dom.setAttribute('data-language', this.__language);
      } else {
        dom.removeAttribute('data-language');
      }
    }
    
    // Content is now managed by Lexical TextNode children
    return false;
  }

  static importJSON(serializedNode: SerializedDocCodeNode): DocCodeNode {
    return $createDocCodeNode(
      serializedNode.blockId,
      serializedNode.code,
      serializedNode.language
    );
  }

  exportJSON(): SerializedDocCodeNode {
    return {
      ...super.exportJSON(),
      type: 'doc-code',
      version: 1,
      blockId: this.__blockId,
      code: this.__code,
      language: this.__language,
    };
  }

  getBlockId(): string {
    return this.__blockId;
  }

  setBlockId(blockId: string): void {
    const writable = this.getWritable();
    writable.__blockId = blockId;
  }

  getCode(): string {
    return this.__code;
  }

  setCode(code: string): void {
    const writable = this.getWritable();
    writable.__code = code;
  }

  getLanguage(): string | undefined {
    return this.__language;
  }

  setLanguage(language: string | undefined): void {
    const writable = this.getWritable();
    writable.__language = language;
  }

  // Make code non-inline
  isInline(): boolean {
    return false;
  }

  // Can have TextNode children
  canBeEmpty(): boolean {
    return false;
  }

  // Handle Enter key - check if we should exit code block
  insertNewAfter(selection: RangeSelection, restoreSelection: boolean): LexicalNode | null {
    // Get the anchor node
    const anchor = selection.anchor.getNode();
    
    // Check if we're at the end of the code block and the line is empty
    const text = anchor.getTextContent();
    const isEmpty = text.trim() === '' || text.endsWith('\n\n');
    
    // If empty, exit code block and create paragraph
    if (isEmpty && selection.anchor.offset === text.length) {
      const newParagraph = $createDocParagraphNode();
      this.insertAfter(newParagraph, restoreSelection);
      return newParagraph;
    }
    
    // Otherwise, stay in code block (return null to use default behavior - line break)
    return null;
  }
}

export function $createDocCodeNode(
  blockId?: string,
  code: string = '',
  language?: string
): DocCodeNode {
  return new DocCodeNode(blockId, code, language);
}

export function $isDocCodeNode(node: LexicalNode | null | undefined): node is DocCodeNode {
  return node instanceof DocCodeNode;
}

