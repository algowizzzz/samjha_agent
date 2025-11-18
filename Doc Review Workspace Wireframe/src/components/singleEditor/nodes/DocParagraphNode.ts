// Custom paragraph node for document structure

import {
  ParagraphNode,
  SerializedParagraphNode,
  LexicalNode,
  Spread,
} from 'lexical';

export type SerializedDocParagraphNode = Spread<{
  sectionKey?: string;
  blockId?: string;
}, SerializedParagraphNode>;

export class DocParagraphNode extends ParagraphNode {
  __sectionKey?: string;
  __blockId: string;

  static getType(): string {
    return 'doc-paragraph';
  }

  static clone(node: DocParagraphNode): DocParagraphNode {
    const cloned = new DocParagraphNode(node.__blockId, node.__key);
    cloned.__sectionKey = node.__sectionKey;
    return cloned;
  }

  constructor(blockId?: string, key?: string) {
    super(key);
    this.__sectionKey = undefined;
    this.__blockId = blockId || `block-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  createDOM(config: any): HTMLElement {
    const dom = super.createDOM(config);
    dom.classList.add('doc-paragraph');
    if (this.__sectionKey) {
      dom.setAttribute('data-section-key', this.__sectionKey);
    }
    if (this.__blockId) {
      dom.setAttribute('data-block-id', this.__blockId);
    }
    return dom;
  }

  updateDOM(prevNode: DocParagraphNode, dom: HTMLElement, config: any): boolean {
    const shouldUpdate = super.updateDOM(prevNode, dom, config);
    if (prevNode.__sectionKey !== this.__sectionKey) {
      if (this.__sectionKey) {
        dom.setAttribute('data-section-key', this.__sectionKey);
      } else {
        dom.removeAttribute('data-section-key');
      }
    }
    if (prevNode.__blockId !== this.__blockId) {
      if (this.__blockId) {
        dom.setAttribute('data-block-id', this.__blockId);
      } else {
        dom.removeAttribute('data-block-id');
      }
    }
    return shouldUpdate;
  }

  static importJSON(
    serializedNode: SerializedDocParagraphNode
  ): DocParagraphNode {
    const node = new DocParagraphNode(serializedNode.blockId);
    node.setFormat(serializedNode.format);
    node.setIndent(serializedNode.indent);
    node.setDirection(serializedNode.direction);
    node.__sectionKey = serializedNode.sectionKey;
    return node;
  }

  exportJSON(): SerializedDocParagraphNode {
    return {
      ...super.exportJSON(),
      type: 'doc-paragraph',
      sectionKey: this.__sectionKey,
      blockId: this.__blockId,
      version: 1,
    };
  }

  // Getters and setters
  getSectionKey(): string | undefined {
    const self = this.getLatest();
    return self.__sectionKey;
  }

  setSectionKey(sectionKey: string | undefined): this {
    const self = this.getWritable();
    self.__sectionKey = sectionKey;
    return self;
  }

  getBlockId(): string {
    const self = this.getLatest();
    return self.__blockId;
  }

  setBlockId(blockId: string): this {
    const self = this.getWritable();
    self.__blockId = blockId;
    return self;
  }
}

// Helper functions
export function $createDocParagraphNode(blockId?: string, sectionKey?: string): DocParagraphNode {
  const node = new DocParagraphNode(blockId);
  if (sectionKey) {
    node.setSectionKey(sectionKey);
  }
  return node;
}

export function $isDocParagraphNode(
  node: LexicalNode | null | undefined
): node is DocParagraphNode {
  return node instanceof DocParagraphNode;
}

