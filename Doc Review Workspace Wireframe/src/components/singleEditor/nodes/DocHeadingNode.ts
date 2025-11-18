// Custom heading node for document structure

import {
  ElementNode,
  SerializedElementNode,
  EditorConfig,
  LexicalNode,
  NodeKey,
  Spread,
} from 'lexical';

export type SerializedDocHeadingNode = Spread<{
  level: 1 | 2 | 3 | 4 | 5 | 6;
  sectionKey?: string;
  blockId?: string;
}, SerializedElementNode>;

export class DocHeadingNode extends ElementNode {
  __level: 1 | 2 | 3 | 4 | 5 | 6;
  __sectionKey?: string;
  __blockId: string;

  static getType(): string {
    return 'doc-heading';
  }

  static clone(node: DocHeadingNode): DocHeadingNode {
    return new DocHeadingNode(node.__level, node.__blockId, node.__sectionKey, node.__key);
  }

  constructor(
    level: 1 | 2 | 3 | 4 | 5 | 6,
    blockId?: string,
    sectionKey?: string,
    key?: NodeKey
  ) {
    super(key);
    this.__level = level;
    this.__blockId = blockId || `block-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    this.__sectionKey = sectionKey;
  }

  createDOM(config: EditorConfig): HTMLElement {
    const tag = `h${this.__level}`;
    const dom = document.createElement(tag);
    dom.setAttribute('data-doc-heading-level', String(this.__level));
    dom.className = `doc-heading doc-heading-${this.__level}`;
    if (this.__sectionKey) {
      dom.setAttribute('data-section-key', this.__sectionKey);
    }
    if (this.__blockId) {
      dom.setAttribute('data-block-id', this.__blockId);
    }
    return dom;
  }

  updateDOM(prevNode: DocHeadingNode, dom: HTMLElement): boolean {
    if (prevNode.__level !== this.__level) {
      // Level changed - need to replace the element
      return true;
    }
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
    return false;
  }

  static importJSON(serialized: SerializedDocHeadingNode): DocHeadingNode {
    const node = new DocHeadingNode(serialized.level, serialized.blockId, serialized.sectionKey);
    node.setFormat(serialized.format);
    node.setIndent(serialized.indent);
    node.setDirection(serialized.direction);
    return node;
  }

  exportJSON(): SerializedDocHeadingNode {
    return {
      ...super.exportJSON(),
      type: 'doc-heading',
      level: this.__level,
      sectionKey: this.__sectionKey,
      blockId: this.__blockId,
      version: 1,
    };
  }

  // Getters and setters
  getLevel(): 1 | 2 | 3 | 4 | 5 | 6 {
    const self = this.getLatest();
    return self.__level;
  }

  setLevel(level: 1 | 2 | 3 | 4 | 5 | 6): this {
    const self = this.getWritable();
    self.__level = level;
    return self;
  }

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

  // Ensure headings can't be empty
  canBeEmpty(): boolean {
    return false;
  }

  // Headings are block-level
  isInline(): boolean {
    return false;
  }
}

// Helper functions
export function $createDocHeadingNode(
  level: 1 | 2 | 3 | 4 | 5 | 6,
  blockId?: string,
  sectionKey?: string
): DocHeadingNode {
  return new DocHeadingNode(level, blockId, sectionKey);
}

export function $isDocHeadingNode(
  node: LexicalNode | null | undefined
): node is DocHeadingNode {
  return node instanceof DocHeadingNode;
}

