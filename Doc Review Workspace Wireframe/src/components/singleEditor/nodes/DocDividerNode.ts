// Horizontal divider/separator node
import {
  ElementNode,
  EditorConfig,
  LexicalNode,
  NodeKey,
  SerializedElementNode,
  Spread,
} from 'lexical';

export type SerializedDocDividerNode = Spread<
  {
    blockId: string;
  },
  SerializedElementNode
>;

export class DocDividerNode extends ElementNode {
  __blockId: string;

  static getType(): string {
    return 'doc-divider';
  }

  static clone(node: DocDividerNode): DocDividerNode {
    return new DocDividerNode(node.__blockId, node.__key);
  }

  constructor(blockId?: string, key?: NodeKey) {
    super(key);
    this.__blockId = blockId || `divider-${Date.now()}`;
  }

  createDOM(config: EditorConfig): HTMLElement {
    const hr = document.createElement('hr');
    hr.setAttribute('data-block-id', this.__blockId);
    hr.className = 'doc-divider';
    return hr;
  }

  updateDOM(prevNode: DocDividerNode, dom: HTMLElement): boolean {
    if (prevNode.__blockId !== this.__blockId) {
      dom.setAttribute('data-block-id', this.__blockId);
    }
    return false;
  }

  static importJSON(serializedNode: SerializedDocDividerNode): DocDividerNode {
    return $createDocDividerNode(serializedNode.blockId);
  }

  exportJSON(): SerializedDocDividerNode {
    return {
      ...super.exportJSON(),
      type: 'doc-divider',
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

  isInline(): boolean {
    return false;
  }

  canBeEmpty(): boolean {
    return true;
  }

  // Prevent text insertion
  canInsertTextBefore(): boolean {
    return false;
  }

  canInsertTextAfter(): boolean {
    return false;
  }
}

export function $createDocDividerNode(blockId?: string): DocDividerNode {
  return new DocDividerNode(blockId);
}

export function $isDocDividerNode(node: LexicalNode | null | undefined): node is DocDividerNode {
  return node instanceof DocDividerNode;
}
