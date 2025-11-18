// Empty/blank line node to preserve spacing
import {
  ElementNode,
  EditorConfig,
  LexicalNode,
  NodeKey,
  SerializedElementNode,
  Spread,
} from 'lexical';

export type SerializedDocEmptyNode = Spread<
  {
    blockId: string;
  },
  SerializedElementNode
>;

export class DocEmptyNode extends ElementNode {
  __blockId: string;

  static getType(): string {
    return 'doc-empty';
  }

  static clone(node: DocEmptyNode): DocEmptyNode {
    return new DocEmptyNode(node.__blockId, node.__key);
  }

  constructor(blockId?: string, key?: NodeKey) {
    super(key);
    this.__blockId = blockId || `empty-${Date.now()}`;
  }

  createDOM(config: EditorConfig): HTMLElement {
    const div = document.createElement('div');
    div.setAttribute('data-block-id', this.__blockId);
    div.className = 'doc-empty';
    div.innerHTML = '&nbsp;'; // Non-breaking space to maintain height
    return div;
  }

  updateDOM(prevNode: DocEmptyNode, dom: HTMLElement): boolean {
    if (prevNode.__blockId !== this.__blockId) {
      dom.setAttribute('data-block-id', this.__blockId);
    }
    return false;
  }

  static importJSON(serializedNode: SerializedDocEmptyNode): DocEmptyNode {
    return $createDocEmptyNode(serializedNode.blockId);
  }

  exportJSON(): SerializedDocEmptyNode {
    return {
      ...super.exportJSON(),
      type: 'doc-empty',
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

export function $createDocEmptyNode(blockId?: string): DocEmptyNode {
  return new DocEmptyNode(blockId);
}

export function $isDocEmptyNode(node: LexicalNode | null | undefined): node is DocEmptyNode {
  return node instanceof DocEmptyNode;
}
