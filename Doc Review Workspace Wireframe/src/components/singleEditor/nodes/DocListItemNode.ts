// Custom list item node for proper Lexical selection support
import {
  ElementNode,
  EditorConfig,
  LexicalNode,
  NodeKey,
  SerializedElementNode,
  Spread,
} from 'lexical';

export type SerializedDocListItemNode = Spread<
  {
    itemId: string;
  },
  SerializedElementNode
>;

export class DocListItemNode extends ElementNode {
  __itemId: string;

  static getType(): string {
    return 'doc-list-item';
  }

  static clone(node: DocListItemNode): DocListItemNode {
    return new DocListItemNode(node.__itemId, node.__key);
  }

  constructor(itemId?: string, key?: NodeKey) {
    super(key);
    this.__itemId = itemId || `item-${Date.now()}-${Math.random()}`;
  }

  createDOM(config: EditorConfig): HTMLElement {
    const li = document.createElement('li');
    li.setAttribute('data-item-id', this.__itemId);
    return li;
  }

  updateDOM(prevNode: DocListItemNode, dom: HTMLElement): boolean {
    if (prevNode.__itemId !== this.__itemId) {
      dom.setAttribute('data-item-id', this.__itemId);
    }
    return false;
  }

  static importJSON(serializedNode: SerializedDocListItemNode): DocListItemNode {
    return $createDocListItemNode(serializedNode.itemId);
  }

  exportJSON(): SerializedDocListItemNode {
    return {
      ...super.exportJSON(),
      type: 'doc-list-item',
      version: 1,
      itemId: this.__itemId,
    };
  }

  getItemId(): string {
    return this.__itemId;
  }

  setItemId(itemId: string): void {
    const writable = this.getWritable();
    writable.__itemId = itemId;
  }

  // Allow this to be a container for text
  canBeEmpty(): boolean {
    return false;
  }

  // Not inline
  isInline(): boolean {
    return false;
  }
}

export function $createDocListItemNode(itemId?: string): DocListItemNode {
  return new DocListItemNode(itemId);
}

export function $isDocListItemNode(node: LexicalNode | null | undefined): node is DocListItemNode {
  return node instanceof DocListItemNode;
}

