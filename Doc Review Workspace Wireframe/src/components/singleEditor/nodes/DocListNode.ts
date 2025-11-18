// Custom list node for BlockEditor-compatible lists
import {
  ElementNode,
  EditorConfig,
  LexicalNode,
  NodeKey,
  SerializedElementNode,
  Spread,
} from 'lexical';

export type ListStyle = 'bullet' | 'number';

export type SerializedDocListNode = Spread<
  {
    blockId: string;
    listStyle: ListStyle;
    items: Array<{ content: string; children?: Array<{ content: string }> }>;
  },
  SerializedElementNode
>;

export class DocListNode extends ElementNode {
  __blockId: string;
  __listStyle: ListStyle;
  __items: Array<{ content: string; children?: Array<{ content: string }> }>;

  static getType(): string {
    return 'doc-list';
  }

  static clone(node: DocListNode): DocListNode {
    return new DocListNode(node.__blockId, node.__listStyle, node.__items, node.__key);
  }

  constructor(
    blockId?: string,
    listStyle: ListStyle = 'bullet',
    items: Array<{ content: string; children?: Array<{ content: string }> }> = [],
    key?: NodeKey
  ) {
    super(key);
    this.__blockId = blockId || `list-${Date.now()}`;
    this.__listStyle = listStyle;
    this.__items = items;
  }

  createDOM(config: EditorConfig): HTMLElement {
    const dom = document.createElement(this.__listStyle === 'bullet' ? 'ul' : 'ol');
    dom.setAttribute('data-block-id', this.__blockId);
    dom.className = 'doc-list';
    // Don't render static items - Lexical children (DocListItemNode) will render
    return dom;
  }

  updateDOM(prevNode: DocListNode, dom: HTMLElement, config: EditorConfig): boolean {
    // Update list type if changed
    const newTag = this.__listStyle === 'bullet' ? 'ul' : 'ol';
    if (dom.tagName.toLowerCase() !== newTag) {
      return true; // Recreate DOM
    }
    
    // Update block ID
    if (prevNode.__blockId !== this.__blockId) {
      dom.setAttribute('data-block-id', this.__blockId);
    }
    
    // Items are now managed by Lexical children, not by this node
    return false;
  }

  static importJSON(serializedNode: SerializedDocListNode): DocListNode {
    const node = $createDocListNode(
      serializedNode.blockId,
      serializedNode.listStyle,
      serializedNode.items
    );
    return node;
  }

  exportJSON(): SerializedDocListNode {
    return {
      ...super.exportJSON(),
      type: 'doc-list',
      version: 1,
      blockId: this.__blockId,
      listStyle: this.__listStyle,
      items: this.__items,
    };
  }

  // Getters and setters
  getBlockId(): string {
    return this.__blockId;
  }

  setBlockId(blockId: string): void {
    const writable = this.getWritable();
    writable.__blockId = blockId;
  }

  getListStyle(): ListStyle {
    return this.__listStyle;
  }

  setListStyle(listStyle: ListStyle): void {
    const writable = this.getWritable();
    writable.__listStyle = listStyle;
  }

  getItems(): Array<{ content: string; children?: Array<{ content: string }> }> {
    return this.__items;
  }

  setItems(items: Array<{ content: string; children?: Array<{ content: string }> }>): void {
    const writable = this.getWritable();
    writable.__items = items;
  }

  // Make list non-inline
  isInline(): boolean {
    return false;
  }

  // Can have children (DocListItemNodes)
  canBeEmpty(): boolean {
    return false;
  }
}

export function $createDocListNode(
  blockId?: string,
  listStyle: ListStyle = 'bullet',
  items: Array<{ content: string; children?: Array<{ content: string }> }> = []
): DocListNode {
  return new DocListNode(blockId, listStyle, items);
}

export function $isDocListNode(node: LexicalNode | null | undefined): node is DocListNode {
  return node instanceof DocListNode;
}

