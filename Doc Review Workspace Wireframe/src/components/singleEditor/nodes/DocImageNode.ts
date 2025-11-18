// Image block node
import {
  ElementNode,
  EditorConfig,
  LexicalNode,
  NodeKey,
  SerializedElementNode,
  Spread,
} from 'lexical';

export type SerializedDocImageNode = Spread<
  {
    blockId: string;
    src: string;
    description?: string;
    widthPx?: number;
    heightPx?: number;
  },
  SerializedElementNode
>;

export class DocImageNode extends ElementNode {
  __blockId: string;
  __src: string;
  __description?: string;
  __widthPx?: number;
  __heightPx?: number;

  static getType(): string {
    return 'doc-image';
  }

  static clone(node: DocImageNode): DocImageNode {
    return new DocImageNode(
      node.__blockId,
      node.__src,
      node.__description,
      node.__widthPx,
      node.__heightPx,
      node.__key
    );
  }

  constructor(
    blockId?: string,
    src: string = '',
    description?: string,
    widthPx?: number,
    heightPx?: number,
    key?: NodeKey
  ) {
    super(key);
    this.__blockId = blockId || `image-${Date.now()}`;
    this.__src = src;
    this.__description = description;
    this.__widthPx = widthPx;
    this.__heightPx = heightPx;
  }

  createDOM(config: EditorConfig): HTMLElement {
    const container = document.createElement('div');
    container.setAttribute('data-block-id', this.__blockId);
    container.className = 'doc-image';

    const img = document.createElement('img');
    img.src = this.__src;
    img.alt = this.__description || 'Document image';
    if (this.__widthPx) {
      img.style.maxWidth = `${this.__widthPx}px`;
    } else {
      img.style.maxWidth = '100%';
    }
    if (this.__heightPx) {
      img.style.height = `${this.__heightPx}px`;
    } else {
      img.style.height = 'auto';
    }
    container.appendChild(img);

    if (this.__description) {
      const caption = document.createElement('div');
      caption.className = 'doc-image-caption';
      caption.textContent = this.__description;
      container.appendChild(caption);
    }

    return container;
  }

  updateDOM(prevNode: DocImageNode, dom: HTMLElement): boolean {
    // If any properties changed, recreate
    if (
      prevNode.__blockId !== this.__blockId ||
      prevNode.__src !== this.__src ||
      prevNode.__description !== this.__description ||
      prevNode.__widthPx !== this.__widthPx ||
      prevNode.__heightPx !== this.__heightPx
    ) {
      return true;
    }
    return false;
  }

  static importJSON(serializedNode: SerializedDocImageNode): DocImageNode {
    return $createDocImageNode(
      serializedNode.blockId,
      serializedNode.src,
      serializedNode.description,
      serializedNode.widthPx,
      serializedNode.heightPx
    );
  }

  exportJSON(): SerializedDocImageNode {
    return {
      ...super.exportJSON(),
      type: 'doc-image',
      version: 1,
      blockId: this.__blockId,
      src: this.__src,
      description: this.__description,
      widthPx: this.__widthPx,
      heightPx: this.__heightPx,
    };
  }

  getBlockId(): string {
    return this.__blockId;
  }

  getSrc(): string {
    return this.__src;
  }

  getDescription(): string | undefined {
    return this.__description;
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

export function $createDocImageNode(
  blockId?: string,
  src: string = '',
  description?: string,
  widthPx?: number,
  heightPx?: number
): DocImageNode {
  return new DocImageNode(blockId, src, description, widthPx, heightPx);
}

export function $isDocImageNode(node: LexicalNode | null | undefined): node is DocImageNode {
  return node instanceof DocImageNode;
}
