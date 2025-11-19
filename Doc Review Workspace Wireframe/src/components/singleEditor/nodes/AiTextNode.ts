// Custom TextNode that supports AI suggestion status tracking

import {
  TextNode,
  SerializedTextNode,
  EditorConfig,
  LexicalNode,
} from 'lexical';
import type { AiSuggestionStatus } from '@/model/docTypes';

export type SerializedAiTextNode = SerializedTextNode & {
  type: 'ai-text';
  version: 1;
  aiSuggestionStatus?: AiSuggestionStatus;
  commentIds?: string[];
};

export class AiTextNode extends TextNode {
  __aiSuggestionStatus: AiSuggestionStatus;
  __commentIds?: string[];

  static getType(): string {
    return 'ai-text';
  }

  static clone(node: AiTextNode): AiTextNode {
    const cloned = new AiTextNode(node.__text, node.__key);
    cloned.__format = node.__format;
    cloned.__style = node.__style;
    cloned.__detail = node.__detail;
    cloned.__mode = node.__mode;
    cloned.__aiSuggestionStatus = node.__aiSuggestionStatus;
    cloned.__commentIds = node.__commentIds;
    return cloned;
  }

  constructor(text: string, key?: string) {
    super(text, key);
    this.__aiSuggestionStatus = null;
    this.__commentIds = undefined;
  }

  // Create DOM and apply CSS classes based on AI status
  createDOM(config: EditorConfig): HTMLElement {
    const dom = super.createDOM(config);
    this._applyAiStatusClass(dom);
    return dom;
  }

  updateDOM(prevNode: AiTextNode, dom: HTMLElement, config: EditorConfig): boolean {
    const shouldUpdate = super.updateDOM(prevNode, dom, config);
    
    // Re-apply classes if AI status or comment IDs changed
    const commentIdsChanged = JSON.stringify(prevNode.__commentIds) !== JSON.stringify(this.__commentIds);
    if (prevNode.__aiSuggestionStatus !== this.__aiSuggestionStatus || commentIdsChanged) {
      this._applyAiStatusClass(dom);
    }
    
    return shouldUpdate;
  }

  _applyAiStatusClass(dom: HTMLElement) {
    // Remove all AI status classes
    dom.classList.remove(
      'ai-suggestion',
      'ai-suggestion-applied',
      'ai-suggestion-rejected'
    );
    
    // Apply appropriate class based on current status
    if (this.__aiSuggestionStatus === 'suggested') {
      dom.classList.add('ai-suggestion');
    } else if (this.__aiSuggestionStatus === 'applied') {
      dom.classList.add('ai-suggestion-applied');
    } else if (this.__aiSuggestionStatus === 'rejected') {
      dom.classList.add('ai-suggestion-rejected');
    }

    // Add comment indicator if comments exist
    if (this.__commentIds && this.__commentIds.length > 0) {
      dom.classList.add('has-comments');
      dom.setAttribute('data-comment-ids', this.__commentIds.join(','));
    } else {
      dom.classList.remove('has-comments');
      dom.removeAttribute('data-comment-ids');
    }
  }

  // Serialization
  static importJSON(serializedNode: SerializedAiTextNode): AiTextNode {
    const node = new AiTextNode(serializedNode.text);
    node.__format = serializedNode.format;
    node.__detail = serializedNode.detail;
    node.__mode = serializedNode.mode;
    node.__style = serializedNode.style;
    node.__aiSuggestionStatus = serializedNode.aiSuggestionStatus ?? null;
    node.__commentIds = serializedNode.commentIds;
    return node;
  }

  exportJSON(): SerializedAiTextNode {
    return {
      ...(super.exportJSON() as SerializedTextNode),
      type: 'ai-text',
      version: 1,
      aiSuggestionStatus: this.__aiSuggestionStatus ?? undefined,
      commentIds: this.__commentIds,
    };
  }

  // API to set AI suggestion status
  setAiSuggestionStatus(status: AiSuggestionStatus): this {
    const self = this.getWritable();
    self.__aiSuggestionStatus = status;
    return self;
  }

  getAiSuggestionStatus(): AiSuggestionStatus {
    const self = this.getLatest();
    return self.__aiSuggestionStatus;
  }

  // API for comments
  addCommentId(commentId: string): this {
    const self = this.getWritable();
    if (!self.__commentIds) {
      self.__commentIds = [];
    }
    if (!self.__commentIds.includes(commentId)) {
      self.__commentIds.push(commentId);
    }
    return self;
  }

  removeCommentId(commentId: string): this {
    const self = this.getWritable();
    if (self.__commentIds) {
      self.__commentIds = self.__commentIds.filter(id => id !== commentId);
      if (self.__commentIds.length === 0) {
        self.__commentIds = undefined;
      }
    }
    return self;
  }

  getCommentIds(): string[] {
    const self = this.getLatest();
    return self.__commentIds ?? [];
  }
}

// Helper functions
export function $createAiTextNode(text: string): AiTextNode {
  return new AiTextNode(text);
}

export function $isAiTextNode(
  node: LexicalNode | null | undefined
): node is AiTextNode {
  return node instanceof AiTextNode;
}

