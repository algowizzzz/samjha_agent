export type BlockType = 
  | 'paragraph' 
  | 'heading1' 
  | 'heading2' 
  | 'heading3' 
  | 'bullet' 
  | 'numbered' 
  | 'quote' 
  | 'code'
  | 'checkbox'
  | 'table' 
  | 'callout' 
  | 'empty';

export type ChangeType = 
  | 'verified' 
  | 'modified' 
  | 'ai_suggested' 
  | 'ai_applied' 
  | 'rejected' 
  | 'none';

export interface ChangeRecord {
  timestamp: string;
  type: ChangeType;
  original: string;
  modified: string;
  reason?: string;
  user?: string;
}

export interface BlockFormatting {
  bold?: boolean;
  italic?: boolean;
  underline?: boolean;
  strikethrough?: boolean;
  code?: boolean;
  highlight?: boolean;
  color?: string;
  backgroundColor?: string;
  has_bold?: boolean;
  has_italic?: boolean;
  has_highlight?: boolean;
  alignment?: 'left' | 'center' | 'right';
  size?: 'small' | 'normal' | 'large';
}

// Inline text segment with formatting (matches API BlockMetadata.InlineSegment)
export interface InlineSegment {
  text: string;
  bold?: boolean;
  italic?: boolean;
  code?: boolean;
  underline?: boolean;
  link?: string;
}

export interface Block {
  id: string;
  type: BlockType;
  content: string;  // Legacy plain text/HTML - kept for backward compatibility
  richContent?: InlineSegment[];  // NEW: Structured content with formatting
  changeType: ChangeType;
  commentCount: number;
  suggestion?: any;
  aiSuggestion?: any;
  changeHistory: ChangeRecord[];
  formatting?: BlockFormatting;
  indent_level?: number;
  checked?: boolean; // for checkbox blocks
  metadata?: Record<string, any>;
}

