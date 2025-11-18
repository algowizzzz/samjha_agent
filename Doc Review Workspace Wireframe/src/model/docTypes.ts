// Core type definitions for the single-editor document model

// AI suggestion status for text runs
export type AiSuggestionStatus = 'suggested' | 'applied' | 'rejected' | null;

// Inline text run with formatting marks
export interface TextRun {
  text: string;
  bold?: boolean;
  italic?: boolean;
  underline?: boolean;
  code?: boolean;
  superscript?: boolean;
  subscript?: boolean;
  color?: string;
  backgroundColor?: string;
  aiSuggestionStatus?: AiSuggestionStatus;
  commentIds?: string[]; // Links to comment threads
}

// Base interface for all document blocks
export interface DocBlockBase {
  id: string;              // Stable block ID
  sectionKey?: string;     // For template compliance (e.g., "overview", "scope")
  order?: number;          // Explicit ordering
  meta?: Record<string, any>; // OSFI refs, page numbers, etc.
}

// Heading block
export interface HeadingBlock extends DocBlockBase {
  type: 'heading';
  level: 1 | 2 | 3 | 4 | 5 | 6;
  text: TextRun[];
}

// Paragraph block
export interface ParagraphBlock extends DocBlockBase {
  type: 'paragraph';
  text: TextRun[];
}

// List item (can be nested)
export interface ListItem {
  id: string;
  text: TextRun[];
  children?: ListItem[];
}

// List block
export interface ListBlock extends DocBlockBase {
  type: 'list';
  style: 'bullet' | 'number';
  items: ListItem[];
}

// Table block
export interface TableBlock extends DocBlockBase {
  type: 'table';
  columns: string[];      // Header labels
  rows: string[][];       // Cell contents (could be TextRun[][] for rich cells)
}

// Divider/separator
export interface DividerBlock extends DocBlockBase {
  type: 'divider';
}

// Note/callout block
export interface NoteBlock extends DocBlockBase {
  type: 'note';
  text: TextRun[];
}

// Image block
export interface ImageBlock extends DocBlockBase {
  type: 'image';
  src: string;            // Base64 or URL
  description?: string;
  widthPx?: number;
  heightPx?: number;
}

// Preformatted/code block
export interface PreformattedBlock extends DocBlockBase {
  type: 'preformatted';
  text: string;           // Raw text with preserved spacing
  language?: string;
}

// Union of all block types
export type DocBlock =
  | HeadingBlock
  | ParagraphBlock
  | ListBlock
  | TableBlock
  | DividerBlock
  | NoteBlock
  | ImageBlock
  | PreformattedBlock;

// Complete document state
export interface DocState {
  id: string;             // Document ID
  title?: string;
  version?: string;
  blocks: DocBlock[];
  meta?: Record<string, any>; // Provenance, timestamps, OSFI guideline ID, etc.
}

