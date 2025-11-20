/**
 * Document Conversion Utilities
 * Converts between backend block_metadata format and frontend DocState format
 */

import { DocState, DocBlock, TextRun, ParagraphBlock, HeadingBlock, ListBlock, PreformattedBlock, QuoteBlock } from '@/model/docTypes';

// Backend types (matching Python backend)
export interface InlineSegment {
  text: string;
  bold?: boolean;
  italic?: boolean;
  underline?: boolean;
  code?: boolean;
  color?: string;
  backgroundColor?: string;
}

export interface BlockMetadata {
  id: string;
  type: 'paragraph' | 'heading' | 'list' | 'code' | 'blockquote' | 'divider' | 'image' | 'empty';
  content: string | InlineSegment[]; // Can be string or array of segments
  level?: number; // For headings (1-3)
  list_type?: 'bulleted' | 'numbered';
  formatting?: {
    bold?: boolean;
    italic?: boolean;
    underline?: boolean;
    color?: string;
    backgroundColor?: string;
  };
  indent_level?: number;
  section_key?: string;
}

export interface BackendDocument {
  id: string;
  title: string;
  version: string;
  block_metadata: BlockMetadata[];
}

/**
 * Convert TextRun[] to plain string
 */
function textRunsToString(runs: TextRun[] | undefined): string {
  if (!runs || !Array.isArray(runs)) return '';
  return runs.map(run => run.text).join('');
}

/**
 * Convert DocState to markdown string
 */
export function docStateToMarkdown(docState: DocState): string {
  if (!docState?.blocks || docState.blocks.length === 0) {
    return '# Empty Document\n\nNo content yet.';
  }

  return docState.blocks.map(block => {
    switch (block.type) {
      case 'heading': {
        const level = '#'.repeat(block.level);
        const text = textRunsToString(block.content);
        return `${level} ${text}`;
      }
      case 'paragraph': {
        const text = textRunsToString(block.content);
        return text || '';
      }
      case 'list': {
        const marker = block.style === 'numbered' ? '1.' : '-';
        const items = block.items.map(item => {
          const text = textRunsToString(item.content);
          return `${marker} ${text}`;
        }).join('\n');
        return items;
      }
      case 'preformatted': {
        const text = textRunsToString(block.content);
        return `\`\`\`\n${text}\n\`\`\``;
      }
      case 'quote': {
        const text = textRunsToString(block.content);
        return `> ${text}`;
      }
      case 'divider':
        return '---';
      case 'image':
        return `![${block.alt || 'image'}](${block.src})`;
      case 'empty':
        return '';
      default:
        return '';
    }
  }).join('\n\n');
}

/**
 * Convert backend BlockMetadata[] to frontend DocState
 */
export function blockMetadataToDocState(
  backendDoc: BackendDocument
): DocState {
  const blocks: DocBlock[] = backendDoc.block_metadata.map((block) => {
    const textRuns = parseContentToTextRuns(block.content, block.formatting);

    switch (block.type) {
      case 'heading':
        return {
          id: block.id,
          type: 'heading',
          level: (block.level || 1) as 1 | 2 | 3,
          content: textRuns,
          sectionKey: block.section_key,
        } as HeadingBlock;

      case 'list':
        return {
          id: block.id,
          type: 'list',
          listStyle: block.list_type === 'numbered' ? 'decimal' : 'disc',
          items: [
            {
              id: `${block.id}-item-0`,
              content: textRuns,
            }
          ],
          sectionKey: block.section_key,
        } as ListBlock;

      case 'code':
        return {
          id: block.id,
          type: 'preformatted',
          content: textRuns,
          sectionKey: block.section_key,
        } as PreformattedBlock;

      case 'blockquote':
        return {
          id: block.id,
          type: 'quote',
          content: textRuns,
          sectionKey: block.section_key,
        } as QuoteBlock;

      case 'paragraph':
      default:
        return {
          id: block.id,
          type: 'paragraph',
          content: textRuns,
          sectionKey: block.section_key,
        } as ParagraphBlock;
    }
  });

  return {
    id: backendDoc.id,
    title: backendDoc.title,
    version: backendDoc.version,
    blocks,
  };
}

/**
 * Convert frontend DocState to backend BlockMetadata[]
 */
export function docStateToBlockMetadata(docState: DocState): BlockMetadata[] {
  const blockMetadata: BlockMetadata[] = [];

  for (const block of docState.blocks) {
    switch (block.type) {
      case 'heading': {
        const headingBlock = block as HeadingBlock;
        blockMetadata.push({
          id: headingBlock.id,
          type: 'heading',
          level: headingBlock.level,
          content: textRunsToPlainText(headingBlock.content),
          formatting: extractFormatting(headingBlock.content),
          section_key: headingBlock.sectionKey,
        });
        break;
      }

      case 'list': {
        const listBlock = block as ListBlock;
        // Convert each list item to a separate block
        for (const item of listBlock.items) {
          blockMetadata.push({
            id: item.id,
            type: 'list',
            list_type: listBlock.listStyle === 'decimal' ? 'numbered' : 'bulleted',
            content: textRunsToPlainText(item.content),
            formatting: extractFormatting(item.content),
            section_key: listBlock.sectionKey,
          });
        }
        break;
      }

      case 'preformatted': {
        const codeBlock = block as PreformattedBlock;
        blockMetadata.push({
          id: codeBlock.id,
          type: 'code',
          content: textRunsToPlainText(codeBlock.content),
          section_key: codeBlock.sectionKey,
        });
        break;
      }

      case 'quote': {
        const quoteBlock = block as QuoteBlock;
        blockMetadata.push({
          id: quoteBlock.id,
          type: 'blockquote',
          content: textRunsToPlainText(quoteBlock.content),
          formatting: extractFormatting(quoteBlock.content),
          section_key: quoteBlock.sectionKey,
        });
        break;
      }

      case 'paragraph':
      default: {
        const paragraphBlock = block as ParagraphBlock;
        blockMetadata.push({
          id: paragraphBlock.id,
          type: 'paragraph',
          content: textRunsToPlainText(paragraphBlock.content),
          formatting: extractFormatting(paragraphBlock.content),
          section_key: paragraphBlock.sectionKey,
        });
        break;
      }
    }
  }

  return blockMetadata;
}

/**
 * Parse content to TextRun[] with formatting
 * Handles both string and InlineSegment[] formats from backend
 */
function parseContentToTextRuns(
  content: string | InlineSegment[],
  formatting?: BlockMetadata['formatting']
): TextRun[] {
  if (!content) {
    return [{ text: '' }];
  }

  // If content is already an array of segments, convert directly
  if (Array.isArray(content)) {
    return content.map(segment => ({
      text: segment.text,
      bold: segment.bold,
      italic: segment.italic,
      underline: segment.underline,
      code: segment.code,
      color: segment.color,
      backgroundColor: segment.backgroundColor,
    }));
  }

  // If content is a string, apply formatting to entire text
  const textRun: TextRun = { text: content };
  
  if (formatting) {
    if (formatting.bold) textRun.bold = true;
    if (formatting.italic) textRun.italic = true;
    if (formatting.underline) textRun.underline = true;
    if (formatting.color) textRun.color = formatting.color;
    if (formatting.backgroundColor) textRun.backgroundColor = formatting.backgroundColor;
  }

  return [textRun];
}

/**
 * Convert TextRun[] to plain text string
 */
function textRunsToPlainText(textRuns: TextRun[]): string {
  return textRuns.map(run => run.text).join('');
}

/**
 * Extract formatting from TextRun[] (uses first formatted run)
 */
function extractFormatting(textRuns: TextRun[]): BlockMetadata['formatting'] | undefined {
  // Find first run with formatting
  const formattedRun = textRuns.find(
    run => run.bold || run.italic || run.underline || run.color || run.backgroundColor
  );

  if (!formattedRun) {
    return undefined;
  }

  return {
    bold: formattedRun.bold,
    italic: formattedRun.italic,
    underline: formattedRun.underline,
    color: formattedRun.color,
    backgroundColor: formattedRun.backgroundColor,
  };
}

/**
 * Generate a unique block ID
 */
export function generateBlockId(): string {
  return `block-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

