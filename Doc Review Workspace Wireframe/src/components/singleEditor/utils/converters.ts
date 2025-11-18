// Converters between old BlockMetadata format and new DocState format

import type { BlockMetadata } from '@/lib/api';
import type { DocState, DocBlock, TextRun, HeadingBlock, ParagraphBlock } from '@/model/docTypes';

/**
 * Convert old BlockMetadata[] format to new DocState format
 */
export function convertBlockMetadataToDocState(
  metadata: BlockMetadata[],
  documentId?: string
): DocState {
  const blocks: DocBlock[] = metadata.map(meta => convertSingleBlock(meta));

  return {
    id: documentId || `doc-${Date.now()}`,
    blocks,
    meta: {
      convertedFrom: 'BlockMetadata',
      convertedAt: new Date().toISOString(),
    },
  };
}

/**
 * Convert a single BlockMetadata to DocBlock
 */
function convertSingleBlock(meta: BlockMetadata): DocBlock {
  // Handle heading types
  if (meta.type === 'heading') {
    const block: HeadingBlock = {
      id: meta.id,
      type: 'heading',
      level: (meta.level || 1) as 1 | 2 | 3 | 4 | 5 | 6,
      text: parseContentToTextRuns(meta.content, meta.formatting),
      sectionKey: (meta as any).sectionKey,
      meta: {
        page: meta.page,
        block_num: meta.block_num,
        start_line: meta.start_line,
        end_line: meta.end_line,
      },
    };
    return block;
  }

  // Handle heading1, heading2, heading3 (legacy format)
  if (meta.type === 'heading1' || meta.type === 'heading2' || meta.type === 'heading3') {
    const levelMap = { heading1: 1, heading2: 2, heading3: 3 };
    const block: HeadingBlock = {
      id: meta.id,
      type: 'heading',
      level: levelMap[meta.type] as 1 | 2 | 3,
      text: parseContentToTextRuns(meta.content, meta.formatting),
      sectionKey: (meta as any).sectionKey,
      meta: {
        page: meta.page,
        block_num: meta.block_num,
      },
    };
    return block;
  }

  // Handle lists
  if (meta.type === 'bullet' || meta.type === 'bulleted_list') {
    // TODO: Implement proper list block conversion
    // For now, convert to paragraph
    const block: ParagraphBlock = {
      id: meta.id,
      type: 'paragraph',
      text: parseContentToTextRuns(meta.content, meta.formatting),
      sectionKey: (meta as any).sectionKey,
      meta: { originalType: meta.type },
    };
    return block;
  }

  // Default to paragraph
  const block: ParagraphBlock = {
    id: meta.id,
    type: 'paragraph',
    text: parseContentToTextRuns(meta.content, meta.formatting),
    sectionKey: (meta as any).sectionKey,
    meta: {
      page: meta.page,
      block_num: meta.block_num,
    },
  };
  return block;
}

/**
 * Parse content (string or InlineSegment[]) to TextRun[]
 */
function parseContentToTextRuns(
  content: string | any[],
  formatting?: any
): TextRun[] {
  // If content is already an array of InlineSegments, use it
  if (Array.isArray(content)) {
    return content.map(segment => ({
      text: segment.text || '',
      bold: segment.bold,
      italic: segment.italic,
      underline: segment.underline,
      code: segment.code,
      aiSuggestionStatus: segment.aiSuggestionStatus,
      commentIds: segment.commentIds,
    }));
  }

  // If content is a string, parse it
  if (typeof content === 'string') {
    // If there's block-level formatting, apply it to the whole text
    if (formatting) {
      return [{
        text: stripHtml(content),
        bold: formatting.bold || formatting.has_bold,
        italic: formatting.italic || formatting.has_italic,
        underline: formatting.underline,
      }];
    }

    // Try to parse HTML to extract formatting
    return parseHtmlToTextRuns(content);
  }

  // Fallback
  return [{ text: String(content) }];
}

/**
 * Parse HTML string to TextRun[] with formatting
 */
function parseHtmlToTextRuns(html: string): TextRun[] {
  // Simple HTML parser for common tags
  const runs: TextRun[] = [];
  
  // Remove line breaks
  html = html.replace(/<br\s*\/?>/gi, ' ');
  
  // Match text with optional formatting tags
  const regex = /<(strong|b|em|i|u|code)>(.*?)<\/\1>|([^<]+)/gi;
  let match;
  
  while ((match = regex.exec(html)) !== null) {
    if (match[1]) {
      // Formatted text
      const tag = match[1].toLowerCase();
      const text = stripHtml(match[2]);
      
      if (text) {
        runs.push({
          text,
          bold: tag === 'strong' || tag === 'b',
          italic: tag === 'em' || tag === 'i',
          underline: tag === 'u',
          code: tag === 'code',
        });
      }
    } else if (match[3]) {
      // Plain text
      const text = stripHtml(match[3]);
      if (text) {
        runs.push({ text });
      }
    }
  }
  
  // If no runs extracted, return plain text
  if (runs.length === 0) {
    const plainText = stripHtml(html);
    if (plainText) {
      runs.push({ text: plainText });
    }
  }
  
  return runs;
}

/**
 * Strip HTML tags from text
 */
function stripHtml(html: string): string {
  return html
    .replace(/<[^>]*>/g, '')
    .replace(/&nbsp;/g, ' ')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&')
    .trim();
}

/**
 * Convert DocState back to BlockMetadata[] for backward compatibility
 */
export function convertDocStateToBlockMetadata(docState: DocState): BlockMetadata[] {
  return docState.blocks.map((block, index) => {
    const baseMetadata: Partial<BlockMetadata> = {
      id: block.id,
      page: block.meta?.page || 1,
      block_num: block.meta?.block_num || index,
      start_line: block.meta?.start_line || 0,
      end_line: block.meta?.end_line || 0,
    };

    if (block.type === 'heading') {
      return {
        ...baseMetadata,
        type: 'heading',
        level: block.level,
        content: textRunsToString(block.text),
      } as BlockMetadata;
    }

    if (block.type === 'paragraph') {
      return {
        ...baseMetadata,
        type: 'paragraph',
        content: textRunsToString(block.text),
      } as BlockMetadata;
    }

    // Default
    return {
      ...baseMetadata,
      type: 'paragraph',
      content: '',
    } as BlockMetadata;
  });
}

/**
 * Convert TextRun[] back to a plain string
 */
function textRunsToString(runs: TextRun[]): string {
  return runs.map(run => run.text).join('');
}

/**
 * Convert TextRun[] to HTML with formatting
 */
export function textRunsToHtml(runs: TextRun[]): string {
  return runs.map(run => {
    let text = run.text;
    
    if (run.bold) text = `<strong>${text}</strong>`;
    if (run.italic) text = `<em>${text}</em>`;
    if (run.underline) text = `<u>${text}</u>`;
    if (run.code) text = `<code>${text}</code>`;
    
    return text;
  }).join('');
}

/**
 * Merge adjacent TextRuns with same formatting to reduce redundancy
 */
export function mergeTextRuns(runs: TextRun[]): TextRun[] {
  if (runs.length <= 1) return runs;
  
  const merged: TextRun[] = [];
  let current = { ...runs[0] };
  
  for (let i = 1; i < runs.length; i++) {
    const next = runs[i];
    
    // Check if formatting matches
    if (
      current.bold === next.bold &&
      current.italic === next.italic &&
      current.underline === next.underline &&
      current.code === next.code &&
      current.aiSuggestionStatus === next.aiSuggestionStatus
    ) {
      // Merge text
      current.text += next.text;
    } else {
      // Push current and start new
      merged.push(current);
      current = { ...next };
    }
  }
  
  // Push last
  merged.push(current);
  
  return merged;
}

