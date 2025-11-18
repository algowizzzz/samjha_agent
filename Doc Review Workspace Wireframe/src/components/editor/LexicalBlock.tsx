import React, { useEffect, useRef } from 'react';
import { LexicalComposer } from '@lexical/react/LexicalComposer';
import { RichTextPlugin } from '@lexical/react/LexicalRichTextPlugin';
import { ContentEditable } from '@lexical/react/LexicalContentEditable';
import { HistoryPlugin } from '@lexical/react/LexicalHistoryPlugin';
import { OnChangePlugin } from '@lexical/react/LexicalOnChangePlugin';
import { useLexicalComposerContext } from '@lexical/react/LexicalComposerContext';
import { HeadingNode, QuoteNode } from '@lexical/rich-text';
import { ListItemNode, ListNode } from '@lexical/list';
import { CodeNode } from '@lexical/code';
import { $getRoot, $createParagraphNode, $createTextNode, EditorState, LexicalEditor, TextNode } from 'lexical';
import LexicalErrorBoundary from '@lexical/react/LexicalErrorBoundary';
import { BlockTypePlugin } from './plugins/BlockTypePlugin';
import { FormattingPlugin } from './plugins/FormattingPlugin';
import { FloatingFormattingToolbar } from './plugins/FloatingFormattingToolbar';
import type { Block, BlockType, InlineSegment } from './types';

interface LexicalBlockProps {
  block: Block;
  onChange: (content: string, htmlContent: string, richContent?: InlineSegment[]) => void;
  onBlur?: () => void;  // NEW: Called when user stops editing
  onKeyDown?: (e: React.KeyboardEvent) => void;
  autoFocus?: boolean;
  className?: string;
}

// Lexical editor theme configuration
const theme = {
  paragraph: 'lexical-paragraph',
  heading: {
    h1: 'lexical-h1',
    h2: 'lexical-h2',
    h3: 'lexical-h3',
  },
  list: {
    ul: 'lexical-ul',
    ol: 'lexical-ol',
    listitem: 'lexical-li',
  },
  quote: 'lexical-quote',
  code: 'lexical-code',
  text: {
    bold: 'lexical-bold',
    italic: 'lexical-italic',
    underline: 'lexical-underline',
    strikethrough: 'lexical-strikethrough',
    code: 'lexical-inline-code',
  },
};

// Plugin to initialize editor content from block
function InitializeContentPlugin({ block }: { block: Block }) {
  const [editor] = useLexicalComposerContext();
  const initializedRef = useRef(false);

  useEffect(() => {
    // Only initialize once when component mounts
    if (initializedRef.current) return;
    initializedRef.current = true;

    editor.update(() => {
      const root = $getRoot();
      root.clear();

      // ✅ NEW: Prefer richContent if available
      if (block.richContent && block.richContent.length > 0) {
        const paragraph = $createParagraphNode();
        
        block.richContent.forEach(segment => {
          const textNode = $createTextNode(segment.text);
          
          // Apply formatting from segment
          if (segment.bold) textNode.toggleFormat('bold');
          if (segment.italic) textNode.toggleFormat('italic');
          if (segment.underline) textNode.toggleFormat('underline');
          if (segment.code) textNode.toggleFormat('code');
          
          paragraph.append(textNode);
        });
        
        root.append(paragraph);
      } else if (block.content) {
        // Fallback: Plain text content (legacy)
        const paragraph = $createParagraphNode();
        // Strip HTML tags for legacy content
        const textContent = block.content.replace(/<[^>]*>/g, '');
        const textNode = $createTextNode(textContent);
        
        // Apply block-level formatting if available
        if (block.formatting?.bold || block.formatting?.has_bold) {
          textNode.toggleFormat('bold');
        }
        if (block.formatting?.italic || block.formatting?.has_italic) {
          textNode.toggleFormat('italic');
        }
        if (block.formatting?.underline) {
          textNode.toggleFormat('underline');
        }
        
        paragraph.append(textNode);
        root.append(paragraph);
      }
    });
  }, [editor]); // ✅ FIXED: Removed block.id dependency - only init once per mount

  return null;
}

// Plugin to handle editor changes and sync back to parent
function OnChangeHandlerPlugin({ 
  onChange 
}: { 
  onChange: (content: string, htmlContent: string, richContent?: InlineSegment[]) => void;
}) {
  const handleChange = (editorState: EditorState, editor: LexicalEditor) => {
    editorState.read(() => {
      const root = $getRoot();
      const textContent = root.getTextContent();
      
      // ✅ NEW: Serialize to InlineSegment[] to preserve formatting
      const richContent: InlineSegment[] = [];
      
      root.getChildren().forEach(child => {
        if (child.getType() === 'paragraph') {
          const textNodes = child.getChildren();
          textNodes.forEach(node => {
            if (node.getType() === 'text') {
              const textNode = node as TextNode;
              const text = textNode.getTextContent();
              
              if (text.length > 0) {
                richContent.push({
                  text,
                  bold: textNode.hasFormat('bold'),
                  italic: textNode.hasFormat('italic'),
                  underline: textNode.hasFormat('underline'),
                  code: textNode.hasFormat('code'),
                });
              }
            }
          });
        }
      });
      
      // Generate HTML for backward compatibility
      let htmlContent = '';
      richContent.forEach(segment => {
        let segmentHtml = segment.text;
        if (segment.bold) segmentHtml = `<strong>${segmentHtml}</strong>`;
        if (segment.italic) segmentHtml = `<em>${segmentHtml}</em>`;
        if (segment.underline) segmentHtml = `<u>${segmentHtml}</u>`;
        if (segment.code) segmentHtml = `<code>${segmentHtml}</code>`;
        htmlContent += segmentHtml;
      });
      
      onChange(textContent, htmlContent, richContent);
    });
  };

  return <OnChangePlugin onChange={handleChange} ignoreSelectionChange />;
}

// Plugin to handle keyboard shortcuts
function KeyboardShortcutsPlugin({ 
  onKeyDown 
}: { 
  onKeyDown?: (e: React.KeyboardEvent) => void;
}) {
  const [editor] = useLexicalComposerContext();
  const contentEditableRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!onKeyDown) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // Create a React synthetic event
      const reactEvent = e as any as React.KeyboardEvent;
      onKeyDown(reactEvent);
    };

    const contentEditable = editor.getRootElement();
    if (contentEditable) {
      contentEditableRef.current = contentEditable;
      contentEditable.addEventListener('keydown', handleKeyDown);
    }

    return () => {
      if (contentEditableRef.current) {
        contentEditableRef.current.removeEventListener('keydown', handleKeyDown);
      }
    };
  }, [editor, onKeyDown]);

  return null;
}

// Auto-focus plugin
function AutoFocusPlugin({ autoFocus }: { autoFocus?: boolean }) {
  const [editor] = useLexicalComposerContext();

  useEffect(() => {
    if (autoFocus) {
      editor.focus();
    }
  }, [editor, autoFocus]);

  return null;
}

// ✅ Memoize component to prevent re-renders when parent updates
export const LexicalBlock = React.memo(({ 
  block, 
  onChange,
  onBlur,
  onKeyDown,
  autoFocus,
  className = ''
}: LexicalBlockProps) => {
  const initialConfig = {
    namespace: `LexicalBlock-${block.id}`,
    theme,
    onError: (error: Error) => {
      console.error('Lexical error:', error);
    },
    nodes: [
      HeadingNode,
      QuoteNode,
      ListNode,
      ListItemNode,
      CodeNode,
    ],
  };

  return (
    <LexicalComposer initialConfig={initialConfig}>
      <div 
        className={`lexical-block-wrapper ${className}`}
        onBlur={(e) => {
          // ✅ Sync content back to parent on blur
          if (onBlur && !e.currentTarget.contains(e.relatedTarget as Node)) {
            onBlur();
          }
        }}
      >
        <RichTextPlugin
          contentEditable={
            <ContentEditable 
              className="lexical-content-editable outline-none"
              style={{ minHeight: '1.5rem' }}
            />
          }
          placeholder={null}
          ErrorBoundary={LexicalErrorBoundary}
        />
            <HistoryPlugin />
            <FormattingPlugin />
            <FloatingFormattingToolbar />
            <BlockTypePlugin blockType={block.type} />
            <InitializeContentPlugin block={block} />
            <OnChangeHandlerPlugin onChange={onChange} />
            <KeyboardShortcutsPlugin onKeyDown={onKeyDown} />
            <AutoFocusPlugin autoFocus={autoFocus} />
      </div>
    </LexicalComposer>
  );
}, (prevProps, nextProps) => {
  // Only re-render if block.id changes or autoFocus changes
  // Don't re-render when content changes - Lexical handles that internally
  return prevProps.block.id === nextProps.block.id && 
         prevProps.autoFocus === nextProps.autoFocus;
});

