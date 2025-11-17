import { useEffect, useRef } from 'react';
import { LexicalComposer } from '@lexical/react/LexicalComposer';
import { RichTextPlugin } from '@lexical/react/LexicalRichTextPlugin';
import { ContentEditable } from '@lexical/react/LexicalContentEditable';
import { HistoryPlugin } from '@lexical/react/LexicalHistoryPlugin';
import { OnChangePlugin } from '@lexical/react/LexicalOnChangePlugin';
import { useLexicalComposerContext } from '@lexical/react/LexicalComposerContext';
import { HeadingNode, QuoteNode } from '@lexical/rich-text';
import { ListItemNode, ListNode } from '@lexical/list';
import { CodeNode } from '@lexical/code';
import { $getRoot, $createParagraphNode, $createTextNode, EditorState, LexicalEditor } from 'lexical';
import LexicalErrorBoundary from '@lexical/react/LexicalErrorBoundary';
import { BlockTypePlugin } from './plugins/BlockTypePlugin';
import { FormattingPlugin } from './plugins/FormattingPlugin';
import type { Block, BlockType } from './types';

interface LexicalBlockProps {
  block: Block;
  onChange: (content: string, htmlContent: string) => void;
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

      // For now, just set plain text content
      // TODO: Parse HTML content to Lexical nodes
      if (block.content) {
        const paragraph = $createParagraphNode();
        // Strip HTML tags for now - we'll implement proper HTML parsing later
        const textContent = block.content.replace(/<[^>]*>/g, '');
        const textNode = $createTextNode(textContent);
        
        // Apply formatting from block metadata
        if (block.formatting?.bold) {
          textNode.toggleFormat('bold');
        }
        if (block.formatting?.italic) {
          textNode.toggleFormat('italic');
        }
        if (block.formatting?.underline) {
          textNode.toggleFormat('underline');
        }
        
        paragraph.append(textNode);
        root.append(paragraph);
      }
    });
  }, [editor, block.id]); // Re-initialize only if block.id changes

  return null;
}

// Plugin to handle editor changes and sync back to parent
function OnChangeHandlerPlugin({ 
  onChange 
}: { 
  onChange: (content: string, htmlContent: string) => void;
}) {
  const handleChange = (editorState: EditorState, editor: LexicalEditor) => {
    editorState.read(() => {
      const root = $getRoot();
      const textContent = root.getTextContent();
      
      // For now, use simple HTML generation
      // TODO: Implement proper HTML generation with formatting preservation
      const htmlContent = textContent;
      
      onChange(textContent, htmlContent);
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

export function LexicalBlock({ 
  block, 
  onChange, 
  onKeyDown,
  autoFocus,
  className = ''
}: LexicalBlockProps) {
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
      <div className={`lexical-block-wrapper ${className}`}>
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
        <BlockTypePlugin blockType={block.type} />
        <InitializeContentPlugin block={block} />
        <OnChangeHandlerPlugin onChange={onChange} />
        <KeyboardShortcutsPlugin onKeyDown={onKeyDown} />
        <AutoFocusPlugin autoFocus={autoFocus} />
      </div>
    </LexicalComposer>
  );
}

