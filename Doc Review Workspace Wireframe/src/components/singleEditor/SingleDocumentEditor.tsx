// Main single-document editor component

import React, { useMemo } from 'react';
import { LexicalComposer } from '@lexical/react/LexicalComposer';
import { RichTextPlugin } from '@lexical/react/LexicalRichTextPlugin';
import { ContentEditable } from '@lexical/react/LexicalContentEditable';
import { HistoryPlugin } from '@lexical/react/LexicalHistoryPlugin';
import LexicalErrorBoundary from '@lexical/react/LexicalErrorBoundary';
import { useLexicalComposerContext } from '@lexical/react/LexicalComposerContext';
import { useEffect } from 'react';
import { singleDocEditorConfig } from './SingleDocEditorConfig';
import { DocInitializerPlugin } from './plugins/DocInitializerPlugin';
import { DocExportOnChangePlugin } from './plugins/DocExportOnChangePlugin';
import { SelectionBridgePlugin, SelectionData } from './plugins/SelectionBridgePlugin';
import { EnterKeyPlugin } from './plugins/EnterKeyPlugin';
import { FloatingToolbarPlugin } from './plugins/FloatingToolbarPlugin';
import { CommentClickPlugin } from './plugins/CommentClickPlugin';
import type { DocState } from '@/model/docTypes';
import type { LexicalEditor } from 'lexical';

interface SingleDocumentEditorProps {
  initialDoc: DocState;
  onDocChange: (doc: DocState) => void;
  readOnly?: boolean;
  className?: string;
  onEditorReady?: (editor: LexicalEditor) => void;
  onSelectionChange?: (data: SelectionData) => void;
  // Floating toolbar handlers
  onFormat?: (format: 'bold' | 'italic' | 'underline' | 'strikethrough') => void;
  onTextColor?: (color: string) => void;
  onBackgroundColor?: (color: string) => void;
  onTurnInto?: (type: string) => void;
  onAddComment?: () => void;
  onImproveText?: () => void;
  // Comment click handler
  onCommentClick?: (commentIds: string[]) => void;
}

// Plugin to expose editor instance to parent
function EditorRefPlugin({ onReady }: { onReady?: (editor: LexicalEditor) => void }) {
  const [editor] = useLexicalComposerContext();
  
  useEffect(() => {
    if (onReady) {
      onReady(editor);
    }
  }, [editor, onReady]);
  
  return null;
}

export function SingleDocumentEditor({
  initialDoc,
  onDocChange,
  readOnly = false,
  className = '',
  onEditorReady,
  onSelectionChange,
  onFormat,
  onTextColor,
  onBackgroundColor,
  onTurnInto,
  onAddComment,
  onImproveText,
  onCommentClick,
}: SingleDocumentEditorProps) {
  const config = useMemo(
    () => ({
      ...singleDocEditorConfig,
      editable: !readOnly,
    }),
    [readOnly]
  );

  return (
    <LexicalComposer initialConfig={config}>
      <div className={`single-doc-editor-root ${className}`}>
        {/* Expose editor instance to parent */}
        <EditorRefPlugin onReady={onEditorReady} />
        
        {/* Initialize editor from DocState */}
        <DocInitializerPlugin initialDoc={initialDoc} />
        
        {/* Track selection and extract block IDs */}
        <SelectionBridgePlugin onSelectionChange={onSelectionChange} />
        
        {/* Handle Enter key for custom nodes */}
        <EnterKeyPlugin />
        
        {/* Handle clicks on commented text */}
        {onCommentClick && <CommentClickPlugin onCommentClick={onCommentClick} />}
        
        {/* Floating toolbar on text selection */}
        {onFormat && onTurnInto && onAddComment && onImproveText && (
          <FloatingToolbarPlugin
            onFormat={onFormat}
            onTextColor={onTextColor || (() => {})}
            onBackgroundColor={onBackgroundColor || (() => {})}
            onTurnInto={onTurnInto}
            onAddComment={onAddComment}
            onImproveText={onImproveText}
          />
        )}
        
        {/* Main editor UI */}
        <RichTextPlugin
          contentEditable={
            <ContentEditable 
              className="single-doc-editor-content outline-none min-h-screen px-16 py-12 max-w-5xl mx-auto"
              aria-label="Document editor"
              style={{
                fontSize: '16px',
                lineHeight: '1.6',
                color: '#37352f'
              }}
            />
          }
          placeholder={
            <div className="single-doc-editor-placeholder absolute top-12 left-16 text-gray-400 pointer-events-none text-lg">
              Start typing or press / for commands...
            </div>
          }
          ErrorBoundary={LexicalErrorBoundary}
        />
        
        {/* History (undo/redo) */}
        <HistoryPlugin />
        
        {/* Export changes back to DocState */}
        {!readOnly && (
          <DocExportOnChangePlugin 
            onDocChange={onDocChange}
            debounceMs={300}
          />
        )}
        
        {/* TODO: Add more plugins:
          - SelectionToAIPlugin
          - TemplateCheckPlugin
          - CommentsPlugin
          - KeyboardShortcutsPlugin
        */}
      </div>
    </LexicalComposer>
  );
}

