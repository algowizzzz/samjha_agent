// Plugin to handle Enter key behavior for custom nodes
import { useLexicalComposerContext } from '@lexical/react/LexicalComposerContext';
import { useEffect } from 'react';
import {
  COMMAND_PRIORITY_LOW,
  KEY_ENTER_COMMAND,
  $getSelection,
  $isRangeSelection,
  $createParagraphNode,
} from 'lexical';
import { $findMatchingParent } from '@lexical/utils';
import { $isDocHeadingNode } from '../nodes/DocHeadingNode';
import { $isDocParagraphNode, $createDocParagraphNode } from '../nodes/DocParagraphNode';

export function EnterKeyPlugin() {
  const [editor] = useLexicalComposerContext();

  useEffect(() => {
    return editor.registerCommand(
      KEY_ENTER_COMMAND,
      (event: KeyboardEvent | null) => {
        const selection = $getSelection();
        
        if (!$isRangeSelection(selection)) {
          return false;
        }

        const anchor = selection.anchor.getNode();
        
        // Find parent heading
        const heading = $findMatchingParent(anchor, $isDocHeadingNode);
        
        if (heading) {
          // Pressing Enter in heading should create a paragraph
          event?.preventDefault();
          
          const newParagraph = $createDocParagraphNode(
            undefined,
            heading.getSectionKey()
          );
          
          heading.insertAfter(newParagraph);
          newParagraph.select();
          
          return true;
        }

        // For paragraphs, let Lexical handle it but ensure blockId
        const paragraph = $findMatchingParent(anchor, $isDocParagraphNode);
        
        if (paragraph) {
          // Let default behavior happen, then fix blockId
          setTimeout(() => {
            editor.update(() => {
              const sel = $getSelection();
              if (!$isRangeSelection(sel)) return;
              
              const node = sel.anchor.getNode();
              const para = $findMatchingParent(node, (n) => {
                return n.getType() === 'paragraph';
              });
              
              // If we created a native paragraph, replace it with DocParagraphNode
              if (para && para.getType() === 'paragraph' && !$isDocParagraphNode(para)) {
                const docPara = $createDocParagraphNode(
                  undefined,
                  paragraph.getSectionKey()
                );
                const children = para.getChildren();
                children.forEach((child) => docPara.append(child));
                para.replace(docPara);
                docPara.select();
              }
            });
          }, 0);
          
          return false;
        }

        return false;
      },
      COMMAND_PRIORITY_LOW
    );
  }, [editor]);

  return null;
}

