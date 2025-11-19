/**
 * Plugin to handle clicks on commented text
 * When user clicks on yellow-highlighted text, show the associated comments
 */

import { useEffect } from 'react';
import { useLexicalComposerContext } from '@lexical/react/LexicalComposerContext';
import { $getNodeByKey, $getSelection, $isRangeSelection } from 'lexical';
import { $isAiTextNode } from '../nodes/AiTextNode';

interface CommentClickPluginProps {
  onCommentClick?: (commentIds: string[]) => void;
}

export function CommentClickPlugin({ onCommentClick }: CommentClickPluginProps): null {
  const [editor] = useLexicalComposerContext();

  useEffect(() => {
    const handleClick = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      
      // Check if clicked element or its parent has comments
      let element = target;
      let commentIds: string[] = [];
      
      // Walk up the DOM tree to find a node with comment-ids
      while (element && element !== editor.getRootElement()) {
        if (element.hasAttribute('data-comment-ids')) {
          const ids = element.getAttribute('data-comment-ids');
          if (ids) {
            commentIds = ids.split(',');
            break;
          }
        }
        element = element.parentElement as HTMLElement;
      }
      
      // If we found comments and have a callback, trigger it
      if (commentIds.length > 0 && onCommentClick) {
        event.preventDefault();
        event.stopPropagation();
        onCommentClick(commentIds);
      }
    };

    const rootElement = editor.getRootElement();
    if (!rootElement) return;

    // Add click listener to the editor root
    rootElement.addEventListener('click', handleClick);

    return () => {
      rootElement.removeEventListener('click', handleClick);
    };
  }, [editor, onCommentClick]);

  return null;
}

