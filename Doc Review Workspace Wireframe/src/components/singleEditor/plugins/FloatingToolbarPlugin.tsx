// Floating toolbar that appears on text selection (Notion-style)
import { useLexicalComposerContext } from '@lexical/react/LexicalComposerContext';
import { $getSelection, $isRangeSelection } from 'lexical';
import { useCallback, useEffect, useRef, useState } from 'react';

interface FloatingToolbarProps {
  onFormat: (format: 'bold' | 'italic' | 'underline' | 'strikethrough') => void;
  onTextColor: (color: string) => void;
  onBackgroundColor: (color: string) => void;
  onTurnInto: (type: string) => void;
  onAddComment: () => void;
  onImproveText: () => void;
}

export function FloatingToolbarPlugin({
  onFormat,
  onTextColor,
  onBackgroundColor,
  onTurnInto,
  onAddComment,
  onImproveText,
}: FloatingToolbarProps) {
  const [editor] = useLexicalComposerContext();
  const toolbarRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState({ top: 0, left: 0 });
  const [showTurnIntoMenu, setShowTurnIntoMenu] = useState(false);
  const [showLinkMenu, setShowLinkMenu] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [manualPosition, setManualPosition] = useState<{ top: number; left: number } | null>(null);

  const updateToolbar = useCallback(() => {
    const selection = $getSelection();
    
    if (!$isRangeSelection(selection) || selection.isCollapsed()) {
      setIsVisible(false);
      setManualPosition(null); // Reset manual position when selection is cleared
      return;
    }

    const nativeSelection = window.getSelection();
    if (!nativeSelection || nativeSelection.rangeCount === 0) {
      setIsVisible(false);
      return;
    }

    const range = nativeSelection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    
    if (rect.width === 0 && rect.height === 0) {
      setIsVisible(false);
      return;
    }

    // Only update position if not manually positioned
    if (!manualPosition) {
      // Position toolbar above selection
      const toolbarHeight = 40; // Reduced from 48
      const gap = 8;
      
      let left = rect.left + window.scrollX + (rect.width / 2);
      let top = rect.top + window.scrollY - toolbarHeight - gap;
      
      // Constrain to viewport
      // Estimate toolbar width (will be centered, so transform will adjust)
      const estimatedToolbarWidth = 600;
      const halfWidth = estimatedToolbarWidth / 2;
      
      // Check horizontal boundaries
      if (left - halfWidth < 10) {
        left = halfWidth + 10;
      } else if (left + halfWidth > window.innerWidth - 10) {
        left = window.innerWidth - halfWidth - 10;
      }
      
      // Check if toolbar would go above viewport
      if (top < 10) {
        // Position below selection instead
        top = rect.bottom + window.scrollY + gap;
      }
      
      setPosition({ top, left });
    }
    
    setIsVisible(true);
  }, [manualPosition]);

  useEffect(() => {
    return editor.registerUpdateListener(() => {
      editor.getEditorState().read(() => {
        updateToolbar();
      });
    });
  }, [editor, updateToolbar]);

  useEffect(() => {
    const handleScroll = () => {
      if (isVisible && !isDragging && !manualPosition) {
        updateToolbar();
      }
    };

    window.addEventListener('scroll', handleScroll, true);
    return () => window.removeEventListener('scroll', handleScroll, true);
  }, [isVisible, isDragging, manualPosition, updateToolbar]);

  // Drag handlers
  const handleDragStart = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    
    const toolbar = toolbarRef.current;
    if (!toolbar) return;
    
    const rect = toolbar.getBoundingClientRect();
    setDragOffset({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  };

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      const toolbar = toolbarRef.current;
      if (!toolbar) return;

      const rect = toolbar.getBoundingClientRect();
      let newLeft = e.clientX - dragOffset.x;
      let newTop = e.clientY - dragOffset.y;
      
      // Constrain to viewport boundaries
      const maxLeft = window.innerWidth - rect.width - 10;
      const maxTop = window.innerHeight - rect.height - 10;
      const minLeft = 10;
      const minTop = 10;
      
      newLeft = Math.max(minLeft, Math.min(maxLeft, newLeft));
      newTop = Math.max(minTop, Math.min(maxTop, newTop));
      
      setManualPosition({
        left: newLeft + window.scrollX,
        top: newTop + window.scrollY,
      });
      setPosition({
        left: newLeft + window.scrollX,
        top: newTop + window.scrollY,
      });
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, dragOffset]);

  if (!isVisible) return null;

  const currentPosition = manualPosition || position;

  return (
    <div
      ref={toolbarRef}
      style={{
        position: 'absolute',
        top: `${currentPosition.top}px`,
        left: `${currentPosition.left}px`,
        transform: manualPosition ? 'none' : 'translateX(-50%)',
        zIndex: 1000,
        backgroundColor: '#1f2937',
        borderRadius: '6px',
        padding: '3px 6px',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
        display: 'flex',
        alignItems: 'center',
        gap: '2px',
        cursor: isDragging ? 'grabbing' : 'default',
      }}
    >
      {/* Drag Handle */}
      <div
        onMouseDown={handleDragStart}
        style={{
          display: 'flex',
          alignItems: 'center',
          padding: '4px 3px',
          cursor: 'grab',
          color: '#9ca3af',
        }}
        title="Drag to move toolbar"
      >
        <svg width="10" height="10" viewBox="0 0 12 12" fill="currentColor">
          <circle cx="3" cy="3" r="1" />
          <circle cx="3" cy="6" r="1" />
          <circle cx="3" cy="9" r="1" />
          <circle cx="9" cy="3" r="1" />
          <circle cx="9" cy="6" r="1" />
          <circle cx="9" cy="9" r="1" />
        </svg>
      </div>
      
      {/* Divider */}
      <div style={{ width: '1px', height: '20px', backgroundColor: '#4b5563' }} />
      {/* Explain (dummy) */}
      <button
        onClick={() => alert('Explain feature - Coming soon!')}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '3px',
          padding: '4px 7px',
          backgroundColor: 'transparent',
          color: '#10b981',
          border: 'none',
          borderRadius: '3px',
          cursor: 'pointer',
          fontSize: '11px',
          fontWeight: 500,
        }}
        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#374151'}
        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
        title="Explain (Coming soon)"
      >
        <svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor">
          <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" fill="none" />
          <text x="8" y="11" fontSize="10" textAnchor="middle" fill="currentColor" fontWeight="bold">?</text>
        </svg>
        Explain
      </button>

      {/* Divider */}
      <div style={{ width: '1px', height: '20px', backgroundColor: '#4b5563' }} />

      {/* Ask AI */}
      <button
        onClick={onImproveText}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '3px',
          padding: '4px 7px',
          backgroundColor: 'transparent',
          color: '#e5e7eb',
          border: 'none',
          borderRadius: '3px',
          cursor: 'pointer',
          fontSize: '11px',
          fontWeight: 500,
        }}
        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#374151'}
        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
        title="Ask AI to improve text"
      >
        <svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor">
          <path d="M8 1a7 7 0 100 14A7 7 0 008 1zm0 1.5a5.5 5.5 0 110 11 5.5 5.5 0 010-11zM7 5v1h2V5H7zm0 2v5h2V7H7z"/>
        </svg>
        Ask AI
      </button>

      {/* Divider */}
      <div style={{ width: '1px', height: '20px', backgroundColor: '#4b5563' }} />

      {/* Comment */}
      <button
        onClick={onAddComment}
        style={{
          display: 'flex',
          alignItems: 'center',
          padding: '4px 6px',
          backgroundColor: 'transparent',
          color: '#e5e7eb',
          border: 'none',
          borderRadius: '3px',
          cursor: 'pointer',
          fontSize: '14px',
        }}
        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#374151'}
        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
        title="Add comment"
      >
        💬
      </button>

      {/* Emoji (dummy) */}
      <button
        onClick={() => alert('Emoji picker - Coming soon!')}
        style={{
          display: 'flex',
          alignItems: 'center',
          padding: '4px 6px',
          backgroundColor: 'transparent',
          color: '#e5e7eb',
          border: 'none',
          borderRadius: '3px',
          cursor: 'pointer',
          fontSize: '14px',
        }}
        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#374151'}
        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
        title="Insert emoji (Coming soon)"
      >
        😀
      </button>

      {/* Edit (dummy) */}
      <button
        onClick={() => alert('Edit feature - Coming soon!')}
        style={{
          display: 'flex',
          alignItems: 'center',
          padding: '4px 6px',
          backgroundColor: 'transparent',
          color: '#e5e7eb',
          border: 'none',
          borderRadius: '3px',
          cursor: 'pointer',
          fontSize: '14px',
        }}
        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#374151'}
        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
        title="Edit (Coming soon)"
      >
        ✏️
      </button>

      {/* Divider */}
      <div style={{ width: '1px', height: '20px', backgroundColor: '#4b5563' }} />

      {/* Turn Into Dropdown */}
      <div style={{ position: 'relative' }}>
        <button
          onClick={() => setShowTurnIntoMenu(!showTurnIntoMenu)}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '3px',
            padding: '4px 7px',
            backgroundColor: showTurnIntoMenu ? '#374151' : 'transparent',
            color: '#e5e7eb',
            border: 'none',
            borderRadius: '3px',
            cursor: 'pointer',
            fontSize: '11px',
            fontWeight: 500,
          }}
          onMouseEnter={(e) => !showTurnIntoMenu && (e.currentTarget.style.backgroundColor = '#374151')}
          onMouseLeave={(e) => !showTurnIntoMenu && (e.currentTarget.style.backgroundColor = 'transparent')}
          title="Turn into"
        >
          Bulleted list
          <svg width="10" height="10" viewBox="0 0 12 12" fill="currentColor">
            <path d="M6 8L3 5h6z"/>
          </svg>
        </button>

        {showTurnIntoMenu && (
          <div
            style={{
              position: 'absolute',
              top: '100%',
              left: 0,
              marginTop: '4px',
              backgroundColor: '#1f2937',
              borderRadius: '5px',
              padding: '3px',
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
              minWidth: '150px',
              zIndex: 1001,
            }}
          >
            {[
              { label: 'Text', value: 'paragraph', icon: 'T' },
              { label: 'Heading 1', value: 'heading-1', icon: 'H1' },
              { label: 'Heading 2', value: 'heading-2', icon: 'H2' },
              { label: 'Heading 3', value: 'heading-3', icon: 'H3' },
              { label: 'Bulleted List', value: 'bulleted-list', icon: '•' },
              { label: 'Numbered List', value: 'numbered-list', icon: '1.' },
              { label: 'Code', value: 'code', icon: '</>' },
              { label: 'Quote', value: 'quote', icon: '"' },
            ].map((item) => (
              <button
                key={item.value}
                onClick={() => {
                  onTurnInto(item.value);
                  setShowTurnIntoMenu(false);
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  width: '100%',
                  padding: '4px 6px',
                  backgroundColor: 'transparent',
                  color: '#e5e7eb',
                  border: 'none',
                  borderRadius: '3px',
                  cursor: 'pointer',
                  fontSize: '11px',
                  textAlign: 'left',
                }}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#374151'}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
              >
                <span style={{ width: '16px', fontWeight: 600 }}>{item.icon}</span>
                {item.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Divider */}
      <div style={{ width: '1px', height: '20px', backgroundColor: '#4b5563' }} />

      {/* Bold */}
      <button
        onClick={() => onFormat('bold')}
        style={{
          padding: '4px 6px',
          backgroundColor: 'transparent',
          color: '#e5e7eb',
          border: 'none',
          borderRadius: '3px',
          cursor: 'pointer',
          fontSize: '12px',
          fontWeight: 'bold',
        }}
        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#374151'}
        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
        title="Bold"
      >
        B
      </button>

      {/* Italic */}
      <button
        onClick={() => onFormat('italic')}
        style={{
          padding: '4px 6px',
          backgroundColor: 'transparent',
          color: '#e5e7eb',
          border: 'none',
          borderRadius: '3px',
          cursor: 'pointer',
          fontSize: '12px',
          fontStyle: 'italic',
          fontWeight: 'bold',
        }}
        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#374151'}
        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
        title="Italic"
      >
        I
      </button>

      {/* Underline */}
      <button
        onClick={() => onFormat('underline')}
        style={{
          padding: '4px 6px',
          backgroundColor: 'transparent',
          color: '#e5e7eb',
          border: 'none',
          borderRadius: '3px',
          cursor: 'pointer',
          fontSize: '12px',
          textDecoration: 'underline',
          fontWeight: 'bold',
        }}
        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#374151'}
        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
        title="Underline"
      >
        U
      </button>

      {/* Strikethrough (dummy) */}
      <button
        onClick={() => alert('Strikethrough - Coming soon!')}
        style={{
          padding: '4px 6px',
          backgroundColor: 'transparent',
          color: '#e5e7eb',
          border: 'none',
          borderRadius: '3px',
          cursor: 'pointer',
          fontSize: '12px',
          textDecoration: 'line-through',
          fontWeight: 'bold',
        }}
        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#374151'}
        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
        title="Strikethrough (Coming soon)"
      >
        S
      </button>

      {/* Code (dummy) */}
      <button
        onClick={() => alert('Inline code - Coming soon!')}
        style={{
          padding: '4px 6px',
          backgroundColor: 'transparent',
          color: '#e5e7eb',
          border: 'none',
          borderRadius: '3px',
          cursor: 'pointer',
          fontSize: '11px',
          fontFamily: 'monospace',
        }}
        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#374151'}
        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
        title="Inline code (Coming soon)"
      >
        &lt;/&gt;
      </button>

      {/* Math (dummy) */}
      <button
        onClick={() => alert('Math equation - Coming soon!')}
        style={{
          padding: '4px 6px',
          backgroundColor: 'transparent',
          color: '#e5e7eb',
          border: 'none',
          borderRadius: '3px',
          cursor: 'pointer',
          fontSize: '12px',
        }}
        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#374151'}
        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
        title="Math equation (Coming soon)"
      >
        √x
      </button>

      {/* Link (dummy) */}
      <button
        onClick={() => setShowLinkMenu(!showLinkMenu)}
        style={{
          padding: '4px 6px',
          backgroundColor: showLinkMenu ? '#374151' : 'transparent',
          color: '#e5e7eb',
          border: 'none',
          borderRadius: '3px',
          cursor: 'pointer',
          fontSize: '13px',
        }}
        onMouseEnter={(e) => !showLinkMenu && (e.currentTarget.style.backgroundColor = '#374151')}
        onMouseLeave={(e) => !showLinkMenu && (e.currentTarget.style.backgroundColor = 'transparent')}
        title="Add link (Coming soon)"
      >
        🔗
      </button>

      {/* Divider */}
      <div style={{ width: '1px', height: '20px', backgroundColor: '#4b5563' }} />

      {/* Text Color (dummy) */}
      <button
        onClick={() => alert('Text color picker - Use right sidebar')}
        style={{
          padding: '4px 6px',
          backgroundColor: 'transparent',
          color: '#e5e7eb',
          border: 'none',
          borderRadius: '3px',
          cursor: 'pointer',
          fontSize: '14px',
        }}
        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#374151'}
        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
        title="Text color (Use right sidebar)"
      >
        A
      </button>

      {/* More options */}
      <button
        onClick={() => alert('More options - Coming soon!')}
        style={{
          padding: '4px 6px',
          backgroundColor: 'transparent',
          color: '#e5e7eb',
          border: 'none',
          borderRadius: '3px',
          cursor: 'pointer',
          fontSize: '14px',
        }}
        onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#374151'}
        onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
        title="More options"
      >
        ⋯
      </button>
    </div>
  );
}

