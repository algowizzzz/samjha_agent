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
  const [showTextColorMenu, setShowTextColorMenu] = useState(false);
  const [showBgColorMenu, setShowBgColorMenu] = useState(false);
  const [showHighlights, setShowHighlights] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [manualPosition, setManualPosition] = useState<{ top: number; left: number } | null>(null);
  const [recentTextColors, setRecentTextColors] = useState<string[]>([]);
  const [recentBgColors, setRecentBgColors] = useState<string[]>([]);

  // Text color palette - comprehensive
  const textColors = [
    { name: 'Black', value: '#000000' },
    { name: 'Dark Gray', value: '#6b7280' },
    { name: 'Brown', value: '#92400e' },
    { name: 'Orange', value: '#ea580c' },
    { name: 'Gold', value: '#ca8a04' },
    { name: 'Green', value: '#16a34a' },
    { name: 'Blue', value: '#2563eb' },
    { name: 'Purple', value: '#9333ea' },
    { name: 'Pink', value: '#db2777' },
    { name: 'Red', value: '#dc2626' },
  ];

  // Background color palette - comprehensive
  const bgColors = [
    { name: 'Transparent', value: 'transparent' },
    { name: 'Light Gray', value: '#f3f4f6' },
    { name: 'Brown', value: '#d4a574' },
    { name: 'Orange', value: '#fed7aa' },
    { name: 'Yellow', value: '#fef3c7' },
    { name: 'Green', value: '#bbf7d0' },
    { name: 'Blue', value: '#bfdbfe' },
    { name: 'Purple', value: '#e9d5ff' },
    { name: 'Pink', value: '#fbcfe8' },
    { name: 'Red', value: '#fecaca' },
  ];

  // Add color to recent colors
  const addToRecentTextColors = (color: string) => {
    setRecentTextColors(prev => {
      const filtered = prev.filter(c => c !== color);
      return [color, ...filtered].slice(0, 4);
    });
  };

  const addToRecentBgColors = (color: string) => {
    setRecentBgColors(prev => {
      const filtered = prev.filter(c => c !== color);
      return [color, ...filtered].slice(0, 4);
    });
  };

  const updateToolbar = useCallback(() => {
    const selection = $getSelection();
    
    if (!$isRangeSelection(selection) || selection.isCollapsed()) {
      setIsVisible(false);
      setManualPosition(null);
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
      // Position toolbar above selection, aligned to left
      const toolbarHeight = 40;
      const gap = 8;
      
      // Get editor container bounds
      const editorContainer = document.querySelector('[data-lexical-editor="true"]')?.parentElement;
      const containerRect = editorContainer?.getBoundingClientRect();
      
      // Align toolbar to left edge of selection
      let left = rect.left + window.scrollX;
      let top = rect.top + window.scrollY - toolbarHeight - gap;
      
      // Constrain to editor container (middle panel)
      const estimatedToolbarWidth = 600;
      
      if (containerRect) {
        // Horizontal constraints relative to container
        const containerLeft = containerRect.left + window.scrollX;
        const containerRight = containerLeft + containerRect.width;
        
        // Ensure toolbar doesn't go outside left edge
        if (left < containerLeft + 10) {
          left = containerLeft + 10;
        }
        // Ensure toolbar doesn't go outside right edge
        if (left + estimatedToolbarWidth > containerRight - 10) {
          left = containerRight - estimatedToolbarWidth - 10;
        }
        
        // Vertical constraints relative to container
        const containerTop = containerRect.top + window.scrollY;
        if (top < containerTop + 10) {
          // Position below selection instead
          top = rect.bottom + window.scrollY + gap;
        }
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
        transform: 'none',
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

      {/* Toggle Highlights */}
      <button
        onClick={() => {
          setShowHighlights(!showHighlights);
          // Toggle CSS class on editor - find ContentEditable
          const editorDiv = document.querySelector('[contenteditable="true"]');
          if (editorDiv) {
            if (showHighlights) {
              editorDiv.classList.add('hide-highlights');
            } else {
              editorDiv.classList.remove('hide-highlights');
            }
          }
        }}
        style={{
          display: 'flex',
          alignItems: 'center',
          padding: '4px 6px',
          backgroundColor: showHighlights ? 'transparent' : '#374151',
          color: '#e5e7eb',
          border: 'none',
          borderRadius: '3px',
          cursor: 'pointer',
          fontSize: '12px',
          fontWeight: 500,
        }}
        onMouseEnter={(e) => !showHighlights && (e.currentTarget.style.backgroundColor = '#4b5563')}
        onMouseLeave={(e) => !showHighlights && (e.currentTarget.style.backgroundColor = '#374151')}
        title={showHighlights ? "Hide comment & AI highlights" : "Show comment & AI highlights"}
      >
        {showHighlights ? (
          <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
            <path d="M10.5 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0z"/>
            <path d="M0 8s3-5.5 8-5.5S16 8 16 8s-3 5.5-8 5.5S0 8 0 8zm8 3.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z"/>
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
            <path d="M13.359 11.238C15.06 9.72 16 8 16 8s-3-5.5-8-5.5a7.028 7.028 0 0 0-2.79.588l.77.771A5.944 5.944 0 0 1 8 3.5c2.12 0 3.879 1.168 5.168 2.457A13.134 13.134 0 0 1 14.828 8c-.058.087-.122.183-.195.288-.335.48-.83 1.12-1.465 1.755-.165.165-.337.328-.517.486l.708.709z"/>
            <path d="M11.297 9.176a3.5 3.5 0 0 0-4.474-4.474l.823.823a2.5 2.5 0 0 1 2.829 2.829l.822.822zm-2.943 1.299.822.822a3.5 3.5 0 0 1-4.474-4.474l.823.823a2.5 2.5 0 0 0 2.829 2.829z"/>
            <path d="M3.35 5.47c-.18.16-.353.322-.518.487A13.134 13.134 0 0 0 1.172 8l.195.288c.335.48.83 1.12 1.465 1.755C4.121 11.332 5.881 12.5 8 12.5c.716 0 1.39-.133 2.02-.36l.77.772A7.029 7.029 0 0 1 8 13.5C3 13.5 0 8 0 8s.939-1.721 2.641-3.238l.708.709zm10.296 8.884-12-12 .708-.708 12 12-.708.708z"/>
          </svg>
        )}
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

      {/* Text Color */}
      <div style={{ position: 'relative', display: 'inline-block' }}>
        <button
          onClick={() => {
            setShowTextColorMenu(!showTextColorMenu);
            setShowBgColorMenu(false);
            setShowTurnIntoMenu(false);
          }}
          style={{
            padding: '4px 6px',
            backgroundColor: showTextColorMenu ? '#374151' : 'transparent',
            color: '#e5e7eb',
            border: 'none',
            borderRadius: '3px',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: 'bold',
          }}
          onMouseEnter={(e) => !showTextColorMenu && (e.currentTarget.style.backgroundColor = '#374151')}
          onMouseLeave={(e) => !showTextColorMenu && (e.currentTarget.style.backgroundColor = 'transparent')}
          title="Text color"
        >
          A
        </button>

        {showTextColorMenu && (
          <div
            style={{
              position: 'absolute',
              top: '100%',
              left: '0',
              marginTop: '6px',
              backgroundColor: '#2d2d2d',
              borderRadius: '8px',
              padding: '12px',
              boxShadow: '0 4px 16px rgba(0, 0, 0, 0.4)',
              zIndex: 1001,
              minWidth: '200px',
            }}
          >
            {/* Recently used */}
            {recentTextColors.length > 0 && (
              <>
                <div style={{ color: '#9ca3af', fontSize: '11px', marginBottom: '8px', fontWeight: 500 }}>
                  Recently used
                </div>
                <div style={{ display: 'flex', gap: '6px', marginBottom: '12px' }}>
                  {recentTextColors.map((color, idx) => (
                    <button
                      key={`recent-${idx}`}
                      onClick={() => {
                        onTextColor(color);
                        setShowTextColorMenu(false);
                      }}
                      style={{
                        width: '36px',
                        height: '36px',
                        backgroundColor: '#1f2937',
                        border: '2px solid #374151',
                        borderRadius: '8px',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '18px',
                        fontWeight: 'bold',
                        color: color,
                      }}
                    >
                      A
                    </button>
                  ))}
                </div>
              </>
            )}

            {/* Text color palette */}
            <div style={{ color: '#9ca3af', fontSize: '11px', marginBottom: '8px', fontWeight: 500 }}>
              Text color
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '6px' }}>
              {textColors.map((color) => (
                <button
                  key={color.value}
                  onClick={() => {
                    onTextColor(color.value);
                    addToRecentTextColors(color.value);
                    setShowTextColorMenu(false);
                  }}
                  style={{
                    width: '36px',
                    height: '36px',
                    backgroundColor: '#1f2937',
                    border: '2px solid #374151',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '18px',
                    fontWeight: 'bold',
                    color: color.value,
                  }}
                  title={color.name}
                >
                  A
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Background Color */}
      <div style={{ position: 'relative', display: 'inline-block' }}>
        <button
          onClick={() => {
            setShowBgColorMenu(!showBgColorMenu);
            setShowTextColorMenu(false);
            setShowTurnIntoMenu(false);
          }}
          style={{
            padding: '4px 6px',
            backgroundColor: showBgColorMenu ? '#374151' : 'transparent',
            color: '#e5e7eb',
            border: 'none',
            borderRadius: '3px',
            cursor: 'pointer',
            fontSize: '13px',
          }}
          onMouseEnter={(e) => !showBgColorMenu && (e.currentTarget.style.backgroundColor = '#374151')}
          onMouseLeave={(e) => !showBgColorMenu && (e.currentTarget.style.backgroundColor = 'transparent')}
          title="Background color"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
            <rect x="2" y="10" width="12" height="4" />
          </svg>
        </button>

        {showBgColorMenu && (
          <div
            style={{
              position: 'absolute',
              top: '100%',
              left: '0',
              marginTop: '6px',
              backgroundColor: '#2d2d2d',
              borderRadius: '8px',
              padding: '12px',
              boxShadow: '0 4px 16px rgba(0, 0, 0, 0.4)',
              zIndex: 1001,
              minWidth: '200px',
            }}
          >
            {/* Recently used */}
            {recentBgColors.length > 0 && (
              <>
                <div style={{ color: '#9ca3af', fontSize: '11px', marginBottom: '8px', fontWeight: 500 }}>
                  Recently used
                </div>
                <div style={{ display: 'flex', gap: '6px', marginBottom: '12px' }}>
                  {recentBgColors.map((color, idx) => (
                    <button
                      key={`recent-bg-${idx}`}
                      onClick={() => {
                        onBackgroundColor(color);
                        setShowBgColorMenu(false);
                      }}
                      style={{
                        width: '36px',
                        height: '36px',
                        backgroundColor: color,
                        border: '2px solid #374151',
                        borderRadius: '8px',
                        cursor: 'pointer',
                      }}
                    />
                  ))}
                </div>
              </>
            )}

            {/* Background color palette */}
            <div style={{ color: '#9ca3af', fontSize: '11px', marginBottom: '8px', fontWeight: 500 }}>
              Background color
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '6px' }}>
              {bgColors.map((color) => (
                <button
                  key={color.value}
                  onClick={() => {
                    onBackgroundColor(color.value);
                    addToRecentBgColors(color.value);
                    setShowBgColorMenu(false);
                  }}
                  style={{
                    width: '36px',
                    height: '36px',
                    backgroundColor: color.value === 'transparent' ? '#1f2937' : color.value,
                    border: '2px solid #374151',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    position: 'relative',
                  }}
                  title={color.name}
                >
                  {color.value === 'transparent' && (
                    <div style={{
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      transform: 'translate(-50%, -50%) rotate(-45deg)',
                      width: '20px',
                      height: '2px',
                      backgroundColor: '#ef4444',
                    }} />
                  )}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Divider */}
      <div style={{ width: '1px', height: '20px', backgroundColor: '#4b5563' }} />

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

