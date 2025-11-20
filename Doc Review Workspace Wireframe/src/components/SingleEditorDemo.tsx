import { useState } from 'react';
import { SingleDocumentEditor } from './singleEditor/SingleDocumentEditor';
import { Button } from './ui/button';
import type { DocState } from '@/model/docTypes';
import type { SelectionData } from './singleEditor/plugins/SelectionBridgePlugin';
import type { Comment, CreateCommentRequest } from '@/model/commentTypes';
import {
  setAiStatusOnSelection,
  getCurrentSelectionText,
  insertAiSuggestion,
} from './singleEditor/utils/aiSuggestionHelpers';
import {
  applyCommentToSelection,
  removeCommentHighlight,
  highlightCommentInEditor,
} from './singleEditor/utils/commentHighlightHelpers';
import { enableFeature, disableFeature, getAllFeatureFlags } from '@/lib/featureFlags';
import { FORMAT_TEXT_COMMAND, $getSelection, $isRangeSelection, $createParagraphNode } from 'lexical';
import { $findMatchingParent } from '@lexical/utils';
import { $createDocParagraphNode } from './singleEditor/nodes/DocParagraphNode';
import { $createDocHeadingNode } from './singleEditor/nodes/DocHeadingNode';
import { $createDocListNode, type ListStyle } from './singleEditor/nodes/DocListNode';
import { $createDocListItemNode } from './singleEditor/nodes/DocListItemNode';
import { $createDocCodeNode } from './singleEditor/nodes/DocCodeNode';
import { $createDocQuoteNode } from './singleEditor/nodes/DocQuoteNode';
import { $createAiTextNode } from './singleEditor/nodes/AiTextNode';

/**
 * Demo page for the new Single Document Editor (Option 3).
 * 
 * This page demonstrates all the key features:
 * - Single Lexical editor for entire document
 * - Rich text formatting (bold, italic, etc.)
 * - AI suggestion workflow (suggested → applied → rejected)
 * - Headings with section keys
 * - Real-time serialization to DocState
 */
interface PendingSuggestion {
  original: string;
  improved: string;
  reason: string;
}

export function SingleEditorDemo() {
  const [docState, setDocState] = useState<DocState>(SAMPLE_DOC_STATE);
  const [editorInstance, setEditorInstance] = useState<any>(null);
  const [selectedText, setSelectedText] = useState('');
  const [showJson, setShowJson] = useState(false);
  const [showControls, setShowControls] = useState(true);
  const [selectionData, setSelectionData] = useState<SelectionData>({
    selectionScope: 'none',
    blockIds: [],
    selectedText: '',
    isEmpty: true,
  });
  const [pendingSuggestion, setPendingSuggestion] = useState<PendingSuggestion | null>(null);
  
  // Comment system state
  const [comments, setComments] = useState<Comment[]>([]);
  const [showCommentModal, setShowCommentModal] = useState(false);
  const [showCommentPanel, setShowCommentPanel] = useState(false);
  const [editingComment, setEditingComment] = useState<Comment | null>(null);
  const [replyingTo, setReplyingTo] = useState<Comment | null>(null);
  const [collapsedComments, setCollapsedComments] = useState<Set<string>>(new Set());
  
  // Panel state for 3-panel layout with resizable widths
  const [scratchpadCollapsed, setScratchpadCollapsed] = useState(false);
  const [scratchpadWidth, setScratchpadWidth] = useState(20); // percentage
  const [scratchpadDoc, setScratchpadDoc] = useState<DocState>(SCRATCHPAD_DOC_STATE);
  const [scratchpadEditorInstance, setScratchpadEditorInstance] = useState<any>(null);
  
  
  const [mainEditorCollapsed, setMainEditorCollapsed] = useState(false);
  // Main editor uses flex: 1, no width state needed
  
  const [controlsCollapsed, setControlsCollapsed] = useState(false);
  const [controlsWidth, setControlsWidth] = useState(20); // percentage

  const handleDocChange = (newDocState: DocState) => {
    setDocState(newDocState);
    console.log('[Demo] Document changed:', newDocState);
  };


  const handleGetSelection = () => {
    if (!editorInstance) return;
    const text = getCurrentSelectionText(editorInstance);
    setSelectedText(text);
    console.log('[Demo] Selected text:', text);
  };

  const handleInsertAiSuggestion = () => {
    if (!editorInstance) return;
    const suggestion = 'This is an AI-generated suggestion with blue underline';
    insertAiSuggestion(editorInstance, suggestion, 'suggested');
  };

  const handleApplySelection = () => {
    if (!editorInstance) return;
    setAiStatusOnSelection(editorInstance, 'applied');
  };

  const handleRejectSelection = () => {
    if (!editorInstance) return;
    setAiStatusOnSelection(editorInstance, 'rejected');
  };

  const handleClearSelection = () => {
    if (!editorInstance) return;
    setAiStatusOnSelection(editorInstance, null);
  };

  const handleImproveText = async () => {
    if (!selectionData.selectedText || selectionData.isEmpty) {
      alert('Please select some text first');
      return;
    }

    try {
      const response = await fetch('http://localhost:8000/api/text-improvement/improve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: selectionData.selectedText,
          instruction: 'Improve clarity and precision',
        }),
      });

      const result = await response.json();

      if (result.success) {
        // Store the suggestion for user review
        setPendingSuggestion({
          original: result.original,
          improved: result.improved,
          reason: result.reason,
        });
      } else {
        alert(`Failed: ${result.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Text improvement error:', error);
      alert(`Error: ${error}`);
    }
  };

  const handleAcceptPendingSuggestion = () => {
    if (!editorInstance || !pendingSuggestion) return;
    insertAiSuggestion(editorInstance, pendingSuggestion.improved, 'suggested');
    setPendingSuggestion(null);
  };

  const handleRejectPendingSuggestion = () => {
    setPendingSuggestion(null);
  };

  const handleFormat = (format: 'bold' | 'italic' | 'underline') => {
    if (!editorInstance) return;
    
    editorInstance.update(() => {
      const selection = $getSelection();
      if ($isRangeSelection(selection)) {
        // Toggle the format on the selected text
        if (format === 'bold') {
          selection.formatText('bold');
        } else if (format === 'italic') {
          selection.formatText('italic');
        } else if (format === 'underline') {
          selection.formatText('underline');
        }
      }
    });
    
    editorInstance.focus();
  };

  const handleTextColor = (color: string) => {
    if (!editorInstance) return;
    
    editorInstance.update(() => {
      const selection = $getSelection();
      if ($isRangeSelection(selection)) {
        selection.getNodes().forEach((node) => {
          if (node.getType() === 'ai-text' || node.getType() === 'text') {
            const currentStyle = node.getStyle() || '';
            // Remove existing color style
            const newStyle = currentStyle.replace(/color:\s*[^;]+;?/g, '').trim();
            // Add new color
            node.setStyle(newStyle ? `${newStyle}; color: ${color};` : `color: ${color};`);
          }
        });
      }
    });
    
    editorInstance.focus();
  };

  const handleBackgroundColor = (color: string) => {
    if (!editorInstance) return;
    
    editorInstance.update(() => {
      const selection = $getSelection();
      if ($isRangeSelection(selection)) {
        selection.getNodes().forEach((node) => {
          if (node.getType() === 'ai-text' || node.getType() === 'text') {
            const currentStyle = node.getStyle() || '';
            // Remove existing background-color style
            const newStyle = currentStyle.replace(/background-color:\s*[^;]+;?/g, '').trim();
            // Add new background color
            node.setStyle(newStyle ? `${newStyle}; background-color: ${color};` : `background-color: ${color};`);
          }
        });
      }
    });
    
    editorInstance.focus();
  };

  const handleTurnInto = (
    type: 'paragraph' | 'heading' | 'list' | 'code' | 'quote',
    options?: { level?: 1 | 2 | 3; listStyle?: ListStyle }
  ) => {
    if (!editorInstance) return;
    
    editorInstance.update(() => {
      const selection = $getSelection();
      if (!$isRangeSelection(selection)) return;
      
      const anchor = selection.anchor.getNode();
      
      // Find the parent block node
      const blockNode = $findMatchingParent(
        anchor,
        (node) => {
          const nodeType = node.getType();
          return nodeType === 'doc-paragraph' || 
                 nodeType === 'doc-heading' || 
                 nodeType === 'doc-list' ||
                 nodeType === 'doc-code' ||
                 nodeType === 'doc-quote' ||
                 nodeType === 'doc-divider' ||
                 nodeType === 'doc-image' ||
                 nodeType === 'doc-empty';
        }
      );
      
      if (!blockNode) return;
      
      // Get the block ID to preserve it
      const blockId = (blockNode as any).getBlockId?.() || `b${Date.now()}`;
      
      // Get current text content (empty for divider/image/empty blocks)
      const nodeType = blockNode.getType();
      const isNonEditableBlock = nodeType === 'doc-divider' || nodeType === 'doc-image' || nodeType === 'doc-empty';
      const textContent = isNonEditableBlock ? '' : blockNode.getTextContent();
      
      // Create new node based on type
      let newNode;
      if (type === 'paragraph') {
        newNode = $createDocParagraphNode(blockId);
      } else if (type === 'heading') {
        newNode = $createDocHeadingNode(options?.level || 1, blockId);
      } else if (type === 'list') {
        // Create list with current text as a single item
        const items = textContent ? [{ content: textContent }] : [{ content: '' }];
        const listNode = $createDocListNode(blockId, options?.listStyle || 'bullet', items);
        
        // Create a list item with text content
        const listItemNode = $createDocListItemNode();
        const textNode = $createAiTextNode(textContent || '');
        listItemNode.append(textNode);
        listNode.append(listItemNode);
        
        // Replace the node
        blockNode.replace(listNode);
        
        // Set selection to the text node inside the list item
        textNode.select();
        return;
      } else if (type === 'code') {
        const codeNode = $createDocCodeNode(blockId, textContent);
        const textNode = $createAiTextNode(textContent || '');
        codeNode.append(textNode);
        
        // Replace the node
        blockNode.replace(codeNode);
        
        // Set selection to the text node inside code
        textNode.select();
        return;
      } else if (type === 'quote') {
        newNode = $createDocQuoteNode(blockId);
      }
      
      if (!newNode) return;
      
      // Transfer children (for paragraph, heading, quote)
      if (isNonEditableBlock) {
        // Non-editable blocks have no text children, create a blank text node
        const textNode = $createAiTextNode('');
        newNode.append(textNode);
      } else {
        // Transfer existing children
        const children = blockNode.getChildren();
        children.forEach(child => {
          newNode.append(child);
        });
      }
      
      // Replace the node
      blockNode.replace(newNode);
      
      // Select the first text node in the new node
      const firstChild = newNode.getFirstChild();
      if (firstChild) {
        firstChild.select();
      }
    });
    
    editorInstance.focus();
  };

  // Comment functions
  const handleAddComment = () => {
    if (selectionData.isEmpty || !selectionData.selectedText) {
      alert('Please select some text first');
      return;
    }
    setShowCommentModal(true);
  };

  const handleCreateComment = async (commentText: string) => {
    if (!commentText.trim() || !editorInstance) return;

    // Generate comment ID
    const commentId = `comment-${Date.now()}`;

    // Apply yellow highlighting to selected text nodes
    const highlightData = applyCommentToSelection(editorInstance, commentId);
    
    if (!highlightData) {
      alert('Failed to apply comment highlight');
      return;
    }

    const newComment: Comment = {
      id: commentId,
      documentId: docState.id,
      blockId: highlightData.blockId,
      selectedText: highlightData.selectedText,
      startOffset: highlightData.startOffset,
      endOffset: highlightData.endOffset,
      commentText: commentText,
      username: 'Current User', // TODO: Get from auth
      timestamp: new Date().toISOString(),
      replies: [],
    };

    // Add comment to state
    setComments([...comments, newComment]);
    setShowCommentModal(false);
    setShowCommentPanel(true);

    // TODO: Send to backend
    console.log('Created comment:', newComment);
  };

  const handleEditComment = (comment: Comment) => {
    setEditingComment(comment);
    setShowCommentModal(true);
  };

  const handleUpdateComment = async (commentId: string, newText: string) => {
    const updatedComments = comments.map(c => 
      c.id === commentId ? { ...c, commentText: newText } : c
    );
    setComments(updatedComments);
    setEditingComment(null);
    setShowCommentModal(false);

    // TODO: Send to backend
    console.log('Updated comment:', commentId);
  };

  const handleDeleteComment = async (commentId: string) => {
    if (!confirm('Delete this comment?')) return;

    // Remove yellow highlighting from text nodes
    if (editorInstance) {
      removeCommentHighlight(editorInstance, commentId);
    }

    const updatedComments = comments.filter(c => c.id !== commentId);
    setComments(updatedComments);

    // TODO: Send to backend
    console.log('Deleted comment:', commentId);
  };

  const handleReplyToComment = (comment: Comment) => {
    setReplyingTo(comment);
    setShowCommentModal(true);
  };

  const handleCreateReply = async (parentComment: Comment, replyText: string) => {
    if (!replyText.trim()) return;

    const newReply: Comment = {
      id: `reply-${Date.now()}`,
      documentId: docState.id,
      blockId: parentComment.blockId,
      selectedText: parentComment.selectedText,
      startOffset: parentComment.startOffset,
      endOffset: parentComment.endOffset,
      commentText: replyText,
      username: 'Current User', // TODO: Get from auth
      timestamp: new Date().toISOString(),
      replies: [],
      parentId: parentComment.id,
    };

    // Add reply to parent comment
    const updatedComments = comments.map(c => 
      c.id === parentComment.id 
        ? { ...c, replies: [...c.replies, newReply] }
        : c
    );
    setComments(updatedComments);
    setReplyingTo(null);
    setShowCommentModal(false);

    // TODO: Send to backend
    console.log('Created reply:', newReply);
  };

  const handleClickComment = (comment: Comment) => {
    // Scroll to and highlight the commented text in the editor
    if (editorInstance) {
      highlightCommentInEditor(
        editorInstance,
        comment.blockId,
        comment.startOffset,
        comment.endOffset
      );
    }
    console.log('Clicked comment:', comment);
  };
  
  const toggleCommentCollapse = (commentId: string) => {
    setCollapsedComments(prev => {
      const newSet = new Set(prev);
      if (newSet.has(commentId)) {
        newSet.delete(commentId);
      } else {
        newSet.add(commentId);
      }
      return newSet;
    });
  };

  // Handle clicks on commented text in the editor
  const handleCommentTextClick = (commentIds: string[]) => {
    // Open the comment panel
    setShowCommentPanel(true);
    
    // Expand all clicked comments (remove from collapsed set)
    setCollapsedComments(prev => {
      const newSet = new Set(prev);
      commentIds.forEach(id => newSet.delete(id));
      return newSet;
    });
    
    // Scroll to the first comment in the panel
    if (commentIds.length > 0) {
      setTimeout(() => {
        const firstCommentId = commentIds[0];
        const commentElement = document.querySelector(`[data-comment-id="${firstCommentId}"]`);
        if (commentElement) {
          commentElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          // Flash highlight effect
          commentElement.classList.add('flash-highlight');
          setTimeout(() => {
            commentElement.classList.remove('flash-highlight');
          }, 1000);
        }
      }, 100);
    }
  };

  // Wrapper for toolbar "Turn Into" handler
  const handleToolbarTurnInto = (type: string) => {
    switch (type) {
      case 'paragraph':
        handleTurnInto('paragraph');
        break;
      case 'heading-1':
        handleTurnInto('heading', { level: 1 });
        break;
      case 'heading-2':
        handleTurnInto('heading', { level: 2 });
        break;
      case 'heading-3':
        handleTurnInto('heading', { level: 3 });
        break;
      case 'bulleted-list':
        handleTurnInto('list', { listStyle: 'bullet' });
        break;
      case 'numbered-list':
        handleTurnInto('list', { listStyle: 'number' });
        break;
      case 'code':
        handleTurnInto('code');
        break;
      case 'quote':
        handleTurnInto('quote');
        break;
    }
  };

  return (
    <div className="h-screen flex flex-col bg-neutral-50">
      {/* Header */}
      <div className="bg-white border-b border-neutral-200 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-neutral-900">
            Single Document Editor - Demo
          </h1>
          <p className="text-sm text-neutral-600 mt-1">
            Notion-like experience: One Lexical editor for the entire document
          </p>
        </div>
        {/* Removed panel toggle buttons - use collapse buttons in panel headers instead */}
      </div>

      {/* 3-Panel Layout: Scratchpad | Main Editor | Controls */}
      <div className="flex-1 flex overflow-hidden" style={{ gap: '0' }}>
        
        {/* LEFT PANEL: Scratchpad */}
        <div 
          style={{
            width: scratchpadCollapsed ? '48px' : `${scratchpadWidth}%`,
            flexShrink: 0,
            display: 'flex',
            flexDirection: 'column',
            backgroundColor: 'white',
            borderRight: '1px solid #e5e5e7',
            transition: scratchpadCollapsed ? 'width 0.3s ease' : 'none',
            overflow: 'hidden',
          }}
        >
          {/* Scratchpad Header */}
          <div 
            style={{
              backgroundColor: '#fef3c7',
              padding: '8px 12px',
              borderBottom: '1px solid #fbbf24',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              userSelect: 'none',
              minHeight: '40px',
            }}
          >
            {!scratchpadCollapsed && (
              <span style={{ fontSize: '13px', fontWeight: '600', color: '#92400e' }}>
                📝 Scratchpad
              </span>
            )}
        <button
              onClick={() => setScratchpadCollapsed(!scratchpadCollapsed)}
              style={{
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                fontSize: '16px',
                padding: '4px',
                marginLeft: 'auto',
              }}
              title={scratchpadCollapsed ? 'Expand Scratchpad' : 'Collapse Scratchpad'}
            >
              {scratchpadCollapsed ? '▶' : '◀'}
        </button>
      </div>

          {/* Scratchpad Content */}
          {!scratchpadCollapsed && (
            <div style={{ flex: 1, overflow: 'auto', padding: '16px' }}>
              <SingleDocumentEditor
                initialDoc={scratchpadDoc}
                onDocChange={setScratchpadDoc}
                readOnly={false}
                onEditorReady={setScratchpadEditorInstance}
              />
            </div>
          )}
        </div>

        {/* Resize Handle: Scratchpad <-> Main */}
        {!scratchpadCollapsed && (
          <div
            style={{
              width: '4px',
              cursor: 'col-resize',
              backgroundColor: '#e5e5e7',
              transition: 'background-color 0.2s',
            }}
            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#3b82f6'}
            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#e5e5e7'}
            onMouseDown={(e) => {
              e.preventDefault();
              const startX = e.clientX;
              const startScratchpadWidth = scratchpadWidth;

              const handleMouseMove = (moveE: MouseEvent) => {
                const deltaX = moveE.clientX - startX;
                const containerWidth = window.innerWidth;
                const deltaPercent = (deltaX / containerWidth) * 100;
                
                // Constrain scratchpad width (10-40%), center panel auto-fills
                const newScratchpadWidth = Math.max(10, Math.min(40, startScratchpadWidth + deltaPercent));
                setScratchpadWidth(newScratchpadWidth);
              };

              const handleMouseUp = () => {
                document.removeEventListener('mousemove', handleMouseMove);
                document.removeEventListener('mouseup', handleMouseUp);
              };

              document.addEventListener('mousemove', handleMouseMove);
              document.addEventListener('mouseup', handleMouseUp);
            }}
          />
        )}

        {/* CENTER PANEL: Main Document Editor */}
        <div 
          style={{
            flex: mainEditorCollapsed ? '0 0 48px' : '1 1 auto',
            minWidth: mainEditorCollapsed ? '48px' : '300px',
            display: 'flex',
            flexDirection: 'column',
            backgroundColor: 'white',
            position: 'relative',
            transition: mainEditorCollapsed ? 'flex 0.3s ease' : 'none',
            overflow: 'hidden',
          }}
        >
          {/* Main Editor Header */}
          <div 
            style={{
              backgroundColor: '#dbeafe',
              padding: '8px 12px',
              borderBottom: '1px solid #3b82f6',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              userSelect: 'none',
              minHeight: '40px',
            }}
          >
            {!mainEditorCollapsed && (
              <span style={{ fontSize: '13px', fontWeight: '600', color: '#1e40af' }}>
                📄 Document
              </span>
            )}
            <button
              onClick={() => setMainEditorCollapsed(!mainEditorCollapsed)}
              style={{
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                fontSize: '16px',
                padding: '4px',
                marginLeft: 'auto',
              }}
              title={mainEditorCollapsed ? 'Expand Editor' : 'Collapse Editor'}
            >
              {mainEditorCollapsed ? '↕' : '−'}
            </button>
          </div>

          {/* Main Editor Content with integrated comment panel */}
          {!mainEditorCollapsed && (
            <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
              {/* Editor Area */}
              <div style={{ flex: 1, overflow: 'auto' }}>
            <SingleDocumentEditor
              initialDoc={docState}
              onDocChange={handleDocChange}
              readOnly={false}
              onEditorReady={setEditorInstance}
                  onSelectionChange={setSelectionData}
                  onFormat={handleFormat}
                  onTextColor={handleTextColor}
                  onBackgroundColor={handleBackgroundColor}
                  onTurnInto={handleToolbarTurnInto}
                  onAddComment={handleAddComment}
                  onImproveText={handleImproveText}
                  onCommentClick={handleCommentTextClick}
            />
          </div>

              {/* Comment Panel - Right Margin (50% smaller, compact) */}
              {showCommentPanel && (
                <div 
                  style={{
                    width: '150px',
                    borderLeft: '1px solid #e5e5e7',
                    backgroundColor: '#f9fafb',
                    display: 'flex',
                    flexDirection: 'column',
                    overflow: 'hidden',
                  }}
                >
                  {/* Comment Panel Header */}
                  <div style={{
                    padding: '6px 8px',
                    borderBottom: '1px solid #e5e5e7',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    backgroundColor: 'white',
                  }}>
                    <span style={{ fontSize: '10px', fontWeight: '600', color: '#6b7280' }}>
                      💬 {comments.length}
                    </span>
                    <button
                      onClick={() => setShowCommentPanel(false)}
                      style={{
                        background: 'transparent',
                        border: 'none',
                        cursor: 'pointer',
                        fontSize: '12px',
                        padding: '2px',
                        color: '#9ca3af',
                      }}
                    >
                      ✕
                    </button>
        </div>

                  {/* Comments List */}
                  <div style={{ flex: 1, overflow: 'auto', padding: '8px' }}>
                    {comments.length === 0 ? (
                      <div style={{
                        textAlign: 'center',
                        color: '#9ca3af',
                        padding: '16px 8px',
                        fontSize: '9px',
                      }}>
                        No comments yet
                      </div>
                    ) : (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        {comments.map((comment) => {
                          const isCollapsed = collapsedComments.has(comment.id);
                          return (
                            <div
                              key={comment.id}
                              data-comment-id={comment.id}
                              style={{
                                borderRadius: '6px',
                                padding: '6px',
                                backgroundColor: 'white',
                                boxShadow: '0 1px 3px rgba(0, 0, 0, 0.08)',
                                fontSize: '9px',
                                cursor: 'pointer',
                              }}
                              onClick={(e) => {
                                e.stopPropagation();
                                toggleCommentCollapse(comment.id);
                              }}
                            >
                              {/* Avatar + Name */}
                              <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginBottom: isCollapsed ? '0' : '4px' }}>
                                <div style={{
                                  width: '14px',
                                  height: '14px',
                                  borderRadius: '50%',
                                  backgroundColor: '#f59e0b',
                                  color: 'white',
                                  display: 'flex',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                  fontSize: '8px',
                                  fontWeight: '600',
                                  flexShrink: 0,
                                }}>
                                  {comment.username.charAt(0).toUpperCase()}
                                </div>
                                <div style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: '9px', fontWeight: '600' }}>
                                  {comment.username}
                                </div>
          </div>
          
                              {/* Comment Text (truncated when collapsed) */}
                              {!isCollapsed && (
                                <div style={{ fontSize: '8px', color: '#374151', lineHeight: '1.4', marginBottom: '4px' }}>
                                  {comment.commentText}
                                </div>
                              )}

                              {/* Actions (when expanded) */}
                              {!isCollapsed && (
                                <div style={{ display: 'flex', gap: '4px', fontSize: '7px' }}>
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleEditComment(comment);
                                    }}
                                    style={{
                                      padding: '2px 4px',
                                      border: 'none',
                                      borderRadius: '3px',
                                      backgroundColor: '#f3f4f6',
                                      cursor: 'pointer',
                                      fontSize: '7px',
                                    }}
                                  >
                                    Edit
                                  </button>
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleDeleteComment(comment.id);
                                    }}
                                    style={{
                                      padding: '2px 4px',
                                      border: 'none',
                                      borderRadius: '3px',
                                      backgroundColor: '#fee2e2',
                                      cursor: 'pointer',
                                      fontSize: '7px',
                                      color: '#dc2626',
                                    }}
                                  >
                                    Del
                                  </button>
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Resize Handle: Main <-> Controls */}
        {!mainEditorCollapsed && !controlsCollapsed && (
          <div
            style={{
              width: '4px',
              cursor: 'col-resize',
              backgroundColor: '#e5e5e7',
              transition: 'background-color 0.2s',
            }}
            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#3b82f6'}
            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#e5e5e7'}
            onMouseDown={(e) => {
              e.preventDefault();
              const startX = e.clientX;
              const startControlsWidth = controlsWidth;

              const handleMouseMove = (moveE: MouseEvent) => {
                const deltaX = moveE.clientX - startX;
                const containerWidth = window.innerWidth;
                const deltaPercent = (deltaX / containerWidth) * 100;
                
                // Constrain controls width (10-40%), center panel auto-fills
                const newControlsWidth = Math.max(10, Math.min(40, startControlsWidth - deltaPercent));
                setControlsWidth(newControlsWidth);
              };

              const handleMouseUp = () => {
                document.removeEventListener('mousemove', handleMouseMove);
                document.removeEventListener('mouseup', handleMouseUp);
              };

              document.addEventListener('mousemove', handleMouseMove);
              document.addEventListener('mouseup', handleMouseUp);
            }}
          />
        )}

        {/* RIGHT PANEL: Controls */}
        <div 
          style={{
            width: controlsCollapsed ? '48px' : `${controlsWidth}%`,
            flexShrink: 0,
            display: 'flex',
            flexDirection: 'column',
            backgroundColor: 'white',
            borderLeft: '1px solid #e5e5e7',
            transition: controlsCollapsed ? 'width 0.3s ease' : 'none',
            overflow: 'hidden',
          }}
        >
          {/* Controls Header */}
          <div 
            style={{
              backgroundColor: '#f3f4f6',
              padding: '8px 12px',
              borderBottom: '1px solid #e5e5e7',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              userSelect: 'none',
              minHeight: '40px',
            }}
          >
            {!controlsCollapsed && (
              <span style={{ fontSize: '13px', fontWeight: '600', color: '#374151' }}>
                🎛️ Controls
              </span>
            )}
            <button
              onClick={() => setControlsCollapsed(!controlsCollapsed)}
              style={{
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                fontSize: '16px',
                padding: '4px',
                marginLeft: 'auto',
              }}
              title={controlsCollapsed ? 'Expand Controls' : 'Collapse Controls'}
            >
              {controlsCollapsed ? '◀' : '▶'}
            </button>
          </div>
          
          {/* Controls Content */}
          {!controlsCollapsed && (
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {/* Pending AI Suggestion - TOP PRIORITY */}
            {pendingSuggestion && (
              <div style={{ border: '2px solid rgb(168, 85, 247)', borderRadius: '8px', padding: '12px', backgroundColor: 'rgb(250, 245, 255)' }}>
                <div style={{ marginBottom: '8px' }}>
                  <span style={{ fontSize: '12px', fontWeight: 'bold', color: 'rgb(88, 28, 135)' }}>🤖 AI Suggestion</span>
                </div>
                
                <div style={{ fontSize: '10px', marginBottom: '4px' }}>
                  <span style={{ color: '#666' }}>Was:</span> {pendingSuggestion.original}
                </div>
                
                <div style={{ fontSize: '10px', marginBottom: '12px', color: 'rgb(29, 78, 216)', fontWeight: '500' }}>
                  <span>→</span> {pendingSuggestion.improved}
                </div>
                
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    onClick={handleAcceptPendingSuggestion}
                    style={{
                      flex: 1,
                      padding: '10px',
                      fontSize: '13px',
                      fontWeight: 'bold',
                      backgroundColor: 'rgb(22, 163, 74)',
                      color: 'white',
                      border: 'none',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      display: 'block'
                    }}
                    onMouseOver={(e) => e.currentTarget.style.backgroundColor = 'rgb(21, 128, 61)'}
                    onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'rgb(22, 163, 74)'}
                  >
                    ✓ Accept
                  </button>
                  <button
                    onClick={handleRejectPendingSuggestion}
                    style={{
                      flex: 1,
                      padding: '10px',
                      fontSize: '13px',
                      fontWeight: 'bold',
                      backgroundColor: 'rgb(220, 38, 38)',
                      color: 'white',
                      border: 'none',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      display: 'block'
                    }}
                    onMouseOver={(e) => e.currentTarget.style.backgroundColor = 'rgb(185, 28, 28)'}
                    onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'rgb(220, 38, 38)'}
                  >
                    ✗ Reject
                  </button>
                </div>
            </div>
            )}

            {/* Selection Bridge (Live) */}
            <div className="border border-neutral-200 rounded-lg p-2">
              <h3 className="text-xs font-semibold mb-1">Selection Bridge (Live)</h3>
              <div className="space-y-1 text-[10px]">
                <div className="bg-neutral-50 p-1 rounded">
                  <span className="font-medium">Scope:</span>
                  <span className={`ml-1 px-1 py-0.5 rounded ${
                    selectionData.selectionScope === 'blocks' ? 'bg-blue-100 text-blue-700' :
                    selectionData.selectionScope === 'text' ? 'bg-green-100 text-green-700' :
                    'bg-neutral-200 text-neutral-600'
                  }`}>
                    {selectionData.selectionScope}
                  </span>
                </div>
                
                {selectionData.currentBlockType && (
                  <div className="bg-neutral-50 p-1 rounded">
                    <span className="font-medium">Block Type:</span>
                    <span className="ml-1 px-1 py-0.5 rounded bg-purple-100 text-purple-700">
                      {selectionData.currentBlockType}
                      {selectionData.currentBlockLevel ? ` (${selectionData.currentBlockLevel})` : ''}
                      {selectionData.currentListStyle ? ` (${selectionData.currentListStyle})` : ''}
                    </span>
                  </div>
                )}
                
                {selectionData.selectedText && (
                  <div className="bg-neutral-50 p-1 rounded">
                    <span className="font-medium">Text ({selectionData.selectedText.length}):</span> "{selectionData.selectedText.substring(0, 50)}{selectionData.selectedText.length > 50 ? '...' : ''}"
                  </div>
                )}
              </div>
            </div>

            {/* Turn Into */}
            <div className="border border-neutral-200 rounded-lg p-2">
              <h3 className="text-xs font-semibold mb-1">Turn Into</h3>
              <div className="space-y-1">
                <button
                  onClick={() => handleTurnInto('paragraph')}
                  style={{
                    width: '100%',
                    padding: '6px 8px',
                    fontSize: '11px',
                    textAlign: 'left',
                    backgroundColor: selectionData.currentBlockType === 'paragraph' ? '#3b82f6' : 'white',
                    color: selectionData.currentBlockType === 'paragraph' ? 'white' : 'black',
                    border: '1px solid #e5e5e7',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontWeight: selectionData.currentBlockType === 'paragraph' ? '600' : 'normal',
                  }}
                  onMouseOver={(e) => {
                    if (selectionData.currentBlockType !== 'paragraph') {
                      e.currentTarget.style.backgroundColor = '#f5f5f5';
                    }
                  }}
                  onMouseOut={(e) => {
                    if (selectionData.currentBlockType !== 'paragraph') {
                      e.currentTarget.style.backgroundColor = 'white';
                    }
                  }}
                >
                  T Text
                </button>
                <button
                  onClick={() => handleTurnInto('heading', 1)}
                  style={{
                    width: '100%',
                    padding: '6px 8px',
                    fontSize: '11px',
                    textAlign: 'left',
                    backgroundColor: selectionData.currentBlockType === 'heading' && selectionData.currentBlockLevel === 1 ? '#3b82f6' : 'white',
                    color: selectionData.currentBlockType === 'heading' && selectionData.currentBlockLevel === 1 ? 'white' : 'black',
                    border: '1px solid #e5e5e7',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontWeight: selectionData.currentBlockType === 'heading' && selectionData.currentBlockLevel === 1 ? '600' : 'normal',
                  }}
                  onMouseOver={(e) => {
                    if (!(selectionData.currentBlockType === 'heading' && selectionData.currentBlockLevel === 1)) {
                      e.currentTarget.style.backgroundColor = '#f5f5f5';
                    }
                  }}
                  onMouseOut={(e) => {
                    if (!(selectionData.currentBlockType === 'heading' && selectionData.currentBlockLevel === 1)) {
                      e.currentTarget.style.backgroundColor = 'white';
                    }
                  }}
                >
                  H1 Heading 1
                </button>
                <button
                  onClick={() => handleTurnInto('heading', 2)}
                  style={{
                    width: '100%',
                    padding: '6px 8px',
                    fontSize: '11px',
                    textAlign: 'left',
                    backgroundColor: selectionData.currentBlockType === 'heading' && selectionData.currentBlockLevel === 2 ? '#3b82f6' : 'white',
                    color: selectionData.currentBlockType === 'heading' && selectionData.currentBlockLevel === 2 ? 'white' : 'black',
                    border: '1px solid #e5e5e7',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontWeight: selectionData.currentBlockType === 'heading' && selectionData.currentBlockLevel === 2 ? '600' : 'normal',
                  }}
                  onMouseOver={(e) => {
                    if (!(selectionData.currentBlockType === 'heading' && selectionData.currentBlockLevel === 2)) {
                      e.currentTarget.style.backgroundColor = '#f5f5f5';
                    }
                  }}
                  onMouseOut={(e) => {
                    if (!(selectionData.currentBlockType === 'heading' && selectionData.currentBlockLevel === 2)) {
                      e.currentTarget.style.backgroundColor = 'white';
                    }
                  }}
                >
                  H2 Heading 2
                </button>
                <button
                  onClick={() => handleTurnInto('heading', 3)}
                  style={{
                    width: '100%',
                    padding: '6px 8px',
                    fontSize: '11px',
                    textAlign: 'left',
                    backgroundColor: selectionData.currentBlockType === 'heading' && selectionData.currentBlockLevel === 3 ? '#3b82f6' : 'white',
                    color: selectionData.currentBlockType === 'heading' && selectionData.currentBlockLevel === 3 ? 'white' : 'black',
                    border: '1px solid #e5e5e7',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontWeight: selectionData.currentBlockType === 'heading' && selectionData.currentBlockLevel === 3 ? '600' : 'normal',
                  }}
                  onMouseOver={(e) => {
                    if (!(selectionData.currentBlockType === 'heading' && selectionData.currentBlockLevel === 3)) {
                      e.currentTarget.style.backgroundColor = '#f5f5f5';
                    }
                  }}
                  onMouseOut={(e) => {
                    if (!(selectionData.currentBlockType === 'heading' && selectionData.currentBlockLevel === 3)) {
                      e.currentTarget.style.backgroundColor = 'white';
                    }
                  }}
                >
                  H3 Heading 3
                </button>
                <button
                  onClick={() => handleTurnInto('list', { listStyle: 'bullet' })}
                  style={{
                    width: '100%',
                    padding: '6px 8px',
                    fontSize: '11px',
                    textAlign: 'left',
                    backgroundColor: selectionData.currentBlockType === 'list' && selectionData.currentListStyle === 'bullet' ? '#3b82f6' : 'white',
                    color: selectionData.currentBlockType === 'list' && selectionData.currentListStyle === 'bullet' ? 'white' : 'black',
                    border: '1px solid #e5e5e7',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontWeight: selectionData.currentBlockType === 'list' && selectionData.currentListStyle === 'bullet' ? '600' : 'normal',
                  }}
                  onMouseOver={(e) => {
                    if (!(selectionData.currentBlockType === 'list' && selectionData.currentListStyle === 'bullet')) {
                      e.currentTarget.style.backgroundColor = '#f5f5f5';
                    }
                  }}
                  onMouseOut={(e) => {
                    if (!(selectionData.currentBlockType === 'list' && selectionData.currentListStyle === 'bullet')) {
                      e.currentTarget.style.backgroundColor = 'white';
                    }
                  }}
                >
                  • Bulleted List
                </button>
                <button
                  onClick={() => handleTurnInto('list', { listStyle: 'number' })}
                  style={{
                    width: '100%',
                    padding: '6px 8px',
                    fontSize: '11px',
                    textAlign: 'left',
                    backgroundColor: selectionData.currentBlockType === 'list' && selectionData.currentListStyle === 'number' ? '#3b82f6' : 'white',
                    color: selectionData.currentBlockType === 'list' && selectionData.currentListStyle === 'number' ? 'white' : 'black',
                    border: '1px solid #e5e5e7',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontWeight: selectionData.currentBlockType === 'list' && selectionData.currentListStyle === 'number' ? '600' : 'normal',
                  }}
                  onMouseOver={(e) => {
                    if (!(selectionData.currentBlockType === 'list' && selectionData.currentListStyle === 'number')) {
                      e.currentTarget.style.backgroundColor = '#f5f5f5';
                    }
                  }}
                  onMouseOut={(e) => {
                    if (!(selectionData.currentBlockType === 'list' && selectionData.currentListStyle === 'number')) {
                      e.currentTarget.style.backgroundColor = 'white';
                    }
                  }}
                >
                  1. Numbered List
                </button>
                <button
                  onClick={() => handleTurnInto('code')}
                  style={{
                    width: '100%',
                    padding: '6px 8px',
                    fontSize: '11px',
                    textAlign: 'left',
                    backgroundColor: selectionData.currentBlockType === 'code' ? '#3b82f6' : 'white',
                    color: selectionData.currentBlockType === 'code' ? 'white' : 'black',
                    border: '1px solid #e5e5e7',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontWeight: selectionData.currentBlockType === 'code' ? '600' : 'normal',
                  }}
                  onMouseOver={(e) => {
                    if (selectionData.currentBlockType !== 'code') {
                      e.currentTarget.style.backgroundColor = '#f5f5f5';
                    }
                  }}
                  onMouseOut={(e) => {
                    if (selectionData.currentBlockType !== 'code') {
                      e.currentTarget.style.backgroundColor = 'white';
                    }
                  }}
                >
                  &lt;/&gt; Code
                </button>
                <button
                  onClick={() => handleTurnInto('quote')}
                  style={{
                    width: '100%',
                    padding: '6px 8px',
                    fontSize: '11px',
                    textAlign: 'left',
                    backgroundColor: selectionData.currentBlockType === 'quote' ? '#3b82f6' : 'white',
                    color: selectionData.currentBlockType === 'quote' ? 'white' : 'black',
                    border: '1px solid #e5e5e7',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontWeight: selectionData.currentBlockType === 'quote' ? '600' : 'normal',
                  }}
                  onMouseOver={(e) => {
                    if (selectionData.currentBlockType !== 'quote') {
                      e.currentTarget.style.backgroundColor = '#f5f5f5';
                    }
                  }}
                  onMouseOut={(e) => {
                    if (selectionData.currentBlockType !== 'quote') {
                      e.currentTarget.style.backgroundColor = 'white';
                    }
                  }}
                >
                  " Quote
                  </button>
                </div>
            </div>

            {/* Text Formatting */}
            <div className="border border-neutral-200 rounded-lg p-2">
              <h3 className="text-xs font-semibold mb-1">Formatting</h3>
              <div className="flex gap-2 mb-3">
                <button
                  onClick={() => handleFormat('bold')}
                  disabled={selectionData.isEmpty}
                  style={{
                    flex: 1,
                    padding: '6px',
                    fontSize: '13px',
                    fontWeight: 'bold',
                    backgroundColor: selectionData.isEmpty ? '#e5e5e5' : '#374151',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: selectionData.isEmpty ? 'not-allowed' : 'pointer',
                  }}
                  title="Bold"
                >
                  B
                </button>
                <button
                  onClick={() => handleFormat('italic')}
                  disabled={selectionData.isEmpty}
                  style={{
                    flex: 1,
                    padding: '6px',
                    fontSize: '13px',
                    fontStyle: 'italic',
                    fontWeight: 'bold',
                    backgroundColor: selectionData.isEmpty ? '#e5e5e5' : '#374151',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: selectionData.isEmpty ? 'not-allowed' : 'pointer',
                  }}
                  title="Italic"
                >
                  I
                </button>
                <button
                  onClick={() => handleFormat('underline')}
                  disabled={selectionData.isEmpty}
                  style={{
                    flex: 1,
                    padding: '6px',
                    fontSize: '13px',
                    fontWeight: 'bold',
                    textDecoration: 'underline',
                    backgroundColor: selectionData.isEmpty ? '#e5e5e5' : '#374151',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: selectionData.isEmpty ? 'not-allowed' : 'pointer',
                  }}
                  title="Underline"
                >
                  U
                </button>
                </div>

              {/* Color Pickers */}
              <div className="mt-3 pt-3 border-t border-neutral-200">
                <div className="mb-3">
                  <label className="text-xs font-medium mb-1 block">Text Color</label>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '4px' }}>
                    {['#000000', '#374151', '#DC2626', '#EA580C', '#D97706', '#CA8A04', '#65A30D', '#16A34A', '#0891B2', '#0EA5E9', '#3B82F6', '#6366F1', '#8B5CF6', '#A855F7', '#D946EF'].map((color) => (
                      <button
                        key={color}
                        onClick={() => handleTextColor(color)}
                        disabled={selectionData.isEmpty}
                        style={{
                          width: '100%',
                          height: '24px',
                          backgroundColor: color,
                          border: '1px solid #e5e5e5',
                          borderRadius: '4px',
                          cursor: selectionData.isEmpty ? 'not-allowed' : 'pointer',
                          opacity: selectionData.isEmpty ? 0.5 : 1,
                        }}
                        title={color}
                      />
                    ))}
                  </div>
                </div>

                <div>
                  <label className="text-xs font-medium mb-1 block">Background Color</label>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '4px' }}>
                    {['#FFFFFF', '#F3F4F6', '#FEE2E2', '#FFEDD5', '#FEF3C7', '#FEF9C3', '#ECFCCB', '#D1FAE5', '#CFFAFE', '#DBEAFE', '#C7D2FE', '#E0E7FF', '#EDE9FE', '#F3E8FF', '#FAE8FF'].map((color) => (
                      <button
                        key={color}
                        onClick={() => handleBackgroundColor(color)}
                        disabled={selectionData.isEmpty}
                        style={{
                          width: '100%',
                          height: '24px',
                          backgroundColor: color,
                          border: '1px solid #d1d5db',
                          borderRadius: '4px',
                          cursor: selectionData.isEmpty ? 'not-allowed' : 'pointer',
                          opacity: selectionData.isEmpty ? 0.5 : 1,
                        }}
                        title={color}
                      />
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* AI Suggestion Tools */}
            <div className="border border-neutral-200 rounded-lg p-3">
              <h3 className="text-sm font-semibold mb-2">AI Suggestions</h3>
              <div className="space-y-2">
                <Button
                  onClick={handleImproveText}
                  size="sm"
                  className="w-full bg-purple-600 hover:bg-purple-700"
                  disabled={selectionData.isEmpty || pendingSuggestion !== null}
                >
                  🤖 Improve Selected Text
                </Button>
                <Button
                  onClick={handleInsertAiSuggestion}
                  size="sm"
                  className="w-full"
                  variant="default"
                >
                  Insert Demo Suggestion
                </Button>
                <Button
                  onClick={handleApplySelection}
                  size="sm"
                  className="w-full bg-green-600 hover:bg-green-700"
                >
                  Mark as Applied (Grey)
                </Button>
                <Button
                  onClick={handleRejectSelection}
                  size="sm"
                  className="w-full bg-red-600 hover:bg-red-700"
                >
                  Mark as Rejected (Red)
                </Button>
                <Button
                  onClick={handleClearSelection}
                  size="sm"
                  className="w-full"
                  variant="outline"
                >
                  Clear Status
                </Button>
              </div>
            </div>

            {/* Comments */}
            <div className="border border-neutral-200 rounded-lg p-3">
              <h3 className="text-sm font-semibold mb-2">Comments</h3>
              <div className="space-y-2">
                <Button
                  onClick={handleAddComment}
                  size="sm"
                  className="w-full bg-amber-600 hover:bg-amber-700"
                  disabled={selectionData.isEmpty}
                >
                  💬 Add Comment
                </Button>
                <Button
                  onClick={() => setShowCommentPanel(!showCommentPanel)}
                  size="sm"
                  className="w-full"
                  variant="outline"
                >
                  {showCommentPanel ? '✕ Hide' : '👁️ View'} Comments ({comments.length})
                </Button>
              </div>
            </div>

            {/* Legend */}
            <div className="border border-neutral-200 rounded-lg p-3">
              <h3 className="text-sm font-semibold mb-2">Track Changes Legend</h3>
              <div className="space-y-2 text-xs">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded border-2 border-blue-500 bg-blue-50"></div>
                  <span>AI Suggested (Blue)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded bg-neutral-200"></div>
                  <span>AI Applied (Grey)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded bg-red-100 border border-red-500"></div>
                  <span>AI Rejected (Red)</span>
                </div>
              </div>
            </div>

            {/* Document State */}
            <div className="border border-neutral-200 rounded-lg p-3">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-semibold">DocState JSON</h3>
                <button
                  onClick={() => setShowJson(!showJson)}
                  className="text-xs text-blue-600 hover:underline"
                >
                  {showJson ? 'Hide' : 'Show'}
                </button>
              </div>
              {showJson && (
                <pre className="text-[10px] bg-neutral-900 text-green-400 p-2 rounded overflow-x-auto max-h-64 overflow-y-auto font-mono">
                  {JSON.stringify(docState, null, 2)}
                </pre>
              )}
              <div className="mt-2 text-xs text-neutral-600">
                <div>Blocks: {docState.blocks.length}</div>
                <div>Version: {docState.version || 'N/A'}</div>
              </div>
            </div>

            {/* Instructions */}
            <div className="border border-amber-200 bg-amber-50 rounded-lg p-3">
              <h3 className="text-sm font-semibold mb-2 text-amber-800">Instructions</h3>
              <ol className="text-xs text-amber-900 space-y-1 list-decimal list-inside">
                <li>Select text in the editor</li>
                <li>Click "Get Selection" to test selection API</li>
                <li>Click "Insert AI Suggestion" to add blue text</li>
                <li>Select the blue text and mark as Applied/Rejected</li>
                <li>Watch the JSON update in real-time</li>
              </ol>
            </div>
          </div>
          )}
        </div>
      </div>

      {/* Comment Modal */}
      {showCommentModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
        }}>
          <div style={{
            backgroundColor: 'white',
            borderRadius: '8px',
            padding: '24px',
            width: '90%',
            maxWidth: '500px',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
          }}>
            <h3 style={{
              fontSize: '18px',
              fontWeight: '600',
              marginBottom: '12px',
            }}>
              {editingComment ? 'Edit Comment' : replyingTo ? 'Reply to Comment' : 'Add Comment'}
            </h3>
            
            {(editingComment || replyingTo) && (
              <div style={{
                backgroundColor: '#f3f4f6',
                padding: '12px',
                borderRadius: '6px',
                marginBottom: '12px',
                fontSize: '14px',
                color: '#6b7280',
              }}>
                <strong>Selected text:</strong> "{editingComment?.selectedText || replyingTo?.selectedText}"
        </div>
        )}

            {!editingComment && !replyingTo && (
              <div style={{
                backgroundColor: '#fef9c3',
                padding: '12px',
                borderRadius: '6px',
                marginBottom: '12px',
                fontSize: '14px',
              }}>
                <strong>Selected text:</strong> "{selectionData.selectedText}"
      </div>
            )}

            <textarea
              autoFocus
              placeholder="Write your comment..."
              defaultValue={editingComment?.commentText || ''}
              style={{
                width: '100%',
                minHeight: '100px',
                padding: '12px',
                border: '1px solid #d1d5db',
                borderRadius: '6px',
                fontSize: '14px',
                fontFamily: 'inherit',
                resize: 'vertical',
                marginBottom: '16px',
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && e.metaKey) {
                  const value = e.currentTarget.value;
                  if (editingComment) {
                    handleUpdateComment(editingComment.id, value);
                  } else if (replyingTo) {
                    handleCreateReply(replyingTo, value);
                  } else {
                    handleCreateComment(value);
                  }
                }
              }}
              id="comment-textarea"
            />

            <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
              <button
                onClick={() => {
                  setShowCommentModal(false);
                  setEditingComment(null);
                  setReplyingTo(null);
                }}
                style={{
                  padding: '8px 16px',
                  border: '1px solid #d1d5db',
                  borderRadius: '6px',
                  backgroundColor: 'white',
                  cursor: 'pointer',
                  fontSize: '14px',
                }}
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  const textarea = document.getElementById('comment-textarea') as HTMLTextAreaElement;
                  const value = textarea?.value || '';
                  if (editingComment) {
                    handleUpdateComment(editingComment.id, value);
                  } else if (replyingTo) {
                    handleCreateReply(replyingTo, value);
                  } else {
                    handleCreateComment(value);
                  }
                }}
                style={{
                  padding: '8px 16px',
                  border: 'none',
                  borderRadius: '6px',
                  backgroundColor: '#f59e0b',
                  color: 'white',
                  cursor: 'pointer',
                  fontSize: '14px',
                  fontWeight: '600',
                }}
              >
                {editingComment ? 'Update' : 'Comment'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Old comment panel removed - now integrated into middle panel */}
      {false && showCommentPanel && (
        <div 
          className="comment-panel-scroll"
          style={{
            position: 'fixed',
            right: '20px',
            top: '80px',
            bottom: '20px',
            width: '300px',
            pointerEvents: 'none',
            overflowY: 'auto',
            zIndex: 150,
            transition: 'right 0.3s ease',
          }}
        >
          {/* Close button - floating at top right */}
          <button
            onClick={() => setShowCommentPanel(false)}
            style={{
              position: 'fixed',
              right: '22px',
              top: '82px',
              width: '28px',
              height: '28px',
              border: 'none',
              background: 'white',
              cursor: 'pointer',
              fontSize: '16px',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#6b7280',
              boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
              pointerEvents: 'auto',
              zIndex: 151,
              transition: 'right 0.3s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#f3f4f6';
              e.currentTarget.style.transform = 'scale(1.1)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'white';
              e.currentTarget.style.transform = 'scale(1)';
            }}
          >
            ✕
          </button>

          {/* Comments List */}
          <div style={{ padding: '0', pointerEvents: 'auto' }}>
            {comments.length === 0 ? (
              <div style={{
                textAlign: 'center',
                color: '#9ca3af',
                padding: '48px 16px',
                fontSize: '13px',
                lineHeight: '1.6',
                backgroundColor: 'white',
                borderRadius: '12px',
                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)',
              }}>
                <div style={{ fontSize: '32px', marginBottom: '8px' }}>💭</div>
                No comments yet<br/>
                <span style={{ fontSize: '11px' }}>Select text and add a comment</span>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {comments.map((comment) => {
                  const isCollapsed = collapsedComments.has(comment.id);
                  return (
                  <div
                    key={comment.id}
                    data-comment-id={comment.id}
                    style={{
                      border: 'none',
                      borderRadius: '12px',
                      padding: '12px',
                      backgroundColor: 'white',
                      boxShadow: '0 2px 12px rgba(0, 0, 0, 0.1)',
                      transition: 'all 0.2s ease',
                      cursor: 'pointer',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.15)';
                      e.currentTarget.style.transform = 'translateY(-2px)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.boxShadow = '0 2px 12px rgba(0, 0, 0, 0.1)';
                      e.currentTarget.style.transform = 'translateY(0)';
                    }}
                  >
                    {/* Comment Header - Clickable to toggle collapse */}
                    <div 
                      style={{
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: '8px',
                        marginBottom: isCollapsed ? '0' : '8px',
                        cursor: 'pointer',
                      }}
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleCommentCollapse(comment.id);
                      }}
                    >
                      {/* Avatar */}
                      <div style={{
                        width: '28px',
                        height: '28px',
                        borderRadius: '50%',
                        backgroundColor: '#f59e0b',
                        color: 'white',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '12px',
                        fontWeight: '600',
                        flexShrink: 0,
                      }}>
                        {comment.username.charAt(0).toUpperCase()}
                      </div>
                      
                      {/* Content preview */}
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '2px' }}>
                          <span style={{ fontSize: '13px', fontWeight: '600', color: '#1f2937' }}>
                            {comment.username}
                          </span>
                          <span style={{ fontSize: '10px', color: '#9ca3af' }}>
                            {new Date(comment.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </span>
                        </div>
                        
                        {isCollapsed ? (
                          <div style={{
                            fontSize: '12px',
                            color: '#6b7280',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}>
                            {comment.commentText.length > 50 
                              ? comment.commentText.substring(0, 50) + '...' 
                              : comment.commentText}
                          </div>
                        ) : null}
                      </div>
                      
                      {/* Collapse/Expand Icon */}
                      <div style={{
                        fontSize: '14px',
                        color: '#9ca3af',
                        transform: isCollapsed ? 'rotate(-90deg)' : 'rotate(0deg)',
                        transition: 'transform 0.2s ease',
                        flexShrink: 0,
                      }}>
                        ▼
                      </div>
                    </div>

                    {/* Expanded Content */}
                    {!isCollapsed && (
                      <div style={{ paddingLeft: '36px' }}>
                        {/* Selected Text Preview */}
                        <div style={{
                          backgroundColor: '#fef3c7',
                          padding: '6px 8px',
                          borderRadius: '4px',
                          fontSize: '11px',
                          marginBottom: '8px',
                          fontStyle: 'italic',
                          borderLeft: '3px solid #fbbf24',
                          color: '#92400e',
                        }}>
                          "{comment.selectedText.length > 60 
                            ? comment.selectedText.substring(0, 60) + '...' 
                            : comment.selectedText}"
                        </div>

                        {/* Comment Text */}
                        <div style={{
                          fontSize: '13px',
                          color: '#374151',
                          marginBottom: '10px',
                          lineHeight: '1.6',
                        }}>
                          {comment.commentText}
                        </div>

                        {/* Action Buttons */}
                        <div style={{
                          display: 'flex',
                          gap: '6px',
                          fontSize: '11px',
                        }}>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleReplyToComment(comment);
                            }}
                            style={{
                              padding: '4px 10px',
                              border: 'none',
                              borderRadius: '4px',
                              backgroundColor: '#f3f4f6',
                              cursor: 'pointer',
                              fontSize: '11px',
                              color: '#374151',
                              fontWeight: '500',
                            }}
                            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#e5e7eb'}
                            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#f3f4f6'}
                          >
                            Reply
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleEditComment(comment);
                            }}
                            style={{
                              padding: '4px 10px',
                              border: 'none',
                              borderRadius: '4px',
                              backgroundColor: '#f3f4f6',
                              cursor: 'pointer',
                              fontSize: '11px',
                              color: '#374151',
                              fontWeight: '500',
                            }}
                            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#e5e7eb'}
                            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#f3f4f6'}
                          >
                            Edit
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeleteComment(comment.id);
                            }}
                            style={{
                              padding: '4px 10px',
                              border: 'none',
                              borderRadius: '4px',
                              backgroundColor: '#fee2e2',
                              cursor: 'pointer',
                              fontSize: '11px',
                              color: '#dc2626',
                              fontWeight: '500',
                            }}
                            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#fecaca'}
                            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#fee2e2'}
                          >
                            Delete
                          </button>
                        </div>

                        {/* Replies */}
                        {comment.replies && comment.replies.length > 0 && (
                          <div style={{
                            marginTop: '12px',
                            paddingTop: '8px',
                            borderTop: '1px solid #f3f4f6',
                          }}>
                            <div style={{ fontSize: '10px', color: '#9ca3af', marginBottom: '8px', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                              {comment.replies.length} {comment.replies.length === 1 ? 'Reply' : 'Replies'}
                            </div>
                            {comment.replies.map((reply) => (
                              <div
                                key={reply.id}
                                style={{
                                  marginTop: '8px',
                                  padding: '8px',
                                  backgroundColor: '#f9fafb',
                                  borderRadius: '6px',
                                  borderLeft: '3px solid #3b82f6',
                                }}
                              >
                                <div style={{
                                  display: 'flex',
                                  alignItems: 'center',
                                  gap: '6px',
                                  marginBottom: '6px',
                                }}>
                                  <div style={{
                                    width: '22px',
                                    height: '22px',
                                    borderRadius: '50%',
                                    backgroundColor: '#3b82f6',
                                    color: 'white',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    fontSize: '10px',
                                    fontWeight: '600',
                                  }}>
                                    {reply.username.charAt(0).toUpperCase()}
                                  </div>
                                  <div style={{ flex: 1 }}>
                                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                      <span style={{ fontSize: '11px', fontWeight: '600', color: '#1f2937' }}>
                                        {reply.username}
                                      </span>
                                      <span style={{ fontSize: '9px', color: '#9ca3af' }}>
                                        {new Date(reply.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                      </span>
                                    </div>
                                  </div>
                                </div>
                                <div style={{ fontSize: '12px', color: '#374151', paddingLeft: '28px', lineHeight: '1.5' }}>
                                  {reply.commentText}
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// Sample document for demo
const SAMPLE_DOC_STATE: DocState = {
  id: 'demo-doc-1',
  title: 'Risk Management Policy',
  version: '1.0',
  blocks: [
    {
      id: 'h1',
      type: 'heading',
      level: 1,
      text: [
        { text: 'Risk Management Policy', bold: true },
      ],
      sectionKey: 'title',
    },
    {
      id: 'overview',
      type: 'heading',
      level: 2,
      text: [{ text: 'Overview' }],
      sectionKey: 'overview',
    },
    {
      id: 'p1',
      type: 'paragraph',
      text: [
        { text: 'This policy establishes the framework for ' },
        { text: 'risk management', bold: true },
        { text: ' within the organization. It applies to all departments and ensures compliance with ' },
        { text: 'OSFI guidelines', italic: true },
        { text: '.' },
      ],
    },
    {
      id: 'p2',
      type: 'paragraph',
      text: [
        { text: 'Key objectives include:', bold: true },
      ],
    },
    {
      id: 'scope',
      type: 'heading',
      level: 2,
      text: [{ text: 'Scope' }],
      sectionKey: 'scope',
    },
    {
      id: 'p3',
      type: 'paragraph',
      text: [
        { text: 'This policy covers ' },
        { text: 'market risk', code: true },
        { text: ', ' },
        { text: 'credit risk', code: true },
        { text: ', and ' },
        { text: 'operational risk', code: true },
        { text: '. All business units must adhere to the established risk appetite and limits.' },
      ],
    },
    {
      id: 'requirements',
      type: 'heading',
      level: 2,
      text: [{ text: 'Policy Requirements' }],
      sectionKey: 'requirements',
    },
    {
      id: 'p4',
      type: 'paragraph',
      text: [
        { text: 'The Chief Risk Officer (CRO) shall:' },
      ],
    },
    {
      id: 'p5',
      type: 'paragraph',
      text: [
        { text: '• Monitor and report risk exposures quarterly', aiSuggestionStatus: 'suggested' },
      ],
    },
    {
      id: 'p6',
      type: 'paragraph',
      text: [
        { text: '• Ensure compliance with regulatory requirements', aiSuggestionStatus: 'applied' },
      ],
    },
    {
      id: 'p7',
      type: 'paragraph',
      text: [
        { text: '• Conduct annual risk assessments', aiSuggestionStatus: 'rejected' },
      ],
    },
  ],
};

// Initial scratchpad document state
const SCRATCHPAD_DOC_STATE: DocState = {
  id: 'scratchpad-doc-1',
  title: 'Scratchpad',
  version: '1.0',
  blocks: [
    {
      id: 's1',
      type: 'heading',
      level: 2,
      text: [{ text: '📝 AI Suggestions' }],
    },
    {
      id: 's2',
      type: 'paragraph',
      text: [{ text: 'AI-generated suggestions will appear here...' }],
    },
    {
      id: 's3',
      type: 'divider',
    },
    {
      id: 's4',
      type: 'heading',
      level: 2,
      text: [{ text: '✏️ My Notes' }],
    },
    {
      id: 's5',
      type: 'paragraph',
      text: [{ text: 'Start taking notes here...' }],
    },
  ],
};

