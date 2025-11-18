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
    mode: 'none',
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

  const handleAcceptSuggestion = () => {
    if (!editorInstance || !pendingSuggestion) return;
    insertAiSuggestion(editorInstance, pendingSuggestion.improved, 'suggested');
    setPendingSuggestion(null);
  };

  const handleRejectSuggestion = () => {
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
                 nodeType === 'doc-quote';
        }
      );
      
      if (!blockNode) return;
      
      // Get the block ID to preserve it
      const blockId = (blockNode as any).getBlockId?.() || `b${Date.now()}`;
      
      // Get current text content
      const textContent = blockNode.getTextContent();
      
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
      const children = blockNode.getChildren();
      children.forEach(child => {
        newNode.append(child);
      });
      
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
    if (!commentText.trim()) return;

    const newComment: Comment = {
      id: `comment-${Date.now()}`,
      documentId: docState.id,
      blockId: selectionData.blockIds[0] || 'unknown',
      selectedText: selectionData.selectedText,
      startOffset: 0, // TODO: Calculate actual offset
      endOffset: selectionData.selectedText.length,
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
    // TODO: Highlight text in editor and scroll to it
    console.log('Clicked comment:', comment);
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
        <button
          onClick={() => setShowControls(!showControls)}
          className="px-4 py-2 text-sm border border-neutral-300 rounded hover:bg-neutral-50 transition-colors"
        >
          {showControls ? 'Hide' : 'Show'} Controls
        </button>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Main: Editor (full width when controls hidden) */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-y-auto bg-white">
            <SingleDocumentEditor
              initialDoc={docState}
              onDocChange={handleDocChange}
              readOnly={false}
              onEditorReady={setEditorInstance}
              onSelectionChange={setSelectionData}
            />
          </div>
        </div>

        {/* Right: Controls & Info (collapsible) */}
        {showControls && (
          <div className="w-96 flex flex-col overflow-hidden bg-white border-l border-neutral-200">
          <div className="bg-neutral-100 px-4 py-2 border-b border-neutral-200">
            <span className="text-sm font-medium text-neutral-700">Controls & Debug</span>
          </div>
          
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
                    onClick={handleAcceptSuggestion}
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
                    onClick={handleRejectSuggestion}
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
                  <span className="font-medium">Mode:</span>
                  <span className={`ml-1 px-1 py-0.5 rounded ${
                    selectionData.mode === 'blocks' ? 'bg-blue-100 text-blue-700' :
                    selectionData.mode === 'text' ? 'bg-green-100 text-green-700' :
                    'bg-neutral-200 text-neutral-600'
                  }`}>
                    {selectionData.mode}
                  </span>
                </div>
                
                {selectionData.selectedText && (
                  <div className="bg-neutral-50 p-1 rounded">
                    <span className="font-medium">Text ({selectionData.selectedText.length}):</span> "{selectionData.selectedText.substring(0, 50)}{selectionData.selectedText.length > 50 ? '...' : ''}"
                  </div>
                )}
              </div>
            </div>

            {/* Turn Into */}
            <div className="border border-neutral-200 rounded-lg p-2" style={{ opacity: selectionData.isConvertible === false ? 0.5 : 1 }}>
              <h3 className="text-xs font-semibold mb-1">Turn Into</h3>
              {selectionData.isConvertible === false && (
                <p className="text-xs text-gray-500 mb-2">Not available for {selectionData.currentBlockType} blocks</p>
              )}
              <div className="space-y-1">
                <button
                  onClick={() => handleTurnInto('paragraph')}
                  disabled={selectionData.isConvertible === false}
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
                  disabled={selectionData.isConvertible === false}
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
                  disabled={selectionData.isConvertible === false}
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
                  disabled={selectionData.isConvertible === false}
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
                  disabled={selectionData.isConvertible === false}
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
                  disabled={selectionData.isConvertible === false}
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
                  disabled={selectionData.isConvertible === false}
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
                  disabled={selectionData.isConvertible === false}
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
                    {['#FFFFFF', '#F3F4F6', '#FEE2E2', '#FFEDD5', '#FEF3C7', '#FEF9C3', '#ECFCCB', '#D1FAE5', '#CFFAFE', '#DBEAFE', '#DBEAFE', '#E0E7FF', '#EDE9FE', '#F3E8FF', '#FAE8FF'].map((color) => (
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
        </div>
        )}
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

      {/* Comment Panel */}
      {showCommentPanel && (
        <div style={{
          position: 'fixed',
          right: showControls ? '320px' : '0',
          top: '73px',
          bottom: 0,
          width: '350px',
          backgroundColor: 'white',
          borderLeft: '1px solid #e5e5e7',
          overflowY: 'auto',
          zIndex: 100,
          boxShadow: '-2px 0 8px rgba(0, 0, 0, 0.05)',
        }}>
          <div style={{
            padding: '16px',
            borderBottom: '1px solid #e5e5e7',
            position: 'sticky',
            top: 0,
            backgroundColor: 'white',
            zIndex: 1,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <h3 style={{ fontSize: '16px', fontWeight: '600' }}>
                Comments ({comments.length})
              </h3>
              <button
                onClick={() => setShowCommentPanel(false)}
                style={{
                  padding: '4px 8px',
                  border: 'none',
                  background: 'none',
                  cursor: 'pointer',
                  fontSize: '18px',
                }}
              >
                ✕
              </button>
            </div>
          </div>

          <div style={{ padding: '16px' }}>
            {comments.length === 0 ? (
              <div style={{
                textAlign: 'center',
                color: '#9ca3af',
                padding: '32px 16px',
                fontSize: '14px',
              }}>
                No comments yet. Select text and click "Add Comment" to start.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                {comments.map((comment) => (
                  <div
                    key={comment.id}
                    style={{
                      border: '1px solid #e5e5e7',
                      borderRadius: '8px',
                      padding: '12px',
                      backgroundColor: '#fafafa',
                      cursor: 'pointer',
                    }}
                    onClick={() => handleClickComment(comment)}
                  >
                    {/* Comment Header */}
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      marginBottom: '8px',
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div style={{
                          width: '24px',
                          height: '24px',
                          borderRadius: '50%',
                          backgroundColor: '#f59e0b',
                          color: 'white',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '12px',
                          fontWeight: '600',
                        }}>
                          {comment.username.charAt(0).toUpperCase()}
                        </div>
                        <div>
                          <div style={{ fontSize: '12px', fontWeight: '600' }}>
                            {comment.username}
                          </div>
                          <div style={{ fontSize: '10px', color: '#6b7280' }}>
                            {new Date(comment.timestamp).toLocaleString()}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Selected Text */}
                    <div style={{
                      backgroundColor: '#fef9c3',
                      padding: '6px 8px',
                      borderRadius: '4px',
                      fontSize: '12px',
                      marginBottom: '8px',
                      fontStyle: 'italic',
                    }}>
                      "{comment.selectedText.length > 60 
                        ? comment.selectedText.substring(0, 60) + '...' 
                        : comment.selectedText}"
                    </div>

                    {/* Comment Text */}
                    <div style={{
                      fontSize: '14px',
                      color: '#374151',
                      marginBottom: '8px',
                      lineHeight: '1.5',
                    }}>
                      {comment.commentText}
                    </div>

                    {/* Action Buttons */}
                    <div style={{
                      display: 'flex',
                      gap: '8px',
                      fontSize: '12px',
                    }}>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleReplyToComment(comment);
                        }}
                        style={{
                          padding: '4px 8px',
                          border: '1px solid #d1d5db',
                          borderRadius: '4px',
                          backgroundColor: 'white',
                          cursor: 'pointer',
                          fontSize: '11px',
                        }}
                      >
                        ↩️ Reply
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleEditComment(comment);
                        }}
                        style={{
                          padding: '4px 8px',
                          border: '1px solid #d1d5db',
                          borderRadius: '4px',
                          backgroundColor: 'white',
                          cursor: 'pointer',
                          fontSize: '11px',
                        }}
                      >
                        ✏️ Edit
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteComment(comment.id);
                        }}
                        style={{
                          padding: '4px 8px',
                          border: '1px solid #ef4444',
                          borderRadius: '4px',
                          backgroundColor: 'white',
                          color: '#ef4444',
                          cursor: 'pointer',
                          fontSize: '11px',
                        }}
                      >
                        🗑️ Delete
                      </button>
                    </div>

                    {/* Replies */}
                    {comment.replies && comment.replies.length > 0 && (
                      <div style={{
                        marginTop: '12px',
                        paddingLeft: '16px',
                        borderLeft: '2px solid #e5e5e7',
                      }}>
                        {comment.replies.map((reply) => (
                          <div
                            key={reply.id}
                            style={{
                              marginTop: '8px',
                              padding: '8px',
                              backgroundColor: 'white',
                              borderRadius: '6px',
                              border: '1px solid #e5e5e7',
                            }}
                          >
                            <div style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '6px',
                              marginBottom: '6px',
                            }}>
                              <div style={{
                                width: '20px',
                                height: '20px',
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
                              <div>
                                <div style={{ fontSize: '11px', fontWeight: '600' }}>
                                  {reply.username}
                                </div>
                                <div style={{ fontSize: '9px', color: '#6b7280' }}>
                                  {new Date(reply.timestamp).toLocaleString()}
                                </div>
                              </div>
                            </div>
                            <div style={{ fontSize: '12px', color: '#374151' }}>
                              {reply.commentText}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
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

