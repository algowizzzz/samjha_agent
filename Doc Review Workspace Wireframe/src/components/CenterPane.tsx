import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Play, FileText, Upload, CheckCircle2 } from 'lucide-react';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { BlockEditor } from './BlockEditor';
import { getDocument, runIngestion, type ApiDocument, updateDocumentMarkdown, type BlockMetadata, runDocumentAnalysis, createComment, type CreateCommentRequest, improveText, getComments, deleteComment, addReply, type Comment, listAISuggestions, createAISuggestion, updateAISuggestionStatus, deleteAISuggestion, type AISuggestion } from '@/lib/api';
import { MarkdownViewer } from './MarkdownViewer';
import { DiffView } from './DiffView';
import { activityLogger } from '@/utils/activityLogger';
import { isFeatureEnabled } from '@/lib/featureFlags';
import { SingleDocumentEditor } from './singleEditor/SingleDocumentEditor';
import { blockMetadataToDocState, docStateToMarkdown } from '../utils/documentConverters';
import { convertDocStateToBlockMetadata } from './singleEditor/utils/converters';
import type { DocState } from '@/model/docTypes';
import type { SelectionData } from './singleEditor/plugins/SelectionBridgePlugin';
import { debounce } from '../utils/debounce';
import { $getSelection, $isRangeSelection, $isTextNode, $isElementNode, type TextNode, type ElementNode } from 'lexical';
import { $findMatchingParent } from '@lexical/utils';
import { $createDocParagraphNode } from './singleEditor/nodes/DocParagraphNode';
import { $createDocHeadingNode } from './singleEditor/nodes/DocHeadingNode';
import { $createDocListNode, type ListStyle } from './singleEditor/nodes/DocListNode';
import { $createDocListItemNode } from './singleEditor/nodes/DocListItemNode';
import { $createDocCodeNode } from './singleEditor/nodes/DocCodeNode';
import { $createDocQuoteNode } from './singleEditor/nodes/DocQuoteNode';
import { $createAiTextNode } from './singleEditor/nodes/AiTextNode';
import { applyCommentToSelection, applyCommentHighlight, applyCommentHighlightByData, removeCommentHighlight } from './singleEditor/utils/commentHighlightHelpers';
import { applyAISuggestionHighlight, applyAISuggestionHighlightByData, updateAISuggestionStatus as updateAISuggestionHighlightStatus, removeAISuggestionHighlight } from './singleEditor/utils/aiSuggestionHelpers';
import { replaceTextBySuggestionId } from './singleEditor/utils/textReplacementHelpers';
import { getSelectionOffsets } from './singleEditor/utils/selectionOffsets';

interface CenterPaneProps {
  mode: 'editing' | 'original' | 'diff';
  onModeChange: (mode: 'editing' | 'original' | 'diff') => void;
  onTextSelect: (text: string) => void;
  selectedIssueId: string | null;
  onCommentClick: (blockId: string) => void;
  fileId?: string;
  onSelectedBlocksChange?: (selectedBlocks: BlockMetadata[]) => void; // NEW: Pass selected blocks to parent
  aiSuggestions?: Array<{ block_id: string; original: string; suggested: string; reason: string }>; // NEW: AI suggestions from chat
  onSuggestionsListChange?: (suggestions: Array<{ block_id: string; original: string; suggested: string; reason: string; block_content: string }>) => void; // NEW: Pass all suggestions to parent
  selectedSuggestionId?: string | null; // NEW: Highlight block when suggestion clicked
  onBlockWithSuggestionClick?: (blockId: string) => void; // NEW: Notify parent when block with suggestion is clicked
  onAcceptSuggestion?: (blockId: string) => void; // NEW: Accept suggestion from left panel
  onRejectSuggestion?: (blockId: string) => void; // NEW: Reject suggestion from left panel
  onSynthesisReceived?: (synthesis: any) => void; // NEW: Pass synthesis summary to parent
  onAISuggestion?: (original: string, suggested: string) => void; // NEW: Send AI text improvement to RightPane
  onTextSuggestionHandlers?: (accept: () => void, reject: () => void) => void; // NEW: Pass text suggestion handlers up
  onAnalysisStateChange?: (isAnalyzing: boolean, handler: () => Promise<void>) => void; // NEW: Pass analysis state to parent
}

export function CenterPane({ mode, onModeChange, onTextSelect, selectedIssueId, onCommentClick, fileId, onSelectedBlocksChange, aiSuggestions = [], onSuggestionsListChange, selectedSuggestionId, onBlockWithSuggestionClick, onAcceptSuggestion, onRejectSuggestion, onSynthesisReceived, onAISuggestion, onTextSuggestionHandlers, onAnalysisStateChange }: CenterPaneProps) {
  const [doc, setDoc] = useState<ApiDocument | null>(null);
  const [loading, setLoading] = useState(false);
  const pollTimer = useRef<number | null>(null);
  const [runningAnalysis, setRunningAnalysis] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved'>('idle');
  const saveStatusTimer = useRef<number | null>(null);
  // Always use SingleDocumentEditor (new editor without block hover handles)
  const [useSingleEditor] = useState(true);
  
  // SingleDocumentEditor state
  const [editorInstance, setEditorInstance] = useState<any>(null);
  const [currentDocState, setCurrentDocState] = useState<DocState | null>(null);
  const [selectionData, setSelectionData] = useState<SelectionData>({
    selectedText: '',
    blockIds: [],
    selectionScope: 'none',
    currentBlockType: 'paragraph',
    isConvertible: true,
    isEmpty: true,
  });
  
  // Comments state (matching demo exactly)
  const [comments, setComments] = useState<Comment[]>([]);
  const [showCommentModal, setShowCommentModal] = useState(false);
  const [showCommentPanel, setShowCommentPanel] = useState(false);
  const [editingComment, setEditingComment] = useState<Comment | null>(null);
  const [replyingTo, setReplyingTo] = useState<Comment | null>(null);
  const [collapsedComments, setCollapsedComments] = useState<Set<string>>(new Set());
  
  // AI Suggestions state (stored in backend)
  const [storedAiSuggestions, setStoredAiSuggestions] = useState<AISuggestion[]>([]);
  
  // Comment panel drag state
  const [commentPanelDragging, setCommentPanelDragging] = useState(false);
  const [commentPanelWidth, setCommentPanelWidth] = useState(150);
  const commentPanelDragStartX = useRef(0);
  const commentPanelStartWidth = useRef(150);
  
  // Text suggestion state - track last suggestion for Accept/Reject
  const lastTextSuggestionRef = useRef<{ original: string; suggested: string; suggestionId?: string } | null>(null);
  
  // Memoize DocState conversion - recalculate when file changes or document data loads
  const initialDocState = useMemo(() => {
    const blockMetadata = doc?.state?.block_metadata || [];
    console.log('%c[CenterPane v2.0] Creating initialDocState for file:', 'color: #FF9800; font-weight: bold', fileId, 'blocks:', blockMetadata.length);
    const docState = blockMetadataToDocState({ id: fileId || '', title: '', version: '1.0', block_metadata: blockMetadata });
    setCurrentDocState(docState); // Initialize current state
    return docState;
  }, [fileId, doc?.state?.block_metadata]); // Depend on fileId AND block_metadata
  
  // sockets disabled for now to avoid connection issues

  function clearPoll() {
    if (pollTimer.current) {
      window.clearInterval(pollTimer.current);
      pollTimer.current = null;
    }
  }

  async function refreshDocument() {
    if (!fileId) return;
    try {
      const res = await getDocument(fileId);
      // eslint-disable-next-line no-console
      console.debug('[CenterPane] getDocument ->', res);
      setDoc(res.document);
    } catch (e) {
      // ignore for now
    }
  }

  // Expose refresh function to window for external refresh button
  useEffect(() => {
    (window as any).__centerPaneRefreshDocument = refreshDocument;
    return () => {
      delete (window as any).__centerPaneRefreshDocument;
    };
  }, [fileId]);

  useEffect(() => {
    clearPoll();
    setDoc(null);
    if (fileId) {
      refreshDocument();
    }
    return () => clearPoll();
  }, [fileId]);

  // Template loading removed - using document analysis instead

  // Template suggestions removed - using document analysis instead
  
  // Load comments when document loads
  useEffect(() => {
    if (!fileId) return;
    
    async function loadComments() {
      try {
        const result = await getComments(fileId!);
        setComments(result.comments || []);
      } catch (e) {
        console.error('[CenterPane] Failed to load comments:', e);
      }
    }

    loadComments();
  }, [fileId]);

  // Apply all comment highlights when editor and comments are ready
  useEffect(() => {
    if (!editorInstance || comments.length === 0) return;

    console.log('[CenterPane] Applying highlights for', comments.length, 'comments');
    
    // Apply highlight for each comment using its stored data (with precise offsets)
    comments.forEach(comment => {
      if (comment.id && comment.blockId && comment.selectedText) {
        applyCommentHighlightByData(
          editorInstance,
          comment.id,
          comment.blockId,
          comment.selectedText,
          comment.startOffset,
          comment.endOffset
        );
      }
    });
  }, [editorInstance, comments]);

  // Load AI suggestions when document loads
  useEffect(() => {
    if (!fileId) return;
    
    async function loadAISuggestions() {
      try {
        const result = await listAISuggestions(fileId!);
        setStoredAiSuggestions(result.suggestions || []);
      } catch (e) {
        console.error('[CenterPane] Failed to load AI suggestions:', e);
      }
    }

    loadAISuggestions();
  }, [fileId]);

  // Apply all AI suggestion highlights when editor and suggestions are ready
  useEffect(() => {
    if (!editorInstance || storedAiSuggestions.length === 0) return;

    console.log('[CenterPane] Applying highlights for', storedAiSuggestions.length, 'AI suggestions');
    
    // Apply highlight for each AI suggestion using its stored data (with precise offsets)
    storedAiSuggestions.forEach(suggestion => {
      if (suggestion.id && suggestion.block_id && suggestion.selection_text) {
        applyAISuggestionHighlightByData(
          editorInstance,
          suggestion.id,
          suggestion.block_id,
          suggestion.selection_text,
          suggestion.status,
          suggestion.start_offset,
          suggestion.end_offset
        );
      }
    });
  }, [editorInstance, storedAiSuggestions]);

  // Auto-run ingestion if raw_markdown is missing
  useEffect(() => {
    if (!doc || !fileId) return;
    const rawMd = (doc.state as any)?.raw_markdown as string | undefined;
    const status = (doc.status || '').toLowerCase();
    if (!rawMd && status !== 'running' && status !== 'completed') {
      activityLogger.info(`[CenterPane] Auto-running ingestion for ${fileId}`);
      setLoading(true);
      runIngestionHandler();
    }
  }, [doc, fileId]);

  useEffect(() => {
    if (!doc) return;
    const status = (doc.status || '').toLowerCase();
    if (status === 'running') {
      if (!pollTimer.current) {
        pollTimer.current = window.setInterval(() => {
          refreshDocument();
    }, 2000);
      }
    } else {
      clearPoll();
    }
  }, [doc]);

  const title = useMemo(() => {
    if (!doc) return 'No document selected';
    const name = (doc.file_metadata as any)?.name as string | undefined;
    return name || doc.file_id;
  }, [doc]);

  const rawMarkdown = (doc?.state as any)?.raw_markdown as string | undefined;
  const improvedMarkdown = (doc?.state as any)?.improved_markdown as string | undefined;
  const docStatus = (doc?.status || 'idle').toLowerCase();

  // Get display status for badge
  const getDisplayStatus = (): { label: string; variant: 'default' | 'secondary' | 'destructive' | 'outline'; className?: string } => {
    if (!doc) return { label: 'No Document', variant: 'default' };
    
    const state = doc.state as any;
    const status = doc.status?.toLowerCase() || 'pending';
    
    // Check for errors
    if (status === 'error' || status === 'failed') {
      return { label: 'Error', variant: 'destructive' };
    }
    
    // Check if running
    if (status === 'running') {
      return { label: 'Processing', variant: 'default', className: 'bg-amber-100 text-amber-800 border-amber-200' };
    }
    
    // Check which phase is completed
    const hasBlocks = state?.block_metadata && state.block_metadata.length > 0;
    const hasSuggestions = state?.verification_suggestions && state.verification_suggestions.length > 0;
    const hasImprovedMarkdown = !!state?.improved_markdown;
    
    if (hasImprovedMarkdown) {
      return { label: 'Improved', variant: 'default', className: 'bg-emerald-100 text-emerald-800 border-emerald-200' };
    } else if (hasSuggestions && hasSuggestions.length > 0) {
      return { label: 'Reviewed', variant: 'default', className: 'bg-emerald-100 text-emerald-800 border-emerald-200' };
    } else if (hasBlocks) {
      return { label: 'Analyzed', variant: 'default', className: 'bg-emerald-100 text-emerald-800 border-emerald-200' };
    } else if (status === 'ready' || status === 'completed') {
      return { label: 'Ready', variant: 'default', className: 'bg-emerald-100 text-emerald-800 border-emerald-200' };
    } else if (status === 'pending') {
      return { label: 'Uploaded', variant: 'secondary' };
    }
    
    return { label: 'Pending', variant: 'default' };
  };

  const displayStatus = getDisplayStatus();

  const runIngestionHandler = async () => {
    if (!fileId) return;
    console.debug('[CenterPane] Running ingestion', { fileId });
    setLoading(true);
    try {
      await runIngestion(fileId);
      await refreshDocument();
    } catch (e) {
      console.error('[CenterPane] Ingestion error', e);
    } finally {
      setLoading(false);
    }
  };

  const handleRunAnalysis = async () => {
    if (!fileId) return;
    console.log('[CenterPane] Running document analysis');
    setRunningAnalysis(true);
    
    try {
      const result = await runDocumentAnalysis(fileId);
      console.log('[CenterPane] Analysis complete:', result);
      
      // Refresh document to get results
      await refreshDocument();
      
    } catch (e) {
      console.error('[CenterPane] Analysis failed:', e);
      alert(`Analysis failed: ${e}`);
    } finally {
      setRunningAnalysis(false);
    }
  };

  // Notify parent of analysis state and handler
  useEffect(() => {
    if (onAnalysisStateChange) {
      onAnalysisStateChange(runningAnalysis, handleRunAnalysis);
    }
  }, [runningAnalysis, onAnalysisStateChange]);

  // ===== SingleDocumentEditor Handlers =====

  // Document change handler with debounced auto-save
  const handleDocChange = (newDocState: DocState) => {
    console.log('[CenterPane] Document changed:', newDocState);
    setCurrentDocState(newDocState);
    debouncedSave(newDocState);
  };

  // Debounced save function
  const debouncedSave = useMemo(
    () => debounce(async (docState: DocState) => {
      if (!fileId) return;
      try {
        setSaveStatus('saving');
        console.log('[CenterPane] Auto-saving document...');
        const blockMetadata = convertDocStateToBlockMetadata(docState);
        const markdown = docStateToMarkdown(docState);
        
        await updateDocumentMarkdown(
          fileId,
          markdown,
          undefined, // toc_markdown
          blockMetadata,
          [], // acceptedSuggestions - handle separately
          []  // rejectedSuggestions - handle separately
        );
        
        console.log('[CenterPane] ✅ Auto-save successful!');
        activityLogger.changesSaved();
        setSaveStatus('saved');
        
        // Reset to idle after 2 seconds
        if (saveStatusTimer.current) {
          clearTimeout(saveStatusTimer.current);
        }
        saveStatusTimer.current = window.setTimeout(() => {
          setSaveStatus('idle');
        }, 2000);
    } catch (e) {
        console.error('[CenterPane] ❌ Auto-save failed:', e);
        setSaveStatus('idle');
      }
    }, 5000),
    [fileId]
  );

  // Selection change handler - pass to parent for right panel
  const handleSelectionChange = useCallback((data: SelectionData) => {
    console.log('[CenterPane] Selection changed:', data);
    setSelectionData(data);
    
    // Pass selected text to parent for AI context
    if (onTextSelect) {
      onTextSelect(data.selectedText);
    }
    
    // Pass selected blocks to parent
    if (onSelectedBlocksChange && currentDocState) {
      const selectedBlocks = currentDocState.blocks
        .filter(block => data.blockIds.includes(block.id))
        .map(block => convertSingleBlockToMetadata(block));
      onSelectedBlocksChange(selectedBlocks);
    }
  }, [onTextSelect, onSelectedBlocksChange, currentDocState]);

  // Format handler (bold, italic, underline, strikethrough)
  const handleFormat = (format: 'bold' | 'italic' | 'underline' | 'strikethrough') => {
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
        } else if (format === 'strikethrough') {
          selection.formatText('strikethrough');
        }
      }
    });
    
    editorInstance.focus();
  };

  // Text color handler
  const handleTextColor = (color: string) => {
    if (!editorInstance) return;
    
    editorInstance.update(() => {
      const selection = $getSelection();
      if ($isRangeSelection(selection)) {
        selection.getNodes().forEach((node) => {
          if ($isTextNode(node)) {
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

  // Background color handler
  const handleBackgroundColor = (color: string) => {
    if (!editorInstance) return;
    
    editorInstance.update(() => {
      const selection = $getSelection();
      if ($isRangeSelection(selection)) {
        selection.getNodes().forEach((node) => {
          if ($isTextNode(node)) {
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

  // Turn into handler (change block type)
  const handleTurnInto = (type: string) => {
    // Map toolbar types to internal types
    const typeMap: Record<string, { type: 'paragraph' | 'heading' | 'list' | 'code' | 'quote'; options?: { level?: 1 | 2 | 3; listStyle?: ListStyle } }> = {
      'paragraph': { type: 'paragraph' },
      'heading-1': { type: 'heading', options: { level: 1 } },
      'heading-2': { type: 'heading', options: { level: 2 } },
      'heading-3': { type: 'heading', options: { level: 3 } },
      'bulleted-list': { type: 'list', options: { listStyle: 'bullet' } },
      'numbered-list': { type: 'list', options: { listStyle: 'number' } },
      'code': { type: 'code' },
      'quote': { type: 'quote' },
    };
    
    const config = typeMap[type];
    if (!config || !editorInstance) return;
    
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
      if (config.type === 'paragraph') {
        newNode = $createDocParagraphNode(blockId);
      } else if (config.type === 'heading') {
        newNode = $createDocHeadingNode(config.options?.level || 1, blockId);
      } else if (config.type === 'list') {
        // Create list with current text as a single item
        const items = textContent ? [{ content: textContent }] : [{ content: '' }];
        const listNode = $createDocListNode(blockId, config.options?.listStyle || 'bullet', items);
        
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
      } else if (config.type === 'code') {
        const codeNode = $createDocCodeNode(blockId, textContent);
        const textNode = $createAiTextNode(textContent || '');
        codeNode.append(textNode);
        
        // Replace the node
        blockNode.replace(codeNode);
        
        // Set selection to the text node inside code
        textNode.select();
        return;
      } else if (config.type === 'quote') {
        newNode = $createDocQuoteNode(blockId);
      }
      
      if (!newNode) return;
      
      // Transfer children (for paragraph, heading, quote)
      if (isNonEditableBlock) {
        // Non-editable blocks have no text children, create a blank text node
        const textNode = $createAiTextNode('');
        newNode.append(textNode);
      } else if ($isElementNode(blockNode)) {
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

  // Add comment handler - opens modal (matching demo)
  const handleAddComment = () => {
    if (!selectionData.selectedText || selectionData.blockIds.length === 0) {
      alert('Please select text before adding a comment');
      return;
    }
    setShowCommentModal(true);
  };

  // Create comment with backend (matching demo)
  const handleCreateComment = async (commentText: string) => {
    if (!commentText.trim() || !fileId || !editorInstance) return;
    
    try {
      // Calculate precise character offsets
      const offsets = getSelectionOffsets(editorInstance);
      if (!offsets) {
        console.error('[CenterPane] Failed to calculate selection offsets');
        return;
      }
      
      const request: CreateCommentRequest = {
        documentId: fileId,
        blockId: offsets.blockId,
        selectedText: offsets.selectedText,
        startOffset: offsets.startOffset,
        endOffset: offsets.endOffset,
        commentText: commentText.trim(),
        username: 'user', // TODO: Get from auth
      };
      
      console.log('[CenterPane] Creating comment with offsets:', request);
      const newComment = await createComment(request);
      
      // Apply yellow highlight to commented text using precise offsets
      if (editorInstance && newComment.id) {
        applyCommentHighlightByData(
          editorInstance,
          newComment.id,
          request.blockId,
          request.selectedText,
          request.startOffset,
          request.endOffset
        );
      }
      
      // Reload comments
      const result = await getComments(fileId);
      setComments(result.comments || []);
      setShowCommentModal(false);
      setShowCommentPanel(true);
    } catch (error) {
      console.error('[CenterPane] Failed to create comment:', error);
      alert(`Failed to add comment: ${error}`);
    }
  };

  // Edit comment (matching demo)
  const handleEditComment = (comment: Comment) => {
    setEditingComment(comment);
    setShowCommentModal(true);
  };

  // Update comment (matching demo)
  const handleUpdateComment = async (commentId: string, newText: string) => {
    // TODO: Backend API call
    const updatedComments = comments.map(c => 
      c.id === commentId ? { ...c, commentText: newText } : c
    );
    setComments(updatedComments);
    setEditingComment(null);
    setShowCommentModal(false);
  };

  // Delete comment (matching demo)
  const handleDeleteComment = async (commentId: string) => {
    if (!confirm('Delete this comment?') || !fileId) return;
    
    try {
      // Remove yellow highlight from editor
      if (editorInstance) {
        removeCommentHighlight(editorInstance, commentId);
      }
      
      await deleteComment(fileId, commentId);
      const result = await getComments(fileId);
      setComments(result.comments || []);
    } catch (error) {
      console.error('[CenterPane] Failed to delete comment:', error);
      alert('Failed to delete comment');
    }
  };

  // Reply to comment (matching demo)
  const handleReplyToComment = (comment: Comment) => {
    setReplyingTo(comment);
    setShowCommentModal(true);
  };

  // Create reply (matching demo)
  const handleCreateReply = async (parentComment: Comment, replyText: string) => {
    if (!replyText.trim() || !fileId) return;

    try {
      await addReply(fileId, parentComment.id, replyText, 'user');
      const result = await getComments(fileId);
      setComments(result.comments || []);
      setReplyingTo(null);
      setShowCommentModal(false);
    } catch (error) {
      console.error('[CenterPane] Failed to add reply:', error);
      alert('Failed to add reply');
    }
  };

  // Toggle collapse (matching demo)
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

  // Comment panel drag handlers
  const handleCommentPanelDragStart = (e: React.MouseEvent) => {
    e.preventDefault();
    setCommentPanelDragging(true);
    commentPanelDragStartX.current = e.clientX;
    commentPanelStartWidth.current = commentPanelWidth;
  };

  useEffect(() => {
    if (!commentPanelDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      const deltaX = commentPanelDragStartX.current - e.clientX; // Negative delta = dragging right
      const newWidth = Math.max(120, Math.min(400, commentPanelStartWidth.current + deltaX));
      setCommentPanelWidth(newWidth);
    };

    const handleMouseUp = () => {
      setCommentPanelDragging(false);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [commentPanelDragging]);

  // Improve text handler (AI suggestions) - uses current selection from selectionData
  const handleImproveText = async () => {
    console.log('[CenterPane] Improve text - current selection:', selectionData.selectedText);
    
    if (!selectionData.selectedText || !selectionData.selectedText.trim() || !fileId || !editorInstance) {
      alert('Please select text to improve');
      return;
    }
    
    if (selectionData.blockIds.length === 0) {
      alert('Please select text within a block');
      return;
    }
    
    try {
      // Calculate precise character offsets
      const offsets = getSelectionOffsets(editorInstance);
      if (!offsets) {
        console.error('[CenterPane] Failed to calculate selection offsets');
        return;
      }
      
      const result = await improveText(offsets.selectedText);
      console.log('[CenterPane] ✅ Text improved:', result);
      
      if (!result.success) {
        alert(`Failed to improve text: ${result}`);
        return;
      }
      
      // Save AI suggestion to backend with precise offsets
      const newSuggestion = await createAISuggestion(fileId, {
        block_id: offsets.blockId,
        selection_text: result.original,
        improved_text: result.improved,
        status: 'pending',
        start_offset: offsets.startOffset,
        end_offset: offsets.endOffset
      });
      
      // Apply blue highlight to the CURRENTLY SELECTED text (not searching for it)
      if (editorInstance && newSuggestion.id) {
        applyAISuggestionHighlight(editorInstance, newSuggestion.id);
      }
      
      // Reload AI suggestions
      const suggestionsResult = await listAISuggestions(fileId);
      setStoredAiSuggestions(suggestionsResult.suggestions || []);
      
      // Store the suggestion for Accept/Reject handlers
      lastTextSuggestionRef.current = {
        original: result.original,
        suggested: result.improved,
        suggestionId: newSuggestion.id
      };
      
      // Send suggestion to RightPane chat
      if (onAISuggestion) {
        onAISuggestion(result.original, result.improved);
      }
    } catch (error) {
      console.error('[CenterPane] ❌ Failed to improve text:', error);
      alert(`Failed to improve text: ${error}`);
    }
  };

  // Accept text suggestion - replace selected text with improved version
  const handleAcceptTextSuggestion = useCallback(async () => {
    if (!editorInstance || !lastTextSuggestionRef.current || !fileId) {
      console.error('[CenterPane] Cannot accept: no editor or suggestion');
      return;
    }

    const { original, suggested, suggestionId } = lastTextSuggestionRef.current;
    console.log('[CenterPane] Accepting text suggestion:', { original, suggested, suggestionId });

    // Update AI suggestion status in backend
    if (suggestionId) {
      try {
        // First, replace the text in the editor with the improved version
        replaceTextBySuggestionId(editorInstance, suggestionId, suggested);
        
        // Then update the backend status
        await updateAISuggestionStatus(fileId, suggestionId, 'accepted');
        
        // Update the suggestion data in backend to reflect the new text
        // (The suggestion now points to the improved text, not the original)
        const result = await listAISuggestions(fileId);
        setStoredAiSuggestions(result.suggestions || []);
        
        console.log('[CenterPane] ✅ Text replaced and status updated to accepted');
      } catch (error) {
        console.error('[CenterPane] Failed to accept AI suggestion:', error);
      }
    }

    editorInstance.focus();
    lastTextSuggestionRef.current = null;
  }, [editorInstance, fileId]);

  // Reject text suggestion - update status in backend and remove highlight
  const handleRejectTextSuggestion = useCallback(async () => {
    if (!fileId || !lastTextSuggestionRef.current) {
      console.log('[CenterPane] Rejecting text suggestion (no backend update)');
      lastTextSuggestionRef.current = null;
      return;
    }

    const { suggestionId } = lastTextSuggestionRef.current;
    console.log('[CenterPane] Rejecting text suggestion:', suggestionId);

    // Update AI suggestion status in backend
    if (suggestionId) {
      try {
        await updateAISuggestionStatus(fileId, suggestionId, 'rejected');
        
        // Update highlight color to show rejected status (or remove it)
        if (editorInstance) {
          updateAISuggestionHighlightStatus(editorInstance, suggestionId, 'rejected');
        }
        
        // Reload AI suggestions
        const result = await listAISuggestions(fileId);
        setStoredAiSuggestions(result.suggestions || []);
      } catch (error) {
        console.error('[CenterPane] Failed to update AI suggestion status:', error);
      }
    }

    lastTextSuggestionRef.current = null;
  }, [editorInstance, fileId]);

  // Pass text suggestion handlers to parent on mount
  useEffect(() => {
    if (onTextSuggestionHandlers) {
      onTextSuggestionHandlers(handleAcceptTextSuggestion, handleRejectTextSuggestion);
    }
  }, [onTextSuggestionHandlers, handleAcceptTextSuggestion, handleRejectTextSuggestion]);

  // Comment click handler - receives array of comment IDs
  const handleCommentTextClick = (commentIds: string[]) => {
    console.log('[CenterPane] Comments clicked:', commentIds);
    
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
        const element = document.getElementById(`comment-${commentIds[0]}`);
        if (element) {
          element.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          // Flash highlight
          element.style.animation = 'none';
          setTimeout(() => {
            element.style.animation = 'flash 1s ease';
          }, 10);
        }
      }, 100);
    }
  };

  // ===== Helper Functions =====

  // Convert BlockMetadata[] to markdown string
  function blocksToMarkdown(blocks: BlockMetadata[]): string {
    return blocks.map(block => {
      if (block.type === 'heading') {
        const level = block.level || 1;
        return `${'#'.repeat(level)} ${block.content}`;
      }
      return block.content;
    }).join('\n\n');
  }

  // Convert single DocBlock to BlockMetadata
  function convertSingleBlockToMetadata(block: any): BlockMetadata {
    return {
      id: block.id,
      type: block.type,
      content: block.text?.map((t: any) => t.text).join('') || block.content?.map((t: any) => t.text).join('') || '',
      level: block.level,
      page: block.meta?.page || 1,
      block_num: block.meta?.block_num || 0,
      start_line: block.meta?.start_line || 0,
      end_line: block.meta?.end_line || 0,
    };
  }


  return (
    <>
      <style>{`
        @keyframes flash {
          0%, 100% { backgroundColor: white; }
          50% { backgroundColor: #fef3c7; }
        }
      `}</style>
    <div className="flex flex-col h-full">
      {/* Top Bar */}
      <div className="border-b border-neutral-200 px-6 py-3">
        <div className="flex items-start justify-between mb-2">
          <div className="flex items-center gap-3">
            <div>
              <h1 className="text-neutral-900 mb-1">{title}</h1>
              <p className="text-neutral-500 text-xs">{doc?.file_id || ''}</p>
            </div>
            {doc && (
              <Badge variant={displayStatus.variant} className={`mt-1 ${displayStatus.className || ''}`}>
                {displayStatus.label}
              </Badge>
            )}
          </div>
          <div className="flex gap-2 items-center relative">
            {/* View Comments Button */}
              <Button 
                variant="outline" 
                size="sm"
              onClick={() => setShowCommentPanel(!showCommentPanel)}
              className="min-w-[120px]"
            >
              {showCommentPanel ? '✕ Hide' : '👁️ View'} Comments ({comments.length})
              </Button>
              
            {/* Save Button */}
            <Button 
              variant="default" 
              size="sm"
              onClick={async () => {
                if (currentDocState && fileId) {
                  setSaveStatus('saving');
                  try {
                    const blockMetadata = convertDocStateToBlockMetadata(currentDocState);
                    const markdown = docStateToMarkdown(currentDocState);
                    
                    await updateDocumentMarkdown(
                      fileId,
                      markdown,
                      undefined,
                      blockMetadata,
                      [],
                      []
                    );
                    
                    setSaveStatus('saved');
                    
                    // Reset to idle after 2 seconds
                    if (saveStatusTimer.current) {
                      clearTimeout(saveStatusTimer.current);
                    }
                    saveStatusTimer.current = window.setTimeout(() => {
                      setSaveStatus('idle');
                    }, 2000);
                  } catch (e) {
                    console.error('[CenterPane] Manual save failed:', e);
                    setSaveStatus('idle');
                  }
                }
              }}
              disabled={!fileId || saveStatus === 'saving'}
              className={`${
                saveStatus === 'saved' 
                  ? '!bg-green-600 hover:!bg-green-700' 
                  : '!bg-blue-600 hover:!bg-blue-700'
              } text-white min-w-[100px]`}
            >
              {saveStatus === 'saving' ? (
                <>
                  <div className="w-3 h-3 mr-2 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  Saving...
                </>
              ) : saveStatus === 'saved' ? (
                <>
                  <CheckCircle2 className="w-3 h-3 mr-2" />
                  Saved
                </>
              ) : (
                <>
                  💾
                  Save
                </>
              )}
            </Button>
          </div>
        </div>

        {/* Phase badges removed - not needed */}
      </div>

      {/* Mode Toggle & Track Changes */}
      <div className="border-b border-neutral-200 px-6 py-2 flex items-center justify-between">
        <div className="inline-flex bg-neutral-100 rounded p-1">
          <button
            onClick={() => onModeChange('editing')}
                onMouseDown={() => console.debug('[UI] Switch mode -> editing')}
            className={`px-4 py-1.5 rounded transition-colors text-sm ${
              mode === 'editing' 
                ? 'bg-white text-neutral-900 shadow-sm' 
                : 'text-neutral-600 hover:text-neutral-900'
            }`}
          >
            Editing
          </button>
          <button
            onClick={() => onModeChange('original')}
            className={`px-4 py-1.5 rounded transition-colors text-sm ${
              mode === 'original' 
                ? 'bg-white text-neutral-900 shadow-sm' 
                : 'text-neutral-600 hover:text-neutral-900'
            }`}
          >
            Original
          </button>
          <button
            onClick={() => onModeChange('diff')}
                onMouseDown={() => console.debug('[UI] Switch mode -> diff')}
            className={`px-4 py-1.5 rounded transition-colors text-sm ${
              mode === 'diff' 
                ? 'bg-white text-neutral-900 shadow-sm' 
                : 'text-neutral-600 hover:text-neutral-900'
            }`}
          >
            Diff
          </button>
        </div>

        {/* Track Changes toggle removed per user request */}
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-hidden">
        {loading || docStatus === 'running' ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-neutral-900 mx-auto mb-4"></div>
              <p className="text-neutral-700 font-medium">Running Phase 0 Ingestion...</p>
              <p className="text-neutral-500 text-sm mt-2">Converting document to markdown and extracting structure</p>
            </div>
          </div>
        ) : mode === 'editing' ? (
          <div className="flex h-full">
            <div className="flex-1 overflow-y-auto">
              {(initialDocState && initialDocState.blocks.length > 0) || improvedMarkdown || rawMarkdown ? (
                useSingleEditor ? (
                  // NEW: Single Document Editor (Option 3) - Clean Notion-like experience with comment panel
                  <div style={{ display: 'flex', height: '100%', position: 'relative' }}>
                    <div style={{ flex: 1, overflow: 'auto', backgroundColor: 'white' }}>
                    <SingleDocumentEditor
                      key={fileId}
                      initialDoc={initialDocState}
                        onDocChange={handleDocChange}
                      readOnly={false}
                        onEditorReady={setEditorInstance}
                        onSelectionChange={handleSelectionChange}
                        onFormat={handleFormat}
                        onTextColor={handleTextColor}
                        onBackgroundColor={handleBackgroundColor}
                        onTurnInto={handleTurnInto}
                        onAddComment={handleAddComment}
                        onImproveText={handleImproveText}
                        onCommentClick={handleCommentTextClick}
                      />
                    </div>
                    
                    {/* Comment Panel - Right Margin (matching demo exactly) */}
                    {showCommentPanel && (
                      <div 
                        style={{
                          width: `${commentPanelWidth}px`,
                          borderLeft: '1px solid #e5e5e7',
                          backgroundColor: '#f9fafb',
                          display: 'flex',
                          flexDirection: 'column',
                          overflow: 'hidden',
                          position: 'relative',
                        }}
                      >
                        {/* Drag Handle */}
                        <div
                          onMouseDown={handleCommentPanelDragStart}
                          style={{
                            position: 'absolute',
                            left: 0,
                            top: 0,
                            bottom: 0,
                            width: '4px',
                            cursor: 'ew-resize',
                            backgroundColor: commentPanelDragging ? '#3b82f6' : 'transparent',
                            transition: 'background-color 0.2s',
                            zIndex: 10,
                          }}
                          onMouseEnter={(e) => {
                            if (!commentPanelDragging) {
                              e.currentTarget.style.backgroundColor = '#e5e5e7';
                            }
                          }}
                          onMouseLeave={(e) => {
                            if (!commentPanelDragging) {
                              e.currentTarget.style.backgroundColor = 'transparent';
                            }
                          }}
                        />
                        {/* Comment Panel Header */}
                        <div style={{
                          padding: '6px 8px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                          borderBottom: '1px solid #e5e5e7',
                          backgroundColor: 'white',
                        }}>
                          <span style={{ fontSize: '10px', fontWeight: 600, color: '#666' }}>
                            Comments ({comments.length})
                          </span>
                          <button
                            onClick={() => setShowCommentPanel(false)}
                            style={{
                              background: 'transparent',
                              border: 'none',
                              cursor: 'pointer',
                              padding: '2px',
                              fontSize: '12px',
                              color: '#999',
                            }}
                            title="Hide comments"
                          >
                            ✕
                          </button>
                        </div>

                        {/* Comments List */}
                        <div style={{ flex: 1, overflowY: 'auto', padding: '8px' }}>
                          {comments.length === 0 ? (
                            <div style={{
                              textAlign: 'center',
                              color: '#9ca3af',
                              padding: '24px 8px',
                              fontSize: '10px',
                              lineHeight: '1.6',
                            }}>
                              <div style={{ fontSize: '20px', marginBottom: '4px' }}>💭</div>
                              No comments yet<br/>
                              <span style={{ fontSize: '9px' }}>Select text and add a comment</span>
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
                                        {(comment.author || 'U').charAt(0).toUpperCase()}
                                      </div>
                                      <div style={{ flex: 1, minWidth: 0, fontSize: '8px', fontWeight: '600', color: '#1f2937' }}>
                                        {comment.author || 'Unknown'}
                                      </div>
                                      <span style={{ fontSize: '7px', color: '#9ca3af' }}>
                                        {new Date(comment.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                      </span>
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
                                            handleReplyToComment(comment);
                                          }}
                                          style={{
                                            padding: '2px 4px',
                                            border: 'none',
                                            borderRadius: '3px',
                                            backgroundColor: '#dbeafe',
                                            cursor: 'pointer',
                                            fontSize: '7px',
                                            color: '#1e40af',
                                          }}
                                        >
                                          Reply
                                        </button>
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
                                          Delete
                                        </button>
                                      </div>
                                    )}

                                    {/* Replies (when expanded) */}
                                    {!isCollapsed && comment.replies && comment.replies.length > 0 && (
                                      <div style={{
                                        marginTop: '6px',
                                        paddingTop: '4px',
                                        borderTop: '1px solid #f3f4f6',
                                      }}>
                                        <div style={{ fontSize: '7px', color: '#9ca3af', marginBottom: '4px', fontWeight: '600' }}>
                                          {comment.replies.length} {comment.replies.length === 1 ? 'Reply' : 'Replies'}
                                        </div>
                                        {comment.replies.map((reply) => (
                                          <div
                                            key={reply.id}
                                            style={{
                                              marginTop: '4px',
                                              padding: '4px',
                                              backgroundColor: '#f9fafb',
                                              borderRadius: '4px',
                                              borderLeft: '2px solid #3b82f6',
                                            }}
                                          >
                                            <div style={{
                                              display: 'flex',
                                              alignItems: 'center',
                                              gap: '3px',
                                              marginBottom: '3px',
                                            }}>
                                              <div style={{
                                                width: '12px',
                                                height: '12px',
                                                borderRadius: '50%',
                                                backgroundColor: '#3b82f6',
                                                color: 'white',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center',
                                                fontSize: '7px',
                                                fontWeight: '600',
                                              }}>
                                                {(reply.author || 'U').charAt(0).toUpperCase()}
                                              </div>
                                              <span style={{ fontSize: '7px', fontWeight: '600', color: '#1f2937' }}>
                                                {reply.author || 'Unknown'}
                                              </span>
                                              <span style={{ fontSize: '6px', color: '#9ca3af' }}>
                                                {new Date(reply.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                              </span>
                                            </div>
                                            <div style={{ fontSize: '8px', color: '#374151', paddingLeft: '15px', lineHeight: '1.4' }}>
                                              {reply.commentText}
                                            </div>
                                          </div>
                                        ))}
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
                ) : (
                  // LEGACY: Per-block editor with header
                  <>
                    <div className="bg-neutral-100 px-6 py-2 sticky top-0 border-b border-neutral-200">
                      <span className="text-neutral-700 text-sm font-medium">
                        {improvedMarkdown ? 'Improved Markdown' : rawMarkdown ? 'Original Markdown' : 'No content yet'}
                      </span>
                    </div>
                    <div className="px-10 py-6">
                    // LEGACY: Per-block editor
                    <BlockEditor 
                        trackChangesEnabled={false}
                        onCommentClick={onCommentClick}
                        selectedIssueId={selectedIssueId}
                        initialMarkdown={improvedMarkdown || rawMarkdown || ''}
                        blockMetadata={doc?.state?.block_metadata}
                        verificationSuggestions={doc?.state?.verification_suggestions}
                        fileId={fileId}
                        onSelectedBlocksChange={onSelectedBlocksChange}
                        aiSuggestions={aiSuggestions}
                        onSuggestionsListChange={onSuggestionsListChange}
                        selectedSuggestionId={selectedSuggestionId}
                        onBlockWithSuggestionClick={onBlockWithSuggestionClick}
                        onAcceptSuggestion={onAcceptSuggestion}
                        onRejectSuggestion={onRejectSuggestion}
                        onSave={async (data: { 
                          markdown: string; 
                          blockMetadata: any[]; 
                          acceptedSuggestions: string[]; 
                          rejectedSuggestions: string[] 
                        }) => {
                          activityLogger.info('Saving document...');
                          console.log('[CenterPane] onSave called with data:', {
                            markdownLength: data.markdown.length,
                            blockMetadataCount: data.blockMetadata.length,
                            acceptedCount: data.acceptedSuggestions.length,
                            rejectedCount: data.rejectedSuggestions.length
                          });
                          
                          try {
                            const fileId = doc?.file_id;
                            if (!fileId) throw new Error('No file ID available');
                            
                            // FIX: Pass parameters in correct order (markdown, toc_markdown, block_metadata, ...)
                            await updateDocumentMarkdown(
                              fileId,
                              data.markdown,
                              undefined, // toc_markdown
                              data.blockMetadata,
                              data.acceptedSuggestions,
                              data.rejectedSuggestions
                            );
                            console.log('[CenterPane] ✅ Save successful! Current editor state persisted.');
                            activityLogger.changesSaved();
                            
                            // Update local state with new accepted/rejected counts without reloading editor
                            if (doc && doc.state) {
                              setDoc({
                                ...doc,
                                state: {
                                  ...doc.state,
                                  accepted_suggestions: data.acceptedSuggestions,
                                  rejected_suggestions: data.rejectedSuggestions,
                                  block_metadata: data.blockMetadata,
                                  improved_markdown: data.markdown
                                }
                              });
                            }
                          } catch (e) {
                            console.error('[CenterPane] ❌ Save failed:', e);
                          alert(`Failed to save: ${e}`);
                        }
                      }}
                    />
                    </div>
                  </>
                )
              ) : (
                <div className="px-10 py-6">
                  <div className="text-sm text-neutral-600">
                    Run Phase 1 to ingest and normalize the document, then content will appear here.
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : mode === 'original' ? (
          <div className="flex-1 overflow-y-auto">
            <div className="bg-neutral-100 px-6 py-2 sticky top-0 border-b border-neutral-200">
              <span className="text-neutral-700 text-sm font-medium">Original Document</span>
            </div>
            <div className="px-10 py-6">
              {doc?.state?.original_markdown || doc?.state?.raw_markdown ? (
                <MarkdownViewer 
                  content={doc?.state?.original_markdown || doc?.state?.raw_markdown || ''} 
                  title={`${title} (original)`} 
                  onCommentClick={onCommentClick} 
                />
              ) : (
                <div className="text-sm text-neutral-600">No original markdown available.</div>
              )}
            </div>
          </div>
        ) : (
          // Diff mode - Show line-by-line comparison
          (() => {
            // Prepare blocks data for DiffView
            const blockMetadata = doc?.state?.block_metadata as BlockMetadata[] | undefined;
            const verificationSuggestions = doc?.state?.verification_suggestions || [];
            const templateImprovements = doc?.state?.template_improvements || [];
            
            // Convert metadata to blocks with change history
            const blocks = (blockMetadata || []).map((meta) => {
              const changeHistory: Array<{
                timestamp: string;
                type: string;
                original: string;
                modified: string;
                reason?: string;
                user?: string;
              }> = [];
              
              // Check for verification suggestions
              const verifySuggestion = verificationSuggestions.find((s: any) => s.block_id === meta.id);
              if (verifySuggestion) {
                changeHistory.push({
                  timestamp: new Date().toISOString(),
                  type: 'verified',
                  original: verifySuggestion.original,
                  modified: verifySuggestion.suggested,
                  reason: verifySuggestion.reason,
                  user: 'system'
                });
              }
              
              // Check for template improvements
              const templateImprovement = templateImprovements.find((imp: any) => imp.block_id === meta.id);
              if (templateImprovement) {
                changeHistory.push({
                  timestamp: new Date().toISOString(),
                  type: 'ai_suggested',
                  original: templateImprovement.original,
                  modified: templateImprovement.improved,
                  reason: `${templateImprovement.reasoning}\n\nChanges: ${templateImprovement.changes_made?.join(', ') || 'N/A'}`,
                  user: 'riskgpt'
                });
              }
              
              return {
                id: meta.id,
                type: meta.type || 'paragraph',
                content: meta.content,
                changeHistory
              };
            });
            
            return blocks.length > 0 ? (
              <DiffView blocks={blocks} blockMetadata={blockMetadata} />
            ) : (
              <div className="flex items-center justify-center h-full">
                <div className="text-center text-neutral-500">
                  <p className="text-sm">No content available for diff comparison.</p>
                  <p className="text-xs mt-2">Please ensure the document has been processed.</p>
                </div>
              </div>
            );
          })()
        )}
      </div>
    </div>

    {/* Comment Modal (matching demo exactly) */}
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
    </>
  );
}