import { useState, useEffect, useCallback } from 'react';
import { 
  Comment, 
  fetchComments, 
  addComment as apiAddComment,
  addReply as apiAddReply,
  resolveComment as apiResolveComment,
  deleteComment as apiDeleteComment,
  updateComment as apiUpdateComment,
  fetchCommentCounts
} from '@/lib/comments-api';

export function useComments(fileId: string | null, blockId?: string) {
  const [comments, setComments] = useState<Comment[]>([]);
  const [commentCounts, setCommentCounts] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadComments = useCallback(async () => {
    if (!fileId) {
      setComments([]);
      setError(null);
      return;
    }
    
    setLoading(true);
    setError(null);
    try {
      const data = await fetchComments(fileId, blockId);
      setComments(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load comments';
      setError(errorMessage);
      // Only log non-404 errors to avoid noise
      if (!errorMessage.includes('NOT FOUND')) {
        console.error('Error loading comments:', err);
      }
    } finally {
      setLoading(false);
    }
  }, [fileId, blockId]);

  const loadCommentCounts = useCallback(async () => {
    if (!fileId) return;
    
    try {
      const counts = await fetchCommentCounts(fileId);
      setCommentCounts(counts);
    } catch (err) {
      console.error('Error loading comment counts:', err);
    }
  }, [fileId]);

  useEffect(() => {
    loadComments();
    loadCommentCounts();
  }, [loadComments, loadCommentCounts]);

  const addComment = useCallback(async (
    blockId: string,
    blockTitle: string,
    content: string,
    selectionText?: string
  ) => {
    if (!fileId) return;
    
    try {
      const newComment = await apiAddComment(fileId, blockId, blockTitle, content, selectionText);
      setComments(prev => [...prev, newComment]);
      loadCommentCounts(); // Refresh counts
      return newComment;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add comment');
      throw err;
    }
  }, [fileId, loadCommentCounts]);

  const addReply = useCallback(async (commentId: string, content: string) => {
    if (!fileId) return;
    
    try {
      const updatedComment = await apiAddReply(fileId, commentId, content);
      setComments(prev => prev.map(c => c.id === commentId ? updatedComment : c));
      return updatedComment;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add reply');
      throw err;
    }
  }, [fileId]);

  const resolveComment = useCallback(async (commentId: string) => {
    if (!fileId) return;
    
    try {
      const updatedComment = await apiResolveComment(fileId, commentId);
      setComments(prev => prev.map(c => c.id === commentId ? updatedComment : c));
      loadCommentCounts(); // Refresh counts
      return updatedComment;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to resolve comment');
      throw err;
    }
  }, [fileId, loadCommentCounts]);

  const deleteComment = useCallback(async (commentId: string) => {
    if (!fileId) return;
    
    try {
      await apiDeleteComment(fileId, commentId);
      setComments(prev => prev.filter(c => c.id !== commentId));
      loadCommentCounts(); // Refresh counts
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete comment');
      throw err;
    }
  }, [fileId, loadCommentCounts]);

  const updateComment = useCallback(async (commentId: string, content: string) => {
    if (!fileId) return;
    
    try {
      const updatedComment = await apiUpdateComment(fileId, commentId, content);
      setComments(prev => prev.map(c => c.id === commentId ? updatedComment : c));
      return updatedComment;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update comment');
      throw err;
    }
  }, [fileId]);

  return {
    comments,
    commentCounts,
    loading,
    error,
    addComment,
    addReply,
    resolveComment,
    deleteComment,
    updateComment,
    refresh: loadComments,
  };
}

