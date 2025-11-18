import { useState } from 'react';
import { MessageSquare, Check, MoreVertical, Send } from 'lucide-react';
import { Button } from './ui/button';
import { useComments } from '@/hooks/useComments';
import type { Comment } from '@/lib/comments-api';

interface CommentsPaneProps {
  fileId: string | null;
  selectedBlockId: string | null;
  onCommentClick: (blockId: string) => void;
}

export function CommentsPane({ fileId, selectedBlockId, onCommentClick }: CommentsPaneProps) {
  const {
    comments,
    loading,
    error,
    addComment: apiAddComment,
    addReply: apiAddReply,
    resolveComment: apiResolveComment,
  } = useComments(fileId);
  
  const [showResolved, setShowResolved] = useState(false);
  const [replyingTo, setReplyingTo] = useState<string | null>(null);
  const [replyText, setReplyText] = useState('');
  const [newCommentText, setNewCommentText] = useState('');

  const handleResolve = async (commentId: string) => {
    try {
      await apiResolveComment(commentId);
    } catch (err) {
      console.error('Failed to resolve comment:', err);
    }
  };

  const handleAddReply = async (commentId: string) => {
    if (!replyText.trim()) return;

    try {
      await apiAddReply(commentId, replyText);
      setReplyText('');
      setReplyingTo(null);
    } catch (err) {
      console.error('Failed to add reply:', err);
    }
  };

  const handleAddComment = async () => {
    if (!newCommentText.trim() || !selectedBlockId || !fileId) return;

    try {
      await apiAddComment(selectedBlockId, 'Selected Block', newCommentText);
      setNewCommentText('');
    } catch (err) {
      console.error('Failed to add comment:', err);
    }
  };

  const filteredComments = showResolved 
    ? comments 
    : comments.filter(c => !c.resolved);

  // Group comments by block
  const groupedComments = filteredComments.reduce((acc, comment) => {
    const key = comment.block_title;
    if (!acc[key]) {
      acc[key] = [];
    }
    acc[key].push(comment);
    return acc;
  }, {} as Record<string, Comment[]>);

  if (loading) {
    return (
      <div className="flex flex-col h-full bg-white items-center justify-center">
        <p className="text-neutral-600 text-sm">Loading comments...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col h-full bg-white items-center justify-center p-4 text-center">
        <MessageSquare className="w-12 h-12 text-neutral-300 mb-3" />
        <p className="text-neutral-700 font-medium mb-2">Unable to load comments</p>
        <p className="text-neutral-500 text-sm">{error}</p>
        {error.includes('NOT FOUND') && (
          <p className="text-neutral-400 text-xs mt-2">
            Document may not be saved yet. Try saving the document first.
          </p>
        )}
      </div>
    );
  }

  if (!fileId) {
    return (
      <div className="flex flex-col h-full bg-white items-center justify-center p-4 text-center">
        <MessageSquare className="w-12 h-12 text-neutral-300 mb-3" />
        <p className="text-neutral-600 text-sm">No document selected</p>
        <p className="text-neutral-400 text-xs mt-2">
          Open a document to view and add comments
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className="border-b border-neutral-200 p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-neutral-900 flex items-center gap-2">
            <MessageSquare className="w-4 h-4" />
            Comments
          </h3>
          <button
            onClick={() => setShowResolved(!showResolved)}
            className="text-xs text-neutral-600 hover:text-neutral-900"
          >
            {showResolved ? 'Hide' : 'Show'} resolved
          </button>
        </div>

        <div className="text-xs text-neutral-600">
          {filteredComments.length} {filteredComments.length === 1 ? 'comment' : 'comments'}
        </div>
      </div>

      {/* Comments List */}
      <div className="flex-1 overflow-y-auto">
        {Object.keys(groupedComments).length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full p-8 text-center">
            <MessageSquare className="w-12 h-12 text-neutral-300 mb-3" />
            <p className="text-neutral-600 text-sm">No comments yet</p>
            <p className="text-neutral-500 text-xs mt-1">
              Select text to add a comment
            </p>
          </div>
        ) : (
          Object.entries(groupedComments).map(([blockTitle, blockComments]) => (
            <div key={blockTitle} className="border-b border-neutral-100 p-4">
              {/* Block Title */}
              <button
                onClick={() => onCommentClick(blockComments[0].blockId)}
                className="text-xs font-medium text-neutral-700 mb-3 hover:text-blue-600 transition-colors"
              >
                {blockTitle}
              </button>

              {/* Comments in this block */}
              <div className="space-y-4">
                {blockComments.map((comment) => (
                  <div
                    key={comment.id}
                    className={`${
                      comment.resolved ? 'opacity-60' : ''
                    }`}
                  >
                    {/* Main Comment */}
                    <div className="bg-neutral-50 rounded-lg p-3">
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-xs font-medium text-neutral-900">
                              {comment.author}
                            </span>
                            <span className="text-xs text-neutral-500">
                              {comment.timestamp}
                            </span>
                          </div>
                          <p className="text-xs text-neutral-700">
                            {comment.content}
                          </p>
                        </div>
                        <button className="p-1 hover:bg-neutral-200 rounded">
                          <MoreVertical className="w-3 h-3 text-neutral-500" />
                        </button>
                      </div>

                      {/* Actions */}
                      <div className="flex items-center gap-2 mt-2">
                        <button
                          onClick={() => setReplyingTo(comment.id)}
                          className="text-xs text-blue-600 hover:text-blue-700"
                        >
                          Reply
                        </button>
                        <button
                          onClick={() => handleResolve(comment.id)}
                          className={`text-xs flex items-center gap-1 ${
                            comment.resolved
                              ? 'text-neutral-600'
                              : 'text-green-600 hover:text-green-700'
                          }`}
                        >
                          <Check className="w-3 h-3" />
                          {comment.resolved ? 'Resolved' : 'Resolve'}
                        </button>
                      </div>
                    </div>

                    {/* Replies */}
                    {comment.replies.length > 0 && (
                      <div className="ml-4 mt-2 space-y-2">
                        {comment.replies.map((reply) => (
                          <div
                            key={reply.id}
                            className="bg-white border border-neutral-200 rounded-lg p-3"
                          >
                            <div className="flex items-center gap-2 mb-1">
                              <span className="text-xs font-medium text-neutral-900">
                                {reply.author}
                              </span>
                              <span className="text-xs text-neutral-500">
                                {reply.timestamp}
                              </span>
                            </div>
                            <p className="text-xs text-neutral-700">
                              {reply.content}
                            </p>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Reply Input */}
                    {replyingTo === comment.id && (
                      <div className="ml-4 mt-2">
                        <div className="flex gap-2">
                          <input
                            type="text"
                            value={replyText}
                            onChange={(e) => setReplyText(e.target.value)}
                            placeholder="Write a reply..."
                            className="flex-1 px-3 py-2 border border-neutral-300 rounded text-xs focus:outline-none focus:ring-2 focus:ring-neutral-900"
                            onKeyPress={(e) => {
                              if (e.key === 'Enter') {
                                handleAddReply(comment.id);
                              }
                            }}
                          />
                          <Button
                            size="sm"
                            onClick={() => handleAddReply(comment.id)}
                            className="px-3"
                          >
                            <Send className="w-3 h-3" />
                          </Button>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Add Comment Input */}
      {selectedBlockId && (
        <div className="border-t border-neutral-200 p-4">
          <div className="flex flex-col gap-2">
            <label className="text-xs text-neutral-700">
              Add comment to selected block
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={newCommentText}
                onChange={(e) => setNewCommentText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleAddComment();
                  }
                }}
                placeholder="Type your comment..."
                className="flex-1 px-3 py-2 border border-neutral-300 rounded text-xs focus:outline-none focus:ring-2 focus:ring-neutral-900"
              />
              <Button size="sm" className="px-3" onClick={handleAddComment}>
                <Send className="w-3 h-3" />
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
