/**
 * API client for comment operations
 */

export interface Comment {
  id: string;
  block_id: string;
  block_title: string;
  author: string;
  timestamp: string;
  content: string;
  resolved: boolean;
  replies: Reply[];
  selection_text?: string;
}

export interface Reply {
  id: string;
  author: string;
  timestamp: string;
  content: string;
}

const API_BASE = '/api/doc_review';

/**
 * Fetch all comments for a document
 */
export async function fetchComments(fileId: string, blockId?: string): Promise<Comment[]> {
  const url = blockId 
    ? `${API_BASE}/${fileId}/comments?block_id=${blockId}`
    : `${API_BASE}/${fileId}/comments`;
  
  const response = await fetch(url, {
    credentials: 'include',
  });
  
  if (!response.ok) {
    throw new Error(`Failed to fetch comments: ${response.statusText}`);
  }
  
  const data = await response.json();
  return data.comments || [];
}

/**
 * Add a new comment
 */
export async function addComment(
  fileId: string,
  blockId: string,
  blockTitle: string,
  content: string,
  selectionText?: string
): Promise<Comment> {
  const response = await fetch(`${API_BASE}/${fileId}/comments`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    credentials: 'include',
    body: JSON.stringify({
      block_id: blockId,
      block_title: blockTitle,
      content,
      selection_text: selectionText,
      author: 'User', // TODO: Get from auth context
    }),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to add comment: ${response.statusText}`);
  }
  
  return response.json();
}

/**
 * Add a reply to a comment
 */
export async function addReply(
  fileId: string,
  commentId: string,
  content: string
): Promise<Comment> {
  const response = await fetch(`${API_BASE}/${fileId}/comments/${commentId}/reply`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    credentials: 'include',
    body: JSON.stringify({
      content,
      author: 'User', // TODO: Get from auth context
    }),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to add reply: ${response.statusText}`);
  }
  
  return response.json();
}

/**
 * Toggle resolved status of a comment
 */
export async function resolveComment(
  fileId: string,
  commentId: string
): Promise<Comment> {
  const response = await fetch(`${API_BASE}/${fileId}/comments/${commentId}/resolve`, {
    method: 'POST',
    credentials: 'include',
  });
  
  if (!response.ok) {
    throw new Error(`Failed to resolve comment: ${response.statusText}`);
  }
  
  return response.json();
}

/**
 * Delete a comment
 */
export async function deleteComment(
  fileId: string,
  commentId: string
): Promise<void> {
  const response = await fetch(`${API_BASE}/${fileId}/comments/${commentId}`, {
    method: 'DELETE',
    credentials: 'include',
  });
  
  if (!response.ok) {
    throw new Error(`Failed to delete comment: ${response.statusText}`);
  }
}

/**
 * Update a comment's content
 */
export async function updateComment(
  fileId: string,
  commentId: string,
  content: string
): Promise<Comment> {
  const response = await fetch(`${API_BASE}/${fileId}/comments/${commentId}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
    },
    credentials: 'include',
    body: JSON.stringify({ content }),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to update comment: ${response.statusText}`);
  }
  
  return response.json();
}

/**
 * Get comment counts by block
 */
export async function fetchCommentCounts(fileId: string): Promise<Record<string, number>> {
  const response = await fetch(`${API_BASE}/${fileId}/comments/counts`, {
    credentials: 'include',
  });
  
  if (!response.ok) {
    throw new Error(`Failed to fetch comment counts: ${response.statusText}`);
  }
  
  const data = await response.json();
  return data.counts || {};
}

