// Comment system types for document review

export interface Comment {
  id: string;
  documentId: string;
  blockId: string;           // Which block this comment is in
  selectedText: string;       // The exact text that was selected
  startOffset: number;        // Character offset within block
  endOffset: number;          // Character offset within block
  commentText: string;        // The comment content
  username: string;           // Who created it
  timestamp: string;          // ISO timestamp
  replies: Comment[];         // Nested replies
  parentId?: string;          // For replies
}

export interface CommentThread {
  id: string;
  documentId: string;
  blockId: string;
  selectedText: string;
  startOffset: number;
  endOffset: number;
  comments: Comment[];        // Thread of comments
}

// For creating a new comment
export interface CreateCommentRequest {
  documentId: string;
  blockId: string;
  selectedText: string;
  startOffset: number;
  endOffset: number;
  commentText: string;
  username?: string;
  parentId?: string;          // For replies
}

// For updating a comment
export interface UpdateCommentRequest {
  commentId: string;
  commentText: string;
}

// Backend API response format
export interface CommentResponse {
  success: boolean;
  comment?: Comment;
  error?: string;
}

