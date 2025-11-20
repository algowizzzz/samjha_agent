import { useEffect, useRef, useState, memo, useCallback } from 'react';
import { Send, Sparkles, X, Copy, Check, RotateCcw, FileDown, Search, Moon, Sun, ChevronDown, ChevronUp, MessageSquare } from 'lucide-react';
import { getDocument, askRiskGPT, type BlockMetadata, listChatMessages, addChatMessage, type ChatMessage as ApiChatMessage } from '@/lib/api';
import { CommentsPane } from './CommentsPane';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  analysis?: string; // Analysis section from RiskGPT
  suggestions?: Array<{
    block_id: string;
    original: string;
    suggested: string;
    reason: string;
  }>;
  selectedBlocks?: BlockMetadata[];
  timestamp?: Date;
  isCollapsed?: boolean;
}

interface RightPaneProps {
  selectedText: string;
  selectedBlockId: string | null;
  onCommentClick: (blockId: string) => void;
  fileId: string | null;
  selectedBlocks?: BlockMetadata[]; // NEW: Selected blocks from BlockEditor
  onSuggestionsReceived?: (suggestions: Array<{ block_id: string; original: string; suggested: string; reason: string }>) => void; // NEW: Callback when suggestions received
  synthesisData?: any; // NEW: Template synthesis summary
  onDeselectBlock?: (blockId: string) => void; // NEW: Callback to deselect a specific block
  onClearAllBlocks?: () => void; // NEW: Callback to clear all selected blocks
  textSuggestion?: { original: string; suggested: string } | null; // NEW: AI text improvement suggestion
  onAcceptTextSuggestion?: () => void; // NEW: Accept text suggestion
  onRejectTextSuggestion?: () => void; // NEW: Reject text suggestion
}

function RightPaneComponent({ selectedText, selectedBlockId, onCommentClick, fileId, selectedBlocks = [], onSuggestionsReceived, synthesisData, onDeselectBlock, onClearAllBlocks, textSuggestion, onAcceptTextSuggestion, onRejectTextSuggestion }: RightPaneProps) {
  console.log('🔵 [RightPane] RENDER START - fileId:', fileId);
  
  const [chatMessages, setChatMessagesRaw] = useState<ChatMessage[]>(() => {
    console.log('🟢 [RightPane] chatMessages initial state');
    return [];
  });
  
  // Wrap setChatMessages to log all calls
  const setChatMessages = useCallback((value: ChatMessage[] | ((prev: ChatMessage[]) => ChatMessage[])) => {
    const stack = new Error().stack?.split('\n').slice(2, 6).join('\n');
    console.log('🚨 [RightPane] setChatMessages CALLED! Stack:', stack);
    console.log('🚨 [RightPane] New value:', typeof value === 'function' ? 'function' : value);
    setChatMessagesRaw(value);
  }, []);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const loadedFileIdRef = useRef<string | null>(null);
  const renderCountRef = useRef(0);
  
  renderCountRef.current += 1;
  console.log('🔵 [RightPane] Render #', renderCountRef.current, '- Messages count:', chatMessages.length);
  
  // NEW: Feature states
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [showSearch, setShowSearch] = useState(false);
  const [collapsedMessages, setCollapsedMessages] = useState<Set<string>>(new Set());

  // Component mount/unmount logging
  useEffect(() => {
    console.log('🟢 [RightPane] COMPONENT MOUNTED');
    return () => {
      console.log('🔴 [RightPane] COMPONENT UNMOUNTING');
    };
  }, []);

  // Apply dark mode to entire page
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
      document.body.style.backgroundColor = '#1a1a1a';
    } else {
      document.documentElement.classList.remove('dark');
      document.body.style.backgroundColor = '';
    }
  }, [isDarkMode]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  // NEW: Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [inputMessage]);

  // Load chat history when fileId changes
  useEffect(() => {
    console.log('🟡 [RightPane] useEffect(fileId) running, fileId:', fileId, 'loadedFileIdRef:', loadedFileIdRef.current, 'current messages:', chatMessages.length);
    console.log('🟡 [RightPane] Stack:', new Error().stack?.split('\n').slice(1, 5).join('\n'));
    
    // Skip if we've already loaded this fileId AND have messages
    if (fileId && loadedFileIdRef.current === fileId && chatMessages.length > 0) {
      console.log('⚪ [RightPane] Already loaded this fileId with messages, skipping');
      return;
    }
    
    if (!fileId) {
      console.log('🔴 [RightPane] NO FILEID - BUT NOT CLEARING (keep messages)');
      console.log('🔴 [RightPane] Stack:', new Error().stack?.split('\n').slice(1, 5).join('\n'));
      // DON'T clear messages when fileId becomes null temporarily
      // setChatMessages([]);
      // loadedFileIdRef.current = null;
      return;
    }

    const loadChatHistory = async () => {
      try {
        console.log('🟢 [RightPane] Loading chat for fileId:', fileId);
        const result = await listChatMessages(fileId);
        // Convert API messages to local ChatMessage format
        const loadedMessages: ChatMessage[] = result.messages.map(apiMsg => ({
          id: apiMsg.id,
          role: apiMsg.role,
          content: apiMsg.content,
          timestamp: new Date(apiMsg.timestamp),
        }));
        console.log('🟢 [RightPane] Calling setChatMessages with', loadedMessages.length, 'messages');
        setChatMessages(loadedMessages);
        loadedFileIdRef.current = fileId;
        console.log('✅ [RightPane] Loaded', loadedMessages.length, 'chat messages');
      } catch (error) {
        console.error('❌ [RightPane] Error loading chat history:', error);
        // Don't clear existing messages on error
      }
    };

    loadChatHistory();
  }, [fileId]);

  // Debug: Log chatMessages state changes
  useEffect(() => {
    console.log('💙 [RightPane] chatMessages changed to length:', chatMessages.length);
    if (chatMessages.length === 0) {
      console.log('💙 [RightPane] MESSAGES ARE EMPTY! Stack:', new Error().stack?.split('\n').slice(1, 5).join('\n'));
    }
  }, [chatMessages]);

  // NEW: Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd/Ctrl + K to focus input
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        textareaRef.current?.focus();
      }
      // Cmd/Ctrl + / to toggle search
      if ((e.metaKey || e.ctrlKey) && e.key === '/') {
        e.preventDefault();
        setShowSearch(prev => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // NEW: Copy message to clipboard
  const copyMessage = async (message: ChatMessage) => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopiedMessageId(message.id);
      setTimeout(() => setCopiedMessageId(null), 2000);
    } catch (err) {
      console.error('Failed to copy message:', err);
    }
  };

  // NEW: Export conversation
  const exportConversation = () => {
    const markdown = chatMessages
      .map(msg => {
        const timestamp = msg.timestamp ? new Date(msg.timestamp).toLocaleString() : '';
        const role = msg.role === 'user' ? 'You' : 'RiskGPT';
        return `## ${role} ${timestamp ? `(${timestamp})` : ''}\n\n${msg.content}\n`;
      })
      .join('\n---\n\n');
    
    const blob = new Blob([markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `riskgpt-conversation-${Date.now()}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // NEW: Toggle message collapse
  const toggleMessageCollapse = (messageId: string) => {
    setCollapsedMessages(prev => {
      const next = new Set(prev);
      if (next.has(messageId)) {
        next.delete(messageId);
      } else {
        next.add(messageId);
      }
      return next;
    });
  };

  // NEW: Regenerate response
  const regenerateResponse = async (messageId: string) => {
    const messageIndex = chatMessages.findIndex(msg => msg.id === messageId);
    if (messageIndex === -1 || !fileId) return;
    
    // Find the previous user message
    let userMessageIndex = messageIndex - 1;
    while (userMessageIndex >= 0 && chatMessages[userMessageIndex].role !== 'user') {
      userMessageIndex--;
    }
    
    if (userMessageIndex < 0) return;
    
    const userMessage = chatMessages[userMessageIndex];
    setIsLoading(true);
    
    try {
      const conversationHistory = chatMessages
        .slice(0, userMessageIndex)
        .slice(-10)
        .map(msg => ({
          role: msg.role,
          content: msg.content
        }));
      
      const selectedBlockIds = userMessage.selectedBlocks?.map(b => b.id) || [];
      const response = await askRiskGPT(
        fileId, 
        selectedBlockIds,
        userMessage.content,
        conversationHistory
      );
      
      const assistantContent = response.analysis || 'Here are my suggestions:';
      
      // Save regenerated assistant message to backend
      let savedAssistantMessage: ApiChatMessage | null = null;
      try {
        savedAssistantMessage = await addChatMessage(fileId, {
          role: 'assistant',
          content: assistantContent,
        });
      } catch (error) {
        console.error('[RightPane] Error saving regenerated message:', error);
      }

      const newMessage: ChatMessage = {
        id: savedAssistantMessage?.id || `a${Date.now()}`,
        role: 'assistant',
        content: assistantContent,
        analysis: response.analysis,
        suggestions: response.suggestions,
        timestamp: savedAssistantMessage ? new Date(savedAssistantMessage.timestamp) : new Date(),
      };

      // Replace the old response with the new one
      setChatMessages(prev => [
        ...prev.slice(0, messageIndex),
        newMessage,
        ...prev.slice(messageIndex + 1)
      ]);
      
      if (onSuggestionsReceived && response.suggestions && response.suggestions.length > 0) {
        onSuggestionsReceived(response.suggestions);
      }
    } catch (error) {
      console.error('[RightPane] Error regenerating response:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // NEW: Apply all suggestions from a message
  const applyAllSuggestions = (message: ChatMessage) => {
    if (message.suggestions && onSuggestionsReceived) {
      onSuggestionsReceived(message.suggestions);
    }
  };

  // Auto-inject synthesis as first message when template is applied (for live updates)
  useEffect(() => {
    if (synthesisData) {
      const synthesisMessage = formatSynthesisMessage(synthesisData);
      setChatMessages(prev => {
        // Check if synthesis message already exists
        const hasSynthesis = prev.some(msg => msg.id === 'synthesis');
        if (hasSynthesis) {
          // Replace existing synthesis
          return prev.map(msg => msg.id === 'synthesis' ? synthesisMessage : msg);
        } else {
          // Add as first message (only if not already loaded from backend)
          return [synthesisMessage, ...prev];
        }
      });
    }
  }, [synthesisData]);

  // Format synthesis data into a nice chat message
  const formatSynthesisMessage = (synthesis: any): ChatMessage => {
    // New format: synthesis is now markdown directly
    if (synthesis?.summary_markdown) {
      const content = synthesis.summary_markdown;
      const stats = synthesis.statistics || {};
      
      return {
        id: 'synthesis',
        role: 'assistant',
        content: content,
      };
    }
    
    // Fallback for old format (if any exists)
    const { overall_assessment, critical_gaps, improvement_areas, strengths, priority_recommendations, statistics } = synthesis || {};
    
    if (!overall_assessment) {
      // Empty or invalid synthesis
      return {
        id: 'synthesis',
        role: 'assistant',
        content: 'Template analysis completed. Check the suggestions panel for details.',
      };
    }
    
    let content = `I've analyzed your document against the template.\n\n`;
    content += `**Overall Assessment: ${overall_assessment.compliance_level} Compliance (${overall_assessment.compliance_percentage}%)**\n\n`;
    content += `${overall_assessment.summary}\n\n`;
    
    if (critical_gaps && critical_gaps.length > 0) {
      content += `🔴 **Critical Gaps (${critical_gaps.length}):**\n`;
      critical_gaps.slice(0, 3).forEach((gap: any) => {
        content += `• **${gap.title}** - ${gap.impact}\n`;
        if (gap.affected_pages && gap.affected_pages.length > 0) {
          content += `  Pages: ${gap.affected_pages.join(', ')}\n`;
        }
      });
      content += `\n`;
    }
    
    if (improvement_areas && improvement_areas.length > 0) {
      content += `🟡 **Improvements Needed (${statistics?.total_issues || improvement_areas.length}):**\n`;
      improvement_areas.slice(0, 5).forEach((area: any) => {
        content += `• **${area.title}** (${area.issue_count} issues)\n`;
      });
      content += `\n`;
    }
    
    if (strengths && strengths.length > 0) {
      content += `✅ **Strengths:**\n`;
      strengths.forEach((strength: string) => {
        content += `• ${strength}\n`;
      });
      content += `\n`;
    }
    
    if (priority_recommendations && priority_recommendations.length > 0) {
      content += `**Priority Recommendations:**\n`;
      priority_recommendations.forEach((rec: string, idx: number) => {
        content += `${idx + 1}. ${rec}\n`;
      });
      content += `\n`;
    }
    
    content += `I've highlighted ${statistics?.total_issues || 0} specific suggestions in the document. What would you like to address first?`;
    
    return {
      id: 'synthesis',
      role: 'assistant',
      content,
    };
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !fileId) return;

    const userMessageContent = inputMessage;
    const userContext = selectedText || undefined;

    // Save user message to backend first
    let savedUserMessage: ApiChatMessage | null = null;
    try {
      savedUserMessage = await addChatMessage(fileId, {
        role: 'user',
        content: userMessageContent,
        context: userContext,
      });
    } catch (error) {
      console.error('[RightPane] Error saving user message:', error);
      // Continue anyway with local ID
    }

    // Support both modes: with blocks (block-specific) or without (general chat)
    const userMessage: ChatMessage = {
      id: savedUserMessage?.id || `u${Date.now()}`,
      role: 'user',
      content: userMessageContent,
      selectedBlocks: selectedBlocks.length > 0 ? selectedBlocks : undefined,
      timestamp: savedUserMessage ? new Date(savedUserMessage.timestamp) : new Date(),
    };

    setChatMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      // Prepare conversation history (last 5 messages, alternating user/assistant)
      const conversationHistory = chatMessages
        .slice(-10) // Get last 10 messages (5 pairs)
        .map(msg => ({
          role: msg.role,
          content: msg.content
        }));
      
      const selectedBlockIds = selectedBlocks.map(b => b.id);
      const response = await askRiskGPT(
        fileId, 
        selectedBlockIds, // Empty array for general chat
        userMessageContent,
        conversationHistory
      );
      
      const assistantContent = response.analysis || 'Here are my suggestions:';
      
      // Save assistant message to backend
      let savedAssistantMessage: ApiChatMessage | null = null;
      try {
        savedAssistantMessage = await addChatMessage(fileId, {
          role: 'assistant',
          content: assistantContent,
        });
      } catch (error) {
        console.error('[RightPane] Error saving assistant message:', error);
        // Continue anyway with local ID
      }

      const assistantMessage: ChatMessage = {
        id: savedAssistantMessage?.id || `a${Date.now()}`,
        role: 'assistant',
        content: assistantContent,
        analysis: response.analysis,
        suggestions: response.suggestions,
        timestamp: savedAssistantMessage ? new Date(savedAssistantMessage.timestamp) : new Date(),
      };

      setChatMessages(prev => [...prev, assistantMessage]);
      
      // Apply suggestions to blocks in editor (only if block-specific)
      if (onSuggestionsReceived && response.suggestions && response.suggestions.length > 0) {
        onSuggestionsReceived(response.suggestions);
      }
    } catch (error) {
      console.error('[RightPane] Error asking RiskGPT:', error);
      const errorMessage: ChatMessage = {
        id: `e${Date.now()}`,
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
      };
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Filter messages by search query
  const filteredMessages = searchQuery
    ? chatMessages.filter(msg => 
        msg.content.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : chatMessages;

  console.log('🔵 [RightPane] RENDER END - chatMessages:', chatMessages.length, 'filteredMessages:', filteredMessages.length);

  return (
    <div className={`flex flex-col h-full ${isDarkMode ? 'bg-neutral-900' : 'bg-white'}`}>
      {/* Header - RiskGPT Only (Comments hidden) */}
      <div className={`border-b ${isDarkMode ? 'border-neutral-700 bg-neutral-800' : 'border-neutral-200 bg-white'} px-4`}>
        <div className="flex items-center gap-2 py-3 justify-end">
          <span className="text-sm font-medium">RiskGPT</span>
          {chatMessages.length > 0 && (
            <span className={`text-xs px-1.5 py-0.5 rounded ${
              isDarkMode ? 'bg-neutral-700 text-neutral-300' : 'bg-neutral-100 text-neutral-600'
            }`}>
              {chatMessages.length}
            </span>
          )}
        </div>
      </div>

      {/* Chat Header with actions */}
      <div className={`border-b ${isDarkMode ? 'border-neutral-700 bg-neutral-800' : 'border-neutral-200 bg-neutral-50'} px-4 py-3`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <h3 className={`text-xs font-medium ${isDarkMode ? 'text-neutral-300' : 'text-neutral-600'}`}>
                Ask about your document
              </h3>
            </div>
            <div className="flex items-center gap-1">
            {/* Search button */}
            <button
              onClick={() => setShowSearch(!showSearch)}
              className={`p-1.5 rounded ${isDarkMode ? 'hover:bg-neutral-700' : 'hover:bg-neutral-200'} transition-colors`}
              title="Search messages (Cmd+/)"
            >
              <Search className={`w-4 h-4 ${isDarkMode ? 'text-neutral-300' : 'text-neutral-600'}`} />
            </button>
            {/* Export button */}
            <button
              onClick={exportConversation}
              disabled={chatMessages.length === 0}
              className={`p-1.5 rounded ${isDarkMode ? 'hover:bg-neutral-700 disabled:opacity-30' : 'hover:bg-neutral-200 disabled:opacity-30'} transition-colors disabled:cursor-not-allowed`}
              title="Export conversation"
            >
              <FileDown className={`w-4 h-4 ${isDarkMode ? 'text-neutral-300' : 'text-neutral-600'}`} />
            </button>
            {/* Dark mode toggle */}
            <button
              onClick={() => setIsDarkMode(!isDarkMode)}
              className={`p-1.5 rounded ${isDarkMode ? 'hover:bg-neutral-700' : 'hover:bg-neutral-200'} transition-colors`}
              title="Toggle dark mode"
            >
              {isDarkMode ? (
                <Sun className="w-4 h-4 text-neutral-300" />
              ) : (
                <Moon className="w-4 h-4 text-neutral-600" />
              )}
            </button>
          </div>
        </div>
        
        {/* Search bar */}
        {showSearch && (
          <div className="mt-2">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search messages..."
              className={`w-full px-3 py-1.5 text-sm rounded border ${
                isDarkMode 
                  ? 'bg-neutral-700 border-neutral-600 text-neutral-100 placeholder-neutral-400' 
                  : 'bg-white border-neutral-300 text-neutral-900 placeholder-neutral-500'
              } focus:outline-none focus:ring-2 focus:ring-blue-500`}
              autoFocus
            />
          </div>
        )}
      </div>

      {/* Chat Content */}
      {/* Messages Area - Full Height */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {filteredMessages.length === 0 && chatMessages.length === 0 && (
          <div className="flex items-center justify-center h-full text-center">
            <div className={isDarkMode ? 'text-neutral-400' : 'text-neutral-500'}>
              <p className="text-sm font-medium mb-2">Welcome to RiskGPT</p>
              <p className="text-xs max-w-xs mx-auto">Select blocks and ask me to improve them, or ask general questions about your document</p>
              <div className="mt-4 text-xs space-y-1">
                <p className={isDarkMode ? 'text-neutral-500' : 'text-neutral-400'}>💡 Try: "Improve the clarity of this section"</p>
                <p className={isDarkMode ? 'text-neutral-500' : 'text-neutral-400'}>💡 Try: "Check for compliance issues"</p>
              </div>
            </div>
          </div>
        )}
        
        {filteredMessages.length === 0 && chatMessages.length > 0 && (
          <div className="flex items-center justify-center h-full text-center">
            <div className={isDarkMode ? 'text-neutral-400' : 'text-neutral-500'}>
              <Search className={`w-12 h-12 mx-auto mb-3 ${isDarkMode ? 'text-neutral-600' : 'text-neutral-300'}`} />
              <p className="text-sm">No messages match your search</p>
            </div>
          </div>
        )}
        
        {filteredMessages.map((message) => {
          const isCollapsed = collapsedMessages.has(message.id);
          const shouldShowCollapse = message.content.length > 500;
          const displayContent = isCollapsed && shouldShowCollapse 
            ? message.content.substring(0, 500) + '...' 
            : message.content;
          
          return (
          <div
            key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} group`}
          >
            <div className={`max-w-[85%] flex flex-col gap-2`}>
              {/* Timestamp (hover to show) */}
              {message.timestamp && (
                <div className={`text-xs ${isDarkMode ? 'text-neutral-500' : 'text-neutral-400'} ${message.role === 'user' ? 'text-right' : 'text-left'} opacity-0 group-hover:opacity-100 transition-opacity`}>
                  {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              )}
              
              {/* Selected Blocks Attachment */}
              {message.selectedBlocks && message.selectedBlocks.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mb-1">
                  {message.selectedBlocks.map((block) => (
                    <div
                      key={block.id}
                      className={`px-2 py-1 rounded text-xs flex items-center gap-1 ${
                        isDarkMode 
                          ? 'bg-blue-900 text-blue-200 border border-blue-700' 
                          : 'bg-blue-100 text-blue-800 border border-blue-200'
                      }`}
                      title={block.content}
                    >
                      <span className="font-medium">B{block.block_num}</span>
                      <span className={`${isDarkMode ? 'text-blue-400' : 'text-blue-600'} text-[10px]`}>
                        {block.content.substring(0, 5)}...
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {/* Message Bubble with header actions */}
              <div className="relative">
              <div
                className={`px-4 py-3 rounded-2xl text-sm ${
                  message.role === 'user'
                      ? isDarkMode 
                        ? 'bg-blue-700 text-white rounded-br-sm' 
                        : 'bg-blue-600 text-white rounded-br-sm'
                      : isDarkMode 
                        ? 'bg-neutral-800 text-neutral-100 rounded-bl-sm border border-neutral-700' 
                    : 'bg-neutral-100 text-neutral-900 rounded-bl-sm'
                }`}
              >
                {message.role === 'assistant' ? (
                  <div 
                    className={`prose prose-sm max-w-none prose-headings:mt-3 prose-headings:mb-2 prose-p:my-2 prose-ul:my-2 prose-li:my-0 ${isDarkMode ? 'prose-invert' : ''}`}
                    dangerouslySetInnerHTML={{ 
                      __html: displayContent
                        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
                        .replace(/\*(.+?)\*/g, '<em>$1</em>')
                        .replace(/^### (.+)$/gm, '<h3 class="font-semibold text-base">$1</h3>')
                        .replace(/^## (.+)$/gm, '<h2 class="font-semibold text-lg">$1</h2>')
                        .replace(/^# (.+)$/gm, '<h1 class="font-bold text-xl">$1</h1>')
                        .replace(/^- (.+)$/gm, '<li>$1</li>')
                        .replace(/(<li>.*<\/li>\n?)+/g, '<ul class="list-disc pl-5">$&</ul>')
                        .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
                        .replace(/\n\n/g, '<br/><br/>')
                        .replace(/\n/g, '<br/>')
                    }}
                  />
                ) : (
                  <p className="whitespace-pre-wrap">{displayContent}</p>
                )}
                
                {/* Collapse toggle for long messages */}
                {shouldShowCollapse && (
                  <button
                    onClick={() => toggleMessageCollapse(message.id)}
                    className={`mt-2 text-xs flex items-center gap-1 ${
                      message.role === 'user'
                        ? 'text-blue-100 hover:text-white'
                        : isDarkMode
                          ? 'text-blue-400 hover:text-blue-300'
                          : 'text-blue-600 hover:text-blue-800'
                    }`}
                  >
                    {isCollapsed ? (
                      <>
                        <ChevronDown className="w-3 h-3" />
                        Show more
                      </>
                    ) : (
                      <>
                        <ChevronUp className="w-3 h-3" />
                        Show less
                      </>
                    )}
                  </button>
                )}
              </div>

              {/* Action buttons (hover to show) */}
              <div className={`flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                {/* Copy button */}
                <button
                  onClick={() => copyMessage(message)}
                  className={`p-1.5 rounded text-xs flex items-center gap-1 ${
                    isDarkMode 
                      ? 'hover:bg-neutral-700 text-neutral-400 hover:text-neutral-200' 
                      : 'hover:bg-neutral-200 text-neutral-600 hover:text-neutral-900'
                  }`}
                  title="Copy message"
                >
                  {copiedMessageId === message.id ? (
                    <><Check className="w-3 h-3" /> Copied</>
                  ) : (
                    <><Copy className="w-3 h-3" /> Copy</>
                  )}
                </button>
                
                {/* Regenerate button (assistant messages only) */}
                {message.role === 'assistant' && (
                  <button
                    onClick={() => regenerateResponse(message.id)}
                    disabled={isLoading}
                    className={`p-1.5 rounded text-xs flex items-center gap-1 ${
                      isDarkMode 
                        ? 'hover:bg-neutral-700 text-neutral-400 hover:text-neutral-200 disabled:opacity-30' 
                        : 'hover:bg-neutral-200 text-neutral-600 hover:text-neutral-900 disabled:opacity-30'
                    } disabled:cursor-not-allowed`}
                    title="Regenerate response"
                  >
                    <RotateCcw className="w-3 h-3" />
                    Regenerate
                  </button>
                )}
                
                {/* Apply all button (if there are suggestions) */}
                {message.role === 'assistant' && message.suggestions && message.suggestions.length > 0 && (
                  <button
                    onClick={() => applyAllSuggestions(message)}
                    className={`p-1.5 rounded-lg text-xs flex items-center gap-1.5 font-semibold shadow-sm transition-all hover:scale-105 active:scale-95 ${
                      isDarkMode 
                        ? 'bg-gradient-to-r from-green-700 to-green-600 text-white hover:from-green-600 hover:to-green-500 border border-green-500' 
                        : 'bg-gradient-to-r from-green-500 to-emerald-500 text-white hover:from-green-600 hover:to-emerald-600 border border-green-400'
                    }`}
                    title="Apply all suggestions to document"
                  >
                    <Check className="w-3.5 h-3.5" />
                    Apply All ({message.suggestions.length})
                  </button>
                )}
                </div>
            </div>
          </div>
        </div>
        );
        })}
        
        {isLoading && (
          <div className="flex justify-start">
            <div className={`px-4 py-3 rounded-2xl rounded-bl-sm ${isDarkMode ? 'bg-neutral-800' : 'bg-neutral-100'}`}>
              <div className="flex gap-1">
                <div className={`w-2 h-2 rounded-full animate-bounce ${isDarkMode ? 'bg-neutral-500' : 'bg-neutral-400'}`} style={{ animationDelay: '0ms' }}></div>
                <div className={`w-2 h-2 rounded-full animate-bounce ${isDarkMode ? 'bg-neutral-500' : 'bg-neutral-400'}`} style={{ animationDelay: '150ms' }}></div>
                <div className={`w-2 h-2 rounded-full animate-bounce ${isDarkMode ? 'bg-neutral-500' : 'bg-neutral-400'}`} style={{ animationDelay: '300ms' }}></div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area - Fixed at Bottom */}
      <div className={`border-t ${isDarkMode ? 'border-neutral-700 bg-neutral-800' : 'border-neutral-200 bg-white'} p-4`}>
        {/* Smart prompt suggestions */}
        {!inputMessage && selectedBlocks.length > 0 && chatMessages.length === 0 && (
          <div className="mb-3 flex flex-wrap gap-2">
            <span className={`text-xs ${isDarkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>Quick prompts:</span>
            {['Improve clarity', 'Check compliance', 'Simplify language', 'Add examples'].map((prompt) => (
              <button
                key={prompt}
                onClick={() => setInputMessage(prompt)}
                className={`px-2 py-1 text-xs rounded border ${
                  isDarkMode 
                    ? 'border-neutral-600 text-neutral-300 hover:bg-neutral-700' 
                    : 'border-neutral-300 text-neutral-700 hover:bg-neutral-100'
                } transition-colors`}
              >
                {prompt}
              </button>
            ))}
          </div>
        )}
        
        {/* Current Text Selection Display */}
        {selectedText && selectedText.trim().length > 0 && (
          <div className={`mb-3 px-3 py-2 rounded-lg border ${
            isDarkMode 
              ? 'bg-neutral-700/50 border-neutral-600' 
              : 'bg-blue-50/50 border-blue-200'
          }`}>
            <div className={`text-xs font-medium mb-1 ${isDarkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>
              Selection:
            </div>
            <div className={`text-sm ${isDarkMode ? 'text-neutral-300' : 'text-neutral-700'} line-clamp-2`}>
              "{selectedText.substring(0, 80)}{selectedText.length > 80 ? '...' : ''}"
            </div>
          </div>
        )}
        
        {/* AI Text Suggestion Display - Compact & Pretty */}
        {textSuggestion && (
          <div style={{
            marginBottom: '8px',
            padding: '8px',
            borderRadius: '6px',
            border: '1px solid #3b82f6',
            backgroundColor: isDarkMode ? 'rgba(59, 130, 246, 0.08)' : '#f0f9ff',
            boxShadow: '0 1px 3px rgba(59, 130, 246, 0.15)',
          }}>
            {/* Header with title and buttons */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
              <div style={{ fontSize: '10px', fontWeight: '600', color: isDarkMode ? '#93c5fd' : '#2563eb', display: 'flex', alignItems: 'center', gap: '3px' }}>
                <span style={{ fontSize: '11px' }}>✨</span>
                AI
              </div>
              <div style={{ display: 'flex', gap: '4px' }}>
                <button
                  onClick={onAcceptTextSuggestion}
                  style={{
                    padding: '3px 8px',
                    fontSize: '10px',
                    fontWeight: '600',
                    borderRadius: '4px',
                    border: 'none',
                    cursor: 'pointer',
                    backgroundColor: '#10b981',
                    color: 'white',
                    transition: 'all 0.15s',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#059669';
                    e.currentTarget.style.transform = 'scale(1.05)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = '#10b981';
                    e.currentTarget.style.transform = 'scale(1)';
                  }}
                  title="Accept suggestion"
                >
                  ✓
                </button>
                <button
                  onClick={onRejectTextSuggestion}
                  style={{
                    padding: '3px 8px',
                    fontSize: '10px',
                    fontWeight: '600',
                    borderRadius: '4px',
                    border: 'none',
                    cursor: 'pointer',
                    backgroundColor: '#ef4444',
                    color: 'white',
                    transition: 'all 0.15s',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = '#dc2626';
                    e.currentTarget.style.transform = 'scale(1.05)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = '#ef4444';
                    e.currentTarget.style.transform = 'scale(1)';
                  }}
                  title="Reject suggestion"
                >
                  ✕
                </button>
              </div>
            </div>

            {/* Original and Improved in compact layout */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              {/* Original */}
              <div style={{
                padding: '4px 6px',
                borderRadius: '4px',
                backgroundColor: isDarkMode ? 'rgba(0,0,0,0.1)' : 'rgba(0,0,0,0.02)',
              }}>
                <div style={{ fontSize: '8px', fontWeight: '600', marginBottom: '2px', color: isDarkMode ? '#9ca3af' : '#6b7280', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  From
                </div>
                <div style={{ fontSize: '11px', color: isDarkMode ? '#d1d5db' : '#4b5563', textDecoration: 'line-through', opacity: 0.6 }}>
                  {textSuggestion.original}
                </div>
              </div>

              {/* Arrow */}
              <div style={{ textAlign: 'center', fontSize: '10px', color: isDarkMode ? '#60a5fa' : '#3b82f6', lineHeight: '1' }}>
                ↓
              </div>

              {/* Improved */}
              <div style={{
                padding: '4px 6px',
                borderRadius: '4px',
                backgroundColor: isDarkMode ? 'rgba(59, 130, 246, 0.15)' : 'rgba(59, 130, 246, 0.08)',
                border: '1px solid rgba(59, 130, 246, 0.3)',
              }}>
                <div style={{ fontSize: '8px', fontWeight: '600', marginBottom: '2px', color: isDarkMode ? '#93c5fd' : '#2563eb', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                  To
                </div>
                <div style={{ fontSize: '12px', fontWeight: '600', color: isDarkMode ? '#dbeafe' : '#1e40af' }}>
                  {textSuggestion.suggested}
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* Selected Blocks Display */}
        {selectedBlocks.length > 0 && (
          <div className="mb-3 flex flex-wrap gap-2 items-center">
            <span className={`text-xs ${isDarkMode ? 'text-neutral-400' : 'text-neutral-600'}`}>Selected ({selectedBlocks.length}):</span>
            {selectedBlocks.map((block) => (
              <div
                key={block.id}
                className={`px-2 py-1 rounded text-xs flex items-center gap-1.5 transition-colors ${
                  isDarkMode 
                    ? 'bg-blue-900 text-blue-200 hover:bg-blue-800 border border-blue-700' 
                    : 'bg-blue-100 text-blue-800 hover:bg-blue-200 border border-blue-200'
                }`}
                title={block.content}
              >
                <span className="font-medium">B{block.block_num}</span>
                <span className={`${isDarkMode ? 'text-blue-400' : 'text-blue-600'} text-[10px]`}>
                  {block.content.substring(0, 5)}...
                </span>
                {onDeselectBlock && (
                  <button
                    onClick={() => onDeselectBlock(block.id)}
                    className={`rounded p-0.5 transition-colors ${isDarkMode ? 'hover:bg-blue-700' : 'hover:bg-blue-300'}`}
                    title="Remove block"
                  >
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>
            ))}
            {selectedBlocks.length > 0 && onClearAllBlocks && (
              <button
                onClick={onClearAllBlocks}
                className={`px-2 py-1 text-xs rounded font-medium transition-colors ${
                  isDarkMode 
                    ? 'text-red-400 hover:text-red-300 hover:bg-red-900/30 border border-red-800' 
                    : 'text-red-600 hover:text-red-700 hover:bg-red-50 border border-red-200'
                }`}
                title="Clear all selected blocks"
              >
                ✕ Clear all
              </button>
            )}
          </div>
        )}
        
        <div className="flex gap-2 items-end">
          <textarea
            ref={textareaRef}
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
                e.preventDefault();
                handleSendMessage();
              }
            }}
            placeholder={selectedBlocks.length > 0 ? "Ask RiskGPT to improve selected blocks... (Cmd+Enter to send)" : "Ask RiskGPT about the document... (Cmd+Enter to send)"}
            disabled={isLoading}
            rows={1}
            className={`flex-1 px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm resize-none ${
              isDarkMode 
                ? 'bg-neutral-700 border-neutral-600 text-neutral-100 placeholder-neutral-400 disabled:bg-neutral-800 disabled:text-neutral-500' 
                : 'bg-white border-neutral-300 text-neutral-900 placeholder-neutral-500 disabled:bg-neutral-100'
            } disabled:cursor-not-allowed`}
            style={{ maxHeight: '200px' }}
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading}
            className={`px-4 py-2 rounded-lg flex items-center justify-center transition-colors ${
              isDarkMode 
                ? 'bg-blue-700 text-white hover:bg-blue-600 disabled:bg-neutral-700 disabled:text-neutral-500' 
                : 'bg-blue-600 text-white hover:bg-blue-700 disabled:bg-neutral-300 disabled:text-neutral-500'
            } disabled:cursor-not-allowed`}
            title="Send message (Cmd+Enter)"
          >
            {isLoading ? (
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>
        
        {/* Keyboard shortcuts hint */}
        <div className={`mt-2 text-xs ${isDarkMode ? 'text-neutral-500' : 'text-neutral-400'} flex justify-between`}>
          <span>Cmd+K to focus • Cmd+Enter to send • Cmd+/ to search</span>
          {chatMessages.length > 0 && (
            <span>{chatMessages.length} message{chatMessages.length !== 1 ? 's' : ''}</span>
          )}
        </div>
      </div>
    </div>
  );
}

// Export memoized version to prevent unnecessary remounts
export const RightPane = memo(RightPaneComponent);
