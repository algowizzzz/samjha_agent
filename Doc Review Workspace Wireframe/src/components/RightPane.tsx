import { useEffect, useRef, useState } from 'react';
import { Send, Sparkles, X } from 'lucide-react';
import { getDocument, askRiskGPT, type BlockMetadata } from '@/lib/api';

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
}

interface RightPaneProps {
  selectedText: string;
  selectedBlockId: string | null;
  onCommentClick: (blockId: string) => void;
  fileId: string | null;
  selectedBlocks?: BlockMetadata[]; // NEW: Selected blocks from BlockEditor
  onSuggestionsReceived?: (suggestions: Array<{ block_id: string; original: string; suggested: string; reason: string }>) => void; // NEW: Callback when suggestions received
}

export function RightPane({ selectedText, selectedBlockId, onCommentClick, fileId, selectedBlocks = [], onSuggestionsReceived }: RightPaneProps) {
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  // Load chat history from backend
  useEffect(() => {
    let cancelled = false;
    async function load() {
      if (!fileId) {
        setChatMessages([]);
        return;
      }
      try {
        const res = await getDocument(fileId);
        const state: any = res.document?.state || {};
        const history: any[] = state.chat_history || [];
        const mappedChat: ChatMessage[] = [];
        
        // Map chat history
        history.forEach((h, idx) => {
          if (h.message) {
            mappedChat.push({
              id: `u${idx}`,
              role: 'user',
              content: h.message,
            });
          }
          if (h.response) {
            mappedChat.push({
              id: `a${idx}`,
              role: 'assistant',
              content: h.response,
              analysis: h.analysis,
              suggestions: h.suggestions,
            });
          }
        });
        
        if (!cancelled) {
          setChatMessages(mappedChat);
        }
      } catch {
        if (!cancelled) {
          setChatMessages([]);
        }
      }
    }
    load();
    return () => { cancelled = true; };
  }, [fileId]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !fileId) return;

    // Support both modes: with blocks (block-specific) or without (general chat)
    const userMessage: ChatMessage = {
      id: `u${Date.now()}`,
      role: 'user',
      content: inputMessage,
      selectedBlocks: selectedBlocks.length > 0 ? selectedBlocks : undefined,
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
        inputMessage,
        conversationHistory
      );
      
      const assistantMessage: ChatMessage = {
        id: `a${Date.now()}`,
        role: 'assistant',
        content: response.analysis || 'Here are my suggestions:',
        analysis: response.analysis,
        suggestions: response.suggestions,
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

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Chat Header */}
      <div className="border-b border-neutral-200 px-4 py-3 bg-neutral-50">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-blue-600" />
          <h2 className="text-sm font-semibold text-neutral-900">RiskGPT</h2>
        </div>
      </div>

      {/* Messages Area - Full Height */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {chatMessages.length === 0 && (
          <div className="flex items-center justify-center h-full text-center">
            <div className="text-neutral-500">
              <Sparkles className="w-12 h-12 mx-auto mb-3 text-neutral-300" />
              <p className="text-sm">Select blocks and ask RiskGPT to improve them</p>
            </div>
          </div>
        )}
        
        {chatMessages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div className={`max-w-[85%] flex flex-col gap-2`}>
              {/* Selected Blocks Attachment */}
              {message.selectedBlocks && message.selectedBlocks.length > 0 && (
                <div className="flex flex-wrap gap-2 mb-2">
                  {message.selectedBlocks.map((block) => (
                    <div
                      key={block.id}
                      className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs flex items-center gap-1"
                    >
                      <span>Block {block.block_num}</span>
                      <button
                        onClick={() => {/* TODO: Scroll to block */}}
                        className="hover:text-blue-600"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* Message Bubble */}
              <div
                className={`px-4 py-3 rounded-2xl text-sm ${
                  message.role === 'user'
                    ? 'bg-blue-600 text-white rounded-br-sm'
                    : 'bg-neutral-100 text-neutral-900 rounded-bl-sm'
                }`}
              >
                {message.role === 'assistant' ? (
                  <div 
                    className="prose prose-sm max-w-none prose-headings:mt-3 prose-headings:mb-2 prose-p:my-2 prose-ul:my-2 prose-li:my-0"
                    dangerouslySetInnerHTML={{ 
                      __html: message.content
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
                  <p className="whitespace-pre-wrap">{message.content}</p>
                )}
              </div>

              {/* Analysis Section (only show if there are suggestions too) */}
              {message.analysis && message.suggestions && message.suggestions.length > 0 && (
                <div className="px-4 py-3 bg-blue-50 border border-blue-200 rounded-lg text-sm">
                  <p className="font-semibold text-blue-900 mb-2">Analysis:</p>
                  <p className="text-blue-800 whitespace-pre-wrap">{message.analysis}</p>
                </div>
              )}

              {/* Suggestions Section (if present) */}
              {message.suggestions && message.suggestions.length > 0 && (
                <div className="px-4 py-3 bg-neutral-50 border border-neutral-200 rounded-lg text-sm">
                  <p className="font-semibold text-neutral-900 mb-3">Suggested Changes:</p>
                  <div className="space-y-3">
                    {message.suggestions.map((sug, idx) => (
                      <div key={idx} className="border-l-4 border-blue-500 pl-3">
                        <p className="text-xs text-neutral-600 mb-1">{sug.reason}</p>
                        <p className="text-xs text-neutral-500 line-through mb-1">{sug.original}</p>
                        <p className="text-xs text-neutral-900 font-medium">{sug.suggested}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="flex justify-start">
            <div className="px-4 py-3 bg-neutral-100 rounded-2xl rounded-bl-sm">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                <div className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                <div className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area - Fixed at Bottom */}
      <div className="border-t border-neutral-200 p-4 bg-white">
        {/* Selected Blocks Display */}
        {selectedBlocks.length > 0 && (
          <div className="mb-3 flex flex-wrap gap-2">
            <span className="text-xs text-neutral-600 self-center">Selected:</span>
            {selectedBlocks.map((block) => (
              <div
                key={block.id}
                className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs flex items-center gap-1"
              >
                <span>{block.content.substring(0, 30)}...</span>
              </div>
            ))}
          </div>
        )}
        
        <div className="flex gap-2">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
              }
            }}
            placeholder={selectedBlocks.length > 0 ? "Ask RiskGPT to improve selected blocks..." : "Ask RiskGPT about the document..."}
            disabled={isLoading}
            className="flex-1 px-4 py-2 border border-neutral-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm disabled:bg-neutral-100 disabled:cursor-not-allowed"
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-neutral-300 disabled:cursor-not-allowed flex items-center justify-center"
          >
            {isLoading ? (
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
