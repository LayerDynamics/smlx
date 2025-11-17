import { useEffect, useRef, useState } from 'react';
import { useChatStore } from '../../store/chatStore';

export function ChatList() {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const activeConversationId = useChatStore((state) => state.activeConversationId);
  const getConversationMessages = useChatStore((state) => state.getConversationMessages);
  const streamingState = useChatStore((state) => state.streamingState);

  const messages = activeConversationId ? getConversationMessages(activeConversationId) : [];

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages.length, streamingState.content]);

  const formatTimestamp = (timestamp: number): string => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;

    return date.toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined,
    });
  };

  const handleCopyMessage = async (messageId: string, content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedId(messageId);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (err) {
      console.error('Failed to copy message:', err);
    }
  };

  // Empty state: no conversation selected
  const renderNoConversation = () => (
    <div className="flex items-center justify-center h-full p-8 text-gray-500 dark:text-gray-400">
      <div className="text-center max-w-md">
        <svg
          className="w-16 h-16 mx-auto mb-4 opacity-50"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
          />
        </svg>
        <h3 className="text-lg font-medium mb-2">No conversation selected</h3>
        <p className="text-sm">
          Select a conversation from the sidebar or create a new one to start chatting
        </p>
      </div>
    </div>
  );

  // Empty state: conversation exists but no messages
  const renderEmptyConversation = () => (
    <div className="flex items-center justify-center h-full p-8 text-gray-500 dark:text-gray-400">
      <div className="text-center max-w-md">
        <svg
          className="w-16 h-16 mx-auto mb-4 opacity-50"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
          />
        </svg>
        <h3 className="text-lg font-medium mb-2">Start a conversation</h3>
        <p className="text-sm">
          Type a message below to begin chatting with the AI assistant
        </p>
      </div>
    </div>
  );

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0"
      style={{ scrollBehavior: 'smooth' }}
    >
      {!activeConversationId ? (
        renderNoConversation()
      ) : messages.length === 0 && !streamingState.isStreaming ? (
        renderEmptyConversation()
      ) : (
        <>
          {messages.map((message) => {
            const isUser = message.role === 'user';
            const isSystem = message.role === 'system';
            const isCopied = copiedId === message.id;

            // Skip system messages in display
            if (isSystem) return null;

            return (
              <div
                key={message.id}
                className={`flex ${isUser ? 'justify-end' : 'justify-start'} group`}
              >
                <div
                  className={`
                    max-w-[80%] rounded-lg px-4 py-3 relative
                    ${
                      isUser
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100'
                    }
                  `}
                >
                  {/* Message content */}
                  <div className="whitespace-pre-wrap break-words">{message.content}</div>

                  {/* Message metadata */}
                  <div
                    className={`
                      flex items-center gap-2 mt-2 text-xs
                      ${isUser ? 'text-blue-100' : 'text-gray-500 dark:text-gray-400'}
                    `}
                  >
                    <span>{formatTimestamp(message.timestamp)}</span>
                    {message.tokenCount && (
                      <>
                        <span>•</span>
                        <span>{message.tokenCount} tokens</span>
                      </>
                    )}
                    {message.isStreaming && (
                      <>
                        <span>•</span>
                        <span className="flex items-center gap-1">
                          <span className="inline-block w-1 h-1 rounded-full bg-current animate-pulse" />
                          Streaming
                        </span>
                      </>
                    )}
                  </div>

                  {/* Copy button */}
                  <button
                    onClick={() => handleCopyMessage(message.id, message.content)}
                    className={`
                      absolute -top-2 ${isUser ? '-left-8' : '-right-8'}
                      p-1.5 rounded-md
                      bg-gray-200 dark:bg-gray-700
                      hover:bg-gray-300 dark:hover:bg-gray-600
                      opacity-0 group-hover:opacity-100
                      transition-opacity duration-200
                      ${isCopied ? 'opacity-100' : ''}
                    `}
                    title={isCopied ? 'Copied!' : 'Copy message'}
                  >
                    {isCopied ? (
                      <svg
                        className="w-4 h-4 text-green-600 dark:text-green-400"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                    ) : (
                      <svg
                        className="w-4 h-4 text-gray-600 dark:text-gray-400"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                        />
                      </svg>
                    )}
                  </button>
                </div>
              </div>
            );
          })}

          {/* Streaming message */}
          {streamingState.isStreaming &&
            streamingState.conversationId === activeConversationId && (
              <div className="flex justify-start group">
                <div className="max-w-[80%] rounded-lg px-4 py-3 bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100">
                  <div className="whitespace-pre-wrap break-words">
                    {streamingState.content || (
                      <span className="text-gray-500 dark:text-gray-400 italic">Thinking...</span>
                    )}
                  </div>
                  <div className="flex items-center gap-2 mt-2 text-xs text-gray-500 dark:text-gray-400">
                    <span className="flex items-center gap-1">
                      <span className="inline-block w-1 h-1 rounded-full bg-current animate-pulse" />
                      <span className="inline-block w-1 h-1 rounded-full bg-current animate-pulse animation-delay-150" />
                      <span className="inline-block w-1 h-1 rounded-full bg-current animate-pulse animation-delay-300" />
                      <span className="ml-1">Streaming</span>
                    </span>
                  </div>
                </div>
              </div>
            )}

          {/* Scroll anchor */}
          <div ref={messagesEndRef} />
        </>
      )}
    </div>
  );
}
