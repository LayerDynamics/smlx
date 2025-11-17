import { useState, useRef, useCallback } from 'react';
import type { KeyboardEvent } from 'react';
import { useChatStore } from '../../store/chatStore';
import { useChatCompletion } from '../../hooks/useChatCompletion';
import { Button } from '../common/Button';

export function ChatComposer() {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const activeConversationId = useChatStore((state) => state.activeConversationId);
  const streamingState = useChatStore((state) => state.streamingState);

  const { sendMessage, isLoading, error, cancelGeneration } = useChatCompletion();

  const handleSubmit = useCallback(async () => {
    const trimmedInput = input.trim();
    if (!trimmedInput || !activeConversationId || isLoading) return;

    setInput('');

    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    await sendMessage(trimmedInput, activeConversationId);
  }, [input, activeConversationId, isLoading, sendMessage]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      // Enter to send, Shift+Enter for newline
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  const handleInputChange = useCallback((value: string) => {
    setInput(value);

    // Auto-resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, []);

  const isDisabled = !activeConversationId || isLoading;
  const isStreaming = streamingState.isStreaming;

  return (
    <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 p-4">
      {/* Error display */}
      {error && (
        <div className="mb-3 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Input area */}
      <div className="flex gap-2 items-end">
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => handleInputChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              activeConversationId
                ? 'Type a message... (Enter to send, Shift+Enter for new line)'
                : 'Select or create a conversation to start chatting'
            }
            disabled={isDisabled}
            rows={1}
            className="
              w-full px-4 py-3
              bg-gray-50 dark:bg-gray-800
              border border-gray-300 dark:border-gray-600
              rounded-lg resize-none
              focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
              disabled:opacity-50 disabled:cursor-not-allowed
              text-gray-900 dark:text-gray-100
              placeholder-gray-500 dark:placeholder-gray-400
              transition-colors duration-200
            "
            style={{
              maxHeight: '200px',
              overflowY: 'auto',
            }}
          />
        </div>

        {/* Action buttons */}
        <div className="flex gap-2">
          {isStreaming ? (
            <Button
              variant="danger"
              size="md"
              onClick={cancelGeneration}
              title="Cancel generation"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
              Cancel
            </Button>
          ) : (
            <Button
              variant="primary"
              size="md"
              onClick={handleSubmit}
              disabled={isDisabled || !input.trim()}
              isLoading={isLoading && !isStreaming}
              title="Send message (Enter)"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                />
              </svg>
              Send
            </Button>
          )}
        </div>
      </div>

      {/* Helper text */}
      <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 flex justify-between items-center">
        <span>Press Enter to send, Shift+Enter for new line</span>
        {input.length > 0 && (
          <span className={input.length > 1000 ? 'text-orange-600 dark:text-orange-400' : ''}>
            {input.length} characters
          </span>
        )}
      </div>
    </div>
  );
}
