import { useState, useCallback } from 'react';
import { useChatStore } from '../store/chatStore';
import { useSettingsStore } from '../store/settingsStore';
import { smlxClient, SMLXAPIError } from '../services/api';
import type { ChatMessage } from '../types/api';

interface UseChatCompletionReturn {
  sendMessage: (content: string, conversationId?: string) => Promise<void>;
  isLoading: boolean;
  error: string | null;
  cancelGeneration: () => void;
}

export function useChatCompletion(): UseChatCompletionReturn {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [abortController, setAbortController] = useState<AbortController | null>(null);

  const {
    addMessage,
    startStreaming,
    appendStreamContent,
    endStreaming,
    getConversationMessages,
    updateTokenUsage,
  } = useChatStore();

  const settings = useSettingsStore();

  const sendMessage = useCallback(
    async (content: string, conversationId?: string) => {
      const activeConversationId =
        conversationId || useChatStore.getState().activeConversationId;

      if (!activeConversationId) {
        setError('No active conversation');
        return;
      }

      setIsLoading(true);
      setError(null);

      // Create abort controller for cancellation
      const controller = new AbortController();
      setAbortController(controller);

      try {
        // Add user message
        addMessage(activeConversationId, {
          role: 'user',
          content,
        });

        // Get conversation history
        const messages = getConversationMessages(activeConversationId);

        // Build messages array for API
        const apiMessages: ChatMessage[] = [];

        // Add system message if configured
        if (settings.systemPrompt) {
          apiMessages.push({
            role: 'system',
            content: settings.systemPrompt,
          });
        }

        // Add conversation history
        apiMessages.push(
          ...messages.map((msg) => ({
            role: msg.role,
            content: msg.content,
          }))
        );

        if (settings.streamEnabled) {
          // Create placeholder message for streaming
          const assistantMessageId = addMessage(activeConversationId, {
            role: 'assistant',
            content: '',
            isStreaming: true,
          });

          startStreaming(activeConversationId, assistantMessageId);

          // Stream the response
          for await (const chunk of smlxClient.streamChatCompletion({
            model: settings.model,
            messages: apiMessages,
            temperature: settings.temperature,
            max_tokens: settings.maxTokens,
            top_p: settings.topP,
            top_k: settings.topK,
            stream: true,
          })) {
            if (controller.signal.aborted) {
              break;
            }

            const content = chunk.choices[0]?.delta?.content || '';
            if (content) {
              appendStreamContent(content);
            }
          }

          endStreaming();
        } else {
          // Non-streaming request
          const response = await smlxClient.createChatCompletion({
            model: settings.model,
            messages: apiMessages,
            temperature: settings.temperature,
            max_tokens: settings.maxTokens,
            top_p: settings.topP,
            top_k: settings.topK,
            stream: false,
          });

          // Add assistant response
          addMessage(activeConversationId, {
            role: 'assistant',
            content: response.choices[0].message.content,
          });

          // Update token usage (map snake_case to camelCase)
          if (response.usage) {
            updateTokenUsage(activeConversationId, {
              promptTokens: response.usage.prompt_tokens,
              completionTokens: response.usage.completion_tokens,
              totalTokens: response.usage.total_tokens,
            });
          }
        }

        setIsLoading(false);
      } catch (err) {
        setIsLoading(false);
        endStreaming();

        if (err instanceof SMLXAPIError) {
          setError(`API Error: ${err.message}`);
        } else if (err instanceof Error) {
          if (err.name === 'AbortError') {
            setError('Generation cancelled');
          } else {
            setError(err.message);
          }
        } else {
          setError('Unknown error occurred');
        }

        console.error('Chat completion error:', err);
      } finally {
        setAbortController(null);
      }
    },
    [
      settings,
      addMessage,
      startStreaming,
      appendStreamContent,
      endStreaming,
      getConversationMessages,
      updateTokenUsage,
    ]
  );

  const cancelGeneration = useCallback(() => {
    if (abortController) {
      abortController.abort();
      setAbortController(null);
      endStreaming();
    }
  }, [abortController, endStreaming]);

  return {
    sendMessage,
    isLoading,
    error,
    cancelGeneration,
  };
}
