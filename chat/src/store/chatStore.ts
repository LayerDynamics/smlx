import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type {
  Conversation,
  Message,
  StreamingState,
  TokenUsage,
  ConversationMetadata,
} from '../types/chat';
import { nanoid } from 'nanoid';

interface ChatStore {
  conversations: Record<string, Conversation>;
  activeConversationId: string | null;
  streamingState: StreamingState;
  tokenUsage: Record<string, TokenUsage>;

  // Conversation actions
  createConversation: (model?: string, systemPrompt?: string) => string;
  deleteConversation: (id: string) => void;
  setActiveConversation: (id: string) => void;
  updateConversationTitle: (id: string, title: string) => void;
  getConversationMetadata: () => ConversationMetadata[];

  // Message actions
  addMessage: (conversationId: string, message: Omit<Message, 'id' | 'timestamp' | 'conversationId'>) => string;
  updateMessage: (messageId: string, content: string) => void;
  deleteMessage: (messageId: string) => void;
  clearMessages: (conversationId: string) => void;

  // Streaming actions
  startStreaming: (conversationId: string, messageId: string) => void;
  appendStreamContent: (content: string) => void;
  endStreaming: () => void;

  // Token usage
  updateTokenUsage: (conversationId: string, usage: Omit<TokenUsage, 'conversationId'>) => void;

  // Utility
  getActiveConversation: () => Conversation | null;
  getConversationMessages: (conversationId: string) => Message[];
}

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      conversations: {},
      activeConversationId: null,
      streamingState: {
        isStreaming: false,
        conversationId: null,
        content: '',
        messageId: null,
      },
      tokenUsage: {},

      // Conversation actions
      createConversation: (model = 'mlx-community/SmolLM2-135M-Instruct', systemPrompt) => {
        const id = nanoid();
        const now = Date.now();
        const conversation: Conversation = {
          id,
          title: 'New Chat',
          messages: [],
          createdAt: now,
          updatedAt: now,
          model,
          systemPrompt,
        };

        set((state) => ({
          conversations: { ...state.conversations, [id]: conversation },
          activeConversationId: id,
        }));

        return id;
      },

      deleteConversation: (id) => {
        set((state) => {
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
          const { [id]: _, ...remainingConversations } = state.conversations;
          const conversationIds = Object.keys(remainingConversations);
          const newActiveId =
            state.activeConversationId === id && conversationIds.length > 0
              ? conversationIds[0]
              : state.activeConversationId;

          return {
            conversations: remainingConversations,
            activeConversationId:
              newActiveId === id ? null : newActiveId,
          };
        });
      },

      setActiveConversation: (id) => {
        set({ activeConversationId: id });
      },

      updateConversationTitle: (id, title) => {
        set((state) => ({
          conversations: {
            ...state.conversations,
            [id]: {
              ...state.conversations[id],
              title,
              updatedAt: Date.now(),
            },
          },
        }));
      },

      getConversationMetadata: () => {
        const state = get();
        return Object.values(state.conversations).map((conv) => ({
          id: conv.id,
          title: conv.title,
          createdAt: conv.createdAt,
          updatedAt: conv.updatedAt,
          messageCount: conv.messages.length,
          model: conv.model,
        }));
      },

      // Message actions
      addMessage: (conversationId, message) => {
        const messageId = nanoid();
        const newMessage: Message = {
          ...message,
          id: messageId,
          timestamp: Date.now(),
          conversationId,
        };

        set((state) => {
          const conversation = state.conversations[conversationId];
          if (!conversation) return state;

          const messages = [...conversation.messages, newMessage];

          // Auto-generate title from first user message
          let title = conversation.title;
          if (title === 'New Chat' && message.role === 'user' && messages.length <= 2) {
            title = message.content.slice(0, 50) + (message.content.length > 50 ? '...' : '');
          }

          return {
            conversations: {
              ...state.conversations,
              [conversationId]: {
                ...conversation,
                messages,
                title,
                updatedAt: Date.now(),
              },
            },
          };
        });

        return messageId;
      },

      updateMessage: (messageId, content) => {
        set((state) => {
          const conversations = { ...state.conversations };

          for (const conv of Object.values(conversations)) {
            const messageIndex = conv.messages.findIndex((m) => m.id === messageId);
            if (messageIndex !== -1) {
              const updatedMessages = [...conv.messages];
              updatedMessages[messageIndex] = {
                ...updatedMessages[messageIndex],
                content,
              };

              conversations[conv.id] = {
                ...conv,
                messages: updatedMessages,
                updatedAt: Date.now(),
              };
              break;
            }
          }

          return { conversations };
        });
      },

      deleteMessage: (messageId) => {
        set((state) => {
          const conversations = { ...state.conversations };

          for (const conv of Object.values(conversations)) {
            const messageIndex = conv.messages.findIndex((m) => m.id === messageId);
            if (messageIndex !== -1) {
              conversations[conv.id] = {
                ...conv,
                messages: conv.messages.filter((m) => m.id !== messageId),
                updatedAt: Date.now(),
              };
              break;
            }
          }

          return { conversations };
        });
      },

      clearMessages: (conversationId) => {
        set((state) => ({
          conversations: {
            ...state.conversations,
            [conversationId]: {
              ...state.conversations[conversationId],
              messages: [],
              updatedAt: Date.now(),
            },
          },
        }));
      },

      // Streaming actions
      startStreaming: (conversationId, messageId) => {
        set({
          streamingState: {
            isStreaming: true,
            conversationId,
            content: '',
            messageId,
          },
        });
      },

      appendStreamContent: (content) => {
        set((state) => ({
          streamingState: {
            ...state.streamingState,
            content: state.streamingState.content + content,
          },
        }));
      },

      endStreaming: () => {
        const state = get();
        const { streamingState } = state;

        if (streamingState.messageId && streamingState.content) {
          get().updateMessage(streamingState.messageId, streamingState.content);
        }

        set({
          streamingState: {
            isStreaming: false,
            conversationId: null,
            content: '',
            messageId: null,
          },
        });
      },

      // Token usage
      updateTokenUsage: (conversationId, usage) => {
        set((state) => ({
          tokenUsage: {
            ...state.tokenUsage,
            [conversationId]: {
              ...usage,
              conversationId,
            },
          },
        }));
      },

      // Utility
      getActiveConversation: () => {
        const state = get();
        return state.activeConversationId
          ? state.conversations[state.activeConversationId] || null
          : null;
      },

      getConversationMessages: (conversationId) => {
        const state = get();
        return state.conversations[conversationId]?.messages || [];
      },
    }),
    {
      name: 'smlx-chat-data',
      partialize: (state) => ({
        conversations: state.conversations,
        activeConversationId: state.activeConversationId,
        tokenUsage: state.tokenUsage,
        // Don't persist streaming state
      }),
    }
  )
);
