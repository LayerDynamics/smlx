// Chat-specific types for the application

export interface Message {
  id: string;
  role: 'system' | 'user' | 'assistant';
  content: string;
  timestamp: number;
  conversationId: string;
  isStreaming?: boolean;
  tokenCount?: number;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
  model: string;
  systemPrompt?: string;
}

export interface ConversationMetadata {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messageCount: number;
  model: string;
}

export interface StreamingState {
  isStreaming: boolean;
  conversationId: string | null;
  content: string;
  messageId: string | null;
}

export interface ChatSettings {
  model: string;
  temperature: number;
  maxTokens: number;
  topP: number;
  topK: number | null;
  systemPrompt: string;
  streamEnabled: boolean;
}

export interface TokenUsage {
  conversationId: string;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}
