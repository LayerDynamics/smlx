// Model-related types

// Re-export ModelInfo from API types
export type { ModelInfo } from './api';

export type ModelType = 'language' | 'vision' | 'audio' | 'embedding';

export interface Model {
  id: string;
  name: string;
  type: ModelType;
  description: string;
  parameters: string; // e.g., "135M", "360M"
  capabilities: string[];
  isLoaded: boolean;
  maxTokens: number;
}

export interface ModelCapabilities {
  supportsStreaming: boolean;
  supportsVision: boolean;
  supportsAudio: boolean;
  supportsChat: boolean;
  supportsCompletion: boolean;
}

export const AVAILABLE_MODELS: Model[] = [
  {
    id: 'mlx-community/SmolLM2-135M-Instruct',
    name: 'SmolLM2-135M',
    type: 'language',
    description: 'Lightweight language model (135M parameters)',
    parameters: '135M',
    capabilities: ['text-generation', 'chat', 'streaming'],
    isLoaded: false,
    maxTokens: 2048,
  },
  {
    id: 'mlx-community/SmolLM2-360M-Instruct',
    name: 'SmolLM2-360M',
    type: 'language',
    description: 'Enhanced language model (360M parameters)',
    parameters: '360M',
    capabilities: ['text-generation', 'chat', 'streaming'],
    isLoaded: false,
    maxTokens: 2048,
  },
];

export const DEFAULT_MODEL = 'mlx-community/SmolLM2-135M-Instruct';
