import { create } from 'zustand';
import type { Model, ModelInfo } from '../types/models';
import { AVAILABLE_MODELS } from '../types/models';

interface ModelStore {
  availableModels: Model[];
  loadedModels: Set<string>;
  isLoading: boolean;
  error: string | null;

  // Actions
  fetchModels: () => Promise<void>;
  setLoadedModels: (modelIds: string[]) => void;
  addLoadedModel: (modelId: string) => void;
  removeLoadedModel: (modelId: string) => void;
  getModelById: (modelId: string) => Model | undefined;
}

export const useModelStore = create<ModelStore>((set, get) => ({
  availableModels: AVAILABLE_MODELS,
  loadedModels: new Set(),
  isLoading: false,
  error: null,

  // Actions
  fetchModels: async () => {
    set({ isLoading: true, error: null });

    try {
      const response = await fetch('http://localhost:8000/v1/models');

      if (!response.ok) {
        throw new Error(`Failed to fetch models: ${response.statusText}`);
      }

      const data: { data: ModelInfo[] } = await response.json();

      // Map API models to our Model type
      const models: Model[] = data.data.map((apiModel) => {
        // Find matching model in our available models or create a new one
        const existingModel = AVAILABLE_MODELS.find(
          (m) => m.id === apiModel.id
        );

        if (existingModel) {
          return { ...existingModel, isLoaded: true };
        }

        // Create a new model entry for unknown models
        return {
          id: apiModel.id,
          name: apiModel.id.split('/').pop() || apiModel.id,
          type: 'language' as const,
          description: 'Model from SMLX server',
          parameters: 'Unknown',
          capabilities: ['text-generation'],
          isLoaded: true,
          maxTokens: 2048,
        };
      });

      set({
        availableModels: models,
        loadedModels: new Set(models.map((m) => m.id)),
        isLoading: false,
      });
    } catch (error) {
      set({
        isLoading: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    }
  },

  setLoadedModels: (modelIds) => {
    set({ loadedModels: new Set(modelIds) });
  },

  addLoadedModel: (modelId) => {
    set((state) => {
      const newLoadedModels = new Set(state.loadedModels);
      newLoadedModels.add(modelId);
      return { loadedModels: newLoadedModels };
    });
  },

  removeLoadedModel: (modelId) => {
    set((state) => {
      const newLoadedModels = new Set(state.loadedModels);
      newLoadedModels.delete(modelId);
      return { loadedModels: newLoadedModels };
    });
  },

  getModelById: (modelId) => {
    return get().availableModels.find((m) => m.id === modelId);
  },
}));
