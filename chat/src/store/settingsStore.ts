import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { ChatSettings } from '../types/chat';
import { DEFAULT_MODEL } from '../types/models';

interface SettingsStore extends ChatSettings {
  // Actions
  updateSettings: (settings: Partial<ChatSettings>) => void;
  resetToDefaults: () => void;
  setModel: (model: string) => void;
  setTemperature: (temperature: number) => void;
  setMaxTokens: (maxTokens: number) => void;
  setTopP: (topP: number) => void;
  setSystemPrompt: (systemPrompt: string) => void;
  toggleStream: () => void;
}

const DEFAULT_SETTINGS: ChatSettings = {
  model: DEFAULT_MODEL,
  temperature: 0.7,
  maxTokens: 1024,
  topP: 1.0,
  topK: null,
  systemPrompt: 'You are a helpful assistant.',
  streamEnabled: true,
};

export const useSettingsStore = create<SettingsStore>()(
  persist(
    (set) => ({
      // Initial state
      ...DEFAULT_SETTINGS,

      // Actions
      updateSettings: (settings) =>
        set((state) => ({
          ...state,
          ...settings,
        })),

      resetToDefaults: () => set(DEFAULT_SETTINGS),

      setModel: (model) => set({ model }),

      setTemperature: (temperature) =>
        set({ temperature: Math.max(0, Math.min(2, temperature)) }),

      setMaxTokens: (maxTokens) =>
        set({ maxTokens: Math.max(1, Math.min(4096, maxTokens)) }),

      setTopP: (topP) => set({ topP: Math.max(0, Math.min(1, topP)) }),

      setSystemPrompt: (systemPrompt) => set({ systemPrompt }),

      toggleStream: () => set((state) => ({ streamEnabled: !state.streamEnabled })),
    }),
    {
      name: 'smlx-chat-settings',
    }
  )
);
