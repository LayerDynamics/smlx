import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { UIState, Theme, PanelState } from '../types/ui';

interface UIStore extends UIState {
  // Actions
  setSidebarWidth: (width: number) => void;
  setSettingsWidth: (width: number) => void;
  toggleSidebar: () => void;
  toggleSettings: () => void;
  setTheme: (theme: Theme) => void;
  collapseSidebar: () => void;
  expandSidebar: () => void;
  collapseSettings: () => void;
  expandSettings: () => void;
}

const DEFAULT_SIDEBAR: PanelState = {
  isCollapsed: false,
  width: 280,
  minWidth: 200,
  maxWidth: 500,
};

const DEFAULT_SETTINGS: PanelState = {
  isCollapsed: false,
  width: 300,
  minWidth: 250,
  maxWidth: 450,
};

export const useUIStore = create<UIStore>()(
  persist(
    (set) => ({
      // Initial state
      sidebar: DEFAULT_SIDEBAR,
      settings: DEFAULT_SETTINGS,
      theme: 'system' as Theme,

      // Actions
      setSidebarWidth: (width) =>
        set((state) => ({
          sidebar: {
            ...state.sidebar,
            width: Math.max(
              state.sidebar.minWidth,
              Math.min(state.sidebar.maxWidth, width)
            ),
          },
        })),

      setSettingsWidth: (width) =>
        set((state) => ({
          settings: {
            ...state.settings,
            width: Math.max(
              state.settings.minWidth,
              Math.min(state.settings.maxWidth, width)
            ),
          },
        })),

      toggleSidebar: () =>
        set((state) => ({
          sidebar: {
            ...state.sidebar,
            isCollapsed: !state.sidebar.isCollapsed,
          },
        })),

      toggleSettings: () =>
        set((state) => ({
          settings: {
            ...state.settings,
            isCollapsed: !state.settings.isCollapsed,
          },
        })),

      collapseSidebar: () =>
        set((state) => ({
          sidebar: {
            ...state.sidebar,
            isCollapsed: true,
          },
        })),

      expandSidebar: () =>
        set((state) => ({
          sidebar: {
            ...state.sidebar,
            isCollapsed: false,
          },
        })),

      collapseSettings: () =>
        set((state) => ({
          settings: {
            ...state.settings,
            isCollapsed: true,
          },
        })),

      expandSettings: () =>
        set((state) => ({
          settings: {
            ...state.settings,
            isCollapsed: false,
          },
        })),

      setTheme: (theme) => set({ theme }),
    }),
    {
      name: 'smlx-chat-ui',
    }
  )
);
