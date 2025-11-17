// UI-specific types

export type Theme = 'light' | 'dark' | 'system';

export interface PanelState {
  isCollapsed: boolean;
  width: number;
  minWidth: number;
  maxWidth: number;
}

export interface UIState {
  sidebar: PanelState;
  settings: PanelState;
  theme: Theme;
}

export interface KeyboardShortcut {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  meta?: boolean;
  description: string;
  action: () => void;
}

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
export type ButtonSize = 'sm' | 'md' | 'lg';

export interface SelectOption<T = string> {
  value: T;
  label: string;
  disabled?: boolean;
}

export interface TooltipProps {
  content: string;
  position?: 'top' | 'right' | 'bottom' | 'left';
  delay?: number;
}

export interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
}

// Drag and resize types
export interface DragState {
  isDragging: boolean;
  startX: number;
  startY: number;
  startWidth: number;
  startHeight: number;
}

export interface ResizeHandleProps {
  direction: 'horizontal' | 'vertical';
  onResizeStart: (e: React.MouseEvent) => void;
  onResize: (delta: number) => void;
  onResizeEnd: () => void;
}
