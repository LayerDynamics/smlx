import type { ReactNode } from 'react';

interface PanelProps {
  children: ReactNode;
  width?: number | string;
  className?: string;
  isCollapsed?: boolean;
}

export function Panel({ children, width, className = '', isCollapsed = false }: PanelProps) {
  if (isCollapsed) {
    return null;
  }

  return (
    <div
      className={`
        h-full overflow-hidden flex flex-col
        bg-white dark:bg-gray-900
        border-r border-gray-200 dark:border-gray-700
        ${className}
      `}
      style={{ width: typeof width === 'number' ? `${width}px` : width }}
    >
      {children}
    </div>
  );
}
