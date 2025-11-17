interface PanelResizeHandleProps {
  onMouseDown: (e: React.MouseEvent) => void;
  onTouchStart: (e: React.TouchEvent) => void;
  isResizing?: boolean;
}

export function PanelResizeHandle({
  onMouseDown,
  onTouchStart,
  isResizing = false,
}: PanelResizeHandleProps) {
  return (
    <div
      className={`
        relative w-1 hover:w-1.5
        bg-gray-200 dark:bg-gray-700
        hover:bg-blue-500 dark:hover:bg-blue-600
        cursor-col-resize
        transition-all duration-150
        group
        ${isResizing ? 'w-1.5 bg-blue-500 dark:bg-blue-600' : ''}
      `}
      onMouseDown={onMouseDown}
      onTouchStart={onTouchStart}
    >
      {/* Visual indicator */}
      <div
        className={`
          absolute inset-y-0 -left-1 -right-1
          group-hover:bg-blue-500/10
          ${isResizing ? 'bg-blue-500/10' : ''}
        `}
      />
    </div>
  );
}
