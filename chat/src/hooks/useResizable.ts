import { useState, useCallback, useEffect, useRef } from 'react';

interface UseResizableOptions {
  initialWidth: number;
  minWidth: number;
  maxWidth: number;
  onResize?: (width: number) => void;
  direction?: 'left' | 'right';
}

interface UseResizableReturn {
  width: number;
  isResizing: boolean;
  handleMouseDown: (e: React.MouseEvent) => void;
  handleTouchStart: (e: React.TouchEvent) => void;
}

export function useResizable({
  initialWidth,
  minWidth,
  maxWidth,
  onResize,
  direction = 'right',
}: UseResizableOptions): UseResizableReturn {
  const [width, setWidth] = useState(initialWidth);
  const [isResizing, setIsResizing] = useState(false);
  const startXRef = useRef(0);
  const startWidthRef = useRef(initialWidth);

  const handleResize = useCallback(
    (clientX: number) => {
      const delta =
        direction === 'right'
          ? clientX - startXRef.current
          : startXRef.current - clientX;

      const newWidth = Math.max(
        minWidth,
        Math.min(maxWidth, startWidthRef.current + delta)
      );

      setWidth(newWidth);
      onResize?.(newWidth);
    },
    [direction, minWidth, maxWidth, onResize]
  );

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isResizing) return;
      handleResize(e.clientX);
    },
    [isResizing, handleResize]
  );

  const handleTouchMove = useCallback(
    (e: TouchEvent) => {
      if (!isResizing || e.touches.length === 0) return;
      handleResize(e.touches[0].clientX);
    },
    [isResizing, handleResize]
  );

  const handleEnd = useCallback(() => {
    setIsResizing(false);
  }, []);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      setIsResizing(true);
      startXRef.current = e.clientX;
      startWidthRef.current = width;
    },
    [width]
  );

  const handleTouchStart = useCallback(
    (e: React.TouchEvent) => {
      if (e.touches.length === 0) return;
      setIsResizing(true);
      startXRef.current = e.touches[0].clientX;
      startWidthRef.current = width;
    },
    [width]
  );

  useEffect(() => {
    if (!isResizing) return;

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleEnd);
    document.addEventListener('touchmove', handleTouchMove);
    document.addEventListener('touchend', handleEnd);

    // Prevent text selection while resizing
    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'col-resize';

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleEnd);
      document.removeEventListener('touchmove', handleTouchMove);
      document.removeEventListener('touchend', handleEnd);

      document.body.style.userSelect = '';
      document.body.style.cursor = '';
    };
  }, [isResizing, handleMouseMove, handleTouchMove, handleEnd]);

  return {
    width,
    isResizing,
    handleMouseDown,
    handleTouchStart,
  };
}
