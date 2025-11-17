import type { ReactNode } from 'react';
import { Panel } from './Panel';
import { PanelResizeHandle } from './PanelResizeHandle';
import { useResizable } from '../../hooks/useResizable';
import { useUIStore } from '../../store/uiStore';

interface PanelLayoutProps {
  sidebar: ReactNode;
  main: ReactNode;
  settings: ReactNode;
}

export function PanelLayout({ sidebar, main, settings }: PanelLayoutProps) {
  const {
    sidebar: sidebarState,
    settings: settingsState,
    setSidebarWidth,
    setSettingsWidth,
  } = useUIStore();

  const sidebarResizable = useResizable({
    initialWidth: sidebarState.width,
    minWidth: sidebarState.minWidth,
    maxWidth: sidebarState.maxWidth,
    onResize: setSidebarWidth,
    direction: 'right',
  });

  const settingsResizable = useResizable({
    initialWidth: settingsState.width,
    minWidth: settingsState.minWidth,
    maxWidth: settingsState.maxWidth,
    onResize: setSettingsWidth,
    direction: 'left',
  });

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      {/* Sidebar Panel */}
      {!sidebarState.isCollapsed && (
        <>
          <Panel width={sidebarResizable.width} className="flex-shrink-0">
            {sidebar}
          </Panel>
          <PanelResizeHandle
            onMouseDown={sidebarResizable.handleMouseDown}
            onTouchStart={sidebarResizable.handleTouchStart}
            isResizing={sidebarResizable.isResizing}
          />
        </>
      )}

      {/* Main Content Panel */}
      <div className="flex-1 h-full overflow-hidden bg-white dark:bg-gray-900">
        {main}
      </div>

      {/* Settings Panel */}
      {!settingsState.isCollapsed && (
        <>
          <PanelResizeHandle
            onMouseDown={settingsResizable.handleMouseDown}
            onTouchStart={settingsResizable.handleTouchStart}
            isResizing={settingsResizable.isResizing}
          />
          <Panel width={settingsResizable.width} className="flex-shrink-0">
            {settings}
          </Panel>
        </>
      )}
    </div>
  );
}
