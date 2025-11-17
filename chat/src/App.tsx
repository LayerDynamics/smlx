import { useEffect } from 'react';
import { PanelLayout } from './components/layout/PanelLayout';
import { ConversationList } from './components/sidebar/ConversationList';
import { ChatContainer } from './components/chat/ChatContainer';
import { SettingsPanel } from './components/sidebar/SettingsPanel';
import { useChatStore } from './store/chatStore';
import { useUIStore } from './store/uiStore';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { useSettingsStore } from './store/settingsStore';
import { IconButton } from './components/common/IconButton';

function App() {
  const {
    createConversation,
    activeConversationId,
    getConversationMetadata,
    setActiveConversation,
  } = useChatStore();
  const { toggleSidebar, toggleSettings } = useUIStore();
  const { model, systemPrompt } = useSettingsStore();

  // Initialize conversation on first load
  useEffect(() => {
    if (!activeConversationId) {
      const conversations = getConversationMetadata();
      if (conversations.length > 0) {
        // If conversations exist, activate the most recent one
        const mostRecent = conversations.sort((a, b) => b.updatedAt - a.updatedAt)[0];
        setActiveConversation(mostRecent.id);
      } else {
        // Otherwise create a new conversation
        createConversation(model, systemPrompt);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Keyboard shortcuts
  useKeyboardShortcuts([
    {
      key: 'k',
      ctrl: true,
      description: 'New conversation',
      action: () => createConversation(model, systemPrompt),
    },
    {
      key: 'b',
      ctrl: true,
      description: 'Toggle sidebar',
      action: toggleSidebar,
    },
    {
      key: ',',
      ctrl: true,
      description: 'Toggle settings',
      action: toggleSettings,
    },
  ]);

  return (
    <div className="h-screen w-screen overflow-hidden bg-gray-50 dark:bg-gray-900">
      {/* Top Bar */}
      <div className="h-14 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between px-4">
        <div className="flex items-center gap-3">
          <IconButton
            icon={
              <svg
                className="w-5 h-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 6h16M4 12h16M4 18h16"
                />
              </svg>
            }
            label="Toggle sidebar"
            onClick={toggleSidebar}
          />
          <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            SMLX Chat
          </h1>
        </div>

        <div className="flex items-center gap-2">
          <IconButton
            icon={
              <svg
                className="w-5 h-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                />
              </svg>
            }
            label="Toggle settings"
            onClick={toggleSettings}
          />
        </div>
      </div>

      {/* Main Layout */}
      <div className="h-[calc(100vh-3.5rem)]">
        <PanelLayout
          sidebar={<ConversationList />}
          main={<ChatContainer />}
          settings={<SettingsPanel />}
        />
      </div>
    </div>
  );
}

export default App;
