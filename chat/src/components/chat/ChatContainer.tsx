import { ChatList } from './ChatList';
import { ChatComposer } from './ChatComposer';

export function ChatContainer() {
  return (
    <div className="flex flex-col h-full bg-white dark:bg-gray-900">
      {/* Message display area */}
      <ChatList />

      {/* Message input area */}
      <ChatComposer />
    </div>
  );
}
