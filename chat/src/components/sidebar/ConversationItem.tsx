import type { ConversationMetadata } from '../../types/chat';
import { IconButton } from '../common/IconButton';

interface ConversationItemProps {
  conversation: ConversationMetadata;
  isActive: boolean;
  onClick: () => void;
  onDelete: () => void;
}

export function ConversationItem({
  conversation,
  isActive,
  onClick,
  onDelete,
}: ConversationItemProps) {
  return (
    <div
      className={`
        group flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer
        transition-colors duration-150
        ${
          isActive
            ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-900 dark:text-blue-100'
            : 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300'
        }
      `}
      onClick={onClick}
    >
      <div className="flex-1 min-w-0">
        <div className="font-medium text-sm truncate">{conversation.title}</div>
        <div className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-2">
          <span>{conversation.messageCount} messages</span>
          <span>·</span>
          <span>{new Date(conversation.updatedAt).toLocaleDateString()}</span>
        </div>
      </div>

      <IconButton
        icon={
          <svg
            className="w-4 h-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
            />
          </svg>
        }
        label="Delete conversation"
        size="sm"
        className="opacity-0 group-hover:opacity-100 transition-opacity"
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
      />
    </div>
  );
}
