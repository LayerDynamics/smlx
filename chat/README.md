# SMLX Chat Module

A modern, panel-based chat interface built with React 19, TypeScript, Tailwind CSS 4.0, Zustand, and the React Compiler. This chat module integrates with the SMLX server API to provide a seamless chat experience with small MLX models.

## 🎯 Features

### Core Features
- ✅ **Real-time streaming** - Stream responses in real-time via Server-Sent Events (SSE)
- ✅ **Multi-conversation support** - Manage multiple conversations with automatic title generation
- ✅ **Message history persistence** - All conversations saved to localStorage
- ✅ **Resizable panels** - Drag to resize sidebar and settings panels
- ✅ **Keyboard shortcuts** - Quick actions with keyboard shortcuts
- ✅ **Dark mode** - Automatic dark mode support
- ✅ **Model switching** - Switch between different SMLX models
- ✅ **Configurable settings** - Adjust temperature, max tokens, top-p, and system prompt

### Technical Features
- 🚀 **React Compiler** - Automatic optimization (no manual memoization needed)
- ⚡ **Tailwind CSS 4.0** - 5x faster builds with CSS-first configuration
- 🐻 **Zustand** - Lightweight state management (~1KB)
- 📦 **TypeScript** - Fully type-safe with strict mode
- 🎨 **Custom components** - No component frameworks, fully custom UI
- 🔄 **SSE streaming** - Efficient real-time communication

## 🚀 Quick Start

### Prerequisites

1. **SMLX Server** must be running on `http://localhost:8000`
2. Node.js >= 18
3. npm or yarn

### Installation

```bash
# Navigate to the chat directory
cd chat

# Install dependencies (if not already done)
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:5173`

## 🏗️ Architecture

### Project Structure

```
src/
├── components/
│   ├── layout/          # Panel system (Panel, PanelLayout, PanelResizeHandle)
│   ├── chat/            # Chat components (ChatContainer, MessageList, Message, ChatInput)
│   ├── sidebar/         # Sidebar components (ConversationList, SettingsPanel)
│   └── common/          # Reusable UI components (Button, Input, Select, etc.)
├── store/               # Zustand stores
│   ├── chatStore.ts     # Chat state (conversations, messages, streaming)
│   ├── uiStore.ts       # UI state (panels, theme)
│   ├── settingsStore.ts # Settings state (model config)
│   └── modelStore.ts    # Model management
├── hooks/               # Custom React hooks
│   ├── useChatCompletion.ts  # Chat API integration
│   ├── useResizable.ts       # Panel resizing logic
│   └── useKeyboardShortcuts.ts # Keyboard shortcuts
├── services/            # API and services
│   ├── api.ts           # SMLX API client
│   └── storage.ts       # LocalStorage wrapper
└── types/               # TypeScript type definitions
    ├── api.ts           # API types (OpenAI-compatible)
    ├── chat.ts          # Chat types
    ├── models.ts        # Model types
    └── ui.ts            # UI types
```

### State Management (Zustand)

The application uses four Zustand stores:

1. **Chat Store** - Manages conversations, messages, and streaming state
2. **UI Store** - Manages panel sizes, collapsed state, and theme
3. **Settings Store** - Manages model configuration and parameters
4. **Model Store** - Manages available models and loaded models

All stores persist to localStorage using Zustand's persist middleware.

### Panel System

The custom panel system provides:
- **Resizable panels** - Drag handles between panels to resize
- **Collapsible panels** - Hide/show sidebar and settings
- **Min/max constraints** - Panels have minimum and maximum widths
- **Persistent sizes** - Panel sizes saved to localStorage

## 📖 Usage

### Keyboard Shortcuts

- **Ctrl+K** - Create new conversation
- **Ctrl+B** - Toggle sidebar
- **Ctrl+,** - Toggle settings panel
- **Enter** - Send message
- **Shift+Enter** - New line in message input

### Creating a Conversation

1. Click "New Chat" button in the sidebar
2. Or use the keyboard shortcut **Ctrl+K**
3. Start typing in the message input

### Sending Messages

1. Type your message in the input field
2. Press **Enter** to send (or click the Send button)
3. Use **Shift+Enter** for multi-line messages

### Configuring Settings

1. Open settings panel (Ctrl+,)
2. Select a model from the dropdown
3. Adjust temperature, max tokens, and top-p sliders
4. Customize the system prompt
5. Toggle streaming on/off

### Managing Conversations

- **View conversations** - All conversations listed in the sidebar
- **Switch conversations** - Click any conversation to switch
- **Delete conversation** - Hover over conversation and click trash icon
- **Auto-title** - First user message becomes the conversation title

## 🔧 Configuration

### API Endpoint

The SMLX API endpoint is configured in `src/services/api.ts`:

```typescript
const baseURL = 'http://localhost:8000/v1';
```

To change the endpoint, edit this value or set an environment variable.

### Available Models

Models are defined in `src/types/models.ts`:

```typescript
export const AVAILABLE_MODELS: Model[] = [
  {
    id: 'mlx-community/SmolLM2-135M-Instruct',
    name: 'SmolLM2-135M',
    type: 'language',
    parameters: '135M',
    // ...
  },
  {
    id: 'mlx-community/SmolLM2-360M-Instruct',
    name: 'SmolLM2-360M',
    type: 'language',
    parameters: '360M',
    // ...
  },
];
```

Add more models by adding entries to this array.

### Default Settings

Default settings are defined in `src/store/settingsStore.ts`:

```typescript
const DEFAULT_SETTINGS: ChatSettings = {
  model: 'mlx-community/SmolLM2-135M-Instruct',
  temperature: 0.7,
  maxTokens: 1024,
  topP: 1.0,
  topK: null,
  systemPrompt: 'You are a helpful assistant.',
  streamEnabled: true,
};
```

## 🎨 Customization

### Styling

All styles use Tailwind CSS 4.0. Customize colors, spacing, and other design tokens in `src/index.css`:

```css
@theme {
  --color-primary-500: #3b82f6;
  --spacing-panel-min: 200px;
  --spacing-panel-max: 600px;
  /* ... */
}
```

### Panel Sizes

Default panel sizes are defined in `src/store/uiStore.ts`:

```typescript
const DEFAULT_SIDEBAR: PanelState = {
  isCollapsed: false,
  width: 280,
  minWidth: 200,
  maxWidth: 500,
};
```

## 🐛 Troubleshooting

### Connection Issues

**Problem**: "Failed to fetch" or connection errors

**Solution**:
1. Ensure SMLX server is running: `python -m smlx.server.app`
2. Check server is accessible at `http://localhost:8000`
3. Verify CORS is configured properly on the server

### Streaming Not Working

**Problem**: Messages appear all at once instead of streaming

**Solution**:
1. Enable streaming in settings panel
2. Check that the server supports SSE streaming
3. Verify browser supports EventSource API

### State Not Persisting

**Problem**: Conversations lost after refresh

**Solution**:
1. Check browser's localStorage is enabled
2. Clear localStorage if corrupted: `localStorage.clear()`
3. Check browser console for persistence errors

## 🚀 Building for Production

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

The built files will be in the `dist/` directory.

## 📝 API Integration

The chat module integrates with the SMLX server using OpenAI-compatible endpoints:

### Chat Completions (Streaming)

```typescript
POST /v1/chat/completions
{
  "model": "mlx-community/SmolLM2-135M-Instruct",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 1024
}
```

### List Models

```typescript
GET /v1/models
```

### Health Check

```typescript
GET /health
```

## 🔒 Security

- No authentication required for local development
- All data stored in browser's localStorage
- No data sent to external services
- CORS configured for localhost only

## 🤝 Contributing

To add new features:

1. **Add types** - Define TypeScript interfaces in `src/types/`
2. **Create store** - Add Zustand store if needed in `src/store/`
3. **Build components** - Create React components in `src/components/`
4. **Add hooks** - Create custom hooks in `src/hooks/`
5. **Integrate** - Wire up in `App.tsx`

## 📄 License

This chat module is part of the SMLX project.

## 🙏 Acknowledgments

Built with:
- [React 19](https://react.dev/) - UI framework
- [React Compiler](https://react.dev/learn/react-compiler) - Automatic optimization
- [TypeScript](https://www.typescriptlang.org/) - Type safety
- [Tailwind CSS 4.0](https://tailwindcss.com/) - Styling
- [Zustand](https://github.com/pmndrs/zustand) - State management
- [Vite](https://vitejs.dev/) - Build tool
- [nanoid](https://github.com/ai/nanoid) - ID generation

---

**Happy Chatting! 🎉**
