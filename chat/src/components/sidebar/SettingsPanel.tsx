import { useSettingsStore } from '../../store/settingsStore';
import { AVAILABLE_MODELS } from '../../types/models';
import { Select } from '../common/Select';
import { Textarea } from '../common/Input';

export function SettingsPanel() {
  const settings = useSettingsStore();

  const modelOptions = AVAILABLE_MODELS.map((model) => ({
    value: model.id,
    label: `${model.name} (${model.parameters})`,
  }));

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
          Settings
        </h2>
      </div>

      {/* Settings Form */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Model Selection */}
        <div>
          <Select
            label="Model"
            value={settings.model}
            options={modelOptions}
            onChange={(value) => settings.setModel(value)}
          />
        </div>

        {/* System Prompt */}
        <div>
          <Textarea
            label="System Prompt"
            value={settings.systemPrompt}
            onChange={(e) => settings.setSystemPrompt(e.target.value)}
            rows={3}
            helperText="Instructions for the AI assistant"
          />
        </div>

        {/* Temperature */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Temperature: {settings.temperature.toFixed(2)}
          </label>
          <input
            type="range"
            min="0"
            max="2"
            step="0.1"
            value={settings.temperature}
            onChange={(e) => settings.setTemperature(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
          />
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Higher values make output more random
          </p>
        </div>

        {/* Max Tokens */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Max Tokens: {settings.maxTokens}
          </label>
          <input
            type="range"
            min="64"
            max="2048"
            step="64"
            value={settings.maxTokens}
            onChange={(e) => settings.setMaxTokens(parseInt(e.target.value))}
            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
          />
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Maximum length of generated response
          </p>
        </div>

        {/* Top P */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Top P: {settings.topP.toFixed(2)}
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={settings.topP}
            onChange={(e) => settings.setTopP(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer"
          />
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Nucleus sampling threshold
          </p>
        </div>

        {/* Streaming Toggle */}
        <div className="flex items-center justify-between">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Streaming
            </label>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Stream responses in real-time
            </p>
          </div>
          <button
            type="button"
            role="switch"
            aria-checked={settings.streamEnabled}
            onClick={settings.toggleStream}
            className={`
              relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent
              transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
              ${settings.streamEnabled ? 'bg-blue-600' : 'bg-gray-200 dark:bg-gray-700'}
            `}
          >
            <span
              aria-hidden="true"
              className={`
                pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0
                transition duration-200 ease-in-out
                ${settings.streamEnabled ? 'translate-x-5' : 'translate-x-0'}
              `}
            />
          </button>
        </div>
      </div>
    </div>
  );
}
