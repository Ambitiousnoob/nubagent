import React, { useEffect, useState } from 'react';
import { Input } from '../UI/Input.jsx';
import { Button } from '../UI/Button.jsx';
import { useSettingsStore } from '../../store/useSettingsStore.js';
import { useToast } from '../UI/ToastProvider.jsx';
import { Key, Eye, EyeOff, Check, Save } from 'lucide-react';

/**
 * ApiKeyForm Component
 * API key management form
 */
export function ApiKeyForm() {
  const { setApiKey, removeApiKey, getApiKey } = useSettingsStore();
  const { success, error } = useToast();
  const [showKey, setShowKey] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const [provider, setProvider] = useState('default');

  const providers = [
    { id: 'default', label: 'Default' },
    { id: 'openrouter', label: 'OpenRouter' },
    { id: 'tavily', label: 'Tavily Search' },
    { id: 'serper', label: 'Serper Search' },
    { id: 'brave', label: 'Brave Search' },
    { id: 'jina', label: 'Jina Search' },
    { id: 'openai', label: 'OpenAI' },
    { id: 'anthropic', label: 'Anthropic' },
    { id: 'google', label: 'Google' },
  ];

  const activeKey = getApiKey(provider);
  const hasKey = Boolean(activeKey);

  useEffect(() => {
    setShowKey(false);
    setInputValue('');
  }, [provider]);

  const handleSave = () => {
    if (!inputValue.trim()) {
      error('API Key Required', 'Please enter a valid API key');
      return;
    }

    setApiKey(inputValue.trim(), provider);
    setInputValue('');
    success('API Key Saved', 'Your API key has been saved securely');
  };

  const handleRemove = () => {
    removeApiKey(provider);
    success('API Key Removed', 'Your API key has been removed');
  };

  const maskKey = (key) => {
    if (!key) return '';
    if (key.length < 10) return '••••••••';
    return `${key.slice(0, 4)}••••••••••••${key.slice(-4)}`;
  };

  return (
    <div className="api-key-form">
      <div className="api-key-form__header">
        <Key size={20} />
        <h3 className="api-key-form__title">API Keys</h3>
      </div>
      <p className="api-key-form__description">
        Manage your API keys for different providers. Keys are stored locally in your browser.
      </p>

      <div className="api-key-form__provider">
        <label className="api-key-form__label">Provider</label>
        <select
          className="api-key-form__select"
          value={provider}
          onChange={(e) => setProvider(e.target.value)}
        >
          {providers.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </select>
      </div>

      {hasKey ? (
        <div className="api-key-form__existing">
          <div className="api-key-form__key-display">
            <span className="api-key-form__key-masked">
              {maskKey(activeKey)}
            </span>
            <button
              className="api-key-form__key-toggle"
              onClick={() => setShowKey(!showKey)}
              aria-label={showKey ? 'Hide key' : 'Show key'}
            >
              {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
            </button>
          </div>
          {showKey && (
            <div className="api-key-form__key-visible">
              {activeKey}
            </div>
          )}
          <div className="api-key-form__actions">
            <Button variant="danger" size="sm" onClick={handleRemove}>
              Remove Key
            </Button>
          </div>
        </div>
      ) : (
        <div className="api-key-form__new">
          <Input
            type={showKey ? 'text' : 'password'}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Enter your API key"
            label="API Key"
            icon={<Key size={16} />}
          />
          <div className="api-key-form__actions">
            <Button
              variant="primary"
              onClick={handleSave}
              disabled={!inputValue.trim()}
            >
              <Save size={16} />
              Save Key
            </Button>
          </div>
        </div>
      )}

      <div className="api-key-form__info">
        <h4>Where to get API keys</h4>
        <ul>
          <li>
            <strong>Default:</strong> Configure in your environment variables
          </li>
          <li>
            <strong>OpenRouter:</strong>{' '}
            <a href="https://openrouter.ai/keys" target="_blank" rel="noopener noreferrer">
              openrouter.ai
            </a>
          </li>
          <li>
            <strong>Tavily:</strong>{' '}
            <a href="https://app.tavily.com/home" target="_blank" rel="noopener noreferrer">
              app.tavily.com
            </a>
          </li>
          <li>
            <strong>Serper:</strong>{' '}
            <a href="https://serper.dev" target="_blank" rel="noopener noreferrer">
              serper.dev
            </a>
          </li>
          <li>
            <strong>Brave Search:</strong>{' '}
            <a href="https://api.search.brave.com" target="_blank" rel="noopener noreferrer">
              api.search.brave.com
            </a>
          </li>
          <li>
            <strong>Jina:</strong>{' '}
            <a href="https://jina.ai" target="_blank" rel="noopener noreferrer">
              jina.ai
            </a>
          </li>
          <li>
            <strong>OpenAI:</strong>{' '}
            <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer">
              platform.openai.com
            </a>
          </li>
          <li>
            <strong>Anthropic:</strong>{' '}
            <a href="https://console.anthropic.com/settings/keys" target="_blank" rel="noopener noreferrer">
              console.anthropic.com
            </a>
          </li>
          <li>
            <strong>Google:</strong>{' '}
            <a href="https://makersuite.google.com/app/apikey" target="_blank" rel="noopener noreferrer">
              makersuite.google.com
            </a>
          </li>
        </ul>
      </div>
    </div>
  );
}

export default ApiKeyForm;
