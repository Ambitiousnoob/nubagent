import React from 'react';
import { useSettingsStore } from '../../store/useSettingsStore.js';
import { Cpu, Sparkles } from 'lucide-react';

/**
 * ModelSelector Component
 * AI model selection
 */
export function ModelSelector() {
  const { selectedModel, setSelectedModel } = useSettingsStore();

  const models = [
    {
      id: 'nub-agent',
      name: 'Nub Agent',
      description: 'Default research-optimized model',
      badge: 'Recommended',
    },
    {
      id: 'gemini-pro',
      name: 'Gemini Pro',
      description: 'Google\'s multimodal model',
      badge: null,
    },
    {
      id: 'claude-3',
      name: 'Claude 3',
      description: 'Anthropic\'s advanced model',
      badge: 'Premium',
    },
    {
      id: 'gpt-4',
      name: 'GPT-4',
      description: 'OpenAI\'s flagship model',
      badge: 'Premium',
    },
  ];

  return (
    <div className="model-selector">
      <div className="model-selector__header">
        <Cpu size={20} />
        <h3 className="model-selector__title">AI Model</h3>
      </div>
      <p className="model-selector__description">
        Select the AI model to use for chat and research tasks.
      </p>

      <div className="model-selector__grid">
        {models.map((model) => (
          <button
            key={model.id}
            className={`model-selector__card ${selectedModel === model.id ? 'model-selector__card--selected' : ''}`}
            onClick={() => setSelectedModel(model.id)}
          >
            <div className="model-selector__card-header">
              <Sparkles size={20} className="model-selector__card-icon" />
              {model.badge && (
                <span className={`model-selector__badge model-selector__badge--${model.badge.toLowerCase()}`}>
                  {model.badge}
                </span>
              )}
            </div>
            <h4 className="model-selector__card-name">{model.name}</h4>
            <p className="model-selector__card-description">{model.description}</p>
            {selectedModel === model.id && (
              <div className="model-selector__card-check">✓</div>
            )}
          </button>
        ))}
      </div>

      <div className="model-selector__info">
        <h4>Model Capabilities</h4>
        <ul>
          <li><strong>Nub Agent:</strong> Optimized for web research and source citation</li>
          <li><strong>Gemini Pro:</strong> Strong multimodal understanding</li>
          <li><strong>Claude 3:</strong> Excellent reasoning and analysis</li>
          <li><strong>GPT-4:</strong> General purpose with broad knowledge</li>
        </ul>
      </div>
    </div>
  );
}

export default ModelSelector;
