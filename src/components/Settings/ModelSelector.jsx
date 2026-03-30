import React from 'react';
import { useSettingsStore } from '../../store/useSettingsStore.js';
import { Cpu, Sparkles } from 'lucide-react';

/**
 * ModelSelector Component
 * AI model selection
 */
export function ModelSelector() {
  const {
    selectedModel,
    researchSelectedModel,
    setResearchSelectedModel,
  } = useSettingsStore();

  const mainModelLabel = selectedModel === 'gemini-2.5-flash'
    ? 'Gemini 2.5 Flash'
    : 'Gemini 2.5 Flash Lite';

  const models = [
    {
      id: 'nvidia/nemotron-3-super-120b-a12b:free',
      name: 'Nemotron 3 Super 120B',
      description: 'OpenRouter research default for orchestration subagents',
      badge: 'Recommended',
    },
    {
      id: 'openrouter-round-robin',
      name: 'OpenRouter Round Robin',
      description: 'Rotate orchestration subagents across the configured research model chain',
      badge: 'Adaptive',
    },
    {
      id: 'gemini-2.5-flash-lite',
      name: 'Gemini 2.5 Flash Lite',
      description: 'Fast Google fallback for lighter synthesis passes',
      badge: null,
    },
    {
      id: 'gemini-2.5-flash',
      name: 'Gemini 2.5 Flash',
      description: 'Google fallback with more reasoning depth',
      badge: 'Fallback',
    },
  ];

  return (
    <div className="model-selector">
      <div className="model-selector__header">
        <Cpu size={20} />
        <h3 className="model-selector__title">Research Subagents</h3>
      </div>
      <p className="model-selector__description">
        The main NubAgent lane stays on {mainModelLabel}. Only the research orchestration subagents use the selector below.
      </p>

      <div className="model-selector__grid">
        {models.map((model) => (
          <button
            key={model.id}
            className={`model-selector__card ${researchSelectedModel === model.id ? 'model-selector__card--selected' : ''}`}
            onClick={() => setResearchSelectedModel(model.id)}
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
            {researchSelectedModel === model.id && (
              <div className="model-selector__card-check">✓</div>
            )}
          </button>
        ))}
      </div>

      <div className="model-selector__info">
        <h4>Model Capabilities</h4>
        <ul>
          <li><strong>Main agent:</strong> NubAgent remains on {mainModelLabel} for the primary chat lane</li>
          <li><strong>Nemotron 3 Super 120B:</strong> OpenRouter-backed default for synthesis, critics, and verifier subagents</li>
          <li><strong>OpenRouter Round Robin:</strong> Uses the backend research model chain so subagents can rotate instead of pinning to one model</li>
          <li><strong>Gemini 2.5 Flash Lite:</strong> Fast fallback lane for lighter research passes</li>
          <li><strong>Gemini 2.5 Flash:</strong> Deeper Google fallback when you want stronger reasoning</li>
        </ul>
      </div>
    </div>
  );
}

export default ModelSelector;
