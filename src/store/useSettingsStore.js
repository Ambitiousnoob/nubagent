import { create } from 'zustand';
import { persist } from 'zustand/middleware';

const DEFAULT_MAIN_MODEL = 'gemini-2.5-flash-lite';
const DEFAULT_RESEARCH_MODEL = 'nvidia/nemotron-3-super-120b-a12b:free';
const RESEARCH_MODEL_PRESETS = new Set([
  'nvidia/nemotron-3-super-120b-a12b:free',
  'openrouter-round-robin',
  'gemini-2.5-flash-lite',
  'gemini-2.5-flash',
]);

const migrateSettingsState = (persistedState, version) => {
  const state = persistedState && typeof persistedState === 'object' ? persistedState : {};

  if (version >= 2) {
    return {
      ...state,
      selectedModel: state.selectedModel || DEFAULT_MAIN_MODEL,
      researchSelectedModel: state.researchSelectedModel || DEFAULT_RESEARCH_MODEL,
    };
  }

  const legacySelectedModel = typeof state.selectedModel === 'string' ? state.selectedModel : '';
  const migratedResearchModel = RESEARCH_MODEL_PRESETS.has(legacySelectedModel)
    ? legacySelectedModel
    : DEFAULT_RESEARCH_MODEL;

  return {
    ...state,
    selectedModel: DEFAULT_MAIN_MODEL,
    researchSelectedModel: state.researchSelectedModel || migratedResearchModel,
  };
};

/**
 * Settings store for managing application preferences
 * Handles API keys, theme, model selection, and user preferences
 */
export const useSettingsStore = create(
  persist(
    (set, get) => ({
      // State
      theme: 'light',
      apiKey: null,
      apiKeys: {},
      selectedModel: DEFAULT_MAIN_MODEL,
      researchSelectedModel: DEFAULT_RESEARCH_MODEL,
      preferences: {
        autoSave: true,
        showSources: true,
        enableAnalytics: true,
        enableErrorTracking: true,
        compactMode: false,
      },

      // Actions
      setTheme: (theme) => set({ theme }),

      toggleTheme: () => {
        const current = get().theme;
        const newTheme = current === 'dark' ? 'light' : 'dark';
        get().setTheme(newTheme);
      },

      setApiKey: (key, provider = 'default') =>
        set((state) => {
          const nextApiKeys = { ...state.apiKeys, [provider]: key };
          return {
            apiKey: provider === 'default' ? key : state.apiKey,
            apiKeys: nextApiKeys,
          };
        }),

      getApiKey: (provider = 'default') => {
        const state = get();
        if (provider === 'default') {
          return state.apiKeys.default ?? state.apiKey ?? undefined;
        }
        return state.apiKeys[provider];
      },

      removeApiKey: (provider = 'default') =>
        set((state) => {
          const newKeys = { ...state.apiKeys };
          delete newKeys[provider];
          return {
            apiKeys: newKeys,
            apiKey: provider === 'default' ? null : state.apiKey,
          };
        }),

      setSelectedModel: (model) => set({ selectedModel: model }),
      setResearchSelectedModel: (model) => set({ researchSelectedModel: model }),

      setPreference: (key, value) =>
        set((state) => ({
          preferences: { ...state.preferences, [key]: value },
        })),

      setPreferences: (preferences) =>
        set((state) => ({
          preferences: { ...state.preferences, ...preferences },
        })),

      resetPreferences: () =>
        set({
          preferences: {
            autoSave: true,
            showSources: true,
            enableAnalytics: true,
            enableErrorTracking: true,
            compactMode: false,
          },
        }),

      clearAllSettings: () => {
        // Clear all persisted data
        set({
          theme: 'light',
          apiKey: null,
          apiKeys: {},
          selectedModel: DEFAULT_MAIN_MODEL,
          researchSelectedModel: DEFAULT_RESEARCH_MODEL,
          preferences: {
            autoSave: true,
            showSources: true,
            enableAnalytics: true,
            enableErrorTracking: true,
            compactMode: false,
          },
        });
      },
    }),
    {
      name: 'nubagent-settings-storage',
      version: 2,
      migrate: migrateSettingsState,
    }
  )
);

export default useSettingsStore;
