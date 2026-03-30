import { create } from 'zustand';
import { persist } from 'zustand/middleware';

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
      selectedModel: 'nub-agent',
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
          selectedModel: 'nub-agent',
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
    }
  )
);

export default useSettingsStore;
