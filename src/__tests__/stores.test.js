/**
 * Store Tests
 */

import { describe, it, expect, beforeEach } from 'vitest';

describe('Settings Store', () => {
  let useSettingsStore;

  beforeEach(() => {
    vi.resetModules();
    useSettingsStore = require('../store/useSettingsStore').useSettingsStore;
  });

  it('initializes with default state', () => {
    const state = useSettingsStore.getState();
    expect(state.theme).toBe('dark');
    expect(state.apiKey).toBeNull();
    expect(state.selectedModel).toBe('nub-agent');
    expect(state.preferences.autoSave).toBe(true);
  });

  it('sets theme', () => {
    const { setTheme } = useSettingsStore.getState();
    setTheme('light');
    expect(useSettingsStore.getState().theme).toBe('light');
  });

  it('toggles theme', () => {
    const { toggleTheme } = useSettingsStore.getState();
    const initialTheme = useSettingsStore.getState().theme;
    toggleTheme();
    expect(useSettingsStore.getState().theme).not.toBe(initialTheme);
  });

  it('sets API key', () => {
    const { setApiKey, getApiKey } = useSettingsStore.getState();
    setApiKey('test-key', 'openai');
    expect(getApiKey('openai')).toBe('test-key');
  });

  it('removes API key', () => {
    const { setApiKey, removeApiKey, getApiKey } = useSettingsStore.getState();
    setApiKey('test-key', 'openai');
    removeApiKey('openai');
    expect(getApiKey('openai')).toBeUndefined();
  });

  it('sets selected model', () => {
    const { setSelectedModel } = useSettingsStore.getState();
    setSelectedModel('gpt-4');
    expect(useSettingsStore.getState().selectedModel).toBe('gpt-4');
  });

  it('sets preference', () => {
    const { setPreference } = useSettingsStore.getState();
    setPreference('compactMode', true);
    expect(useSettingsStore.getState().preferences.compactMode).toBe(true);
  });

  it('resets preferences', () => {
    const { setPreference, resetPreferences } = useSettingsStore.getState();
    setPreference('compactMode', true);
    resetPreferences();
    expect(useSettingsStore.getState().preferences.compactMode).toBe(false);
  });
});

describe('UI Store', () => {
  let useUIStore;

  beforeEach(() => {
    vi.resetModules();
    useUIStore = require('../store/useUIStore').useUIStore;
  });

  it('initializes with default state', () => {
    const state = useUIStore.getState();
    expect(state.sidebar.isOpen).toBe(true);
    expect(state.sidebar.activeTab).toBe('chat');
    expect(state.modals.settings).toBe(false);
    expect(state.toasts).toEqual([]);
  });

  it('toggles sidebar', () => {
    const { toggleSidebar } = useUIStore.getState();
    const initialOpen = useUIStore.getState().sidebar.isOpen;
    toggleSidebar();
    expect(useUIStore.getState().sidebar.isOpen).not.toBe(initialOpen);
  });

  it('sets active tab', () => {
    const { setActiveTab } = useUIStore.getState();
    setActiveTab('library');
    expect(useUIStore.getState().sidebar.activeTab).toBe('library');
    expect(useUIStore.getState().currentRoute).toBe('library');
  });

  it('opens and closes modals', () => {
    const { openModal, closeModal } = useUIStore.getState();
    openModal('settings');
    expect(useUIStore.getState().modals.settings).toBe(true);
    closeModal('settings');
    expect(useUIStore.getState().modals.settings).toBe(false);
  });

  it('closes all modals', () => {
    const { openModal, closeAllModals } = useUIStore.getState();
    openModal('settings');
    openModal('library');
    closeAllModals();
    const modals = useUIStore.getState().modals;
    expect(Object.values(modals).every(v => v === false)).toBe(true);
  });

  it('adds and removes toasts', () => {
    const { addToast, removeToast } = useUIStore.getState();
    const id = addToast({ title: 'Test', type: 'success' });
    expect(useUIStore.getState().toasts).toHaveLength(1);
    removeToast(id);
    expect(useUIStore.getState().toasts).toHaveLength(0);
  });

  it('clears toasts', () => {
    const { addToast, clearToasts } = useUIStore.getState();
    addToast({ title: 'Test 1' });
    addToast({ title: 'Test 2' });
    clearToasts();
    expect(useUIStore.getState().toasts).toHaveLength(0);
  });

  it('sets route', () => {
    const { setRoute } = useUIStore.getState();
    setRoute('library');
    expect(useUIStore.getState().currentRoute).toBe('library');
  });
});
