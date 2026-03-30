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
    expect(state.theme).toBe('light');
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

describe('Library Store', () => {
  let useLibraryStore;
  const defaultLibraryFilters = {
    dateRange: 'all',
    hasAttachments: false,
    sortBy: 'date',
    sortOrder: 'desc',
  };

  beforeEach(() => {
    vi.useRealTimers();
    vi.resetModules();
    window.localStorage.clear();
    useLibraryStore = require('../store/useLibraryStore').useLibraryStore;
    useLibraryStore.setState({
      sessions: [],
      isLoading: false,
      error: null,
      searchQuery: '',
      filters: defaultLibraryFilters,
      selectedSession: null,
      selectedSessions: [],
    });
  });

  it('updates a saved session and keeps selected session in sync', async () => {
    window.localStorage.setItem('nubagent-library', JSON.stringify([
      {
        id: 'session-1',
        query: 'Original query',
        heading: 'Original heading',
        body: 'Original body',
        sources: [],
        attachments: [],
        createdAt: '2026-03-20T12:00:00.000Z',
        updatedAt: '2026-03-20T12:00:00.000Z',
      },
    ]));

    await useLibraryStore.getState().loadSessions();
    const original = useLibraryStore.getState().sessions[0];
    useLibraryStore.getState().selectSession(original);

    const updated = useLibraryStore.getState().updateSession('session-1', {
      query: 'Updated query',
      body: 'Updated body',
    });

    expect(updated.query).toBe('Updated query');
    expect(useLibraryStore.getState().sessions[0].query).toBe('Updated query');
    expect(useLibraryStore.getState().selectedSession.query).toBe('Updated query');
  });

  it('filters and sorts sessions correctly when persisted timestamps are strings', async () => {
    const now = new Date('2026-03-30T12:00:00.000Z');
    vi.useFakeTimers();
    vi.setSystemTime(now);

    window.localStorage.setItem('nubagent-library', JSON.stringify([
      {
        id: 'recent',
        query: 'Recent session',
        heading: 'Recent',
        body: 'Body',
        sources: [],
        attachments: [],
        createdAt: '2026-03-28T12:00:00.000Z',
        updatedAt: '2026-03-28T12:00:00.000Z',
      },
      {
        id: 'older',
        query: 'Older session',
        heading: 'Older',
        body: 'Body',
        sources: [],
        attachments: [],
        createdAt: '2026-02-10T12:00:00.000Z',
        updatedAt: '2026-02-10T12:00:00.000Z',
      },
    ]));

    const loadPromise = useLibraryStore.getState().loadSessions();
    await vi.advanceTimersByTimeAsync(200);
    await loadPromise;
    useLibraryStore.getState().setFilters({ dateRange: 'week' });
    const filtered = useLibraryStore.getState().getFilteredSessions();

    expect(filtered).toHaveLength(1);
    expect(filtered[0].id).toBe('recent');

    vi.useRealTimers();
  });

  it('matches attachment names in library search', async () => {
    vi.useFakeTimers();
    window.localStorage.setItem('nubagent-library', JSON.stringify([
      {
        id: 'session-attachment',
        query: 'Analyze uploads',
        heading: 'Attachment summary',
        body: 'Body',
        sources: [],
        attachments: [{ id: 'file-1', name: 'roadmap.pdf' }],
        createdAt: '2026-03-20T12:00:00.000Z',
        updatedAt: '2026-03-20T12:00:00.000Z',
      },
    ]));

    const loadPromise = useLibraryStore.getState().loadSessions();
    await vi.advanceTimersByTimeAsync(200);
    await loadPromise;
    useLibraryStore.getState().resetFilters();
    useLibraryStore.getState().setSearchQuery('roadmap');

    const filtered = useLibraryStore.getState().getFilteredSessions();
    expect(filtered).toHaveLength(1);
    expect(filtered[0].id).toBe('session-attachment');

    vi.useRealTimers();
  });
});
