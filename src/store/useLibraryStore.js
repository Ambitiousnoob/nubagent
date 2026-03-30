import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
  getSavedSessions,
  saveSession,
  deleteSession,
  getSessionById,
  getSessionTimestamp,
} from '../lib/library.js';

const DEFAULT_FILTERS = {
  dateRange: 'all',
  hasAttachments: false,
  sortBy: 'date',
  sortOrder: 'desc',
};

const matchesSessionSearch = (session, searchQuery) => {
  const term = searchQuery.trim().toLowerCase();
  if (!term) return true;

  return [
    session?.query,
    session?.heading,
    session?.body,
    ...(Array.isArray(session?.attachments) ? session.attachments.map((attachment) => attachment?.name) : []),
  ].some((value) => typeof value === 'string' && value.toLowerCase().includes(term));
};

/**
 * Library store for managing session library state
 * Handles loading, filtering, and CRUD operations on saved sessions
 */
export const useLibraryStore = create(
  persist(
    (set, get) => ({
      // State
      sessions: [],
      isLoading: false,
      error: null,
      searchQuery: '',
      filters: DEFAULT_FILTERS,
      selectedSession: null,
      selectedSessions: [],

      // Actions
      loadSessions: async () => {
        set({ isLoading: true, error: null });
        try {
          // Simulate async delay for UX
          await new Promise((resolve) => setTimeout(resolve, 150));
          const sessions = getSavedSessions();
          set({ sessions, isLoading: false });
          return sessions;
        } catch (error) {
          set({ error: error.message, isLoading: false });
          return [];
        }
      },

      addSession: (session) => {
        const newSession = {
          ...session,
          id: session.id || `${Date.now()}-${Math.random().toString(16).slice(2)}`,
          createdAt: session.createdAt || Date.now(),
          updatedAt: Date.now(),
        };
        const savedSession = saveSession(newSession);
        set((state) => ({
          sessions: [savedSession, ...state.sessions.filter((item) => item.id !== savedSession.id)],
        }));
        return savedSession;
      },

      updateSession: (id, updates) => {
        const existing = getSessionById(id);
        if (!existing) return null;

        const updated = {
          ...existing,
          ...updates,
          updatedAt: Date.now(),
        };
        const savedSession = saveSession(updated);
        set((state) => ({
          sessions: state.sessions.some((s) => s.id === id)
            ? state.sessions.map((s) => (s.id === id ? savedSession : s))
            : [savedSession, ...state.sessions],
          selectedSession: state.selectedSession?.id === id ? savedSession : state.selectedSession,
        }));
        return savedSession;
      },

      deleteSession: async (id) => {
        deleteSession(id);
        set((state) => ({
          sessions: state.sessions.filter((s) => s.id !== id),
          selectedSession: state.selectedSession?.id === id ? null : state.selectedSession,
          selectedSessions: state.selectedSessions.filter((sid) => sid !== id),
        }));
      },

      deleteSessions: async (ids) => {
        ids.forEach((id) => deleteSession(id));
        set((state) => ({
          sessions: state.sessions.filter((s) => !ids.includes(s.id)),
          selectedSessions: [],
        }));
      },

      getSessionById: (id) => getSessionById(id),

      setSearchQuery: (query) => set({ searchQuery: query }),

      setFilters: (filters) =>
        set((state) => ({
          filters: { ...state.filters, ...filters },
        })),

      resetFilters: () =>
        set({
          searchQuery: '',
          filters: DEFAULT_FILTERS,
        }),

      selectSession: (session) => set({ selectedSession: session }),

      clearSelectedSession: () => set({ selectedSession: null }),

      setSelectedSessions: (ids = []) =>
        set({
          selectedSessions: Array.from(new Set(ids)),
        }),

      toggleSessionSelection: (id) =>
        set((state) => ({
          selectedSessions: state.selectedSessions.includes(id)
            ? state.selectedSessions.filter((sid) => sid !== id)
            : [...state.selectedSessions, id],
        })),

      clearSelection: () => set({ selectedSessions: [] }),

      selectAll: (sessionIds = []) =>
        set({
          selectedSessions: Array.from(new Set(sessionIds)),
        }),

      getFilteredSessions: () => {
        const state = get();
        const { sessions, searchQuery, filters } = state;

        let filtered = [...sessions];

        // Apply search
        if (searchQuery.trim()) {
          filtered = filtered.filter((session) => matchesSessionSearch(session, searchQuery));
        }

        // Apply date filter
        if (filters.dateRange !== 'all') {
          const now = Date.now();
          const ranges = {
            today: 24 * 60 * 60 * 1000,
            week: 7 * 24 * 60 * 60 * 1000,
            month: 30 * 24 * 60 * 60 * 1000,
            year: 365 * 24 * 60 * 60 * 1000,
          };
          const range = ranges[filters.dateRange] || 0;
          filtered = filtered.filter((s) => {
            const createdAt = getSessionTimestamp(s.createdAt);
            return createdAt > 0 && now - createdAt <= range;
          });
        }

        // Apply attachment filter
        if (filters.hasAttachments) {
          filtered = filtered.filter((s) => s.attachments?.length > 0);
        }

        // Apply sorting
        filtered.sort((a, b) => {
          let comparison = 0;
          if (filters.sortBy === 'date') {
            comparison = getSessionTimestamp(a.createdAt) - getSessionTimestamp(b.createdAt);
          } else if (filters.sortBy === 'title') {
            comparison = a.query.localeCompare(b.query);
          } else if (filters.sortBy === 'sources') {
            comparison = (a.sources?.length || 0) - (b.sources?.length || 0);
          }
          return filters.sortOrder === 'desc' ? -comparison : comparison;
        });

        return filtered;
      },

      exportSessions: async (format = 'json', sessionIds = []) => {
        const state = get();
        const sessionsToExport = sessionIds.length
          ? state.sessions.filter((s) => sessionIds.includes(s.id))
          : state.sessions;

        if (format === 'json') {
          return JSON.stringify(sessionsToExport, null, 2);
        }

        if (format === 'markdown') {
          return sessionsToExport
            .map((s) => {
              const date = new Date(s.createdAt).toISOString();
              return `# ${s.query}\n\n**Date:** ${date}\n\n${s.body || ''}\n\n---\n`;
            })
            .join('\n');
        }

        return null;
      },
    }),
    {
      name: 'nubagent-library-storage',
      partialize: (state) => ({
        searchQuery: state.searchQuery,
        filters: state.filters,
      }),
    }
  )
);

export default useLibraryStore;
