import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
  getSavedSessions,
  saveSession,
  deleteSession,
  getSessionById,
  searchSessions,
} from '../lib/library.js';

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
      filters: {
        dateRange: 'all',
        hasAttachments: false,
        sortBy: 'date',
        sortOrder: 'desc',
      },
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
        saveSession(newSession);
        set((state) => ({
          sessions: [newSession, ...state.sessions],
        }));
        return newSession;
      },

      updateSession: (id, updates) => {
        const updated = {
          ...getSessionById(id),
          ...updates,
          updatedAt: Date.now(),
        };
        saveSession(updated);
        set((state) => ({
          sessions: state.sessions.map((s) => (s.id === id ? updated : s)),
        }));
        return updated;
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
          filters: {
            dateRange: 'all',
            hasAttachments: false,
            sortBy: 'date',
            sortOrder: 'desc',
          },
        }),

      selectSession: (session) => set({ selectedSession: session }),

      clearSelectedSession: () => set({ selectedSession: null }),

      toggleSessionSelection: (id) =>
        set((state) => ({
          selectedSessions: state.selectedSessions.includes(id)
            ? state.selectedSessions.filter((sid) => sid !== id)
            : [...state.selectedSessions, id],
        })),

      clearSelection: () => set({ selectedSessions: [] }),

      selectAll: () =>
        set((state) => ({
          selectedSessions: state.filteredSessions?.map((s) => s.id) || [],
        })),

      getFilteredSessions: () => {
        const state = get();
        const { sessions, searchQuery, filters } = state;

        let filtered = [...sessions];

        // Apply search
        if (searchQuery.trim()) {
          filtered = searchSessions(searchQuery);
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
          filtered = filtered.filter((s) => now - s.createdAt <= range);
        }

        // Apply attachment filter
        if (filters.hasAttachments) {
          filtered = filtered.filter((s) => s.attachments?.length > 0);
        }

        // Apply sorting
        filtered.sort((a, b) => {
          let comparison = 0;
          if (filters.sortBy === 'date') {
            comparison = a.createdAt - b.createdAt;
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
        filters: state.filters,
      }),
    }
  )
);

export default useLibraryStore;
