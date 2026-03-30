import React, { useState, useCallback, useEffect } from 'react';
import { useLibraryStore } from './store/useLibraryStore.js';
import { SessionList } from './components/Library/SessionList.jsx';
import { SessionActions, BulkActions } from './components/Library/SessionActions.jsx';
import { SearchFilters } from './components/Search/SearchFilters.jsx';
import { Button } from './components/UI/Button.jsx';
import { useToast } from './components/UI/ToastProvider.jsx';
import { buildSessionShareUrl, promptToCopySessionUrl } from './lib/library.js';
import './styles/knowledge-surfaces.css';
import {
  Search,
  ArrowLeft,
  Plus,
  CheckSquare,
  Square,
  Sparkles,
  FileStack,
  SlidersHorizontal,
} from 'lucide-react';

const triggerDownload = (filename, content, contentType) => {
  const blob = new Blob([content], { type: contentType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
};

const slugifyFilename = (value = 'session') => (
  value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48) || 'session'
);

/**
 * Enhanced Library Component
 * Modern library view with search, filters, and bulk actions
 */
export default function Library({ onBack, onViewSession, onNewSearch }) {
  const {
    sessions,
    isLoading,
    searchQuery,
    filters,
    selectedSessions,
    setSearchQuery,
    setFilters,
    resetFilters,
    deleteSession,
    deleteSessions,
    updateSession,
    exportSessions,
    getFilteredSessions,
    loadSessions,
    clearSelection,
    setSelectedSessions,
    toggleSessionSelection,
  } = useLibraryStore();

  const { success, error } = useToast();
  const [actionsModalOpen, setActionsModalOpen] = useState(false);
  const [selectedSessionForActions, setSelectedSessionForActions] = useState(null);
  const [isSelectionMode, setIsSelectionMode] = useState(false);

  const filteredSessions = getFilteredSessions();
  const totalSources = sessions.reduce((count, session) => count + (session.sources?.length || 0), 0);
  const totalAttachments = sessions.reduce((count, session) => count + (session.attachments?.length || 0), 0);
  const activeFilterCount = [
    filters.dateRange && filters.dateRange !== 'all',
    Boolean(filters.hasAttachments),
    filters.sortBy && filters.sortBy !== 'date',
    filters.sortOrder && filters.sortOrder !== 'desc',
  ].filter(Boolean).length;
  const archiveModeLabel = isSelectionMode
    ? `${selectedSessions.length} selected for bulk actions`
    : `${filteredSessions.length} session${filteredSessions.length === 1 ? '' : 's'} in view`;
  const archiveSignals = [
    {
      icon: <Sparkles size={16} />,
      label: 'Curation posture',
      value: activeFilterCount === 0 ? 'Broad review lane' : `${activeFilterCount} filters shaping the current slice`,
    },
    {
      icon: <FileStack size={16} />,
      label: 'Export readiness',
      value: sessions.length === 0 ? 'No reusable runs yet' : 'Saved runs remain exportable as working artifacts',
    },
    {
      icon: <SlidersHorizontal size={16} />,
      label: 'Selection mode',
      value: isSelectionMode ? 'Bulk actions armed for archive cleanup' : 'Single-session browsing is active',
    },
  ];

  // Reload sessions on mount
  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  const handleSearchChange = useCallback((e) => {
    setSearchQuery(e.target.value);
  }, [setSearchQuery]);

  const handleFilterChange = useCallback((newFilters) => {
    setFilters(newFilters);
  }, [setFilters]);

  const handleResetFilters = useCallback(() => {
    resetFilters();
    setSearchQuery('');
  }, [resetFilters, setSearchQuery]);

  const handleView = useCallback((session) => {
    onViewSession?.(session);
  }, [onViewSession]);

  const handleDelete = useCallback((id) => {
    deleteSession(id);
    success('Session Deleted', 'The session has been removed from your library');
  }, [deleteSession, success]);

  const handleBulkDelete = useCallback(() => {
    if (selectedSessions.length === 0) return;

    deleteSessions(selectedSessions);
    setIsSelectionMode(false);
    clearSelection();
    success('Sessions Deleted', `${selectedSessions.length} sessions have been removed`);
  }, [selectedSessions, deleteSessions, clearSelection, success]);

  const handleBulkExport = useCallback(async (format = 'markdown') => {
    try {
      const content = await exportSessions(format, selectedSessions);
      if (!content) throw new Error('No sessions were exported.');

      const extension = format === 'json' ? 'json' : 'md';
      triggerDownload(
        `nubagent-library-${selectedSessions.length}-sessions.${extension}`,
        content,
        format === 'json' ? 'application/json' : 'text/markdown',
      );
      success('Export Ready', `${selectedSessions.length} session${selectedSessions.length === 1 ? '' : 's'} exported.`);
    } catch (err) {
      error('Export Failed', 'Unable to export sessions');
    }
  }, [error, exportSessions, selectedSessions, success]);

  const handleSelectAll = useCallback(() => {
    if (selectedSessions.length === filteredSessions.length) {
      clearSelection();
    } else {
      setSelectedSessions(filteredSessions.map((session) => session.id));
    }
  }, [filteredSessions, selectedSessions.length, clearSelection, setSelectedSessions]);

  const handleShare = useCallback(async (session, options = {}) => {
    try {
      const shareUrl = buildSessionShareUrl(session?.id);
      if (!shareUrl) throw new Error('No share URL available.');

      if (options?.copied) {
        success('Link Copied', 'A direct link to this saved session is in your clipboard.');
        return;
      }

      if (navigator.share) {
        await navigator.share({
          title: session?.query || 'nubagent session',
          text: session?.heading || session?.body || session?.query || 'Saved NubAgent session',
          url: shareUrl,
        });
        success('Shared', 'Your saved session link is ready to send.');
        return;
      }

      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(shareUrl);
        success('Link Copied', 'A direct link to this saved session is in your clipboard.');
        return;
      }

      if (promptToCopySessionUrl(shareUrl)) {
        success('Copy Link', 'Use the dialog to copy your saved session link.');
        return;
      }

      throw new Error('Sharing unavailable.');
    } catch (shareError) {
      if (promptToCopySessionUrl(buildSessionShareUrl(session?.id))) {
        success('Copy Link', 'Use the dialog to copy your saved session link.');
        return;
      }
      error('Share Failed', 'Unable to share this saved session.');
    }
  }, [error, success]);

  const handleEdit = useCallback((session, updates = {}) => {
    if (!session?.id) return;

    const nextQuery = updates.query?.trim() || session.query || 'Untitled session';
    const nextHeading = updates.heading?.trim();
    const updatedSession = updateSession(session.id, {
      query: nextQuery,
      heading: nextHeading || nextQuery,
      body: updates.body ?? session.body ?? '',
    });

    if (!updatedSession) {
      error('Save Failed', 'This session could not be updated.');
      return;
    }

    setSelectedSessionForActions(updatedSession);
    success('Session Saved', 'Your library changes were saved.');
  }, [error, success, updateSession]);

  const handleExport = useCallback(async (session, format) => {
    try {
      const content = await exportSessions(format, session?.id ? [session.id] : []);
      if (!content || !session?.id) throw new Error('Unable to export this session.');

      const extension = format === 'json' ? 'json' : 'md';
      triggerDownload(
        `${slugifyFilename(session.query || 'nubagent-session')}.${extension}`,
        content,
        format === 'json' ? 'application/json' : 'text/markdown',
      );
      success('Export Ready', 'The saved session export has started.');
    } catch (exportError) {
      error('Export Failed', 'Unable to export this session.');
    }
  }, [error, exportSessions, success]);

  const openSessionActions = useCallback((session) => {
    setSelectedSessionForActions(session);
    setActionsModalOpen(true);
  }, []);

  return (
    <div className="library-page library-page--refined">
      <section className="library-page__hero">
        <div className="library-page__hero-copy">
          <div className="library-page__eyebrow">Knowledge archive</div>
          <h1 className="library-page__hero-title">Research worth keeping, shaped into a working library.</h1>
          <p className="library-page__hero-body">
            Re-open strong runs, refine saved answers, export useful work, and keep a cleaner signal path between active research and retained knowledge.
          </p>
          <div className="library-page__hero-note">
            Built for retrieval, review, and clean reuse instead of a flat pile of old transcripts.
          </div>
        </div>
        <div className="library-page__hero-stats">
          <div className="library-page__hero-stat">
            <span>{sessions.length}</span>
            <small>Saved sessions</small>
          </div>
          <div className="library-page__hero-stat">
            <span>{totalSources}</span>
            <small>Cited sources</small>
          </div>
          <div className="library-page__hero-stat">
            <span>{totalAttachments}</span>
            <small>Attachments tracked</small>
          </div>
        </div>
      </section>

      <section className="library-page__signal-grid" aria-label="Library operating signals">
        {archiveSignals.map((signal) => (
          <article key={signal.label} className="library-page__signal-card">
            <div className="library-page__signal-icon">{signal.icon}</div>
            <div>
              <div className="library-page__signal-label">{signal.label}</div>
              <p className="library-page__signal-value">{signal.value}</p>
            </div>
          </article>
        ))}
      </section>

      <div className="library-page__workspace">
        <div className="library-page__workspace-overview">
          <div className="library-page__workspace-copy">
            <div className="library-page__section-label">Archive control room</div>
            <h2 className="library-page__workspace-title">Curate strong sessions, not just stored runs.</h2>
            <p className="library-page__workspace-body">
              Search by intent, tighten the view with filters, and move from passive storage to an actively usable research shelf.
            </p>
          </div>
          <div className="library-page__workspace-metrics">
            <div className="library-page__workspace-metric">
              <span className="library-page__workspace-metric-label">View</span>
              <strong>{archiveModeLabel}</strong>
            </div>
            <div className="library-page__workspace-metric">
              <span className="library-page__workspace-metric-label">Filters</span>
              <strong>{activeFilterCount === 0 ? 'Default' : `${activeFilterCount} active`}</strong>
            </div>
            <div className="library-page__workspace-metric">
              <span className="library-page__workspace-metric-label">Attachments</span>
              <strong>{totalAttachments} tracked</strong>
            </div>
          </div>
        </div>

        <div className="library-page__header">
          <div className="library-page__title-section">
            <button className="library-page__back" onClick={onBack}>
              <ArrowLeft size={20} />
            </button>
            <div className="library-page__title-copy">
              <div className="library-page__section-label">Library</div>
              <h1 className="library-page__title">Saved research sessions</h1>
            </div>
          </div>
          <div className="library-page__actions">
            {isSelectionMode ? (
              <>
                <Button variant="outline" size="sm" onClick={() => {
                  clearSelection();
                  setIsSelectionMode(false);
                }}>
                  Cancel
                </Button>
                <Button variant="ghost" size="sm" onClick={handleSelectAll}>
                  {selectedSessions.length === filteredSessions.length ? (
                    <CheckSquare size={16} />
                  ) : (
                    <Square size={16} />
                  )}
                </Button>
              </>
            ) : (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setIsSelectionMode(true)}
                  disabled={filteredSessions.length === 0}
                >
                  Select
                </Button>
                <Button variant="primary" size="sm" onClick={onNewSearch}>
                  <Plus size={16} />
                  New Search
                </Button>
              </>
            )}
          </div>
        </div>

        <div className="library-page__search-shell">
          <div className="library-page__search-intro">
            <div className="library-page__section-label">Find and refine</div>
            <p className="library-page__search-body">
              Narrow by recency, attachments, or ordering, then jump back into any saved answer without losing the provenance around it.
            </p>
          </div>

          <div className="library-page__search-section">
            <div className="library-page__search">
              <Search size={18} className="library-page__search-icon" />
              <input
                type="text"
                className="library-page__search-input"
                placeholder="Search saved queries, answers, or cited domains..."
                value={searchQuery}
                onChange={handleSearchChange}
              />
              {searchQuery && (
                <button
                  className="library-page__search-clear"
                  onClick={() => setSearchQuery('')}
                >
                  ×
                </button>
              )}
              <span className="library-page__search-count">
                {filteredSessions.length} session{filteredSessions.length !== 1 ? 's' : ''}
              </span>
            </div>
            <SearchFilters
              filters={filters}
              onFilterChange={handleFilterChange}
              onReset={handleResetFilters}
            />
          </div>
        </div>

        {isSelectionMode && selectedSessions.length > 0 && (
          <BulkActions
            selectedCount={selectedSessions.length}
            onDelete={handleBulkDelete}
            onExport={() => handleBulkExport('markdown')}
            onClearSelection={clearSelection}
          />
        )}

        <div className="library-page__results-head">
          <div>
            <div className="library-page__section-label">Archive view</div>
            <h2 className="library-page__results-title">
              {filteredSessions.length === 0 ? 'No sessions in the current slice' : `${filteredSessions.length} saved session${filteredSessions.length === 1 ? '' : 's'} ready to review`}
            </h2>
          </div>
          <div className="library-page__results-copy">
            <p className="library-page__results-body">
              {searchQuery
                ? `Showing matches for "${searchQuery}".`
                : 'Sorted for quick re-entry into prior research, exports, and follow-up edits.'}
            </p>
            <div className="library-page__results-chip">
              {isSelectionMode ? 'Bulk archive mode' : 'Single-session review'}
            </div>
          </div>
        </div>

        <SessionList
          sessions={filteredSessions}
          onView={handleView}
          onDelete={handleDelete}
          onEdit={openSessionActions}
          onShare={handleShare}
          isLoading={isLoading}
          selectable={isSelectionMode}
          selectedSessions={selectedSessions}
          onSelect={toggleSessionSelection}
        />

        {selectedSessionForActions && (
          <SessionActions
            session={selectedSessionForActions}
            isOpen={actionsModalOpen}
            onClose={() => {
              setActionsModalOpen(false);
              setSelectedSessionForActions(null);
            }}
            onDelete={handleDelete}
            onEdit={handleEdit}
            onShare={handleShare}
            onExport={handleExport}
          />
        )}
      </div>
    </div>
  );
}
