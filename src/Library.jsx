import React, { useState, useCallback, useEffect } from 'react';
import { useLibraryStore } from './store/useLibraryStore.js';
import { useUIStore } from './store/useUIStore.js';
import { SessionList } from './components/Library/SessionList.jsx';
import { SessionActions, BulkActions } from './components/Library/SessionActions.jsx';
import { SearchFilters } from './components/Search/SearchFilters.jsx';
import { Button } from './components/UI/Button.jsx';
import { Input } from './components/UI/Input.jsx';
import { useToast } from './components/UI/ToastProvider.jsx';
import {
  Search,
  ArrowLeft,
  Plus,
  Filter,
  Trash2,
  Download,
  CheckSquare,
  Square,
} from 'lucide-react';

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
    getFilteredSessions,
    loadSessions,
    clearSelection,
    toggleSessionSelection,
  } = useLibraryStore();

  const { openModal } = useUIStore();
  const { success, error, info } = useToast();
  const [actionsModalOpen, setActionsModalOpen] = useState(false);
  const [selectedSessionForActions, setSelectedSessionForActions] = useState(null);
  const [isSelectionMode, setIsSelectionMode] = useState(false);

  const filteredSessions = getFilteredSessions();

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
      // Export functionality would be implemented here
      info('Export Started', `Preparing ${selectedSessions.length} sessions for export...`);
    } catch (err) {
      error('Export Failed', 'Unable to export sessions');
    }
  }, [selectedSessions, info, error]);

  const handleSelectAll = useCallback(() => {
    if (selectedSessions.length === filteredSessions.length) {
      clearSelection();
    } else {
      filteredSessions.forEach((session) => {
        toggleSessionSelection(session.id);
      });
    }
  }, [filteredSessions, selectedSessions.length, clearSelection, toggleSessionSelection]);

  const handleShare = useCallback((session) => {
    info('Share', 'Share functionality coming soon');
  }, [info]);

  const handleEdit = useCallback((session) => {
    info('Edit', 'Edit functionality coming soon');
  }, [info]);

  const handleExport = useCallback((session, format) => {
    info('Export', `Exporting session as ${format}...`);
  }, [info]);

  const openSessionActions = useCallback((session) => {
    setSelectedSessionForActions(session);
    setActionsModalOpen(true);
  }, []);

  return (
    <div className="library-page">
      {/* Header */}
      <div className="library-page__header">
        <div className="library-page__title-section">
          <button className="library-page__back" onClick={onBack}>
            <ArrowLeft size={20} />
          </button>
          <h1 className="library-page__title">📚 Library</h1>
        </div>
        <div className="library-page__actions">
          {isSelectionMode ? (
            <>
              <Button variant="outline" size="sm" onClick={() => setIsSelectionMode(false)}>
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

      {/* Bulk Actions Bar */}
      {isSelectionMode && selectedSessions.length > 0 && (
        <BulkActions
          selectedCount={selectedSessions.length}
          onDelete={handleBulkDelete}
          onExport={() => handleBulkExport('markdown')}
          onClearSelection={clearSelection}
        />
      )}

      {/* Search and Filters */}
      <div className="library-page__search-section">
        <div className="library-page__search">
          <Search size={18} className="library-page__search-icon" />
          <input
            type="text"
            className="library-page__search-input"
            placeholder="Search your library..."
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

      {/* Session List */}
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

      {/* Session Actions Modal */}
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
  );
}
