import React from 'react';
import { SessionCard } from './SessionCard.jsx';
import { CardSkeleton } from '../UI/Skeleton.jsx';

/**
 * SessionList Component
 * Filterable list of session cards
 * 
 * @param {array} sessions - Array of session objects
 * @param {function} onView - View handler
 * @param {function} onDelete - Delete handler
 * @param {function} onEdit - Edit handler
 * @param {function} onShare - Share handler
 * @param {boolean} isLoading - Loading state
 * @param {boolean} selectable - Enable selection mode
 * @param {array} selectedSessions - Array of selected session IDs
 * @param {function} onSelect - Select handler
 */
export function SessionList({
  sessions = [],
  onView,
  onDelete,
  onEdit,
  onShare,
  isLoading = false,
  selectable = false,
  selectedSessions = [],
  onSelect,
}) {
  if (isLoading) {
    return (
      <div className="session-list session-list--loading">
        {Array.from({ length: 6 }).map((_, i) => (
          <CardSkeleton key={i} />
        ))}
      </div>
    );
  }

  if (sessions.length === 0) {
    return (
      <div className="session-list session-list--empty">
        <div className="session-list__empty">
          <div className="session-list__empty-icon">📚</div>
          <h3 className="session-list__empty-title">No sessions found</h3>
          <p className="session-list__empty-text">
            {selectable
              ? 'No sessions match your filters.'
              : 'Your library is empty. Start a new search to create your first session.'}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="session-list">
      {selectable && selectedSessions.length > 0 && (
        <div className="session-list__selection-bar">
          <span>{selectedSessions.length} selected</span>
        </div>
      )}
      <div className="session-list__grid">
        {sessions.map((session) => (
          <SessionCard
            key={session.id}
            session={session}
            onView={onView}
            onDelete={onDelete}
            onEdit={onEdit}
            onShare={onShare}
            isSelected={selectedSessions.includes(session.id)}
            onSelect={selectable ? onSelect : undefined}
          />
        ))}
      </div>
    </div>
  );
}

export default SessionList;
