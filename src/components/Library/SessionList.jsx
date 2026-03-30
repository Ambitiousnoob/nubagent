import React from "react";
import { SessionCard } from "./SessionCard.jsx";
import { CardSkeleton } from "../UI/Skeleton.jsx";

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
  const sessionCountLabel = `${sessions.length} saved session${sessions.length === 1 ? "" : "s"}`;

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
          <div className="session-list__empty-eyebrow">Archive empty</div>
          <h3 className="session-list__empty-title">
            No sessions match this view
          </h3>
          <p className="session-list__empty-text">
            {selectable
              ? "Widen the current slice or leave selection mode to see the rest of the archive."
              : "Saved research shows up here once a run is worth keeping."}
          </p>
          <p className="session-list__empty-note">
            {selectable
              ? "Relax filters or leave bulk mode."
              : "Each saved session keeps the query, answer, attachments, and source trail together."}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="session-list">
      <div className="session-list__summary">
        <div className="session-list__summary-copy">
          <div className="session-list__summary-eyebrow">
            {selectable ? "Selection mode" : "Archive view"}
          </div>
          <h2 className="session-list__summary-title">{sessionCountLabel}</h2>
          <p className="session-list__summary-body">
            {selectable
              ? "Choose the sessions you want to export or remove."
              : "Open saved answers quickly with the source trail still attached."}
          </p>
        </div>

        {selectable && selectedSessions.length > 0 && (
          <div className="session-list__selection-bar">
            <span>{selectedSessions.length} selected</span>
            <small>Bulk actions apply to the current selection.</small>
          </div>
        )}
      </div>

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
