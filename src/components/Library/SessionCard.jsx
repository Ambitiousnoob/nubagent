import React, { useState } from 'react';
import { Calendar, FileText, Trash2, Edit2, Share2, ExternalLink } from 'lucide-react';

/**
 * SessionCard Component
 * Library item card with session preview
 * 
 * @param {object} session - Session object
 * @param {function} onView - View handler
 * @param {function} onDelete - Delete handler
 * @param {function} onEdit - Edit handler
 * @param {function} onShare - Share handler
 * @param {boolean} isSelected - Selection state
 * @param {function} onSelect - Select handler
 */
export function SessionCard({
  session,
  onView,
  onDelete,
  onEdit,
  onShare,
  isSelected = false,
  onSelect,
}) {
  const [isDeleting, setIsDeleting] = useState(false);
  const [showActions, setShowActions] = useState(false);

  const getDomain = (url) => {
    try {
      return new URL(url).hostname.replace(/^www\./, '');
    } catch {
      return url;
    }
  };

  const getFavicon = (url) => {
    try {
      return `https://www.google.com/s2/favicons?domain=${new URL(url).origin}&sz=32`;
    } catch {
      return null;
    }
  };

  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
    if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;
    return `${Math.floor(diffDays / 365)} years ago`;
  };

  const handleDelete = (e) => {
    e.stopPropagation();
    setIsDeleting(true);
    setTimeout(() => {
      onDelete?.(session.id);
    }, 300);
  };

  const handleShare = (e) => {
    e.stopPropagation();
    onShare?.(session);
  };

  const handleEdit = (e) => {
    e.stopPropagation();
    onEdit?.(session);
  };

  const sourceCount = session.sources?.length || 0;
  const firstSource = session.sources?.[0];
  const hasAttachments = session.attachments?.length > 0;

  return (
    <div
      className={`session-card ${isDeleting ? 'session-card--deleting' : ''} ${isSelected ? 'session-card--selected' : ''}`}
      onClick={() => onView?.(session)}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && onView?.(session)}
    >
      {onSelect && (
        <input
          type="checkbox"
          className="session-card__checkbox"
          checked={isSelected}
          onChange={(e) => {
            e.stopPropagation();
            onSelect?.(session.id);
          }}
          onClick={(e) => e.stopPropagation()}
        />
      )}

      <div className="session-card__header">
        <div className="session-card__meta">
          <span className="session-card__date">
            <Calendar size={12} />
            {formatDate(session.createdAt)}
          </span>
          {sourceCount > 0 && (
            <span className="session-card__sources">
              <FileText size={12} />
              {sourceCount} source{sourceCount !== 1 ? 's' : ''}
            </span>
          )}
          {hasAttachments && (
            <span className="session-card__attachments">
              📎 {session.attachments.length}
            </span>
          )}
        </div>
      </div>

      <h3 className="session-card__query">{session.query}</h3>

      <p className="session-card__preview">
        {(session.heading || session.body || '').replace(/[#*`\[\]]/g, '').slice(0, 120)}
        {(session.heading || session.body || '').length > 120 ? '...' : ''}
      </p>

      {firstSource && (
        <div className="session-card__source">
          {getFavicon(firstSource.url) && (
            <img
              src={getFavicon(firstSource.url)}
              alt=""
              className="session-card__favicon"
              onError={(e) => { e.currentTarget.style.display = 'none'; }}
            />
          )}
          <span className="session-card__domain">{getDomain(firstSource.url)}</span>
        </div>
      )}

      <div className={`session-card__actions ${showActions ? 'session-card__actions--visible' : ''}`}>
        <button
          className="session-card__action session-card__action--share"
          onClick={handleShare}
          title="Share"
        >
          <Share2 size={14} />
        </button>
        <button
          className="session-card__action session-card__action--edit"
          onClick={handleEdit}
          title="Edit"
        >
          <Edit2 size={14} />
        </button>
        <button
          className="session-card__action session-card__action--delete"
          onClick={handleDelete}
          title="Delete"
        >
          <Trash2 size={14} />
        </button>
        <a
          href={session.sources?.[0]?.url}
          target="_blank"
          rel="noopener noreferrer"
          className="session-card__action session-card__action--external"
          onClick={(e) => e.stopPropagation()}
          title="Open source"
        >
          <ExternalLink size={14} />
        </a>
      </div>
    </div>
  );
}

export default SessionCard;
