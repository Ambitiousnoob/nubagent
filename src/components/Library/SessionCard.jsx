import React, { useState } from 'react';
import { Calendar, FileText, Trash2, Edit2, Share2, ExternalLink, Paperclip } from 'lucide-react';

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
  const isSelectionMode = typeof onSelect === 'function';
  const cleanedPreview = (session.heading || session.body || '').replace(/[#*`\[\]]/g, '');

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
  const attachmentCount = session.attachments?.length || 0;
  const handleCardActivate = () => {
    if (isSelectionMode) {
      onSelect?.(session.id);
      return;
    }
    onView?.(session);
  };

  return (
    <div
      className={`session-card ${isDeleting ? 'session-card--deleting' : ''} ${isSelected ? 'session-card--selected' : ''}`}
      onClick={handleCardActivate}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
      onFocus={() => setShowActions(true)}
      onBlur={() => setShowActions(false)}
      role="button"
      tabIndex={0}
      aria-pressed={isSelectionMode ? isSelected : undefined}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleCardActivate();
        }
      }}
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
        <div className="session-card__eyebrow-row">
          <div className="session-card__eyebrow">Saved run</div>
          <div className="session-card__state">{sourceCount > 0 ? 'Evidence attached' : 'Transcript only'}</div>
        </div>
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
              <Paperclip size={12} />
              {session.attachments.length}
            </span>
          )}
        </div>
      </div>

      <h3 className="session-card__query">{session.query}</h3>

      <p className="session-card__preview">
        {cleanedPreview.slice(0, 150)}
        {cleanedPreview.length > 150 ? '...' : ''}
      </p>

      <div className="session-card__signal-row">
        <span className="session-card__signal-chip">{sourceCount} source{sourceCount !== 1 ? 's' : ''}</span>
        <span className="session-card__signal-chip">{hasAttachments ? `${attachmentCount} attachment${attachmentCount === 1 ? '' : 's'}` : 'No attachments'}</span>
      </div>

      {firstSource?.url && (
        <div className="session-card__source">
          <div className="session-card__source-brand">
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
          <div className="session-card__source-tail">
            {sourceCount > 1 && (
              <span className="session-card__source-more">+{sourceCount - 1} more</span>
            )}
          </div>
        </div>
      )}

      <div className="session-card__footer">
        <span className="session-card__open">Open session</span>
        {hasAttachments ? (
          <span className="session-card__footer-note">{session.attachments.length} attachment{session.attachments.length === 1 ? '' : 's'} attached</span>
        ) : (
          <span className="session-card__footer-note">Research transcript and saved answer</span>
        )}
      </div>

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
        {firstSource?.url && (
          <a
            href={firstSource.url}
            target="_blank"
            rel="noopener noreferrer"
            className="session-card__action session-card__action--external"
            onClick={(e) => e.stopPropagation()}
            title="Open source"
          >
            <ExternalLink size={14} />
          </a>
        )}
      </div>
    </div>
  );
}

export default SessionCard;
