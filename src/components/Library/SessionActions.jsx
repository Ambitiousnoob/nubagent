import React from 'react';
import { Trash2, Edit2, Share2, Download, Copy } from 'lucide-react';
import { Button } from '../UI/Button.jsx';
import { Modal } from '../UI/Modal.jsx';

/**
 * SessionActions Component
 * Edit/delete/share actions for sessions
 * 
 * @param {object} session - Session object
 * @param {function} onDelete - Delete handler
 * @param {function} onEdit - Edit handler
 * @param {function} onShare - Share handler
 * @param {function} onExport - Export handler
 * @param {boolean} isOpen - Modal visibility
 * @param {function} onClose - Close handler
 */
export function SessionActions({
  session,
  onDelete,
  onEdit,
  onShare,
  onExport,
  isOpen = false,
  onClose,
}) {
  const [confirmDelete, setConfirmDelete] = React.useState(false);

  const handleDelete = () => {
    onDelete?.(session.id);
    onClose?.();
  };

  const handleShare = () => {
    onShare?.(session);
    onClose?.();
  };

  const handleEdit = () => {
    onEdit?.(session);
    onClose?.();
  };

  const handleExport = (format) => {
    onExport?.(session, format);
    onClose?.();
  };

  const handleCopyLink = async () => {
    try {
      const shareUrl = `${window.location.origin}/session/${session.id}`;
      await navigator.clipboard.writeText(shareUrl);
      onShare?.(session, { copied: true });
      onClose?.();
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  if (!isOpen) return null;

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title={confirmDelete ? 'Confirm Delete' : 'Session Actions'}
      size="sm"
    >
      {confirmDelete ? (
        <div className="session-actions__confirm">
          <div className="session-actions__confirm-icon">⚠️</div>
          <h3 className="session-actions__confirm-title">Delete this session?</h3>
          <p className="session-actions__confirm-text">
            "{session.query.slice(0, 50)}{session.query.length > 50 ? '...' : ''}"
          </p>
          <p className="session-actions__confirm-sub">This action cannot be undone.</p>
          <div className="session-actions__confirm-buttons">
            <Button variant="outline" onClick={() => setConfirmDelete(false)}>
              Cancel
            </Button>
            <Button variant="danger" onClick={handleDelete}>
              <Trash2 size={16} />
              Delete
            </Button>
          </div>
        </div>
      ) : (
        <div className="session-actions__menu">
          <button className="session-actions__item" onClick={handleEdit}>
            <Edit2 size={18} />
            <span>Edit Session</span>
          </button>
          <button className="session-actions__item" onClick={handleShare}>
            <Share2 size={18} />
            <span>Share</span>
          </button>
          <button className="session-actions__item" onClick={handleCopyLink}>
            <Copy size={18} />
            <span>Copy Link</span>
          </button>
          <div className="session-actions__divider" />
          <div className="session-actions__submenu">
            <span className="session-actions__submenu-label">Export as</span>
            <button className="session-actions__item session-actions__item--nested" onClick={() => handleExport('markdown')}>
              <Download size={18} />
              <span>Markdown</span>
            </button>
            <button className="session-actions__item session-actions__item--nested" onClick={() => handleExport('json')}>
              <Download size={18} />
              <span>JSON</span>
            </button>
          </div>
          <div className="session-actions__divider" />
          <button
            className="session-actions__item session-actions__item--danger"
            onClick={() => setConfirmDelete(true)}
          >
            <Trash2 size={18} />
            <span>Delete Session</span>
          </button>
        </div>
      )}
    </Modal>
  );
}

/**
 * BulkActions Component
 * Actions for multiple selected sessions
 */
export function BulkActions({
  selectedCount,
  onDelete,
  onExport,
  onClearSelection,
}) {
  if (selectedCount === 0) return null;

  return (
    <div className="bulk-actions">
      <span className="bulk-actions__count">{selectedCount} selected</span>
      <div className="bulk-actions__buttons">
        <Button variant="outline" size="sm" onClick={onClearSelection}>
          Clear
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => onExport?.('markdown')}
        >
          <Download size={14} />
          Export
        </Button>
        <Button
          variant="danger"
          size="sm"
          onClick={() => onDelete?.()}
        >
          <Trash2 size={14} />
          Delete All
        </Button>
      </div>
    </div>
  );
}

export default SessionActions;
