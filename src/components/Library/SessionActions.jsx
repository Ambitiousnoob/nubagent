import React from "react";
import { Trash2, Edit2, Share2, Download, Copy } from "lucide-react";
import { Button } from "../UI/Button.jsx";
import { Modal } from "../UI/Modal.jsx";
import {
  buildSessionShareUrl,
  promptToCopySessionUrl,
} from "../../lib/library.js";

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
  const [isEditing, setIsEditing] = React.useState(false);
  const [draft, setDraft] = React.useState({
    query: "",
    heading: "",
    body: "",
  });
  const sourceCount = session?.sources?.length || 0;
  const attachmentCount = session?.attachments?.length || 0;
  const summaryPreview = (session?.heading || session?.body || "")
    .replace(/[[\]#*`]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  React.useEffect(() => {
    if (!isOpen || !session) return;
    setConfirmDelete(false);
    setIsEditing(false);
    setDraft({
      query: session.query || "",
      heading: session.heading || "",
      body: session.body || "",
    });
  }, [isOpen, session]);

  const closeWithReset = () => {
    setConfirmDelete(false);
    setIsEditing(false);
    onClose?.();
  };

  const handleDelete = () => {
    onDelete?.(session.id);
    closeWithReset();
  };

  const handleShare = () => {
    onShare?.(session);
    closeWithReset();
  };

  const handleEditSave = () => {
    onEdit?.(session, {
      query: draft.query.trim() || session.query || "Untitled session",
      heading: draft.heading.trim(),
      body: draft.body.trim(),
    });
    closeWithReset();
  };

  const handleExport = (format) => {
    onExport?.(session, format);
    closeWithReset();
  };

  const handleCopyLink = async () => {
    try {
      const shareUrl = buildSessionShareUrl(session.id);
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(shareUrl);
      } else if (!promptToCopySessionUrl(shareUrl)) {
        throw new Error("Clipboard unavailable");
      }
      onShare?.(session, { copied: true });
      closeWithReset();
    } catch (err) {
      const shareUrl = buildSessionShareUrl(session.id);
      if (promptToCopySessionUrl(shareUrl)) {
        closeWithReset();
        return;
      }
      console.error("Failed to copy:", err);
    }
  };

  if (!isOpen) return null;

  return (
    <Modal
      isOpen={isOpen}
      onClose={closeWithReset}
      title={
        confirmDelete
          ? "Confirm Delete"
          : isEditing
            ? "Edit Session"
            : "Session Actions"
      }
      size="sm"
    >
      {confirmDelete ? (
        <div className="session-actions__confirm">
          <div className="session-actions__confirm-icon">Delete</div>
          <h3 className="session-actions__confirm-title">
            Delete this session?
          </h3>
          <p className="session-actions__confirm-text">
            "{session.query.slice(0, 50)}
            {session.query.length > 50 ? "..." : ""}"
          </p>
          <p className="session-actions__confirm-sub">
            This action cannot be undone.
          </p>
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
      ) : isEditing ? (
        <form
          className="session-actions__form"
          onSubmit={(event) => {
            event.preventDefault();
            handleEditSave();
          }}
        >
          <div className="session-actions__lead">
            <div className="session-actions__eyebrow">Session editor</div>
            <p className="session-actions__lead-copy">
              Update the saved label and summary without changing the
              transcript.
            </p>
          </div>
          <label className="session-actions__field">
            <span className="session-actions__label">Query</span>
            <input
              className="session-actions__input"
              type="text"
              value={draft.query}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  query: event.target.value,
                }))
              }
              placeholder="Search query"
              autoFocus
            />
          </label>
          <label className="session-actions__field">
            <span className="session-actions__label">Heading</span>
            <input
              className="session-actions__input"
              type="text"
              value={draft.heading}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  heading: event.target.value,
                }))
              }
              placeholder="Answer heading"
            />
          </label>
          <label className="session-actions__field">
            <span className="session-actions__label">Body</span>
            <textarea
              className="session-actions__textarea"
              value={draft.body}
              onChange={(event) =>
                setDraft((current) => ({
                  ...current,
                  body: event.target.value,
                }))
              }
              placeholder="Saved answer text"
              rows={8}
            />
          </label>
          <div className="session-actions__form-buttons">
            <Button
              type="button"
              variant="outline"
              onClick={() => setIsEditing(false)}
            >
              Cancel
            </Button>
            <Button type="submit" variant="primary">
              Save changes
            </Button>
          </div>
        </form>
      ) : (
        <div className="session-actions__menu">
          <div className="session-actions__summary">
            <div className="session-actions__eyebrow">Saved session</div>
            <h3 className="session-actions__summary-title">{session.query}</h3>
            <p className="session-actions__summary-copy">
              Share it, export it, edit the saved copy, or remove it.
            </p>
            {summaryPreview && (
              <p className="session-actions__summary-preview">
                {summaryPreview.slice(0, 180)}
                {summaryPreview.length > 180 ? "…" : ""}
              </p>
            )}
            <div className="session-actions__summary-meta">
              <span>
                {sourceCount} source{sourceCount === 1 ? "" : "s"}
              </span>
              <span>
                {attachmentCount} attachment{attachmentCount === 1 ? "" : "s"}
              </span>
            </div>
          </div>
          <button
            className="session-actions__item"
            onClick={() => setIsEditing(true)}
          >
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
            <button
              className="session-actions__item session-actions__item--nested"
              onClick={() => handleExport("markdown")}
            >
              <Download size={18} />
              <span>Markdown</span>
            </button>
            <button
              className="session-actions__item session-actions__item--nested"
              onClick={() => handleExport("json")}
            >
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
      <div className="bulk-actions__copy">
        <span className="bulk-actions__count">{selectedCount} selected</span>
        <small className="bulk-actions__note">
          Apply actions to the current selection.
        </small>
      </div>
      <div className="bulk-actions__buttons">
        <Button variant="outline" size="sm" onClick={onClearSelection}>
          Clear
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => onExport?.("markdown")}
        >
          <Download size={14} />
          Export
        </Button>
        <Button variant="danger" size="sm" onClick={() => onDelete?.()}>
          <Trash2 size={14} />
          Delete All
        </Button>
      </div>
    </div>
  );
}

export default SessionActions;
