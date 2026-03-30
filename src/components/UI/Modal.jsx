import React, { useEffect, useId } from 'react';
import { X } from 'lucide-react';

let activeModalCount = 0;

/**
 * Modal Component
 * Reusable modal dialog with overlay
 * 
 * @param {boolean} isOpen - Modal visibility state
 * @param {function} onClose - Close handler
 * @param {string} title - Modal title
 * @param {React.ReactNode} children - Modal content
 * @param {React.ReactNode} footer - Modal footer content
 * @param {string} size - Modal size (sm, md, lg, xl)
 * @param {boolean} closeOnOverlay - Close when clicking overlay
 * @param {boolean} showCloseButton - Show close button
 */
export function Modal({
  isOpen,
  onClose,
  title,
  children,
  footer,
  size = 'md',
  closeOnOverlay = true,
  showCloseButton = true,
  className = '',
}) {
  const titleId = useId();

  // Handle escape key
  useEffect(() => {
    if (!isOpen) return;

    const handleEscape = (e) => {
      if (e.key === 'Escape') {
        onClose?.();
      }
    };

    document.addEventListener('keydown', handleEscape);
    activeModalCount += 1;
    document.body.style.overflow = 'hidden';

    return () => {
      document.removeEventListener('keydown', handleEscape);
      activeModalCount = Math.max(0, activeModalCount - 1);
      if (activeModalCount === 0) {
        document.body.style.overflow = '';
      }
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleOverlayClick = (e) => {
    if (closeOnOverlay && e.target === e.currentTarget) {
      onClose?.();
    }
  };

  return (
    <div className="modal-overlay" onClick={handleOverlayClick}>
      <div
        className={`modal modal--${size} ${className}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby={title ? titleId : undefined}
      >
        {(title || showCloseButton) && (
          <div className="modal__header">
            {title && <h2 id={titleId} className="modal__title">{title}</h2>}
            {showCloseButton && (
              <button className="modal__close" onClick={onClose} aria-label="Close modal">
                <X size={20} />
              </button>
            )}
          </div>
        )}
        <div className="modal__content">{children}</div>
        {footer && <div className="modal__footer">{footer}</div>}
      </div>
    </div>
  );
}

export default Modal;
