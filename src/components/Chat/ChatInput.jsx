import React, { useRef, useState, useCallback } from 'react';
import { Send, Image, X, Paperclip } from 'lucide-react';
import { Button } from '../UI/Button.jsx';

/**
 * ChatInput Component
 * Input area for chat with send button and image upload
 * 
 * @param {function} onSend - Send message handler
 * @param {function} onAttachmentAdd - Attachment add handler
 * @param {array} attachments - Current attachments
 * @param {boolean} disabled - Disabled state
 * @param {boolean} isLoading - Loading state
 */
export function ChatInput({
  onSend,
  onAttachmentAdd,
  onAttachmentRemove,
  attachments = [],
  disabled = false,
  isLoading = false,
  placeholder = 'Type your message...',
}) {
  const [value, setValue] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  const MAX_TEXT_LENGTH = 8000;

  const handleSubmit = useCallback(() => {
    if (!value.trim() || isLoading || disabled) return;
    onSend?.(value.trim());
    setValue('');
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }, [value, isLoading, disabled, onSend]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleChange = (e) => {
    const newValue = e.target.value.slice(0, MAX_TEXT_LENGTH);
    setValue(newValue);
    // Auto-resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  };

  const handleFileSelect = useCallback((e) => {
    const files = Array.from(e.target.files || []);
    files.forEach((file) => {
      const reader = new FileReader();
      reader.onload = (event) => {
        const isImage = file.type.startsWith('image/');
        onAttachmentAdd?.({
          id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
          name: file.name,
          kind: isImage ? 'image' : 'text',
          size: file.size,
          dataUrl: isImage ? event.target.result : '',
          file,
        });
      };
      if (isImage) {
        reader.readAsDataURL(file);
      } else {
        reader.readAsText(file);
      }
    });
    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, [onAttachmentAdd]);

  const triggerFileSelect = () => {
    fileInputRef.current?.click();
  };

  const remainingChars = MAX_TEXT_LENGTH - value.length;
  const isNearLimit = remainingChars < 500;
  const isOverLimit = remainingChars <= 0;

  return (
    <div className={`chat-input ${isFocused ? 'chat-input--focused' : ''} ${disabled ? 'chat-input--disabled' : ''}`}>
      {attachments.length > 0 && (
        <div className="chat-input__attachments">
          {attachments.map((att) => (
            <div key={att.id} className="chat-input__attachment">
              {att.kind === 'image' ? (
                <img src={att.dataUrl} alt={att.name} className="chat-input__attachment-preview" />
              ) : (
                <div className="chat-input__attachment-file">
                  <Paperclip size={14} />
                  <span className="chat-input__attachment-name">{att.name}</span>
                </div>
              )}
              <button
                className="chat-input__attachment-remove"
                onClick={() => onAttachmentRemove?.(att.id)}
                aria-label={`Remove ${att.name}`}
              >
                <X size={14} />
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="chat-input__container">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          placeholder={placeholder}
          disabled={disabled || isLoading}
          className="chat-input__textarea"
          rows={1}
        />
        <div className="chat-input__actions">
          <div className="chat-input__chars">
            <span className={`chat-input__chars-count ${isNearLimit ? 'chat-input__chars-count--warning' : ''} ${isOverLimit ? 'chat-input__chars-count--error' : ''}`}>
              {remainingChars}
            </span>
          </div>
          <Button
            variant="primary"
            size="sm"
            onClick={handleSubmit}
            disabled={!value.trim() || isLoading || disabled || isOverLimit}
            isLoading={isLoading}
            className="chat-input__send"
          >
            <Send size={16} />
          </Button>
        </div>
      </div>

      <div className="chat-input__toolbar">
        <button
          className="chat-input__toolbar-btn"
          onClick={triggerFileSelect}
          disabled={disabled || isLoading}
          aria-label="Attach file"
        >
          <Image size={16} />
          <span>Attach</span>
        </button>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*,.txt,.md,.json,.csv,.js,.ts,.jsx,.tsx,.py,.rb,.go,.rs,.java,.c,.h,.cpp,.hpp,.html,.css,.scss,.xml,.yaml,.yml"
        multiple
        onChange={handleFileSelect}
        className="chat-input__file-input"
      />
    </div>
  );
}

export default ChatInput;
