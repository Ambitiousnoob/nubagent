import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check } from 'lucide-react';
import { useSettingsStore } from '../../store/useSettingsStore.js';

/**
 * ChatMessage Component
 * Displays individual chat messages with markdown and code highlighting
 * 
 * @param {object} message - Message object with role, content, timestamp
 * @param {boolean} isStreaming - Whether message is being streamed
 */
export function ChatMessage({ message, isStreaming = false }) {
  const { theme } = useSettingsStore();
  const isUser = message.role === 'user';
  const [copiedCode, setCopiedCode] = React.useState(null);

  const handleCopyCode = async (code, index) => {
    try {
      await navigator.clipboard.writeText(code);
      setCopiedCode(index);
      setTimeout(() => setCopiedCode(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className={`chat-message ${isUser ? 'chat-message--user' : 'chat-message--assistant'}`}>
      <div className="chat-message__avatar">
        {isUser ? (
          <div className="chat-message__avatar--user">U</div>
        ) : (
          <div className="chat-message__avatar--assistant">AI</div>
        )}
      </div>
      <div className="chat-message__content">
        <div className="chat-message__header">
          <span className="chat-message__role">
            {isUser ? 'You' : 'Assistant'}
          </span>
          <span className="chat-message__time">
            {formatTime(message.timestamp)}
          </span>
        </div>
        <div className="chat-message__body">
          {message.attachments?.length > 0 && (
            <div className="chat-message__attachments">
              {message.attachments.map((att, i) => (
                <div key={i} className="chat-message__attachment">
                  {att.kind === 'image' ? (
                    <img src={att.dataUrl} alt={att.name} className="chat-message__attachment-img" />
                  ) : (
                    <div className="chat-message__attachment-file">
                      <span className="chat-message__attachment-icon">📄</span>
                      <span className="chat-message__attachment-name">{att.name}</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
          <ReactMarkdown
            className="chat-message__markdown"
            components={{
              code({ node, inline, className, children, ...props }) {
                const match = /language-(\w+)/.exec(className || '');
                const code = String(children).replace(/\n$/, '');
                const codeIndex = `${message.id}-${i}`;

                if (!inline && match) {
                  return (
                    <div className="chat-message__code-block">
                      <div className="chat-message__code-header">
                        <span className="chat-message__code-lang">{match[1]}</span>
                        <button
                          className="chat-message__code-copy"
                          onClick={() => handleCopyCode(code, codeIndex)}
                          aria-label="Copy code"
                        >
                          {copiedCode === codeIndex ? (
                            <Check size={14} />
                          ) : (
                            <Copy size={14} />
                          )}
                        </button>
                      </div>
                      <SyntaxHighlighter
                        style={theme === 'dark' ? oneDark : oneLight}
                        language={match[1]}
                        PreTag="div"
                        {...props}
                      >
                        {code}
                      </SyntaxHighlighter>
                    </div>
                  );
                }

                return (
                  <code className={className} {...props}>
                    {children}
                  </code>
                );
              },
            }}
          >
            {message.content}
          </ReactMarkdown>
          {isStreaming && (
            <span className="chat-message__cursor">▊</span>
          )}
        </div>
      </div>
    </div>
  );
}

export default ChatMessage;
