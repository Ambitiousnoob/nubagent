import React, { useRef, useEffect } from 'react';
import { ChatMessage } from './ChatMessage.jsx';

/**
 * ChatHistory Component
 * Scrollable message list with auto-scroll
 * 
 * @param {array} messages - Array of message objects
 * @param {string} streamingMessageId - ID of message being streamed
 * @param {boolean} isLoading - Loading state
 */
export function ChatHistory({ messages = [], streamingMessageId, isLoading = false }) {
  const containerRef = useRef(null);
  const scrollRef = useRef(null);
  const shouldAutoScroll = useRef(true);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (shouldAutoScroll.current && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messages]);

  // Handle scroll to detect user scroll position
  const handleScroll = () => {
    if (!containerRef.current) return;

    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
    shouldAutoScroll.current = isNearBottom;
  };

  // Scroll to bottom button
  const scrollToBottom = () => {
    if (containerRef.current) {
      containerRef.current.scrollTo({
        top: containerRef.current.scrollHeight,
        behavior: 'smooth',
      });
      shouldAutoScroll.current = true;
    }
  };

  // Check if we should show scroll button
  const [showScrollButton, setShowScrollButton] = React.useState(false);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const checkScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      setShowScrollButton(scrollHeight - scrollTop - clientHeight > 200);
    };

    container.addEventListener('scroll', checkScroll);
    checkScroll();

    return () => container.removeEventListener('scroll', checkScroll);
  }, []);

  if (messages.length === 0 && !isLoading) {
    return (
      <div className="chat-history chat-history--empty" ref={containerRef}>
        <div className="chat-history__empty">
          <div className="chat-history__empty-icon">💬</div>
          <h3 className="chat-history__empty-title">Start a Conversation</h3>
          <p className="chat-history__empty-text">
            Ask a question or describe what you need help with.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-history" ref={containerRef} onScroll={handleScroll}>
      <div className="chat-history__content" ref={scrollRef}>
        {messages.map((message, index) => (
          <ChatMessage
            key={message.id}
            message={message}
            isStreaming={message.id === streamingMessageId}
          />
        ))}
        {isLoading && messages.length === 0 && (
          <div className="chat-history__loading">
            <div className="chat-history__loading-dots">
              <span className="chat-history__loading-dot" />
              <span className="chat-history__loading-dot" />
              <span className="chat-history__loading-dot" />
            </div>
            <span className="chat-history__loading-text">Thinking...</span>
          </div>
        )}
      </div>

      {showScrollButton && (
        <button
          className="chat-history__scroll-bottom"
          onClick={scrollToBottom}
          aria-label="Scroll to bottom"
        >
          ↓
        </button>
      )}
    </div>
  );
}

export default ChatHistory;
