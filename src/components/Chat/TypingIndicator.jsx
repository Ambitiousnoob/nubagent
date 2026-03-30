import React from 'react';

/**
 * TypingIndicator Component
 * Animated typing indicator for AI responses
 * 
 * @param {string} text - Optional text to display with indicator
 */
export function TypingIndicator({ text = 'Thinking' }) {
  return (
    <div className="typing-indicator">
      <div className="typing-indicator__dots">
        <span className="typing-indicator__dot" />
        <span className="typing-indicator__dot" />
        <span className="typing-indicator__dot" />
      </div>
      {text && (
        <span className="typing-indicator__text">
          {text}
          <span className="typing-indicator__dots-text">...</span>
        </span>
      )}
    </div>
  );
}

/**
 * StreamingProgress Component
 * Shows progress during streaming response
 * 
 * @param {number} progress - Progress percentage (0-100)
 * @param {string} status - Status text
 */
export function StreamingProgress({ progress = 0, status = 'Generating response' }) {
  return (
    <div className="streaming-progress">
      <div className="streaming-progress__header">
        <span className="streaming-progress__status">{status}</span>
        <span className="streaming-progress__percent">{Math.round(progress)}%</span>
      </div>
      <div className="streaming-progress__bar">
        <div
          className="streaming-progress__fill"
          style={{ width: `${progress}%` }}
        />
      </div>
    </div>
  );
}

export default TypingIndicator;
