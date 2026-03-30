import React, { forwardRef, useId } from 'react';
import { AlertCircle, CheckCircle2, XCircle } from 'lucide-react';

/**
 * Input Component
 * Styled input field with validation states
 * 
 * @param {string} type - Input type (text, email, password, etc.)
 * @param {string} value - Input value
 * @param {string} placeholder - Placeholder text
 * @param {string} label - Label text
 * @param {string} error - Error message
 * @param {string} success - Success message
 * @param {boolean} disabled - Disabled state
 * @param {function} onChange - Change handler
 * @param {function} onBlur - Blur handler
 * @param {string} className - Additional CSS classes
 */
export const Input = forwardRef(function Input(
  {
    type = 'text',
    value,
    placeholder,
    label,
    error,
    success,
    disabled = false,
    onChange,
    onBlur,
    className = '',
    icon,
    ...props
  },
  ref
) {
  const generatedId = useId();
  const inputId = props.id || generatedId;
  const hasError = Boolean(error);
  const hasSuccess = Boolean(success);
  const stateClass = hasError
    ? 'input--error'
    : hasSuccess
    ? 'input--success'
    : '';

  return (
    <div className={`input-wrapper ${className}`}>
      {label && (
        <label className="input__label" htmlFor={inputId}>
          {label}
          {props.required && <span className="input__required">*</span>}
        </label>
      )}
      <div className="input__container">
        {icon && <span className="input__icon">{icon}</span>}
        <input
          id={inputId}
          ref={ref}
          type={type}
          value={value}
          placeholder={placeholder}
          disabled={disabled}
          onChange={onChange}
          onBlur={onBlur}
          className={`input ${stateClass} ${icon ? 'input--with-icon' : ''}`}
          {...props}
        />
        {hasError && (
          <span className="input__status input__status--error">
            <XCircle size={16} />
          </span>
        )}
        {hasSuccess && (
          <span className="input__status input__status--success">
            <CheckCircle2 size={16} />
          </span>
        )}
      </div>
      {error && <span className="input__message input__message--error">{error}</span>}
      {success && <span className="input__message input__message--success">{success}</span>}
    </div>
  );
});

/**
 * Textarea Component
 * Styled textarea field
 */
export const Textarea = forwardRef(function Textarea(
  {
    value,
    placeholder,
    label,
    error,
    disabled = false,
    onChange,
    onBlur,
    className = '',
    rows = 4,
    ...props
  },
  ref
) {
  const generatedId = useId();
  const textareaId = props.id || generatedId;
  const hasError = Boolean(error);
  const stateClass = hasError ? 'input--error' : '';

  return (
    <div className={`input-wrapper ${className}`}>
      {label && (
        <label className="input__label" htmlFor={textareaId}>
          {label}
          {props.required && <span className="input__required">*</span>}
        </label>
      )}
      <textarea
        id={textareaId}
        ref={ref}
        value={value}
        placeholder={placeholder}
        disabled={disabled}
        onChange={onChange}
        onBlur={onBlur}
        rows={rows}
        className={`input input--textarea ${stateClass}`}
        {...props}
      />
      {error && <span className="input__message input__message--error">{error}</span>}
    </div>
  );
});

export default Input;
