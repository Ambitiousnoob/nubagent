import React from "react";
import { Loader2 } from "lucide-react";

/**
 * Button Component
 * Reusable button with multiple variants and states
 *
 * @param {string} variant - Button style (primary, secondary, outline, ghost, danger)
 * @param {string} size - Button size (sm, md, lg)
 * @param {boolean} isLoading - Loading state
 * @param {boolean} disabled - Disabled state
 * @param {string} className - Additional CSS classes
 * @param {React.ReactNode} children - Button content
 * @param {function} onClick - Click handler
 */
export function Button({
  variant = "primary",
  size = "md",
  isLoading = false,
  disabled = false,
  className = "",
  children,
  onClick,
  type = "button",
  ...props
}) {
  const baseClasses = "btn";
  const variantClasses = `btn--${variant}`;
  const sizeClasses = `btn--${size}`;
  const loadingClasses = isLoading ? "btn--loading" : "";
  const disabledClasses = disabled || isLoading ? "btn--disabled" : "";

  return (
    <button
      type={type}
      className={`${baseClasses} ${variantClasses} ${sizeClasses} ${loadingClasses} ${disabledClasses} ${className}`.trim()}
      onClick={onClick}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading && <Loader2 className="btn__spinner" size={16} />}
      <span className="btn__content">{children}</span>
    </button>
  );
}

export default Button;
