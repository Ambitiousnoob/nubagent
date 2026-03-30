import React from "react";

/**
 * Skeleton Component
 * Loading placeholder for content
 *
 * @param {string} variant - Skeleton variant (text, circle, rect)
 * @param {string} width - Width of skeleton
 * @param {string} height - Height of skeleton
 * @param {number} lines - Number of lines for text variant
 * @param {string} className - Additional CSS classes
 */
export function Skeleton({
  variant = "text",
  width,
  height,
  lines = 1,
  className = "",
}) {
  const baseClasses = "skeleton";
  const variantClasses = `skeleton--${variant}`;

  if (variant === "text" && lines > 1) {
    return (
      <div className={`${baseClasses} ${baseClasses}--lines ${className}`}>
        {Array.from({ length: lines }).map((_, i) => (
          <div
            key={i}
            className={`${baseClasses}__line ${variantClasses}`}
            style={{
              width: i === lines - 1 ? "60%" : undefined,
              ...styleProps(width, height),
            }}
          />
        ))}
      </div>
    );
  }

  return (
    <div
      className={`${baseClasses} ${variantClasses} ${className}`}
      style={styleProps(width, height)}
    />
  );
}

function styleProps(width, height) {
  const style = {};
  if (width) style.width = width;
  if (height) style.height = height;
  return style;
}

/**
 * Card Skeleton
 * Pre-built skeleton for card components
 */
export function CardSkeleton() {
  return (
    <div className="skeleton-card">
      <Skeleton variant="rect" height="160px" />
      <div className="skeleton-card__content">
        <Skeleton variant="text" width="80%" height="20px" />
        <Skeleton variant="text" lines={2} />
      </div>
    </div>
  );
}

/**
 * List Skeleton
 * Pre-built skeleton for list items
 */
export function ListSkeleton({ count = 5 }) {
  return (
    <div className="skeleton-list">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="skeleton-list__item">
          <Skeleton variant="circle" width="40px" height="40px" />
          <div className="skeleton-list__content">
            <Skeleton variant="text" width="60%" height="16px" />
            <Skeleton variant="text" width="40%" height="14px" />
          </div>
        </div>
      ))}
    </div>
  );
}

export default Skeleton;
