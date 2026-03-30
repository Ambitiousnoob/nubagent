/**
 * Centralized Error Handler Middleware
 * Catches and formats all errors consistently
 */

/**
 * Custom error classes
 */
class AppError extends Error {
  constructor(message, statusCode = 500, code = 'INTERNAL_ERROR') {
    super(message);
    this.name = 'AppError';
    this.statusCode = statusCode;
    this.code = code;
    this.isOperational = true;

    Error.captureStackTrace(this, this.constructor);
  }
}

class ValidationError extends AppError {
  constructor(message = 'Validation error', details = []) {
    super(message, 400, 'VALIDATION_ERROR');
    this.name = 'ValidationError';
    this.details = details;
  }
}

class NotFoundError extends AppError {
  constructor(message = 'Resource not found') {
    super(message, 404, 'NOT_FOUND');
    this.name = 'NotFoundError';
  }
}

class UnauthorizedError extends AppError {
  constructor(message = 'Unauthorized') {
    super(message, 401, 'UNAUTHORIZED');
    this.name = 'UnauthorizedError';
  }
}

class ForbiddenError extends AppError {
  constructor(message = 'Forbidden') {
    super(message, 403, 'FORBIDDEN');
    this.name = 'ForbiddenError';
  }
}

class RateLimitError extends AppError {
  constructor(message = 'Too many requests') {
    super(message, 429, 'RATE_LIMIT_EXCEEDED');
    this.name = 'RateLimitError';
  }
}

/**
 * Format error response
 */
function formatError(err, isDev = false) {
  const error = {
    error: err.message || 'An unexpected error occurred',
    code: err.code || 'INTERNAL_ERROR',
  };

  if (err.details) {
    error.details = err.details;
  }

  if (isDev && err.stack) {
    error.stack = err.stack;
  }

  return error;
}

/**
 * Error handler middleware
 */
module.exports = function errorHandler(options = {}) {
  const { isDev = process.env.NODE_ENV === 'development' } = options;

  return (err, req, res, next) => {
    // Determine status code
    let statusCode = err.statusCode || err.status || 500;

    // Don't leak server errors in production
    if (statusCode === 500 && !isDev) {
      console.error('Unhandled error:', err);
    }

    // Log error
    if (statusCode >= 500) {
      console.error(`[Error ${statusCode}]`, err.message, err.stack);
    } else {
      console.log(`[Error ${statusCode}]`, err.message);
    }

    // Send response
    res.status(statusCode);
    res.setHeader('Content-Type', 'application/json');

    // Add CORS headers if not already set
    if (!res.getHeader('Access-Control-Allow-Origin')) {
      res.setHeader('Access-Control-Allow-Origin', '*');
    }

    res.end(JSON.stringify(formatError(err, isDev)));
  };
};

/**
 * Async handler wrapper to catch errors
 */
module.exports.asyncHandler = (fn) => (req, res, next) => {
  Promise.resolve(fn(req, res, next)).catch(next);
};

/**
 * Not found handler (404)
 */
module.exports.notFound = (req, res, next) => {
  const err = new NotFoundError(`Cannot ${req.method} ${req.url}`);
  next(err);
};

/**
 * Export error classes
 */
module.exports.errors = {
  AppError,
  ValidationError,
  NotFoundError,
  UnauthorizedError,
  ForbiddenError,
  RateLimitError,
};

/**
 * Create specific error types
 */
module.exports.createError = {
  validation: (message, details) => new ValidationError(message, details),
  notFound: (message) => new NotFoundError(message),
  unauthorized: (message) => new UnauthorizedError(message),
  forbidden: (message) => new ForbiddenError(message),
  rateLimit: (message) => new RateLimitError(message),
  app: (message, statusCode, code) => new AppError(message, statusCode, code),
};
