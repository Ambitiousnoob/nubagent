/**
 * Middleware Exports
 * Centralized export for all middleware
 */

module.exports = {
  rateLimit: require('./rateLimit'),
  validate: require('./validate'),
  errorHandler: require('./errorHandler'),
  cors: require('./cors'),
};

// Export individual middleware
module.exports.rateLimitMiddleware = require('./rateLimit');
module.exports.validateMiddleware = require('./validate');
module.exports.errorHandlerMiddleware = require('./errorHandler');
module.exports.corsMiddleware = require('./cors');

// Export error classes
module.exports.errors = require('./errorHandler').errors;
module.exports.createError = require('./errorHandler').createError;
module.exports.asyncHandler = require('./errorHandler').asyncHandler;
module.exports.notFound = require('./errorHandler').notFound;

// Export validation schemas
module.exports.schemas = require('./validate').schemas;
module.exports.jsonBody = require('./validate').jsonBody;
