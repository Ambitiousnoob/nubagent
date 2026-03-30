/**
 * Request Validation Middleware
 * Validates request body, query, and params using Zod schemas
 */

const { z } = require('zod');

/**
 * Validation error response
 */
function validationError(res, errors) {
  res.status(400);
  res.setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify({
    error: 'Validation failed',
    details: errors.map((err) => ({
      field: err.path.join('.'),
      message: err.message,
    })),
  }));
}

/**
 * Validation middleware factory
 * @param {object} schemas - Zod schemas for validation
 * @param {z.ZodSchema} schemas.body - Schema for request body
 * @param {z.ZodSchema} schemas.query - Schema for query parameters
 * @param {z.ZodSchema} schemas.params - Schema for URL parameters
 * @returns {function} Middleware function
 */
module.exports = function validate(schemas = {}) {
  return (req, res, next) => {
    const errors = [];

    // Validate body
    if (schemas.body && req.body !== undefined) {
      const result = schemas.body.safeParse(req.body);
      if (!result.success) {
        errors.push(...result.error.errors);
      } else {
        req.body = result.data; // Use parsed/transformed data
      }
    }

    // Validate query
    if (schemas.query && req.query !== undefined) {
      const result = schemas.query.safeParse(req.query);
      if (!result.success) {
        errors.push(...result.error.errors);
      } else {
        req.query = result.data;
      }
    }

    // Validate params
    if (schemas.params && req.params !== undefined) {
      const result = schemas.params.safeParse(req.params);
      if (!result.success) {
        errors.push(...result.error.errors);
      } else {
        req.params = result.data;
      }
    }

    if (errors.length > 0) {
      validationError(res, errors);
      return;
    }

    next();
  };
};

/**
 * Common Zod schemas for reuse
 */
module.exports.schemas = {
  // Chat request schema
  chatRequest: z.object({
    model: z.string().optional(),
    messages: z.array(z.object({
      role: z.enum(['system', 'user', 'assistant']),
      content: z.string(),
    })).min(1),
    stream: z.boolean().optional(),
    temperature: z.number().min(0).max(2).optional(),
    max_tokens: z.number().positive().optional(),
  }),

  // Search request schema
  searchRequest: z.object({
    query: z.string().min(1).max(2000),
    limit: z.number().int().positive().max(100).optional(),
    offset: z.number().int().nonnegative().optional(),
  }),

  // Session ID schema
  sessionId: z.object({
    id: z.string().min(1),
  }),

  // Export request schema
  exportRequest: z.object({
    format: z.enum(['json', 'markdown', 'pdf']),
    sessionIds: z.array(z.string()).optional(),
  }),

  // Analytics event schema
  analyticsEvent: z.object({
    event: z.string().min(1),
    properties: z.record(z.any()).optional(),
    timestamp: z.number().optional(),
  }),

  // Pagination schema
  pagination: z.object({
    page: z.number().int().positive().optional(),
    limit: z.number().int().positive().max(100).optional(),
    sortBy: z.string().optional(),
    sortOrder: z.enum(['asc', 'desc']).optional(),
  }),

  // API key schema
  apiKey: z.object({
    key: z.string().min(1),
    provider: z.string().optional(),
  }),
};

/**
 * Validate JSON body middleware (simplified)
 */
module.exports.jsonBody = (req, res, next) => {
  if (['POST', 'PUT', 'PATCH'].includes(req.method)) {
    const contentType = req.headers['content-type'];
    if (!contentType || !contentType.includes('application/json')) {
      res.status(400);
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify({ error: 'Content-Type must be application/json' }));
      return;
    }
  }
  next();
};
