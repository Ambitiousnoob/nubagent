/**
 * API Middleware Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

describe('Rate Limit Middleware', () => {
  let rateLimit;
  let req, res, next;

  beforeEach(() => {
    // Mock request and response
    req = {
      headers: { 'x-forwarded-for': '127.0.0.1' },
      socket: { remoteAddress: '127.0.0.1' },
      url: '/api/test',
    };
    res = {
      setHeader: vi.fn(),
      status: vi.fn().mockReturnThis(),
      end: vi.fn(),
    };
    next = vi.fn();

    // Import fresh instance
    vi.resetModules();
    rateLimit = require('../../api/middleware/rateLimit');
  });

  it('allows requests under the limit', () => {
    const middleware = rateLimit({ windowMs: 60000, max: 100 });
    middleware(req, res, next);
    expect(next).toHaveBeenCalled();
    expect(res.status).not.toHaveBeenCalled();
  });

  it('blocks requests over the limit', () => {
    const middleware = rateLimit({ windowMs: 60000, max: 1 });
    
    // First request should pass
    middleware(req, res, next);
    expect(next).toHaveBeenCalled();
    
    // Reset mocks
    next.mockClear();
    res.status.mockClear();
    res.end.mockClear();
    
    // Second request should be blocked
    middleware(req, res, next);
    expect(next).not.toHaveBeenCalled();
    expect(res.status).toHaveBeenCalledWith(429);
    expect(res.end).toHaveBeenCalled();
  });

  it('sets rate limit headers', () => {
    const middleware = rateLimit({ windowMs: 60000, max: 100 });
    middleware(req, res, next);
    
    expect(res.setHeader).toHaveBeenCalledWith('X-RateLimit-Limit', 100);
    expect(res.setHeader).toHaveBeenCalledWith(
      'X-RateLimit-Remaining',
      expect.any(Number)
    );
    expect(res.setHeader).toHaveBeenCalledWith(
      'X-RateLimit-Reset',
      expect.any(Number)
    );
  });

  it('uses default configuration', () => {
    const middleware = rateLimit();
    middleware(req, res, next);
    expect(next).toHaveBeenCalled();
  });
});

describe('CORS Middleware', () => {
  let cors;
  let req, res, next;

  beforeEach(() => {
    req = { method: 'GET', headers: {} };
    res = { setHeader: vi.fn(), status: vi.fn().mockReturnThis(), end: vi.fn() };
    next = vi.fn();

    vi.resetModules();
    cors = require('../../api/middleware/cors');
  });

  it('sets CORS headers', () => {
    const middleware = cors();
    middleware(req, res, next);
    
    expect(res.setHeader).toHaveBeenCalledWith('Access-Control-Allow-Origin', '*');
    expect(res.setHeader).toHaveBeenCalledWith(
      'Access-Control-Allow-Methods',
      expect.stringContaining('GET')
    );
    expect(res.setHeader).toHaveBeenCalledWith(
      'Access-Control-Allow-Headers',
      expect.stringContaining('Content-Type')
    );
  });

  it('handles preflight requests', () => {
    req.method = 'OPTIONS';
    const middleware = cors();
    middleware(req, res, next);
    
    expect(res.status).toHaveBeenCalledWith(204);
    expect(res.end).toHaveBeenCalled();
    expect(next).not.toHaveBeenCalled();
  });

  it('respects custom origin', () => {
    const middleware = cors({ origin: 'https://example.com' });
    middleware(req, res, next);
    
    expect(res.setHeader).toHaveBeenCalledWith('Access-Control-Allow-Origin', 'https://example.com');
  });

  it('handles array of origins', () => {
    const middleware = cors({ origin: ['https://example.com', 'https://test.com'] });
    req.headers.origin = 'https://example.com';
    middleware(req, res, next);
    
    expect(res.setHeader).toHaveBeenCalledWith('Access-Control-Allow-Origin', 'https://example.com');
  });
});

describe('Error Handler Middleware', () => {
  let errorHandler, errors;

  beforeEach(() => {
    vi.resetModules();
    const eh = require('../../api/middleware/errorHandler');
    errorHandler = eh;
    errors = eh.errors;
  });

  it('creates AppError with correct properties', () => {
    const err = new errors.AppError('Test error', 400, 'TEST_ERROR');
    expect(err.message).toBe('Test error');
    expect(err.statusCode).toBe(400);
    expect(err.code).toBe('TEST_ERROR');
    expect(err.name).toBe('AppError');
  });

  it('creates ValidationError', () => {
    const err = new errors.ValidationError('Invalid input', [{ field: 'email', message: 'Required' }]);
    expect(err.message).toBe('Invalid input');
    expect(err.statusCode).toBe(400);
    expect(err.code).toBe('VALIDATION_ERROR');
    expect(err.details).toHaveLength(1);
  });

  it('creates NotFoundError', () => {
    const err = new errors.NotFoundError('Resource not found');
    expect(err.message).toBe('Resource not found');
    expect(err.statusCode).toBe(404);
    expect(err.code).toBe('NOT_FOUND');
  });

  it('creates UnauthorizedError', () => {
    const err = new errors.UnauthorizedError('Invalid token');
    expect(err.message).toBe('Invalid token');
    expect(err.statusCode).toBe(401);
    expect(err.code).toBe('UNAUTHORIZED');
  });

  it('creates RateLimitError', () => {
    const err = new errors.RateLimitError('Too many requests');
    expect(err.message).toBe('Too many requests');
    expect(err.statusCode).toBe(429);
    expect(err.code).toBe('RATE_LIMIT_EXCEEDED');
  });
});

describe('Validation Middleware', () => {
  let validate, schemas;

  beforeEach(() => {
    vi.resetModules();
    const v = require('../../api/middleware/validate');
    validate = v;
    schemas = v.schemas;
  });

  it('exports common schemas', () => {
    expect(schemas.chatRequest).toBeDefined();
    expect(schemas.searchRequest).toBeDefined();
    expect(schemas.sessionId).toBeDefined();
    expect(schemas.analyticsEvent).toBeDefined();
  });

  it('validates chat request schema', () => {
    const validData = {
      messages: [{ role: 'user', content: 'Hello' }],
    };
    const result = schemas.chatRequest.safeParse(validData);
    expect(result.success).toBe(true);
  });

  it('rejects invalid chat request', () => {
    const invalidData = {
      messages: [], // Empty messages
    };
    const result = schemas.chatRequest.safeParse(invalidData);
    expect(result.success).toBe(false);
  });

  it('validates search request schema', () => {
    const validData = {
      query: 'test query',
      limit: 10,
    };
    const result = schemas.searchRequest.safeParse(validData);
    expect(result.success).toBe(true);
  });

  it('rejects empty search query', () => {
    const invalidData = {
      query: '',
    };
    const result = schemas.searchRequest.safeParse(invalidData);
    expect(result.success).toBe(false);
  });
});
