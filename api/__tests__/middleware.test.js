/**
 * API Middleware Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

describe('CORS Middleware', () => {
  let cors;
  let req, res, next;

  beforeEach(() => {
    req = { method: 'GET', headers: {} };
    res = { setHeader: vi.fn(), status: vi.fn().mockReturnThis(), end: vi.fn() };
    next = vi.fn();

    vi.resetModules();
    cors = require('../../lib/middleware/cors.cjs');
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
  
  it('exports a direct header helper', () => {
    cors.setCorsHeaders(res, { origin: 'https://example.com', credentials: false });

    expect(res.setHeader).toHaveBeenCalledWith('Access-Control-Allow-Origin', 'https://example.com');
    expect(res.setHeader).toHaveBeenCalledWith('Access-Control-Allow-Credentials', 'false');
  });
});
