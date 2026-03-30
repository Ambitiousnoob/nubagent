/**
 * API Endpoint Tests
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

describe('Health Endpoint', () => {
  let healthHandler;
  let req, res;

  beforeEach(() => {
    req = { method: 'GET', headers: {} };
    res = {
      status: vi.fn().mockReturnThis(),
      setHeader: vi.fn(),
      end: vi.fn(),
    };

    vi.resetModules();
    healthHandler = require('../../api/health');
  });

  it('returns healthy status', async () => {
    await healthHandler(req, res);
    
    expect(res.status).toHaveBeenCalledWith(200);
    expect(res.setHeader).toHaveBeenCalledWith('Content-Type', 'application/json; charset=utf-8');
    
    const response = JSON.parse(res.end.mock.calls[0][0]);
    expect(response.status).toBe('healthy');
    expect(response.version).toBeDefined();
    expect(response.uptime).toBeDefined();
  });

  it('includes version information', async () => {
    await healthHandler(req, res);
    
    const response = JSON.parse(res.end.mock.calls[0][0]);
    expect(response.version).toMatch(/^\d+\.\d+\.\d+$/);
  });

  it('includes uptime', async () => {
    await healthHandler(req, res);
    
    const response = JSON.parse(res.end.mock.calls[0][0]);
    expect(response.uptime).toBeGreaterThanOrEqual(0);
    expect(response.uptimeFormatted).toBeDefined();
  });

  it('handles non-GET methods', async () => {
    req.method = 'POST';
    await healthHandler(req, res);
    
    expect(res.status).toHaveBeenCalledWith(405);
    const response = JSON.parse(res.end.mock.calls[0][0]);
    expect(response.error).toBe('Method not allowed');
  });

  it('handles OPTIONS preflight', async () => {
    req.method = 'OPTIONS';
    await healthHandler(req, res);
    
    expect(res.status).toHaveBeenCalledWith(204);
    expect(res.end).toHaveBeenCalled();
  });
});

describe('Analytics Endpoint', () => {
  let analyticsHandler;
  let req, res;

  beforeEach(() => {
    req = {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'user-agent': 'Test Agent',
      },
      socket: { remoteAddress: '127.0.0.1' },
    };
    res = {
      status: vi.fn().mockReturnThis(),
      setHeader: vi.fn(),
      end: vi.fn(),
    };

    vi.resetModules();
    analyticsHandler = require('../../api/analytics');
  });

  it('tracks events successfully', async () => {
    const bodyPromise = Promise.resolve({
      event: 'test_event',
      properties: { key: 'value' },
    });
    req.body = await bodyPromise;

    // Mock readBody
    vi.mock('../../lib/web', () => ({
      readBody: vi.fn().mockResolvedValue({
        event: 'test_event',
        properties: { key: 'value' },
      }),
    }));

    await analyticsHandler(req, res);
    
    expect(res.status).toHaveBeenCalledWith(200);
    const response = JSON.parse(res.end.mock.calls[0][0]);
    expect(response.success).toBe(true);
    expect(response.eventId).toBeDefined();
  });

  it('requires event name', async () => {
    vi.mock('../../lib/web', () => ({
      readBody: vi.fn().mockResolvedValue({}),
    }));

    await analyticsHandler(req, res);
    
    expect(res.status).toHaveBeenCalledWith(400);
    const response = JSON.parse(res.end.mock.calls[0][0]);
    expect(response.error).toBeDefined();
  });

  it('returns stats on GET', async () => {
    req.method = 'GET';
    await analyticsHandler(req, res);
    
    expect(res.status).toHaveBeenCalledWith(200);
    const response = JSON.parse(res.end.mock.calls[0][0]);
    expect(response.totalEvents).toBeDefined();
    expect(response.eventsLastHour).toBeDefined();
  });

  it('handles OPTIONS preflight', async () => {
    req.method = 'OPTIONS';
    await analyticsHandler(req, res);
    
    expect(res.status).toHaveBeenCalledWith(204);
  });
});

describe('Export Endpoint', () => {
  let exportHandler;
  let req, res;

  beforeEach(() => {
    req = {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
    };
    res = {
      status: vi.fn().mockReturnThis(),
      setHeader: vi.fn(),
      end: vi.fn(),
    };

    vi.resetModules();
    exportHandler = require('../../api/export');
  });

  it('exports sessions as JSON', async () => {
    vi.mock('../../lib/library', () => ({
      getSavedSessions: vi.fn().mockReturnValue([
        { id: '1', query: 'Test', body: 'Content' },
      ]),
      getSessionById: vi.fn(),
    }));

    vi.mock('../../lib/web', () => ({
      readBody: vi.fn().mockResolvedValue({ format: 'json' }),
    }));

    await exportHandler(req, res);
    
    expect(res.setHeader).toHaveBeenCalledWith('Content-Type', 'application/json');
    expect(res.setHeader).toHaveBeenCalledWith(
      'Content-Disposition',
      expect.stringContaining('.json')
    );
  });

  it('exports sessions as Markdown', async () => {
    vi.mock('../../lib/library', () => ({
      getSavedSessions: vi.fn().mockReturnValue([
        { id: '1', query: 'Test', body: 'Content' },
      ]),
      getSessionById: vi.fn(),
    }));

    vi.mock('../../lib/web', () => ({
      readBody: vi.fn().mockResolvedValue({ format: 'markdown' }),
    }));

    await exportHandler(req, res);
    
    expect(res.setHeader).toHaveBeenCalledWith('Content-Type', 'text/markdown');
    expect(res.setHeader).toHaveBeenCalledWith(
      'Content-Disposition',
      expect.stringContaining('.md')
    );
  });

  it('handles empty sessions', async () => {
    vi.mock('../../lib/library', () => ({
      getSavedSessions: vi.fn().mockReturnValue([]),
      getSessionById: vi.fn(),
    }));

    vi.mock('../../lib/web', () => ({
      readBody: vi.fn().mockResolvedValue({ format: 'json' }),
    }));

    await exportHandler(req, res);
    
    expect(res.status).toHaveBeenCalledWith(404);
    const response = JSON.parse(res.end.mock.calls[0][0]);
    expect(response.error).toBe('No sessions found to export');
  });

  it('rejects unsupported formats', async () => {
    vi.mock('../../lib/library', () => ({
      getSavedSessions: vi.fn().mockReturnValue([
        { id: '1', query: 'Test' },
      ]),
      getSessionById: vi.fn(),
    }));

    vi.mock('../../lib/web', () => ({
      readBody: vi.fn().mockResolvedValue({ format: 'xml' }),
    }));

    await exportHandler(req, res);
    
    expect(res.status).toHaveBeenCalledWith(400);
    const response = JSON.parse(res.end.mock.calls[0][0]);
    expect(response.error).toBe('Unsupported format');
  });

  it('handles non-POST methods', async () => {
    req.method = 'GET';
    await exportHandler(req, res);
    
    expect(res.status).toHaveBeenCalledWith(405);
    const response = JSON.parse(res.end.mock.calls[0][0]);
    expect(response.error).toBe('Method not allowed');
  });
});
