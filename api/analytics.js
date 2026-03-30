/**
 * Analytics Endpoint
 * Track user events and application metrics
 */

const { setCorsHeaders } = require('../middleware/cors');
const { readBody } = require('../lib/web');

// In-memory analytics store (use database in production)
const analyticsStore = {
  events: [],
  sessions: new Map(),
};

// Clean up old events periodically
setInterval(() => {
  const oneHourAgo = Date.now() - 60 * 60 * 1000;
  analyticsStore.events = analyticsStore.events.filter(
    (event) => event.timestamp > oneHourAgo
  );
}, 60 * 60 * 1000);

const writeJson = (res, status, data) => {
  res.status(status);
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.end(JSON.stringify(data));
};

const getClientInfo = (req) => {
  const userAgent = req.headers['user-agent'] || '';
  const ip = req.headers['x-forwarded-for']?.split(',')[0] ||
             req.headers['x-real-ip'] ||
             req.socket?.remoteAddress ||
             'unknown';

  // Simple user agent parsing
  let browser = 'Unknown';
  let os = 'Unknown';

  if (userAgent.includes('Chrome')) browser = 'Chrome';
  else if (userAgent.includes('Firefox')) browser = 'Firefox';
  else if (userAgent.includes('Safari')) browser = 'Safari';
  else if (userAgent.includes('Edge')) browser = 'Edge';

  if (userAgent.includes('Windows')) os = 'Windows';
  else if (userAgent.includes('Mac')) os = 'macOS';
  else if (userAgent.includes('Linux')) os = 'Linux';
  else if (userAgent.includes('Android')) os = 'Android';
  else if (userAgent.includes('iOS')) os = 'iOS';

  return { browser, os, ip };
};

module.exports = async (req, res) => {
  setCorsHeaders(res);

  // Handle preflight
  if (req.method === 'OPTIONS') {
    res.status(204).end();
    return;
  }

  // Handle GET for stats
  if (req.method === 'GET') {
    const stats = getAnalyticsStats();
    writeJson(res, 200, stats);
    return;
  }

  // Handle POST for tracking
  if (req.method === 'POST') {
    try {
      const body = await readBody(req);
      const { event, properties = {}, sessionId } = body;

      if (!event || typeof event !== 'string') {
        writeJson(res, 400, { error: 'Event name is required' });
        return;
      }

      const clientInfo = getClientInfo(req);
      const timestamp = Date.now();

      const analyticsEvent = {
        id: generateId(),
        event,
        properties,
        sessionId: sessionId || generateSessionId(),
        timestamp,
        ...clientInfo,
      };

      // Store event
      analyticsStore.events.push(analyticsEvent);

      // Track session
      if (!analyticsStore.sessions.has(analyticsEvent.sessionId)) {
        analyticsStore.sessions.set(analyticsEvent.sessionId, {
          id: analyticsEvent.sessionId,
          startedAt: timestamp,
          lastActivity: timestamp,
          events: [],
        });
      }

      const session = analyticsStore.sessions.get(analyticsEvent.sessionId);
      session.events.push(analyticsEvent.id);
      session.lastActivity = timestamp;

      // Return success
      writeJson(res, 200, {
        success: true,
        eventId: analyticsEvent.id,
        sessionId: analyticsEvent.sessionId,
      });
    } catch (error) {
      console.error('[Analytics Error]', error);
      writeJson(res, 500, { error: 'Failed to track event' });
    }
    return;
  }

  // Method not allowed
  writeJson(res, 405, { error: 'Method not allowed' });
};

/**
 * Get analytics statistics
 */
function getAnalyticsStats() {
  const now = Date.now();
  const oneHourAgo = now - 60 * 60 * 1000;
  const oneDayAgo = now - 24 * 60 * 60 * 1000;

  const recentEvents = analyticsStore.events.filter((e) => e.timestamp > oneHourAgo);
  const dailyEvents = analyticsStore.events.filter((e) => e.timestamp > oneDayAgo);

  // Count events by type
  const eventCounts = {};
  recentEvents.forEach((event) => {
    eventCounts[event.event] = (eventCounts[event.event] || 0) + 1;
  });

  // Active sessions
  const activeSessions = Array.from(analyticsStore.sessions.values())
    .filter((s) => s.lastActivity > oneHourAgo)
    .length;

  return {
    totalEvents: analyticsStore.events.length,
    eventsLastHour: recentEvents.length,
    eventsLastDay: dailyEvents.length,
    eventCounts,
    activeSessions,
    totalSessions: analyticsStore.sessions.size,
  };
}

/**
 * Generate unique ID
 */
function generateId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

/**
 * Generate session ID
 */
function generateSessionId() {
  return `sess_${generateId()}`;
}
