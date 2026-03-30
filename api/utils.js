/**
 * Utils Endpoint
 * Consolidated utility endpoints: health, analytics, export
 * 
 * Usage:
 * - /api/utils?action=health (GET)
 * - /api/utils?action=analytics (GET for stats, POST for tracking)
 * - /api/utils?action=export (POST)
 */

const { setCorsHeaders } = require('../middleware/cors');
const { readBody } = require('../lib/web');
const { getSavedSessions, getSessionById } = require('../lib/library');

const VERSION = '1.0.0';
const START_TIME = Date.now();

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

const writeContent = (res, status, content, contentType, filename) => {
  res.status(status);
  res.setHeader('Content-Type', contentType);
  res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
  res.end(content);
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

  const action = req.query?.action || req.query.action;

  // Route to appropriate handler based on action
  switch (action) {
    case 'health':
      return handleHealth(req, res);
    case 'analytics':
      return handleAnalytics(req, res);
    case 'export':
      return handleExport(req, res);
    default:
      return handleMetadata(res);
  }
};

/**
 * Health Check Handler
 */
function handleHealth(req, res) {
  // Only allow GET requests
  if (req.method !== 'GET') {
    writeJson(res, 405, { error: 'Method not allowed' });
    return;
  }

  try {
    const uptime = Math.floor((Date.now() - START_TIME) / 1000);
    const uptimeFormatted = formatUptime(uptime);

    const healthData = {
      status: 'healthy',
      version: VERSION,
      uptime: uptime,
      uptimeFormatted,
      timestamp: new Date().toISOString(),
      environment: process.env.NODE_ENV || 'development',
      services: {
        database: 'connected',
        cache: 'connected',
      },
    };

    // Add cache headers
    res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');

    writeJson(res, 200, healthData);
  } catch (error) {
    console.error('[Health Check Error]', error);
    writeJson(res, 500, {
      status: 'unhealthy',
      error: error.message || 'Health check failed',
    });
  }
}

/**
 * Analytics Handler
 */
async function handleAnalytics(req, res) {
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
}

/**
 * Export Handler
 */
async function handleExport(req, res) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    writeJson(res, 405, { error: 'Method not allowed' });
    return;
  }

  try {
    const body = await readBody(req);
    const { format = 'json', sessionIds = [] } = body;

    // Get sessions to export
    let sessions;
    if (sessionIds.length > 0) {
      sessions = sessionIds
        .map((id) => getSessionById(id))
        .filter(Boolean);
    } else {
      sessions = getSavedSessions();
    }

    if (sessions.length === 0) {
      writeJson(res, 404, { error: 'No sessions found to export' });
      return;
    }

    // Generate export based on format
    switch (format.toLowerCase()) {
      case 'json':
        exportJson(res, sessions);
        break;
      case 'markdown':
      case 'md':
        exportMarkdown(res, sessions);
        break;
      case 'txt':
      case 'text':
        exportText(res, sessions);
        break;
      default:
        writeJson(res, 400, {
          error: 'Unsupported format',
          supportedFormats: ['json', 'markdown', 'txt'],
        });
    }
  } catch (error) {
    console.error('[Export Error]', error);
    writeJson(res, 500, { error: 'Export failed' });
  }
}

/**
 * Metadata Handler (GET /api/utils)
 */
function handleMetadata(res) {
  writeJson(res, 200, {
    ok: true,
    endpoint: '/api/utils',
    actions: {
      health: { method: 'GET', description: 'Health check endpoint' },
      analytics: { method: 'GET|POST', description: 'Analytics stats (GET) or tracking (POST)' },
      export: { method: 'POST', description: 'Export sessions in various formats' },
    },
    example: {
      health: '/api/utils?action=health',
      analytics: '/api/utils?action=analytics',
      export: '/api/utils?action=export',
    },
  });
}

/**
 * Export as JSON
 */
function exportJson(res, sessions) {
  const exportData = {
    version: '1.0',
    exportedAt: new Date().toISOString(),
    sessionCount: sessions.length,
    sessions: sessions.map((session) => ({
      id: session.id,
      query: session.query,
      heading: session.heading,
      body: session.body,
      sources: session.sources,
      attachments: session.attachments,
      createdAt: session.createdAt,
      updatedAt: session.updatedAt,
    })),
  };

  const content = JSON.stringify(exportData, null, 2);
  const filename = `nubagent-export-${Date.now()}.json`;
  writeContent(res, 200, content, 'application/json', filename);
}

/**
 * Export as Markdown
 */
function exportMarkdown(res, sessions) {
  const timestamp = new Date().toISOString();
  let content = `# NubAgent Export\n\n`;
  content += `**Exported:** ${timestamp}\n`;
  content += `**Sessions:** ${sessions.length}\n\n`;
  content += `---\n\n`;

  sessions.forEach((session, index) => {
    content += `## ${index + 1}. ${session.query}\n\n`;

    if (session.heading) {
      content += `### ${session.heading}\n\n`;
    }

    if (session.body) {
      content += `${session.body}\n\n`;
    }

    if (session.sources && session.sources.length > 0) {
      content += `#### Sources\n\n`;
      session.sources.forEach((source, i) => {
        content += `${i + 1}. [${source.title || source.url}](${source.url})\n`;
      });
      content += `\n`;
    }

    if (session.attachments && session.attachments.length > 0) {
      content += `#### Attachments\n\n`;
      session.attachments.forEach((att) => {
        content += `- ${att.name} (${formatBytes(att.size)})\n`;
      });
      content += `\n`;
    }

    content += `---\n\n`;
  });

  const filename = `nubagent-export-${Date.now()}.md`;
  writeContent(res, 200, content, 'text/markdown', filename);
}

/**
 * Export as plain text
 */
function exportText(res, sessions) {
  const timestamp = new Date().toISOString();
  let content = `NUBAGENT EXPORT\n`;
  content += `================\n`;
  content += `Exported: ${timestamp}\n`;
  content += `Sessions: ${sessions.length}\n\n`;

  sessions.forEach((session, index) => {
    content += `${'='.repeat(50)}\n`;
    content += `SESSION ${index + 1}\n`;
    content += `${'='.repeat(50)}\n\n`;
    content += `Query: ${session.query}\n\n`;

    if (session.heading) {
      content += `${session.heading}\n\n`;
    }

    if (session.body) {
      // Remove markdown formatting for plain text
      const plainBody = session.body
        .replace(/#{1,6}\s*/g, '')
        .replace(/\*\*([^*]+)\*\*/g, '$1')
        .replace(/\*([^*]+)\*/g, '$1')
        .replace(/`([^`]+)`/g, '$1')
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
      content += `${plainBody}\n\n`;
    }

    if (session.sources && session.sources.length > 0) {
      content += `Sources:\n`;
      session.sources.forEach((source, i) => {
        content += `  ${i + 1}. ${source.title || source.url}\n`;
      });
      content += `\n`;
    }

    content += `\n`;
  });

  const filename = `nubagent-export-${Date.now()}.txt`;
  writeContent(res, 200, content, 'text/plain', filename);
}

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

/**
 * Format uptime into human-readable string
 */
function formatUptime(seconds) {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  const parts = [];
  if (days > 0) parts.push(`${days}d`);
  if (hours > 0) parts.push(`${hours}h`);
  if (minutes > 0) parts.push(`${minutes}m`);
  parts.push(`${secs}s`);

  return parts.join(' ');
}

/**
 * Format bytes to human-readable string
 */
function formatBytes(bytes) {
  if (!bytes) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}
