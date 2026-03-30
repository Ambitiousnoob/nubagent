/**
 * Utils Endpoint
 * Consolidated utility endpoints: health, analytics, export, messenger
 *
 * Usage:
 * - /api/utils?action=health (GET)
 * - /api/utils?action=analytics (GET for stats, POST for tracking)
 * - /api/utils?action=export (POST)
 * - /api/utils?action=messenger (GET for verification, POST for webhooks)
 */

const path = require("node:path");
const dotenv = require("dotenv");
dotenv.config();
dotenv.config({ path: path.join(__dirname, "..", ".env") });

const { setCorsHeaders } = require('../lib/middleware/cors');
const { readBody } = require('../lib/web');
const { getSavedSessions, getSessionById } = require('../lib/db');
const { runLiteHostChat, PUBLIC_MODEL_NAME, PUBLIC_DEVELOPER_NAME } = require("../lib/litehost-chat");

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

const sendText = (res, status, value) => {
  res.status(status);
  res.setHeader("Content-Type", "text/plain; charset=utf-8");
  res.end(String(value));
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

// ============================================================================
// MESSENGER WEBHOOK FUNCTIONS
// ============================================================================

const DEFAULT_FB_GRAPH_API = "https://graph.facebook.com/v21.0";
const DEFAULT_MESSENGER_SYSTEM_PROMPT = [
    `You are ${PUBLIC_MODEL_NAME} on Messenger.`,
    `${PUBLIC_DEVELOPER_NAME} built you.`,
    "Reply with concise, useful text suitable for Facebook Messenger.",
    "Prefer short paragraphs and avoid markdown tables.",
    "If the user asks for a long answer, keep it readable and practical.",
].join(" ");

const getVerifyToken = () => (
    String(process.env.MESSENGER_VERIFY_TOKEN || process.env.VERIFY_TOKEN || "").trim()
);

const getPageAccessToken = () => String(process.env.PAGE_ACCESS_TOKEN || "").trim();

const getGraphApiBase = () => (
    String(process.env.FB_GRAPH_API || DEFAULT_FB_GRAPH_API).trim().replace(/\/+$/, "")
);

const getMessengerSystemPrompt = () => (
    String(process.env.MESSENGER_SYSTEM_PROMPT || DEFAULT_MESSENGER_SYSTEM_PROMPT).trim()
);

const getMaxMessageChars = () => {
    const value = Number(process.env.MESSENGER_MAX_MESSAGE_CHARS) || 640;
    return Math.max(160, Math.min(1800, value));
};

const normalizeReplyText = (value) => {
    const text = String(value || "").replace(/\r\n/g, "\n").trim();
    return text || "I'm here, but I couldn't generate a useful reply just now. Please try again.";
};

const splitLongLine = (line, maxChars) => {
    const chunks = [];
    let rest = String(line || "").trim();

    while (rest.length > maxChars) {
        let splitAt = rest.lastIndexOf(" ", maxChars);
        if (splitAt < Math.floor(maxChars * 0.5)) splitAt = maxChars;
        chunks.push(rest.slice(0, splitAt).trim());
        rest = rest.slice(splitAt).trim();
    }

    if (rest) chunks.push(rest);
    return chunks;
};

const splitMessengerText = (value, maxChars = getMaxMessageChars()) => {
    const text = normalizeReplyText(value);
    const chunks = [];
    const paragraphs = text.split(/\n{2,}/).map((paragraph) => paragraph.trim()).filter(Boolean);

    for (const paragraph of paragraphs) {
        if (paragraph.length <= maxChars) {
            chunks.push(paragraph);
            continue;
        }

        const lines = paragraph.split("\n").map((line) => line.trim()).filter(Boolean);
        for (const line of lines) {
            chunks.push(...splitLongLine(line, maxChars));
        }
    }

    return chunks.length ? chunks : [text];
};

const extractGraphError = async (response) => {
    const raw = await response.text();
    try {
        const parsed = JSON.parse(raw);
        if (parsed?.error?.message) return parsed.error.message;
    } catch {
        return raw;
    }
    return raw;
};

const sendMessengerMessage = async (recipientId, text) => {
    const accessToken = getPageAccessToken();
    if (!accessToken) {
        const error = new Error("PAGE_ACCESS_TOKEN is not configured on the server.");
        error.status = 503;
        throw error;
    }

    const response = await fetch(`${getGraphApiBase()}/me/messages?access_token=${encodeURIComponent(accessToken)}`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            recipient: { id: recipientId },
            message: { text },
            messaging_type: "RESPONSE",
        }),
    });

    if (!response.ok) {
        const error = new Error(`Facebook Graph API error: ${response.status} ${await extractGraphError(response)}`);
        error.status = response.status;
        throw error;
    }

    return response.json();
};

const sendMessengerReply = async (recipientId, text) => {
    const parts = splitMessengerText(text);
    const receipts = [];

    for (const part of parts) {
        receipts.push(await sendMessengerMessage(recipientId, part));
    }

    return receipts;
};

const getIncomingText = (event) => {
    const messageText = String(event?.message?.text || "").trim();
    if (messageText) return messageText;

    const postbackTitle = String(event?.postback?.title || "").trim();
    if (postbackTitle) return postbackTitle;

    const postbackPayload = String(event?.postback?.payload || "").trim();
    return postbackPayload;
};

const isReplyableEvent = (event) => {
    if (!event || typeof event !== "object") return false;
    if (!event.sender?.id) return false;
    if (event.message?.is_echo) return false;
    if (!event.message && !event.postback) return false;
    return true;
};

const buildMessengerChatBody = (incomingText) => ({
    messages: [
        { role: "system", content: getMessengerSystemPrompt() },
        { role: "user", content: incomingText },
    ],
    stream: false,
});

const handleMessagingEvent = async (event) => {
    if (!isReplyableEvent(event)) return;

    const senderId = String(event.sender.id).trim();

    try {
        const incomingText = getIncomingText(event);

        if (!incomingText) {
            await sendMessengerReply(
                senderId,
                "I can reply to text messages right now. Send a question or prompt and I'll answer there.",
            );
            return;
        }

        const { reply } = await runLiteHostChat(buildMessengerChatBody(incomingText));
        await sendMessengerReply(senderId, reply.content);
    } catch (error) {
        console.error("[Messenger Bot Error]", error);
        try {
            await sendMessengerReply(
                senderId,
                `${PUBLIC_MODEL_NAME} is temporarily unavailable. Please try again in a moment.`,
            );
        } catch (fallbackError) {
            console.error("[Messenger Bot Fallback Error]", fallbackError);
        }
    }
};

const handleMessengerWebhook = async (req, res) => {
    if (req.method === "GET") {
        const mode = String(req.query?.["hub.mode"] || req.query?.hub_mode || "").trim();
        const verifyToken = String(req.query?.["hub.verify_token"] || req.query?.hub_verify_token || "").trim();
        const challenge = String(req.query?.["hub.challenge"] || req.query?.hub_challenge || "").trim();

        if (!mode && !verifyToken && !challenge) {
            writeJson(res, 200, {
                ok: true,
                endpoint: "/api/messenger",
                brand: PUBLIC_MODEL_NAME,
                developer: PUBLIC_DEVELOPER_NAME,
                verify_token_env: process.env.MESSENGER_VERIFY_TOKEN ? "MESSENGER_VERIFY_TOKEN" : "VERIFY_TOKEN",
                graph_api_env: "FB_GRAPH_API",
                page_access_token_env: "PAGE_ACCESS_TOKEN",
            });
            return;
        }

        if (mode === "subscribe" && verifyToken && challenge && verifyToken === getVerifyToken()) {
            sendText(res, 200, challenge);
            return;
        }

        writeJson(res, 403, { error: "Messenger webhook verification failed." });
        return;
    }

    if (req.method !== "POST") {
        writeJson(res, 405, { error: "Method not allowed" });
        return;
    }

    let body;
    try {
        body = await readBody(req);
    } catch (error) {
        writeJson(res, 400, { error: "Invalid JSON body" });
        return;
    }

    if (body?.object !== "page" || !Array.isArray(body.entry)) {
        writeJson(res, 400, { error: "Unsupported Messenger webhook payload." });
        return;
    }

    for (const entry of body.entry) {
        const events = Array.isArray(entry?.messaging) ? entry.messaging : [];
        for (const event of events) {
            await handleMessagingEvent(event);
        }
    }

    sendText(res, 200, "EVENT_RECEIVED");
};

// ============================================================================
// MAIN UTILS ENDPOINT
// ============================================================================

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
    case 'messenger':
      return handleMessengerWebhook(req, res);
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
      messenger: { method: 'GET|POST', description: 'Facebook Messenger webhook (GET for verification, POST for events)' },
    },
    example: {
      health: '/api/utils?action=health',
      analytics: '/api/utils?action=analytics',
      export: '/api/utils?action=export',
      messenger: '/api/utils?action=messenger',
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
