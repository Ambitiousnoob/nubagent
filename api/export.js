/**
 * Export Endpoint
 * Export conversations and sessions in various formats
 */

const { setCorsHeaders } = require('../middleware/cors');
const { readBody } = require('../lib/web');
const { getSavedSessions, getSessionById } = require('../lib/library');

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

module.exports = async (req, res) => {
  setCorsHeaders(res);

  // Handle preflight
  if (req.method === 'OPTIONS') {
    res.status(204).end();
    return;
  }

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
};

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
 * Format bytes to human-readable string
 */
function formatBytes(bytes) {
  if (!bytes) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}
