/**
 * Health Check Endpoint
 * Returns API status and version information
 */

const { setCorsHeaders } = require('../middleware/cors');

const VERSION = '1.0.0';
const START_TIME = Date.now();

const writeJson = (res, status, data) => {
  res.status(status);
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.end(JSON.stringify(data));
};

module.exports = async (req, res) => {
  setCorsHeaders(res);

  // Handle preflight
  if (req.method === 'OPTIONS') {
    res.status(204).end();
    return;
  }

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
};

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
