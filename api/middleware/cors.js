/**
 * CORS Configuration Middleware
 * Handles Cross-Origin Resource Sharing
 */

const defaultOptions = {
  origin: '*',
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-API-Key', 'X-State-Key'],
  exposedHeaders: ['X-RateLimit-Limit', 'X-RateLimit-Remaining', 'X-RateLimit-Reset'],
  credentials: true,
  maxAge: 86400, // 24 hours
};

/**
 * CORS middleware factory
 * @param {object} options - CORS options
 * @returns {function} Middleware function
 */
module.exports = function cors(options = {}) {
  const config = { ...defaultOptions, ...options };

  // Normalize origin to array
  const origins = Array.isArray(config.origin) ? config.origin : [config.origin];

  return (req, res, next) => {
    const requestOrigin = req.headers.origin;

    // Check if request origin is allowed
    let allowedOrigin = config.origin;
    if (origins.includes('*')) {
      allowedOrigin = '*';
    } else if (requestOrigin && origins.includes(requestOrigin)) {
      allowedOrigin = requestOrigin;
    }

    // Set CORS headers
    res.setHeader('Access-Control-Allow-Origin', allowedOrigin);
    res.setHeader('Access-Control-Allow-Methods', config.methods.join(', '));
    res.setHeader('Access-Control-Allow-Headers', config.allowedHeaders.join(', '));
    res.setHeader('Access-Control-Expose-Headers', config.exposedHeaders.join(', '));
    res.setHeader('Access-Control-Allow-Credentials', String(config.credentials));
    res.setHeader('Access-Control-Max-Age', String(config.maxAge));

    // Handle preflight requests
    if (req.method === 'OPTIONS') {
      res.status(204).end();
      return;
    }

    // Add Vary header for caching
    if (allowedOrigin !== '*') {
      res.setHeader('Vary', 'Origin');
    }

    next();
  };
};

/**
 * Pre-configured CORS presets
 */

// Public API - allow all origins
module.exports.public = cors({
  origin: '*',
  credentials: false,
});

// Restricted - specific origins only
module.exports.restricted = (allowedOrigins) => cors({
  origin: allowedOrigins,
  credentials: true,
});

// Development - allow localhost
module.exports.development = cors({
  origin: ['http://localhost:3000', 'http://localhost:5173', 'http://127.0.0.1:3000', 'http://127.0.0.1:5173'],
  credentials: true,
});

/**
 * Simple CORS headers helper (for individual routes)
 */
module.exports.setCorsHeaders = (res, options = {}) => {
  const config = { ...defaultOptions, ...options };
  res.setHeader('Access-Control-Allow-Origin', config.origin);
  res.setHeader('Access-Control-Allow-Methods', config.methods.join(', '));
  res.setHeader('Access-Control-Allow-Headers', config.allowedHeaders.join(', '));
  res.setHeader('Access-Control-Allow-Credentials', String(config.credentials));
};
