/**
 * Rate Limiting Middleware
 * In-memory rate limiting with configurable limits
 */

// In-memory store for rate limiting (use Redis in production)
const rateLimitStore = new Map();

/**
 * Clean up expired entries periodically
 */
setInterval(() => {
  const now = Date.now();
  for (const [key, data] of rateLimitStore.entries()) {
    if (data.resetAt < now) {
      rateLimitStore.delete(key);
    }
  }
}, 60000); // Clean up every minute

/**
 * Rate limit configuration
 */
const defaultConfig = {
  windowMs: 60 * 1000, // 1 minute window
  max: 100, // 100 requests per window
  message: { error: 'Too many requests, please try again later.' },
  statusCode: 429,
};

/**
 * Generate a unique key for each request
 */
const generateKey = (req) => {
  // Use IP address, or fallback to a generic key
  const ip = req.headers['x-forwarded-for']?.split(',')[0] ||
             req.headers['x-real-ip'] ||
             req.socket?.remoteAddress ||
             'unknown';
  return `ratelimit:${ip}:${req.url}`;
};

/**
 * Rate limiting middleware factory
 * @param {object} config - Rate limit configuration
 * @returns {function} Middleware function
 */
module.exports = function rateLimit(config = {}) {
  const options = { ...defaultConfig, ...config };

  return (req, res, next) => {
    const key = generateKey(req);
    const now = Date.now();

    let record = rateLimitStore.get(key);

    if (!record) {
      record = {
        count: 1,
        resetAt: now + options.windowMs,
      };
      rateLimitStore.set(key, record);
    } else if (now > record.resetAt) {
      // Reset the window
      record = {
        count: 1,
        resetAt: now + options.windowMs,
      };
      rateLimitStore.set(key, record);
    } else {
      // Increment count
      record.count++;
    }

    // Set rate limit headers
    res.setHeader('X-RateLimit-Limit', options.max);
    res.setHeader('X-RateLimit-Remaining', Math.max(0, options.max - record.count));
    res.setHeader('X-RateLimit-Reset', Math.ceil(record.resetAt / 1000));

    // Check if limit exceeded
    if (record.count > options.max) {
      res.setHeader('Retry-After', Math.ceil((record.resetAt - now) / 1000));
      res.status(options.statusCode);
      res.setHeader('Content-Type', 'application/json');
      res.end(JSON.stringify(options.message));
      return;
    }

    next();
  };
};

/**
 * Pre-configured rate limiters for different use cases
 */
module.exports.strict = rateLimit({
  windowMs: 60 * 1000,
  max: 10,
});

module.exports.moderate = rateLimit({
  windowMs: 60 * 1000,
  max: 50,
});

module.exports.relaxed = rateLimit({
  windowMs: 60 * 1000,
  max: 200,
});
