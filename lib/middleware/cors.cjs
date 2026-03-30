/**
 * CORS middleware helper
 */

function resolveOrigin(origin, requestOrigin) {
  if (Array.isArray(origin)) {
    return origin.includes(requestOrigin) ? requestOrigin : origin[0];
  }
  return origin;
}

function setCorsHeaders(res, options = {}, req = null) {
  const origin = resolveOrigin(options.origin || '*', req?.headers?.origin);
  const methods = options.methods || 'GET,POST,PUT,PATCH,DELETE,OPTIONS';
  const headers = options.headers || 'Content-Type, Authorization, X-Requested-With, X-API-Key, X-State-Key';
  const credentials = options.credentials !== false;

  res.setHeader('Access-Control-Allow-Origin', origin);
  res.setHeader('Access-Control-Allow-Methods', methods);
  res.setHeader('Access-Control-Allow-Headers', headers);
  res.setHeader('Access-Control-Allow-Credentials', String(credentials));
}

function cors(options = {}) {
  return (req, res, next = () => {}) => {
    setCorsHeaders(res, options, req);

    if (req.method === 'OPTIONS') {
      res.status(204);
      res.end();
      return;
    }

    next();
  };
}

cors.setCorsHeaders = setCorsHeaders;

module.exports = cors;
