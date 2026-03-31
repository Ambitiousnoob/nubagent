const createError = (status, message) => {
  const error = new Error(message);
  error.status = status;
  return error;
};

const readBody = async (req, options = {}) => {
  const maxBytes = Number.isFinite(Number(options?.maxBytes))
    ? Number(options.maxBytes)
    : null;

  if (req.body && typeof req.body === "object") return req.body;

  if (typeof req.body === "string") {
    if (maxBytes && Buffer.byteLength(req.body, "utf8") > maxBytes) {
      throw createError(413, "Request body too large.");
    }
    return req.body.trim() ? JSON.parse(req.body) : {};
  }

  const chunks = [];
  let totalBytes = 0;

  for await (const chunk of req) {
    const buffer = Buffer.from(chunk);
    totalBytes += buffer.length;
    if (maxBytes && totalBytes > maxBytes) {
      throw createError(413, "Request body too large.");
    }
    chunks.push(buffer);
  }

  const raw = Buffer.concat(chunks).toString("utf8");
  return raw.trim() ? JSON.parse(raw) : {};
};

module.exports = {
  readBody,
};
