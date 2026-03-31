const { readBody } = require("../lib/web");
const {
  metadataPayload,
  createChatResponsePayload,
  runGeminiChat,
} = require("../lib/gemini-chat");

const DEFAULT_MAX_BODY_BYTES = 64 * 1024;
const envLimit = Number.parseInt(process.env.CHAT_BODY_LIMIT_BYTES || "", 10);
const MAX_BODY_BYTES =
  Number.isFinite(envLimit) && envLimit > 0 ? envLimit : DEFAULT_MAX_BODY_BYTES;

const writeCorsHeaders = (res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
};

const sendJson = (res, status, payload) => {
  res.status(status);
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(payload));
};

module.exports = async (req, res) => {
  writeCorsHeaders(res);

  if (req.method === "OPTIONS") {
    res.status(204).end();
    return;
  }

  if (req.method === "HEAD" || req.method === "GET") {
    sendJson(res, 200, metadataPayload());
    return;
  }

  if (req.method !== "POST") {
    sendJson(res, 405, { error: "Method not allowed" });
    return;
  }

  let body;
  try {
    body = await readBody(req, { maxBytes: MAX_BODY_BYTES });
  } catch (error) {
    sendJson(res, Number(error?.status) || 400, {
      error:
        error?.status === 413
          ? error.message
          : "Invalid JSON body",
    });
    return;
  }

  try {
    const result = await runGeminiChat(body);
    sendJson(res, 200, createChatResponsePayload(result));
  } catch (error) {
    if (Number(error?.status) >= 500 || !error?.status) {
      console.error("[nub-agent API Error]", error);
    }
    sendJson(res, Number(error?.status) || 500, {
      error: error?.message || "Chat request failed.",
    });
  }
};
