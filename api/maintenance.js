import { getRuntimeConfig } from "../lib/config.js";
import { getConversationStore } from "../lib/history.js";
import { ensureProfileState } from "../lib/profile-state.js";

function sendJson(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(payload));
}

function readHeader(req, name) {
  const value = req.headers[name];
  return Array.isArray(value) ? value[0] : value;
}

function isAuthorizedRepairRequest(req, cronSecret) {
  const authorization = readHeader(req, "authorization");
  return authorization === `Bearer ${cronSecret}`;
}

export default async function handler(req, res) {
  const config = getRuntimeConfig();

  if (req.method !== "GET" && req.method !== "POST") {
    sendJson(res, 405, { error: "Method not allowed." });
    return;
  }

  if (!config.cronSecret) {
    sendJson(res, 404, { error: "Not found." });
    return;
  }

  if (!isAuthorizedRepairRequest(req, config.cronSecret)) {
    sendJson(res, 401, { error: "Unauthorized." });
    return;
  }

  if (!config.pageAccessToken) {
    sendJson(res, 500, {
      error: "Missing PAGE_ACCESS_TOKEN.",
    });
    return;
  }

  try {
    const store = await getConversationStore(config);
    await ensureProfileState(config);
    const cleanupResult = await store.cleanupOperationalData({
      retentionDays: config.reliabilityRetentionDays,
    });

    sendJson(res, 200, {
      ok: true,
      cleanup: cleanupResult,
    });
  } catch (error) {
    sendJson(res, 500, {
      error: error instanceof Error ? error.message : String(error),
    });
  }
}
