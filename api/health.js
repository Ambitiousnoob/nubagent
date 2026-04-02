import {
  getRuntimeConfig,
  hasMessagingConfig,
  hasVerificationConfig,
} from "../lib/config.js";
import { getConversationStore } from "../lib/history.js";

function readHeader(req, name) {
  const value = req.headers[name];
  return Array.isArray(value) ? value[0] : value;
}

function isAuthorizedRequest(req, cronSecret) {
  if (!cronSecret) {
    return true;
  }

  return readHeader(req, "authorization") === `Bearer ${cronSecret}`;
}

function sendJson(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(payload));
}

export default async function handler(req, res) {
  if (req.method !== "GET") {
    sendJson(res, 405, { error: "Method not allowed." });
    return;
  }

  const config = getRuntimeConfig();
  const authorized = isAuthorizedRequest(req, config.cronSecret);
  const verificationReady = hasVerificationConfig(config);
  const messagingReady = hasMessagingConfig(config);
  let dbReachable = false;
  let snapshot = null;

  try {
    const store = await getConversationStore(config);
    await store.ensureSchema();
    snapshot = await store.getOperationalSnapshot({
      lookbackHours: config.healthLookbackHours,
    });
    dbReachable = true;
  } catch {
    dbReachable = false;
  }

  const ok = verificationReady && messagingReady && dbReachable;
  const response = {
    ok,
    status: ok ? "ok" : "degraded",
    timestamp: new Date().toISOString(),
  };

  if (authorized) {
    response.checks = {
      verificationReady,
      messagingReady,
      dbReachable,
    };
    response.missing = {
      verification: config.missingVerificationKeys,
      messaging: config.missingMessagingKeys,
    };
    response.lookbackHours = config.healthLookbackHours;
    response.metrics = snapshot;
  }

  sendJson(res, ok ? 200 : 503, response);
}
