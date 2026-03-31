import { createHmac, timingSafeEqual } from "node:crypto";

import {
  getRuntimeConfig,
  hasMessagingConfig,
  hasVerificationConfig,
} from "../lib/config.js";
import { getConversationStore } from "../lib/history.js";
import { generateGeminiReply } from "../lib/gemini.js";
import { sendSenderAction, sendTextMessage } from "../lib/messenger.js";

const ATTACHMENT_FALLBACK =
  "I can reply to text messages right now. Send text and I will forward it to Gemini.";
const UPSTREAM_FAILURE_REPLY =
  "I hit an upstream error while talking to Gemini. Please try again in a moment.";

export function sendJson(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(payload));
}

export function sendText(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader("Content-Type", "text/plain; charset=utf-8");
  res.end(payload);
}

export function buildRequestUrl(req) {
  return new URL(req.url, `https://${req.headers.host || "localhost"}`);
}

function readHeader(req, name) {
  const value = req.headers[name];

  return Array.isArray(value) ? value[0] : value;
}

async function readRawBody(req) {
  const chunks = [];

  for await (const chunk of req) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }

  return Buffer.concat(chunks);
}

export function verifySignature(rawBody, signatureHeader, appSecret) {
  if (!appSecret) {
    return true;
  }

  if (!signatureHeader?.startsWith("sha256=")) {
    return false;
  }

  const providedSignature = signatureHeader.slice("sha256=".length);
  const expectedSignature = createHmac("sha256", appSecret)
    .update(rawBody)
    .digest("hex");

  if (providedSignature.length !== expectedSignature.length) {
    return false;
  }

  return timingSafeEqual(
    Buffer.from(providedSignature, "utf8"),
    Buffer.from(expectedSignature, "utf8"),
  );
}

export function extractInboundPrompt(event) {
  if (event?.message?.is_echo) {
    return "";
  }

  if (typeof event?.message?.text === "string") {
    return event.message.text.trim();
  }

  if (typeof event?.postback?.payload === "string") {
    return `Postback payload: ${event.postback.payload}`;
  }

  return "";
}

export function extractSourceEventId(event) {
  return event?.message?.mid || event?.postback?.mid || null;
}

function withStage(error, stage) {
  if (error instanceof Error) {
    error.stage = stage;
    return error;
  }

  const wrapped = new Error(String(error));
  wrapped.stage = stage;
  return wrapped;
}

function handleVerification(req, res, config) {
  if (!hasVerificationConfig(config)) {
    sendJson(res, 500, {
      error: `Missing environment variable(s): ${config.missingVerificationKeys.join(", ")}`,
    });
    return;
  }

  const url = buildRequestUrl(req);
  const mode = url.searchParams.get("hub.mode");
  const token = url.searchParams.get("hub.verify_token");
  const challenge = url.searchParams.get("hub.challenge");

  if (mode !== "subscribe" || !challenge) {
    sendJson(res, 400, {
      error: "Missing Messenger webhook verification parameters.",
    });
    return;
  }

  if (token !== config.verifyToken) {
    sendJson(res, 403, {
      error: "Invalid verify token.",
    });
    return;
  }

  sendText(res, 200, challenge);
}

export function createWebhookHandler({
  configLoader = getRuntimeConfig,
  conversationStoreFactory = getConversationStore,
  geminiReply = generateGeminiReply,
  sendAction = sendSenderAction,
  sendTextMessageImpl = sendTextMessage,
  logger = console,
} = {}) {
  async function processEvent(event, config) {
    const senderId = event?.sender?.id;

    if (!senderId || event?.delivery || event?.read) {
      return;
    }

    const prompt = extractInboundPrompt(event);
    const sourceEventId = extractSourceEventId(event);

    await sendAction(senderId, "mark_seen", config).catch(() => {});

    if (!prompt) {
      if (event?.message?.attachments?.length) {
        await sendTextMessageImpl(senderId, ATTACHMENT_FALLBACK, config).catch(
          () => {},
        );
      }

      return;
    }

    let typingEnabled = false;

    try {
      const store = await conversationStoreFactory(config);
      const inboundResult = await store
        .saveInboundTurn({
          senderId,
          text: prompt,
          sourceEventId,
        })
        .catch((error) => {
          throw withStage(error, "db:save_inbound");
        });

      if (!inboundResult.inserted) {
        logger.info?.("Skipping duplicate Messenger event", {
          senderId,
          sourceEventId,
        });
        return;
      }

      await sendAction(senderId, "typing_on", config).catch(() => {});
      typingEnabled = true;

      const history = await store
        .getConversationHistory(senderId, {
          excludeMessageId: inboundResult.messageId,
        })
        .catch((error) => {
          throw withStage(error, "db:load_history");
        });

      const reply = await geminiReply({
        prompt,
        history,
        config,
      }).catch((error) => {
        throw withStage(error, "gemini");
      });

      await sendTextMessageImpl(senderId, reply, config).catch((error) => {
        throw withStage(error, "messenger:send_text");
      });

      await store.saveModelTurn({ senderId, text: reply }).catch((error) => {
        throw withStage(error, "db:save_model");
      });
    } catch (error) {
      logger.error?.("Failed to process Messenger event", {
        senderId,
        sourceEventId,
        stage: error?.stage || "unknown",
        message: error instanceof Error ? error.message : String(error),
      });

      await sendTextMessageImpl(senderId, UPSTREAM_FAILURE_REPLY, config).catch(
        () => {},
      );
    } finally {
      if (typingEnabled) {
        await sendAction(senderId, "typing_off", config).catch(() => {});
      }
    }
  }

  return async function handler(req, res) {
    const config = configLoader();

    if (req.method === "GET") {
      handleVerification(req, res, config);
      return;
    }

    if (req.method !== "POST") {
      sendJson(res, 405, { error: "Method not allowed." });
      return;
    }

    if (!hasMessagingConfig(config)) {
      sendJson(res, 500, {
        error: `Missing environment variable(s): ${config.missingMessagingKeys.join(", ")}`,
      });
      return;
    }

    const rawBody = await readRawBody(req);
    const signature = readHeader(req, "x-hub-signature-256");

    if (!verifySignature(rawBody, signature, config.facebookAppSecret)) {
      sendJson(res, 403, { error: "Invalid webhook signature." });
      return;
    }

    let payload;

    try {
      payload = rawBody.length > 0 ? JSON.parse(rawBody.toString("utf8")) : {};
    } catch {
      sendJson(res, 400, { error: "Invalid JSON body." });
      return;
    }

    if (payload.object !== "page") {
      sendJson(res, 404, { error: "Unsupported webhook object." });
      return;
    }

    const tasks = [];

    for (const entry of payload.entry ?? []) {
      for (const event of entry.messaging ?? []) {
        tasks.push(processEvent(event, config));
      }
    }

    await Promise.allSettled(tasks);

    sendText(res, 200, "EVENT_RECEIVED");
  };
}

export default createWebhookHandler();
