import { createHmac, timingSafeEqual } from "node:crypto";
import { waitUntil } from "@vercel/functions";

import {
  getRuntimeConfig,
  hasMessagingConfig,
  hasVerificationConfig,
} from "../lib/config.js";
import { getConversationStore } from "../lib/history.js";
import { generateGeminiReply } from "../lib/gemini.js";
import { sendSenderAction, sendTextMessage } from "../lib/messenger.js";
import { maybeRepairProfileState } from "../lib/profile-state.js";
import {
  buildImageContextPrompt,
  buildImageReadyReply,
  buildPromptWithImageContext,
  buildStoredInboundText,
  buildVisionPrompt,
  extractImageAttachments,
  loadInlineImageParts,
} from "../lib/vision.js";

const ATTACHMENT_FALLBACK =
  "I can reply to text and supported image messages right now. Send text or a PNG, JPEG, WEBP, HEIC, or HEIF image.";
const UPSTREAM_FAILURE_REPLY =
  "I hit an upstream error while talking to Gemini. Please try again in a moment.";
const TYPING_REFRESH_INTERVAL_MS = 5000;

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

async function sendActionSafely({
  senderId,
  action,
  config,
  sendAction,
  logger,
}) {
  try {
    await sendAction(senderId, action, config);
    return true;
  } catch (error) {
    logger.warn?.("Failed to send Messenger sender action", {
      senderId,
      action,
      message: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
}

function createTypingController({ senderId, config, sendAction, logger }) {
  let stopped = false;
  let heartbeatTimer = null;
  let actionQueue = Promise.resolve();

  const enqueueSenderAction = (action) => {
    actionQueue = actionQueue.then(async () => {
      if (stopped && action === "typing_on") {
        return false;
      }

      return sendActionSafely({
        senderId,
        action,
        config,
        sendAction,
        logger,
      });
    });

    return actionQueue;
  };

  const scheduleHeartbeat = () => {
    heartbeatTimer = setTimeout(async () => {
      heartbeatTimer = null;

      if (stopped) {
        return;
      }

      await enqueueSenderAction("typing_on");

      if (!stopped) {
        scheduleHeartbeat();
      }
    }, TYPING_REFRESH_INTERVAL_MS);

    if (typeof heartbeatTimer?.unref === "function") {
      heartbeatTimer.unref();
    }
  };

  return {
    async start() {
      const started = await enqueueSenderAction("typing_on");

      if (started && !stopped) {
        scheduleHeartbeat();
      }

      return started;
    },

    async stop() {
      if (stopped) {
        return;
      }

      stopped = true;

      if (heartbeatTimer) {
        clearTimeout(heartbeatTimer);
        heartbeatTimer = null;
      }

      await enqueueSenderAction("typing_off");
    },
  };
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

    const imageAttachments = extractImageAttachments(event);
    const rawPrompt = extractInboundPrompt(event);
    const prompt = buildVisionPrompt(rawPrompt, imageAttachments.length);
    const sourceEventId = extractSourceEventId(event);

    await sendActionSafely({
      senderId,
      action: "mark_seen",
      config,
      sendAction,
      logger,
    });

    if (!prompt) {
      if (event?.message?.attachments?.length) {
        await sendTextMessageImpl(senderId, ATTACHMENT_FALLBACK, config).catch(
          () => {},
        );
      }

      return;
    }

    const typingController = createTypingController({
      senderId,
      config,
      sendAction,
      logger,
    });
    let typingEnabled = false;

    try {
      typingEnabled = await typingController.start();

      const store = await conversationStoreFactory(config);
      const inboundResult = await store
        .saveInboundTurn({
          senderId,
          text: buildStoredInboundText(rawPrompt, imageAttachments.length),
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

      const inlineParts = await loadInlineImageParts(imageAttachments).catch(
        (error) => {
          throw withStage(error, "attachments:load_inline_images");
        },
      );

      if (
        imageAttachments.length > 0 &&
        inlineParts.length === 0 &&
        !rawPrompt
      ) {
        await sendTextMessageImpl(senderId, ATTACHMENT_FALLBACK, config).catch(
          () => {},
        );
        await store
          .saveModelTurn({ senderId, text: ATTACHMENT_FALLBACK })
          .catch(() => {});
        return;
      }

      if (imageAttachments.length > 0 && !rawPrompt) {
        const imageSummary = await geminiReply({
          prompt: buildImageContextPrompt(inlineParts.length),
          history: [],
          config,
          inlineParts,
        }).catch((error) => {
          throw withStage(error, "gemini:image_context");
        });

        await store
          .saveLatestImageContext({
            senderId,
            summary: imageSummary,
            sourceEventId,
          })
          .catch((error) => {
            throw withStage(error, "db:save_image_context");
          });

        const imageReadyReply = buildImageReadyReply(
          imageSummary,
          inlineParts.length,
        );

        await sendTextMessageImpl(senderId, imageReadyReply, config).catch(
          (error) => {
            throw withStage(error, "messenger:send_text");
          },
        );

        await store
          .saveModelTurn({ senderId, text: imageReadyReply })
          .catch((error) => {
            throw withStage(error, "db:save_model");
          });

        return;
      }

      const history = await store
        .getConversationHistory(senderId, {
          excludeMessageId: inboundResult.messageId,
        })
        .catch((error) => {
          throw withStage(error, "db:load_history");
        });

      const imageContext =
        imageAttachments.length === 0
          ? await store.getLatestImageContext(senderId).catch((error) => {
              throw withStage(error, "db:load_image_context");
            })
          : null;

      const reply = await geminiReply({
        prompt: imageContext
          ? buildPromptWithImageContext(prompt, imageContext)
          : prompt,
        history,
        config,
        inlineParts,
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
        await typingController.stop();
      }
    }
  }

  return async function handler(req, res) {
    const config = configLoader();

    const profileRepairTask = maybeRepairProfileState(config, logger);

    if (profileRepairTask) {
      waitUntil(profileRepairTask.catch(() => {}));
    }

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
