import { createHmac, timingSafeEqual } from "node:crypto";
import { waitUntil } from "@vercel/functions";

import { handleCommand, parseCommand } from "../lib/commands.js";
import {
  getRuntimeConfig,
  hasMessagingConfig,
  hasVerificationConfig,
} from "../lib/config.js";
import {
  buildPromptWithPersistentContext,
  buildRollingConversationSummary,
  extractMemoryCandidates,
} from "../lib/context.js";
import { generateGeminiReply, isRetryableGeminiError } from "../lib/gemini.js";
import { getConversationStore } from "../lib/history.js";
import {
  isRetryableMessengerError,
  sendSenderAction,
  sendTextMessage,
} from "../lib/messenger.js";
import { maybeRepairProfileState } from "../lib/profile-state.js";
import { retryAsync, summarizeError } from "../lib/reliability.js";
import {
  buildImageContextPrompt,
  buildImageReadyReply,
  buildLocationReadyReply,
  buildStoredInboundText,
  buildVisionPrompt,
  extractImageAttachments,
  extractLocationAttachment,
  extractLocationCoordinates,
  loadInlineImageParts,
} from "../lib/vision.js";

const ATTACHMENT_FALLBACK =
  "I can reply to text, supported image messages, and Messenger location pins right now. Send text, a PNG/JPEG/WEBP/HEIC/HEIF image, or a location pin.";
const UPSTREAM_FAILURE_REPLY =
  "I hit an upstream error while talking to Gemini. Please try again in a moment.";
const TYPING_REFRESH_INTERVAL_MS = 5000;

function isGeminiStage(stage) {
  return typeof stage === "string" && stage.startsWith("gemini");
}

function extractGeminiFailureReply(error) {
  const apiMessage =
    typeof error?.apiMessage === "string" ? error.apiMessage.trim() : "";

  if (apiMessage) {
    return apiMessage;
  }

  const message =
    typeof error?.message === "string" ? error.message.trim() : "";

  if (!message) {
    return "";
  }

  if (/^Gemini API \d+:/i.test(message)) {
    return message.replace(/^Gemini API \d+:\s*/i, "").trim();
  }

  if (/^Gemini\b/i.test(message)) {
    return message;
  }

  return "";
}

export function buildFailureReply(error) {
  if (!isGeminiStage(error?.stage)) {
    return UPSTREAM_FAILURE_REPLY;
  }

  return extractGeminiFailureReply(error) || UPSTREAM_FAILURE_REPLY;
}

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

function extractEventType(event, imageAttachments, sharedLocation) {
  if (event?.postback) {
    return "postback";
  }

  if (Array.isArray(imageAttachments) && imageAttachments.length > 0) {
    return "image_message";
  }

  if (
    Number.isFinite(sharedLocation?.latitude) &&
    Number.isFinite(sharedLocation?.longitude)
  ) {
    return "location_message";
  }

  return "message";
}

function logStructured(logger, level, eventName, payload) {
  logger[level]?.(eventName, payload);
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
    logStructured(logger, "warn", "messenger.sender_action.failed", {
      senderId,
      action,
      message: summarizeError(error),
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

async function sendTextWithRetries({
  senderId,
  text,
  config,
  sendTextMessageImpl,
  logger,
  retryStage,
}) {
  return retryAsync(
    async () => {
      await sendTextMessageImpl(senderId, text, config);
    },
    {
      retries: config.reliabilityRetryLimit,
      baseDelayMs: config.reliabilityRetryBaseMs,
      shouldRetry: isRetryableMessengerError,
      onRetry: (error, attempt) => {
        logStructured(logger, "warn", "messenger.send.retry", {
          senderId,
          stage: retryStage,
          attempt,
          message: summarizeError(error),
        });
      },
    },
  );
}

async function generateReplyWithRetries({
  prompt,
  history,
  config,
  inlineParts,
  geminiReply,
  logger,
  senderId,
  sourceEventId,
  stage,
}) {
  return retryAsync(
    async () =>
      geminiReply({
        prompt,
        history,
        config,
        inlineParts,
      }),
    {
      retries: config.reliabilityRetryLimit,
      baseDelayMs: config.reliabilityRetryBaseMs,
      shouldRetry: isRetryableGeminiError,
      onRetry: (error, attempt) => {
        logStructured(logger, "warn", "gemini.reply.retry", {
          senderId,
          sourceEventId,
          stage,
          attempt,
          message: summarizeError(error),
        });
      },
    },
  );
}

async function saveAutoMemory({ senderId, rawPrompt, store, logger }) {
  const candidates = extractMemoryCandidates(rawPrompt);

  if (candidates.length === 0) {
    return [];
  }

  const saved = [];

  for (const candidate of candidates) {
    try {
      saved.push(
        await store.saveMemory({
          senderId,
          content: candidate.content,
          kind: candidate.kind,
          source: "auto",
        }),
      );
    } catch (error) {
      logStructured(logger, "warn", "memory.auto_save.failed", {
        senderId,
        message: summarizeError(error),
      });
    }
  }

  return saved;
}

async function refreshConversationSummary({
  senderId,
  store,
  summaryMaxChars,
  logger,
}) {
  try {
    const previousSummary = await store.getConversationSummary(senderId);
    const history = await store.getSummarySourceHistory(senderId);
    const nextSummary = buildRollingConversationSummary(history, {
      previousSummary,
      maxChars: summaryMaxChars,
    });

    if (nextSummary) {
      await store.saveConversationSummary({
        senderId,
        summary: nextSummary,
      });
    }
  } catch (error) {
    logStructured(logger, "warn", "summary.refresh.failed", {
      senderId,
      message: summarizeError(error),
    });
  }
}

async function recordFailedOutboundSafely({
  store,
  senderId,
  sourceEventId,
  stage,
  replyText,
  error,
  logger,
}) {
  try {
    await store.recordFailedOutbound({
      senderId,
      sourceEventId,
      stage,
      replyText,
      errorMessage: summarizeError(error),
      retryCount: error?.retryCount || 0,
    });
  } catch (recordError) {
    logStructured(logger, "warn", "messenger.failed_outbound.record_failed", {
      senderId,
      sourceEventId,
      stage,
      message: summarizeError(recordError),
    });
  }
}

export function createWebhookHandler({
  configLoader = getRuntimeConfig,
  conversationStoreFactory = getConversationStore,
  geminiReply = generateGeminiReply,
  sendAction = sendSenderAction,
  sendTextMessageImpl = sendTextMessage,
  profileRepair = maybeRepairProfileState,
  waitUntilImpl = waitUntil,
  logger = console,
} = {}) {
  async function processEvent(event, config) {
    const senderId = event?.sender?.id;

    if (!senderId || event?.delivery || event?.read) {
      return;
    }

    const imageAttachments = extractImageAttachments(event);
    const locationAttachment = extractLocationAttachment(event);
    const sharedLocationFromAttachment =
      extractLocationCoordinates(locationAttachment);
    const rawPrompt = extractInboundPrompt(event);
    const prompt = buildVisionPrompt(rawPrompt, imageAttachments.length);
    const sourceEventId = extractSourceEventId(event);
    const command = parseCommand(rawPrompt);
    const eventType = extractEventType(
      event,
      imageAttachments,
      sharedLocationFromAttachment,
    );
    let store;
    let sharedLocation = null;
    let typingEnabled = false;
    let inboundSaved = false;
    let modelTurnSaved = false;
    let replyGenerated = false;
    let replySent = false;
    let replyText = "";
    let totalRetryCount = 0;
    let currentStage = "received";

    try {
      store = await conversationStoreFactory(config);
      const eventRecord = await store.beginEventProcessing({
        senderId,
        sourceEventId,
        eventType,
      });

      if (!eventRecord.inserted) {
        logStructured(logger, "info", "webhook.event.duplicate", {
          senderId,
          sourceEventId,
        });
        return;
      }

      logStructured(logger, "info", "webhook.event.received", {
        senderId,
        sourceEventId,
        eventType,
        hasCommand: Boolean(command),
        imageCount: imageAttachments.length,
      });

      if (sharedLocationFromAttachment) {
        currentStage = "db:save_location";
        sharedLocation = await store.saveLatestLocation({
          senderId,
          latitude: sharedLocationFromAttachment.latitude,
          longitude: sharedLocationFromAttachment.longitude,
          sourceEventId,
        });
      }

      await sendActionSafely({
        senderId,
        action: "mark_seen",
        config,
        sendAction,
        logger,
      });

      if (!prompt && !command) {
        if (sharedLocation) {
          replyText = buildLocationReadyReply(sharedLocation);
          replyGenerated = true;
          currentStage = "messenger:send_location_ready";
          const sendResult = await sendTextWithRetries({
            senderId,
            text: replyText,
            config,
            sendTextMessageImpl,
            logger,
            retryStage: currentStage,
          });
          totalRetryCount += sendResult.retryCount;
          replySent = true;
        } else if (event?.message?.attachments?.length) {
          replyText = ATTACHMENT_FALLBACK;
          replyGenerated = true;
          currentStage = "messenger:send_attachment_fallback";
          const sendResult = await sendTextWithRetries({
            senderId,
            text: replyText,
            config,
            sendTextMessageImpl,
            logger,
            retryStage: currentStage,
          });
          totalRetryCount += sendResult.retryCount;
          replySent = true;
        }

        await store.updateEventProcessing({
          senderId,
          sourceEventId,
          status: "completed",
          stage: replyGenerated ? currentStage : "ignored",
          replyGenerated,
          replySent,
          retryCount: totalRetryCount,
        });
        return;
      }

      const typingController = createTypingController({
        senderId,
        config,
        sendAction,
        logger,
      });

      typingEnabled = await typingController.start();

      try {
        if (command) {
          currentStage = "command";

          const commandResult = await handleCommand({
            command,
            senderId,
            store,
          });

          replyText =
            commandResult?.reply || "I could not process that command.";
          replyGenerated = true;

          const sendResult = await sendTextWithRetries({
            senderId,
            text: replyText,
            config,
            sendTextMessageImpl,
            logger,
            retryStage: "command_reply",
          });
          totalRetryCount += sendResult.retryCount;
          replySent = true;

          await store.updateEventProcessing({
            senderId,
            sourceEventId,
            status: "completed",
            stage: "command",
            replyGenerated: true,
            replySent: true,
            retryCount: totalRetryCount,
          });
          return;
        }

        currentStage = "db:save_inbound";
        const inboundResult = await store.saveInboundTurn({
          senderId,
          text: `${buildStoredInboundText(rawPrompt, imageAttachments.length)}${
            sharedLocationFromAttachment
              ? `\n\n[User shared location: ${sharedLocationFromAttachment.latitude}, ${sharedLocationFromAttachment.longitude}]`
              : ""
          }`.trim(),
          sourceEventId,
        });

        if (!inboundResult.inserted) {
          logStructured(logger, "info", "webhook.event.duplicate_inbound", {
            senderId,
            sourceEventId,
          });
          await store.updateEventProcessing({
            senderId,
            sourceEventId,
            status: "completed",
            stage: "duplicate_inbound",
            retryCount: totalRetryCount,
          });
          return;
        }

        inboundSaved = true;
        await store.updateEventProcessing({
          senderId,
          sourceEventId,
          stage: "inbound_saved",
          inboundSaved: true,
        });

        waitUntil(saveAutoMemory({ senderId, rawPrompt, store, logger }));

        currentStage = "attachments:load_inline_images";
        const inlineParts = await loadInlineImageParts(imageAttachments);

        if (
          imageAttachments.length > 0 &&
          inlineParts.length === 0 &&
          !rawPrompt
        ) {
          replyText = ATTACHMENT_FALLBACK;
          replyGenerated = true;

          const sendResult = await sendTextWithRetries({
            senderId,
            text: replyText,
            config,
            sendTextMessageImpl,
            logger,
            retryStage: "attachment_fallback",
          });
          totalRetryCount += sendResult.retryCount;
          replySent = true;

          currentStage = "db:save_model";
          await store.saveModelTurn({ senderId, text: replyText });
          modelTurnSaved = true;
          waitUntil(
            refreshConversationSummary({
              senderId,
              store,
              summaryMaxChars: config.summaryMaxChars,
              logger,
            }),
          );

          await store.updateEventProcessing({
            senderId,
            sourceEventId,
            status: "completed",
            stage: "completed",
            inboundSaved,
            replyGenerated,
            replySent,
            modelTurnSaved,
            retryCount: totalRetryCount,
          });
          return;
        }

        if (imageAttachments.length > 0 && !rawPrompt) {
          currentStage = "gemini:image_context";
          const imageSummaryResult = await generateReplyWithRetries({
            prompt: buildImageContextPrompt(inlineParts.length),
            history: [],
            config,
            inlineParts,
            geminiReply,
            logger,
            senderId,
            sourceEventId,
            stage: currentStage,
          });
          totalRetryCount += imageSummaryResult.retryCount;
          const imageSummary = imageSummaryResult.value;

          currentStage = "db:save_image_context";
          await store.saveLatestImageContext({
            senderId,
            summary: imageSummary,
            sourceEventId,
          });

          replyText = buildImageReadyReply(imageSummary, inlineParts.length);
          replyGenerated = true;

          currentStage = "messenger:send_text";
          const sendResult = await sendTextWithRetries({
            senderId,
            text: replyText,
            config,
            sendTextMessageImpl,
            logger,
            retryStage: currentStage,
          });
          totalRetryCount += sendResult.retryCount;
          replySent = true;

          currentStage = "db:save_model";
          await store.saveModelTurn({ senderId, text: replyText });
          modelTurnSaved = true;
          waitUntil(
            refreshConversationSummary({
              senderId,
              store,
              summaryMaxChars: config.summaryMaxChars,
              logger,
            }),
          );

          await store.updateEventProcessing({
            senderId,
            sourceEventId,
            status: "completed",
            stage: "completed",
            inboundSaved,
            replyGenerated,
            replySent,
            modelTurnSaved,
            retryCount: totalRetryCount,
          });
          return;
        }

        currentStage = "db:load_history";
        const history = await store.getConversationHistory(senderId, {
          excludeMessageId: inboundResult.messageId,
        });
        const conversationSummary =
          await store.getConversationSummary(senderId);
        const memoryEntries = await store.findRelevantMemory(senderId, prompt, {
          limit: config.memoryMaxItems,
        });
        if (!sharedLocation) {
          currentStage = "db:load_location";
          sharedLocation = await store.getLatestLocation(senderId);
        }
        const imageContext =
          imageAttachments.length === 0
            ? await store.getLatestImageContext(senderId)
            : null;
        const activeConfig =
          Number.isFinite(sharedLocation?.latitude) &&
          Number.isFinite(sharedLocation?.longitude)
            ? {
                ...config,
                geminiGoogleMapsLocation: {
                  latitude: sharedLocation.latitude,
                  longitude: sharedLocation.longitude,
                },
              }
            : config;
        const contextualPrompt = buildPromptWithPersistentContext({
          prompt,
          imageContext,
          conversationSummary,
          memoryEntries,
          sharedLocation,
          maxContextChars: config.promptContextMaxChars,
          maxMemoryItems: config.memoryMaxItems,
          maxMemoryChars: config.memoryMaxChars,
        });

        currentStage = "gemini";
        const replyResult = await generateReplyWithRetries({
          prompt: contextualPrompt,
          history,
          config: activeConfig,
          inlineParts,
          geminiReply,
          logger,
          senderId,
          sourceEventId,
          stage: currentStage,
        });
        totalRetryCount += replyResult.retryCount;
        replyText = replyResult.value;
        replyGenerated = true;

        currentStage = "messenger:send_text";
        const sendResult = await sendTextWithRetries({
          senderId,
          text: replyText,
          config,
          sendTextMessageImpl,
          logger,
          retryStage: currentStage,
        });
        totalRetryCount += sendResult.retryCount;
        replySent = true;

        currentStage = "db:save_model";
        await store.saveModelTurn({ senderId, text: replyText });
        modelTurnSaved = true;
        waitUntil(
          refreshConversationSummary({
            senderId,
            store,
            summaryMaxChars: config.summaryMaxChars,
            logger,
          }),
        );

        await store.updateEventProcessing({
          senderId,
          sourceEventId,
          status: "completed",
          stage: "completed",
          inboundSaved,
          replyGenerated,
          replySent,
          modelTurnSaved,
          retryCount: totalRetryCount,
        });
      } finally {
        if (typingEnabled) {
          await typingController.stop();
        }
      }
    } catch (error) {
      const failedStage = error?.stage || currentStage || "unknown";

      logStructured(logger, "error", "webhook.event.failed", {
        senderId,
        sourceEventId,
        stage: failedStage,
        replyGenerated,
        replySent,
        modelTurnSaved,
        message: summarizeError(error),
      });

      if (store) {
        await store
          .updateEventProcessing({
            senderId,
            sourceEventId,
            status: "failed",
            stage: failedStage,
            inboundSaved,
            replyGenerated,
            replySent,
            modelTurnSaved,
            retryCount: error?.retryCount ?? totalRetryCount,
            failureMessage: summarizeError(error),
          })
          .catch(() => {});
      }

      if (store && replyGenerated && !replySent && replyText) {
        await recordFailedOutboundSafely({
          store,
          senderId,
          sourceEventId,
          stage: failedStage,
          replyText,
          error,
          logger,
        });
      }

      if (!replyGenerated && !replySent) {
        const failureReply = buildFailureReply(error);

        try {
          const failureSendResult = await sendTextWithRetries({
            senderId,
            text: failureReply,
            config,
            sendTextMessageImpl,
            logger,
            retryStage: "messenger:send_failure_reply",
          });
          totalRetryCount += failureSendResult.retryCount;
        } catch (sendError) {
          if (store) {
            await recordFailedOutboundSafely({
              store,
              senderId,
              sourceEventId,
              stage: "messenger:send_failure_reply",
              replyText: failureReply,
              error: sendError,
              logger,
            });
          }
        }
      }
    }
  }

  return async function handler(req, res) {
    const config = configLoader();
    const profileRepairTask = profileRepair(config, logger);

    if (profileRepairTask) {
      waitUntilImpl(profileRepairTask.catch(() => {}));
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
