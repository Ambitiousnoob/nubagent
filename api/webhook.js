import { waitUntil } from "@vercel/functions";
import {
  normalizeChatBody,
  runGeminiChat,
  createChatResponsePayload,
} from "../lib/gemini-chat.js";

const GRAPH_API_BASE_URL = "https://graph.facebook.com/v21.0";
const PAGE_MESSAGES_URL = `${GRAPH_API_BASE_URL}/me/messages`;
const MESSENGER_MAX_TEXT_CHARS = 1900;
const MESSENGER_PROFILE_FIELDS = ["name", "first_name", "last_name"];
const MESSENGER_TYPING_MIN_MS = 400;
const MESSENGER_TYPING_MAX_MS = 1500;
const MESSENGER_TYPING_MS_PER_CHAR = 8;
const LEADING_INLINE_WHITESPACE_RE = /^[ \t]+/u;
const TRAILING_INLINE_WHITESPACE_RE = /[ \t]+$/u;
const WHITESPACE_ONLY_RE = /^\s+$/u;

function log(event, details = {}) {
  console.log(JSON.stringify({ event, ...details }));
}

function logError(event, details = {}) {
  console.error(JSON.stringify({ event, ...details }));
}

function countCodePoints(value) {
  return Array.from(value).length;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function estimateTypingDelayMs(text) {
  const estimatedDelayMs = countCodePoints(text) * MESSENGER_TYPING_MS_PER_CHAR;
  return Math.min(
    MESSENGER_TYPING_MAX_MS,
    Math.max(MESSENGER_TYPING_MIN_MS, estimatedDelayMs),
  );
}

function formatUtcDate(currentUtc) {
  return new Date(currentUtc).toLocaleDateString("en-US", {
    day: "numeric",
    month: "long",
    timeZone: "UTC",
    year: "numeric",
  });
}

function formatUtcTime(currentUtc) {
  return new Date(currentUtc).toLocaleTimeString("en-GB", {
    hour: "2-digit",
    hour12: false,
    minute: "2-digit",
    second: "2-digit",
    timeZone: "UTC",
  });
}

function classifyDirectReplyIntent(userText) {
  const normalizedText = String(userText ?? "").toLowerCase();

  const asksCurrentDate =
    /\b(today'?s date|current date|what(?:'s| is) the date|what day is it|date today)\b/u.test(
      normalizedText,
    ) ||
    (/\bdate\b/u.test(normalizedText) && /\btoday\b/u.test(normalizedText));

  const asksCurrentTime =
    /\b(current time|what(?:'s| is) the time|time now|time today)\b/u.test(
      normalizedText,
    ) ||
    (/\btime\b/u.test(normalizedText) &&
      /\b(now|today|current)\b/u.test(normalizedText));

  const asksOwnName =
    /\b(what(?:'s| is) my name|do you know my name|who am i|my name)\b/u.test(
      normalizedText,
    );

  return {
    asksCurrentDate,
    asksCurrentTime,
    asksOwnName,
  };
}

function buildDirectReply(userText, currentUtc, userProfile) {
  const { asksCurrentDate, asksCurrentTime, asksOwnName } =
    classifyDirectReplyIntent(userText);

  if (asksCurrentDate || asksCurrentTime) {
    const dateReply = `Today's date is ${formatUtcDate(currentUtc)} UTC.`;
    const timeReply = `The current time is ${formatUtcTime(currentUtc)} UTC.`;

    if (asksCurrentDate && asksCurrentTime) {
      return {
        reason: "current_datetime",
        text: `${dateReply} ${timeReply}`,
      };
    }

    if (asksCurrentDate) {
      return {
        reason: "current_date",
        text: dateReply,
      };
    }

    return {
      reason: "current_time",
      text: timeReply,
    };
  }

  if (asksOwnName) {
    if (userProfile?.fullName) {
      return {
        reason: "user_name_available",
        text: `Your name is ${userProfile.fullName}.`,
      };
    }

    return {
      reason: "user_name_unavailable",
      text: "Meta did not provide your Messenger profile name for this conversation.",
    };
  }

  return null;
}

function normalizeMessengerUserProfile(profileData) {
  const firstName =
    typeof profileData?.first_name === "string" && profileData.first_name.trim()
      ? profileData.first_name.trim()
      : null;
  const lastName =
    typeof profileData?.last_name === "string" && profileData.last_name.trim()
      ? profileData.last_name.trim()
      : null;
  const fullName =
    typeof profileData?.name === "string" && profileData.name.trim()
      ? profileData.name.trim()
      : [firstName, lastName].filter(Boolean).join(" ") || null;

  if (!fullName && !firstName && !lastName) {
    return null;
  }

  return {
    firstName,
    fullName,
    lastName,
  };
}

function buildAiMessages(userText, currentUtc, userProfile) {
  const systemInstructions = [
    `Current UTC date and time: ${currentUtc}. If the user asks for today's date or current time without specifying a location or timezone, answer using UTC and say it is UTC.`,
    "Do not claim that you cannot access the current date or time. It is already provided in this system message.",
    "Use available tools when external or real-time information is needed.",
  ];

  if (userProfile?.fullName) {
    systemInstructions.push(
      `The user's Messenger profile name is ${JSON.stringify(userProfile.fullName)}.${userProfile.firstName ? ` Their first name is ${JSON.stringify(userProfile.firstName)}.` : ""} Use their name naturally only when it is relevant. Do not claim to know anything else about them beyond this profile name.`,
    );
  } else {
    systemInstructions.push(
      "The user's Messenger profile name is unavailable. Do not guess it.",
    );
  }

  return [
    {
      role: "system",
      content: systemInstructions.join(" "),
    },
    {
      role: "user",
      content: userText,
    },
  ];
}

function extractMessengerImageUrls(message) {
  const attachments = Array.isArray(message?.attachments)
    ? message.attachments
    : [];

  return attachments
    .filter(
      (attachment) =>
        attachment?.type === "image" &&
        typeof attachment?.payload?.url === "string",
    )
    .map((attachment) => String(attachment.payload.url).trim())
    .filter(Boolean);
}

async function fetchMessengerUserProfile(sender_psid, pageAccessToken) {
  const profileUrl = `${GRAPH_API_BASE_URL}/${sender_psid}?fields=${MESSENGER_PROFILE_FIELDS.join(",")}&access_token=${pageAccessToken}`;

  try {
    const res = await fetch(profileUrl);
    const data = await res.json();

    log("messenger.profile.fetched", {
      sender_psid,
      status: res.status,
      ok: res.ok,
      has_name: Boolean(data?.name),
    });

    return res.ok ? normalizeMessengerUserProfile(data) : null;
  } catch (error) {
    logError("messenger.profile.fetch_error", {
      sender_psid,
      error: error instanceof Error ? error.message : String(error),
    });
    return null;
  }
}

async function sendMessengerTyping(sender_psid, pageAccessToken) {
  await sendMessengerSenderAction(sender_psid, "typing_on", pageAccessToken);
}

async function sendMessengerTypingStop(sender_psid, pageAccessToken) {
  await sendMessengerSenderAction(sender_psid, "typing_off", pageAccessToken);
}

function splitLongToken(token, maxChars) {
  const codePoints = Array.from(token);
  const parts = [];

  for (let start = 0; start < codePoints.length; start += maxChars) {
    parts.push(codePoints.slice(start, start + maxChars).join(""));
  }

  return parts;
}

function chunkMessengerText(text, maxChars = MESSENGER_MAX_TEXT_CHARS) {
  const normalizedText = String(text ?? "")
    .replace(/\r\n/g, "\n")
    .trim();

  if (!normalizedText) {
    return [];
  }

  const tokens = normalizedText.match(/\S+|\s+/gu) ?? [];
  const chunks = [];
  let currentChunk = "";
  let currentLength = 0;

  function pushCurrentChunk() {
    const trimmedChunk = currentChunk.replace(
      TRAILING_INLINE_WHITESPACE_RE,
      "",
    );

    if (trimmedChunk.trim()) {
      chunks.push(trimmedChunk);
    }

    currentChunk = "";
    currentLength = 0;
  }

  for (const token of tokens) {
    if (!currentChunk && WHITESPACE_ONLY_RE.test(token)) {
      continue;
    }

    const tokenLength = countCodePoints(token);

    if (tokenLength > maxChars) {
      pushCurrentChunk();

      const trimmedToken = token.trim();
      if (trimmedToken) {
        chunks.push(...splitLongToken(trimmedToken, maxChars));
      }
      continue;
    }

    if (currentLength + tokenLength <= maxChars) {
      currentChunk += token;
      currentLength += tokenLength;
      continue;
    }

    pushCurrentChunk();
    currentChunk = token.replace(LEADING_INLINE_WHITESPACE_RE, "");
    currentLength = countCodePoints(currentChunk);
  }

  pushCurrentChunk();
  return chunks;
}

async function postMessengerPayload(payload, pageAccessToken) {
  const messengerRes = await fetch(
    `${PAGE_MESSAGES_URL}?access_token=${pageAccessToken}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );

  const messengerBody = await messengerRes.text();

  return {
    messengerBody,
    messengerRes,
  };
}

async function sendMessengerSenderAction(
  senderPsid,
  senderAction,
  pageAccessToken,
) {
  const { messengerBody, messengerRes } = await postMessengerPayload(
    {
      recipient: { id: senderPsid },
      sender_action: senderAction,
    },
    pageAccessToken,
  );

  log("messenger.sender_action.sent", {
    sender_psid: senderPsid,
    sender_action: senderAction,
    status: messengerRes.status,
    ok: messengerRes.ok,
  });
}

async function sendMessengerReply(
  sender_psid,
  text,
  pageAccessToken,
  options = {},
) {
  const chunks = chunkMessengerText(text);

  if (!chunks.length) {
    log("messenger.reply.empty", { sender_psid });
    return;
  }

  const initialTypingStartedAtMs =
    options?.initialTypingStartedAtMs ?? Date.now();

  for (let index = 0; index < chunks.length; index += 1) {
    const chunk = chunks[index];
    const isLastChunk = index === chunks.length - 1;
    const typingDuration = estimateTypingDelayMs(chunk);
    const elapsedSinceTypingStartMs = Date.now() - initialTypingStartedAtMs;

    if (elapsedSinceTypingStartMs < typingDuration) {
      const remainingTypingMs = typingDuration - elapsedSinceTypingStartMs;
      log("messenger.reply.delay", {
        sender_psid,
        chunk_index: index,
        delay_ms: remainingTypingMs,
      });
      await sleep(remainingTypingMs);
    }

    const { messengerBody, messengerRes } = await postMessengerPayload(
      {
        recipient: { id: sender_psid },
        message: { text: chunk },
      },
      pageAccessToken,
    );

    log("messenger.reply.sent", {
      sender_psid,
      chunk_index: index,
      chunk_length: chunk.length,
      total_chunks: chunks.length,
      status: messengerRes.status,
      ok: messengerRes.ok,
    });

    if (!messengerRes.ok) {
      logError("messenger.reply.failed", {
        sender_psid,
        chunk_index: index,
        status: messengerRes.status,
        body: messengerBody.slice(0, 200),
      });
    }

    if (!isLastChunk) {
      await sleep(200);
    }
  }

  await sendMessengerTypingStop(sender_psid, pageAccessToken);
}

async function handleMessagingEntry(entry, pageAccessToken) {
  const messagingEvents = Array.isArray(entry?.messaging)
    ? entry.messaging
    : [];

  for (const event of messagingEvents) {
    const sender_psid = event?.sender?.id;
    const messageData = event?.message;

    if (!sender_psid || !messageData) {
      continue;
    }

    const userText = String(messageData?.text ?? "").trim();
    if (!userText && !extractMessengerImageUrls(messageData).length) {
      log("messenger.message.empty", { sender_psid });
      continue;
    }

    const currentUtc = new Date().toISOString();

    log("messenger.message.received", {
      sender_psid,
      has_text: Boolean(userText),
      text_length: userText.length,
      has_image: extractMessengerImageUrls(messageData).length > 0,
    });

    const userProfile = await fetchMessengerUserProfile(
      sender_psid,
      pageAccessToken,
    );

    await sendMessengerTyping(sender_psid, pageAccessToken);
    const initialTypingStartedAtMs = Date.now();

    const directReply = buildDirectReply(userText, currentUtc, userProfile);

    if (directReply) {
      log("messenger.direct_reply.sent", {
        sender_psid,
        reason: directReply.reason,
        text_length: directReply.text.length,
      });
      await sendMessengerReply(sender_psid, directReply.text, pageAccessToken, {
        initialTypingStartedAtMs,
      });
      return;
    }

    const messages = buildAiMessages(userText, currentUtc, userProfile);
    const systemPrompt =
      messages.find((message) => message?.role === "system")?.content || "";

    log("messenger.profile.attached", {
      has_profile_name: Boolean(userProfile?.fullName),
      sender_psid,
    });

    log("ai.request.sent", {
      sender_psid,
      has_user_name: Boolean(userProfile?.fullName),
      messages_count: 2,
      user_message_length: userText.length,
      current_utc: currentUtc,
    });

    try {
      const chatBody = {
        system: systemPrompt,
        messages: [messages.find((m) => m.role === "user")],
      };

      const aiResult = await runGeminiChat(chatBody);
      const botReply = aiResult.reply.content;

      log("ai.reply.normalized", {
        sender_psid,
        reply_length: countCodePoints(botReply),
      });

      await sendMessengerReply(sender_psid, botReply, pageAccessToken, {
        initialTypingStartedAtMs,
      });
    } catch (error) {
      const fallbackReply =
        "The AI service is under heavy load right now. Please try again in a moment.";
      logError("ai.request.failed", {
        sender_psid,
        error: error instanceof Error ? error.message : String(error),
      });
      await sendMessengerReply(sender_psid, fallbackReply, pageAccessToken, {
        initialTypingStartedAtMs,
      });
    }
  }
}

export default async function handler(req, res) {
  const PAGE_ACCESS_TOKEN = process.env.PAGE_ACCESS_TOKEN;
  const VERIFY_TOKEN = process.env.VERIFY_TOKEN;

  log("webhook.request.received", {
    method: req.method,
    has_body: Boolean(req.body),
  });

  if (req.method === "GET") {
    if (!VERIFY_TOKEN) {
      logError("webhook.verify.misconfigured", { missing_env: "VERIFY_TOKEN" });
      return res.status(500).send("Missing VERIFY_TOKEN");
    }

    const mode = req.query["hub.mode"];
    const token = req.query["hub.verify_token"];
    const challenge = req.query["hub.challenge"];

    log("webhook.verify.attempt", {
      mode,
      has_token: Boolean(token),
      challenge_length: challenge?.length ?? 0,
    });

    if (mode === "subscribe" && token === VERIFY_TOKEN) {
      log("webhook.verify.success");
      res.setHeader("Content-Type", "text/plain");
      return res.status(200).end(challenge);
    }

    log("webhook.verify.failed", { mode });
    return res.status(403).send("Verification failed");
  }

  if (req.method === "POST") {
    const body = req.body;
    const entries = Array.isArray(body?.entry) ? body.entry : [];

    log("webhook.post.received", {
      object: body?.object ?? null,
      entry_count: entries.length,
      has_page_access_token: Boolean(PAGE_ACCESS_TOKEN),
    });

    if (body?.object === "page") {
      if (!PAGE_ACCESS_TOKEN) {
        logError("webhook.post.misconfigured", {
          missing_env: "PAGE_ACCESS_TOKEN",
        });
        return res.status(500).send("Missing PAGE_ACCESS_TOKEN");
      }

      const backgroundWork = Promise.allSettled(
        entries.map(async (entry, index) => {
          try {
            log("webhook.entry.started", { index });
            await handleMessagingEntry(entry, PAGE_ACCESS_TOKEN);
            log("webhook.entry.completed", { index });
          } catch (error) {
            logError("webhook.entry.failed", {
              index,
              error: error instanceof Error ? error.message : String(error),
            });
          }
        }),
      ).then((results) => {
        log("webhook.background.complete", {
          settled: results.length,
        });
      });

      waitUntil(backgroundWork);
      return res.status(200).send("EVENT_RECEIVED");
    }

    log("webhook.post.ignored", { object: body?.object ?? null });
    return res.status(404).end();
  }

  log("webhook.method.unsupported", { method: req.method });
  return res.status(405).end();
}
