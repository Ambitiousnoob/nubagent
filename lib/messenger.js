const MAX_MESSAGE_LENGTH = 1800;

export function buildMessengerEndpoint(config) {
  const pagePath = config.pageId || "me";
  return `https://graph.facebook.com/${config.graphApiVersion}/${pagePath}/messages`;
}

function summarizeMessengerError(payload, status) {
  const message =
    payload?.error?.message ||
    payload?.error?.error_user_msg ||
    payload?.message ||
    "Unknown Messenger API error";

  return `Messenger API ${status}: ${message}`;
}

function normalizeText(text) {
  return text
    .replace(/\r\n/g, "\n")
    .replace(/\u0000/g, "")
    .trim();
}

function takeLeadingCharacters(text, maxLength) {
  return Array.from(text).slice(0, maxLength).join("");
}

export function splitTextIntoChunks(text) {
  const normalized = normalizeText(text);

  if (!normalized) {
    return ["I do not have a text response for that yet."];
  }

  const chunks = [];
  let remaining = normalized;

  while (remaining) {
    const candidate = takeLeadingCharacters(remaining, MAX_MESSAGE_LENGTH);

    if (candidate === remaining) {
      chunks.push(remaining);
      break;
    }

    let splitAt = candidate.lastIndexOf("\n\n");

    if (splitAt < candidate.length / 2) {
      splitAt = candidate.lastIndexOf("\n");
    }

    if (splitAt < candidate.length / 2) {
      splitAt = candidate.lastIndexOf(". ");
    }

    if (splitAt < candidate.length / 2) {
      splitAt = candidate.lastIndexOf(" ");
    }

    if (splitAt <= 0) {
      splitAt = candidate.length;
    }

    const consumedText = candidate.slice(0, splitAt);
    chunks.push(consumedText.trim());
    remaining = remaining.slice(consumedText.length).trim();
  }

  return chunks;
}

async function postMessenger(body, config) {
  const response = await fetch(buildMessengerEndpoint(config), {
    method: "POST",
    headers: {
      Authorization: `Bearer ${config.pageAccessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(summarizeMessengerError(payload, response.status));
  }

  return payload;
}

export async function sendSenderAction(recipientId, action, config) {
  return postMessenger(
    {
      recipient: { id: recipientId },
      sender_action: action,
    },
    config,
  );
}

export async function sendTextMessage(recipientId, text, config) {
  const chunks = splitTextIntoChunks(text);

  for (const chunk of chunks) {
    await postMessenger(
      {
        recipient: { id: recipientId },
        message_type: "RESPONSE",
        message: { text: chunk },
      },
      config,
    );
  }
}
