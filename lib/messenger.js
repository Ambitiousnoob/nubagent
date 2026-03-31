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

export function splitTextIntoChunks(text) {
  const normalized = normalizeText(text);

  if (!normalized) {
    return ["I do not have a text response for that yet."];
  }

  if (normalized.length <= MAX_MESSAGE_LENGTH) {
    return [normalized];
  }

  const chunks = [];
  let remaining = normalized;

  while (remaining.length > MAX_MESSAGE_LENGTH) {
    let splitAt = remaining.lastIndexOf("\n\n", MAX_MESSAGE_LENGTH);

    if (splitAt < MAX_MESSAGE_LENGTH / 2) {
      splitAt = remaining.lastIndexOf("\n", MAX_MESSAGE_LENGTH);
    }

    if (splitAt < MAX_MESSAGE_LENGTH / 2) {
      splitAt = remaining.lastIndexOf(". ", MAX_MESSAGE_LENGTH);
    }

    if (splitAt < MAX_MESSAGE_LENGTH / 2) {
      splitAt = remaining.lastIndexOf(" ", MAX_MESSAGE_LENGTH);
    }

    if (splitAt <= 0) {
      splitAt = MAX_MESSAGE_LENGTH;
    }

    chunks.push(remaining.slice(0, splitAt).trim());
    remaining = remaining.slice(splitAt).trim();
  }

  if (remaining) {
    chunks.push(remaining);
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
