const DEFAULT_MODEL =
  process.env.GEMINI_CHAT_MODEL || "gemini-3-flash-preview";
const DEFAULT_THINKING_LEVEL =
  process.env.GEMINI_CHAT_THINKING_LEVEL || "low";

const createError = (status, message) => {
  const error = new Error(message);
  error.status = status;
  return error;
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const normalizeTextContent = (content) => {
  if (typeof content === "string") return content.trim();
  if (!Array.isArray(content)) return "";

  return content
    .map((part) => {
      if (typeof part === "string") return part;
      if (part?.type === "text" && typeof part.text === "string") {
        return part.text;
      }
      return "";
    })
    .filter(Boolean)
    .join("\n\n")
    .trim();
};

const normalizeMessages = (body = {}) => {
  const baseMessages = Array.isArray(body.messages) ? body.messages : [];
  const singleMessage = normalizeTextContent(body.message);
  const candidateMessages =
    baseMessages.length > 0
      ? baseMessages
      : singleMessage
        ? [{ role: "user", content: singleMessage }]
        : [];

  const normalizedMessages = candidateMessages
    .map((message) => ({
      role:
        message?.role === "assistant" || message?.role === "system"
          ? message.role
          : "user",
      content: normalizeTextContent(message?.content),
    }))
    .filter((message) => message.content);

  if (!normalizedMessages.length) {
    throw createError(400, "Provide a non-empty `message` or `messages` array.");
  }

  if (!normalizedMessages.some((message) => message.role === "user")) {
    throw createError(400, "At least one user message is required.");
  }

  return normalizedMessages;
};

const normalizeInteger = (value) => {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
};

const normalizeFloat = (value) => {
  const parsed = Number.parseFloat(String(value ?? ""));
  return Number.isFinite(parsed) ? parsed : null;
};

const normalizeThinkingLevel = (value) => {
  const normalized = String(value || "").trim().toLowerCase();
  if (
    normalized === "minimal" ||
    normalized === "low" ||
    normalized === "medium" ||
    normalized === "high"
  ) {
    return normalized;
  }
  return DEFAULT_THINKING_LEVEL;
};

const normalizeChatBody = (body = {}) => {
  if (body?.stream === true) {
    throw createError(
      400,
      "Streaming is not supported by this simple chat endpoint.",
    );
  }

  const messages = normalizeMessages(body);
  const temperature = normalizeFloat(body.temperature);
  const maxOutputTokens = normalizeInteger(
    body.maxOutputTokens ?? body.max_tokens,
  );

  return {
    model: DEFAULT_MODEL,
    messages,
    system:
      normalizeTextContent(body.system) ||
      messages
        .filter((message) => message.role === "system")
        .map((message) => message.content)
        .join("\n\n"),
    temperature:
      temperature === null ? null : clamp(temperature, 0, 2),
    maxOutputTokens,
    thinkingLevel: normalizeThinkingLevel(
      body.thinkingLevel ?? body.thinking_level,
    ),
  };
};

const buildGeminiPayload = (body) => {
  const contents = body.messages
    .filter((message) => message.role !== "system")
    .map((message) => ({
      role: message.role === "assistant" ? "model" : "user",
      parts: [{ text: message.content }],
    }));

  if (!contents.length) {
    throw createError(400, "At least one non-system message is required.");
  }

  const generationConfig = {
    thinkingConfig: {
      thinkingLevel: body.thinkingLevel,
    },
  };

  if (body.temperature !== null) {
    generationConfig.temperature = body.temperature;
  }

  if (body.maxOutputTokens) {
    generationConfig.maxOutputTokens = body.maxOutputTokens;
  }

  return {
    contents,
    ...(body.system
      ? {
          systemInstruction: {
            parts: [{ text: body.system }],
          },
        }
      : {}),
    generationConfig,
  };
};

const extractReplyContent = (data = {}) => {
  const candidate = Array.isArray(data?.candidates) ? data.candidates[0] : null;
  const parts = Array.isArray(candidate?.content?.parts)
    ? candidate.content.parts
    : [];
  const text = parts
    .map((part) => (typeof part?.text === "string" ? part.text : ""))
    .filter(Boolean)
    .join("\n\n")
    .trim();

  if (text) {
    return {
      content: text,
      finishReason: String(candidate?.finishReason || "STOP").toLowerCase(),
    };
  }

  if (data?.promptFeedback?.blockReason) {
    throw createError(
      400,
      `Gemini blocked the prompt (${data.promptFeedback.blockReason}).`,
    );
  }

  throw createError(502, "Gemini returned no text response.");
};

const normalizeUsage = (usageMetadata = {}) => ({
  prompt_tokens: Number(usageMetadata.promptTokenCount || 0),
  completion_tokens: Number(usageMetadata.candidatesTokenCount || 0),
  total_tokens: Number(usageMetadata.totalTokenCount || 0),
});

const parseErrorMessage = async (response) => {
  try {
    const payload = await response.json();
    return (
      payload?.error?.message ||
      payload?.message ||
      `Gemini request failed with HTTP ${response.status}.`
    );
  } catch {
    return `Gemini request failed with HTTP ${response.status}.`;
  }
};

const runGeminiChat = async (body, fetchImpl = globalThis.fetch) => {
  if (typeof fetchImpl !== "function") {
    throw createError(500, "Fetch is not available in this runtime.");
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw createError(
      500,
      "Missing GEMINI_API_KEY. Set it before using /api/chat.",
    );
  }

  const normalizedBody = normalizeChatBody(body);
  const response = await fetchImpl(
    `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(normalizedBody.model)}:generateContent`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": apiKey,
      },
      body: JSON.stringify(buildGeminiPayload(normalizedBody)),
    },
  );

  if (!response.ok) {
    throw createError(response.status, await parseErrorMessage(response));
  }

  const data = await response.json();
  const reply = extractReplyContent(data);

  return {
    model: normalizedBody.model,
    reply,
    usage: normalizeUsage(data?.usageMetadata),
  };
};

const metadataPayload = () => ({
  ok: true,
  endpoint: "/api/chat",
  mode: "simple-chatbot",
  provider: "google-gemini",
  model: DEFAULT_MODEL,
  streaming: false,
  agentic: false,
});

const createChatResponsePayload = (result) => ({
  ok: true,
  id: `chatcmpl-${Date.now()}`,
  object: "chat.completion",
  created: Math.floor(Date.now() / 1000),
  model: result.model,
  provider: "google-gemini",
  output_text: result.reply.content,
  choices: [
    {
      index: 0,
      message: {
        role: "assistant",
        content: result.reply.content,
      },
      finish_reason: result.reply.finishReason,
    },
  ],
  usage: result.usage,
});

module.exports = {
  DEFAULT_MODEL,
  metadataPayload,
  normalizeChatBody,
  runGeminiChat,
  createChatResponsePayload,
};
