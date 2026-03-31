const VERIFY_KEYS = ["VERIFY_TOKEN"];
const BASE_MESSAGING_KEYS = [
  "PAGE_ACCESS_TOKEN",
  "VERIFY_TOKEN",
  "POSTGRES_URL",
];

const THINKING_LEVELS = {
  none: null,
  off: null,
  minimal: "MINIMAL",
  low: "LOW",
  medium: "MEDIUM",
  high: "HIGH",
};

function readEnv(name) {
  const value = process.env[name];
  return typeof value === "string" ? value.trim() : "";
}

function missingKeys(keys) {
  return keys.filter((key) => !readEnv(key));
}

export function parseGeminiApiKeys(value) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

export function parseGeminiModels(value) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => item.replace(/^models\//, ""));
}

function normalizeThinkingLevel(value) {
  const normalized = THINKING_LEVELS[value.toLowerCase()];
  return normalized === undefined ? "LOW" : normalized;
}

export function getRuntimeConfig() {
  const geminiApiKeys = parseGeminiApiKeys(readEnv("GEMINI_API_KEY"));
  const geminiModels = parseGeminiModels(readEnv("GEMINI_CHAT_MODEL"));
  const geminiModel = geminiModels[0] || "";

  return {
    geminiApiKeys,
    geminiModels,
    pageAccessToken: readEnv("PAGE_ACCESS_TOKEN"),
    pageId: readEnv("PAGE_ID"),
    postgresUrl: readEnv("POSTGRES_URL"),
    verifyToken: readEnv("VERIFY_TOKEN"),
    facebookAppSecret: readEnv("FACEBOOK_APP_SECRET"),
    graphApiVersion: readEnv("GRAPH_API_VERSION") || "v23.0",
    geminiModel,
    geminiThinkingLevel: normalizeThinkingLevel(
      readEnv("GEMINI_CHAT_THINKING_LEVEL") || "low",
    ),
    systemPrompt:
      readEnv("SYSTEM_PROMPT") ||
      "You are NubAgent, a concise and helpful assistant replying inside Facebook Messenger. Keep answers plain text, short paragraphs, and easy to read on mobile.",
    missingVerificationKeys: missingKeys(VERIFY_KEYS),
    missingMessagingKeys: [
      ...(geminiApiKeys.length === 0 ? ["GEMINI_API_KEY"] : []),
      ...(geminiModels.length === 0 ? ["GEMINI_CHAT_MODEL"] : []),
      ...missingKeys(BASE_MESSAGING_KEYS),
    ],
  };
}

export function hasVerificationConfig(config) {
  return config.missingVerificationKeys.length === 0;
}

export function hasMessagingConfig(config) {
  return config.missingMessagingKeys.length === 0;
}

export function supportsThinkingLevel(modelName) {
  return /^gemini-(3|[4-9])/.test(modelName);
}
