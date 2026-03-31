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

const TRUE_VALUES = new Set(["1", "true", "yes", "on"]);

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

function normalizeBooleanEnv(value) {
  return TRUE_VALUES.has(value.toLowerCase());
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
    cronSecret: readEnv("CRON_SECRET"),
    graphApiVersion: readEnv("GRAPH_API_VERSION") || "v23.0",
    geminiModel,
    geminiEnableGoogleSearch: normalizeBooleanEnv(
      readEnv("GEMINI_ENABLE_GOOGLE_SEARCH"),
    ),
    geminiEnableCodeExecution: normalizeBooleanEnv(
      readEnv("GEMINI_ENABLE_CODE_EXECUTION"),
    ),
    geminiEnableUrlContext: normalizeBooleanEnv(
      readEnv("GEMINI_ENABLE_URL_CONTEXT"),
    ),
    geminiThinkingLevel: normalizeThinkingLevel(
      readEnv("GEMINI_CHAT_THINKING_LEVEL") || "low",
    ),
    optionalInstruction: readEnv("OPTIONAL_INSTRUCTION"),
    systemPrompt:
      readEnv("SYSTEM_PROMPT") ||
      "You are NubAgent, a concise and helpful assistant replying inside Facebook Messenger. Keep answers plain text, short paragraphs, and easy to read on mobile. Prefer precise, practical wording over marketing language.",
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

export function supportsGoogleSearch(modelName) {
  if (typeof modelName !== "string") {
    return false;
  }

  const normalized = modelName.toLowerCase();

  if (normalized.includes("preview") || normalized.includes("experimental")) {
    return false;
  }

  return (
    normalized.startsWith("gemini-2.5-pro") ||
    normalized.startsWith("gemini-2.5-flash") ||
    normalized.startsWith("gemini-2.5-flash-lite") ||
    normalized.startsWith("gemini-2.0-flash") ||
    normalized.startsWith("gemini-1.5-pro") ||
    normalized.startsWith("gemini-1.5-flash")
  );
}

export function supportsUrlContext(modelName) {
  if (typeof modelName !== "string") {
    return false;
  }

  const normalized = modelName.toLowerCase();

  if (normalized.includes("preview") || normalized.includes("experimental")) {
    return false;
  }

  return (
    normalized.startsWith("gemini-3-flash") ||
    normalized.startsWith("gemini-3-pro") ||
    normalized.startsWith("gemini-2.5-pro") ||
    normalized.startsWith("gemini-2.5-flash") ||
    normalized.startsWith("gemini-2.5-flash-lite")
  );
}
