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
const DEFAULT_SYSTEM_PROMPT =
  "You are NubAgent, a concise and helpful assistant replying inside Facebook Messenger. Keep answers plain text, short paragraphs, and easy to read on mobile. Prefer precise, practical wording over marketing language.";

function readEnv(name) {
  const value = process.env[name];
  return typeof value === "string" ? value.trim() : "";
}

function readNumericEnv(name) {
  const value = readEnv(name);

  if (!value) {
    return null;
  }

  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
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

export function parseGeminiCachedContent(value) {
  const entries = typeof value === "string" ? value.split(",") : [];
  const byModel = {};
  let fallback = "";

  for (const entry of entries.map((item) => item.trim()).filter(Boolean)) {
    const separatorIndex = entry.indexOf("=");

    if (separatorIndex < 0) {
      if (!fallback) {
        fallback = entry;
      }

      continue;
    }

    const modelName = normalizeModelName(entry.slice(0, separatorIndex));
    const cacheName = entry.slice(separatorIndex + 1).trim();

    if (modelName && cacheName) {
      byModel[modelName] = cacheName;
    }
  }

  return {
    fallback,
    byModel,
  };
}

function normalizeThinkingLevel(value) {
  const normalized = THINKING_LEVELS[value.toLowerCase()];
  return normalized === undefined ? "LOW" : normalized;
}

function normalizeBooleanEnv(value, defaultValue = false) {
  if (!value) {
    return defaultValue;
  }

  return TRUE_VALUES.has(value.toLowerCase());
}

function normalizeModelName(modelName) {
  return typeof modelName === "string"
    ? modelName
        .replace(/^models\//, "")
        .trim()
        .toLowerCase()
    : "";
}

function isSpecializedGeminiVariant(modelName) {
  return (
    modelName.includes("-live") ||
    modelName.includes("native-audio") ||
    modelName.includes("-tts") ||
    modelName.includes("image-generation") ||
    modelName.includes("-image-preview") ||
    modelName.includes("-image")
  );
}

function isGemini3TextModel(modelName) {
  return (
    (modelName.startsWith("gemini-3-pro") ||
      modelName.startsWith("gemini-3-flash")) &&
    !isSpecializedGeminiVariant(modelName)
  );
}

function isGemini25TextModel(modelName) {
  return (
    (modelName.startsWith("gemini-2.5-pro") ||
      modelName.startsWith("gemini-2.5-flash-lite") ||
      modelName.startsWith("gemini-2.5-flash")) &&
    !isSpecializedGeminiVariant(modelName)
  );
}

function isGemini20FlashTextModel(modelName) {
  return (
    modelName.startsWith("gemini-2.0-flash") &&
    !modelName.startsWith("gemini-2.0-flash-lite") &&
    !isSpecializedGeminiVariant(modelName)
  );
}

function isGemini15TextModel(modelName) {
  return (
    (modelName.startsWith("gemini-1.5-pro") ||
      modelName.startsWith("gemini-1.5-flash")) &&
    !isSpecializedGeminiVariant(modelName)
  );
}

export function getRuntimeConfig() {
  const geminiApiKeys = parseGeminiApiKeys(readEnv("GEMINI_API_KEY"));
  const geminiModels = parseGeminiModels(readEnv("GEMINI_CHAT_MODEL"));
  const geminiCachedContent = parseGeminiCachedContent(
    readEnv("GEMINI_CACHED_CONTENT"),
  );
  const geminiModel = geminiModels[0] || "";
  const geminiGoogleMapsLatitude = readNumericEnv(
    "GEMINI_GOOGLE_MAPS_LATITUDE",
  );
  const geminiGoogleMapsLongitude = readNumericEnv(
    "GEMINI_GOOGLE_MAPS_LONGITUDE",
  );

  return {
    geminiApiKeys,
    geminiModels,
    geminiCachedContent,
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
      true,
    ),
    geminiEnableCodeExecution: normalizeBooleanEnv(
      readEnv("GEMINI_ENABLE_CODE_EXECUTION"),
      true,
    ),
    geminiEnableGoogleMaps: normalizeBooleanEnv(
      readEnv("GEMINI_ENABLE_GOOGLE_MAPS"),
      true,
    ),
    geminiEnableUrlContext: normalizeBooleanEnv(
      readEnv("GEMINI_ENABLE_URL_CONTEXT"),
      true,
    ),
    geminiGoogleMapsLocation:
      geminiGoogleMapsLatitude === null || geminiGoogleMapsLongitude === null
        ? null
        : {
            latitude: geminiGoogleMapsLatitude,
            longitude: geminiGoogleMapsLongitude,
          },
    geminiThinkingLevel: normalizeThinkingLevel(
      readEnv("GEMINI_CHAT_THINKING_LEVEL") || "low",
    ),
    optionalInstruction: readEnv("OPTIONAL_INSTRUCTION"),
    systemPrompt: readEnv("SYSTEM_PROMPT") || DEFAULT_SYSTEM_PROMPT,
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
  return /^gemini-(3|[4-9])/.test(normalizeModelName(modelName));
}

export function supportsGoogleSearch(modelName) {
  const normalized = normalizeModelName(modelName);
  return (
    isGemini3TextModel(normalized) ||
    isGemini25TextModel(normalized) ||
    isGemini20FlashTextModel(normalized) ||
    isGemini15TextModel(normalized)
  );
}

export function supportsCodeExecution(modelName) {
  const normalized = normalizeModelName(modelName);
  return (
    isGemini15TextModel(normalized) ||
    isGemini3TextModel(normalized) ||
    isGemini25TextModel(normalized) ||
    isGemini20FlashTextModel(normalized)
  );
}

export function supportsGoogleMapsGrounding(modelName) {
  const normalized = normalizeModelName(modelName);

  return isGemini25TextModel(normalized) || isGemini20FlashTextModel(normalized);
}

export function supportsUrlContext(modelName) {
  const normalized = normalizeModelName(modelName);
  return isGemini3TextModel(normalized) || isGemini25TextModel(normalized);
}

export function getCachedContentForModel(config, modelName) {
  const normalized = normalizeModelName(modelName);
  const configuredModels = Array.isArray(config?.geminiModels)
    ? config.geminiModels
    : [];

  return (
    config?.geminiCachedContent?.byModel?.[normalized] ||
    (configuredModels.length <= 1 ? config?.geminiCachedContent?.fallback : "") ||
    ""
  );
}
