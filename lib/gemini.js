import { supportsThinkingLevel } from "./config.js";

let geminiApiKeyCursor = 0;

export function buildEndpoint(modelName, apiKey) {
  const url = new URL(
    `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(modelName)}:generateContent`,
  );
  url.searchParams.set("key", apiKey);
  return url;
}

function getConfiguredGeminiApiKeys(config) {
  if (Array.isArray(config.geminiApiKeys)) {
    return config.geminiApiKeys;
  }

  if (typeof config.geminiApiKey === "string" && config.geminiApiKey.trim()) {
    return [config.geminiApiKey.trim()];
  }

  return [];
}

export function getNextGeminiApiKey(config) {
  const apiKeys = getConfiguredGeminiApiKeys(config);

  if (apiKeys.length === 0) {
    throw new Error("No Gemini API key is configured.");
  }

  const selectedIndex = geminiApiKeyCursor % apiKeys.length;
  const selectedKey = apiKeys[selectedIndex];

  geminiApiKeyCursor = (selectedIndex + 1) % apiKeys.length;

  return selectedKey;
}

export function resetGeminiApiKeyRoundRobinForTesting() {
  geminiApiKeyCursor = 0;
}

function getConfiguredGeminiModels(config) {
  if (Array.isArray(config.geminiModels) && config.geminiModels.length > 0) {
    return config.geminiModels;
  }

  if (typeof config.geminiModel === "string" && config.geminiModel.trim()) {
    return [config.geminiModel.trim()];
  }

  return [];
}

export function getOrderedGeminiModels(config) {
  const models = getConfiguredGeminiModels(config);

  if (models.length === 0) {
    throw new Error("No Gemini model is configured.");
  }

  return models;
}

function buildThinkingConfig(config, modelName) {
  if (!config.geminiThinkingLevel) {
    return undefined;
  }

  if (!supportsThinkingLevel(modelName)) {
    return undefined;
  }

  return {
    thinkingLevel: config.geminiThinkingLevel,
  };
}

export function buildRequestBody({ prompt, history, config }) {
  const generationConfig = {};
  const activeModel = config.activeGeminiModel || config.geminiModel;
  const thinkingConfig = buildThinkingConfig(config, activeModel);

  if (thinkingConfig) {
    generationConfig.thinkingConfig = thinkingConfig;
  }

  const body = {
    contents: [...history, { role: "user", parts: [{ text: prompt }] }],
  };

  if (config.systemPrompt) {
    body.systemInstruction = {
      parts: [{ text: config.systemPrompt }],
    };
  }

  if (Object.keys(generationConfig).length > 0) {
    body.generationConfig = generationConfig;
  }

  return body;
}

function buildGeminiApiError(payload, status, modelName) {
  const detail =
    payload?.error?.message ||
    payload?.message ||
    payload?.promptFeedback?.blockReason ||
    "Unknown Gemini API error";

  const error = new Error(`Gemini API ${status}: ${detail}`);
  error.httpStatus = status;
  error.apiStatus = payload?.error?.status || "";
  error.apiMessage = detail;
  error.modelName = modelName;
  return error;
}

export function shouldFallbackToNextGeminiModel(error) {
  const retryableHttpStatuses = new Set([429, 500, 503]);
  const retryableApiStatuses = new Set([
    "RESOURCE_EXHAUSTED",
    "INTERNAL",
    "UNAVAILABLE",
  ]);
  const message = String(
    error?.apiMessage || error?.message || "",
  ).toLowerCase();

  return (
    retryableHttpStatuses.has(error?.httpStatus) ||
    retryableApiStatuses.has(error?.apiStatus) ||
    message.includes("overloaded") ||
    message.includes("temporarily running out of capacity") ||
    message.includes("resource has been exhausted")
  );
}

function extractTextFromCandidate(candidate) {
  const parts = candidate?.content?.parts;

  if (!Array.isArray(parts)) {
    return "";
  }

  return parts
    .map((part) => (typeof part?.text === "string" ? part.text.trim() : ""))
    .filter(Boolean)
    .join("\n\n")
    .trim();
}

export function normalizeReply(text) {
  return text
    .replace(/\r\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export async function generateGeminiReply({ prompt, history, config }) {
  const models = getOrderedGeminiModels(config);
  let lastError;

  for (let index = 0; index < models.length; index += 1) {
    const modelName = models[index];
    const apiKey = getNextGeminiApiKey(config);
    const response = await fetch(buildEndpoint(modelName, apiKey), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(
        buildRequestBody({
          prompt,
          history,
          config: {
            ...config,
            activeGeminiModel: modelName,
          },
        }),
      ),
    });

    const payload = await response.json().catch(() => null);

    if (!response.ok) {
      lastError = buildGeminiApiError(payload, response.status, modelName);

      if (
        index < models.length - 1 &&
        shouldFallbackToNextGeminiModel(lastError)
      ) {
        continue;
      }

      throw lastError;
    }

    const reply =
      payload?.candidates?.map(extractTextFromCandidate).find(Boolean) ||
      payload?.promptFeedback?.blockReason ||
      "";

    if (!reply) {
      throw new Error("Gemini returned no text response.");
    }

    return normalizeReply(reply);
  }

  throw lastError || new Error("Gemini request failed.");
}
