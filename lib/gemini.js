import {
  buildDuckDuckGoGroundedPrompt,
  searchDuckDuckGo,
} from "./duckduckgo.js";
import {
  supportsGoogleSearch,
  supportsThinkingLevel,
  supportsUrlContext,
} from "./config.js";

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

function buildTools(config, modelName) {
  const tools = [];

  if (config.geminiEnableCodeExecution) {
    tools.push({ code_execution: {} });
  }

  if (config.geminiEnableGoogleSearch) {
    if (
      !modelName.toLowerCase().includes("preview") &&
      supportsGoogleSearch(modelName)
    ) {
      if (modelName.toLowerCase().startsWith("gemini-1.5-")) {
        tools.push({
          google_search_retrieval: {
            dynamic_retrieval_config: {
              mode: "MODE_DYNAMIC",
              dynamic_threshold: 0.7,
            },
          },
        });
      } else {
        tools.push({ google_search: {} });
      }
    }
  }

  if (config.geminiEnableUrlContext && supportsUrlContext(modelName)) {
    tools.push({ url_context: {} });
  }

  return tools.length > 0 ? tools : undefined;
}

function buildUserParts({ prompt, inlineParts = [] }) {
  return [...inlineParts, { text: prompt }];
}

function buildOptionalInstructionTurn(optionalInstruction) {
  const normalized =
    typeof optionalInstruction === "string" ? optionalInstruction.trim() : "";

  if (!normalized) {
    return null;
  }

  return {
    role: "user",
    parts: [
      {
        text: `Optional project guidance for this bot. Apply it only when it does not conflict with the system instruction or the user's latest message.\n\n${normalized}`,
      },
    ],
  };
}

export function buildRequestBody({
  prompt,
  history,
  config,
  inlineParts = [],
}) {
  const generationConfig = {};
  const activeModel = config.activeGeminiModel || config.geminiModel;
  const thinkingConfig = buildThinkingConfig(config, activeModel);
  const tools = buildTools(config, activeModel);
  const optionalInstructionTurn = buildOptionalInstructionTurn(
    config.optionalInstruction,
  );

  const contents = [...history];

  if (optionalInstructionTurn) {
    contents.push(optionalInstructionTurn);
  }

  contents.push({
    role: "user",
    parts: buildUserParts({ prompt, inlineParts }),
  });

  if (thinkingConfig) {
    generationConfig.thinkingConfig = thinkingConfig;
  }

  const body = {
    contents,
  };

  if (config.systemPrompt) {
    body.systemInstruction = {
      parts: [{ text: config.systemPrompt }],
    };
  }

  if (tools) {
    body.tools = tools;
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

  const textParts = parts
    .map((part) => (typeof part?.text === "string" ? part.text.trim() : ""))
    .filter(Boolean);

  if (textParts.length > 0) {
    return textParts.join("\n\n").trim();
  }

  const executionOutputs = parts
    .map((part) => {
      const value =
        part?.code_execution_result?.output ||
        part?.codeExecutionResult?.output ||
        "";
      return typeof value === "string" ? value.trim() : "";
    })
    .filter(Boolean);

  return executionOutputs.join("\n\n").trim();
}

export function normalizeReply(text) {
  return text
    .replace(/\r\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function shouldUseDuckDuckGoGrounding(config, modelName) {
  return (
    config.geminiEnableGoogleSearch &&
    typeof modelName === "string" &&
    modelName.toLowerCase().includes("preview")
  );
}

async function buildPromptWithGrounding(prompt, modelName, config) {
  if (!shouldUseDuckDuckGoGrounding(config, modelName)) {
    return prompt;
  }

  try {
    const results = await searchDuckDuckGo(prompt);
    return buildDuckDuckGoGroundedPrompt(prompt, results);
  } catch {
    return prompt;
  }
}

export async function generateGeminiReply({
  prompt,
  history,
  config,
  inlineParts = [],
}) {
  const configuredModels = getOrderedGeminiModels(config);
  const models = config.geminiEnableGoogleSearch
    ? configuredModels.filter(
        (modelName) =>
          supportsGoogleSearch(modelName) ||
          modelName.toLowerCase().includes("preview"),
      )
    : configuredModels;
  let lastError;

  if (models.length === 0) {
    throw new Error(
      config.geminiEnableGoogleSearch
        ? "Google Search grounding is enabled, but none of the configured Gemini models support it."
        : "No Gemini model is configured.",
    );
  }

  for (let index = 0; index < models.length; index += 1) {
    const modelName = models[index];
    const apiKey = getNextGeminiApiKey(config);
    const groundedPrompt = await buildPromptWithGrounding(
      prompt,
      modelName,
      config,
    );
    const response = await fetch(buildEndpoint(modelName, apiKey), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(
        buildRequestBody({
          prompt: groundedPrompt,
          history,
          inlineParts,
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
