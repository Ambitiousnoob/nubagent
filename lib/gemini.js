import {
  buildDuckDuckGoGroundedPrompt,
  searchDuckDuckGo,
} from "./duckduckgo.js";
import {
  getCachedContentForModel,
  supportsCodeExecution,
  supportsGoogleMapsGrounding,
  supportsGoogleSearch,
  supportsThinkingLevel,
  supportsUrlContext,
} from "./config.js";

let geminiApiKeyCursor = 0;

const EXACT_REASONING_HINT =
  "For arithmetic, algebra, combinatorics, graph optimization, or other exact numeric tasks, use code execution when available to verify the result instead of relying on mental math. Return the checked final answer clearly.";
const URL_CONTEXT_HINT =
  "When the user includes one or more URLs, use the URL context tool to read those exact URLs before answering. If a URL cannot be retrieved, say so briefly instead of guessing.";
const GOOGLE_MAPS_HINT =
  "For place-specific, local, or proximity-based questions, use Grounding with Google Maps when available. Base local recommendations and factual place details on Google Maps results and do not invent them.";
const TOOL_CAPABILITY_HINT_PREFIX =
  "Available tools in this request:";
const TOOL_CAPABILITY_BEHAVIOR_HINT =
  "Use the available tools automatically when helpful. If the user asks about your capabilities or access, answer from the tools listed for this request and do not deny access to an enabled tool.";
const MATH_KEYWORD_PATTERN =
  /\b(?:math|mathematics|calculate|calculation|compute|solve|equation|algebra|arithmetic|combinatorics|combinatorial|binomial|probability|statistics|matrix|determinant|integral|derivative|factorial|prime|modulo|mod|gcd|lcm|dynamic programming|shortest path|minimum cost|maximum cost|optimi[sz]e|dag)\b/i;
const MATH_EXPRESSION_PATTERN =
  /\b\d+(?:\.\d+)?\s*(?:\+|-|\*|\/|%|\^|=|<=|>=|<|>)\s*\d+(?:\.\d+)?\b/;
const COMBINATORICS_PATTERN = /\\binom|\bchoose\b|ncr/i;
const URL_PATTERN = /\bhttps?:\/\/[^\s<>"')\]]+/gi;
const MAPS_NEARBY_PATTERN =
  /\b(?:near me|nearby|near here|around here|close by|closest|walking distance|driving distance|within \d+\s*(?:min|mins|minute|minutes|km|kilometers|miles?|mi))\b/i;
const MAPS_PLACE_PATTERN =
  /\b(?:restaurant|restaurants|cafe|cafes|coffee shop|coffee shops|bar|bars|hotel|hotels|museum|museums|park|parks|pharmacy|pharmacies|hospital|hospitals|grocery store|grocery stores|supermarket|supermarkets|gas station|gas stations|shop|shops|mall|malls|airport|airports|station|stations|bus stop|bus stops|train station|train stations|itinerary|directions|route|routes|things to do|attractions|landmarks|neighborhood|neighbourhood|address|open now|hours)\b/i;
const MAPS_COORDINATE_PATTERN = /\b-?\d{1,3}\.\d+\s*,\s*-?\d{1,3}\.\d+\b/;
const GOOGLE_MAPS_MENTION_PATTERN = /\bgoogle maps?\b/i;
const TOOL_CAPABILITY_PATTERN =
  /\b(?:what tools|which tools|tool access|tool support|capabilities|what can you do|do you have access|can you access|can you use|google maps access|google maps|url context|code execution|google search)\b/i;

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

function extractUrls(prompt) {
  return Array.from(
    new Set(
      (typeof prompt === "string" ? prompt.match(URL_PATTERN) : [])
        ?.map((url) => url.trim())
        .filter(Boolean) || [],
    ),
  );
}

function looksLikeMapsPrompt(prompt) {
  const normalized = typeof prompt === "string" ? prompt.trim() : "";

  if (!normalized) {
    return false;
  }

  return (
    MAPS_NEARBY_PATTERN.test(normalized) ||
    MAPS_PLACE_PATTERN.test(normalized) ||
    MAPS_COORDINATE_PATTERN.test(normalized)
  );
}

export function getPromptCapabilities(prompt, config) {
  const normalizedPrompt =
    typeof prompt === "string" ? prompt.trim() : "";

  return {
    wantsExactReasoning: looksLikeExactReasoningPrompt(normalizedPrompt),
    wantsUrlContext:
      config.geminiEnableUrlContext && extractUrls(normalizedPrompt).length > 0,
    wantsMapsGrounding:
      config.geminiEnableGoogleMaps &&
      (looksLikeMapsPrompt(normalizedPrompt) ||
        GOOGLE_MAPS_MENTION_PATTERN.test(normalizedPrompt)),
    wantsToolCapabilityAnswer: TOOL_CAPABILITY_PATTERN.test(normalizedPrompt),
  };
}

function wantsGoogleMapsTool(promptCapabilities) {
  return Boolean(promptCapabilities?.wantsMapsGrounding);
}

function wantsCodeExecutionTool({
  config,
  modelName,
  promptCapabilities,
  includeGoogleMaps,
}) {
  return (
    config.geminiEnableCodeExecution &&
    supportsCodeExecution(modelName) &&
    !includeGoogleMaps
  );
}

function hasGoogleMapsTool(tools) {
  return Array.isArray(tools) && tools.some((tool) => Boolean(tool?.googleMaps));
}

function buildTools({ config, modelName, promptCapabilities }) {
  const tools = [];
  const includeGoogleMaps =
    config.geminiEnableGoogleMaps &&
    supportsGoogleMapsGrounding(modelName) &&
    wantsGoogleMapsTool(promptCapabilities);
  const includeCodeExecution = wantsCodeExecutionTool({
    config,
    modelName,
    promptCapabilities,
    includeGoogleMaps,
  });

  if (includeCodeExecution) {
    tools.push({ code_execution: {} });
  }

  if (includeGoogleMaps) {
    // Gemini rejects requests that include Google Maps grounding and code
    // execution together, so geo-intent prompts use Maps and omit code exec.
    tools.push({ googleMaps: {} });
  }

  if (config.geminiEnableGoogleSearch && supportsGoogleSearch(modelName)) {
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

  if (config.geminiEnableUrlContext && supportsUrlContext(modelName)) {
    tools.push({ url_context: {} });
  }

  return tools.length > 0 ? tools : undefined;
}

function buildToolConfig({ config, tools }) {
  if (
    !hasGoogleMapsTool(tools) ||
    !config.geminiGoogleMapsLocation
  ) {
    return undefined;
  }

  return {
    retrievalConfig: {
      latLng: config.geminiGoogleMapsLocation,
    },
  };
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
  const promptCapabilities = getPromptCapabilities(prompt, config);
  const thinkingConfig = buildThinkingConfig(config, activeModel);
  const tools = buildTools({
    config,
    modelName: activeModel,
    promptCapabilities,
  });
  const toolConfig = buildToolConfig({
    config,
    tools,
  });
  const optionalInstructionTurn = buildOptionalInstructionTurn(
    config.optionalInstruction,
  );
  const systemInstruction = buildSystemInstruction({
    config,
    modelName: activeModel,
    promptCapabilities,
    tools,
  });

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
  const cachedContent = getCachedContentForModel(config, activeModel);

  if (systemInstruction) {
    body.systemInstruction = {
      parts: [{ text: systemInstruction }],
    };
  }

  if (cachedContent) {
    body.cachedContent = cachedContent;
  }

  if (tools) {
    body.tools = tools;
  }

  if (toolConfig) {
    body.toolConfig = toolConfig;
  }

  if (Object.keys(generationConfig).length > 0) {
    body.generationConfig = generationConfig;
  }

  return body;
}

function looksLikeExactReasoningPrompt(prompt) {
  const normalized = typeof prompt === "string" ? prompt.trim() : "";

  if (!normalized) {
    return false;
  }

  return (
    MATH_KEYWORD_PATTERN.test(normalized) ||
    COMBINATORICS_PATTERN.test(normalized) ||
    MATH_EXPRESSION_PATTERN.test(normalized)
  );
}

function getAvailableToolLabels(tools) {
  if (!Array.isArray(tools)) {
    return [];
  }

  return tools.flatMap((tool) => {
    if (tool?.code_execution) {
      return ["Code Execution"];
    }

    if (tool?.googleMaps) {
      return ["Google Maps grounding"];
    }

    if (tool?.google_search || tool?.google_search_retrieval) {
      return ["Google Search grounding"];
    }

    if (tool?.url_context) {
      return ["URL Context"];
    }

    return [];
  });
}

function buildSystemInstruction({
  config,
  modelName,
  promptCapabilities,
  tools,
}) {
  const baseInstruction =
    typeof config.systemPrompt === "string" ? config.systemPrompt.trim() : "";

  if (!baseInstruction) {
    return "";
  }

  const hints = [];
  const availableToolLabels = getAvailableToolLabels(tools);

  if (availableToolLabels.length > 0) {
    hints.push(
      `${TOOL_CAPABILITY_HINT_PREFIX} ${availableToolLabels.join(", ")}.`,
    );
    hints.push(TOOL_CAPABILITY_BEHAVIOR_HINT);
  }

  if (
    config.geminiEnableCodeExecution &&
    promptCapabilities.wantsExactReasoning &&
    supportsCodeExecution(modelName)
  ) {
    hints.push(EXACT_REASONING_HINT);
  }

  if (
    config.geminiEnableUrlContext &&
    promptCapabilities.wantsUrlContext &&
    supportsUrlContext(modelName)
  ) {
    hints.push(URL_CONTEXT_HINT);
  }

  if (
    config.geminiEnableGoogleMaps &&
    promptCapabilities.wantsMapsGrounding &&
    supportsGoogleMapsGrounding(modelName)
  ) {
    hints.push(GOOGLE_MAPS_HINT);
  }

  if (promptCapabilities.wantsToolCapabilityAnswer) {
    hints.push(
      "When the user asks whether you have access to Google Maps, URL Context, Google Search, or Code Execution, answer directly from the available tools in this request.",
    );
  }

  return hints.length > 0
    ? `${baseInstruction}\n\n${hints.join("\n")}`
    : baseInstruction;
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

function extractUsageMetadata(payload) {
  const usage =
    payload?.usageMetadata ||
    payload?.usage_metadata ||
    payload?.responseMetadata?.usageMetadata ||
    payload?.response_metadata?.usage_metadata ||
    null;

  return usage && typeof usage === "object" ? usage : null;
}

function readUsageCount(usageMetadata, ...keys) {
  for (const key of keys) {
    const value = usageMetadata?.[key];

    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }
  }

  return 0;
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

function extractGroundingSources(candidate) {
  const chunks =
    candidate?.groundingMetadata?.groundingChunks ||
    candidate?.grounding_metadata?.grounding_chunks;

  if (!Array.isArray(chunks)) {
    return [];
  }

  const sources = [];
  const seen = new Set();

  for (const chunk of chunks) {
    const mapsTitle =
      typeof chunk?.maps?.title === "string" ? chunk.maps.title.trim() : "";
    const mapsUri =
      typeof chunk?.maps?.uri === "string" ? chunk.maps.uri.trim() : "";
    const webTitle =
      typeof chunk?.web?.title === "string" ? chunk.web.title.trim() : "";
    const webUri =
      typeof chunk?.web?.uri === "string" ? chunk.web.uri.trim() : "";

    const source = mapsUri
      ? {
          kind: "maps",
          title: mapsTitle || "Google Maps",
          uri: mapsUri,
        }
      : webUri
        ? {
            kind: "web",
            title: webTitle || webUri,
            uri: webUri,
          }
        : null;

    if (!source) {
      continue;
    }

    const sourceKey = `${source.kind}:${source.uri}`;

    if (seen.has(sourceKey)) {
      continue;
    }

    seen.add(sourceKey);
    sources.push(source);
  }

  return sources;
}

function buildGroundingFooter(candidate) {
  const sources = extractGroundingSources(candidate);

  if (sources.length === 0) {
    return "";
  }

  const mapsSources = sources.filter((source) => source.kind === "maps");
  const webSources = sources.filter((source) => source.kind === "web");
  const sections = [];

  if (mapsSources.length > 0) {
    sections.push(
      `Google Maps sources:\n${mapsSources
        .slice(0, 5)
        .map((source) => `- ${source.title}: ${source.uri}`)
        .join("\n")}`,
    );
  }

  if (webSources.length > 0) {
    sections.push(
      `Sources:\n${webSources
        .slice(0, 5)
        .map((source) => `- ${source.title}: ${source.uri}`)
        .join("\n")}`,
    );
  }

  return sections.join("\n\n");
}

function extractReplyFromCandidate(candidate) {
  const text = extractTextFromCandidate(candidate);

  if (!text) {
    return "";
  }

  const groundingFooter = buildGroundingFooter(candidate);
  return groundingFooter ? `${text}\n\n${groundingFooter}` : text;
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
    modelName.toLowerCase().includes("preview") &&
    !supportsGoogleSearch(modelName)
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

function countEnabledSupportedToolsForModel(modelName, config) {
  let count = 0;

  if (config.geminiEnableCodeExecution && supportsCodeExecution(modelName)) {
    count += 1;
  }

  if (config.geminiEnableGoogleMaps && supportsGoogleMapsGrounding(modelName)) {
    count += 1;
  }

  if (config.geminiEnableGoogleSearch && supportsGoogleSearch(modelName)) {
    count += 1;
  }

  if (config.geminiEnableUrlContext && supportsUrlContext(modelName)) {
    count += 1;
  }

  return count;
}

export function getOrderedGeminiModelsForPrompt(prompt, config) {
  const models = getOrderedGeminiModels(config);
  const promptCapabilities = getPromptCapabilities(prompt, config);

  return models
    .map((modelName, index) => {
      let score = 0;

      if (
        promptCapabilities.wantsMapsGrounding &&
        config.geminiEnableGoogleMaps &&
        supportsGoogleMapsGrounding(modelName)
      ) {
        score += 4;
      }

      if (
        promptCapabilities.wantsUrlContext &&
        config.geminiEnableUrlContext &&
        supportsUrlContext(modelName)
      ) {
        score += 2;
      }

      if (
        promptCapabilities.wantsExactReasoning &&
        config.geminiEnableCodeExecution &&
        supportsCodeExecution(modelName)
      ) {
        score += 1;
      }

      if (promptCapabilities.wantsToolCapabilityAnswer) {
        score += countEnabledSupportedToolsForModel(modelName, config) * 10;
      }

      return {
        modelName,
        index,
        score,
      };
    })
    .sort((left, right) => right.score - left.score || left.index - right.index)
    .map((entry) => entry.modelName);
}

export async function generateGeminiReply({
  prompt,
  history,
  config,
  inlineParts = [],
}) {
  const models = getOrderedGeminiModelsForPrompt(prompt, config);
  let lastError;

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

    const usageMetadata = extractUsageMetadata(payload);
    const cachedContentTokenCount = readUsageCount(
      usageMetadata,
      "cachedContentTokenCount",
      "cached_content_token_count",
    );

    if (cachedContentTokenCount > 0) {
      console.info("Gemini cached content used", {
        modelName,
        cachedContent: getCachedContentForModel(config, modelName) || null,
        cachedContentTokenCount,
        promptTokenCount: readUsageCount(
          usageMetadata,
          "promptTokenCount",
          "prompt_token_count",
        ),
        totalTokenCount: readUsageCount(
          usageMetadata,
          "totalTokenCount",
          "total_token_count",
        ),
      });
    }

    const reply =
      payload?.candidates?.map(extractReplyFromCandidate).find(Boolean) ||
      payload?.promptFeedback?.blockReason ||
      "";

    if (!reply) {
      throw new Error("Gemini returned no text response.");
    }

    return normalizeReply(reply);
  }

  throw lastError || new Error("Gemini request failed.");
}
