const path = require("node:path");
const dotenv = require("dotenv");
dotenv.config();
dotenv.config({ path: path.join(__dirname, "..", ".env") });

const {
  getApiKeysFromEnv,
  getRotatingApiKey,
  getRotatingValue,
} = require("./api-key-rotation.cjs");
const {
  clampCompletionTokens,
  getBackoffDelayMs,
  sleep,
} = require("./ai-control");
const { assertPublicHttpUrl } = require("./web");
const { GoogleGenAI } = require("@google/genai");

const {
  definition: calcDef,
  handler: calcHandler,
} = require("./tools/calculate");
const {
  definition: searchDef,
  handler: searchHandler,
} = require("./tools/web_search");
const {
  definition: fetchDef,
  handler: fetchHandler,
} = require("./tools/web_fetch");
const {
  definition: imageSearchDef,
  handler: imageSearchHandler,
} = require("./tools/search_images");
const {
  definition: viewImageDef,
  handler: viewImageHandler,
} = require("./tools/view_image");
const {
  definition: delegateResearchDef,
  handler: delegateResearchHandler,
} = require("./tools/delegate_research");

const GEMINI_PROVIDER = "gemini";
const OPENROUTER_PROVIDER = "openrouter";
const DEFAULT_MODEL = "gemini-2.5-flash-lite";
const DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free";
const OPENROUTER_API_URL =
  process.env.OPENROUTER_API_URL ||
  "https://openrouter.ai/api/v1/chat/completions";
const PUBLIC_MODEL_NAME = "nub-agent";
const PUBLIC_DEVELOPER_NAME = "Ambitiousnoob";
const BRAND_SYSTEM_PROMPT = [
  `You are ${PUBLIC_MODEL_NAME}.`,
  `When asked about your model, identity, creator, builder, or developer, identify as "${PUBLIC_MODEL_NAME}" and say "${PUBLIC_DEVELOPER_NAME}" built you.`,
  "Do not reveal internal provider names or backend model IDs unless the user explicitly asks for implementation details.",
  "When a user asks for source-grounded, contested, or multi-step research, use the delegate_research tool instead of improvising from memory.",
].join(" ");
/** Used when `research_mode` / `research_synthesis` is true — avoids identity lines in synthesized reports. */
const RESEARCH_SYSTEM_PROMPT = [
  `You are ${PUBLIC_MODEL_NAME}'s research synthesis backend.`,
  "Do not introduce yourself, your model name, or your creator in the answer body.",
  "Do not write chatty meta; start with substantive content (title or first sentence of the report).",
  "Cite sources using bracketed indices like [1] or [12] that match the provided evidence index only.",
  'Never append bare comma-separated numbers after sentences (e.g. "4, 9, 11") — those are invalid.',
].join(" ");
const FALLBACK_MODELS = [
  DEFAULT_MODEL,
  "gemini-2.5-flash",
  "gemini-3-flash-preview",
];
const MAX_TOOL_TURNS = 6;
const MAX_INLINE_IMAGE_BYTES = 10 * 1024 * 1024;
const DEFAULT_IMAGE_ANALYSIS_PROMPT =
  "Analyze the provided image and answer based on what is visible.";
const SAFE_MAX_COMPLETION_TOKENS = clampCompletionTokens(
  process.env.GEMINI_MAX_COMPLETION_TOKENS || 4096,
);
const SIMPLE_IMAGE_FIELDS = ["image", "image_url", "images", "image_urls"];
const OPENROUTER_REFERER =
  process.env.OPENROUTER_HTTP_REFERER ||
  process.env.NUBAGENT_PUBLIC_URL ||
  (process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : "");
const OPENROUTER_TITLE = process.env.OPENROUTER_APP_TITLE || PUBLIC_MODEL_NAME;
const OPENROUTER_MODEL_ALIASES = Object.freeze({
  nemotron: DEFAULT_OPENROUTER_MODEL,
  "nemotron-free": DEFAULT_OPENROUTER_MODEL,
  "research-nemotron": DEFAULT_OPENROUTER_MODEL,
  "openrouter-nemotron": DEFAULT_OPENROUTER_MODEL,
  "openrouter-nemotron-free": DEFAULT_OPENROUTER_MODEL,
  "nvidia/nemotron-3-super-120b-a12b": DEFAULT_OPENROUTER_MODEL,
  "nvidia/nemotron-3-super-120b-a12b:free": DEFAULT_OPENROUTER_MODEL,
});

const TOOL_DEFINITIONS = [
  calcDef,
  searchDef,
  fetchDef,
  imageSearchDef,
  viewImageDef,
  delegateResearchDef,
];

const wantsStream = (body = {}) => {
  const responseType = String(
    body.responseType || body.format || "",
  ).toLowerCase();
  if (responseType === "stream" || responseType === "sse") return true;
  if (responseType === "json" || responseType === "raw") return false;
  return body.stream === true;
};

const getMessageText = (content) => {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((part) => part?.type === "text" && typeof part.text === "string")
    .map((part) => part.text)
    .join("\n\n");
};

const normalizeText = (value) => String(value ?? "").trim();

const unique = (values = []) => [
  ...new Set((Array.isArray(values) ? values : []).filter(Boolean)),
];
const normalizeApiKeyList = (value = []) =>
  unique(
    (Array.isArray(value) ? value : [value])
      .flatMap((item) => String(item || "").split(","))
      .map((item) => normalizeText(item))
      .filter(Boolean),
  );

const toBase64 = (arrayBuffer) => Buffer.from(arrayBuffer).toString("base64");

const normalizeHttpImageUrl = (value) => {
  const raw = normalizeText(value);
  if (!raw) return "";

  try {
    const parsed = new URL(raw);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return "";
    return parsed.toString();
  } catch {
    return "";
  }
};

const isImageDataUrl = (value) =>
  /^data:image\/[^;,]+;base64,[a-z0-9+/=\s]+$/i.test(normalizeText(value));

const normalizeImageLocator = (value) => {
  const raw = normalizeText(value);
  if (!raw) return "";
  if (isImageDataUrl(raw)) return raw;
  return normalizeHttpImageUrl(raw);
};

const normalizeInlineDataPart = (inlineData) => {
  const mimeType = normalizeText(inlineData?.mimeType || inlineData?.mime_type);
  const data = normalizeText(inlineData?.data);
  if (!mimeType || !data || !mimeType.toLowerCase().startsWith("image/"))
    return null;
  return {
    type: "image_inline",
    inlineData: { mimeType, data },
  };
};

const normalizeImageInputValue = (value) => {
  if (!value) return null;
  if (typeof value === "string") {
    const url = normalizeImageLocator(value);
    return url ? { type: "image_url", image_url: { url } } : null;
  }

  if (typeof value !== "object") return null;

  if (value.type === "image_inline" && value.inlineData) {
    return normalizeInlineDataPart(value.inlineData);
  }

  if (value.inlineData || value.inline_data) {
    return normalizeInlineDataPart(value.inlineData || value.inline_data);
  }

  const directUrl = normalizeText(
    value.url ||
      value.image ||
      value.imageUrl ||
      value.image_url?.url ||
      value.image_url,
  );
  const normalizedUrl = normalizeImageLocator(directUrl);
  if (normalizedUrl) {
    return {
      type: "image_url",
      image_url: { url: normalizedUrl },
    };
  }

  return null;
};

const normalizeMessageContentForModel = (content) => {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return content;

  return content
    .map((part) => {
      if (!part || typeof part !== "object") return null;
      if (
        (part.type === "text" || part.type === "input_text") &&
        typeof part.text === "string"
      ) {
        return { type: "text", text: part.text };
      }
      if (part.type === "image_url" || part.type === "input_image") {
        return normalizeImageInputValue(part);
      }
      if (part.type === "image_inline" && part.inlineData)
        return normalizeInlineDataPart(part.inlineData);
      if (part.inlineData || part.inline_data)
        return normalizeInlineDataPart(part.inlineData || part.inline_data);
      return null;
    })
    .filter(Boolean);
};

const normalizeMessagesForModel = (messages = []) =>
  (Array.isArray(messages) ? messages : []).map((message) =>
    message && typeof message === "object"
      ? {
          ...message,
          content: normalizeMessageContentForModel(message.content),
        }
      : message,
  );

const hasBrandSystemMessage = (messages = []) =>
  (Array.isArray(messages) ? messages : []).some(
    (message) =>
      message?.role === "system" &&
      (getMessageText(message.content).includes(PUBLIC_MODEL_NAME) ||
        getMessageText(message.content).includes(PUBLIC_DEVELOPER_NAME)),
  );

const injectBrandSystemMessage = (messages = [], options = {}) => {
  const normalized = Array.isArray(messages) ? [...messages] : [];
  const researchMode =
    options.research_mode === true || options.research_synthesis === true;
  if (researchMode) {
    return [{ role: "system", content: RESEARCH_SYSTEM_PROMPT }, ...normalized];
  }
  if (hasBrandSystemMessage(normalized)) return normalized;
  return [{ role: "system", content: BRAND_SYSTEM_PROMPT }, ...normalized];
};

const collectTopLevelImageParts = (body = {}) => {
  const result = [];
  for (const field of SIMPLE_IMAGE_FIELDS) {
    const raw = body?.[field];
    if (Array.isArray(raw)) {
      for (const item of raw) {
        const normalized = normalizeImageInputValue(item);
        if (normalized) result.push(normalized);
      }
    } else {
      const normalized = normalizeImageInputValue(raw);
      if (normalized) result.push(normalized);
    }
  }
  return result;
};

const resolveTopLevelTextPrompt = (body = {}) => {
  const candidates = [body.prompt, body.input, body.message];
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim())
      return candidate.trim();
  }
  return "";
};

const buildMessagesFromSimpleBody = (body = {}) => {
  const prompt = resolveTopLevelTextPrompt(body);
  const imageParts = collectTopLevelImageParts(body);
  const systemPrompt = normalizeText(
    body.system || body.systemPrompt || body.system_prompt,
  );
  const messages = [];

  if (systemPrompt) {
    messages.push({ role: "system", content: systemPrompt });
  }

  if (!prompt && !imageParts.length) return messages;

  if (!imageParts.length) {
    messages.push({ role: "user", content: prompt });
    return messages;
  }

  const content = [];
  if (prompt) content.push({ type: "text", text: prompt });
  content.push(...imageParts);
  messages.push({ role: "user", content });
  return messages;
};

const findLastUserIndex = (messages = []) => {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "user") return index;
  }
  return -1;
};

const mergeMessageContent = (baseContent, extraContent) => {
  if (!extraContent) return baseContent;
  if (!baseContent) return extraContent;

  const baseIsString = typeof baseContent === "string";
  const extraIsString = typeof extraContent === "string";

  if (baseIsString && extraIsString) {
    const merged = [baseContent.trim(), extraContent.trim()]
      .filter(Boolean)
      .join("\n\n");
    return merged || baseContent;
  }

  const baseParts = baseIsString
    ? [{ type: "text", text: baseContent }]
    : Array.isArray(baseContent)
      ? [...baseContent]
      : [];
  const extraParts = extraIsString
    ? [{ type: "text", text: extraContent }]
    : Array.isArray(extraContent)
      ? extraContent
      : [];

  return [...baseParts, ...extraParts];
};

const resolveIncomingMessages = (body = {}) => {
  const explicitMessages = Array.isArray(body.messages) ? body.messages : [];
  const simpleMessages = buildMessagesFromSimpleBody(body);

  if (!explicitMessages.length) return simpleMessages;
  if (!simpleMessages.length) return explicitMessages;

  const explicitSystem = explicitMessages.filter(
    (message) => message?.role === "system",
  );
  const explicitNonSystem = explicitMessages.filter(
    (message) => message?.role !== "system",
  );
  const simpleSystem = simpleMessages.filter(
    (message) => message?.role === "system",
  );
  const simpleNonSystem = simpleMessages.filter(
    (message) => message?.role !== "system",
  );

  if (!explicitNonSystem.length) {
    return [
      ...(explicitSystem.length ? explicitSystem : simpleSystem),
      ...simpleNonSystem,
    ];
  }

  if (!simpleNonSystem.length) return explicitMessages;

  const merged = explicitMessages.map((message) =>
    message && typeof message === "object" ? { ...message } : message,
  );
  const lastUserIndex = findLastUserIndex(merged);
  if (lastUserIndex >= 0) {
    const latestSimpleUser = simpleNonSystem[simpleNonSystem.length - 1];
    merged[lastUserIndex] = {
      ...merged[lastUserIndex],
      content: mergeMessageContent(
        merged[lastUserIndex]?.content,
        latestSimpleUser?.content,
      ),
    };
    return merged;
  }

  return [...explicitMessages, ...simpleNonSystem];
};

const getLatestUserText = (messages = []) => {
  const latestUserMessage = [...(Array.isArray(messages) ? messages : [])]
    .reverse()
    .find((message) => message?.role === "user");
  return getMessageText(latestUserMessage?.content || latestUserMessage) || "";
};

const extractAttachmentManifestEntries = (messages = []) => {
  const source = getLatestUserText(messages);
  if (!source) return [];

  return source
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.startsWith("- Image:") || line.startsWith("- Text:"))
    .map((line) => {
      const kind = line.startsWith("- Image:") ? "image" : "text";
      const prefix = kind === "image" ? "- Image:" : "- Text:";
      const afterPrefix = line.slice(prefix.length).trim();
      const parenIndex = afterPrefix.lastIndexOf(" (");
      const name = (
        parenIndex >= 0 ? afterPrefix.slice(0, parenIndex) : afterPrefix
      ).trim();
      return name ? { kind, name } : null;
    })
    .filter(Boolean);
};

const extractAttachmentOcrEntries = (messages = []) => {
  const source = getLatestUserText(messages);
  if (!source) return [];

  const entries = [];
  const pattern =
    /--- OCR:\s*(.+?) ---\n([\s\S]*?)(?=\n{2}--- (?:OCR|File):|\n{2}Relevant retrieved context:|$)/g;
  let match;
  while ((match = pattern.exec(source)) !== null) {
    const name = String(match[1] || "").trim();
    const text = String(match[2] || "").trim();
    if (!name || !text) continue;
    entries.push({ name, text });
  }
  return entries;
};

const resolveAttachmentViewImage = (args, messages = []) => {
  const url = String(args?.url || "").trim();
  if (!/^https?:\/\//i.test(url)) return null;

  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    return null;
  }

  const ocrEntries = extractAttachmentOcrEntries(messages);
  const manifestEntries = extractAttachmentManifestEntries(messages).filter(
    (entry) => entry.kind === "image",
  );
  if (!ocrEntries.length && !manifestEntries.length) return null;

  const fileName = decodeURIComponent(
    parsed.pathname.split("/").pop() || "",
  ).toLowerCase();
  const matched =
    ocrEntries.find((entry) => entry.name.toLowerCase() === fileName) ||
    ocrEntries.find(
      (entry) => fileName && entry.name.toLowerCase().includes(fileName),
    ) ||
    manifestEntries.find((entry) => entry.name.toLowerCase() === fileName) ||
    manifestEntries.find(
      (entry) => fileName && entry.name.toLowerCase().includes(fileName),
    ) ||
    (parsed.hostname.toLowerCase() === "example.com" && ocrEntries.length === 1
      ? ocrEntries[0]
      : null) ||
    (parsed.hostname.toLowerCase() === "example.com" &&
    manifestEntries.length === 1
      ? manifestEntries[0]
      : null);

  if (!matched && parsed.hostname.toLowerCase() !== "example.com") return null;

  if (!matched) {
    const payload = {
      url,
      source: "attachment_context",
      note: "This URL appears to be a placeholder for an uploaded attachment that is already present in the conversation. Do not fetch it remotely. Use the attachment OCR/context already provided in the prompt.",
    };
    return {
      content: JSON.stringify(payload),
      meta: {
        rewritten: true,
        source: payload.source,
        note: payload.note,
      },
    };
  }

  const hasOcr =
    typeof matched.text === "string" && matched.text.trim().length > 0;
  const ocrText = hasOcr
    ? matched.text.length > 5000
      ? `${matched.text.slice(0, 5000)}...`
      : matched.text
    : "";
  const payload = hasOcr
    ? {
        url,
        source: "attachment_ocr",
        attachmentName: matched.name,
        summary: ocrText.slice(0, 1200),
        ocrText,
        note: "This URL maps to an uploaded attachment already present in the conversation. Use this OCR/context instead of attempting a remote fetch.",
      }
    : {
        url,
        source: "attachment_context",
        attachmentName: matched.name,
        note: "This URL maps to an uploaded attachment already present in the conversation, but no OCR text was extracted from it. Do not remote-fetch the placeholder URL; reason from the existing attachment context instead.",
      };
  return {
    content: JSON.stringify(payload),
    meta: {
      rewritten: true,
      source: payload.source,
      attachmentName: payload.attachmentName,
      note: payload.note,
    },
  };
};

const executeTool = async (toolCall, context = {}) => {
  const name = toolCall?.function?.name || toolCall?.name;
  const args = toolCall?.function?.arguments || toolCall?.args || {};

  let processedArgs = args;
  if (typeof args === "string") {
    try {
      processedArgs = JSON.parse(args);
    } catch (error) {
      return {
        content: `Error: could not parse arguments (${error.message})`,
        meta: null,
      };
    }
  }

  if (name === "calculate")
    return { content: await calcHandler(processedArgs), meta: null };
  if (name === "web_search")
    return { content: await searchHandler(processedArgs), meta: null };
  if (name === "web_fetch")
    return { content: await fetchHandler(processedArgs), meta: null };
  if (name === "search_images")
    return { content: await imageSearchHandler(processedArgs), meta: null };
  if (name === "view_image") {
    const attachmentProxy = resolveAttachmentViewImage(
      processedArgs,
      context.messages,
    );
    if (attachmentProxy) return attachmentProxy;
    return { content: await viewImageHandler(processedArgs), meta: null };
  }
  if (name === "delegate_research") {
    const content = await delegateResearchHandler(processedArgs, context);
    if (/^error:/i.test(normalizeText(content))) {
      throw createHttpError(
        502,
        normalizeText(content) || "Delegated research failed.",
      );
    }
    let delegatedResearch = null;

    try {
      const parsed = JSON.parse(content);
      if (parsed && typeof parsed === "object" && parsed.delegated === true) {
        const outputMode =
          parsed.outputMode && typeof parsed.outputMode === "object"
            ? {
                id: normalizeText(parsed.outputMode.id),
                label: normalizeText(parsed.outputMode.label),
              }
            : null;
        const sources = (Array.isArray(parsed.sources) ? parsed.sources : [])
          .map((source) => ({
            citationIndex: Number(source?.citationIndex || 0) || undefined,
            title: normalizeText(source?.title),
            url: normalizeText(source?.url),
            tier: normalizeText(source?.tier || "supporting") || "supporting",
          }))
          .filter((source) => source.title && source.url);

        delegatedResearch = {
          delegated: true,
          runId: normalizeText(parsed.runId),
          heading:
            normalizeText(parsed.heading || "Research Answer") ||
            "Research Answer",
          outputMode: outputMode?.id ? outputMode : null,
          noGroundedSources: parsed.noGroundedSources === true,
          sourceSelection: parsed.sourceSelection || null,
          sourceCount:
            Number(parsed.sourceCount || sources.length || 0) || sources.length,
          sources,
          tribunal: parsed.tribunal || null,
          convergence: parsed.convergence || null,
          verifierSummary: parsed.verifierSummary || null,
        };
      }
    } catch {
      delegatedResearch = null;
    }

    return {
      content,
      meta: delegatedResearch
        ? {
            runId: delegatedResearch.runId,
            sourceCount: delegatedResearch.sourceCount,
            noGroundedSources: delegatedResearch.noGroundedSources,
            outputMode: delegatedResearch.outputMode?.id || "",
          }
        : null,
      delegatedResearch,
    };
  }
  return { content: `Error: unknown tool ${name}`, meta: null };
};

const normalizeModel = (model) => {
  const id = typeof model === "string" ? model.trim() : "";
  const lowered = id.toLowerCase();
  const aliasedOpenRouterModel = OPENROUTER_MODEL_ALIASES[lowered];

  if (aliasedOpenRouterModel) return aliasedOpenRouterModel;
  if (lowered === "claude-3") return "anthropic/claude-3.7-sonnet";
  if (lowered === "gpt-4") return "openai/gpt-4.1-mini";
  if (lowered.startsWith("openrouter:"))
    return id.slice(id.indexOf(":") + 1).trim() || DEFAULT_OPENROUTER_MODEL;
  if (!id || lowered === "polly" || lowered.startsWith("polly-"))
    return DEFAULT_MODEL;
  if (
    lowered === PUBLIC_MODEL_NAME ||
    lowered === "nub agent" ||
    lowered === "gemini-3-flash"
  )
    return DEFAULT_MODEL;
  if (lowered === "gemini-pro") return "gemini-2.5-flash";
  if (lowered === "gemini-3-flash-preview") return "gemini-3-flash-preview";
  if (lowered === "gemini-2.5-flash") return "gemini-2.5-flash";
  if (lowered === "gemini-2.5-flash-lite") return "gemini-2.5-flash-lite";
  if (lowered.startsWith("openrouter/")) return id.slice("openrouter/".length);
  if (id.includes("/") || lowered.includes(":free")) return id;
  return DEFAULT_MODEL;
};

const isGeminiModel = (modelId = "") =>
  normalizeText(modelId).toLowerCase().startsWith("gemini-");

const normalizeProviderApiKeys = (value = {}) => {
  const candidate = value && typeof value === "object" ? value : {};
  const normalized = {
    openrouter: normalizeText(candidate?.openrouter || candidate?.openRouter),
    google: normalizeText(candidate?.google || candidate?.gemini),
  };
  return Object.fromEntries(
    Object.entries(normalized).filter(([, apiKey]) => apiKey),
  );
};

const normalizeProvider = (provider, model) => {
  const requestedProvider = normalizeText(provider).toLowerCase();
  if (requestedProvider === OPENROUTER_PROVIDER) return OPENROUTER_PROVIDER;
  if (requestedProvider === GEMINI_PROVIDER || requestedProvider === "google")
    return GEMINI_PROVIDER;
  return isGeminiModel(model) ? GEMINI_PROVIDER : OPENROUTER_PROVIDER;
};

const resolveMaxTokens = (value) =>
  Math.min(
    clampCompletionTokens(value, SAFE_MAX_COMPLETION_TOKENS),
    SAFE_MAX_COMPLETION_TOKENS,
  );

const getGeminiApiKeys = (body = {}) =>
  unique([
    ...normalizeApiKeyList(body?.providerApiKeys?.google),
    ...getApiKeysFromEnv("GEMINI_API_KEYS", "GEMINI_API_KEY"),
  ]);
const getGeminiApiKey = (body = {}) => {
  const requestKeys = normalizeApiKeyList(body?.providerApiKeys?.google);
  if (requestKeys.length)
    return getRotatingValue("gemini-request", requestKeys);
  return getRotatingApiKey("gemini", "GEMINI_API_KEYS", "GEMINI_API_KEY");
};
const getOpenRouterApiKeys = (body = {}) =>
  unique([
    ...normalizeApiKeyList(body?.providerApiKeys?.openrouter),
    ...getApiKeysFromEnv("OPENROUTER_API_KEYS", "OPENROUTER_API_KEY"),
  ]);
const getOpenRouterApiKey = (body = {}) => {
  const requestKeys = normalizeApiKeyList(body?.providerApiKeys?.openrouter);
  if (requestKeys.length)
    return getRotatingValue("openrouter-request", requestKeys);
  return getRotatingApiKey(
    "openrouter",
    "OPENROUTER_API_KEYS",
    "OPENROUTER_API_KEY",
  );
};

const getGeminiFallbackModelChain = (requestedModel, configuredChain = []) => {
  const configured = (
    process.env.GEMINI_FALLBACK_MODELS || FALLBACK_MODELS.join(",")
  )
    .split(",")
    .concat(Array.isArray(configuredChain) ? configuredChain : [])
    .map((item) => normalizeModel(item))
    .filter((item) => isGeminiModel(item));
  return [
    ...new Set(
      [normalizeModel(requestedModel), ...configured].filter((item) =>
        isGeminiModel(item),
      ),
    ),
  ];
};

const getOpenRouterFallbackModelChain = (
  requestedModel,
  configuredChain = [],
) => {
  const configured = [
    ...normalizeApiKeyList(configuredChain),
    ...normalizeApiKeyList(
      process.env.OPENROUTER_FALLBACK_MODELS ||
        process.env.OPENROUTER_MODEL_CHAIN,
    ),
  ]
    .map((item) => normalizeModel(item))
    .filter((item) => item && !isGeminiModel(item));
  return [
    ...new Set([normalizeModel(requestedModel), ...configured].filter(Boolean)),
  ];
};

const getThinkingConfigForModel = (modelId, body = {}) => {
  const explicitBudget = body.thinking_budget ?? body.thinkingBudget;
  if (Number.isFinite(Number(explicitBudget))) {
    return { thinkingBudget: Number(explicitBudget) };
  }

  const normalized = normalizeModel(modelId);
  if (
    normalized === "gemini-2.5-flash" ||
    normalized === "gemini-2.5-flash-lite"
  ) {
    return { thinkingBudget: 0 };
  }
  if (normalized === "gemini-3-flash-preview") {
    return { thinkingLevel: "minimal" };
  }
  return undefined;
};

const parseInlineDataUrl = (value) => {
  const match = String(value || "").match(
    /^data:(image\/[^;,]+);base64,(.+)$/i,
  );
  if (!match) return null;
  return {
    mimeType: match[1],
    data: match[2],
  };
};

const fetchRemoteImageInlineData = async (url) => {
  const normalizedUrl = await assertPublicHttpUrl(url);
  const response = await fetch(normalizedUrl, {
    headers: {
      Accept: "image/*",
      "User-Agent": `${PUBLIC_MODEL_NAME}/1.0`,
    },
  });

  if (!response.ok) {
    const accessHint =
      response.status === 401 || response.status === 403
        ? "The image URL is not publicly accessible."
        : `Remote image fetch failed with status ${response.status}.`;
    throw createHttpError(
      400,
      `${accessHint} Provide a public image URL, a data URL, or upload the image directly.`,
    );
  }

  const mimeType = normalizeText(response.headers.get("content-type"))
    .split(";")[0]
    .trim()
    .toLowerCase();
  if (!mimeType.startsWith("image/")) {
    throw createHttpError(
      400,
      `Remote resource is not an image (${mimeType || "unknown content-type"}).`,
    );
  }

  const contentLength = Number(response.headers.get("content-length") || 0);
  if (contentLength && contentLength > MAX_INLINE_IMAGE_BYTES) {
    throw createHttpError(
      400,
      `Remote image exceeds ${MAX_INLINE_IMAGE_BYTES} bytes.`,
    );
  }

  const bytes = await response.arrayBuffer();
  if (!bytes.byteLength) {
    throw createHttpError(400, "Remote image response was empty.");
  }
  if (bytes.byteLength > MAX_INLINE_IMAGE_BYTES) {
    throw createHttpError(
      400,
      `Remote image exceeds ${MAX_INLINE_IMAGE_BYTES} bytes.`,
    );
  }

  return {
    mimeType,
    data: toBase64(bytes),
  };
};

const mapOpenAiContentToGeminiParts = async (content, options = {}) => {
  if (typeof content === "string") {
    return content ? [{ text: content }] : [];
  }

  if (!Array.isArray(content)) return [];

  const parts = [];
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    if (
      part.type === "text" &&
      typeof part.text === "string" &&
      part.text.trim()
    ) {
      parts.push({ text: part.text });
      continue;
    }
    if (part.type === "image_url" && typeof part.image_url?.url === "string") {
      const inlineData = parseInlineDataUrl(part.image_url.url);
      if (inlineData) {
        parts.push({ inlineData });
        continue;
      }
      parts.push({
        inlineData: await fetchRemoteImageInlineData(part.image_url.url),
      });
      continue;
    }
    if (
      part.type === "image_inline" &&
      part.inlineData?.mimeType &&
      part.inlineData?.data
    ) {
      parts.push({ inlineData: part.inlineData });
    }
  }

  const hasInlineImage = parts.some(
    (part) => part?.inlineData?.mimeType && part?.inlineData?.data,
  );
  const hasText = parts.some(
    (part) => typeof part?.text === "string" && part.text.trim(),
  );
  if (options.ensureImagePrompt && hasInlineImage && !hasText) {
    parts.unshift({ text: DEFAULT_IMAGE_ANALYSIS_PROMPT });
  }

  return parts;
};

const getSystemInstruction = (messages = []) => {
  const text = (Array.isArray(messages) ? messages : [])
    .filter((message) => message?.role === "system")
    .map((message) => getMessageText(message.content))
    .filter(Boolean)
    .join("\n\n");
  return text || undefined;
};

const mapOpenAiToGeminiMessages = async (messages = []) => {
  const result = [];
  for (const message of (Array.isArray(messages) ? messages : []).filter(
    (item) => item?.role !== "system",
  )) {
    const role = message.role === "assistant" ? "model" : "user";
    const parts = await mapOpenAiContentToGeminiParts(message.content, {
      ensureImagePrompt: role === "user",
    });
    if (!parts.length) parts.push({ text: "" });
    result.push({ role, parts });
  }
  return result;
};

const mapOpenAiToolsToGemini = (tools = []) => {
  const scrub = (obj) => {
    if (!obj || typeof obj !== "object") return obj;
    if (Array.isArray(obj)) return obj.map(scrub);
    const result = {};
    for (const [key, value] of Object.entries(obj)) {
      if (key === "additionalProperties") continue;
      result[key] = scrub(value);
    }
    return result;
  };

  return [
    {
      functionDeclarations: tools.map((tool) => ({
        name: tool.function.name,
        description: tool.function.description,
        parameters: scrub(tool.function.parameters),
      })),
    },
  ];
};

const getGeminiResponseText = (response) => {
  if (!response) return "";
  if (typeof response.text === "function") return response.text() || "";
  if (typeof response.text === "string") return response.text;
  return (
    response?.candidates?.[0]?.content?.parts
      ?.map((part) => (typeof part?.text === "string" ? part.text : ""))
      .filter(Boolean)
      .join("") || ""
  );
};

const getGeminiResponseContent = (response) =>
  response?.candidates?.[0]?.content || {
    role: "model",
    parts: [{ text: getGeminiResponseText(response) }],
  };

const getGeminiFunctionCalls = (response) => {
  if (!response) return [];
  if (Array.isArray(response.functionCalls)) return response.functionCalls;
  if (typeof response.functionCalls === "function") {
    const calls = response.functionCalls();
    return Array.isArray(calls) ? calls : [];
  }
  return [];
};

const normalizeGeminiError = (error) => {
  if (!error) return new Error(`${PUBLIC_MODEL_NAME} upstream error`);
  const next =
    error instanceof Error ? error : new Error(String(error?.message || error));
  const text = String(next.message || "");
  if (!Number.isFinite(Number(next.status))) {
    const statusMatch = text.match(/"code"\s*:\s*(\d{3})/);
    if (statusMatch) next.status = Number(statusMatch[1]);
  }
  return next;
};

const isRetryableGeminiModelError = (error) => {
  const normalized = normalizeGeminiError(error);
  const text = String(normalized.message || "");
  return (
    Number(normalized.status) === 503 ||
    Number(normalized.status) === 429 ||
    /UNAVAILABLE|high demand|overloaded|RESOURCE_EXHAUSTED|rate limit/i.test(
      text,
    )
  );
};

const runGeminiWithModel = async (body, modelId, streamCallback) => {
  const apiKey = getGeminiApiKey(body);
  if (!apiKey) throw new Error("Server model credentials are not configured.");

  const client = new GoogleGenAI({ apiKey });
  const messages = await mapOpenAiToGeminiMessages(body.messages);
  const systemInstruction = getSystemInstruction(body.messages);
  const thinkingConfig = getThinkingConfigForModel(modelId, body);
  const config = {
    maxOutputTokens: resolveMaxTokens(body.max_tokens ?? body.maxTokens),
    ...(typeof body.temperature === "number"
      ? { temperature: body.temperature }
      : {}),
    ...(systemInstruction ? { systemInstruction } : {}),
    ...(thinkingConfig ? { thinkingConfig } : {}),
    ...(body.use_tools !== false
      ? { tools: mapOpenAiToolsToGemini(TOOL_DEFINITIONS) }
      : {}),
  };

  if (body.stream && body.use_tools === false) {
    const result = await client.models.generateContentStream({
      model: modelId,
      contents: messages,
      config,
    });
    let fullText = "";
    for await (const chunk of result) {
      const chunkText =
        typeof chunk?.text === "function" ? chunk.text() : chunk?.text || "";
      fullText += chunkText;
      if (streamCallback) streamCallback(chunkText);
    }
    return { content: fullText, usage: null };
  }

  const contents = [...messages];
  const toolsUsed = [];
  let delegatedResearch = null;

  for (let turn = 0; turn < MAX_TOOL_TURNS; turn += 1) {
    const responseResult = await client.models.generateContent({
      model: modelId,
      contents,
      config,
    });
    const response = responseResult?.response || responseResult;
    const functionCalls = getGeminiFunctionCalls(response);

    if (!functionCalls.length) {
      return {
        content: getGeminiResponseText(response),
        toolsUsed,
        usage: null,
        ...(delegatedResearch ? { delegatedResearch } : {}),
      };
    }

    contents.push(getGeminiResponseContent(response));

    const functionResponseParts = [];
    for (const call of functionCalls) {
      const toolResult = await executeTool(call, {
        messages: body.messages,
        requestBody: body,
        delegationContext: body.delegationContext || {},
      });
      if (toolResult.delegatedResearch)
        delegatedResearch = toolResult.delegatedResearch;
      toolsUsed.push({
        name: call.name,
        args: JSON.stringify(call.args || {}),
        ...(toolResult.meta || {}),
      });
      functionResponseParts.push({
        functionResponse: {
          name: call.name,
          response: { content: toolResult.content },
        },
      });
    }

    contents.push({
      role: "user",
      parts: functionResponseParts,
    });
  }

  throw new Error(
    `${PUBLIC_MODEL_NAME} tool loop exceeded max tool turns (${MAX_TOOL_TURNS})`,
  );
};

const runGeminiChat = async (body, streamCallback) => {
  const modelChain = getGeminiFallbackModelChain(
    body.model,
    body.modelChain || body.model_chain,
  );
  let lastError = null;

  for (let attempt = 0; attempt < modelChain.length; attempt += 1) {
    const modelId = modelChain[attempt];
    try {
      return await runGeminiWithModel(body, modelId, streamCallback);
    } catch (error) {
      lastError = normalizeGeminiError(error);
      if (
        !isRetryableGeminiModelError(lastError) ||
        attempt === modelChain.length - 1
      ) {
        throw lastError;
      }
      await sleep(getBackoffDelayMs(attempt, lastError.headers));
    }
  }

  throw (
    lastError ||
    new Error(`${PUBLIC_MODEL_NAME} upstream fallback chain failed`)
  );
};

const createProviderError = (status, message, payload = null) => {
  const error = new Error(message);
  error.status = status;
  error.payload = payload;
  return error;
};

const normalizeOpenRouterError = (status, payload, fallbackMessage = "") => {
  const message = normalizeText(
    payload?.error?.message ||
      payload?.message ||
      fallbackMessage ||
      "OpenRouter request failed.",
  );
  return createProviderError(status, message, payload);
};

const normalizeOpenRouterCaughtError = (error) =>
  error instanceof Error
    ? error
    : createProviderError(
        Number(error?.status) || 502,
        normalizeText(error?.message || error || "OpenRouter request failed."),
        error?.payload || null,
      );

const parseOpenRouterMessageText = (message = {}) => {
  const content = message?.content;
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((part) => {
      if (typeof part === "string") return part;
      if (typeof part?.text === "string") return part.text;
      return "";
    })
    .filter(Boolean)
    .join("");
};

const buildOpenRouterHeaders = (apiKey) => {
  const headers = {
    Authorization: `Bearer ${apiKey}`,
    "Content-Type": "application/json",
  };
  if (OPENROUTER_REFERER) headers["HTTP-Referer"] = OPENROUTER_REFERER;
  if (OPENROUTER_TITLE) headers["X-Title"] = OPENROUTER_TITLE;
  return headers;
};

const parseOpenRouterResponse = async (response) => {
  const contentType = normalizeText(
    response.headers.get("content-type"),
  ).toLowerCase();
  if (contentType.includes("application/json")) {
    const payload = await response.json().catch(() => null);
    return payload;
  }
  const text = await response.text().catch(() => "");
  try {
    return JSON.parse(text);
  } catch {
    return { message: text };
  }
};

const consumeOpenRouterStream = async (response, streamCallback) => {
  const reader = response.body?.getReader?.();
  if (!reader) {
    throw createProviderError(
      502,
      "OpenRouter streaming response was unavailable.",
    );
  }

  const decoder = new globalThis.TextDecoder();
  let buffer = "";
  let content = "";

  const flushChunk = (block) => {
    const lines = String(block || "")
      .split(/\r?\n/)
      .filter(Boolean);
    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const raw = line.slice(5).trim();
      if (!raw || raw === "[DONE]") continue;
      try {
        const payload = JSON.parse(raw);
        const delta = payload?.choices?.[0]?.delta?.content;
        if (typeof delta === "string" && delta) {
          content += delta;
          if (typeof streamCallback === "function") streamCallback(delta);
        }
      } catch {
        // Ignore malformed stream chunks and continue accumulating usable deltas.
      }
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let separatorIndex = buffer.indexOf("\n\n");
    while (separatorIndex >= 0) {
      flushChunk(buffer.slice(0, separatorIndex));
      buffer = buffer.slice(separatorIndex + 2);
      separatorIndex = buffer.indexOf("\n\n");
    }

    if (done) {
      if (buffer.trim()) flushChunk(buffer);
      break;
    }
  }

  return { content, usage: null, toolsUsed: [] };
};

const runOpenRouterWithModel = async (body, modelId, streamCallback) => {
  const apiKey = getOpenRouterApiKey(body);
  if (!apiKey) throw new Error("OpenRouter credentials are not configured.");

  const messages = Array.isArray(body.messages)
    ? JSON.parse(JSON.stringify(body.messages))
    : [];
  const basePayload = {
    model: modelId,
    messages,
    max_tokens: resolveMaxTokens(body.max_tokens ?? body.maxTokens),
    ...(typeof body.temperature === "number"
      ? { temperature: body.temperature }
      : {}),
    ...(body.use_tools !== false
      ? { tools: TOOL_DEFINITIONS, tool_choice: "auto" }
      : {}),
  };

  if (body.stream && body.use_tools === false) {
    const response = await fetch(OPENROUTER_API_URL, {
      method: "POST",
      headers: buildOpenRouterHeaders(apiKey),
      body: JSON.stringify({
        ...basePayload,
        stream: true,
      }),
    });
    if (!response.ok) {
      throw normalizeOpenRouterError(
        response.status,
        await parseOpenRouterResponse(response),
        `OpenRouter stream failed with status ${response.status}.`,
      );
    }
    return consumeOpenRouterStream(response, streamCallback);
  }

  const toolsUsed = [];
  let delegatedResearch = null;
  for (let turn = 0; turn < MAX_TOOL_TURNS; turn += 1) {
    const response = await fetch(OPENROUTER_API_URL, {
      method: "POST",
      headers: buildOpenRouterHeaders(apiKey),
      body: JSON.stringify(basePayload),
    });
    const payload = await parseOpenRouterResponse(response);
    if (!response.ok) {
      throw normalizeOpenRouterError(
        response.status,
        payload,
        `OpenRouter request failed with status ${response.status}.`,
      );
    }

    const message = payload?.choices?.[0]?.message || {};
    const toolCalls = Array.isArray(message?.tool_calls)
      ? message.tool_calls
      : [];
    if (!toolCalls.length) {
      return {
        content: parseOpenRouterMessageText(message),
        usage: payload?.usage || null,
        toolsUsed,
        ...(delegatedResearch ? { delegatedResearch } : {}),
      };
    }

    messages.push({
      role: "assistant",
      content: message.content || "",
      tool_calls: toolCalls,
    });

    for (const call of toolCalls) {
      const toolResult = await executeTool(call, {
        messages: body.messages,
        requestBody: body,
        delegationContext: body.delegationContext || {},
      });
      if (toolResult.delegatedResearch)
        delegatedResearch = toolResult.delegatedResearch;
      toolsUsed.push({
        name: call?.function?.name || call?.name,
        args:
          typeof call?.function?.arguments === "string"
            ? call.function.arguments
            : JSON.stringify(call?.args || call?.function?.arguments || {}),
        ...(toolResult.meta || {}),
      });
      messages.push({
        role: "tool",
        tool_call_id:
          call?.id || `${call?.function?.name || call?.name || "tool"}-${turn}`,
        content: toolResult.content,
      });
    }
  }

  throw new Error(
    `${PUBLIC_MODEL_NAME} tool loop exceeded max tool turns (${MAX_TOOL_TURNS})`,
  );
};

const runOpenRouterChat = async (body, streamCallback) => {
  const modelChain = getOpenRouterFallbackModelChain(
    body.model,
    body.modelChain || body.model_chain,
  );
  let lastError = null;

  for (let attempt = 0; attempt < modelChain.length; attempt += 1) {
    const modelId = modelChain[attempt];
    try {
      return await runOpenRouterWithModel(
        {
          ...body,
          model: modelId,
        },
        modelId,
        streamCallback,
      );
    } catch (error) {
      lastError = normalizeOpenRouterCaughtError(error);
      if (
        !Number.isFinite(Number(lastError?.status)) ||
        Number(lastError.status) < 429 ||
        attempt === modelChain.length - 1
      ) {
        throw lastError;
      }
      await sleep(getBackoffDelayMs(attempt, lastError.headers));
    }
  }

  throw lastError || new Error("OpenRouter upstream fallback chain failed");
};

const metadataPayload = () => ({
  ok: true,
  endpoint: "/api/chat",
  provider: PUBLIC_MODEL_NAME,
  brand: PUBLIC_MODEL_NAME,
  developer: PUBLIC_DEVELOPER_NAME,
  models: {
    default: PUBLIC_MODEL_NAME,
    available: [PUBLIC_MODEL_NAME],
    label: PUBLIC_MODEL_NAME,
  },
  default_execution_mode: "completion",
  agentic: true,
  stream: "SSE",
  rate_limit_strategy: {
    exact_completion_cache: true,
    max_completion_tokens: SAFE_MAX_COMPLETION_TOKENS,
  },
  tools: TOOL_DEFINITIONS.map((tool) => tool.function.name),
  example: {
    model: PUBLIC_MODEL_NAME,
    messages: [{ role: "user", content: "Hello!" }],
    stream: true,
  },
  image_input: {
    supported: true,
    accepted_fields: [
      "messages[].content[].image_url.url",
      "image",
      "image_url",
      "images",
      "image_urls",
    ],
  },
  memory: {
    headers: ["X-API-Key", "Authorization: Bearer <key>", "X-State-Key"],
    behavior:
      "Scoped memory is stored server-side. Thin requests can send only the latest turn and the backend will search the key-scoped memory table for related context before answering. When CEREBRAS_API_KEY is configured, a Cerebras Qwen helper reranks memory matches before the primary model answers.",
  },
});

const createHttpError = (status, message) => {
  const error = new Error(message);
  error.status = status;
  return error;
};

const normalizeChatBody = (body = {}) => {
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    throw createHttpError(400, "Invalid JSON body");
  }

  const resolvedMessages = resolveIncomingMessages(body);
  if (!Array.isArray(resolvedMessages) || resolvedMessages.length === 0) {
    throw createHttpError(
      400,
      "Provide a non-empty messages array or send prompt/message/input with image/image_url/images.",
    );
  }

  const requestedModel =
    normalizeText(body.model) ||
    (normalizeText(body.provider).toLowerCase() === OPENROUTER_PROVIDER
      ? DEFAULT_OPENROUTER_MODEL
      : DEFAULT_MODEL);
  const normalizedProvider = normalizeProvider(
    body.provider,
    normalizeModel(requestedModel),
  );
  const normalizedProviderApiKeys = normalizeProviderApiKeys(
    body.providerApiKeys || body.provider_api_keys,
  );
  const hasCredentials =
    normalizedProvider === OPENROUTER_PROVIDER
      ? getOpenRouterApiKeys({ providerApiKeys: normalizedProviderApiKeys })
          .length > 0
      : getGeminiApiKeys({ providerApiKeys: normalizedProviderApiKeys })
          .length > 0;
  if (!hasCredentials) {
    throw createHttpError(
      503,
      normalizedProvider === OPENROUTER_PROVIDER
        ? "OpenRouter credentials are not configured."
        : "Server model credentials are not configured.",
    );
  }

  const normalizedMessages = normalizeMessagesForModel(
    injectBrandSystemMessage(resolvedMessages, {
      research_mode:
        body.research_mode === true || body.research_synthesis === true,
    }),
  );
  const hasUsablePrompt = normalizedMessages.some((message) => {
    if (message?.role === "system") return false;
    if (typeof message?.content === "string")
      return Boolean(message.content.trim());
    return Array.isArray(message?.content) && message.content.length > 0;
  });
  if (!hasUsablePrompt) {
    throw createHttpError(
      400,
      "Provide at least one user message, prompt, or image input.",
    );
  }

  return {
    ...body,
    provider: normalizedProvider,
    providerApiKeys: normalizedProviderApiKeys,
    model: normalizeModel(requestedModel),
    max_tokens: resolveMaxTokens(body.max_tokens ?? body.maxTokens),
    messages: normalizedMessages,
  };
};

const runLiteHostChat = async (body, streamCallback) => {
  const normalizedBody = normalizeChatBody(body);
  let reply;
  if (normalizedBody.provider === OPENROUTER_PROVIDER) {
    reply = await runOpenRouterChat(normalizedBody, streamCallback);
  } else {
    reply = await runGeminiChat(normalizedBody, streamCallback);
  }
  return { body: normalizedBody, reply };
};

const createChatResponsePayload = (body, reply) => {
  const delegatedResearch =
    reply?.delegatedResearch && typeof reply.delegatedResearch === "object"
      ? reply.delegatedResearch
      : null;
  const researchMeta = delegatedResearch
    ? {
        delegated: true,
        runId: delegatedResearch.runId || "",
        outputMode: delegatedResearch.outputMode || null,
        sourceSelection: delegatedResearch.sourceSelection || null,
        noGroundedSources: delegatedResearch.noGroundedSources === true,
        tribunal: delegatedResearch.tribunal || null,
        convergence: delegatedResearch.convergence || null,
        verifierSummary: delegatedResearch.verifierSummary || null,
      }
    : null;

  return {
    ok: true,
    model: PUBLIC_MODEL_NAME,
    developer: PUBLIC_DEVELOPER_NAME,
    output_text: reply.content,
    choices: [{ message: { content: reply.content }, finish_reason: "stop" }],
    finish_reason: "stop",
    agentic: body.use_tools !== false,
    tools_used: reply.toolsUsed || [],
    usage: reply.usage,
    ...(delegatedResearch
      ? {
          sources: Array.isArray(delegatedResearch.sources)
            ? delegatedResearch.sources
            : [],
          researchMeta,
          delegated_research: delegatedResearch,
        }
      : {}),
  };
};

module.exports = {
  PUBLIC_MODEL_NAME,
  PUBLIC_DEVELOPER_NAME,
  BRAND_SYSTEM_PROMPT,
  metadataPayload,
  wantsStream,
  normalizeChatBody,
  runLiteHostChat,
  createChatResponsePayload,
};
