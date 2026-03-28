const path = require("node:path");
const dotenv = require("dotenv");
dotenv.config();
dotenv.config({ path: path.join(__dirname, "..", ".env") });

const { readBody } = require("../lib/web");
const {
    SlidingWindowTokenLimiter,
    ExactResponseCache,
    clampCompletionTokens,
    estimatePayloadTokens,
    extractUsage,
    getBackoffDelayMs,
    mergeUsage,
    parseHeaderNumber,
    sleep,
} = require("../lib/ai-control");
require("dotenv/config");

const { GoogleGenAI } = require("@google/genai");

const DEFAULT_MODEL = "gemini-2.5-flash-lite";
const PUBLIC_MODEL_NAME = "nub-agent";
const FALLBACK_MODELS = [
    DEFAULT_MODEL,
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
];
const MAX_TOOL_TURNS = 6;
const SAFE_MAX_COMPLETION_TOKENS = clampCompletionTokens(process.env.GEMINI_MAX_COMPLETION_TOKENS || 4096);
const MAX_GEMINI_ATTEMPTS = Math.max(1, Number(process.env.GEMINI_MAX_RETRIES) || 3);
const RATE_LIMITER = new SlidingWindowTokenLimiter({
    tokensPerMinute: Number(process.env.GEMINI_TPM_LIMIT) || 30000,
    remoteHeadroom: Number(process.env.GEMINI_RATE_LIMIT_HEADROOM) || 1000,
});
const RESPONSE_CACHE = new ExactResponseCache({
    ttlMs: Number(process.env.EXACT_COMPLETION_CACHE_TTL_MS) || 5 * 60 * 1000,
    maxEntries: Number(process.env.EXACT_COMPLETION_CACHE_MAX_ENTRIES) || 100,
});

const { definition: calcDef, handler: calcHandler } = require("./tools/calculate");
const { definition: searchDef, handler: searchHandler } = require("./tools/web_search");
const { definition: fetchDef, handler: fetchHandler } = require("./tools/web_fetch");
const { definition: imageSearchDef, handler: imageSearchHandler } = require("./tools/search_images");
const { definition: viewImageDef, handler: viewImageHandler } = require("./tools/view_image");

const writeCorsHeaders = (res) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
};

const sendJson = (res, status, payload) => {
    res.status(status);
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(JSON.stringify(payload));
};

const wantsStream = (body = {}) => {
    const responseType = String(body.responseType || body.format || "").toLowerCase();
    if (responseType === "stream" || responseType === "sse") return true;
    if (responseType === "json" || responseType === "raw") return false;
    return body.stream === true;
};

const TOOL_DEFINITIONS = [calcDef, searchDef, fetchDef, imageSearchDef, viewImageDef];

const getMessageText = (content) => {
    if (typeof content === "string") return content;
    if (!Array.isArray(content)) return "";
    return content
        .filter((part) => part?.type === "text" && typeof part.text === "string")
        .map((part) => part.text)
        .join("\n\n");
};

const normalizeMessageContentForModel = (content) => {
    if (typeof content === "string") return content;
    if (!Array.isArray(content)) return content;

    return content
        .map((part) => {
            if (!part || typeof part !== "object") return null;
            if (part.type === "text" && typeof part.text === "string") {
                return { type: "text", text: part.text };
            }
            if (part.type === "image_url" && typeof part.image_url?.url === "string") {
                return {
                    type: "image_url",
                    image_url: { url: part.image_url.url },
                };
            }
            return null;
        })
        .filter(Boolean);
};

const normalizeMessagesForModel = (messages = []) => (
    (Array.isArray(messages) ? messages : []).map((message) => (
        message && typeof message === "object"
            ? {
                ...message,
                content: normalizeMessageContentForModel(message.content),
            }
            : message
    ))
);

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
            const name = (parenIndex >= 0 ? afterPrefix.slice(0, parenIndex) : afterPrefix).trim();
            return name ? { kind, name } : null;
        })
        .filter(Boolean);
};

const extractAttachmentOcrEntries = (messages = []) => {
    const source = getLatestUserText(messages);
    if (!source) return [];

    const entries = [];
    const pattern = /--- OCR:\s*(.+?) ---\n([\s\S]*?)(?=\n{2}--- (?:OCR|File):|\n{2}Relevant retrieved context:|$)/g;
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
    const manifestEntries = extractAttachmentManifestEntries(messages).filter((entry) => entry.kind === "image");
    if (!ocrEntries.length && !manifestEntries.length) return null;

    const fileName = decodeURIComponent(parsed.pathname.split("/").pop() || "").toLowerCase();
    const matched = ocrEntries.find((entry) => entry.name.toLowerCase() === fileName)
        || ocrEntries.find((entry) => fileName && entry.name.toLowerCase().includes(fileName))
        || manifestEntries.find((entry) => entry.name.toLowerCase() === fileName)
        || manifestEntries.find((entry) => fileName && entry.name.toLowerCase().includes(fileName))
        || (parsed.hostname.toLowerCase() === "example.com" && ocrEntries.length === 1 ? ocrEntries[0] : null)
        || (parsed.hostname.toLowerCase() === "example.com" && manifestEntries.length === 1 ? manifestEntries[0] : null);

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

    const hasOcr = typeof matched.text === "string" && matched.text.trim().length > 0;
    const ocrText = hasOcr
        ? (matched.text.length > 5000 ? `${matched.text.slice(0, 5000)}...` : matched.text)
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
        try { processedArgs = JSON.parse(args); } catch (e) { return `Error: could not parse arguments (${e.message})`; }
    }

    if (name === "calculate") return { content: await calcHandler(processedArgs), meta: null };
    if (name === "web_search") return { content: await searchHandler(processedArgs), meta: null };
    if (name === "web_fetch") return { content: await fetchHandler(processedArgs), meta: null };
    if (name === "search_images") return { content: await imageSearchHandler(processedArgs), meta: null };
    if (name === "view_image") {
        const attachmentProxy = resolveAttachmentViewImage(processedArgs, context.messages);
        if (attachmentProxy) return attachmentProxy;
        return { content: await viewImageHandler(processedArgs), meta: null };
    }
    return { content: `Error: unknown tool ${name}`, meta: null };
};

const normalizeModel = (model) => {
    const id = typeof model === "string" ? model.trim() : "";
    const lowered = id.toLowerCase();

    if (!id || lowered === "polly" || lowered.startsWith("polly-")) return DEFAULT_MODEL;
    if (lowered === PUBLIC_MODEL_NAME || lowered === "nub agent" || lowered === "gemini-3-flash") return DEFAULT_MODEL;
    if (lowered === "gemini-3-flash-preview") return "gemini-3-flash-preview";
    if (lowered === "gemini-2.5-flash") return "gemini-2.5-flash";
    if (lowered === "gemini-2.5-flash-lite") return "gemini-2.5-flash-lite";
    return DEFAULT_MODEL;
};

const resolveMaxTokens = (value) => (
    Math.min(clampCompletionTokens(value, SAFE_MAX_COMPLETION_TOKENS), SAFE_MAX_COMPLETION_TOKENS)
);

const getGeminiApiKey = () => {
    const keys = (process.env.GEMINI_API_KEYS || "").split(",").map(k => k.trim()).filter(Boolean);
    if (!keys.length) return null;
    return keys[Math.floor(Math.random() * keys.length)];
};

const getFallbackModelChain = (requestedModel) => {
    const configured = (process.env.GEMINI_FALLBACK_MODELS || FALLBACK_MODELS.join(","))
        .split(",")
        .map((item) => normalizeModel(item))
        .filter(Boolean);
    return [...new Set([normalizeModel(requestedModel), ...configured])];
};

const getThinkingConfigForModel = (modelId, body = {}) => {
    const explicitBudget = body.thinking_budget ?? body.thinkingBudget;
    if (Number.isFinite(Number(explicitBudget))) {
        return { thinkingBudget: Number(explicitBudget) };
    }

    const normalized = normalizeModel(modelId);
    if (normalized === "gemini-2.5-flash" || normalized === "gemini-2.5-flash-lite") {
        return { thinkingBudget: 0 };
    }
    if (normalized === "gemini-3-flash-preview") {
        return { thinkingLevel: "minimal" };
    }
    return undefined;
};

const parseInlineDataUrl = (value) => {
    const match = String(value || "").match(/^data:([^;,]+);base64,(.+)$/i);
    if (!match) return null;
    return {
        mimeType: match[1],
        data: match[2],
    };
};

const mapOpenAiContentToGeminiParts = (content) => {
    if (typeof content === "string") {
        return content ? [{ text: content }] : [];
    }

    if (!Array.isArray(content)) return [];

    return content.flatMap((part) => {
        if (!part || typeof part !== "object") return [];
        if (part.type === "text" && typeof part.text === "string" && part.text.trim()) {
            return [{ text: part.text }];
        }
        if (part.type === "image_url" && typeof part.image_url?.url === "string") {
            const inlineData = parseInlineDataUrl(part.image_url.url);
            if (inlineData) {
                return [{ inlineData }];
            }
            return [{ text: `[Remote image URL provided: ${part.image_url.url}]` }];
        }
        return [];
    });
};

const getSystemInstruction = (messages = []) => {
    const text = (Array.isArray(messages) ? messages : [])
        .filter((message) => message?.role === "system")
        .map((message) => getMessageText(message.content))
        .filter(Boolean)
        .join("\n\n");
    return text || undefined;
};

const mapOpenAiToGeminiMessages = (messages = []) => {
    return (Array.isArray(messages) ? messages : [])
        .filter((message) => message?.role !== "system")
        .map((message) => {
            const role = message.role === "assistant" ? "model" : "user";
            const parts = mapOpenAiContentToGeminiParts(message.content);
            if (!parts.length) parts.push({ text: "" });
            return { role, parts };
        });
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

    return [{
        functionDeclarations: tools.map(t => ({
            name: t.function.name,
            description: t.function.description,
            parameters: scrub(t.function.parameters)
        }))
    }];
};

const getGeminiResponseText = (response) => {
    if (!response) return "";
    if (typeof response.text === "function") return response.text() || "";
    if (typeof response.text === "string") return response.text;
    return response?.candidates?.[0]?.content?.parts
        ?.map((part) => (typeof part?.text === "string" ? part.text : ""))
        .filter(Boolean)
        .join("") || "";
};

const getGeminiResponseContent = (response) => (
    response?.candidates?.[0]?.content || {
        role: "model",
        parts: [{ text: getGeminiResponseText(response) }],
    }
);

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
    if (!error) return new Error("Unknown Gemini error");
    const next = error instanceof Error ? error : new Error(String(error?.message || error));
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
    return Number(normalized.status) === 503
        || Number(normalized.status) === 429
        || /UNAVAILABLE|high demand|overloaded|RESOURCE_EXHAUSTED|rate limit/i.test(text);
};

const runGeminiWithModel = async (body, modelId, streamCallback) => {
    const apiKey = getGeminiApiKey();
    if (!apiKey) throw new Error("GEMINI_API_KEYS is not configured.");

    const client = new GoogleGenAI({ apiKey });
    const messages = mapOpenAiToGeminiMessages(body.messages);
    const systemInstruction = getSystemInstruction(body.messages);
    const thinkingConfig = getThinkingConfigForModel(modelId, body);
    const config = {
        maxOutputTokens: resolveMaxTokens(body.max_tokens ?? body.maxTokens),
        ...(typeof body.temperature === "number" ? { temperature: body.temperature } : {}),
        ...(systemInstruction ? { systemInstruction } : {}),
        ...(thinkingConfig ? { thinkingConfig } : {}),
        ...(body.use_tools !== false ? { tools: mapOpenAiToolsToGemini(TOOL_DEFINITIONS) } : {}),
    };

    if (body.stream && body.use_tools === false) {
        const result = await client.models.generateContentStream({
            model: modelId,
            contents: messages,
            config,
        });
        let fullText = "";
        for await (const chunk of result) {
            const chunkText = typeof chunk?.text === "function" ? chunk.text() : (chunk?.text || "");
            fullText += chunkText;
            if (streamCallback) streamCallback(chunkText);
        }
        return { content: fullText, usage: null };
    }

    const contents = [...messages];
    const toolsUsed = [];

    for (let turn = 0; turn < MAX_TOOL_TURNS; turn += 1) {
        const responseResult = await client.models.generateContent({
            model: modelId,
            contents,
            config,
        });
        const response = responseResult?.response || responseResult;
        const functionCalls = getGeminiFunctionCalls(response);

        if (!functionCalls.length) {
            return { content: getGeminiResponseText(response), toolsUsed, usage: null };
        }

        contents.push(getGeminiResponseContent(response));

        const functionResponseParts = [];
        for (const call of functionCalls) {
            const toolResult = await executeTool(call, { messages: body.messages });
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

    throw new Error(`Gemini API error: exceeded max tool turns (${MAX_TOOL_TURNS})`);
};

const runGeminiChat = async (body, streamCallback) => {
    const modelChain = getFallbackModelChain(body.model);
    let lastError = null;

    for (let attempt = 0; attempt < modelChain.length; attempt += 1) {
        const modelId = modelChain[attempt];
        try {
            return await runGeminiWithModel(body, modelId, streamCallback);
        } catch (error) {
            lastError = normalizeGeminiError(error);
            if (!isRetryableGeminiModelError(lastError) || attempt === modelChain.length - 1) {
                throw lastError;
            }
            await sleep(getBackoffDelayMs(attempt, lastError.headers));
        }
    }

    throw lastError || new Error("Gemini API error: no fallback model succeeded");
};

const metadataPayload = () => ({
    ok: true,
    endpoint: "/api/chat",
    provider: PUBLIC_MODEL_NAME,
    brand: PUBLIC_MODEL_NAME,
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
    tools: TOOL_DEFINITIONS.map((t) => t.function.name),
    example: {
        model: PUBLIC_MODEL_NAME,
        messages: [{ role: "user", content: "Hello!" }],
        stream: true,
    },
});

module.exports = async (req, res) => {
    writeCorsHeaders(res);

    if (req.method === "OPTIONS") {
        res.status(204).end();
        return;
    }

    if (req.method === "HEAD" || req.method === "GET") {
        sendJson(res, 200, metadataPayload());
        return;
    }

    if (req.method !== "POST") {
        sendJson(res, 405, { error: "Method not allowed" });
        return;
    }

    let body;
    try {
        body = await readBody(req);
    } catch (error) {
        sendJson(res, 400, { error: "Invalid JSON body" });
        return;
    }

    if (!body.messages || !Array.isArray(body.messages) || body.messages.length === 0) {
        sendJson(res, 400, { error: "Provide a non-empty messages array." });
        return;
    }

    if (!process.env.GEMINI_API_KEYS) {
        sendJson(res, 503, { error: "GEMINI_API_KEYS is not configured on the server." });
        return;
    }

    const normalizedModel = normalizeModel(body.model);
    body = {
        ...body,
        model: normalizedModel,
        max_tokens: resolveMaxTokens(body.max_tokens ?? body.maxTokens),
        messages: normalizeMessagesForModel(body.messages),
    };

    try {
        const wantsSse = wantsStream(body) && body.use_tools === false;
        
        if (wantsSse) {
            res.status(200);
            res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
            res.setHeader("Cache-Control", "no-cache, no-transform");
            res.setHeader("Connection", "keep-alive");

            const streamCallback = (chunk) => {
                res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: chunk } }] })}\n\n`);
            };

            await runGeminiChat(body, streamCallback);
            res.write("data: [DONE]\n\n");
            res.end();
        } else {
            const reply = await runGeminiChat(body);
            const responsePayload = {
                ok: true,
                model: PUBLIC_MODEL_NAME,
                output_text: reply.content,
                choices: [{ message: { content: reply.content }, finish_reason: "stop" }],
                finish_reason: "stop",
                agentic: body.use_tools !== false,
                tools_used: reply.toolsUsed || [],
                usage: reply.usage,
            };

            sendJson(res, 200, responsePayload);
        }
    } catch (error) {
        console.error("[Gemini API Error]", error);
        sendJson(res, 500, { error: error?.message || "Chat request failed." });
    }
};
