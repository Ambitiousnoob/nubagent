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
} = require("../lib/cerebras-control");
require("dotenv/config");

const DEFAULT_MODEL = "qwen-3-235b-a22b-instruct-2507";
const PUBLIC_MODEL_NAME = "nub-agent";
const API_URL = "https://api.cerebras.ai/v1/chat/completions";
const ALLOWED_MODELS = new Set([DEFAULT_MODEL]);
const MAX_TOOL_TURNS = 6;
const SAFE_MAX_COMPLETION_TOKENS = clampCompletionTokens(process.env.CEREBRAS_MAX_COMPLETION_TOKENS || 4096);
const MAX_CEREBRAS_ATTEMPTS = Math.max(1, Number(process.env.CEREBRAS_MAX_RETRIES) || 3);
const RATE_LIMITER = new SlidingWindowTokenLimiter({
    tokensPerMinute: Number(process.env.CEREBRAS_TPM_LIMIT) || 60000,
    remoteHeadroom: Number(process.env.CEREBRAS_RATE_LIMIT_HEADROOM) || 2000,
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

    const textParts = content
        .filter((part) => part?.type === "text" && typeof part.text === "string")
        .map((part) => part.text.trim())
        .filter(Boolean);

    if (textParts.length) return textParts.join("\n\n");
    if (content.some((part) => part?.type === "image_url")) {
        return "[Image input omitted: this backend currently accepts text content only. Use attachment OCR/context or a dedicated vision backend.]";
    }
    return "";
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
    const name = toolCall?.function?.name;
    const argsRaw = toolCall?.function?.arguments || "{}";
    let args = {};
    try {
        args = JSON.parse(argsRaw);
    } catch (e) {
        return `Error: could not parse arguments (${e.message})`;
    }

    if (name === "calculate") return { content: await calcHandler(args), meta: null };
    if (name === "web_search") return { content: await searchHandler(args), meta: null };
    if (name === "web_fetch") return { content: await fetchHandler(args), meta: null };
    if (name === "search_images") return { content: await imageSearchHandler(args), meta: null };
    if (name === "view_image") {
        const attachmentProxy = resolveAttachmentViewImage(args, context.messages);
        if (attachmentProxy) return attachmentProxy;
        return { content: await viewImageHandler(args), meta: null };
    }
    return { content: `Error: unknown tool ${name}`, meta: null };
};

const normalizeModel = (model) => {
    const id = typeof model === "string" ? model.trim() : "";
    const lowered = id.toLowerCase();

    // Map any legacy "polly" selections or unknown models to the current default.
    if (!id || lowered === "polly" || lowered.startsWith("polly-")) return DEFAULT_MODEL;
    if (lowered === PUBLIC_MODEL_NAME) return DEFAULT_MODEL;
    if (lowered === "nub agent") return DEFAULT_MODEL;

    if (ALLOWED_MODELS.has(id)) return id;
    if (ALLOWED_MODELS.has(lowered)) return lowered;

    return DEFAULT_MODEL;
};

const resolveMaxTokens = (value) => (
    Math.min(clampCompletionTokens(value, SAFE_MAX_COMPLETION_TOKENS), SAFE_MAX_COMPLETION_TOKENS)
);

const getRateLimitState = (headers) => {
    const remainingTokensMinute = parseHeaderNumber(headers, "x-ratelimit-remaining-tokens-minute");
    const resetTokensMinute = parseHeaderNumber(headers, "x-ratelimit-reset-tokens-minute");
    if (!Number.isFinite(remainingTokensMinute) && !Number.isFinite(resetTokensMinute)) return null;
    return {
        remaining_tokens_minute: Number.isFinite(remainingTokensMinute) ? remainingTokensMinute : null,
        reset_tokens_minute: Number.isFinite(resetTokensMinute) ? resetTokensMinute : null,
    };
};

const createCerebrasError = ({ status, message, headers }) => {
    const error = new Error(message);
    error.status = status;
    error.headers = headers;
    return error;
};

const estimateResponseUsage = (payload, content = "") => {
    const promptTokens = estimatePayloadTokens({
        messages: payload.messages,
        tools: payload.tools,
        max_tokens: 0,
    });
    const completionTokens = Math.min(
        resolveMaxTokens(payload.max_tokens ?? payload.maxTokens),
        Math.max(1, Math.ceil(String(content || "").length / 4)),
    );
    return {
        prompt_tokens: promptTokens,
        completion_tokens: completionTokens,
        total_tokens: promptTokens + completionTokens,
    };
};

const makeExactCacheKey = (body) => RESPONSE_CACHE.makeKey({
    model: normalizeModel(body.model),
    messages: normalizeMessagesForModel(body.messages),
    temperature: body.temperature ?? 0.2,
    max_tokens: resolveMaxTokens(body.max_tokens ?? body.maxTokens),
});

const callCerebras = async (payload, streamCallback) => {
    const apiKey = process.env.CEREBRAS_API_KEY;
    const normalizedPayload = {
        ...payload,
        max_tokens: resolveMaxTokens(payload.max_tokens ?? payload.maxTokens),
    };
    const estimatedTokens = estimatePayloadTokens(normalizedPayload);
    const reservationId = await RATE_LIMITER.reserve(estimatedTokens);

    try {
        for (let attempt = 0; attempt < MAX_CEREBRAS_ATTEMPTS; attempt += 1) {
            const response = await fetch(API_URL, {
                method: "POST",
                headers: {
                    Authorization: `Bearer ${apiKey}`,
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(normalizedPayload),
            });

            RATE_LIMITER.applyHeaders(response.headers);
            const rateLimit = getRateLimitState(response.headers);

            if (!response.ok) {
                const errorText = await response.text();
                console.error("Cerebras API Error:", errorText);
                const error = createCerebrasError({
                    status: response.status,
                    headers: response.headers,
                    message: `Cerebras API error: ${response.status} ${errorText}`,
                });

                if (response.status === 429 && attempt + 1 < MAX_CEREBRAS_ATTEMPTS) {
                    await sleep(getBackoffDelayMs(attempt, response.headers));
                    continue;
                }

                throw error;
            }

            if (normalizedPayload.stream) {
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let fullText = "";
                let buffer = "";

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split("\\n");
                    buffer = lines.pop() || "";

                    for (const line of lines) {
                        if (!line.startsWith("data: ")) continue;
                        if (line === "data: [DONE]") continue;
                        try {
                            const data = JSON.parse(line.slice(6));
                            const chunk = data.choices?.[0]?.delta?.content || "";
                            if (chunk) {
                                fullText += chunk;
                                streamCallback(chunk);
                            }
                        } catch (e) {
                            console.error("Error parsing stream chunk:", e);
                        }
                    }
                }

                const usage = estimateResponseUsage(normalizedPayload, fullText);
                RATE_LIMITER.commit(reservationId, usage.total_tokens);
                return { content: fullText, usage, rateLimit };
            }

            const json = await response.json();
            const choice = json.choices?.[0] || {};
            const content = choice.message?.content || json.output_text || "";
            const usage = extractUsage(json) || estimateResponseUsage(normalizedPayload, content);
            RATE_LIMITER.commit(reservationId, usage.total_tokens);
            return {
                content,
                message: choice.message,
                usage,
                rateLimit,
            };
        }
    } catch (error) {
        RATE_LIMITER.release(reservationId);
        throw error;
    }

    RATE_LIMITER.release(reservationId);
    throw createCerebrasError({
        status: 500,
        message: "Cerebras API error: exhausted retry budget",
    });
};

const runChatWithTools = async (body) => {
    const model = normalizeModel(body.model);
    const messages = normalizeMessagesForModel(body.messages);
    const base = {
        model,
        temperature: body.temperature,
        max_tokens: resolveMaxTokens(body.max_tokens ?? body.maxTokens),
        tools: TOOL_DEFINITIONS,
        parallel_tool_calls: true,
        stream: false, // tool path uses non-stream for determinism
    };

    let lastContent = "";
    const toolsUsed = [];
    const usageEntries = [];
    let lastRateLimit = null;
    for (let i = 0; i < MAX_TOOL_TURNS; i += 1) {
        const payload = { ...base, messages };
        const result = await callCerebras(payload);
        if (result?.usage) usageEntries.push(result.usage);
        if (result?.rateLimit) lastRateLimit = result.rateLimit;
        const assistantMessage = result.message || { role: "assistant", content: result.content };
        messages.push(assistantMessage);

        const toolCalls = assistantMessage?.tool_calls || assistantMessage?.toolCalls || [];
        if (!toolCalls.length) {
            lastContent = assistantMessage?.content || "";
            break;
        }

        for (const call of toolCalls) {
            const toolResult = await executeTool(call, { messages });
            toolsUsed.push({
                name: call.function?.name || call.id,
                args: call.function?.arguments || "{}",
                ...(toolResult?.meta || {}),
            });
            messages.push({
                role: "tool",
                tool_call_id: call.id || call?.function?.name || "tool-call",
                content: toolResult?.content || "",
            });
        }
    }

    return {
        content: lastContent,
        toolsUsed,
        usage: mergeUsage(usageEntries),
        rateLimit: lastRateLimit,
    };
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
    stream: "SSE (streaming disabled during tool calls)",
    rate_limit_strategy: {
        local_token_limiter: true,
        adaptive_header_throttling: true,
        retry_on_429: true,
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

    if (!process.env.CEREBRAS_API_KEY) {
        sendJson(res, 503, { error: "CEREBRAS_API_KEY is not configured on the server." });
        return;
    }

    const normalizedModel = normalizeModel(body.model);
    body = {
        ...body,
        model: normalizedModel,
        max_tokens: resolveMaxTokens(body.max_tokens ?? body.maxTokens),
        messages: normalizeMessagesForModel(body.messages),
    };

    const HARD_TIMEOUT_MS = 60000;
    try {
        const allowTools = body.use_tools !== false;
        const wantsSse = wantsStream(body) && !allowTools;
        const exactCacheEnabled = !allowTools && !wantsSse && body.cache !== false;
        const exactCacheKey = exactCacheEnabled ? makeExactCacheKey(body) : null;

        if (exactCacheKey) {
            const cached = RESPONSE_CACHE.get(exactCacheKey);
            if (cached) {
                sendJson(res, 200, {
                    ...cached,
                    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
                    rate_limit: null,
                    cache: {
                        hit: true,
                        strategy: "exact",
                        saved_tokens: cached?.usage?.total_tokens || null,
                    },
                });
                return;
            }
        }

        if (wantsSse) {
            res.status(200);
            res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
            res.setHeader("Cache-Control", "no-cache, no-transform");
            res.setHeader("Connection", "keep-alive");

            const streamCallback = (chunk) => {
                res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: chunk } }] })}\n\n`);
            };

            await Promise.race([
                callCerebras({
                    model: normalizedModel,
                    messages: body.messages,
                    stream: true,
                    temperature: body.temperature,
                    max_tokens: body.max_tokens,
                }, streamCallback),
                new Promise((_, reject) => setTimeout(() => reject(new Error(`Timed out after ${HARD_TIMEOUT_MS}ms`)), HARD_TIMEOUT_MS)),
            ]);

            res.write("data: [DONE]\n\n");
            res.end();
        } else {
            const reply = await Promise.race([
                allowTools
                    ? runChatWithTools(body)
                    : callCerebras({
                        model: normalizedModel,
                        messages: body.messages,
                        stream: false,
                        temperature: body.temperature,
                        max_tokens: body.max_tokens,
                    }),
                new Promise((_, reject) => setTimeout(() => reject(new Error(`Timed out after ${HARD_TIMEOUT_MS}ms`)), HARD_TIMEOUT_MS)),
            ]);

            const outputContent = reply && typeof reply === "object" && reply.content !== undefined
                ? reply.content
                : reply?.content || reply;
            const toolsUsed = reply && typeof reply === "object" && Array.isArray(reply.toolsUsed)
                ? reply.toolsUsed
                : [];
            const usage = reply && typeof reply === "object" && reply.usage
                ? reply.usage
                : null;
            const rateLimit = reply && typeof reply === "object" && reply.rateLimit
                ? reply.rateLimit
                : null;

            const responsePayload = {
                ok: true,
                model: PUBLIC_MODEL_NAME,
                output_text: outputContent,
                choices: [{ message: { content: outputContent }, finish_reason: "stop" }],
                finish_reason: "stop",
                agentic: allowTools,
                tools_used: toolsUsed,
                usage,
                rate_limit: rateLimit,
            };

            if (exactCacheKey) {
                RESPONSE_CACHE.set(exactCacheKey, responsePayload);
            }

            sendJson(res, 200, responsePayload);
        }
    } catch (error) {
        console.error("[Cerebras API Error]", error);
        sendJson(res, 500, { error: error?.message || "Chat request failed." });
    }
};
