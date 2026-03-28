const path = require("node:path");
const dotenv = require("dotenv");
dotenv.config();
dotenv.config({ path: path.join(__dirname, "..", ".env") });

const { readBody } = require("../lib/web");
require("dotenv/config");

const DEFAULT_MODEL = "qwen-3-235b-a22b-instruct-2507";
const API_URL = "https://api.cerebras.ai/v1/chat/completions";
const ALLOWED_MODELS = new Set([DEFAULT_MODEL]);
const MAX_TOOL_TURNS = 6;

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

const executeTool = async (toolCall) => {
    const name = toolCall?.function?.name;
    const argsRaw = toolCall?.function?.arguments || "{}";
    let args = {};
    try {
        args = JSON.parse(argsRaw);
    } catch (e) {
        return `Error: could not parse arguments (${e.message})`;
    }

    if (name === "calculate") return calcHandler(args);
    if (name === "web_search") return searchHandler(args);
    if (name === "web_fetch") return fetchHandler(args);
    if (name === "search_images") return imageSearchHandler(args);
    if (name === "view_image") return viewImageHandler(args);
    return `Error: unknown tool ${name}`;
};

const normalizeModel = (model) => {
    const id = typeof model === "string" ? model.trim() : "";
    const lowered = id.toLowerCase();

    // Map any legacy "polly" selections or unknown models to the current default.
    if (!id || lowered === "polly" || lowered.startsWith("polly-")) return DEFAULT_MODEL;

    if (ALLOWED_MODELS.has(id)) return id;
    if (ALLOWED_MODELS.has(lowered)) return lowered;

    return DEFAULT_MODEL;
};

const callCerebras = async (payload, streamCallback) => {
    const apiKey = process.env.CEREBRAS_API_KEY;
    const response = await fetch(API_URL, {
        method: "POST",
        headers: {
            Authorization: `Bearer ${apiKey}`,
            "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        const errorText = await response.text();
        console.error("Cerebras API Error:", errorText);
        throw new Error(`Cerebras API error: ${response.status} ${errorText}`);
    }

    if (payload.stream) {
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
        return { content: fullText };
    }

    const json = await response.json();
    const choice = json.choices?.[0] || {};
    return {
        content: choice.message?.content || json.output_text || "",
        message: choice.message,
    };
};

const runChatWithTools = async (body) => {
    const model = normalizeModel(body.model);
    const messages = Array.isArray(body.messages) ? [...body.messages] : [];
    const base = {
        model,
        temperature: body.temperature,
        max_tokens: body.max_tokens,
        tools: TOOL_DEFINITIONS,
        parallel_tool_calls: true,
        stream: false, // tool path uses non-stream for determinism
    };

    let lastContent = "";
    const toolsUsed = [];
    for (let i = 0; i < MAX_TOOL_TURNS; i += 1) {
        const payload = { ...base, messages };
        const result = await callCerebras(payload);
        const assistantMessage = result.message || { role: "assistant", content: result.content };
        messages.push(assistantMessage);

        const toolCalls = assistantMessage?.tool_calls || assistantMessage?.toolCalls || [];
        if (!toolCalls.length) {
            lastContent = assistantMessage?.content || "";
            break;
        }

        for (const call of toolCalls) {
            const toolResult = await executeTool(call);
            toolsUsed.push({ name: call.function?.name || call.id, args: call.function?.arguments || "{}" });
            messages.push({
                role: "tool",
                tool_call_id: call.id || call?.function?.name || "tool-call",
                content: toolResult,
            });
        }
    }

    return { content: lastContent, toolsUsed };
};

const metadataPayload = () => ({
    ok: true,
    endpoint: "/api/chat",
    provider: "nub-agent (ambitiousnoob) via Cerebras",
    brand: "nub-agent (ambitiousnoob)",
    models: {
        default: DEFAULT_MODEL,
        available: Array.from(ALLOWED_MODELS),
        label: "nub-agent (ambitiousnoob) · Cerebras Qwen 3 235B",
    },
    default_execution_mode: "completion",
    agentic: true,
    stream: "SSE (streaming disabled during tool calls)",
    tools: TOOL_DEFINITIONS.map((t) => t.function.name),
    example: {
        model: DEFAULT_MODEL,
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
    body.model = normalizedModel;

    const HARD_TIMEOUT_MS = 60000;
    try {
        const allowTools = body.use_tools !== false;
        const wantsSse = wantsStream(body) && !allowTools;

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

            sendJson(res, 200, {
                ok: true,
                model: normalizedModel,
                output_text: outputContent,
                choices: [{ message: { content: outputContent }, finish_reason: "stop" }],
                finish_reason: "stop",
                agentic: allowTools,
                tools_used: toolsUsed,
            });
        }
    } catch (error) {
        console.error("[Cerebras API Error]", error);
        sendJson(res, 500, { error: error?.message || "Chat request failed." });
    }
};
