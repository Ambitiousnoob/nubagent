const path = require("node:path");
const dotenv = require("dotenv");
dotenv.config();
dotenv.config({ path: path.join(__dirname, "..", ".env") });

const { readBody } = require("../lib/web");
require("dotenv/config");

const DEFAULT_MODEL = "qwen-3-235b-a22b-instruct-2507";
const API_URL = "https://api.cerebras.ai/v1/chat/completions";
const ALLOWED_MODELS = new Set([DEFAULT_MODEL]);

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

const normalizeModel = (model) => {
    const id = typeof model === "string" ? model.trim() : "";
    const lowered = id.toLowerCase();

    // Map any legacy "polly" selections or unknown models to the current default.
    if (!id || lowered === "polly" || lowered.startsWith("polly-")) return DEFAULT_MODEL;

    if (ALLOWED_MODELS.has(id)) return id;
    if (ALLOWED_MODELS.has(lowered)) return lowered;

    return DEFAULT_MODEL;
};

const askAgent = async (body, streamCallback) => {
    const model = normalizeModel(body.model);
    const payload = {
        model,
        messages: Array.isArray(body.messages) ? body.messages : [],
        stream: wantsStream(body),
    };
    if (Array.isArray(body.tools)) payload.tools = body.tools;
    if (body.tool_choice) payload.tool_choice = body.tool_choice;
    if (body.temperature !== undefined) payload.temperature = body.temperature;
    if (body.max_tokens !== undefined) payload.max_tokens = body.max_tokens;

    const apiKey = process.env.CEREBRAS_API_KEY;
    const response = await fetch(API_URL, {
        method: "POST",
        headers: {
            "Authorization": `Bearer ${apiKey}`,
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
            const lines = buffer.split('\n');
            buffer = lines.pop() || "";

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                if (line === 'data: [DONE]') continue;
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
        return fullText;
    }

    const json = await response.json();
    return json.choices?.[0]?.message?.content || json.output_text || "";
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
    agentic: false,
    stream: "SSE",
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
        if (wantsStream(body)) {
            res.status(200);
            res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
            res.setHeader("Cache-Control", "no-cache, no-transform");
            res.setHeader("Connection", "keep-alive");

            const streamCallback = (chunk) => {
                res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: chunk } }] })}\n\n`);
            };

            await Promise.race([
                askAgent(body, streamCallback),
                new Promise((_, reject) => setTimeout(() => reject(new Error(`Timed out after ${HARD_TIMEOUT_MS}ms`)), HARD_TIMEOUT_MS)),
            ]);

            res.write("data: [DONE]\n\n");
            res.end();
        } else {
            const reply = await Promise.race([
                askAgent(body),
                new Promise((_, reject) => setTimeout(() => reject(new Error(`Timed out after ${HARD_TIMEOUT_MS}ms`)), HARD_TIMEOUT_MS)),
            ]);

            sendJson(res, 200, {
                ok: true,
                model: normalizedModel,
                output_text: reply,
                choices: [{ message: { content: reply }, finish_reason: "stop" }],
                finish_reason: "stop",
                agentic: false,
            });
        }
    } catch (error) {
        console.error("[Cerebras API Error]", error);
        sendJson(res, 500, { error: error?.message || "Chat request failed." });
    }
};
