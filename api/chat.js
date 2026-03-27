const path = require("node:path");
const dotenv = require("dotenv");
dotenv.config();
dotenv.config({ path: path.join(__dirname, "..", ".env") });

const { readBody } = require("../lib/web");
const { getAiClient, resetAiClient } = require("../lib/ai-client");
const mysql = require("mysql2/promise");
require("dotenv/config");

const GEMINI_MODEL = "gemini-1.5-flash";
const ROUTER_MODEL = "gemini-1.5-flash";
const MAX_AGENT_ITERATIONS = 3;

const writeCorsHeaders = (res) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
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

const normalizeText = (value) => String(value ?? "").replace(/\u0000/g, "").trim();

const extractTextFromMessage = (message = {}) => {
    if (typeof message.content === "string") return normalizeText(message.content);
    if (Array.isArray(message.content)) {
        return normalizeText(
            message.content
                .filter((part) => part?.type === "text" && typeof part.text === "string")
                .map((part) => part.text)
                .join("\n")
        );
    }
    return "";
};

const extractPrompt = (body = {}) => {
    const direct = normalizeText(body.prompt || body.input || body.message || "");
    if (direct) return direct;

    if (Array.isArray(body.messages)) {
        for (let i = body.messages.length - 1; i >= 0; i -= 1) {
            const msg = body.messages[i];
            if (msg?.role === "user" || msg?.role === "system") {
                const text = extractTextFromMessage(msg);
                if (text) return text;
            }
        }
    }
    return "";
};

const getEnvConfig = () => ({
    databaseUrl: process.env.DATABASE_URL || process.env.DATEBASE_URL,
});

let dbPool = null;
let tableReady = null;

const ensureDb = () => {
    const { databaseUrl } = getEnvConfig();
    const missing = [];
    if (!databaseUrl) missing.push("DATABASE_URL");
    if (missing.length) {
        throw new Error(`Missing environment variables: ${missing.join(", ")}`);
    }

    if (!dbPool) {
        dbPool = mysql.createPool(databaseUrl);
        tableReady = ensureTable();
    }
};

const ensureTable = async () => {
    try {
        await dbPool.execute(`CREATE TABLE IF NOT EXISTS agent_memory (
            id INT AUTO_INCREMENT PRIMARY KEY,
            topic VARCHAR(100),
            information TEXT
        );`);
    } catch (error) {
        console.error("[DB Error] Failed to ensure agent_memory table", error.message);
        throw error;
    }
};

const agentTools = [{
    functionDeclarations: [
        {
            name: "search_memory",
            description: "Search the database for facts, past user interactions, or system rules.",
            parameters: {
                type: "OBJECT",
                properties: {
                    searchKeyword: { type: "STRING", description: "The main topic to look up." },
                },
                required: ["searchKeyword"],
            },
        },
        {
            name: "save_memory",
            description: "Save a new fact, rule, or preference to the database so you can remember it later.",
            parameters: {
                type: "OBJECT",
                properties: {
                    topic: { type: "STRING", description: "A short category (e.g., 'server rule', 'lore')" },
                    information: { type: "STRING", description: "The complete fact to remember." },
                },
                required: ["topic", "information"],
            },
        },
    ],
}];

const toolExecutors = {
    search_memory: async (args) => {
        const keyword = normalizeText(args?.searchKeyword);
        if (!keyword) return { result: "No keyword provided." };
        console.log(`[DB Search] Looking up: "${keyword}"`);
        try {
            await tableReady;
            const [rows] = await dbPool.execute(
                "SELECT information FROM agent_memory WHERE topic LIKE ? LIMIT 3",
                [`%${keyword}%`]
            );
            return rows.length === 0 ? { result: "No memory found." } : { result: rows };
        } catch (error) {
            console.error("[DB Error]", error.message);
            return { error: "Database query failed." };
        }
    },

    save_memory: async (args) => {
        const topic = normalizeText(args?.topic);
        const information = normalizeText(args?.information);
        if (!topic || !information) return { error: "Both topic and information are required." };
        console.log(`[DB Insert] Learning: "${topic}" -> "${information}"`);
        try {
            await tableReady;
            const [result] = await dbPool.execute(
                "INSERT INTO agent_memory (topic, information) VALUES (?, ?)",
                [topic, information]
            );
            return { status: "success", message: `Memory saved. ID: ${result.insertId}` };
        } catch (error) {
            console.error("[DB Error]", error.message);
            return { error: "Failed to write memory." };
        }
    },
};

const pruneHistory = (chatHistory, keepLast = 4) => {
    if (chatHistory.length > keepLast) {
        return [chatHistory[0], ...chatHistory.slice(-keepLast)];
    }
    return chatHistory;
};

const runToolWorker = async (prompt) => {
    try {
        ensureDb();
        await tableReady;
    } catch (error) {
        return `[DB Error] ${error?.message || "Database is not configured (set DATABASE_URL)."}`;
    }

    let chatHistory = [{ role: "user", parts: [{ text: prompt }] }];
    let iterations = 0;

    while (iterations < MAX_AGENT_ITERATIONS) {
        iterations += 1;
        chatHistory = pruneHistory(chatHistory);

        const response = await getAiClient().generateContent({
            model: GEMINI_MODEL,
            contents: chatHistory,
            tools: agentTools,
            generationConfig: { maxOutputTokens: 200 },
            systemInstruction: "You are an agent with a permanent database memory. Use 'search_memory' to answer questions about the past. Use 'save_memory' when the user tells you a new rule or fact to remember. Be concise.",
        });

        const call = Array.isArray(response?.functionCalls) ? response.functionCalls[0] : response?.functionCalls?.[0];

        if (call) {
            const executor = toolExecutors[call.name];
            if (!executor) break;

            const result = await executor(call.args);

            chatHistory.push({ role: "model", parts: [{ functionCall: call }] });
            chatHistory.push({
                role: "user",
                parts: [{ functionResponse: { name: call.name, response: result, id: call.id } }],
            });
        } else {
            if (response?.text) return response.text;
            const candidate = response?.candidates?.[0];
            if (candidate?.content?.parts?.[0]?.text) return candidate.content.parts[0].text;
            return "";
        }
    }
    return "Task aborted: Tool loop limit reached.";
};

const runReasoningWorker = async (prompt) => {
    const response = await getAiClient().generateContent({
        model: GEMINI_MODEL,
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: { maxOutputTokens: 400 },
    });
    if (response?.text) return response.text;
    const candidate = response?.candidates?.[0];
    if (candidate?.content?.parts?.[0]?.text) return candidate.content.parts[0].text;
    return "";
};

const askAgent = async (userPrompt) => {
    console.log(`\n[User] ${userPrompt}`);

    if (userPrompt.startsWith("!ping")) return "Pong! System is online.";
    if (userPrompt.toLowerCase() === "hello") return "Hey! Ready to work.";
    // Keep default cheap path; tools only if explicitly requested via prefix
    if (userPrompt.startsWith("!tool ")) {
        return runToolWorker(userPrompt.slice("!tool ".length));
    }
    return runReasoningWorker(userPrompt);
};

const metadataPayload = () => ({
    ok: true,
    endpoint: "/api/chat",
    provider: "nub-agent by ambitiousnoob (Gemini backend)",
    models: {
        router: ROUTER_MODEL,
        reasoning: GEMINI_MODEL,
    },
    default_execution_mode: "completion",
    agentic: false,
    tools: ["search_memory", "save_memory"],
    stream: "SSE (final chunk only)",
    database: {
        table: "agent_memory",
        fields: ["id", "topic", "information"],
        urlEnv: "DATABASE_URL",
    },
    example: {
        prompt: "Remember that the admin email is admin@example.com",
        stream: false,
    },
});

module.exports = async (req, res) => {
    writeCorsHeaders(res);

    if (req.method === "OPTIONS") {
        res.status(204).end();
        return;
    }

    if (req.method === "HEAD") {
        res.status(200);
        res.setHeader("Content-Type", "application/json; charset=utf-8");
        res.end();
        return;
    }

    if (req.method === "GET") {
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

    const prompt = extractPrompt(body);
    if (!prompt) {
        sendJson(res, 400, { error: "Provide a prompt string or a non-empty messages array." });
        return;
    }

    // Ensure API keys exist before invoking Gemini; allow per-request override
    let hasKey = Boolean((process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || "").trim());
    if (!hasKey && body?.apiKey) {
        process.env.GEMINI_API_KEY = String(body.apiKey).trim();
        process.env.GEMINI_API_KEYS = process.env.GEMINI_API_KEY;
        resetAiClient();
        hasKey = true;
    }
    if (!hasKey) {
        sendJson(res, 503, { error: "GEMINI_API_KEYS (or GEMINI_API_KEY) is not configured on the server. Provide apiKey in request body or set env." });
        return;
    }

    const HARD_TIMEOUT_MS = 60000;
    try {
        const reply = await Promise.race([
            askAgent(prompt),
            new Promise((_, reject) => setTimeout(() => reject(new Error(`Timed out after ${HARD_TIMEOUT_MS}ms`)), HARD_TIMEOUT_MS)),
        ]);
        if (wantsStream(body)) {
            res.status(200);
            res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
            res.setHeader("Cache-Control", "no-cache, no-transform");
            res.setHeader("Connection", "keep-alive");
            res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: reply } }] })}\n\n`);
            res.write("data: [DONE]\n\n");
            res.end();
            return;
        }

        sendJson(res, 200, {
            ok: true,
            model: GEMINI_MODEL,
            output_text: reply,
            finish_reason: "stop",
            agentic: false,
        });
    } catch (error) {
        console.error("[Gemini Error]", error);
        sendJson(res, 500, { error: error?.message || "Chat request failed." });
    }
};
