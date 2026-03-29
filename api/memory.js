const { readBody } = require("../lib/web");
const { getScopedStateKeyFromRequest } = require("../lib/state-scope");
const {
    ensureApiKeyMemoryTable,
    saveApiKeyMemoryEntries,
    searchApiKeyMemoryDetailed,
    formatApiKeyMemoryContext,
} = require("../lib/api-key-memory");

const writeCorsHeaders = (res) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key");
};

const sendJson = (res, status, payload) => {
    res.status(status);
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(JSON.stringify(payload));
};

const normalizeInsertEntries = (body = {}) => {
    if (Array.isArray(body.entries) && body.entries.length) {
        return body.entries
            .map((entry) => ({
                role: entry?.role === "assistant" ? "assistant" : "user",
                content: String(entry?.content || "").trim(),
            }))
            .filter((entry) => entry.content);
    }

    const content = String(body.content || "").trim();
    if (!content) return [];

    return [{
        role: body.role === "assistant" ? "assistant" : "user",
        content,
    }];
};

module.exports = async (req, res) => {
    writeCorsHeaders(res);

    if (req.method === "OPTIONS") {
        res.status(204).end();
        return;
    }

    const scope = getScopedStateKeyFromRequest(req);
    if (scope.scope !== "api_key" || !scope.stateKey) {
        sendJson(res, 400, {
            error: "Send X-API-Key or Authorization: Bearer <key> to access API-key memory.",
        });
        return;
    }

    if (req.method === "HEAD" || req.method === "GET") {
        try {
            await ensureApiKeyMemoryTable();
            sendJson(res, 200, {
                ok: true,
                endpoint: "/api/memory",
                scope: "api_key",
                actions: ["insert", "search"],
                search: {
                    provider: process.env.CEREBRAS_API_KEY ? "cerebras+local" : "local",
                    helper_model: process.env.CEREBRAS_MEMORY_MODEL || "qwen-3-235b-a22b-instruct-2507",
                },
                example: {
                    action: "search",
                    query: "what is my favorite color",
                    limit: 4,
                },
            });
        } catch (error) {
            console.error("[Memory API Error]", error);
            sendJson(res, 503, { error: error?.message || "Memory storage unavailable." });
        }
        return;
    }

    if (req.method !== "POST") {
        sendJson(res, 405, { error: "Method not allowed" });
        return;
    }

    let body;
    try {
        body = await readBody(req);
    } catch {
        sendJson(res, 400, { error: "Invalid JSON body" });
        return;
    }

    const action = String(body?.action || "").trim().toLowerCase();
    if (!action) {
        sendJson(res, 400, { error: "Provide an action: insert or search." });
        return;
    }

    try {
        if (action === "insert") {
            const entries = normalizeInsertEntries(body);
            if (!entries.length) {
                sendJson(res, 400, { error: "Provide content or an entries array." });
                return;
            }

            const assistantReply = String(body.assistant_reply || body.assistantReply || "").trim();
            const inserted = await saveApiKeyMemoryEntries(scope.stateKey, entries, assistantReply);
            sendJson(res, 200, {
                ok: true,
                action,
                scope: "api_key",
                inserted,
                stored_entries: entries.length + (assistantReply ? 1 : 0),
            });
            return;
        }

        if (action === "search") {
            const query = String(body.query || "").trim();
            if (!query) {
                sendJson(res, 400, { error: "Provide a query string." });
                return;
            }

            const search = await searchApiKeyMemoryDetailed(scope.stateKey, query, {
                limit: body.limit,
                candidateLimit: body.candidate_limit ?? body.candidateLimit,
                rerankLimit: body.rerank_limit ?? body.rerankLimit,
            });

            sendJson(res, 200, {
                ok: true,
                action,
                scope: "api_key",
                query,
                search: search.meta,
                results: search.results,
                context: body.include_context ? formatApiKeyMemoryContext(search.results) : undefined,
            });
            return;
        }

        sendJson(res, 400, { error: "Unknown action. Use insert or search." });
    } catch (error) {
        console.error("[Memory API Error]", error);
        sendJson(res, 503, { error: error?.message || "Memory request failed." });
    }
};
