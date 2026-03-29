const { readBody } = require("../lib/web");
const {
    metadataPayload,
    wantsStream,
    runLiteHostChat,
    createChatResponsePayload,
} = require("../lib/litehost-chat");
const {
    loadScopedChatMemory,
    saveScopedChatMemory,
    mergeChatMemory,
    shouldUseScopedChatMemory,
    buildMessagesWithScopedMemory,
} = require("../lib/chat-memory");
const {
    saveApiKeyMemoryEntries,
    searchApiKeyMemoryDetailed,
    formatApiKeyMemoryContext,
} = require("../lib/api-key-memory");

const writeCorsHeaders = (res) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key, X-State-Key");
};

const sendJson = (res, status, payload) => {
    res.status(status);
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(JSON.stringify(payload));
};

const getMessageText = (content) => {
    if (typeof content === "string") return content;
    if (!Array.isArray(content)) return "";
    return content
        .filter((part) => part?.type === "text" && typeof part.text === "string")
        .map((part) => part.text)
        .join("\n\n");
};

const getLatestUserText = (messages = []) => {
    const latestUserMessage = [...(Array.isArray(messages) ? messages : [])]
        .reverse()
        .find((message) => message?.role === "user");
    return getMessageText(latestUserMessage?.content || "").trim();
};

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

    try {
        const scopedMemory = await loadScopedChatMemory(req).catch(() => null);
        const latestUserText = getLatestUserText(body.messages);
        const apiKeyMemorySearch = scopedMemory?.scope === "api_key" && latestUserText
            ? await searchApiKeyMemoryDetailed(scopedMemory.stateKey, latestUserText).catch(() => ({ results: [], meta: null }))
            : { results: [], meta: null };
        const apiKeyMemoryHits = apiKeyMemorySearch.results || [];
        const apiKeyMemoryContext = formatApiKeyMemoryContext(apiKeyMemoryHits);
        const shouldUseMemory = Boolean(scopedMemory?.dbKey) && shouldUseScopedChatMemory(body);
        const requestBody = shouldUseMemory
            ? {
                ...body,
                messages: buildMessagesWithScopedMemory(body.messages, scopedMemory.memory, {
                    retrievedContext: apiKeyMemoryContext,
                }),
            }
            : body;
        const wantsSse = wantsStream(body) && body.use_tools === false;

        if (wantsSse) {
            res.status(200);
            res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
            res.setHeader("Cache-Control", "no-cache, no-transform");
            res.setHeader("Connection", "keep-alive");

            const streamCallback = (chunk) => {
                res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: chunk } }] })}\n\n`);
            };

            const result = await runLiteHostChat(requestBody, streamCallback);
            if (scopedMemory?.scope === "api_key" && body.save_persistent_memory !== false) {
                await saveApiKeyMemoryEntries(scopedMemory.stateKey, body.messages, result?.reply?.content || "").catch(() => {});
            }
            if (scopedMemory?.dbKey && body.save_persistent_memory !== false) {
                const nextMemory = mergeChatMemory(scopedMemory.memory, body.messages, result?.reply?.content || "");
                await saveScopedChatMemory(scopedMemory, nextMemory).catch(() => {});
            }
            res.write("data: [DONE]\n\n");
            res.end();
        } else {
            const result = await runLiteHostChat(requestBody);
            if (scopedMemory?.scope === "api_key" && body.save_persistent_memory !== false) {
                await saveApiKeyMemoryEntries(scopedMemory.stateKey, body.messages, result?.reply?.content || "").catch(() => {});
            }
            let memoryMeta = null;
            if (scopedMemory?.dbKey && body.save_persistent_memory !== false) {
                const nextMemory = mergeChatMemory(scopedMemory.memory, body.messages, result?.reply?.content || "");
                const savedMemory = await saveScopedChatMemory(scopedMemory, nextMemory).catch(() => null);
                const activeMemory = savedMemory?.memory || nextMemory;
                memoryMeta = {
                    scope: scopedMemory.scope,
                    injected: shouldUseMemory,
                    summary: Boolean(activeMemory?.summary),
                    recent_messages: Array.isArray(activeMemory?.recentMessages) ? activeMemory.recentMessages.length : 0,
                    related_entries: apiKeyMemoryHits.length,
                    search_strategy: apiKeyMemorySearch.meta?.strategy || null,
                    search_provider: apiKeyMemorySearch.meta?.provider || null,
                    search_model: apiKeyMemorySearch.meta?.model || null,
                };
            } else if (scopedMemory?.dbKey) {
                memoryMeta = {
                    scope: scopedMemory.scope,
                    injected: shouldUseMemory,
                    summary: Boolean(scopedMemory.memory?.summary),
                    recent_messages: Array.isArray(scopedMemory.memory?.recentMessages) ? scopedMemory.memory.recentMessages.length : 0,
                    related_entries: apiKeyMemoryHits.length,
                    search_strategy: apiKeyMemorySearch.meta?.strategy || null,
                    search_provider: apiKeyMemorySearch.meta?.provider || null,
                    search_model: apiKeyMemorySearch.meta?.model || null,
                };
            }

            sendJson(res, 200, {
                ...createChatResponsePayload(result.body, result.reply),
                ...(memoryMeta ? { memory: memoryMeta } : {}),
            });
        }
    } catch (error) {
        console.error("[nub-agent API Error]", error);
        sendJson(res, Number(error?.status) || 500, { error: error?.message || "Chat request failed." });
    }
};
