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
        const shouldUseMemory = Boolean(scopedMemory?.dbKey) && shouldUseScopedChatMemory(body);
        const requestBody = shouldUseMemory
            ? {
                ...body,
                messages: buildMessagesWithScopedMemory(body.messages, scopedMemory.memory),
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
            if (scopedMemory?.dbKey && body.save_persistent_memory !== false) {
                const nextMemory = mergeChatMemory(scopedMemory.memory, body.messages, result?.reply?.content || "");
                await saveScopedChatMemory(scopedMemory, nextMemory).catch(() => {});
            }
            res.write("data: [DONE]\n\n");
            res.end();
        } else {
            const result = await runLiteHostChat(requestBody);
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
                };
            } else if (scopedMemory?.dbKey) {
                memoryMeta = {
                    scope: scopedMemory.scope,
                    injected: shouldUseMemory,
                    summary: Boolean(scopedMemory.memory?.summary),
                    recent_messages: Array.isArray(scopedMemory.memory?.recentMessages) ? scopedMemory.memory.recentMessages.length : 0,
                };
            }

            sendJson(res, 200, {
                ...createChatResponsePayload(result.body, result.reply),
                ...(memoryMeta ? { memory: memoryMeta } : {}),
            });
        }
    } catch (error) {
        console.error("[Gemini API Error]", error);
        sendJson(res, Number(error?.status) || 500, { error: error?.message || "Chat request failed." });
    }
};
