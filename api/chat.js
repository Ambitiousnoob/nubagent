const { readBody } = require("../lib/web");
const {
    metadataPayload,
    wantsStream,
    runLiteHostChat,
    createChatResponsePayload,
} = require("../lib/litehost-chat");

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
        const wantsSse = wantsStream(body) && body.use_tools === false;

        if (wantsSse) {
            res.status(200);
            res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
            res.setHeader("Cache-Control", "no-cache, no-transform");
            res.setHeader("Connection", "keep-alive");

            const streamCallback = (chunk) => {
                res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: chunk } }] })}\n\n`);
            };

            await runLiteHostChat(body, streamCallback);
            res.write("data: [DONE]\n\n");
            res.end();
        } else {
            const result = await runLiteHostChat(body);
            sendJson(res, 200, createChatResponsePayload(result.body, result.reply));
        }
    } catch (error) {
        console.error("[Gemini API Error]", error);
        sendJson(res, Number(error?.status) || 500, { error: error?.message || "Chat request failed." });
    }
};
