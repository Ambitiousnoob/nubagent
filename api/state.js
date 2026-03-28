const { readBody } = require("../lib/web");
const { loadAppState, saveAppState } = require("../lib/db");

const STATE_KEY = "default";

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
        try {
            const result = await loadAppState(STATE_KEY);
            sendJson(res, 200, {
                ok: true,
                state: result?.state || null,
                updatedAt: result?.updatedAt || null,
            });
        } catch (error) {
            console.error("[State API Error]", error);
            sendJson(res, 503, { error: error?.message || "State storage unavailable." });
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

    const state = body?.state;
    if (!state || typeof state !== "object" || Array.isArray(state)) {
        sendJson(res, 400, { error: "Provide a state object." });
        return;
    }

    const payloadSize = Buffer.byteLength(JSON.stringify(state), "utf8");
    if (payloadSize > 2_500_000) {
        sendJson(res, 413, { error: "State payload too large." });
        return;
    }

    try {
        const saved = await saveAppState(STATE_KEY, state);
        sendJson(res, 200, {
            ok: true,
            updatedAt: saved?.updatedAt || new Date().toISOString(),
        });
    } catch (error) {
        console.error("[State API Error]", error);
        sendJson(res, 503, { error: error?.message || "State storage unavailable." });
    }
};
