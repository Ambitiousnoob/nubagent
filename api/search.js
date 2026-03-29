const { readBody } = require("../lib/web");
const { handler: webSearchHandler } = require("./tools/web_search");

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

    if (req.method === "GET" || req.method === "HEAD") {
        sendJson(res, 200, {
            ok: true,
            endpoint: "/api/search",
            tool: "web_search",
        });
        return;
    }

    if (req.method !== "POST") {
        sendJson(res, 405, { error: "Method not allowed" });
        return;
    }

    try {
        const body = await readBody(req);
        const query = String(body?.query || "").trim();
        if (!query) {
            sendJson(res, 400, { error: "Provide a query string." });
            return;
        }

        const raw = await webSearchHandler({ query });
        if (typeof raw === "string" && raw.startsWith("Error:")) {
            sendJson(res, 502, { error: raw });
            return;
        }

        let results = [];
        try {
            results = JSON.parse(raw);
        } catch {
            results = [];
        }

        sendJson(res, 200, {
            ok: true,
            query,
            results: Array.isArray(results) ? results : [],
        });
    } catch (error) {
        sendJson(res, 500, { error: error?.message || "Search request failed." });
    }
};
