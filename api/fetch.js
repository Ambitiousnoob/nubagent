const { readBody } = require("../lib/web");
const { handler: webFetchHandler } = require("./tools/web_fetch");

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
            endpoint: "/api/fetch",
            tool: "web_fetch",
        });
        return;
    }

    if (req.method !== "POST") {
        sendJson(res, 405, { error: "Method not allowed" });
        return;
    }

    try {
        const body = await readBody(req);
        const url = String(body?.url || "").trim();
        if (!url) {
            sendJson(res, 400, { error: "Provide a url." });
            return;
        }

        const raw = await webFetchHandler({
            url,
            format: body?.format,
            max_chars: body?.max_chars ?? body?.maxChars,
        });

        if (typeof raw === "string" && raw.startsWith("Error:")) {
            sendJson(res, 502, { error: raw });
            return;
        }

        sendJson(res, 200, {
            ok: true,
            url,
            content: String(raw || ""),
        });
    } catch (error) {
        sendJson(res, 500, { error: error?.message || "Fetch request failed." });
    }
};
