const path = require("node:path");
const dotenv = require("dotenv");
dotenv.config();
dotenv.config({ path: path.join(__dirname, "..", ".env") });

const { readBody } = require("../lib/web");
const { runLiteHostChat } = require("../lib/litehost-chat");

const DEFAULT_FB_GRAPH_API = "https://graph.facebook.com/v21.0";
const DEFAULT_MESSENGER_SYSTEM_PROMPT = [
    "You are LiteHost Messenger Bot.",
    "Reply with concise, useful text suitable for Facebook Messenger.",
    "Prefer short paragraphs and avoid markdown tables.",
    "If the user asks for a long answer, keep it readable and practical.",
].join(" ");

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

const sendText = (res, status, value) => {
    res.status(status);
    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    res.end(String(value));
};

const getVerifyToken = () => (
    String(process.env.MESSENGER_VERIFY_TOKEN || process.env.VERIFY_TOKEN || "").trim()
);

const getPageAccessToken = () => String(process.env.PAGE_ACCESS_TOKEN || "").trim();

const getGraphApiBase = () => (
    String(process.env.FB_GRAPH_API || DEFAULT_FB_GRAPH_API).trim().replace(/\/+$/, "")
);

const getMessengerSystemPrompt = () => (
    String(process.env.MESSENGER_SYSTEM_PROMPT || DEFAULT_MESSENGER_SYSTEM_PROMPT).trim()
);

const getMaxMessageChars = () => {
    const value = Number(process.env.MESSENGER_MAX_MESSAGE_CHARS) || 640;
    return Math.max(160, Math.min(1800, value));
};

const normalizeReplyText = (value) => {
    const text = String(value || "").replace(/\r\n/g, "\n").trim();
    return text || "I'm here, but I couldn't generate a useful reply just now. Please try again.";
};

const splitLongLine = (line, maxChars) => {
    const chunks = [];
    let rest = String(line || "").trim();

    while (rest.length > maxChars) {
        let splitAt = rest.lastIndexOf(" ", maxChars);
        if (splitAt < Math.floor(maxChars * 0.5)) splitAt = maxChars;
        chunks.push(rest.slice(0, splitAt).trim());
        rest = rest.slice(splitAt).trim();
    }

    if (rest) chunks.push(rest);
    return chunks;
};

const splitMessengerText = (value, maxChars = getMaxMessageChars()) => {
    const text = normalizeReplyText(value);
    const chunks = [];
    const paragraphs = text.split(/\n{2,}/).map((paragraph) => paragraph.trim()).filter(Boolean);

    for (const paragraph of paragraphs) {
        if (paragraph.length <= maxChars) {
            chunks.push(paragraph);
            continue;
        }

        const lines = paragraph.split("\n").map((line) => line.trim()).filter(Boolean);
        for (const line of lines) {
            chunks.push(...splitLongLine(line, maxChars));
        }
    }

    return chunks.length ? chunks : [text];
};

const extractGraphError = async (response) => {
    const raw = await response.text();
    try {
        const parsed = JSON.parse(raw);
        if (parsed?.error?.message) return parsed.error.message;
    } catch {
        return raw;
    }
    return raw;
};

const sendMessengerMessage = async (recipientId, text) => {
    const accessToken = getPageAccessToken();
    if (!accessToken) {
        const error = new Error("PAGE_ACCESS_TOKEN is not configured on the server.");
        error.status = 503;
        throw error;
    }

    const response = await fetch(`${getGraphApiBase()}/me/messages?access_token=${encodeURIComponent(accessToken)}`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            recipient: { id: recipientId },
            message: { text },
            messaging_type: "RESPONSE",
        }),
    });

    if (!response.ok) {
        const error = new Error(`Facebook Graph API error: ${response.status} ${await extractGraphError(response)}`);
        error.status = response.status;
        throw error;
    }

    return response.json();
};

const sendMessengerReply = async (recipientId, text) => {
    const parts = splitMessengerText(text);
    const receipts = [];

    for (const part of parts) {
        receipts.push(await sendMessengerMessage(recipientId, part));
    }

    return receipts;
};

const getIncomingText = (event) => {
    const messageText = String(event?.message?.text || "").trim();
    if (messageText) return messageText;

    const postbackTitle = String(event?.postback?.title || "").trim();
    if (postbackTitle) return postbackTitle;

    const postbackPayload = String(event?.postback?.payload || "").trim();
    return postbackPayload;
};

const isReplyableEvent = (event) => {
    if (!event || typeof event !== "object") return false;
    if (!event.sender?.id) return false;
    if (event.message?.is_echo) return false;
    if (!event.message && !event.postback) return false;
    return true;
};

const buildMessengerChatBody = (incomingText) => ({
    messages: [
        { role: "system", content: getMessengerSystemPrompt() },
        { role: "user", content: incomingText },
    ],
    stream: false,
});

const handleMessagingEvent = async (event) => {
    if (!isReplyableEvent(event)) return;

    const senderId = String(event.sender.id).trim();

    try {
        const incomingText = getIncomingText(event);

        if (!incomingText) {
            await sendMessengerReply(
                senderId,
                "I can reply to text messages right now. Send a question or prompt and I'll answer there.",
            );
            return;
        }

        const { reply } = await runLiteHostChat(buildMessengerChatBody(incomingText));
        await sendMessengerReply(senderId, reply.content);
    } catch (error) {
        console.error("[Messenger Bot Error]", error);
        try {
            await sendMessengerReply(
                senderId,
                "LiteHost is temporarily unavailable. Please try again in a moment.",
            );
        } catch (fallbackError) {
            console.error("[Messenger Bot Fallback Error]", fallbackError);
        }
    }
};

module.exports = async (req, res) => {
    writeCorsHeaders(res);

    if (req.method === "OPTIONS") {
        res.status(204).end();
        return;
    }

    if (req.method === "GET") {
        const mode = String(req.query?.["hub.mode"] || req.query?.hub_mode || "").trim();
        const verifyToken = String(req.query?.["hub.verify_token"] || req.query?.hub_verify_token || "").trim();
        const challenge = String(req.query?.["hub.challenge"] || req.query?.hub_challenge || "").trim();

        if (!mode && !verifyToken && !challenge) {
            sendJson(res, 200, {
                ok: true,
                endpoint: "/api/messenger",
                verify_token_env: process.env.MESSENGER_VERIFY_TOKEN ? "MESSENGER_VERIFY_TOKEN" : "VERIFY_TOKEN",
                graph_api_env: "FB_GRAPH_API",
                page_access_token_env: "PAGE_ACCESS_TOKEN",
            });
            return;
        }

        if (mode === "subscribe" && verifyToken && challenge && verifyToken === getVerifyToken()) {
            sendText(res, 200, challenge);
            return;
        }

        sendJson(res, 403, { error: "Messenger webhook verification failed." });
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

    if (body?.object !== "page" || !Array.isArray(body.entry)) {
        sendJson(res, 400, { error: "Unsupported Messenger webhook payload." });
        return;
    }

    for (const entry of body.entry) {
        const events = Array.isArray(entry?.messaging) ? entry.messaging : [];
        for (const event of events) {
            await handleMessagingEvent(event);
        }
    }

    sendText(res, 200, "EVENT_RECEIVED");
};
