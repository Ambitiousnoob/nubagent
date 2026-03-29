const crypto = require("node:crypto");

const STATE_KEY_HEADER = "x-state-key";
const API_KEY_HEADER = "x-api-key";
const AUTHORIZATION_HEADER = "authorization";
const STATE_KEY_RE = /^[A-Za-z0-9:_-]{8,191}$/;

const getHeaderValue = (req, name) => {
    const value = req?.headers?.[name];
    return Array.isArray(value) ? value[0] : value;
};

const normalizeStateKey = (value) => {
    const key = String(value || "").trim();
    return STATE_KEY_RE.test(key) ? key : "";
};

const normalizeApiKey = (value) => {
    const key = String(value || "").trim();
    if (!key) return "";
    if (key.length < 8 || key.length > 1024) return "";
    return key;
};

const getAuthorizationApiKey = (req) => {
    const raw = String(getHeaderValue(req, AUTHORIZATION_HEADER) || "").trim();
    const match = raw.match(/^Bearer\s+(.+)$/i);
    return match ? normalizeApiKey(match[1]) : "";
};

const getApiKeyFromRequest = (req) => (
    normalizeApiKey(getHeaderValue(req, API_KEY_HEADER)) || getAuthorizationApiKey(req)
);

const buildApiMemoryStateKey = (apiKey) => (
    `api:${crypto.createHash("sha256").update(normalizeApiKey(apiKey)).digest("hex")}`
);

const getScopedStateKeyFromRequest = (req) => {
    const apiKey = getApiKeyFromRequest(req);
    if (apiKey) {
        return {
            stateKey: buildApiMemoryStateKey(apiKey),
            scope: "api_key",
        };
    }

    const stateKey = normalizeStateKey(getHeaderValue(req, STATE_KEY_HEADER));
    if (stateKey) {
        return {
            stateKey,
            scope: "state_key",
        };
    }

    return {
        stateKey: "",
        scope: "",
    };
};

module.exports = {
    STATE_KEY_HEADER,
    API_KEY_HEADER,
    STATE_KEY_RE,
    normalizeStateKey,
    normalizeApiKey,
    buildApiMemoryStateKey,
    getScopedStateKeyFromRequest,
};
