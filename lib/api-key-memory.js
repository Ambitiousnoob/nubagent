const crypto = require("node:crypto");
const { getPool } = require("./db");

const API_KEY_MEMORY_TABLE = "api_key_memory";
const MEMORY_CONTENT_CHAR_LIMIT = 1200;
const MEMORY_CANDIDATE_LIMIT = 80;
const MEMORY_RESULT_LIMIT = 4;
const TERM_RE = /[a-z0-9_/-]{3,}/g;
const STOP_WORDS = new Set([
    "the", "and", "for", "with", "that", "this", "from", "have", "your", "into", "about", "there",
    "their", "would", "could", "should", "after", "before", "where", "when", "what", "which", "while",
    "then", "than", "them", "they", "were", "been", "being", "also", "just", "over", "under", "through",
    "user", "assistant", "tool", "result", "results", "using", "used", "http", "https", "www", "com",
    "org", "net", "api", "json", "html", "text", "data", "file", "files", "reply", "said", "tell",
]);

let initPromise = null;

const normalizeText = (value) => (
    String(value ?? "")
        .replace(/\u0000/g, "")
        .replace(/\r\n?/g, "\n")
        .trim()
);

const truncateText = (value, max = MEMORY_CONTENT_CHAR_LIMIT) => {
    const text = String(value ?? "");
    if (text.length <= max) return text;
    const suffix = ` ...[${text.length - max} chars omitted]`;
    if (max <= suffix.length + 8) return text.slice(0, max);
    return `${text.slice(0, max - suffix.length)}${suffix}`;
};

const getMessageText = (content) => {
    if (typeof content === "string") return content;
    if (!Array.isArray(content)) return "";
    return content
        .filter((part) => part?.type === "text" && typeof part.text === "string")
        .map((part) => part.text)
        .join("\n\n");
};

const stripAssistantArtifacts = (value) => {
    const text = normalizeText(value);
    if (!text) return "";
    return text
        .replace(/<thought>[\s\S]*?<\/thought>/gi, "")
        .replace(/<tool_call>[\s\S]*?<\/tool_call>/gi, "")
        .replace(/\n{3,}/g, "\n\n")
        .trim();
};

const normalizeMemoryEntry = (role, content) => {
    const normalizedRole = role === "assistant" ? "assistant" : "user";
    const raw = normalizedRole === "assistant"
        ? stripAssistantArtifacts(content)
        : normalizeText(content);
    if (!raw) return null;

    const normalizedContent = truncateText(raw, MEMORY_CONTENT_CHAR_LIMIT);
    return {
        role: normalizedRole,
        content: normalizedContent,
        contentHash: crypto.createHash("sha256").update(`${normalizedRole}\n${normalizedContent}`).digest("hex"),
    };
};

const extractTerms = (value) => (
    Array.from(new Set(
        (normalizeText(value).toLowerCase().match(TERM_RE) || [])
            .filter((term) => !STOP_WORDS.has(term))
            .slice(0, 80),
    ))
);

const buildTermSet = (value) => new Set(extractTerms(value));

const scoreEntry = (queryTerms, entryText, updatedAt) => {
    if (!queryTerms.size) return 0;
    const entryTerms = buildTermSet(entryText);
    if (!entryTerms.size) return 0;

    let overlap = 0;
    for (const term of queryTerms) {
        if (entryTerms.has(term)) overlap += 1;
    }
    if (!overlap) return 0;

    const ageHours = Math.max(0, (Date.now() - new Date(updatedAt || Date.now()).getTime()) / 36e5);
    const recencyBoost = 1 / (1 + ageHours / 24);
    return overlap + recencyBoost;
};

const ensureApiKeyMemoryTable = async () => {
    if (initPromise) return initPromise;

    initPromise = (async () => {
        const db = getPool();
        await db.query(`
            CREATE TABLE IF NOT EXISTS ${API_KEY_MEMORY_TABLE} (
                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                api_key_hash CHAR(68) NOT NULL,
                content_hash CHAR(64) NOT NULL,
                role VARCHAR(16) NOT NULL,
                content TEXT NOT NULL,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY uniq_api_key_memory (api_key_hash, content_hash),
                KEY idx_api_key_memory_lookup (api_key_hash, updated_at)
            ) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin
        `);
    })().catch((error) => {
        initPromise = null;
        throw error;
    });

    return initPromise;
};

const collectMemoryEntries = (messages = [], assistantReply = "") => {
    const output = [];
    for (const message of Array.isArray(messages) ? messages : []) {
        if (!message || message.role === "system") continue;
        const entry = normalizeMemoryEntry(message.role, getMessageText(message.content));
        if (entry) output.push(entry);
    }

    const assistantEntry = normalizeMemoryEntry("assistant", assistantReply);
    if (assistantEntry) output.push(assistantEntry);

    const deduped = [];
    const seen = new Set();
    for (const entry of output) {
        if (seen.has(entry.contentHash)) continue;
        seen.add(entry.contentHash);
        deduped.push(entry);
    }
    return deduped.slice(-6);
};

const saveApiKeyMemoryEntries = async (apiKeyHash, messages = [], assistantReply = "") => {
    const scopedHash = normalizeText(apiKeyHash);
    if (!scopedHash) return 0;

    const entries = collectMemoryEntries(messages, assistantReply);
    if (!entries.length) return 0;

    await ensureApiKeyMemoryTable();
    const db = getPool();
    for (const entry of entries) {
        await db.query(
            `
                INSERT INTO ${API_KEY_MEMORY_TABLE} (api_key_hash, content_hash, role, content)
                VALUES (?, ?, ?, ?)
                ON DUPLICATE KEY UPDATE
                    updated_at = CURRENT_TIMESTAMP,
                    content = VALUES(content),
                    role = VALUES(role)
            `,
            [scopedHash, entry.contentHash, entry.role, entry.content],
        );
    }
    return entries.length;
};

const searchApiKeyMemory = async (apiKeyHash, query, { limit = MEMORY_RESULT_LIMIT, candidateLimit = MEMORY_CANDIDATE_LIMIT } = {}) => {
    const scopedHash = normalizeText(apiKeyHash);
    const normalizedQuery = normalizeText(query);
    if (!scopedHash || !normalizedQuery) return [];

    await ensureApiKeyMemoryTable();
    const db = getPool();
    const [rows] = await db.query(
        `
            SELECT role, content, updated_at
            FROM ${API_KEY_MEMORY_TABLE}
            WHERE api_key_hash = ?
            ORDER BY updated_at DESC
            LIMIT ?
        `,
        [scopedHash, Math.max(limit, candidateLimit)],
    );

    const queryTerms = buildTermSet(normalizedQuery);
    const ranked = (Array.isArray(rows) ? rows : [])
        .map((row) => ({
            role: row.role === "assistant" ? "assistant" : "user",
            content: normalizeText(row.content),
            updatedAt: row.updated_at ? new Date(row.updated_at).toISOString() : null,
            score: scoreEntry(queryTerms, row.content, row.updated_at),
        }))
        .filter((row) => row.content && row.score > 0)
        .sort((left, right) => (
            (right.score - left.score) ||
            (new Date(right.updatedAt || 0).getTime() - new Date(left.updatedAt || 0).getTime())
        ));

    const seen = new Set();
    const results = [];
    for (const row of ranked) {
        const key = `${row.role}:${row.content}`;
        if (seen.has(key)) continue;
        seen.add(key);
        results.push(row);
        if (results.length >= limit) break;
    }
    return results;
};

const formatApiKeyMemoryContext = (entries = []) => {
    const items = (Array.isArray(entries) ? entries : []).filter((entry) => entry?.content);
    if (!items.length) return "";
    const lines = ["Relevant API-key memory:"];
    items.forEach((entry, index) => {
        const label = entry.role === "assistant" ? "Assistant memory" : "User memory";
        lines.push(`${index + 1}. [${label}] ${entry.content}`);
    });
    lines.push("Use this only when it is relevant to the current request.");
    return lines.join("\n");
};

module.exports = {
    API_KEY_MEMORY_TABLE,
    ensureApiKeyMemoryTable,
    saveApiKeyMemoryEntries,
    searchApiKeyMemory,
    formatApiKeyMemoryContext,
};
