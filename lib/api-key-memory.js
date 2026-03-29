const crypto = require("node:crypto");
const { getPool } = require("./db");

const API_KEY_MEMORY_TABLE = "api_key_memory";
const MEMORY_CONTENT_CHAR_LIMIT = 1200;
const MEMORY_CANDIDATE_LIMIT = 80;
const MEMORY_RESULT_LIMIT = 4;
const MEMORY_RERANK_CANDIDATE_LIMIT = 12;
const MEMORY_RERANK_TIMEOUT_MS = 6500;
const CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions";
const DEFAULT_CEREBRAS_MEMORY_MODEL = process.env.CEREBRAS_MEMORY_MODEL || "qwen-3-235b-a22b-instruct-2507";
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

const clampInteger = (value, min, max, fallback) => {
    const number = Number(value);
    if (!Number.isFinite(number)) return fallback;
    return Math.min(max, Math.max(min, Math.floor(number)));
};

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

const byUpdatedAtDesc = (left, right) => (
    new Date(right.updatedAt || 0).getTime() - new Date(left.updatedAt || 0).getTime()
);

const byScoreThenUpdatedAtDesc = (left, right) => (
    (right.score - left.score) || byUpdatedAtDesc(left, right)
);

const dedupeRows = (rows = [], limit = MEMORY_RESULT_LIMIT) => {
    const seen = new Set();
    const result = [];
    for (const row of Array.isArray(rows) ? rows : []) {
        const key = `${row.role}:${row.content}`;
        if (!row?.content || seen.has(key)) continue;
        seen.add(key);
        result.push(row);
        if (result.length >= limit) break;
    }
    return result;
};

const parseJsonObject = (value) => {
    if (value && typeof value === "object" && !Array.isArray(value)) return value;

    const text = normalizeText(value);
    if (!text) return null;

    const cleaned = text
        .replace(/^```(?:json)?\s*/i, "")
        .replace(/\s*```$/i, "")
        .trim();

    try {
        return JSON.parse(cleaned);
    } catch {
        const match = cleaned.match(/\{[\s\S]*\}/);
        if (!match) return null;
        try {
            return JSON.parse(match[0]);
        } catch {
            return null;
        }
    }
};

const buildNormalizedRows = (rows = [], queryTerms = new Set()) => (
    (Array.isArray(rows) ? rows : [])
        .map((row) => ({
            role: row.role === "assistant" ? "assistant" : "user",
            content: normalizeText(row.content),
            updatedAt: row.updated_at ? new Date(row.updated_at).toISOString() : null,
            score: scoreEntry(queryTerms, row.content, row.updated_at),
        }))
        .filter((row) => row.content)
);

const buildFallbackResults = (rows = [], queryTerms = new Set(), limit = MEMORY_RESULT_LIMIT) => {
    const normalizedLimit = clampInteger(limit, 1, 10, MEMORY_RESULT_LIMIT);
    const lexicalMatches = dedupeRows(
        [...rows]
            .filter((row) => row.score > 0)
            .sort(byScoreThenUpdatedAtDesc),
        normalizedLimit,
    );

    if (lexicalMatches.length) {
        return {
            results: lexicalMatches,
            strategy: "lexical",
        };
    }

    return {
        results: dedupeRows([...rows].sort(byUpdatedAtDesc), normalizedLimit),
        strategy: queryTerms.size ? "recent_fallback" : "recent",
    };
};

const buildRerankCandidates = (rows = [], queryTerms = new Set(), rerankLimit = MEMORY_RERANK_CANDIDATE_LIMIT) => {
    const normalizedLimit = clampInteger(rerankLimit, 1, 20, MEMORY_RERANK_CANDIDATE_LIMIT);
    const selected = [];
    const seen = new Set();
    const pushRow = (row) => {
        if (!row?.content) return;
        const key = `${row.role}:${row.content}`;
        if (seen.has(key)) return;
        seen.add(key);
        selected.push(row);
    };

    if (queryTerms.size) {
        for (const row of [...rows].sort(byScoreThenUpdatedAtDesc)) {
            if (row.score <= 0) continue;
            pushRow(row);
            if (selected.length >= normalizedLimit) return selected;
        }
    }

    for (const row of [...rows].sort(byUpdatedAtDesc)) {
        pushRow(row);
        if (selected.length >= normalizedLimit) break;
    }

    return selected;
};

const rerankWithCerebras = async (query, candidates = [], limit = MEMORY_RESULT_LIMIT) => {
    const apiKey = normalizeText(process.env.CEREBRAS_API_KEY);
    if (!apiKey || !Array.isArray(candidates) || !candidates.length) return null;

    const selectedLimit = clampInteger(limit, 1, 10, MEMORY_RESULT_LIMIT);
    const payloadCandidates = candidates.map((entry, index) => ({
        id: index + 1,
        role: entry.role,
        updatedAt: entry.updatedAt,
        content: truncateText(entry.content, 900),
    }));

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), MEMORY_RERANK_TIMEOUT_MS);

    try {
        const response = await fetch(CEREBRAS_API_URL, {
            method: "POST",
            signal: controller.signal,
            headers: {
                "Authorization": `Bearer ${apiKey}`,
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                model: DEFAULT_CEREBRAS_MEMORY_MODEL,
                temperature: 0,
                max_completion_tokens: 220,
                stream: false,
                response_format: {
                    type: "json_schema",
                    json_schema: {
                        name: "memory_selection",
                        strict: true,
                        schema: {
                            type: "object",
                            properties: {
                                selected_ids: {
                                    type: "array",
                                    items: { type: "integer" },
                                    maxItems: selectedLimit,
                                },
                                notes: {
                                    type: "string",
                                },
                            },
                            required: ["selected_ids", "notes"],
                            additionalProperties: false,
                        },
                    },
                },
                messages: [
                    {
                        role: "system",
                        content: [
                            "You are a retrieval coprocessor for nub-agent.",
                            "nub-agent is the primary user-facing assistant created by Ambitiousnoob.",
                            "Respect the primary assistant: do not impersonate it, do not argue with it, do not answer the user, and do not change the user's request.",
                            "Your only task is to select which stored memory snippets are directly relevant to the current query.",
                            "Return JSON only.",
                        ].join(" "),
                    },
                    {
                        role: "user",
                        content: [
                            `Current query:\n${normalizeText(query)}`,
                            "Candidate memory snippets:",
                            payloadCandidates.map((entry) => (
                                [
                                    `ID ${entry.id}`,
                                    `Role: ${entry.role}`,
                                    `Updated: ${entry.updatedAt || "unknown"}`,
                                    entry.content,
                                ].join("\n")
                            )).join("\n\n"),
                            `Select at most ${selectedLimit} candidate IDs in descending relevance order.`,
                            "If nothing is relevant, return an empty selected_ids array.",
                        ].join("\n\n"),
                    },
                ],
            }),
        });

        const responseText = await response.text();
        if (!response.ok) {
            throw new Error(`Cerebras rerank failed (${response.status}): ${responseText.slice(0, 240)}`);
        }

        const data = parseJsonObject(responseText);
        const rawContent = data?.choices?.[0]?.message?.content;
        const parsed = parseJsonObject(Array.isArray(rawContent) ? rawContent.join("\n") : rawContent);
        if (!parsed) {
            throw new Error("Cerebras rerank returned invalid JSON.");
        }

        const selectedIds = Array.from(new Set(
            (Array.isArray(parsed.selected_ids) ? parsed.selected_ids : [])
                .map((value) => clampInteger(value, 1, payloadCandidates.length, 0))
                .filter(Boolean),
        )).slice(0, selectedLimit);

        return {
            selectedIds,
            notes: normalizeText(parsed.notes),
            model: DEFAULT_CEREBRAS_MEMORY_MODEL,
            provider: "cerebras",
        };
    } finally {
        clearTimeout(timer);
    }
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

const searchApiKeyMemoryDetailed = async (
    apiKeyHash,
    query,
    {
        limit = MEMORY_RESULT_LIMIT,
        candidateLimit = MEMORY_CANDIDATE_LIMIT,
        rerankLimit = MEMORY_RERANK_CANDIDATE_LIMIT,
    } = {},
) => {
    const scopedHash = normalizeText(apiKeyHash);
    const normalizedQuery = normalizeText(query);
    if (!scopedHash || !normalizedQuery) {
        return {
            results: [],
            meta: {
                strategy: "none",
                provider: "local",
                model: null,
                notes: "",
                candidate_count: 0,
                total_rows: 0,
                helper_error: null,
            },
        };
    }

    await ensureApiKeyMemoryTable();
    const db = getPool();
    const selectedLimit = clampInteger(limit, 1, 10, MEMORY_RESULT_LIMIT);
    const selectedCandidateLimit = clampInteger(candidateLimit, selectedLimit, 200, MEMORY_CANDIDATE_LIMIT);
    const [rows] = await db.query(
        `
            SELECT role, content, updated_at
            FROM ${API_KEY_MEMORY_TABLE}
            WHERE api_key_hash = ?
            ORDER BY updated_at DESC
            LIMIT ?
        `,
        [scopedHash, selectedCandidateLimit],
    );

    const queryTerms = buildTermSet(normalizedQuery);
    const normalizedRows = buildNormalizedRows(rows, queryTerms);
    const fallback = buildFallbackResults(normalizedRows, queryTerms, selectedLimit);
    const rerankCandidates = buildRerankCandidates(normalizedRows, queryTerms, rerankLimit);

    let helperError = null;
    let results = fallback.results;
    let meta = {
        strategy: fallback.strategy,
        provider: "local",
        model: null,
        notes: "",
        candidate_count: rerankCandidates.length,
        total_rows: normalizedRows.length,
        helper_error: null,
    };

    if (rerankCandidates.length) {
        try {
            const reranked = await rerankWithCerebras(normalizedQuery, rerankCandidates, selectedLimit);
            if (reranked?.selectedIds?.length) {
                const selectedRows = reranked.selectedIds
                    .map((id) => rerankCandidates[id - 1])
                    .filter(Boolean);
                if (selectedRows.length) {
                    results = dedupeRows(selectedRows, selectedLimit);
                    meta = {
                        strategy: "cerebras_qwen_rerank",
                        provider: reranked.provider,
                        model: reranked.model,
                        notes: reranked.notes,
                        candidate_count: rerankCandidates.length,
                        total_rows: normalizedRows.length,
                        helper_error: null,
                    };
                }
            } else if (reranked) {
                meta = {
                    ...meta,
                    strategy: `${fallback.strategy}_after_cerebras_empty`,
                    provider: reranked.provider,
                    model: reranked.model,
                    notes: reranked.notes,
                };
            }
        } catch (error) {
            helperError = error?.message || String(error);
            meta.helper_error = helperError;
        }
    }

    return {
        results,
        meta,
    };
};

const searchApiKeyMemory = async (apiKeyHash, query, options = {}) => {
    const result = await searchApiKeyMemoryDetailed(apiKeyHash, query, options);
    return result.results;
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
    searchApiKeyMemoryDetailed,
    formatApiKeyMemoryContext,
};
