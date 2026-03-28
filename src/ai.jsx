import React, { useState, useRef, useEffect } from "react";
import TerminalMessage from "./TerminalMessage.jsx";

const HOSTED_CHAT_API_PATH = "/api/chat";
const BRAND_SYSTEM_MESSAGE = {
    role: "system",
    content: "You are nub-agent (ambitiousnoob), running on Cerebras Qwen 3 235B via Cerebras Inference. When asked about your model, identify exactly as: \"nub-agent (ambitiousnoob) on Cerebras Qwen 3 235B\" and avoid claiming affiliation with other providers.",
};
const AVAILABLE_MODELS = [
    {
        id: "qwen-3-235b-a22b-instruct-2507",
        label: "nub-agent (ambitiousnoob) · Cerebras Qwen 3 235B",
        icon: "",
        capabilities: ["text", "streaming"],
    },
];
const MAX_ITERATIONS = 20;
const MAX_RETRIES = 0;
const MAX_TOOL_CALLS_PER_ITERATION = 3;
const MAX_VERIFICATION_CYCLES = 1;
const MAX_OBSERVATION_CHARS = 2400;
const MAX_COMPLETION_TOKENS = 4096;
const MAX_SUBAGENT_TOKENS = 900;
const MAX_VERIFIER_TOKENS = 1200;
const MAX_RENDER_CHARS = 12000;
const STREAM_RENDER_CAP = 12000;
const MAX_UPLOAD_FILES = 6;
const MAX_FILE_INSERT_CHARS = 18000;
const MAX_TOTAL_UPLOAD_CHARS = 60000;
const MAX_IMAGE_UPLOAD_BYTES = 6 * 1024 * 1024;
const MAX_CONTEXT_CHARS = 120000;
const MIN_CONTEXT_CHARS = 24000;
const MAX_CURRENT_QUERY_CHARS = 60000;
const MAX_CONTEXT_MESSAGE_CHARS = 16000;
const MAX_OLDER_CONTEXT_MESSAGE_CHARS = 8000;
const MAX_CONTEXT_MESSAGES = 14;
const WORKING_MEMORY_TURNS = 5;
const WORKING_MEMORY_MESSAGES = WORKING_MEMORY_TURNS * 2;
const DEFAULT_PROMPT_TOKEN_LIMIT = 100000;
const DEFAULT_COMPLETION_TOKEN_RESERVE = 6000;
const IMAGE_ATTACHMENT_TOKEN_COST = 1200;
const TOOL_FIREWALL_CHUNK_CHARS = 1800;
const TOOL_FIREWALL_MAX_CHUNKS = 5;
const EPISODIC_SUMMARY_MAX_CHARS = 1400;
const SEMANTIC_CHUNK_CHARS = 1100;
const SEMANTIC_CHUNK_OVERLAP = 160;
const SEMANTIC_TOP_K = 3;
const MAX_SEMANTIC_CHUNKS = 90;
const MIN_SEMANTIC_TERM_LENGTH = 3;
const TEXT_FILE_NAME_RE = /\.(txt|md|markdown|json|csv|js|mjs|cjs|ts|tsx|jsx|py|rb|go|rs|java|c|h|cpp|hpp|html|css|scss|sass|xml|yaml|yml|toml|ini|env|log)$/i;
const IMAGE_FILE_NAME_RE = /\.(png|jpe?g|gif|webp|bmp|svg)$/i;
const OCR_LANGUAGE = "eng";
const CONTEXT_LENGTH_ERROR_RE = /(context_length_exceeded|max(?:imum)? context length|requested \d+ tokens)/i;
const HOSTED_API_ENABLED = true;
const HOSTED_API_LABEL = "Server API";
const MOBILE_UI_BREAKPOINT = 768;
const AUTO_SIDEBAR_BREAKPOINT = 900;
const ANDROID_USER_AGENT_RE = /Android/i;

const createId = () => (
    typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(16).slice(2)}`
);

const detectCompactUi = () => (
    typeof window !== "undefined"
        ? window.innerWidth <= MOBILE_UI_BREAKPOINT
        : false
);

const detectAndroidUi = () => (
    typeof navigator !== "undefined"
        ? ANDROID_USER_AGENT_RE.test(navigator.userAgent || "")
        : false
);

const createRunMetrics = () => ({
    iterations: 0,
    toolCalls: 0,
    verificationPasses: 0,
    verificationRevisions: 0,
    updatedAt: null,
});

const MODEL_TOKEN_PROFILES = Object.fromEntries(
    AVAILABLE_MODELS.map(model => [model.id, {
        promptLimit: DEFAULT_PROMPT_TOKEN_LIMIT,
        completionReserve: DEFAULT_COMPLETION_TOKEN_RESERVE,
    }])
);

const FASTEST_INFERENCE_PROFILE = Object.freeze({
    enabled: false,
    label: "1X",
    provider: undefined,
});
const roughTokenEstimate = (value) => {
    const text = String(value ?? "");
    if (!text) return 0;
    return Math.max(1, Math.ceil(text.length / 4));
};

// Lightweight token estimation only; avoids multi‑MB tokenizer chunk.
const countTextTokens = async (value) => roughTokenEstimate(value);

const countContentTokens = async (content) => {
    if (typeof content === "string") return countTextTokens(content);
    if (!Array.isArray(content)) return 0;
    let total = 0;
    for (const part of content) {
        if (part?.type === "text") {
            total += await countTextTokens(part.text || "");
        } else if (part?.type === "image_url") {
            total += IMAGE_ATTACHMENT_TOKEN_COST;
        }
    }
    return total;
};

const countMessagesTokens = async (messages = []) => {
    let total = 2;
    for (const message of Array.isArray(messages) ? messages : []) {
        if (!message) continue;
        total += 4;
        total += await countContentTokens(message.content);
    }
    return total;
};

const getModelTokenProfile = (model) => {
    const key = typeof model === "string" ? model : model?.id;
    return MODEL_TOKEN_PROFILES[key] || {
        promptLimit: DEFAULT_PROMPT_TOKEN_LIMIT,
        completionReserve: DEFAULT_COMPLETION_TOKEN_RESERVE,
    };
};

const buildOpenRouterRequestBody = ({ model, maxTokens, messages, stream }) => ({
    model: typeof model === "string" ? model : model?.id,
    max_tokens: maxTokens,
    messages,
    temperature: 0.2,
    stream,
});

const readApiErrorText = async (response) => {
    const raw = await response.text().catch(() => response.statusText);
    try {
        const parsed = JSON.parse(raw);
        return parsed?.error || parsed?.message || raw;
    } catch {
        return raw;
    }
};

const createStrategyState = () => ({
    router: null,
    plan: null,
    verification: null,
    metrics: createRunMetrics(),
});

const hydrateStrategyState = (raw = {}) => {
    const metrics = raw?.metrics && typeof raw.metrics === "object" ? raw.metrics : {};
    const normalizeItems = (items) => Array.isArray(items) ? items.map(item => String(item).trim()).filter(Boolean) : [];
    const normalizeBlock = (value) => (value && typeof value === "object" ? value : null);
    return {
        router: normalizeBlock(raw.router),
        plan: raw?.plan && typeof raw.plan === "object"
            ? {
                ...raw.plan,
                objectives: normalizeItems(raw.plan.objectives),
                constraints: normalizeItems(raw.plan.constraints),
                success: normalizeItems(raw.plan.success),
                tooling: normalizeItems(raw.plan.tooling),
            }
            : null,
        verification: raw?.verification && typeof raw.verification === "object"
            ? {
                ...raw.verification,
                checks: normalizeItems(raw.verification.checks),
                missing: normalizeItems(raw.verification.missing),
            }
            : null,
        metrics: {
            iterations: Number.isFinite(Number(metrics.iterations)) ? Number(metrics.iterations) : 0,
            toolCalls: Number.isFinite(Number(metrics.toolCalls)) ? Number(metrics.toolCalls) : 0,
            verificationPasses: Number.isFinite(Number(metrics.verificationPasses)) ? Number(metrics.verificationPasses) : 0,
            verificationRevisions: Number.isFinite(Number(metrics.verificationRevisions)) ? Number(metrics.verificationRevisions) : 0,
            updatedAt: typeof metrics.updatedAt === "string" ? metrics.updatedAt : null,
        },
    };
};

const formatConversationTitle = (createdAt) => {
    const dt = new Date(createdAt);
    return `Session ${new Intl.DateTimeFormat("en-US", {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
    }).format(dt)}`;
};

const createConversation = () => {
    const createdAt = new Date().toISOString();
    return {
        id: createId(),
        title: formatConversationTitle(createdAt),
        messages: [],
        pendingSteps: null,
        activeStream: null,
        systemLogs: [],
        currentInput: "",
        pendingUploads: [],
        memoryStore: {},
        memoryEngine: createMemoryEngine(),
        strategyState: createStrategyState(),
        createdAt,
    };
};

const hydrateConversation = (raw = {}) => {
    const createdAt = raw.createdAt || new Date().toISOString();
    return {
        id: raw.id || createId(),
        title: raw.title && raw.title !== "New Chat" ? raw.title : formatConversationTitle(createdAt),
        messages: Array.isArray(raw.messages) ? raw.messages : [],
        pendingSteps: Array.isArray(raw.pendingSteps) ? raw.pendingSteps : null,
        activeStream: typeof raw.activeStream === "string" ? raw.activeStream : null,
        systemLogs: Array.isArray(raw.systemLogs) ? raw.systemLogs : [],
        currentInput: typeof raw.currentInput === "string" ? raw.currentInput : "",
        pendingUploads: Array.isArray(raw.pendingUploads) ? raw.pendingUploads : [],
        memoryStore: raw.memoryStore && typeof raw.memoryStore === "object" ? raw.memoryStore : {},
        memoryEngine: hydrateMemoryEngine(raw.memoryEngine),
        strategyState: hydrateStrategyState(raw.strategyState),
        createdAt,
    };
};

const formatCrawlReport = (payload) => {
    if (!payload?.pages?.length) {
        const errorText = payload?.errors?.length
            ? `\nErrors:\n${payload.errors.map(item => `- ${item.url}: ${item.error}`).join("\n")}`
            : "";
        return `No readable pages were crawled from ${payload?.startUrl || "the requested URL"}.${errorText}`;
    }

    const lines = [
        `Crawl start: ${payload.startUrl}`,
        `Pages read: ${payload.pages.length}/${payload.maxPages} | Max depth: ${payload.maxDepth} | Same origin: ${payload.sameOrigin ? "yes" : "no"}`,
        `Discovered URLs: ${payload.discoveredCount}${payload.errors?.length ? ` | Errors: ${payload.errors.length}` : ""}`,
        "",
    ];

    payload.pages.forEach((page, index) => {
        lines.push(`${index + 1}. ${page.title || page.url}`);
        lines.push(`URL: ${page.url}`);
        lines.push(`Depth: ${page.depth} | Words: ${page.words} | Links found: ${page.linksDiscovered}${page.via ? ` | Via: ${page.via}` : ""}`);
        if (page.queuedLinks?.length) {
            lines.push(`Next links: ${page.queuedLinks.slice(0, 5).join(", ")}`);
        }
        lines.push(page.content || "(No readable text extracted)");
        if (page.contentTruncated) lines.push("[Content truncated]");
        lines.push("");
    });

    if (payload.errors?.length) {
        lines.push("Errors:");
        payload.errors.forEach(item => lines.push(`- ${item.url}: ${item.error}`));
    }

    return lines.join("\n").trim();
};

const getToolApiUrl = (path) => {
    if (typeof window === "undefined") return path;
    return new URL(path, window.location.origin).toString();
};

const getToolApiCandidates = (path) => {
    const candidates = [path];
    if (/^\/api\//.test(path) && !path.endsWith(".js")) {
        candidates.push(`${path}.js`);
    }
    return [...new Set(candidates)];
};

const canRetryToolApiPath = (error) => (
    /Tool backend HTTP (404|405)\b/i.test(error?.message || "") ||
    /unexpected .* content/i.test(error?.message || "")
);

const parseToolBackendResponse = async (response) => {
    const contentType = response.headers.get("content-type") || "";
    const raw = await response.text();
    const isJson = /application\/json/i.test(contentType);
    let data = null;

    if (isJson && raw) {
        try {
            data = JSON.parse(raw);
        } catch {
            throw new Error("Tool backend returned invalid JSON.");
        }
    }

    if (!response.ok) {
        if (/Authentication Required|Vercel Authentication/i.test(raw)) {
            throw new Error("The tool backend is behind Vercel Authentication. Open the deployment directly in a signed-in browser or make the deployment public.");
        }
        throw new Error(data?.error || `Tool backend HTTP ${response.status}`);
    }

    if (!isJson) {
        if (/Authentication Required|Vercel Authentication/i.test(raw)) {
            throw new Error("The tool backend is behind Vercel Authentication. Open the deployment directly in a signed-in browser or make the deployment public.");
        }
        throw new Error(`Tool backend returned unexpected ${contentType || "non-JSON"} content.`);
    }

    return data;
};

const callToolApi = async (path, payload) => {
    const candidates = getToolApiCandidates(path);
    let lastError = null;

    for (const candidate of candidates) {
        let response;
        try {
            response = await fetch(getToolApiUrl(candidate), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
                credentials: "include",
                cache: "no-store",
            });
        } catch (e) {
            lastError = new Error(`Tool backend request failed: ${e.message}`);
            if (candidate !== candidates[candidates.length - 1]) continue;
            throw lastError;
        }

        try {
            return await parseToolBackendResponse(response);
        } catch (e) {
            lastError = e;
            if (candidate !== candidates[candidates.length - 1] && canRetryToolApiPath(e)) continue;
            throw e;
        }
    }

    throw lastError || new Error("Tool backend request failed.");
};

const callWebReader = async ({ url, mode="article", maxChars=3200, includeLinks=true }) => {
    return callToolApi("/api/read.js", { url, mode, maxChars, includeLinks });
};

const formatReadReport = (payload, { includeLinks = true } = {}) => {
    const lines = [
        payload.title || payload.finalUrl,
        `URL: ${payload.finalUrl}`,
        `Content type: ${payload.contentType}`,
    ];

    if (payload.canonicalUrl && payload.canonicalUrl !== payload.finalUrl) {
        lines.push(`Canonical: ${payload.canonicalUrl}`);
    }
    if (payload.description) lines.push(`Description: ${payload.description}`);
    lines.push(`Words: ${payload.wordCount}`);
    if (payload.via) lines.push(`Fetched via: ${payload.via}`);

    if (payload.headings?.length) {
        lines.push("");
        lines.push("Headings:");
        payload.headings.forEach(item => lines.push(`- H${item.level}: ${item.text}`));
    }

    lines.push("");
    lines.push("Content:");
    lines.push(payload.content || "No readable content extracted.");

    if (payload.contentTruncated) {
        lines.push("");
        lines.push("[Content truncated]");
    }

    if (includeLinks && payload.links?.length) {
        lines.push("");
        lines.push("Links:");
        payload.links.forEach(link => lines.push(`- ${link}`));
    }

    return lines.join("\n").trim();
};

const getXmlTagValue = (text, tag) => (
    text.match(new RegExp(`<${tag}>([\\s\\S]*?)<\\/${tag}>`, "i"))?.[1]?.trim() || ""
);

const getXmlSectionItems = (text, sectionTag) => {
    const section = getXmlTagValue(text, sectionTag);
    return [...section.matchAll(/<item>([\s\S]*?)<\/item>/gi)].map(match => match[1].trim()).filter(Boolean);
};

const truncateText = (value, max = MAX_OBSERVATION_CHARS) => {
    const text = String(value ?? "");
    if (text.length <= max) return text;
    const suffix = `\n...[truncated ${text.length - max} chars]`;
    if (max <= suffix.length + 8) return text.slice(0, max);
    return `${text.slice(0, max - suffix.length)}${suffix}`;
};

const formatBytes = (bytes = 0) => {
    if (!Number.isFinite(Number(bytes))) return "0 B";
    const value = Number(bytes);
    if (value < 1024) return `${value} B`;
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
    return `${(value / (1024 * 1024)).toFixed(1)} MB`;
};

const normalizeTextBlock = (value) => String(value ?? "").replace(/\u0000/g, "").replace(/\r\n?/g, "\n").trim();

const isProbablyTextFile = (file) => {
    const type = file?.type || "";
    return TEXT_FILE_NAME_RE.test(file?.name || "") ||
        type.startsWith("text/") ||
        /(json|javascript|typescript|xml|yaml|csv)/i.test(type);
};

const isImageFile = (file) => {
    const type = file?.type || "";
    return type.startsWith("image/") || IMAGE_FILE_NAME_RE.test(file?.name || "");
};

const readFileAsDataUrl = (file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error(`Failed to read ${file?.name || "file"}.`));
    reader.readAsDataURL(file);
});

const STOP_WORDS = new Set([
    "the", "and", "for", "with", "that", "this", "from", "have", "your", "into", "about", "there",
    "their", "would", "could", "should", "after", "before", "where", "when", "what", "which", "while",
    "then", "than", "them", "they", "were", "been", "being", "also", "just", "over", "under", "through",
    "user", "assistant", "tool", "result", "results", "using", "used", "into", "onto", "http", "https",
    "www", "com", "org", "net", "api", "json", "html", "text", "data", "file", "files"
]);

const splitTextIntoChunks = (value, maxChars = SEMANTIC_CHUNK_CHARS, overlap = SEMANTIC_CHUNK_OVERLAP) => {
    const text = normalizeTextBlock(value);
    if (!text) return [];
    if (text.length <= maxChars) return [text];

    const chunks = [];
    let start = 0;

    while (start < text.length) {
        let end = Math.min(text.length, start + maxChars);
        if (end < text.length) {
            const window = text.slice(start, end);
            const preferredBreak = Math.max(
                window.lastIndexOf("\n\n"),
                window.lastIndexOf("\n"),
                window.lastIndexOf(". "),
                window.lastIndexOf(" ")
            );
            if (preferredBreak > Math.floor(maxChars * 0.55)) {
                end = start + preferredBreak + 1;
            }
        }
        const chunk = normalizeTextBlock(text.slice(start, end));
        if (chunk) chunks.push(chunk);
        if (end >= text.length) break;
        start = Math.max(end - overlap, start + 1);
    }

    return chunks;
};

const stripHtmlNoise = (value) => {
    const html = String(value ?? "");
    if (!/<[a-z][\s\S]*>/i.test(html)) return normalizeTextBlock(html);

    if (typeof DOMParser !== "undefined") {
        try {
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, "text/html");
            doc.querySelectorAll("script,style,noscript,nav,header,footer,aside,form,button,svg").forEach(node => node.remove());
            return normalizeTextBlock(doc.body?.textContent || doc.documentElement?.textContent || "");
        } catch {
            // Fall through to regex cleaning.
        }
    }

    return normalizeTextBlock(
        html
            .replace(/<script[\s\S]*?<\/script>/gi, " ")
            .replace(/<style[\s\S]*?<\/style>/gi, " ")
            .replace(/<nav[\s\S]*?<\/nav>/gi, " ")
            .replace(/<header[\s\S]*?<\/header>/gi, " ")
            .replace(/<footer[\s\S]*?<\/footer>/gi, " ")
            .replace(/<aside[\s\S]*?<\/aside>/gi, " ")
            .replace(/<[^>]+>/g, " ")
    );
};

const pruneJsonValue = (value, depth = 0) => {
    if (value == null) return null;
    if (typeof value === "string") {
        const text = normalizeTextBlock(value);
        return text ? truncateText(text, 700) : null;
    }
    if (typeof value === "number" || typeof value === "boolean") return value;
    if (Array.isArray(value)) {
        const cleaned = value.map(item => pruneJsonValue(item, depth + 1)).filter(item => item != null);
        if (!cleaned.length) return null;
        if (cleaned.length > 10) {
            return [...cleaned.slice(0, 10), `...[${cleaned.length - 10} more items]`];
        }
        return cleaned;
    }
    if (typeof value === "object") {
        const entries = Object.entries(value)
            .filter(([key]) => !/^(raw|html|headers|metadata|analytics|tracking|debug|trace)$/i.test(key))
            .map(([key, item]) => [key, pruneJsonValue(item, depth + 1)])
            .filter(([, item]) => item != null);
        if (!entries.length) return null;
        const limited = depth > 1 ? entries.slice(0, 12) : entries.slice(0, 20);
        return Object.fromEntries(limited);
    }
    return truncateText(String(value), 700);
};

const extractSemanticTerms = (value) => {
    const text = normalizeTextBlock(value).toLowerCase();
    if (!text) return [];
    return (text.match(/[a-z0-9_/-]{3,}/g) || [])
        .filter(term => term.length >= MIN_SEMANTIC_TERM_LENGTH && !STOP_WORDS.has(term))
        .slice(0, 400);
};

const buildSemanticVector = (value) => {
    const counts = {};
    extractSemanticTerms(value).forEach((term) => {
        counts[term] = (counts[term] || 0) + 1;
    });
    const ranked = Object.entries(counts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 40);
    const magnitude = Math.sqrt(ranked.reduce((sum, [, count]) => sum + (count * count), 0)) || 1;
    return Object.fromEntries(ranked.map(([term, count]) => [term, count / magnitude]));
};

const scoreSemanticMatch = (queryVector, chunkVector) => {
    let score = 0;
    const terms = Object.keys(queryVector || {});
    if (!terms.length) return 0;
    terms.forEach((term) => {
        score += (queryVector?.[term] || 0) * (chunkVector?.[term] || 0);
    });
    return score;
};

const normalizeSemanticChunk = (chunk = {}) => ({
    id: chunk.id || createId(),
    sourceType: chunk.sourceType || "context",
    sourceLabel: chunk.sourceLabel || "Context",
    text: truncateText(normalizeTextBlock(chunk.text || ""), SEMANTIC_CHUNK_CHARS + 200),
    terms: chunk.terms && typeof chunk.terms === "object" ? chunk.terms : buildSemanticVector(chunk.text || ""),
    tokenEstimate: Number.isFinite(Number(chunk.tokenEstimate)) ? Number(chunk.tokenEstimate) : roughTokenEstimate(chunk.text || ""),
    createdAt: typeof chunk.createdAt === "string" ? chunk.createdAt : new Date().toISOString(),
});

const buildSemanticChunks = (docs = [], { sourceType = "context" } = {}) => (
    (Array.isArray(docs) ? docs : []).flatMap((doc) => {
        const text = normalizeTextBlock(doc?.text || "");
        if (!text) return [];
        return splitTextIntoChunks(text).map((chunkText, index) => normalizeSemanticChunk({
            sourceType: doc?.sourceType || sourceType,
            sourceLabel: doc?.sourceLabel || doc?.name || "Context",
            text: chunkText,
            id: doc?.id ? `${doc.id}:${index}` : createId(),
            createdAt: doc?.createdAt,
        }));
    })
);

const mergeSemanticChunks = (existing = [], additions = []) => {
    const seen = new Set();
    const merged = [...(Array.isArray(existing) ? existing : []), ...(Array.isArray(additions) ? additions : [])]
        .map(normalizeSemanticChunk)
        .filter((chunk) => {
            const key = `${chunk.sourceType}:${chunk.sourceLabel}:${chunk.text.slice(0, 120)}`;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        });
    return merged.slice(-MAX_SEMANTIC_CHUNKS);
};

const createMemoryEngine = () => ({
    episodicSummary: "",
    semanticChunks: [],
    lastPromptStats: null,
    systemPromptCacheKey: "agent-system-v5",
});

const hydrateMemoryEngine = (raw = {}) => ({
    episodicSummary: typeof raw.episodicSummary === "string" ? raw.episodicSummary : "",
    semanticChunks: Array.isArray(raw.semanticChunks) ? raw.semanticChunks.map(normalizeSemanticChunk).slice(-MAX_SEMANTIC_CHUNKS) : [],
    lastPromptStats: raw.lastPromptStats && typeof raw.lastPromptStats === "object" ? raw.lastPromptStats : null,
    systemPromptCacheKey: typeof raw.systemPromptCacheKey === "string" ? raw.systemPromptCacheKey : "agent-system-v5",
});

const buildAttachmentLine = (attachment) => (
    `- ${attachment.kind === "image" ? "Image" : "Text"}: ${attachment.name} (${formatBytes(attachment.size)})${attachment.truncated ? " [truncated]" : ""}`
);

const buildTextAttachmentBlock = (attachment) => {
    const text = normalizeTextBlock(attachment?.textContent || "");
    if (!text) return "";
    return `--- File: ${attachment.name} (${formatBytes(attachment.size)})${attachment.truncated ? " [truncated]" : ""} ---\n${text}`;
};

const buildImageOcrBlock = (attachment) => {
    const text = normalizeTextBlock(attachment?.ocrText || "");
    if (!text) return "";
    return `--- OCR: ${attachment.name} ---\n${text}`;
};

const buildAttachmentManifest = (attachments = []) => (
    attachments.map(buildAttachmentLine).filter(Boolean).join("\n")
);

const buildUserHistoryText = (prompt, attachments = [], {
    includeManifest = true,
    includeTextBlocks = true,
    includeImageOcr = true,
    extraContext = "",
} = {}) => {
    const normalizedPrompt = normalizeTextBlock(prompt);
    const sections = [];
    if (normalizedPrompt) {
        sections.push(normalizedPrompt);
    } else if (attachments.length) {
        sections.push("Analyze the attached file(s).");
    }

    const manifest = includeManifest ? buildAttachmentManifest(attachments) : "";
    if (manifest) {
        sections.push(`Attached files:\n${manifest}`);
    }

    if (includeTextBlocks) {
        const textBlocks = attachments
            .filter(attachment => attachment?.kind === "text")
            .map(buildTextAttachmentBlock)
            .filter(Boolean);
        if (textBlocks.length) {
            sections.push(textBlocks.join("\n\n"));
        }
    }

    if (includeImageOcr) {
        const imageOcrBlocks = attachments
            .filter(attachment => attachment?.kind === "image")
            .map(buildImageOcrBlock)
            .filter(Boolean);
        if (imageOcrBlocks.length) {
            sections.push(imageOcrBlocks.join("\n\n"));
        }
    }

    const extra = normalizeTextBlock(extraContext);
    if (extra) {
        sections.push(extra);
    }

    return sections.join("\n\n").trim();
};

const buildUserModelContent = (prompt, attachments = [], options = {}) => {
    const historyText = buildUserHistoryText(prompt, attachments, options);
    if (!attachments.some(attachment => attachment?.kind === "image" && attachment?.dataUrl)) {
        return historyText;
    }

    const parts = historyText ? [{ type: "text", text: historyText }] : [];
    attachments.forEach((attachment) => {
        if (attachment?.kind === "image" && attachment?.dataUrl) {
            parts.push({
                type: "image_url",
                image_url: { url: attachment.dataUrl },
            });
        }
    });
    return parts;
};

const getMessageContentText = (content) => {
    if (typeof content === "string") return normalizeTextBlock(content);
    if (!Array.isArray(content)) return "";
    return normalizeTextBlock(
        content
            .filter(part => part?.type === "text" && typeof part.text === "string")
            .map(part => part.text)
            .join("\n\n")
    );
};

const messageHasContent = (message) => {
    if (!message) return false;
    if (getMessageContentText(message.content)) return true;
    return Array.isArray(message.content) && message.content.some(part => part?.type === "image_url" && part?.image_url?.url);
};

const rebuildMessageContent = (content, text) => {
    if (typeof content === "string" || !Array.isArray(content)) {
        return text;
    }
    const images = content.filter(part => part?.type === "image_url" && part?.image_url?.url);
    const parts = [];
    if (text) parts.push({ type: "text", text });
    parts.push(...images);
    return parts;
};

const isOcrPendingUpload = (attachment) => (
    attachment?.kind === "image" && ["queued", "running"].includes(attachment?.ocrStatus)
);

const buildAttachmentBadge = (attachment) => {
    if (!attachment) return "";
    const parts = [attachment.kind === "image" ? "Image" : "Text", formatBytes(attachment.size)];
    if (attachment.truncated) parts.push("truncated");
    if (attachment.kind === "image") {
        if (attachment.ocrStatus === "queued") {
            parts.push("OCR queued");
        } else if (attachment.ocrStatus === "running") {
            const progress = Math.max(0, Math.min(100, Math.round((Number(attachment.ocrProgress) || 0) * 100)));
            parts.push(progress > 0 ? `OCR ${progress}%` : "OCR running");
        } else if (attachment.ocrStatus === "done") {
            parts.push("OCR ready");
        } else if (attachment.ocrStatus === "empty") {
            parts.push("No text found");
        } else if (attachment.ocrStatus === "error") {
            parts.push("OCR failed");
        }
    }
    return parts.join(" • ");
};

const createPendingUpload = (attachment) => ({
    id: attachment.id || createId(),
    name: attachment.name || "attachment",
    size: Number.isFinite(Number(attachment.size)) ? Number(attachment.size) : 0,
    kind: attachment.kind === "image" ? "image" : "text",
    mimeType: attachment.mimeType || "",
    truncated: Boolean(attachment.truncated),
    textContent: typeof attachment.textContent === "string" ? attachment.textContent : "",
    dataUrl: typeof attachment.dataUrl === "string" ? attachment.dataUrl : "",
    ocrStatus: attachment.kind === "image" ? (attachment.ocrStatus || "queued") : "",
    ocrProgress: Number.isFinite(Number(attachment.ocrProgress)) ? Number(attachment.ocrProgress) : 0,
    ocrText: typeof attachment.ocrText === "string" ? attachment.ocrText : "",
    ocrError: typeof attachment.ocrError === "string" ? attachment.ocrError : "",
});

const createMessageAttachment = (attachment) => ({
    id: attachment.id || createId(),
    name: attachment.name || "attachment",
    size: Number.isFinite(Number(attachment.size)) ? Number(attachment.size) : 0,
    kind: attachment.kind === "image" ? "image" : "text",
    mimeType: attachment.mimeType || "",
    truncated: Boolean(attachment.truncated),
    previewUrl: attachment.kind === "image"
        ? (typeof attachment.previewUrl === "string" && attachment.previewUrl) || (typeof attachment.dataUrl === "string" ? attachment.dataUrl : "")
        : "",
    ocrStatus: attachment.kind === "image" ? (attachment.ocrStatus || "") : "",
    ocrProgress: Number.isFinite(Number(attachment.ocrProgress)) ? Number(attachment.ocrProgress) : 0,
    ocrError: typeof attachment.ocrError === "string" ? attachment.ocrError : "",
});

const serializeConversation = (conversation = {}) => ({
    ...conversation,
    activeStream: null,
    pendingSteps: null,
    pendingUploads: [],
    memoryEngine: hydrateMemoryEngine(conversation.memoryEngine),
    messages: Array.isArray(conversation.messages)
        ? conversation.messages.map(message => ({
            ...message,
            content: message.role === "assistant"
                ? sanitizeAssistantHistoryText(message.content)
                : message.content,
            steps: sanitizeStepsForHistory(message.steps),
            attachments: Array.isArray(message?.attachments)
                ? message.attachments.map(attachment => ({
                    ...attachment,
                    previewUrl: "",
                }))
                : [],
        }))
        : [],
});

const stripAgentArtifacts = (value) => {
    const text = normalizeTextBlock(value);
    if (!text) return "";

    const withoutThoughts = text.replace(/<thought>[\s\S]*?<\/thought>/gi, "").trim();
    const withoutToolCalls = withoutThoughts.replace(/<tool_call>[\s\S]*?<\/tool_call>/gi, "").trim();
    if (withoutToolCalls) return withoutToolCalls;

    const toolNames = [...text.matchAll(/<function=([^>]+)>/gi)]
        .map(match => match[1]?.trim())
        .filter(Boolean);
    if (toolNames.length) {
        return `Tool calls executed: ${[...new Set(toolNames)].join(", ")}.`;
    }

    return text.replace(/\s+/g, " ").trim();
};

const sanitizeStepForHistory = (step = {}) => {
    if (!step || typeof step !== "object") return step;
    const { thought, ...rest } = step;
    if (rest.feedback && typeof rest.feedback === "object") {
        rest.feedback = {
            ...rest.feedback,
            analysis: "",
        };
    }
    return rest;
};

const sanitizeStepsForHistory = (steps = []) => (
    Array.isArray(steps) ? steps.map(sanitizeStepForHistory) : []
);

const sanitizeAssistantHistoryText = (value, parsed = null) => {
    const finalAnswer = normalizeTextBlock(parsed?.final_answer || "");
    if (finalAnswer) return finalAnswer;
    return stripAgentArtifacts(getMessageContentText(value) || value);
};

const buildMessageDigest = (message) => {
    if (!message || message.role === "system") return "";

    if (message.error) {
        return `Assistant error: ${truncateText(String(message.error), 220)}`;
    }

    const attachments = Array.isArray(message.attachments) ? message.attachments : [];
    const attachmentSummary = attachments.length
        ? `Attachments: ${attachments.map(attachment => attachment?.name || "attachment").filter(Boolean).join(", ")}.`
        : "";

    const text = message.role === "assistant"
        ? sanitizeAssistantHistoryText(message.content)
        : normalizeTextBlock(message.displayText || getMessageContentText(message.content) || attachmentSummary);

    if (!text) return attachmentSummary || "";

    return `${message.role === "assistant" ? "Assistant" : "User"}: ${truncateText(text.replace(/\s+/g, " "), 220)}`;
};

const buildEpisodicSummary = (messages = []) => {
    const relevant = (Array.isArray(messages) ? messages : []).filter(message => (
        message &&
        message.role !== "system" &&
        (messageHasContent(message) || message.error || (Array.isArray(message.attachments) && message.attachments.length))
    ));

    if (relevant.length <= WORKING_MEMORY_MESSAGES) return "";

    const digests = relevant
        .slice(0, -WORKING_MEMORY_MESSAGES)
        .map(buildMessageDigest)
        .filter(Boolean);

    if (!digests.length) return "";
    return truncateText(`Previously:\n${digests.join("\n")}`, EPISODIC_SUMMARY_MAX_CHARS);
};

const buildWorkingMemoryMessages = (messages = []) => (
    (Array.isArray(messages) ? messages : [])
        .filter(message => (
            message &&
            message.role !== "system" &&
            (messageHasContent(message) || message.error || (Array.isArray(message.attachments) && message.attachments.length))
        ))
        .slice(-WORKING_MEMORY_MESSAGES)
        .map((message) => {
            const attachmentSummary = Array.isArray(message.attachments) && message.attachments.length
                ? `Attached files:\n${buildAttachmentManifest(message.attachments)}`
                : "";
            let text = "";

            if (message.error) {
                text = `Previous error: ${message.error}`;
            } else if (message.role === "assistant") {
                text = sanitizeAssistantHistoryText(message.content);
            } else {
                text = normalizeTextBlock(message.displayText || getMessageContentText(message.content) || attachmentSummary);
            }

            if (!text) return null;
            return {
                role: message.role === "assistant" ? "assistant" : "user",
                content: truncateText(text, MAX_CONTEXT_MESSAGE_CHARS),
            };
        })
        .filter(Boolean)
);

const buildAttachmentSemanticDocs = (attachments = []) => (
    (Array.isArray(attachments) ? attachments : []).flatMap((attachment) => {
        const docs = [];
        const createdAt = new Date().toISOString();
        if (attachment?.kind === "text") {
            const text = normalizeTextBlock(attachment.textContent || "");
            if (text) {
                docs.push({
                    id: attachment.id || createId(),
                    sourceType: "attachment",
                    sourceLabel: attachment.name || "Text attachment",
                    text,
                    createdAt,
                });
            }
        }
        if (attachment?.kind === "image") {
            const text = normalizeTextBlock(attachment.ocrText || "");
            if (text) {
                docs.push({
                    id: `${attachment.id || createId()}:ocr`,
                    sourceType: "attachment",
                    sourceLabel: `${attachment.name || "Image"} OCR`,
                    text,
                    createdAt,
                });
            }
        }
        return docs;
    })
);

const buildToolSemanticDocs = (actionName, firewallOutput, createdAt = new Date().toISOString()) => (
    (Array.isArray(firewallOutput?.chunks) ? firewallOutput.chunks : []).map((chunk, index) => ({
        id: createId(),
        sourceType: "tool",
        sourceLabel: chunk && firewallOutput?.chunkCount > 1
            ? `${actionName} (${index + 1}/${firewallOutput.chunkCount})`
            : actionName,
        text: chunk,
        createdAt,
    }))
);

const buildSemanticContext = (query, semanticChunks = [], { topK = SEMANTIC_TOP_K } = {}) => {
    const chunks = (Array.isArray(semanticChunks) ? semanticChunks : [])
        .map(normalizeSemanticChunk)
        .filter(chunk => chunk.text);

    if (!chunks.length) return { text: "", hits: [] };

    const queryVector = buildSemanticVector(query);
    const ranked = chunks
        .map((chunk) => ({
            chunk,
            score: scoreSemanticMatch(queryVector, chunk.terms),
        }))
        .sort((a, b) => (
            (b.score - a.score) ||
            (new Date(b.chunk.createdAt).getTime() - new Date(a.chunk.createdAt).getTime())
        ));

    let hits = ranked.filter(entry => entry.score > 0).slice(0, topK);
    if (!hits.length) {
        hits = ranked.slice(0, topK);
    }

    const text = hits.length
        ? `Relevant retrieved context:\n${hits.map((entry, index) => `${index + 1}. [${entry.chunk.sourceLabel}] ${entry.chunk.text}`).join("\n\n")}`
        : "";

    return {
        text,
        hits: hits.map(entry => entry.chunk),
    };
};

const formatToolChunkLabel = (actionName, chunkIndex, totalChunks) => (
    `${actionName} chunk ${chunkIndex} of ${totalChunks}`
);

const firewallToolOutput = (actionName, rawOutput) => {
    let cleanedValue = rawOutput;
    let wasJson = false;
    let wasHtml = false;

    if (cleanedValue != null && typeof cleanedValue === "object") {
        cleanedValue = pruneJsonValue(cleanedValue);
        wasJson = true;
    }

    if (typeof cleanedValue === "string") {
        const trimmed = cleanedValue.trim();
        if (!wasJson && (trimmed.startsWith("{") || trimmed.startsWith("["))) {
            try {
                cleanedValue = pruneJsonValue(JSON.parse(trimmed));
                wasJson = true;
            } catch {
                cleanedValue = trimmed;
            }
        }
    }

    let cleanedText = "";
    if (typeof cleanedValue === "string") {
        if (/<[a-z][\s\S]*>/i.test(cleanedValue)) {
            cleanedText = stripHtmlNoise(cleanedValue);
            wasHtml = true;
        } else {
            cleanedText = normalizeTextBlock(cleanedValue);
        }
    } else {
        try {
            cleanedText = JSON.stringify(cleanedValue, null, 2);
        } catch {
            cleanedText = String(cleanedValue ?? "");
        }
    }

    cleanedText = normalizeTextBlock(cleanedText);
    if (!cleanedText) cleanedText = "Tool returned no readable content.";

    const allChunks = splitTextIntoChunks(cleanedText, TOOL_FIREWALL_CHUNK_CHARS, 120);
    const chunkCount = allChunks.length || 1;
    const storedChunks = (allChunks.length ? allChunks : [cleanedText]).slice(0, TOOL_FIREWALL_MAX_CHUNKS);
    const omittedChunkCount = Math.max(0, chunkCount - storedChunks.length);
    let agentText = storedChunks[0] || cleanedText;

    if (chunkCount > 1) {
        agentText = `${formatToolChunkLabel(actionName, 1, chunkCount)}\n${agentText}\n\n[Firewall note: only chunk 1 of ${chunkCount} is shown to the agent. ${Math.max(0, storedChunks.length - 1)} more sanitized chunk(s) were indexed for recall${omittedChunkCount ? `; ${omittedChunkCount} chunk(s) were dropped from persistence.` : "."}]`;
    }

    return {
        agentText,
        cleanedText,
        chunks: storedChunks,
        tokenEstimate: roughTokenEstimate(cleanedText),
        chunkCount,
        storedChunkCount: storedChunks.length,
        omittedChunkCount,
        wasJson,
        wasHtml,
    };
};

const buildContextWindow = (messages, { maxChars = MAX_CONTEXT_CHARS, latestUserMaxChars = MAX_CONTEXT_MESSAGE_CHARS } = {}) => {
    const relevant = (Array.isArray(messages) ? messages : []).filter(item => (
        item &&
        item.role !== "system" &&
        messageHasContent(item)
    ));
    const windowed = relevant.slice(-MAX_CONTEXT_MESSAGES);
    const prepared = [];
    let trimmed = relevant.length > windowed.length;
    let remaining = maxChars;

    for (let index = windowed.length - 1; index >= 0; index -= 1) {
        const item = windowed[index];
        const isRecent = index >= windowed.length - 4;
        const isLatestUser = index === windowed.length - 1 && item.role === "user";
        const perMessageCap = isLatestUser
            ? latestUserMaxChars
            : isRecent
                ? MAX_CONTEXT_MESSAGE_CHARS
                : MAX_OLDER_CONTEXT_MESSAGE_CHARS;
        const rawText = getMessageContentText(item.content);
        const baseText = item.role === "assistant"
            ? stripAgentArtifacts(rawText)
            : normalizeTextBlock(rawText);

        if (!baseText) {
            if (Array.isArray(item.content) && item.content.some(part => part?.type === "image_url" && part?.image_url?.url)) {
                prepared.unshift({
                    role: item.role === "assistant" ? "assistant" : "user",
                    content: rebuildMessageContent(item.content, ""),
                });
                continue;
            }
            trimmed = true;
            continue;
        }

        let content = baseText;
        if (content.length > perMessageCap) {
            content = truncateText(content, perMessageCap);
            trimmed = true;
        }
        if (content.length > remaining) {
            if (remaining < 800) {
                trimmed = true;
                continue;
            }
            content = truncateText(content, remaining);
            trimmed = true;
        }

        prepared.unshift({
            role: item.role === "assistant" ? "assistant" : "user",
            content: rebuildMessageContent(item.content, content),
        });
        remaining -= content.length;
        if (remaining <= 0) break;
    }

    return { messages: prepared, trimmed };
};

const buildPreparedModelMessages = (messages, { maxChars = MAX_CONTEXT_CHARS, latestUserMaxChars = MAX_CONTEXT_MESSAGE_CHARS } = {}) => {
    const systemMessages = [];
    const conversational = [];

    for (const message of Array.isArray(messages) ? messages : []) {
        if (!message || !messageHasContent(message)) continue;
        if (message.role === "system") {
            systemMessages.push({ role: "system", content: getMessageContentText(message.content).trim() });
        } else {
            conversational.push(message);
        }
    }

    const window = buildContextWindow(conversational, { maxChars, latestUserMaxChars });
    return {
        messages: [...systemMessages, ...window.messages],
        trimmed: window.trimmed,
    };
};

const compactSupplementalSystemMessages = (messages = [], cap = 1500) => {
    let firstSystem = true;
    return (Array.isArray(messages) ? messages : [])
        .map((message) => {
            if (!message || message.role !== "system") return message;
            const text = getMessageContentText(message.content);
            if (!text) return null;
            const next = {
                role: "system",
                content: truncateText(text, firstSystem ? Math.max(cap * 2, 3200) : cap),
            };
            firstSystem = false;
            return next;
        })
        .filter(Boolean);
};

const prepareMessagesForDispatch = async (messages, {
    chain = [],
    maxTokens = MAX_COMPLETION_TOKENS,
    maxContextChars = MAX_CONTEXT_CHARS,
    latestUserMaxChars = MAX_CONTEXT_MESSAGE_CHARS,
} = {}) => {
    const primaryModel = Array.isArray(chain) && chain.length ? chain[0] : AVAILABLE_MODELS[0];
    const profile = getModelTokenProfile(primaryModel);
    const promptLimit = profile.promptLimit;
    const completionReserve = Math.max(maxTokens, profile.completionReserve || 0);
    const promptBudget = Math.max(3000, promptLimit - completionReserve);

    let sourceMessages = Array.isArray(messages) ? messages : [];
    let charBudget = maxContextChars;
    let latestBudget = latestUserMaxChars;
    let prepared = buildPreparedModelMessages(sourceMessages, {
        maxChars: charBudget,
        latestUserMaxChars: latestBudget,
    });
    let tokenCount = await countMessagesTokens(prepared.messages);
    let trimmed = prepared.trimmed;
    let attempts = 0;

    while (tokenCount > promptBudget && attempts < 7) {
        attempts += 1;
        trimmed = true;

        if (attempts === 4) {
            sourceMessages = compactSupplementalSystemMessages(sourceMessages, 1200);
        }

        charBudget = Math.max(Math.floor(maxContextChars * 0.18), Math.floor(charBudget * 0.72));
        latestBudget = Math.max(1200, Math.min(latestBudget, Math.floor(charBudget * 0.7)));

        prepared = buildPreparedModelMessages(sourceMessages, {
            maxChars: charBudget,
            latestUserMaxChars: latestBudget,
        });
        tokenCount = await countMessagesTokens(prepared.messages);
    }

    return {
        messages: prepared.messages,
        tokenCount,
        promptBudget,
        promptLimit,
        completionReserve,
        trimmed,
        charBudget,
        latestUserMaxChars: latestBudget,
    };
};

const isContextLengthError = (error) => CONTEXT_LENGTH_ERROR_RE.test(String(error?.message || error || ""));

const formatPlanForSystem = (plan) => {
    if (!plan) return "";
    const lines = ["Execution plan:"];
    if (plan.goal) lines.push(`Goal: ${plan.goal}`);
    if (plan.complexity) lines.push(`Complexity: ${plan.complexity}`);
    if (plan.objectives?.length) lines.push(`Objectives: ${plan.objectives.join(" | ")}`);
    if (plan.constraints?.length) lines.push(`Constraints: ${plan.constraints.join(" | ")}`);
    if (plan.success?.length) lines.push(`Success criteria: ${plan.success.join(" | ")}`);
    if (plan.tooling?.length) lines.push(`Preferred tooling: ${plan.tooling.join(" | ")}`);
    if (plan.notes) lines.push(`Manager notes: ${plan.notes}`);
    return lines.join("\n");
};

const formatMemoryContext = (memoryStore) => {
    const entries = Object.entries(memoryStore || {});
    if (!entries.length) return "";
    const lines = ["Session memory:"];
    entries.forEach(([key, value]) => lines.push(`- ${key}: ${value?.value || ""}`));
    lines.push("Use this only when relevant and do not invent extra memory.");
    return lines.join("\n");
};

const formatStepsForVerifier = (steps = []) => {
    const actionSteps = steps.filter(step => step.type === "action").slice(-8);
    if (!actionSteps.length) return "No tool evidence was collected.";
    return actionSteps.map((step, index) => {
        const input = step.input && Object.keys(step.input).length ? truncateText(JSON.stringify(step.input), 320) : "{}";
        const observation = truncateText(String(step.observation || "").replace(/\s+/g, " ").trim(), 900);
        return `Step ${index + 1}: ${step.action}\nInput: ${input}\nObservation: ${observation}`;
    }).join("\n\n");
};



const formatObservationBundle = (actions, wasTrimmed = false) => {
    const lines = ["Tool results:"];
    actions.forEach((item, index) => {
        lines.push(`${index + 1}. ${item.action}`);
        if (item.input && Object.keys(item.input).length) {
            lines.push(`Input: ${truncateText(JSON.stringify(item.input), 320)}`);
        }
        const firewallNotes = [];
        if (item.firewallMeta?.wasJson) firewallNotes.push("JSON pruned");
        if (item.firewallMeta?.wasHtml) firewallNotes.push("HTML cleaned");
        if (item.firewallMeta?.chunkCount > 1) firewallNotes.push(`showing chunk 1 of ${item.firewallMeta.chunkCount}`);
        if (firewallNotes.length) {
            lines.push(`Firewall: ${firewallNotes.join(" | ")}`);
        }
        lines.push(`Observation:\n${truncateText(item.observation, MAX_OBSERVATION_CHARS)}`);
        if (index < actions.length - 1) lines.push("---");
    });
    if (wasTrimmed) {
        lines.push("");
        lines.push(`Only the first ${MAX_TOOL_CALLS_PER_ITERATION} tool calls were executed from this batch.`);
    }
    lines.push("");
    lines.push("Continue. Use more tools if needed, or provide the final answer if the task is complete.");
    return lines.join("\n");
};

const buildPlannerPrompt = (desc) => `You are Execution Planner, a specialist sub-agent.

Your only job is to create a concise execution blueprint for the next agent pass.

AVAILABLE TOOLS:
${desc}

Respond using this exact XML format:
<plan>
<goal>one sentence goal</goal>
<complexity>low|medium|high</complexity>
<objectives>
<item>objective 1</item>
</objectives>
<constraints>
<item>constraint 1</item>
</constraints>
<success>
<item>success criterion 1</item>
</success>
<tooling>
<item>tool_name: why it helps</item>
</tooling>
<notes>short execution note</notes>
</plan>

Rules:
- Do not solve the user request.
- Keep each list short and high signal.
- Mention evidence requirements when facts or web content matter.
- Prefer tool-supported plans over pure reasoning.`;

const parsePlannerResponse = (text) => ({
    goal: getXmlTagValue(text, "goal"),
    complexity: getXmlTagValue(text, "complexity") || "medium",
    objectives: getXmlSectionItems(text, "objectives"),
    constraints: getXmlSectionItems(text, "constraints"),
    success: getXmlSectionItems(text, "success"),
    tooling: getXmlSectionItems(text, "tooling"),
    notes: getXmlTagValue(text, "notes"),
});

const buildVerifierPrompt = ({ query, plan, steps, answer }) => `You are Answer Verifier, a specialist sub-agent.

Your only job is to decide whether the draft answer is ready for the user.

USER REQUEST:
${query}

EXECUTION PLAN:
${plan ? formatPlanForSystem(plan) : "No structured plan was available."}

RECENT TOOL EVIDENCE:
${formatStepsForVerifier(steps)}

DRAFT ANSWER:
${answer}

Respond using this exact XML format:
<verification>
<status>pass|revise</status>
<summary>one sentence</summary>
<checks>
<item>what is already good</item>
</checks>
<missing>
<item>what still needs work</item>
</missing>
<next_action>single best next action</next_action>
</verification>

Rules:
- Use "pass" only if the draft is complete, grounded, and directly answers the request.
- Use "revise" if there is a material gap, unsupported claim, or missing execution step.
- Keep the response concise and execution-focused.`;

const parseVerifierResponse = (text) => {
    const status = (getXmlTagValue(text, "status") || "pass").toLowerCase();
    return {
        status: status === "revise" ? "revise" : "pass",
        summary: getXmlTagValue(text, "summary"),
        checks: getXmlSectionItems(text, "checks"),
        missing: getXmlSectionItems(text, "missing"),
        nextAction: getXmlTagValue(text, "next_action"),
    };
};

const formatVerifierFeedback = (report) => {
    const lines = [
        `Verifier status: ${report.status}`,
        `Summary: ${report.summary || "No summary provided."}`,
    ];
    if (report.checks?.length) lines.push(`Checks: ${report.checks.join(" | ")}`);
    if (report.missing?.length) lines.push(`Missing: ${report.missing.join(" | ")}`);
    if (report.nextAction) lines.push(`Next action: ${report.nextAction}`);
    return lines.join("\n");
};

const buildRouterPrompt = (desc) => `You are Tool Router, a specialist sub-agent.

Your only job is to choose the best starting tool for the request.

AVAILABLE TOOLS:
${desc}

Respond using this exact XML format:
<analysis>short reasoning</analysis>
<route>
<primary>tool_name_or_none</primary>
<secondary>comma,separated,backup,tools</secondary>
<reason>one sentence</reason>
</route>

Rules:
- Pick one primary tool or "none".
- Prefer web_search for external info; use web_fetch only when a specific URL is provided.
- Prefer specialized tools over generic ones.
- Do not answer the user request directly.
- Keep the analysis short.`;

const parseRouterResponse = (text) => {
    const analysis = text.match(/<analysis>([\s\S]*?)<\/analysis>/)?.[1]?.trim() || "";
    const primary = text.match(/<primary>([\s\S]*?)<\/primary>/)?.[1]?.trim() || "none";
    const secondaryRaw = text.match(/<secondary>([\s\S]*?)<\/secondary>/)?.[1]?.trim() || "";
    const reason = text.match(/<reason>([\s\S]*?)<\/reason>/)?.[1]?.trim() || "";
    const secondary = secondaryRaw
        ? secondaryRaw.split(",").map(item => item.trim()).filter(Boolean)
        : [];
    return { analysis, primary, secondary, reason };
};

function createTools() {
    return {
        calculate: {
            category: "Core",
            icon: "",
            description: "Evaluate arithmetic expressions (use when math is required).",
            example: "calculate: 15*7+(2/3)",
        },
        web_search: {
            category: "Search",
            icon: "",
            description: "Search the web for recent information and return top snippets.",
            example: "web_search: latest cerebras tool calling docs",
        },
        web_fetch: {
            category: "Search",
            icon: "",
            description: "Fetch a URL (text/html) and return up to ~3000 characters.",
            example: "web_fetch: https://example.com/page",
        },
        search_images: {
            category: "Search",
            icon: "",
            description: "Search the web for relevant images and return up to 10 image URLs/snippets.",
            example: "search_images: solar eclipse nasa photo",
        },
        view_image: {
            category: "Search",
            icon: "",
            description: "Inspect an image URL for metadata (content type, size) to verify a link.",
            example: "view_image: https://example.com/pic.jpg",
        },
    };
}

const buildSystemPrompt = (n, desc) => `You are an exceptionally capable autonomous AI agent.

PHILOSOPHY:
- Think deeply before acting. Break complex problems into subtasks.
- Use multiple tools in sequence when needed.
- When several tool calls are independent, batch up to ${MAX_TOOL_CALLS_PER_ITERATION} <tool_call> blocks in one response.
- Cross-verify important information using different tools.
- Be precise, comprehensive, and honest about uncertainty.
- Respect the manager plan, router guidance, memory context, and verifier feedback when they are provided.
- Preserve evidence from web tools and include source URLs when that helps the user.
- When you have enough information, synthesize a clear final answer.

AVAILABLE TOOLS (${n} total):
${desc}

STRICT XML FORMAT — you MUST use the following XML tags:

To think about what to do next:
<thought>
I need to calculate 25 * 4, I will use the calculator tool.
</thought>

To use a tool (must follow a <thought>):
<tool_call>
<function=tool_name>
<parameter=param_name>value</parameter>
</function>
</tool_call>

To give your final answer to the user (must follow a <thought>):
<thought>
I have the result. I will tell the user.
</thought>
The answer is 100.

RULES:
1. ALWAYS write a <thought> before taking an action or answering.
2. Use tools for any factual, computational, or data-retrieval need.
3. If a tool errors, adapt: try different params or tools.
4. Before finalizing, make sure the critical objectives are satisfied or explicitly state what is still missing.
5. Final answers should be complete, structured, and use markdown.
6. Do NOT use JSON for tool calls, ONLY use the exact <tool_call> XML format.`;

async function callLLMStream(messages, model, signal, onUpdate, { maxTokens = MAX_COMPLETION_TOKENS, log, clientTimeoutMs = 70000, onToolCall } = {}) {
    const ensureBrandedMessages = (msgs = []) => (
        Array.isArray(msgs) && msgs.some(m => m?.role === "system")
            ? msgs
            : [BRAND_SYSTEM_MESSAGE, ...msgs]
    );

    const maxAttempts = MAX_RETRIES + 1;
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        try {
            const brandedMessages = ensureBrandedMessages(messages);
            log?.(`Dispatching to ${model} (attempt ${attempt + 1}/${maxAttempts})`);
            const controller = new AbortController();
            if (signal) {
                if (signal.aborted) controller.abort(signal.reason);
                else signal.addEventListener("abort", () => controller.abort(signal.reason), { once: true });
            }
            const timeoutId = setTimeout(() => controller.abort(), clientTimeoutMs);
                const res = await fetch(getToolApiUrl(HOSTED_CHAT_API_PATH), {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(buildOpenRouterRequestBody({
                        model,
                        maxTokens,
                        messages: brandedMessages,
                        stream: true,
                    })),
                    credentials: "include",
                    cache: "no-store",
                    signal: controller.signal,
            });
            clearTimeout(timeoutId);

            if (!res.ok) {
                const errText = await readApiErrorText(res);
                const details = `${res.status} ${res.statusText}: ${errText}`;
                log?.(`Upstream error: ${details}`);
                if (res.status === 429 || res.status === 503) {
                    if (attempt < maxAttempts - 1) {
                        const backoff = 1000 * Math.pow(2, attempt);
                        log?.(`Backing off ${backoff}ms and retrying...`);
                        await new Promise(r => setTimeout(r, backoff));
                        continue;
                    }
                    throw new Error(details);
                }
                if (isContextLengthError(errText)) {
                    throw new Error(`Context window exceeded. ${errText}`);
                }
                throw new Error(`${HOSTED_API_LABEL} ${details}`);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let fullText = "";
            let collectedTools = [];
            let buffer = "";
            let chunkCount = 0;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || "";

                for (const line of lines) {
                    if (line.startsWith('data: ') && line !== 'data: [DONE]') {
                        try {
                            const data = JSON.parse(line.slice(6));
                            const chunk = data.choices?.[0]?.delta?.content || "";
                            if (chunk) {
                                chunkCount += 1;
                                fullText += chunk;
                                onUpdate(fullText);
                            }
                        } catch (e) { /* ignore parse error on partial chunks */ }
                    }
                }
            }
            log?.(`Stream finished with ${chunkCount} chunk(s).`);

            if (!fullText && chunkCount === 0) {
                log?.("Stream returned no chunks; retrying once as non-stream.");
                const fallbackRes = await fetch(getToolApiUrl(HOSTED_CHAT_API_PATH), {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(buildOpenRouterRequestBody({
                        model,
                        maxTokens,
                        messages: brandedMessages,
                        stream: false,
                    })),
                    credentials: "include",
                    cache: "no-store",
                    signal: controller.signal,
                });
                if (!fallbackRes.ok) {
                    const errText = await readApiErrorText(fallbackRes);
                    throw new Error(`${HOSTED_API_LABEL} fallback HTTP ${fallbackRes.status}: ${errText}`);
                }
                const json = await fallbackRes.json();
                const text = json?.choices?.[0]?.message?.content || json?.output_text || "";
                const toolsUsed = json?.tools_used || json?.toolsUsed;
                collectedTools = Array.isArray(toolsUsed) ? toolsUsed : [];
                const logTool = typeof onToolCall === "function" ? onToolCall : () => {};
                if (Array.isArray(collectedTools)) {
                    collectedTools.forEach((entry) => {
                        const name = entry?.name || "tool";
                        let args = entry?.args;
                        try {
                            if (typeof args === "string") args = JSON.parse(args);
                        } catch {
                            /* ignore JSON parse errors */
                        }
                        logTool(name, args);
                    });
                }
                onUpdate(text || "");
                log?.("Non-stream retry completed.");
                return { text, toolsUsed: collectedTools };
            }

            return { text: fullText, toolsUsed: collectedTools };
        } catch (e) {
            if (e.name === "AbortError") throw e;
            if (attempt === maxAttempts - 1) throw e;
            log?.(`Attempt ${attempt + 1} failed: ${e?.message || e}. Retrying...`);
        }
    }
}

async function callLLMText(messages, chain, signal, {
    maxTokens = MAX_SUBAGENT_TOKENS,
    maxContextChars = MAX_CONTEXT_CHARS,
    latestUserMaxChars = MAX_CONTEXT_MESSAGE_CHARS,
    onCompacted,
    onPrepared,
} = {}) {
    const prepared = await prepareMessagesForDispatch(messages, {
        chain,
        maxTokens,
        maxContextChars,
        latestUserMaxChars,
    });
    if (prepared.trimmed && typeof onCompacted === "function") {
        onCompacted(prepared);
    }
    if (typeof onPrepared === "function") onPrepared(prepared);
    let lastErr = null;
    for (const model of chain) {
        if (signal?.aborted) throw new Error("aborted");
        try {
            const { text } = await callLLMStream(prepared.messages, model, signal, () => {}, { maxTokens });
            return text;
        } catch (e) {
            lastErr = e;
            if (e.name === "AbortError") throw e;
            if (e.message.includes("402") || e.message.includes("429") || e.message === "rate_limited") continue;
            throw e;
        }
    }
    throw lastErr || new Error("All models failed");
}

function parseXMLAgentResponse(text) {
    const res = { thought: "", action: null, action_input: {}, actions: [], final_answer: "" };
    
    // Extract thought
    const thoughtMatches = [...text.matchAll(/<thought>([\s\S]*?)(?:<\/thought>|$)/g)];
    if (thoughtMatches.length) res.thought = thoughtMatches[thoughtMatches.length - 1][1].trim();

    // Extract tool calls
    const toolCallMatches = [...text.matchAll(/<tool_call>([\s\S]*?)(?:<\/tool_call>|$)/g)];
    if (toolCallMatches.length) {
        toolCallMatches.forEach((toolCallMatch) => {
            const tc = toolCallMatch[1];
            const fnMatch = tc.match(/<function=([^>]+)>([\s\S]*?)(?:<\/function>|$)/);
            if (!fnMatch) return;
            const actionName = fnMatch[1].trim();
            const paramsStr = fnMatch[2];
            const actionInput = {};
            const paramRegex = /<parameter=([^>]+)>([\s\S]*?)<\/parameter>/g;
            let m;
            while ((m = paramRegex.exec(paramsStr)) !== null) {
                let val = m[2].trim();
                try { val = JSON.parse(val); } catch(e){}
                actionInput[m[1].trim()] = val;
            }
            res.actions.push({ name: actionName, input: actionInput });
        });
        if (res.actions.length) {
            res.action = res.actions[0].name;
            res.action_input = res.actions[0].input;
        }
        return res;
    }

    // If no tool call, everything after </thought> is final answer
    const finalSplit = text.split(/<\/thought>/);
    if (finalSplit.length > 1) {
        res.final_answer = finalSplit.slice(1).join("</thought>").trim();
    } else if (!thoughtMatches.length && text.trim()) {
         // Fallback if model ignored tags completely
         res.final_answer = text.trim();
    }

    return res;
}

function renderMarkdown(text) {
    if (!text) return "";
    return text
        .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
        .replace(/&lt;thought&gt;([\s\S]*?)&lt;\/thought&gt;/g, '<details class="af-thought-details"><summary>Thinking process</summary><div class="af-thought-inner">$1</div></details>')
        .replace(/(https?:\/\/[^\s&<"']+)/g, '<a href="$1" target="_blank" rel="noopener noreferrer" style="color:#a78bfa;text-decoration:underline;">$1</a>')
        .replace(/```(\w*)\n([\s\S]*?)```/g, '<div style="position:relative"><button onclick="navigator.clipboard.writeText(this.nextElementSibling.innerText)" style="position:absolute;top:8px;right:8px;background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);color:#fff;border-radius:4px;padding:4px 8px;font-size:10px;cursor:pointer;z-index:10;transition:all 0.2s" onmousedown="this.innerText=\'Copied!\';setTimeout(()=>this.innerText=\'Copy\',2000)">Copy</button><pre class="md-code" style="margin-top:0;"><code>$2</code></pre></div>')
        .replace(/`([^`]+)`/g, '<code class="md-inline">$1</code>')
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.+?)\*/g, "<em>$1</em>")
        .replace(/^### (.+)$/gm, '<div class="md-h3">$1</div>')
        .replace(/^## (.+)$/gm, '<div class="md-h2">$1</div>')
        .replace(/^# (.+)$/gm, '<div class="md-h1">$1</div>')
        .replace(/^[-*] (.+)$/gm, '<div class="md-li">• $1</div>')
        .replace(/\n/g, "<br/>");
}

function DocCodeBlock({ code, language = "text" }) {
    return (
        <div style={{ position: "relative" }}>
            <button
                onClick={(e) => {
                    navigator.clipboard?.writeText(code);
                    e.currentTarget.innerText = "Copied!";
                    setTimeout(() => { e.currentTarget.innerText = "Copy"; }, 2000);
                }}
                style={{
                    position: "absolute",
                    top: 8,
                    right: 8,
                    background: "rgba(255,255,255,0.1)",
                    border: "1px solid rgba(255,255,255,0.2)",
                    color: "#fff",
                    borderRadius: 4,
                    padding: "4px 8px",
                    fontSize: 10,
                    cursor: "pointer",
                    zIndex: 10,
                }}
            >
                Copy
            </button>
            <pre className="md-code" style={{ marginTop: 0 }}>
                <code data-language={language}>{code}</code>
            </pre>
        </div>
    );
}

function MessageContent({ text }) {
    if (!text) return null;
    const extractHtmlBlock = (value) => {
        if (typeof value !== "string") return null;
        const match = value.match(/```html\n([\s\S]*?)\n```/i);
        return match ? match[1] : null;
    };

    const uiCode = extractHtmlBlock(text);
    const cleanedText = uiCode ? text.replace(/```html[\s\S]*?```/gi, "[UI code generated]") : text;
    const isLong = typeof text === "string" && text.length > MAX_RENDER_CHARS;
    if (!isLong) {
        return (
            <div className="af-msg-answer text">
                <div dangerouslySetInnerHTML={{ __html: renderMarkdown(cleanedText) }} />
                {uiCode && (
                    <div className="af-preview">
                        <div className="af-preview__bar">Live Preview</div>
                        <iframe
                            title="AI Code Preview"
                            sandbox="allow-scripts"
                            srcDoc={uiCode}
                            className="af-preview__frame"
                        />
                    </div>
                )}
            </div>
        );
    }
    const preview = text.slice(0, MAX_RENDER_CHARS);
    return (
        <div className="af-msg-answer text">
            <div dangerouslySetInnerHTML={{ __html: renderMarkdown(preview + " …") }} />
            <details className="af-long-msg">
            <summary>Show full message ({text.length.toLocaleString()} chars)</summary>
            <div dangerouslySetInnerHTML={{ __html: renderMarkdown(cleanedText) }} />
            {uiCode && (
                <div className="af-preview">
                    <div className="af-preview__bar">Live Preview</div>
                    <iframe
                        title="AI Code Preview"
                        sandbox="allow-scripts"
                        srcDoc={uiCode}
                        className="af-preview__frame"
                    />
                </div>
            )}
            </details>
        </div>
    );
}

// ====================== MAIN COMPONENT ======================
export default function AgentFramework() {
    const initialConversationRef = useRef(null);
    if (initialConversationRef.current === null) {
        initialConversationRef.current = createConversation();
    }
    const [settingsOpen, setSettingsOpen] = useState(false);
    const [multiThink, setMultiThink] = useState(false);
    const [doublePass, setDoublePass] = useState(false);
    const [headerOpen, setHeaderOpen] = useState(false);
    const [thinkAloud, setThinkAloud] = useState(false);
    const [drawerOpen, setDrawerOpen] = useState(false);
    const [uploadStatus, setUploadStatus] = useState("");
    const [activeTab, setActiveTab] = useState("chat");
    const [isCompactUi, setIsCompactUi] = useState(detectCompactUi);
    const [isAndroidUi, setIsAndroidUi] = useState(detectAndroidUi);
    const [keyboardOpen, setKeyboardOpen] = useState(false);
    const [sidebarOpen, setSidebarOpen] = useState(() => {
        if (typeof window === "undefined") return true;
        return window.innerWidth > AUTO_SIDEBAR_BREAKPOINT;
    });
    const [logCopyStatus, setLogCopyStatus] = useState("Copy");
    const [primaryModel, setPrimaryModel] = useState(() => AVAILABLE_MODELS.find(m => m.id === localStorage.getItem("primary_model")) || AVAILABLE_MODELS[0]);
    const [fallbackModels, setFallbackModels] = useState(() => {
        const s = localStorage.getItem("fallback_models");
        if (!s) return [AVAILABLE_MODELS[1] || AVAILABLE_MODELS[0]];
        return JSON.parse(s).map(id => AVAILABLE_MODELS.find(m => m.id === id)).filter(Boolean);
    });
    const [conversations, setConversations] = useState(() => {
        const s = localStorage.getItem("agent_convos_v4");
        if (s) return JSON.parse(s).map(hydrateConversation);
        return [initialConversationRef.current];
    });
    const [currentConversationId, setCurrentConversationId] = useState(() => {
        const s = localStorage.getItem("agent_convos_v4");
        return s ? JSON.parse(s)[0]?.id : initialConversationRef.current.id;
    });
    const [running, setRunning] = useState(false);
    const [preparingSend, setPreparingSend] = useState(false);
    const [expandedSteps, setExpandedSteps] = useState({});
    const conversationsRef = useRef(conversations);
    const initialVhRef = useRef(null);
    const wideLayoutRef = useRef(typeof window !== "undefined" ? window.innerWidth > AUTO_SIDEBAR_BREAKPOINT : true);
    const bottomRef = useRef(null);
    const composerRef = useRef(null);
    const inputRef = useRef(null);
    const abortRef = useRef(null);
    const fileInputRef = useRef(null);
    const settingsFlyRef = useRef(null);
    const settingsBtnRef = useRef(null);
    const ocrWorkerRef = useRef(null);
    const ocrWorkerPromiseRef = useRef(null);
    const ocrQueueRef = useRef(Promise.resolve());
    const ocrJobsRef = useRef(new Map());
    const ocrLoggerRef = useRef(() => {});

    // Track both the stable viewport and the visual viewport so Android keyboards do not cover the composer.
    useEffect(() => {
        const syncViewport = () => {
            if (typeof window === "undefined") return;
            if (initialVhRef.current === null) initialVhRef.current = window.innerHeight;
            const viewport = window.visualViewport;
            const visualHeight = viewport?.height || window.innerHeight;
            const stableHeight = Math.max(initialVhRef.current, visualHeight);
            if (stableHeight > initialVhRef.current) initialVhRef.current = stableHeight;
            const keyboardDelta = Math.round(stableHeight - visualHeight - (viewport?.offsetTop || 0));
            const keyboardOffset = keyboardDelta > 140 ? keyboardDelta : 0;
            document.documentElement.style.setProperty("--vh", `${stableHeight * 0.01}px`);
            document.documentElement.style.setProperty("--shell-vh", `${visualHeight * 0.01}px`);
            document.documentElement.style.setProperty("--keyboard-offset", `${keyboardOffset}px`);
            setKeyboardOpen(keyboardOffset > 0);
            setIsCompactUi(detectCompactUi());
            setIsAndroidUi(detectAndroidUi());
        };
        syncViewport();
        window.addEventListener("resize", syncViewport);
        window.addEventListener("orientationchange", syncViewport);
        window.visualViewport?.addEventListener("resize", syncViewport);
        window.visualViewport?.addEventListener("scroll", syncViewport);
        return () => {
            window.removeEventListener("resize", syncViewport);
            window.removeEventListener("orientationchange", syncViewport);
            window.visualViewport?.removeEventListener("resize", syncViewport);
            window.visualViewport?.removeEventListener("scroll", syncViewport);
        };
    }, []);
    useEffect(() => {
        conversationsRef.current = conversations;
    }, [conversations]);
    useEffect(() => {
        if (typeof window === "undefined") return;
        localStorage.removeItem("agent_api_keys_pool");
    }, []);
    useEffect(() => () => {
        const worker = ocrWorkerRef.current;
        ocrWorkerRef.current = null;
        ocrWorkerPromiseRef.current = null;
        ocrLoggerRef.current = () => {};
        if (worker?.terminate) {
            worker.terminate().catch(() => {});
        }
    }, []);
    const handleFileUpload = async (e) => {
        const files = Array.from(e.target.files || []);
        e.target.value = null;
        if (!files.length) return;
        const conversationId = currentConversationId;
        const existingUploads = Array.isArray(conv?.pendingUploads) ? conv.pendingUploads : [];
        const remainingSlots = Math.max(0, MAX_UPLOAD_FILES - existingUploads.length);
        if (remainingSlots <= 0) {
            setUploadStatus(`Attachment limit reached (${MAX_UPLOAD_FILES}). Remove one to add another.`);
            setTimeout(() => setUploadStatus(""), 3200);
            return;
        }

        const selected = files.slice(0, remainingSlots);
        setUploadStatus(`Preparing ${selected.length} attachment${selected.length > 1 ? "s" : ""}...`);

        let totalChars = existingUploads
            .filter(upload => upload?.kind === "text")
            .reduce((sum, upload) => sum + (upload?.textContent?.length || 0), 0);
        let truncatedCount = 0;
        let skippedCount = Math.max(0, files.length - selected.length);
        let unreadableCount = 0;
        let oversizedCount = 0;
        const nextUploads = [];
        const imageUploads = [];

        for (const file of selected) {
            if (isImageFile(file)) {
                if ((file?.size || 0) > MAX_IMAGE_UPLOAD_BYTES) {
                    oversizedCount += 1;
                    skippedCount += 1;
                    continue;
                }
                try {
                    const dataUrl = await readFileAsDataUrl(file);
                    if (!dataUrl) {
                        unreadableCount += 1;
                        continue;
                    }
                    const imageUpload = createPendingUpload({
                        name: file.name,
                        size: file.size,
                        kind: "image",
                        mimeType: file.type,
                        dataUrl,
                        ocrStatus: "queued",
                    });
                    nextUploads.push(imageUpload);
                    imageUploads.push(imageUpload);
                } catch {
                    unreadableCount += 1;
                }
                continue;
            }

            if (!isProbablyTextFile(file)) {
                skippedCount += 1;
                continue;
            }

            try {
                let text = normalizeTextBlock(await file.text());
                if (!text) {
                    skippedCount += 1;
                    continue;
                }

                const remainingBudget = MAX_TOTAL_UPLOAD_CHARS - totalChars;
                if (remainingBudget <= 0) {
                    skippedCount += 1;
                    continue;
                }

                let wasTruncated = false;
                if (text.length > MAX_FILE_INSERT_CHARS) {
                    text = truncateText(text, MAX_FILE_INSERT_CHARS);
                    wasTruncated = true;
                }
                if (text.length > remainingBudget) {
                    text = truncateText(text, remainingBudget);
                    wasTruncated = true;
                }
                if (!text.trim()) {
                    skippedCount += 1;
                    continue;
                }

                totalChars += text.length;
                if (wasTruncated) truncatedCount += 1;
                nextUploads.push(createPendingUpload({
                    name: file.name,
                    size: file.size,
                    kind: "text",
                    mimeType: file.type,
                    textContent: text,
                    truncated: wasTruncated,
                }));
            } catch {
                unreadableCount += 1;
            }
        }

        if (!nextUploads.length) {
            const reason = unreadableCount
                ? "Selected files could not be read."
                : skippedCount
                    ? "No supported files were added."
                    : "No readable content found.";
            setUploadStatus(`Upload error: ${reason}`);
            setTimeout(() => setUploadStatus(""), 3200);
            return;
        }

        updateConv(c => ({
            ...c,
            pendingUploads: [...(Array.isArray(c.pendingUploads) ? c.pendingUploads : []), ...nextUploads],
        }));
        imageUploads.forEach(upload => {
            enqueueOcrForUpload(conversationId, upload).catch(() => {});
        });

        const notes = [];
        if (truncatedCount) notes.push(`${truncatedCount} truncated`);
        if (imageUploads.length) notes.push(`${imageUploads.length} OCR`);
        if (oversizedCount) notes.push(`${oversizedCount} too large`);
        if (skippedCount) notes.push(`${skippedCount} skipped`);
        if (unreadableCount) notes.push(`${unreadableCount} unreadable`);
        setUploadStatus(
            `Attached ${nextUploads.length} file${nextUploads.length > 1 ? "s" : ""}${notes.length ? ` (${notes.join(", ")})` : ""}.`
        );
        setTimeout(() => setUploadStatus(""), 3200);
    };

    const removePendingUpload = (uploadId) => {
        updateConv(c => ({
            ...c,
            pendingUploads: (Array.isArray(c.pendingUploads) ? c.pendingUploads : []).filter(upload => upload?.id !== uploadId),
        }));
    };

    useEffect(() => localStorage.setItem("primary_model", primaryModel.id), [primaryModel]);
    useEffect(() => localStorage.setItem("fallback_models", JSON.stringify(fallbackModels.map(m => m.id))), [fallbackModels]);
    useEffect(() => localStorage.setItem("agent_convos_v4", JSON.stringify(conversations.map(serializeConversation))), [conversations]);
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [currentConversationId, conversations.length, running, activeTab]);
    useEffect(() => {
        setFallbackModels(prev => {
            const next = prev.filter((model, index, models) => (
                model &&
                model.id !== primaryModel.id &&
                models.findIndex(candidate => candidate?.id === model.id) === index
            ));
            return next.length === prev.length && next.every((model, index) => model.id === prev[index]?.id) ? prev : next;
        });
    }, [primaryModel.id]);
    useEffect(() => {
        if (typeof window === "undefined") return;
        const onResize = () => {
            const isWideLayout = window.innerWidth > AUTO_SIDEBAR_BREAKPOINT;
            if (isWideLayout !== wideLayoutRef.current) {
                setSidebarOpen(isWideLayout);
                wideLayoutRef.current = isWideLayout;
            }
            setIsCompactUi(detectCompactUi());
        };
        window.addEventListener("resize", onResize);
        return () => window.removeEventListener("resize", onResize);
    }, []);
    useEffect(() => {
        if (!settingsOpen) return;
        const onClick = (e) => {
            if (settingsFlyRef.current && settingsBtnRef.current) {
                if (!settingsFlyRef.current.contains(e.target) && !settingsBtnRef.current.contains(e.target)) {
                    setSettingsOpen(false);
                }
            }
        };
        document.addEventListener("mousedown", onClick);
        return () => document.removeEventListener("mousedown", onClick);
    }, [settingsOpen]);
    useEffect(() => {
        if (!headerOpen && settingsOpen) {
            setSettingsOpen(false);
        }
    }, [headerOpen, settingsOpen]);
    useEffect(() => {
        const header = document.getElementById("agentHeader");
        const toggle = document.getElementById("toggleHeader");
        if (!header || !toggle) return undefined;

        const handleToggle = () => {
            setHeaderOpen((current) => {
                const next = !current;
                header.classList.toggle("open", next);
                toggle.textContent = next ? "↑" : "↓";
                return next;
            });
        };

        toggle.addEventListener("click", handleToggle);
        return () => toggle.removeEventListener("click", handleToggle);
    }, []);

    const conv = conversations.find(c => c.id === currentConversationId);
    useEffect(() => {
        if (!conv && conversations[0]) setCurrentConversationId(conversations[0].id);
    }, [conv, conversations]);
    useEffect(() => {
        if (activeTab !== "chat") return;
        const textarea = inputRef.current;
        if (!textarea) return;
        textarea.style.height = "0px";
        const maxHeight = isCompactUi
            ? Math.max(110, Math.round((window.visualViewport?.height || window.innerHeight) * 0.22))
            : 110;
        const nextHeight = Math.max(36, Math.min(textarea.scrollHeight, maxHeight));
        textarea.style.height = `${nextHeight}px`;
        textarea.style.overflowY = textarea.scrollHeight > maxHeight ? "auto" : "hidden";
    }, [activeTab, conv?.currentInput, currentConversationId, isCompactUi]);
    useEffect(() => {
        if (!keyboardOpen || !isCompactUi || activeTab !== "chat") return;
        const timer = window.setTimeout(() => {
            composerRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
            bottomRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
        }, 120);
        return () => window.clearTimeout(timer);
    }, [activeTab, currentConversationId, isCompactUi, keyboardOpen]);
    const getConversationById = (conversationId) => conversationsRef.current.find(entry => entry.id === conversationId);
    const updateConversationById = (conversationId, fn) => setConversations(prev => prev.map(c => c.id === conversationId ? fn(c) : c));
    const updateUploadInConversation = (conversationId, uploadId, patch) => updateConversationById(conversationId, c => ({
        ...c,
        pendingUploads: (Array.isArray(c.pendingUploads) ? c.pendingUploads : []).map(upload => {
            if (upload?.id !== uploadId) return upload;
            const nextPatch = typeof patch === "function" ? patch(upload) : patch;
            return nextPatch ? { ...upload, ...nextPatch } : upload;
        }),
    }));
    const getOcrJobKey = (conversationId, uploadId) => `${conversationId}:${uploadId}`;
    const ensureOcrWorker = async () => {
        if (ocrWorkerRef.current) return ocrWorkerRef.current;
        if (!ocrWorkerPromiseRef.current) {
            ocrWorkerPromiseRef.current = (async () => {
                const { createWorker } = await import("tesseract.js");
                const worker = await createWorker(OCR_LANGUAGE, 1, {
                    logger: (message) => ocrLoggerRef.current?.(message),
                });
                ocrWorkerRef.current = worker;
                return worker;
            })().catch((error) => {
                ocrWorkerPromiseRef.current = null;
                throw error;
            });
        }
        return ocrWorkerPromiseRef.current;
    };
    const enqueueOcrForUpload = (conversationId, attachment) => {
        if (!attachment?.id || !attachment?.dataUrl) return Promise.resolve();
        const jobKey = getOcrJobKey(conversationId, attachment.id);
        const runJob = async () => {
            updateUploadInConversation(conversationId, attachment.id, {
                ocrStatus: "running",
                ocrProgress: 0.02,
                ocrText: "",
                ocrError: "",
            });
            try {
                ocrLoggerRef.current = (message) => {
                    const nextProgress = Number.isFinite(Number(message?.progress)) ? Number(message.progress) : 0;
                    if (!nextProgress) return;
                    updateUploadInConversation(conversationId, attachment.id, (upload) => ({
                        ocrStatus: "running",
                        ocrProgress: Math.max(Number(upload?.ocrProgress) || 0, nextProgress),
                    }));
                };
                const worker = await ensureOcrWorker();
                const result = await worker.recognize(attachment.dataUrl, { rotateAuto: true });
                const ocrText = normalizeTextBlock(result?.data?.text || "");
                updateUploadInConversation(conversationId, attachment.id, {
                    ocrStatus: ocrText ? "done" : "empty",
                    ocrProgress: 1,
                    ocrText,
                    ocrError: "",
                });
            } catch (error) {
                updateUploadInConversation(conversationId, attachment.id, {
                    ocrStatus: "error",
                    ocrProgress: 0,
                    ocrText: "",
                    ocrError: error?.message || "OCR failed.",
                });
            } finally {
                ocrLoggerRef.current = () => {};
            }
        };
        const job = ocrQueueRef.current
            .catch(() => {})
            .then(runJob)
            .finally(() => {
                if (ocrJobsRef.current.get(jobKey) === job) {
                    ocrJobsRef.current.delete(jobKey);
                }
            });
        ocrQueueRef.current = job;
        ocrJobsRef.current.set(jobKey, job);
        return job;
    };
    const waitForPendingOcr = async (conversationId, attachments) => {
        const jobs = (Array.isArray(attachments) ? attachments : [])
            .filter(isOcrPendingUpload)
            .map(attachment => ocrJobsRef.current.get(getOcrJobKey(conversationId, attachment.id)))
            .filter(Boolean);
        if (!jobs.length) return;
        setPreparingSend(true);
        setUploadStatus(`Waiting for OCR on ${jobs.length} image${jobs.length > 1 ? "s" : ""}...`);
        try {
            await Promise.allSettled(jobs);
        } finally {
            setPreparingSend(false);
            setUploadStatus("");
        }
    };
    const memoryStore = conv?.memoryStore || {};
    const memoryEngine = hydrateMemoryEngine(conv?.memoryEngine);
    const strategyState = conv?.strategyState || createStrategyState();
    const updateMemory = (fn) => updateConv(c => ({ ...c, memoryStore: typeof fn === "function" ? fn(c.memoryStore) : fn }));
    const tools = createTools(memoryStore, updateMemory);
    const toolList = Object.entries(tools);

    const handlePrimaryModelChange = (modelId) => {
        const nextPrimary = AVAILABLE_MODELS.find(model => model.id === modelId);
        if (!nextPrimary || nextPrimary.id === primaryModel.id) return;
        setPrimaryModel(nextPrimary);
        setFallbackModels(prev => {
            const next = [primaryModel, ...prev].filter((model, index, models) => (
                model &&
                model.id !== nextPrimary.id &&
                models.findIndex(candidate => candidate?.id === model.id) === index
            ));
            return next;
        });
    };

    const toggleFallbackModel = (model) => {
        if (!model || model.id === primaryModel.id) return;
        setFallbackModels(prev => {
            const exists = prev.some(entry => entry.id === model.id);
            if (exists) return prev.filter(entry => entry.id !== model.id);
            return [...prev, model].filter((entry, index, models) => models.findIndex(candidate => candidate.id === entry.id) === index);
        });
    };

    const updateConv = (fn) => updateConversationById(currentConversationId, fn);
    const buildToolSteps = (items = [], prefix = "") => (
        (Array.isArray(items) ? items : []).map((entry, index) => {
            let input = entry?.args;
            if (typeof input === "string") {
                try { input = JSON.parse(input); } catch { /* keep raw */ }
            }
            return {
                type: "action",
                action: `${prefix}${entry?.name || "tool"}`,
                input: input || {},
                iteration: index + 1,
                batchSize: items.length,
            };
        })
    );
    useEffect(() => {
        if (!conv) return;
        const nextSummary = buildEpisodicSummary(conv.messages || []);
        if (nextSummary === memoryEngine.episodicSummary) return;
        updateConversationById(conv.id, (currentConversation) => {
            const currentEngine = hydrateMemoryEngine(currentConversation.memoryEngine);
            if (currentEngine.episodicSummary === nextSummary) return currentConversation;
            return {
                ...currentConversation,
                memoryEngine: hydrateMemoryEngine({
                    ...currentEngine,
                    episodicSummary: nextSummary,
                }),
            };
        });
    }, [conv?.id, conv?.messages, memoryEngine.episodicSummary]);
    const newConv = () => {
        const next = createConversation();
        setConversations(prev => [...prev, next]);
        setCurrentConversationId(next.id);
        setExpandedSteps({});
    };
    const delConv = (id) => {
        if (conversations.length <= 1) return;
        setConversations(prev => {
            const next = prev.filter(c => c.id !== id);
            if (currentConversationId === id && next[0]) setCurrentConversationId(next[0].id);
            return next;
        });
    };
    const clearConv = () => updateConv(c => ({
        ...c,
        title: formatConversationTitle(c.createdAt),
        messages: [],
        pendingSteps: null,
        activeStream: null,
        systemLogs: [],
        currentInput: "",
        pendingUploads: [],
        memoryStore: {},
        memoryEngine: createMemoryEngine(),
        strategyState: createStrategyState(),
    }));
    const exportConv = () => { if (!conv) return; const b = new Blob([JSON.stringify(serializeConversation(conv), null, 2)], { type: "application/json" }); const a = document.createElement("a"); a.href = URL.createObjectURL(b); a.download = `chat_${conv.id}.json`; a.click(); };

    const runAgent = async () => {
        const query = normalizeTextBlock(conv?.currentInput || "");
        let pendingUploads = Array.isArray(conv?.pendingUploads) ? conv.pendingUploads.map(createPendingUpload) : [];
        if ((!query && !pendingUploads.length) || running || preparingSend) return;

        const conversationId = currentConversationId;
        const updateRunConversation = (fn) => updateConversationById(conversationId, fn);
        const pushRunLog = (module, msg, color = "var(--success)") => {
            const ts = new Date().toLocaleTimeString('en-US', { hour12: false });
            updateRunConversation(c => ({ ...c, systemLogs: [...(c.systemLogs || []), { ts, module, msg, color }] }));
        };
        const updateRunStrategy = (patch) => updateRunConversation(c => {
            const current = hydrateStrategyState(c.strategyState);
            const nextPatch = typeof patch === "function" ? patch(current) : patch;
            return {
                ...c,
                strategyState: {
                    ...current,
                    ...nextPatch,
                    metrics: {
                        ...current.metrics,
                        ...(nextPatch?.metrics || {}),
                    },
                },
            };
        });

        await waitForPendingOcr(conversationId, pendingUploads);
        pendingUploads = Array.isArray(getConversationById(conversationId)?.pendingUploads)
            ? getConversationById(conversationId).pendingUploads.map(createPendingUpload)
            : pendingUploads;
        if ((!query && !pendingUploads.length) || running) return;

        let runMemoryEngine = hydrateMemoryEngine(getConversationById(conversationId)?.memoryEngine || conv?.memoryEngine);
        const persistRunMemoryEngine = (patch) => {
            const nextPatch = typeof patch === "function" ? patch(runMemoryEngine) : patch;
            runMemoryEngine = hydrateMemoryEngine({
                ...runMemoryEngine,
                ...(nextPatch || {}),
            });
            updateRunConversation(c => ({
                ...c,
                memoryEngine: runMemoryEngine,
            }));
            return runMemoryEngine;
        };
        const persistAssistantMessage = (assistantMessageFactory) => {
            updateRunConversation((conversation) => {
                const assistantMessage = typeof assistantMessageFactory === "function"
                    ? assistantMessageFactory(conversation)
                    : assistantMessageFactory;
                const nextMessages = [...(Array.isArray(conversation.messages) ? conversation.messages : []), assistantMessage];
                const nextEngine = hydrateMemoryEngine({
                    ...runMemoryEngine,
                    episodicSummary: buildEpisodicSummary(nextMessages),
                });
                runMemoryEngine = nextEngine;
                return {
                    ...conversation,
                    activeStream: null,
                    pendingSteps: null,
                    messages: nextMessages,
                    memoryEngine: nextEngine,
                };
            });
        };

        const currentAttachmentChunks = buildSemanticChunks(buildAttachmentSemanticDocs(pendingUploads), { sourceType: "attachment" });
        if (currentAttachmentChunks.length) {
            runMemoryEngine = hydrateMemoryEngine({
                ...runMemoryEngine,
                semanticChunks: mergeSemanticChunks(runMemoryEngine.semanticChunks, currentAttachmentChunks),
            });
        }

        const semanticSearchText = query || pendingUploads.map(upload => upload.name).filter(Boolean).join(" ") || "attached files";
        const semanticRecall = buildSemanticContext(semanticSearchText, runMemoryEngine.semanticChunks);
        const rawUserHistoryText = buildUserHistoryText(query, pendingUploads, {
            includeTextBlocks: false,
            includeImageOcr: false,
            extraContext: semanticRecall.text,
        });
        const preparedQuery = truncateText(rawUserHistoryText, MAX_CURRENT_QUERY_CHARS);
        const queryWasTrimmed = preparedQuery !== rawUserHistoryText;
        const modelUserContent = buildUserModelContent(query, pendingUploads, {
            includeTextBlocks: false,
            includeImageOcr: false,
            extraContext: semanticRecall.text,
        });
        const messageAttachments = pendingUploads.map(createMessageAttachment);
        const titleSource = query || pendingUploads.map(upload => upload.name).join(", ");
        const priorMessages = Array.isArray(conv?.messages) ? conv.messages : [];
        const priorRelevantMessages = priorMessages.filter(message => (
            message &&
            message.role !== "system" &&
            (messageHasContent(message) || message.error || (Array.isArray(message.attachments) && message.attachments.length))
        ));
        const workingMemoryMessages = buildWorkingMemoryMessages(priorMessages);
        const workingMemoryTrimmed = priorRelevantMessages.length > workingMemoryMessages.length;
        const episodicSummary = buildEpisodicSummary(priorMessages);
        const historyMessages = workingMemoryMessages;
        const isModelQuestion = /\b(what|which)\s+model\b|model is this|what are you|who are you|identify yourself|which ai|what ai/i.test(query.toLowerCase());

        if (isModelQuestion && !pendingUploads.length) {
            const isFirst = priorMessages.length === 0;
            const assistantReply = "I'm nub-agent (ambitiousnoob) running on Cerebras Qwen 3 235B via Cerebras Inference.";
            updateRunConversation(c => {
                const nextMessages = [...c.messages,
                    { role: "user", content: preparedQuery, displayText: query, attachments: messageAttachments },
                    { role: "assistant", content: assistantReply },
                ];
                const nextEngine = hydrateMemoryEngine({
                    ...runMemoryEngine,
                    episodicSummary: buildEpisodicSummary(nextMessages),
                });
                return {
                    ...c,
                    currentInput: "",
                    pendingUploads: [],
                    messages: nextMessages,
                    activeStream: null,
                    pendingSteps: [],
                    memoryEngine: nextEngine,
                    strategyState: createStrategyState(),
                    ...(isFirst ? { title: titleSource.slice(0, 40) + (titleSource.length > 40 ? "..." : "") } : {}),
                };
            });
            return;
        }

        setRunning(true);
        setExpandedSteps({});
        const isFirst = priorMessages.length === 0;
        updateRunConversation(c => ({
            ...c,
            currentInput: "",
            pendingUploads: [],
            messages: [...c.messages, {
                role: "user",
                content: preparedQuery,
                displayText: query,
                attachments: messageAttachments,
            }],
            pendingSteps: [],
            activeStream: "",
            memoryEngine: runMemoryEngine,
            strategyState: createStrategyState(),
            ...(isFirst ? { title: titleSource.slice(0, 40) + (titleSource.length > 40 ? "..." : "") } : {}),
        }));
        abortRef.current = new AbortController();
        const signal = abortRef.current.signal;

        const logPrefix = priorMessages.length === 0 ? "Starting" : "Continuing";
        pushRunLog("Manager", `${logPrefix} — nub-agent single-call mode`, "var(--text-dim)");
        if (queryWasTrimmed) {
            pushRunLog("Tokens", "Current input or attachment context was compacted before dispatch.", "var(--danger)");
        }
        if (workingMemoryTrimmed || episodicSummary) {
            pushRunLog("Memory", "Older turns were collapsed into episodic summary + sliding working memory.", "var(--text-dim)");
        }
        if (semanticRecall.hits.length) {
            pushRunLog("Memory", `Semantic recall injected ${semanticRecall.hits.length} indexed chunk(s) into the latest prompt.`, "var(--text-dim)");
        }

        pushRunLog("System", "Agentic tools enabled; backend may call calculate/web_search/web_fetch.", "var(--text-dim)");

        try {
            const thinkPrompt = thinkAloud ? [{ role: "system", content: "Think step by step. Show a concise <thought> plan before answering." }] : [];
            const chatMessages = [...historyMessages, ...thinkPrompt, { role: "user", content: preparedQuery }];

            if (!multiThink) {
                const { text: fastText, toolsUsed = [] } = await callLLMStream(chatMessages, primaryModel.id, signal, (text) => {
                    updateRunConversation(c => ({ ...c, activeStream: text }));
                }, {
                    maxTokens: MAX_COMPLETION_TOKENS,
                    log: (msg) => pushRunLog("System", msg, "var(--text-dim)"),
                    onToolCall: (name, args) => pushRunLog("Tool", `Called ${name} with ${JSON.stringify(args)}`, "var(--accent)"),
                });

                if (Array.isArray(toolsUsed) && toolsUsed.length) {
                    toolsUsed.forEach((entry) => {
                        pushRunLog("Tool", `Called ${entry?.name || "tool"} with ${entry?.args || "{}"}`, "var(--accent)");
                    });
                }

                if (!doublePass) {
                    updateRunConversation(c => ({
                        ...c,
                        messages: [...c.messages, { role: "assistant", content: fastText, steps: buildToolSteps(toolsUsed) }],
                        activeStream: "",
                        pendingSteps: [],
                    }));
                } else {
                    // Second pass refinement
                    pushRunLog("System", "Two-pass reasoning: generating refinement draft.", "var(--text-dim)");
                    const reviewMessages = [
                        ...chatMessages,
                        { role: "assistant", content: fastText },
                        { role: "system", content: "You are a meticulous reviewer. Improve the assistant's previous answer. Fix mistakes, add missing details, and keep it concise." },
                        { role: "user", content: "Refine the previous assistant reply. Return the improved final answer only." },
                    ];
                    const { text: refinedText, toolsUsed: refineTools = [] } = await callLLMStream(reviewMessages, primaryModel.id, signal, () => {}, {
                        maxTokens: MAX_COMPLETION_TOKENS,
                        log: (msg) => pushRunLog("System", `[Refine] ${msg}`, "var(--text-dim)"),
                        onToolCall: (name, args) => pushRunLog("Tool", `[Refine] Called ${name} with ${JSON.stringify(args)}`, "var(--accent)"),
                    });
                    if (Array.isArray(refineTools) && refineTools.length) {
                        refineTools.forEach(entry => pushRunLog("Tool", `[Refine] Called ${entry?.name || "tool"} with ${entry?.args || "{}"}`, "var(--accent)"));
                    }
                    const combined = `### Pass 1\n${fastText}\n\n### Pass 2 (refined)\n${refinedText || ""}`;
                    const combinedSteps = [
                        ...buildToolSteps(toolsUsed),
                        ...buildToolSteps(refineTools, "refine: "),
                    ];
                    updateRunConversation(c => ({
                        ...c,
                        messages: [...c.messages, { role: "assistant", content: combined, steps: combinedSteps }],
                        activeStream: "",
                        pendingSteps: [],
                    }));
                }
            } else {
                pushRunLog("System", "Multi-process thinking enabled: launching two parallel drafts.", "var(--text-dim)");
                const thinkerLabels = ["Alpha", "Beta"];
                const thinkerPromises = thinkerLabels.map(label => {
                    const thinkerMessages = [
                        BRAND_SYSTEM_MESSAGE,
                        { role: "system", content: `You are thinker ${label}. Produce an independent best-possible answer. Do not mention other thinkers.` },
                        ...chatMessages,
                    ];
                    return callLLMStream(thinkerMessages, primaryModel.id, signal, () => {}, {
                        maxTokens: MAX_COMPLETION_TOKENS,
                        log: (msg) => pushRunLog(`System`, `[${label}] ${msg}`, "var(--text-dim)"),
                        onToolCall: (name, args) => pushRunLog("Tool", `[${label}] Called ${name} with ${JSON.stringify(args)}`, "var(--accent)"),
                    }).then(res => ({ label, ...res }));
                });

                const drafts = await Promise.all(thinkerPromises);
                const combined = drafts
                    .map(draft => `## ${draft.label}\n${draft.text || draft.content || ""}`.trim())
                    .join("\n\n");
                const combinedSteps = drafts.flatMap((draft) => buildToolSteps(draft.toolsUsed, `${draft.label}: `));

                drafts.forEach(draft => {
                    if (Array.isArray(draft.toolsUsed) && draft.toolsUsed.length) {
                        draft.toolsUsed.forEach(entry => {
                            pushRunLog("Tool", `[${draft.label}] Called ${entry?.name || "tool"} with ${entry?.args || "{}"}`, "var(--accent)");
                        });
                    }
                });

                updateRunConversation(c => ({
                    ...c,
                    messages: [...c.messages, { role: "assistant", content: combined, steps: combinedSteps }],
                    activeStream: "",
                    pendingSteps: [],
                }));
            }
        } catch (e) {
            updateRunConversation(c => ({
                ...c,
                messages: [...c.messages, { role: "assistant", content: null, error: e.message === "aborted" ? "Stopped by user." : e.message, steps: [] }],
                activeStream: "",
                pendingSteps: [],
            }));
        } finally {
            setRunning(false);
            abortRef.current = null;
            inputRef.current?.focus();
        }
    };

    const handleKey = (e) => {
        if (e.nativeEvent?.isComposing) return;
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            runAgent();
        }
    };
    useEffect(() => {
        const h = (e) => { if ((e.ctrlKey||e.metaKey) && e.key==="k") { e.preventDefault(); inputRef.current?.focus(); } if (e.key==="Escape" && running) abortRef.current?.abort(); };
        window.addEventListener("keydown", h); return () => window.removeEventListener("keydown", h);
    }, [running]);

    const categories = [...new Set(Object.values(tools).map(t => t.category))];
    const allMessages = conv?.messages || [];
    const pendingSteps = conv?.pendingSteps;
    const pendingUploadsView = Array.isArray(conv?.pendingUploads) ? conv.pendingUploads : [];
    const systemLogs = conv?.systemLogs || [];
    useEffect(() => { setLogCopyStatus("Copy"); }, [systemLogs.length]);
    const handleCopyLogs = () => {
        if (!systemLogs.length || typeof navigator === "undefined" || !navigator.clipboard) return;
        const text = systemLogs.map((entry) => `[${entry.ts}] ${entry.module}: ${entry.msg}`).join("\n");
        navigator.clipboard.writeText(text)
            .then(() => {
                setLogCopyStatus("Copied!");
                window.setTimeout(() => setLogCopyStatus("Copy"), 1600);
            })
            .catch(() => setLogCopyStatus("Copy failed"));
    };
    const latestAssistantConsoleMessage = [...allMessages].reverse().find(message => (
        message?.role === "assistant" && Boolean(message.error)
    ));
    const drawerCards = conv?.activeStream ? [{
        key: "drawer-live-stream",
        label: running ? "Live response" : "Last streamed response",
        body: truncateText(conv.activeStream, 4000),
    }] : [];
    const rawConsoleWarning = !running ? normalizeTextBlock(latestAssistantConsoleMessage?.error || "") : "";
    const latestConsoleWarning = rawConsoleWarning === "Iteration limit reached before a verified final answer was produced."
        ? ""
        : rawConsoleWarning;
    const drawerWarning = latestConsoleWarning || (
        running && !conv?.activeStream && !drawerCards.length
            ? `${primaryModel.label} is starting. Live agent output will appear here.`
            : ""
    );
    const drawerSummaryLabel = running ? "Agent Console - Live" : "Agent Console";
    const hasRuntimeAccess = HOSTED_API_ENABLED;
    const routerState = strategyState.router;
    const planState = strategyState.plan;
    const verificationState = strategyState.verification;
    const runMetrics = strategyState.metrics || createRunMetrics();
    const lastPromptStats = memoryEngine.lastPromptStats;
    const canSend = Boolean(normalizeTextBlock(conv?.currentInput || "") || pendingUploadsView.length);
    const siteOrigin = typeof window !== "undefined" ? window.location.origin : "";
    const apiUrl = siteOrigin ? `${siteOrigin}${HOSTED_CHAT_API_PATH}` : HOSTED_CHAT_API_PATH;
    const defaultApiModel = AVAILABLE_MODELS[0];
    const agentRequestExample = JSON.stringify({
        prompt: "Find the shortest path from San Francisco to Tokyo and explain which airport you chose.",
        stream: false,
    }, null, 2);
    const completionRequestExample = JSON.stringify({
        prompt: "Reply with OK only.",
        stream: false,
        executionMode: "completion",
    }, null, 2);
    const messagesRequestExample = JSON.stringify({
        messages: [
            {
                role: "user",
                content: [
                    { type: "text", text: "Read the text in this image and summarize it." },
                    { type: "image_url", image_url: { url: "https://example.com/receipt.png" } },
                ],
            },
        ],
        model: defaultApiModel.id,
        stream: false,
    }, null, 2);
    const agentResponseExample = JSON.stringify({
        ok: true,
        agentic: true,
        output_text: "Final answer text",
        strategy: {
            router: { primary: "web_fetch" },
            plan: { goal: "Answer the request" },
            verification: { status: "pass" },
        },
        steps: [
            {
                type: "action",
                action: "web_fetch",
                observation: "Cleaned tool output chunk 1 of 2",
            },
        ],
        choices: [
            {
                index: 0,
                role: "assistant",
                output_text: "Final answer text",
                message: { role: "assistant", content: "Final answer text" },
            },
        ],
    }, null, 2);
    const agentCurlExample = `curl -sS -X POST ${apiUrl} \\
  -H "Content-Type: application/json" \\
  -d '{"prompt":"Find the shortest path from San Francisco to Tokyo and explain which airport you chose.","stream":false}'`;
    const completionCurlExample = `curl -sS -X POST ${apiUrl} \\
  -H "Content-Type: application/json" \\
  -d '{"prompt":"Reply with OK only.","stream":false,"executionMode":"completion"}'`;
    const streamCurlExample = `curl -N -X POST ${apiUrl} \\
  -H "Content-Type: application/json" \\
  -d '{"prompt":"hello","stream":true}'`;
    const rootClassName = [
        "af-root",
        sidebarOpen ? "sidebar-open" : "",
        isCompactUi ? "is-compact" : "",
        isAndroidUi ? "is-android" : "",
        keyboardOpen ? "keyboard-open" : "",
    ].filter(Boolean).join(" ");
    const tabConfig = [
        { id: "chat", label: "💬 Chat" },
        { id: "docs", label: "📘 Docs" },
        { id: "logs", label: `📜 Logs (${systemLogs.length})` },
    ];

    return (
        <div className={rootClassName}>
            <style>{`
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--bg-deep:#0a0a0f;--bg-card:rgba(18,18,28,0.85);--bg-card-hover:rgba(28,28,42,0.9);--bg-input:rgba(22,22,35,0.9);--border:rgba(255,255,255,0.06);--border-focus:rgba(99,102,241,0.5);--text:#e4e4ed;--text-dim:#8b8b9e;--text-muted:#5a5a6e;--accent:linear-gradient(135deg,#6366f1,#8b5cf6);--accent-solid:#7c3aed;--accent-glow:rgba(99,102,241,0.15);--success:#34d399;--success-bg:rgba(52,211,153,0.1);--danger:#f87171;--danger-bg:rgba(248,113,113,0.1);--radius:12px;--font:Inter,system-ui,sans-serif;--mono:'JetBrains Mono',monospace;--vh:1vh;--shell-vh:1vh;--keyboard-offset:0px;}
.af-root{font-family:var(--font);color:var(--text);background:var(--bg-deep);display:flex;width:100%;height:calc(var(--shell-vh,var(--vh,1vh))*100);min-height:0;overflow:hidden;}
.af-root *{box-sizing:border-box;margin:0;}
.af-root button,.af-root a,.af-root input,.af-root textarea,.af-root select{-webkit-tap-highlight-color:transparent;touch-action:manipulation;}
.af-root ::-webkit-scrollbar{width:6px;} .af-root ::-webkit-scrollbar-track{background:transparent;} .af-root ::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.08);border-radius:3px;}
.af-root .container{width:100%;max-width:1200px;margin:0 auto;}
.af-root .tabs{display:flex;flex-wrap:wrap;gap:8px;}
.af-root .text{word-wrap:break-word;overflow-wrap:break-word;}
.agent-header{display:none;}
.drag-handle{display:none;}
.header-content{display:none;}
.drawer-fab{position:fixed;right:16px;bottom:calc(16px + env(safe-area-inset-bottom) + var(--keyboard-offset));z-index:10001;width:48px;height:48px;border:0;border-radius:999px;background:#6d5efc;color:#fff;font-size:18px;font-weight:700;cursor:pointer;box-shadow:0 10px 24px rgba(0,0,0,0.3);}
.agent-drawer{position:fixed;left:0;right:0;bottom:var(--keyboard-offset);z-index:10000;height:min(72vh,760px);background:#0d1020;border-top-left-radius:18px;border-top-right-radius:18px;box-shadow:0 -20px 60px rgba(0,0,0,.45);transition:transform .25s ease;will-change:transform;overflow:hidden;}
.agent-drawer--closed{transform:translateY(calc(100% - 56px));}
.agent-drawer--open{transform:translateY(0);}
.agent-drawer__handle{height:56px;display:flex;align-items:center;justify-content:space-between;padding:0 14px;background:#12162a;border-bottom:1px solid rgba(255,255,255,.08);color:#cfd3ff;font-weight:700;gap:12px;}
.agent-drawer__title{display:flex;align-items:center;gap:8px;min-width:0;}
.agent-drawer__live{display:inline-flex;width:8px;height:8px;border-radius:999px;background:#34d399;box-shadow:0 0 0 6px rgba(52,211,153,0.12);}
.drawer-mini{border:1px solid rgba(255,255,255,.12);background:transparent;color:#cfd3ff;border-radius:999px;padding:8px 12px;cursor:pointer;font-size:12px;font-weight:600;}
.agent-drawer__body{height:calc(100% - 56px);overflow-y:auto;padding:14px;padding-bottom:calc(14px + env(safe-area-inset-bottom));}
.agent-drawer__empty{padding:14px;border-radius:18px;background:#111525;border:1px dashed rgba(255,255,255,.08);color:#98a2c9;font-size:13px;line-height:1.6;}
.snippet-card{margin-top:12px;padding:14px;border-radius:18px;background:#111525;border:1px solid rgba(255,255,255,.08);overflow-x:auto;max-width:100%;}
.snippet-card:first-child{margin-top:0;}
.snippet-card__title{margin-bottom:10px;font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#98a2c9;}
.snippet-card pre,.snippet-card code{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;max-width:100%;color:#dfe6ff;font-size:13px;line-height:1.45;font-family:var(--mono);}
.snippet-card.warning{background:#2b1014;border-color:rgba(255,120,120,.22);color:#ff9ca3;}
.agent-buttons{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;align-items:stretch;margin:10px 0 6px;}
.agent-buttons .btn{display:flex;align-items:center;justify-content:center;gap:6px;height:42px;padding:0 14px;border:1px solid var(--border);border-radius:10px;background:var(--bg-card);color:var(--text-dim);font-weight:600;font-size:13px;cursor:pointer;transition:all .2s;}
.agent-buttons .btn:hover{background:var(--bg-card-hover);color:var(--text);}
.agent-buttons .btn.primary{background:var(--accent);border-color:transparent;color:#fff;}
.agent-buttons .btn.danger{color:var(--danger);border-color:rgba(248,113,113,0.3);}
.agent-buttons .btn:disabled{opacity:0.5;cursor:not-allowed;}
.af-overlay{position:fixed;inset:0;border:none;background:rgba(6,10,18,0.58);backdrop-filter:blur(8px);z-index:90;padding:0;cursor:pointer;}
.af-side{width:280px;background:var(--bg-card);border-right:1px solid var(--border);display:flex;flex-direction:column;backdrop-filter:blur(20px);transition:width .3s,opacity .3s;z-index:95;}
.af-side-hdr{padding:20px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--border);}
.af-side-title{font-size:14px;font-weight:600;letter-spacing:.02em;}
.af-side-list{flex:1;overflow-y:auto;padding:8px;}
.af-side-item{padding:10px 14px;border-radius:8px;cursor:pointer;font-size:13px;display:flex;justify-content:space-between;align-items:center;transition:all .2s;margin-bottom:2px;} .af-side-item:hover{background:var(--bg-card-hover);transform:translateX(2px);} .af-side-item.active{background:var(--accent-glow);border:1px solid var(--border-focus);}
.af-side-item .del{opacity:0;transition:opacity .2s;cursor:pointer;color:var(--text-muted);font-size:12px;} .af-side-item:hover .del{opacity:1;}
.af-main{flex:1;display:flex;flex-direction:row;min-width:0;min-height:0;overflow:hidden;padding-top:0;}
.af-center{flex:1;display:flex;flex-direction:column;min-width:0;min-height:0;overflow:hidden;position:relative;}
.af-header{padding:16px 24px;display:flex;align-items:center;gap:16px;min-width:0;border-bottom:1px solid var(--border);background:var(--bg-card);backdrop-filter:blur(20px);position:relative;}
.af-logo{font-size:20px;font-weight:700;background:var(--accent);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.af-model-tag{font-size:11px;color:var(--text-dim);display:flex;align-items:center;gap:4px;}
.af-header-actions{margin-left:auto;flex:1;min-width:0;display:flex;justify-content:flex-end;}
.af-header-slider{display:flex;gap:8px;align-items:center;flex-wrap:wrap;max-width:100%;overflow:visible;padding-bottom:2px;}
.af-header-slider::-webkit-scrollbar{display:none;}
.af-btn{flex:0 0 auto;padding:7px 14px;min-height:38px;border-radius:8px;border:1px solid var(--border);background:transparent;color:var(--text-dim);cursor:pointer;font-size:12px;font-family:var(--font);font-weight:500;transition:all .2s;white-space:nowrap;} .af-btn:hover{background:var(--bg-card-hover);color:var(--text);border-color:rgba(255,255,255,0.1);}
.af-btn:disabled{opacity:.4;cursor:not-allowed;}
.af-btn-accent{background:var(--accent);border:none;color:#fff;font-weight:600;} .af-btn-accent:hover{opacity:.9;box-shadow:0 0 20px rgba(99,102,241,0.3);}
.af-btn-accent:disabled{opacity:.4;cursor:not-allowed;}
.af-btn-danger{color:var(--danger);border-color:rgba(248,113,113,0.2);} .af-btn-danger:hover{background:var(--danger-bg);}
.af-btn.active{background:var(--accent-glow);border-color:var(--border-focus);color:var(--text);}
.af-toggle{background:none;border:none;color:var(--text-dim);font-size:18px;cursor:pointer;padding:4px 8px;min-width:42px;min-height:42px;border-radius:12px;transition:color .2s,background .2s;} .af-toggle:hover{color:var(--text);background:rgba(255,255,255,0.04);}
.af-api{margin:0 24px;padding:14px 18px;background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);backdrop-filter:blur(20px);margin-top:16px;}
.af-api-head{display:flex;justify-content:space-between;align-items:center;cursor:pointer;} .af-api-head:hover{color:var(--text);}
.af-api-badge{font-size:10px;padding:2px 8px;border-radius:4px;font-weight:600;}
.af-input{width:100%;padding:10px 14px;border-radius:8px;border:1px solid var(--border);background:var(--bg-input);color:var(--text);font-family:var(--font);font-size:13px;outline:none;transition:border-color .2s;} .af-input:focus{border-color:var(--border-focus);}
.af-api-fly{position:absolute;right:16px;top:64px;z-index:50;width:320px;max-width:80vw;background:var(--bg-card);border:1px solid var(--border);border-radius:14px;box-shadow:0 18px 50px rgba(0,0,0,0.35);backdrop-filter:blur(16px);padding:14px;display:flex;flex-direction:column;gap:12px;}
.af-api-fly small{color:var(--text-muted);}
.af-api-actions{display:flex;justify-content:space-between;align-items:center;font-size:11px;color:var(--text-dim);}
.af-content{flex:1;min-height:0;overflow-y:auto;overflow-x:hidden;padding:24px;overscroll-behavior:contain;}
.af-content-chat{padding-bottom:260px;}
.af-chat-thread{display:flex;flex-direction:column;gap:16px;max-width:860px;margin:0 auto;width:100%;min-height:100%;padding-right:8px;}
.af-msg-user{align-self:flex-end;max-width:75%;min-width:0;background:linear-gradient(135deg, rgba(125,211,252,0.2), rgba(168,85,247,0.18));border:1px solid var(--border-focus);border-radius:18px;padding:12px 18px;font-size:14px;line-height:1.6;animation:msgIn .3s ease;box-shadow:0 10px 30px rgba(0,0,0,0.25);display:flex;flex-direction:column;gap:10px;}
.af-msg-user-text{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;}
.af-msg-bot{align-self:flex-start;max-width:85%;min-width:0;overflow:hidden;animation:msgIn .4s ease;box-shadow:0 10px 30px rgba(0,0,0,0.18);border-radius:18px;}
.af-msg-answer{background:var(--bg-card);border:1px solid var(--border);border-radius:18px;padding:18px 20px;min-width:0;overflow-wrap:anywhere;word-break:break-word;backdrop-filter:blur(16px);font-size:14px;line-height:1.7;box-shadow:0 16px 40px rgba(0,0,0,0.28);}
.af-msg-answer strong{color:#a78bfa;} .af-msg-answer em{color:var(--text-dim);}
.af-msg-answer a{overflow-wrap:anywhere;word-break:break-word;}
.af-msg-answer .md-code{max-width:100%;background:rgba(0,0,0,0.4);border:1px solid var(--border);border-radius:8px;padding:12px 16px;display:block;overflow-x:auto;font-family:var(--mono);font-size:12px;margin:8px 0;color:#c4b5fd;}
.af-msg-answer .md-inline{background:rgba(139,92,246,0.15);padding:2px 6px;border-radius:4px;font-family:var(--mono);font-size:12px;color:#c4b5fd;white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;}
.af-msg-answer .md-h1{font-size:18px;font-weight:700;margin:12px 0 6px;color:#e2e8f0;} .af-msg-answer .md-h2{font-size:16px;font-weight:600;margin:10px 0 4px;color:#e2e8f0;} .af-msg-answer .md-h3{font-size:14px;font-weight:600;margin:8px 0 4px;color:#cbd5e1;}
.af-msg-answer .md-li{padding-left:16px;position:relative;margin:2px 0;}
.af-long-msg{margin-top:10px;border:1px solid var(--border);border-radius:12px;padding:10px 12px;background:rgba(255,255,255,0.03);}
.af-long-msg summary{cursor:pointer;font-weight:700;color:var(--text-muted);}
.af-long-msg[open]{border-color:var(--border-focus);}
.af-msg-error{background:var(--danger-bg);border:1px solid rgba(248,113,113,0.2);border-radius:8px;padding:12px 16px;color:var(--danger);font-size:13px;margin-top:8px;}
.af-steps{margin-bottom:8px;}
.af-step{border:1px solid var(--border);border-radius:8px;margin-bottom:6px;overflow:hidden;transition:all .2s;}
.af-step-hdr{padding:10px 14px;display:flex;align-items:center;gap:8px;cursor:pointer;font-size:12px;background:var(--bg-card);transition:background .2s;} .af-step-hdr:hover{background:var(--bg-card-hover);}
.af-step-body{padding:14px;font-size:13px;border-top:1px solid var(--border);animation:fadeIn .3s ease;}
.af-step-thought{font-style:italic;color:var(--text-dim);margin-bottom:10px;font-size:12px;line-height:1.5;}
.af-step-obs{background:var(--success-bg);border:1px solid rgba(52,211,153,0.15);border-radius:8px;padding:10px 14px;margin-top:10px;}
.af-step-obs pre{margin:0;white-space:pre-wrap;font-family:var(--mono);font-size:11px;color:var(--text-dim);}
.af-step-obs-label{font-size:10px;text-transform:uppercase;letter-spacing:.06em;color:var(--success);font-weight:600;margin-bottom:4px;}
.af-step-input{background:rgba(0,0,0,0.3);padding:8px 12px;border-radius:6px;font-family:var(--mono);font-size:11px;overflow-x:auto;margin-top:6px;}
.af-thinking{display:flex;align-items:center;gap:10px;padding:12px 0;color:var(--text-muted);font-size:13px;animation:fadeIn .3s;}
.af-dots{display:flex;gap:4px;} .af-dot{width:6px;height:6px;background:var(--accent-solid);border-radius:50%;animation:pulse 1.4s infinite;}
.af-dot:nth-child(2){animation-delay:.2s;} .af-dot:nth-child(3){animation-delay:.4s;}
.af-attachment-strip{display:flex;flex-wrap:wrap;gap:10px;width:100%;}
.af-attachment-strip-message{margin-top:2px;}
.af-attachment-chip{display:flex;align-items:center;gap:10px;min-width:0;max-width:100%;padding:8px 10px;border-radius:14px;background:rgba(255,255,255,0.04);border:1px solid var(--border);}
.af-attachment-chip-message{background:rgba(255,255,255,0.06);}
.af-attachment-thumb{width:44px;height:44px;border-radius:10px;object-fit:cover;flex:0 0 auto;background:rgba(255,255,255,0.08);}
.af-attachment-thumb-fallback{display:flex;align-items:center;justify-content:center;font-size:18px;}
.af-attachment-copy{display:flex;flex-direction:column;gap:2px;min-width:0;}
.af-attachment-name{font-size:12px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:220px;}
.af-attachment-sub{font-size:11px;color:var(--text-dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:220px;}
.af-attachment-remove{margin-left:auto;border:none;background:transparent;color:var(--text-dim);font-size:18px;line-height:1;cursor:pointer;padding:8px 6px;min-width:36px;min-height:36px;border-radius:10px;transition:color .2s,background .2s;}
.af-attachment-remove:hover{color:var(--text);}
.af-attachment-remove:disabled,.af-btn-attach:disabled{opacity:.45;cursor:not-allowed;}
.af-btn-attach {background: none; border: none; color: var(--text-dim); font-size: 18px; cursor: pointer; padding: 8px; width:42px; height:42px; border-radius:12px; transition: color .2s,background .2s;flex:0 0 auto;} .af-btn-attach:hover {color: var(--text);background:rgba(255,255,255,0.05);}
.af-input-wrap{position:sticky;bottom:calc(68px + env(safe-area-inset-bottom));z-index:50;align-self:center;width:78%;max-width:560px;margin:0 auto 12px;background:var(--bg-card);border:1px solid var(--border);border-radius:16px;display:flex;flex-direction:column;align-items:stretch;gap:6px;padding:8px 10px 10px;backdrop-filter:blur(20px);box-shadow:0 8px 28px rgba(0,0,0,0.4);}
.af-input-row{display:flex;align-items:flex-end;gap:6px;min-width:0;}
.af-upload-status{font-size:11px;color:var(--text-dim);padding:0 2px;}
.af-textarea{flex:1;resize:none;min-height:36px;height:36px;max-height:96px;padding:10px 0;border:none;background:transparent;color:var(--text);font-family:var(--font);font-size:14px;line-height:1.5;outline:none;}
.af-textarea::placeholder{color:var(--text-muted);}
.af-send{padding:7px 16px;border-radius:16px;font-size:13px;font-weight:600;min-width:56px;min-height:40px;}
.af-panel{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);padding:20px;backdrop-filter:blur(20px);}
.af-strategy-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-bottom:16px;}
.af-strategy-card{background:var(--bg-input);border:1px solid var(--border);border-radius:10px;padding:14px;display:flex;flex-direction:column;gap:8px;}
.af-strategy-label{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);}
.af-strategy-kpi{font-size:24px;font-weight:700;color:var(--text);}
.af-strategy-list{display:grid;gap:6px;font-size:13px;line-height:1.6;}
.af-strategy-pill{align-self:flex-start;font-size:11px;padding:4px 8px;border-radius:999px;background:rgba(124,58,237,0.16);color:#c4b5fd;border:1px solid rgba(124,58,237,0.22);}
.af-strategy-muted{font-size:12px;color:var(--text-dim);line-height:1.6;}
.af-tool-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;}
.af-tool-card{background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:14px;transition:all .2s;} .af-tool-card:hover{border-color:var(--border-focus);transform:translateY(-2px);box-shadow:0 4px 20px rgba(0,0,0,0.3);}
.af-mem-item{background:var(--bg-input);padding:12px 16px;border-radius:8px;margin-bottom:8px;display:flex;justify-content:space-between;font-size:13px;}
.af-log-list{display:flex;flex-direction:column;gap:10px;}
.af-log-item{background:var(--bg-input);border:1px solid var(--border);border-radius:10px;padding:12px 14px;}
.af-log-meta{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:6px;font-size:11px;color:var(--text-muted);}
.af-log-module{font-family:var(--mono);font-size:11px;padding:2px 6px;border-radius:999px;background:rgba(255,255,255,0.05);}
.af-log-msg{font-size:13px;line-height:1.6;color:var(--text);}
.af-empty{text-align:center;padding:60px 20px;color:var(--text-muted);}
.af-empty-icon{font-size:48px;margin-bottom:16px;display:block;}
.af-empty-text{font-size:14px;line-height:1.6;}
@keyframes msgIn{from{opacity:0;transform:translateY(12px);}to{opacity:1;transform:translateY(0);}}
@keyframes fadeIn{from{opacity:0;}to{opacity:1;}}
@keyframes pulse{0%,100%{opacity:.3;transform:scale(.8);}50%{opacity:1;transform:scale(1);}}
@media(max-width:768px){
    .af-root{height:calc(var(--shell-vh,var(--vh,1vh))*100);min-height:0;overflow:hidden;}
    .agent-header{height:70vh;transform:translateY(-60vh);}
    .header-content{height:calc(70vh - 40px);padding:12px;}
    .drawer-fab{right:12px;bottom:calc(12px + env(safe-area-inset-bottom) + var(--keyboard-offset));}
    .agent-drawer{height:min(72vh,560px);}
    .af-side{position:fixed;z-index:100;height:calc(var(--shell-vh,var(--vh,1vh))*100);left:0;top:0;width:min(84vw,320px);max-width:84%;box-shadow:24px 0 60px rgba(0,0,0,0.45);}
    .af-main{flex-direction:column;min-height:0;height:100%;padding-top:0;}
    .af-center{overflow:hidden;min-height:0;}
    .af-header{padding:calc(12px + env(safe-area-inset-top)) calc(16px + env(safe-area-inset-right)) 12px calc(16px + env(safe-area-inset-left));flex-wrap:wrap;align-items:flex-start;gap:12px;}
    .af-logo{font-size:18px;}
    .af-model-tag{flex-wrap:wrap;row-gap:6px;}
    .af-header-actions{margin-left:0;flex-basis:100%;justify-content:flex-start;}
    .af-header-slider{padding-bottom:6px;padding-right:calc(4px + env(safe-area-inset-right));}
    .af-btn{min-height:42px;padding:9px 14px;}
    .af-content{padding:16px;min-height:0;}
    .af-content-chat{padding-bottom:260px;}
    .af-input-wrap{position:sticky;bottom:calc(68px + env(safe-area-inset-bottom));z-index:50;align-self:stretch;width:calc(100% - 24px);max-width:none;border-radius:16px;margin:0 auto;padding:9px calc(14px + env(safe-area-inset-right)) 11px calc(14px + env(safe-area-inset-left));padding-bottom:calc(12px + env(safe-area-inset-bottom));box-shadow:0 -14px 34px rgba(0,0,0,0.42);}
    .af-btn-attach{width:44px;height:44px;border-radius:13px;background:rgba(255,255,255,0.04);}
    .af-send{min-width:54px;min-height:44px;border-radius:13px;padding:0 15px;}
    .af-textarea{min-height:42px;max-height:28vh;font-size:15px;padding:11px 0 10px;}
    .af-msg-user,.af-msg-bot{max-width:100%;}
    .af-attachment-chip{padding:10px 12px;border-radius:16px;}
    .af-attachment-remove{min-width:40px;min-height:40px;}
    .af-attachment-name,.af-attachment-sub{max-width:160px;}
    .af-panel{padding:16px;}
    .af-strategy-grid,.af-tool-grid{grid-template-columns:1fr;}
    .af-root .sidebar{display:none;}
    .af-root .sidebar.is-open{display:flex;}
}
@media(max-width:480px){
    .af-content{padding:14px;}
    .af-content-chat{padding-bottom:220px;}
    .af-header-slider{gap:6px;}
    .af-btn{padding:9px 12px;}
}
.af-root.is-android.is-compact .af-input-wrap{border-top-color:rgba(125,211,252,0.18);}
.af-root.keyboard-open .af-input-wrap{box-shadow:0 -18px 42px rgba(0,0,0,0.55);}
.af-thought-details { margin-bottom: 12px; cursor: pointer; background: rgba(0,0,0,0.2); border-left: 3px solid var(--border-focus); border-radius: 0 8px 8px 0; padding: 12px; font-size: 13px; }
.af-thought-details summary { font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--accent-solid); font-size: 10px; list-style: none; display: flex; align-items: center; outline: none; }
.af-thought-details summary::-webkit-details-marker { display: none; }
.af-thought-inner { margin-top: 8px; font-style: italic; color: var(--text-dim); white-space: pre-wrap; font-size: 14px; }
            `}</style>

                    {sidebarOpen && isCompactUi && (
                        <button type="button" className="af-overlay" aria-label="Close sidebar" onClick={() => setSidebarOpen(false)} />
                    )}
                    {sidebarOpen && (
                        <div className={`af-side sidebar ${sidebarOpen ? "is-open" : ""}`}>
                            <div className="af-side-hdr">
                                <span className="af-side-title">💬 Chats</span>
                                <div className={isCompactUi ? "button-grid" : ""} style={{display:"flex", gap:6}}>
                                    <button className="af-btn" onClick={newConv}>+ New</button>
                                    <button className="af-btn af-btn-danger" onClick={() => setSidebarOpen(false)} style={{padding:"6px 10px"}}>Close</button>
                                </div>
                            </div>
                    <div className="af-side-list">
                        {conversations.map(c => (
                            <div key={c.id} onClick={() => { setCurrentConversationId(c.id); setExpandedSteps({}); if (isCompactUi) setSidebarOpen(false); }}
                                className={`af-side-item ${currentConversationId === c.id ? "active" : ""}`}>
                                <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>{c.title}</span>
                                <span className="del" onClick={e => { e.stopPropagation(); delConv(c.id); }}>✕</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <div className={`agent-header ${headerOpen ? "open" : ""}`} id="agentHeader">
                <div className="drag-handle" id="toggleHeader">
                    {headerOpen ? "↑" : "↓"}
                </div>
                <div className="header-content">
                    <div className="af-header">
                        <button className="af-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>{sidebarOpen ? "◀" : "☰"}</button>
                        <div>
                            <div className="af-logo">Agent Console</div>
                            <div className="af-model-tag">
                                <span>{primaryModel.icon} {primaryModel.label}</span>
                                {fallbackModels.length > 0 && <span style={{ opacity: 0.5 }}> +{fallbackModels.length} fallback</span>}
                                {FASTEST_INFERENCE_PROFILE.enabled && (
                                    <span style={{
                                        padding: "2px 8px",
                                        borderRadius: 999,
                                        border: "1px solid rgba(125,211,252,0.22)",
                                        background: "rgba(125,211,252,0.12)",
                                        color: "#7dd3fc",
                                        fontWeight: 600,
                                    }}>
                                        Fastest {FASTEST_INFERENCE_PROFILE.label}
                                    </span>
                                )}
                                <span style={{ color: hasRuntimeAccess ? "var(--success)" : "var(--danger)" }}>
                                    {HOSTED_API_ENABLED ? `${HOSTED_API_LABEL} auth only` : "Server API unavailable"}
                                </span>
                            </div>
                        </div>
                        <div className="af-header-actions">
                            <div className={`af-header-slider tabs ${isCompactUi ? "button-grid" : ""}`}>
                                {tabConfig.map(tab => (
                                    <button key={tab.id} className={`af-btn ${activeTab === tab.id ? "active" : ""}`} onClick={() => setActiveTab(tab.id)}>
                                        {tab.label}
                                    </button>
                                ))}
                                <button className="af-btn" onClick={newConv}>New</button>
                                <button className="af-btn" onClick={exportConv} disabled={!allMessages.length}>Export</button>
                                <button className="af-btn af-btn-danger" onClick={() => { setExpandedSteps({}); clearConv(); }} disabled={running || !allMessages.length}>Reset</button>
                                <button
                                    ref={settingsBtnRef}
                                    className={`af-btn ${settingsOpen ? "active" : ""}`}
                                    onClick={() => {
                                        setHeaderOpen(true);
                                        setSettingsOpen(v => !v);
                                    }}
                                    style={{padding:"6px 10px"}}
                                >
                                    ⚙ Models
                                </button>
                            </div>
                        </div>
                    </div>
                    {settingsOpen && (
                            <div className="af-api-fly" ref={settingsFlyRef}>
                                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                                    <span style={{ fontWeight: 700 }}>Model Settings</span>
                                    <button className="af-btn" style={{ padding: "4px 8px" }} onClick={() => setSettingsOpen(false)}>✕</button>
                                </div>
                            <small>{HOSTED_API_ENABLED ? `${HOSTED_API_LABEL} runs at /api/chat and uses the server-side CEREBRAS_API_KEY. Browser-stored API keys are disabled and any previously saved local keys are removed on load.` : "Server API is unavailable."}</small>
                                <div style={{ display: "grid", gap: 8 }}>
                                    <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: ".08em", color: "var(--text-muted)" }}>Primary model</div>
                                    <select className="af-input" value={primaryModel.id} onChange={e => handlePrimaryModelChange(e.target.value)}>
                                        {AVAILABLE_MODELS.map(model => (
                                            <option key={model.id} value={model.id}>{`${model.icon} ${model.label}`}</option>
                                    ))}
                                </select>
                            </div>
                            <div style={{ display: "grid", gap: 8 }}>
                                <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: ".08em", color: "var(--text-muted)" }}>Fallback models</div>
                                {AVAILABLE_MODELS.filter(model => model.id !== primaryModel.id).map(model => (
                                    <label key={model.id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 12px", borderRadius: 10, border: "1px solid var(--border)", background: "var(--bg-input)", fontSize: 13 }}>
                                        <input type="checkbox" checked={fallbackModels.some(entry => entry.id === model.id)} onChange={() => toggleFallbackModel(model)} />
                                        <span>{model.icon} {model.label}</span>
                                    </label>
                                ))}
                            </div>
                            <div style={{ display: "grid", gap: 6, marginTop: 6, padding: "10px 12px", borderRadius: 10, border: "1px solid var(--border)", background: "var(--bg-input)", fontSize: 13 }}>
                                <label style={{ display: "flex", alignItems: "center", gap: 10 }}>
                                    <input type="checkbox" checked={multiThink} onChange={e => setMultiThink(e.target.checked)} />
                                    <span>Multi-process thinking (run 2 parallel drafts and merge)</span>
                                </label>
                                <small style={{ color: "var(--text-muted)" }}>Runs two independent drafts (Alpha/Beta) in parallel and combines them for a more reliable answer.</small>
                            </div>
                            <div style={{ display: "grid", gap: 6, marginTop: 6, padding: "10px 12px", borderRadius: 10, border: "1px solid var(--border)", background: "var(--bg-input)", fontSize: 13 }}>
                                <label style={{ display: "flex", alignItems: "center", gap: 10 }}>
                                    <input type="checkbox" checked={doublePass} onChange={e => setDoublePass(e.target.checked)} />
                                    <span>Two-pass refinement (draft then self-review)</span>
                                </label>
                                <small style={{ color: "var(--text-muted)" }}>First pass writes an answer; second pass critiques and improves it. Skips refinement when multi-process is on.</small>
                            </div>
                            <div style={{ display: "grid", gap: 6, marginTop: 6, padding: "10px 12px", borderRadius: 10, border: "1px solid var(--border)", background: "var(--bg-input)", fontSize: 13 }}>
                                <label style={{ display: "flex", alignItems: "center", gap: 10 }}>
                                    <input type="checkbox" checked={thinkAloud} onChange={e => setThinkAloud(e.target.checked)} />
                                    <span>Think-aloud plan (injects a &lt;thought&gt; plan before answers)</span>
                                </label>
                                <small style={{ color: "var(--text-muted)" }}>Adds a brief step-by-step plan before the final answer so you can see reasoning.</small>
                            </div>
                            <div className="af-api-actions">
                                <span>{HOSTED_API_ENABLED ? `${HOSTED_API_LABEL} active` : "Server API offline"}</span>
                                <span>Auth stays on the server</span>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            <div className="af-main grid">
                <div className="af-center container">
                <div className={`af-content ${activeTab === "chat" ? "af-content-chat" : ""}`}>
                    {activeTab === "docs" && (
                        <div style={{ display: "grid", gap: 16 }}>
                            <div className="af-strategy-grid">
                                <div className="af-strategy-card">
                                    <div className="af-strategy-label">Endpoint</div>
                                    <div className="af-strategy-kpi" style={{ fontSize: 18, lineHeight: 1.3 }}>POST</div>
                                    <div className="af-strategy-muted"><code style={{ fontFamily: "var(--mono)" }}>{HOSTED_CHAT_API_PATH}</code> on the current host.</div>
                                </div>
                                <div className="af-strategy-card">
                                    <div className="af-strategy-label">Authentication</div>
                                    <div className="af-strategy-kpi" style={{ fontSize: 18, lineHeight: 1.3 }}>Server-side</div>
                                    <div className="af-strategy-muted">Clients do not send model keys. The server keeps authentication in CEREBRAS_API_KEY.</div>
                                </div>
                                <div className="af-strategy-card">
                                    <div className="af-strategy-label">Default Mode</div>
                                    <div className="af-strategy-kpi" style={{ fontSize: 18, lineHeight: 1.3 }}>Agent</div>
                                    <div className="af-strategy-muted">Direct API calls run the server-side router, planner, tools, and verifier.</div>
                                </div>
                                <div className="af-strategy-card">
                                    <div className="af-strategy-label">Streaming</div>
                                    <div className="af-strategy-kpi" style={{ fontSize: 18, lineHeight: 1.3 }}>SSE</div>
                                    <div className="af-strategy-muted">Set <code style={{ fontFamily: "var(--mono)" }}>stream: true</code>. Agent mode currently streams the final answer only.</div>
                                </div>
                            </div>

                            <div className="af-panel">
                                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 16, gap: 8, flexWrap: "wrap", alignItems: "center" }}>
                                    <div>
                                        <div style={{ fontWeight: 700, marginBottom: 6 }}>API Documentation</div>
                                        <div className="af-strategy-muted">This website exposes a public JSON API at <code style={{ fontFamily: "var(--mono)" }}>{apiUrl}</code>.</div>
                                    </div>
                                    <div className={isCompactUi ? "button-grid" : ""} style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                                        <button
                                            className="af-btn"
                                            onClick={(e) => {
                                                navigator.clipboard?.writeText(apiUrl);
                                                e.currentTarget.innerText = "Copied!";
                                                setTimeout(() => { e.currentTarget.innerText = "Copy endpoint"; }, 2000);
                                            }}
                                        >
                                            Copy endpoint
                                        </button>
                                        <a className="af-btn" href={apiUrl} target="_blank" rel="noreferrer" style={{ textDecoration: "none" }}>Open metadata</a>
                                    </div>
                                </div>
                                <div className="af-strategy-grid" style={{ marginBottom: 0 }}>
                                    <div className="af-strategy-card">
                                        <div className="af-strategy-label">Accepted Inputs</div>
                                        <div className="af-strategy-list">
                                            <div><strong>Prompt:</strong> send <code style={{ fontFamily: "var(--mono)" }}>prompt</code>, <code style={{ fontFamily: "var(--mono)" }}>input</code>, or <code style={{ fontFamily: "var(--mono)" }}>message</code>.</div>
                                            <div><strong>Messages:</strong> send a chat array with <code style={{ fontFamily: "var(--mono)" }}>system</code>, <code style={{ fontFamily: "var(--mono)" }}>user</code>, and <code style={{ fontFamily: "var(--mono)" }}>assistant</code> roles.</div>
                                            <div><strong>Images:</strong> message content can include <code style={{ fontFamily: "var(--mono)" }}>text</code> and <code style={{ fontFamily: "var(--mono)" }}>image_url</code> parts.</div>
                                        </div>
                                    </div>
                                    <div className="af-strategy-card">
                                        <div className="af-strategy-label">Controls</div>
                                        <div className="af-strategy-list">
                                            <div><strong>Mode:</strong> <code style={{ fontFamily: "var(--mono)" }}>executionMode: "agent"</code> is the default. Use <code style={{ fontFamily: "var(--mono)" }}>"completion"</code> to bypass tools and planning.</div>
                                            <div><strong>Model:</strong> choose one of {AVAILABLE_MODELS.map(model => <code key={model.id} style={{ fontFamily: "var(--mono)", marginRight: 6 }}>{model.id}</code>)}.</div>
                                            <div><strong>Output:</strong> default is normalized JSON. Use <code style={{ fontFamily: "var(--mono)" }}>format: "raw"</code> for upstream payloads.</div>
                                        </div>
                                    </div>
                                    <div className="af-strategy-card">
                                        <div className="af-strategy-label">Response Shape</div>
                                        <div className="af-strategy-list">
                                            <div><strong>Agent mode:</strong> returns <code style={{ fontFamily: "var(--mono)" }}>agentic</code>, <code style={{ fontFamily: "var(--mono)" }}>strategy</code>, <code style={{ fontFamily: "var(--mono)" }}>steps</code>, and <code style={{ fontFamily: "var(--mono)" }}>output_text</code>.</div>
                                            <div><strong>Completion mode:</strong> returns normalized completion JSON with <code style={{ fontFamily: "var(--mono)" }}>choices</code> and <code style={{ fontFamily: "var(--mono)" }}>output_text</code>.</div>
                                            <div><strong>Reasoning:</strong> normalized JSON strips raw reasoning fields before returning them to callers.</div>
                                        </div>
                                    </div>
                                    <div className="af-strategy-card">
                                        <div className="af-strategy-label">Notes</div>
                                        <div className="af-strategy-list">
                                            <div><strong>Server tools:</strong> the live tool registry is listed in the <strong>Tools</strong> tab.</div>
                                            <div><strong>Provider routing:</strong> requests prefer throughput routing for the fastest available upstream.</div>
                                            <div><strong>Metadata:</strong> <code style={{ fontFamily: "var(--mono)" }}>GET {HOSTED_CHAT_API_PATH}</code> returns endpoint metadata and examples.</div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="af-panel">
                                <div style={{ fontWeight: 700, marginBottom: 12 }}>Quickstart</div>
                                <div className="af-strategy-muted" style={{ marginBottom: 12 }}>Minimal agent request with no client API key.</div>
                                <DocCodeBlock code={agentCurlExample} language="bash" />
                            </div>

                            <div className="af-panel">
                                <div style={{ fontWeight: 700, marginBottom: 12 }}>Request Examples</div>
                                <div style={{ display: "grid", gap: 16 }}>
                                    <div>
                                        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: "var(--text-dim)" }}>Default server-side agent request</div>
                                        <DocCodeBlock code={agentRequestExample} language="json" />
                                    </div>
                                    <div>
                                        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: "var(--text-dim)" }}>Force raw completion mode</div>
                                        <DocCodeBlock code={completionRequestExample} language="json" />
                                    </div>
                                    <div>
                                        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: "var(--text-dim)" }}>Messages payload with image input</div>
                                        <DocCodeBlock code={messagesRequestExample} language="json" />
                                    </div>
                                </div>
                            </div>

                            <div className="af-panel">
                                <div style={{ fontWeight: 700, marginBottom: 12 }}>Response Examples</div>
                                <div style={{ display: "grid", gap: 16 }}>
                                    <div>
                                        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: "var(--text-dim)" }}>Normalized agent response</div>
                                        <DocCodeBlock code={agentResponseExample} language="json" />
                                    </div>
                                    <div>
                                        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: "var(--text-dim)" }}>Completion-mode curl</div>
                                        <DocCodeBlock code={completionCurlExample} language="bash" />
                                    </div>
                                    <div>
                                        <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8, color: "var(--text-dim)" }}>Streaming curl</div>
                                        <DocCodeBlock code={streamCurlExample} language="bash" />
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {activeTab === "strategy" && conv && (
                        <div className="af-panel">
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 16, gap: 8, flexWrap: "wrap" }}>
                                <span style={{ fontWeight: 600 }}>Execution Strategy</span>
                                <span
                                    className="af-strategy-pill"
                                    style={verificationState?.status === "pass"
                                        ? { background: "rgba(52,211,153,0.14)", color: "var(--success)", borderColor: "rgba(52,211,153,0.2)" }
                                        : verificationState?.status === "revise"
                                            ? { background: "rgba(248,113,113,0.12)", color: "var(--danger)", borderColor: "rgba(248,113,113,0.18)" }
                                            : undefined}
                                >
                                    {verificationState?.status === "pass"
                                        ? "Verifier passed"
                                        : verificationState?.status === "revise"
                                            ? "Revision requested"
                                            : running
                                                ? "Run in progress"
                                                : "Awaiting run"}
                                </span>
                            </div>

                            <div className="af-strategy-grid">
                                <div className="af-strategy-card">
                                    <div className="af-strategy-label">Iterations</div>
                                    <div className="af-strategy-kpi">{runMetrics.iterations}</div>
                                    <div className="af-strategy-muted">Latest loop count for this run.</div>
                                </div>
                                <div className="af-strategy-card">
                                    <div className="af-strategy-label">Tool Calls</div>
                                    <div className="af-strategy-kpi">{runMetrics.toolCalls}</div>
                                    <div className="af-strategy-muted">Batched tool execution is enabled.</div>
                                </div>
                                <div className="af-strategy-card">
                                    <div className="af-strategy-label">Verifier Passes</div>
                                    <div className="af-strategy-kpi">{runMetrics.verificationPasses}</div>
                                    <div className="af-strategy-muted">Draft approvals completed this run.</div>
                                </div>
                                <div className="af-strategy-card">
                                    <div className="af-strategy-label">Verifier Revisions</div>
                                    <div className="af-strategy-kpi">{runMetrics.verificationRevisions}</div>
                                    <div className="af-strategy-muted">{runMetrics.updatedAt ? `Updated ${new Date(runMetrics.updatedAt).toLocaleTimeString()}` : "No run metrics yet."}</div>
                                </div>
                            </div>

                            <div className="af-strategy-grid">
                                <div className="af-strategy-card">
                                    <div className="af-strategy-label">Router</div>
                                    {routerState ? (
                                        <>
                                            {routerState.primary && routerState.primary !== "none" && <span className="af-strategy-pill">{routerState.primary}</span>}
                                            <div className="af-strategy-list">
                                                <div><strong>Reason:</strong> {routerState.reason || "No router reason provided."}</div>
                                                {routerState.analysis && <div><strong>Analysis:</strong> {routerState.analysis}</div>}
                                                {routerState.secondary?.length > 0 && <div><strong>Backups:</strong> {routerState.secondary.join(", ")}</div>}
                                            </div>
                                        </>
                                    ) : (
                                        <div className="af-strategy-muted">The router has not produced a routing recommendation yet.</div>
                                    )}
                                </div>

                                <div className="af-strategy-card">
                                    <div className="af-strategy-label">Planner</div>
                                    {planState ? (
                                        <>
                                            {planState.complexity && <span className="af-strategy-pill">{planState.complexity}</span>}
                                            <div className="af-strategy-list">
                                                <div><strong>Goal:</strong> {planState.goal || "No explicit goal captured."}</div>
                                                {planState.objectives?.length > 0 && <div><strong>Objectives:</strong> {planState.objectives.join(" | ")}</div>}
                                                {planState.success?.length > 0 && <div><strong>Success:</strong> {planState.success.join(" | ")}</div>}
                                                {planState.constraints?.length > 0 && <div><strong>Constraints:</strong> {planState.constraints.join(" | ")}</div>}
                                                {planState.tooling?.length > 0 && <div><strong>Tooling:</strong> {planState.tooling.join(" | ")}</div>}
                                                {planState.notes && <div><strong>Notes:</strong> {planState.notes}</div>}
                                            </div>
                                        </>
                                    ) : (
                                        <div className="af-strategy-muted">The planner will store objectives, constraints, and success criteria here.</div>
                                    )}
                                </div>

                                <div className="af-strategy-card">
                                    <div className="af-strategy-label">Verifier</div>
                                    {verificationState ? (
                                        <>
                                            <span className="af-strategy-pill" style={verificationState.status === "pass" ? { background: "rgba(52,211,153,0.14)", color: "var(--success)", borderColor: "rgba(52,211,153,0.2)" } : { background: "rgba(248,113,113,0.12)", color: "var(--danger)", borderColor: "rgba(248,113,113,0.18)" }}>
                                                {verificationState.status}
                                            </span>
                                            <div className="af-strategy-list">
                                                <div><strong>Summary:</strong> {verificationState.summary || "No verifier summary provided."}</div>
                                                {verificationState.checks?.length > 0 && <div><strong>Checks:</strong> {verificationState.checks.join(" | ")}</div>}
                                                {verificationState.missing?.length > 0 && <div><strong>Missing:</strong> {verificationState.missing.join(" | ")}</div>}
                                                {verificationState.nextAction && <div><strong>Next action:</strong> {verificationState.nextAction}</div>}
                                            </div>
                                        </>
                                    ) : (
                                        <div className="af-strategy-muted">The verifier will approve or reject the draft answer here.</div>
                                    )}
                                </div>
                            </div>
                        </div>
                    )}

                    {activeTab === "memory" && (
                        <div className="af-panel">
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 16 }}>
                                <span style={{ fontWeight: 600 }}>Stored Memory ({Object.keys(memoryStore).length})</span>
                                <button className="af-btn af-btn-danger" onClick={() => updateMemory({})}>Clear All</button>
                            </div>
                            {Object.keys(memoryStore).length === 0
                                ? <div className="af-empty"><span className="af-empty-icon">🧠</span><div className="af-empty-text">Named memory entries will appear here.<br/>Use the memory tools to save facts between prompts in this session.</div></div>
                                : Object.entries(memoryStore).map(([k, v]) => (
                                    <div key={k} className="af-mem-item">
                                        <span><code style={{ fontFamily: "var(--mono)", color: "#a78bfa", fontWeight: 500 }}>{k}</code> → {v.value}</span>
                                        <span style={{ fontSize: 11, color: "var(--text-muted)" }}>{new Date(v.timestamp).toLocaleTimeString()}</span>
                                    </div>
                                ))}
                            <div style={{ display: "grid", gap: 12, marginTop: 18 }}>
                                <div className="af-strategy-card">
                                    <div style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: ".08em", color: "var(--text-muted)" }}>Episodic Summary</div>
                                    {memoryEngine.episodicSummary
                                        ? <div className="af-strategy-muted" style={{ whiteSpace: "pre-wrap" }}>{memoryEngine.episodicSummary}</div>
                                        : <div className="af-strategy-muted">Older turns are still inside the live sliding window. The rolling summary will appear here once history compacts.</div>}
                                </div>
                                <div className="af-strategy-card">
                                    <div style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: ".08em", color: "var(--text-muted)" }}>Semantic Index</div>
                                    <div className="af-strategy-list">
                                        <div><strong>Chunks:</strong> {memoryEngine.semanticChunks.length}</div>
                                        <div><strong>System cache key:</strong> <code style={{ fontFamily: "var(--mono)" }}>{memoryEngine.systemPromptCacheKey}</code></div>
                                        <div><strong>Last prompt:</strong> {lastPromptStats ? `${lastPromptStats.tokenCount}/${lastPromptStats.promptBudget} tokens${lastPromptStats.trimmed ? " (compacted)" : ""}` : "No token stats yet."}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}

                    {activeTab === "logs" && conv && (
                        <div className="af-panel">
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 16, gap: 8, flexWrap: "wrap" }}>
                                <span style={{ fontWeight: 600 }}>Run Log ({systemLogs.length})</span>
                                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                                    <button className="af-btn" onClick={handleCopyLogs} disabled={!systemLogs.length}>{logCopyStatus}</button>
                                    <button className="af-btn af-btn-danger" onClick={() => updateConv(c => ({ ...c, systemLogs: [] }))} disabled={!systemLogs.length}>Clear Log</button>
                                </div>
                            </div>
                            {systemLogs.length === 0 ? (
                                <div className="af-empty">
                                    <span className="af-empty-icon">–</span>
                                    <div className="af-empty-text">Execution events appear here when the agent plans, retries, falls back, and calls tools.</div>
                                </div>
                            ) : (
                                <div className="af-log-list">
                                    {systemLogs.map((entry, index) => (
                                        <div key={`${entry.ts}-${entry.module}-${index}`} className="af-log-item">
                                            <div className="af-log-meta">
                                                <span>{entry.ts}</span>
                                                <span className="af-log-module" style={{ color: entry.color }}>{entry.module}</span>
                                            </div>
                                            <div className="af-log-msg">{entry.msg}</div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}

                    {activeTab === "tools" && (
                        <div>
                            {categories.map(cat => (
                                <div key={cat} style={{ marginBottom: 24 }}>
                                    <div style={{ fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: ".08em", color: "var(--text-muted)", marginBottom: 10 }}>{cat}</div>
                                    <div className="af-tool-grid">
                                        {toolList.filter(([, t]) => t.category === cat).map(([name, tool]) => (
                                            <div key={name} className="af-tool-card">
                                                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                                                    <span style={{ fontSize: 16 }}>{tool.icon}</span>
                                                    <code style={{ fontFamily: "var(--mono)", fontWeight: 600, fontSize: 13 }}>{name}</code>
                                                </div>
                                                <p style={{ fontSize: 12, color: "var(--text-dim)", lineHeight: 1.5, marginBottom: 8 }}>{tool.description}</p>
                                                <code style={{ display: "block", fontSize: 10, background: "rgba(0,0,0,0.3)", padding: "6px 10px", borderRadius: 6, fontFamily: "var(--mono)", color: "var(--text-muted)", wordBreak: "break-all" }}>{tool.example}</code>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {activeTab === "chat" && conv && (
                        <div className="af-chat-thread">
                            {allMessages.length === 0 && !running && (
                                <div className="af-empty">
                                    <span className="af-empty-icon">–</span>
                                    <div className="af-empty-text">Plan research, calculations, or file analysis from one place.<br/>This workspace can chain <strong>{toolList.length} tools</strong>, batch tool calls, verify drafts, and keep session memory.</div>
                                </div>
                            )}
                            {allMessages.map((msg, idx) => msg.role === "user" ? (
                                <div key={idx} className="af-msg-user text">
                                    {(() => {
                                        const visibleText = typeof msg.displayText === "string" ? msg.displayText : msg.content;
                                        const attachments = Array.isArray(msg.attachments) ? msg.attachments : [];
                                        return (
                                            <>
                                                {visibleText && <div className="af-msg-user-text text">{visibleText}</div>}
                                                {attachments.length > 0 && (
                                                    <div className="af-attachment-strip af-attachment-strip-message">
                                                        {attachments.map((attachment, attachmentIndex) => (
                                                            <div key={attachment.id || `${idx}-${attachment.name}-${attachmentIndex}`} className="af-attachment-chip af-attachment-chip-message">
                                                                {attachment.kind === "image" && attachment.previewUrl ? (
                                                                    <img className="af-attachment-thumb" src={attachment.previewUrl} alt={attachment.name} />
                                                                ) : (
                                                                    <div className="af-attachment-thumb af-attachment-thumb-fallback">{attachment.kind === "image" ? "IMG" : "FILE"}</div>
                                                                )}
                                                                <div className="af-attachment-copy">
                                                                    <div className="af-attachment-name">{attachment.name}</div>
                                                                    <div className="af-attachment-sub">{buildAttachmentBadge(attachment)}</div>
                                                                </div>
                                                            </div>
                                                        ))}
                                                    </div>
                                                )}
                                                {!visibleText && !attachments.length && msg.content && <div className="af-msg-user-text text">{msg.content}</div>}
                                            </>
                                        );
                                    })()}
                                </div>
                            ) : (
                                <div key={idx} className="af-msg-bot">
                                    {msg.steps?.length > 0 && (
                                        <div className="af-steps">
                                            {msg.steps.filter(s => s.type === "action").map((step, si) => {
                                                const details = [];
                                                if (step.input && Object.keys(step.input).length) details.push(`input=${JSON.stringify(step.input)}`);
                                                if (step.observation) details.push(`obs=${step.observation}`);
                                                const text = `${step.action}${details.length ? " " + details.join(" | ") : ""}`;
                                                return (
                                                    <TerminalMessage
                                                        key={`${idx}-${si}`}
                                                        status={step.status || "info"}
                                                        isRunning={false}
                                                        text={text}
                                                    />
                                                );
                                            })}
                                        </div>
                                    )}
                                    {msg.error ? (
                                        <div className="af-msg-error text">Error: {msg.error}</div>
                                    ) : (
                                        <MessageContent text={msg.content} />
                                    )}
                                </div>
                            ))}

                            {running && pendingSteps?.length > 0 && (
                                <div className="af-steps">
                                    {pendingSteps.filter(s => s.type === "action").map((step, si) => {
                                        const details = [];
                                        if (step.input && Object.keys(step.input).length) details.push(`input=${JSON.stringify(step.input)}`);
                                        const text = `${step.action}${details.length ? " " + details.join(" | ") : ""}`;
                                        return (
                                            <TerminalMessage
                                                key={`pending-${si}`}
                                                status={step.status || "info"}
                                                isRunning
                                                text={text}
                                            />
                                        );
                                    })}
                                </div>
                            )}

                            {running && conv.activeStream && (
                                <div className="af-msg-bot">
                                    <div className="af-msg-answer">
                                        {/* A simple inline parser for streaming XML */}
                                        {(() => {
                                            const streamText = conv.activeStream.length > STREAM_RENDER_CAP
                                                ? conv.activeStream.slice(-STREAM_RENDER_CAP)
                                                : conv.activeStream;
                                            const parts = streamText.split(/(<thought>[\s\S]*?(?:<\/thought>|$))/);
                                            return parts.map((part, i) => {
                                                if (part.startsWith("<thought>")) {
                                                    const thoughtText = part.replace(/^<thought>\n?/, "").replace(/\n?<\/thought>$/, "");
                                                    return (
                                                        <details open key={i} className="af-thought-details" style={{ margin:0, border:"none", padding:0 }}>
                                                            <summary>Thinking...</summary>
                                                            <div className="af-thought-inner">{thoughtText}</div>
                                                        </details>
                                                    );
                                                }
                                                return <div key={i} dangerouslySetInnerHTML={{ __html: renderMarkdown(part) }} />;
                                            });
                                        })()}
                                    </div>
                                </div>
                            )}

                            {running && !conv.activeStream && (
                                <div className="af-thinking">
                                    <div className="af-dots"><div className="af-dot" /><div className="af-dot" /><div className="af-dot" /></div>
                                    <span>{primaryModel.icon} {primaryModel.label} starting...</span>
                                </div>
                            )}
                            <div ref={bottomRef} />
                        </div>
                    )}
                </div>

                {/* Overlaid Input Bar */}
                {activeTab === "chat" && conv && (
                    <div className="af-input-wrap" ref={composerRef}>
                        <input type="file" ref={fileInputRef} multiple accept=".txt,.md,.markdown,.json,.csv,.js,.mjs,.cjs,.ts,.jsx,.tsx,.py,.rb,.go,.rs,.java,.c,.h,.cpp,.hpp,.html,.css,.scss,.sass,.xml,.yaml,.yml,.toml,.ini,.env,.log,text/*,application/json,image/*" style={{display:'none'}} onChange={handleFileUpload} />
                        {pendingUploadsView.length > 0 && (
                            <div className="af-attachment-strip">
                                {pendingUploadsView.map((attachment, attachmentIndex) => (
                                    <div key={attachment.id || `${attachment.name}-${attachmentIndex}`} className="af-attachment-chip">
                                            {attachment.kind === "image" && attachment.dataUrl ? (
                                                <img className="af-attachment-thumb" src={attachment.dataUrl} alt={attachment.name} />
                                            ) : (
                                                <div className="af-attachment-thumb af-attachment-thumb-fallback">{attachment.kind === "image" ? "IMG" : "FILE"}</div>
                                            )}
                                        <div className="af-attachment-copy">
                                            <div className="af-attachment-name">{attachment.name}</div>
                                            <div className="af-attachment-sub">{buildAttachmentBadge(attachment)}</div>
                                        </div>
                                        <button className="af-attachment-remove" type="button" disabled={running || preparingSend} onClick={() => removePendingUpload(attachment.id)}>×</button>
                                    </div>
                                ))}
                            </div>
                        )}
                        <div className="af-input-row">
                            <button className="af-btn-attach" disabled={running || preparingSend} onClick={() => fileInputRef.current?.click()}>Attach</button>
                            <textarea ref={inputRef} className="af-textarea" value={conv.currentInput || ""} onChange={e => updateConv(c => ({ ...c, currentInput: e.target.value }))} onKeyDown={handleKey} onFocus={() => { if (isCompactUi) setSidebarOpen(false); }} disabled={running || preparingSend} enterKeyHint="send" placeholder={"Ask for research, calculations, code, or attached-file analysis..."} />
                            {running ? (
                                <button className="af-btn af-btn-danger af-send" onClick={() => abortRef.current?.abort()}>■</button>
                            ) : (
                                <button className="af-btn af-btn-accent af-send" disabled={!canSend || preparingSend} onClick={runAgent}>Send</button>
                            )}
                        </div>
                        {uploadStatus && <div className="af-upload-status">{uploadStatus}</div>}
                    </div>
                )}
                {conv && (
                    <>
                        <button
                            id="drawerToggle"
                            className="drawer-fab"
                            type="button"
                            aria-expanded={drawerOpen ? "true" : "false"}
                            aria-controls="agentDrawer"
                            aria-label={drawerOpen ? "Collapse agent console" : "Expand agent console"}
                            onClick={() => setDrawerOpen(open => !open)}
                        >
                            {drawerOpen ? "↑" : "↓"}
                        </button>
                        <aside
                            id="agentDrawer"
                            className={`agent-drawer ${drawerOpen ? "agent-drawer--open" : "agent-drawer--closed"}`}
                        >
                            <div className="agent-drawer__handle">
                                <div className="agent-drawer__title">
                                    {running && <span className="agent-drawer__live" aria-hidden="true" />}
                                    <span>{drawerSummaryLabel}</span>
                                </div>
                                <button
                                    id="drawerToggle2"
                                    className="drawer-mini"
                                    type="button"
                                    onClick={() => setDrawerOpen(open => !open)}
                                >
                                    {drawerOpen ? "Close" : "Open"}
                                </button>
                            </div>

                            <div className="agent-drawer__body">
                                <div className="agent-buttons" style={{ marginBottom: 8 }}>
                                    <button className={`btn ${activeTab === "chat" ? "primary" : ""}`} onClick={() => setActiveTab("chat")}>Chat</button>
                                    <button className={`btn ${activeTab === "docs" ? "primary" : ""}`} onClick={() => setActiveTab("docs")}>Docs</button>
                                    <button className={`btn ${activeTab === "logs" ? "primary" : ""}`} onClick={() => setActiveTab("logs")}>Logs ({systemLogs.length})</button>
                                    <button className="btn" onClick={newConv}>New</button>
                                    <button className="btn" onClick={exportConv} disabled={!allMessages.length}>Export</button>
                                    <button className="btn danger" onClick={() => { setExpandedSteps({}); clearConv(); }} disabled={running || !allMessages.length}>Reset</button>
                                    <button
                                        className="btn"
                                        onClick={() => {
                                            setHeaderOpen(true);
                                            setSettingsOpen(v => !v);
                                        }}
                                    >
                                        Models
                                    </button>
                                </div>

                                {!drawerCards.length && !drawerWarning ? (
                                    <div className="agent-drawer__empty">
                                        Agent steps, live output, and warnings stay here between updates so the page stays clear.
                                    </div>
                                ) : (
                                    <>
                                        {drawerCards.map(card => (
                                            <div key={card.key} className="snippet-card">
                                                <div className="snippet-card__title">{card.label}</div>
                                                <pre><code>{card.body}</code></pre>
                                            </div>
                                        ))}
                                        {drawerWarning && (
                                            <div className="snippet-card warning">
                                                {drawerWarning}
                                            </div>
                                        )}
                                    </>
                                )}
                            </div>
                        </aside>
                    </>
                )}
            </div>
        </div>
    </div>
    );
}
