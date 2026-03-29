const { loadAppState, saveAppState } = require("./db");
const { getScopedStateKeyFromRequest } = require("./state-scope");

const CHAT_MEMORY_PREFIX = "chatmem:";
const RECENT_MESSAGE_LIMIT = 4;
const MESSAGE_CHAR_LIMIT = 900;
const DIGEST_CHAR_LIMIT = 220;
const SUMMARY_CHAR_LIMIT = 1400;
const SUMMARY_LINE_LIMIT = 10;

const normalizeText = (value) => (
    String(value ?? "")
        .replace(/\u0000/g, "")
        .replace(/\r\n?/g, "\n")
        .trim()
);

const truncateText = (value, max) => {
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

const normalizeMemoryMessage = (message) => {
    if (!message || message.role === "system") return null;
    const role = message.role === "assistant" ? "assistant" : "user";
    const rawText = getMessageText(message.content);
    const text = role === "assistant"
        ? stripAssistantArtifacts(rawText)
        : normalizeText(rawText);
    if (!text) return null;
    return {
        role,
        content: truncateText(text, MESSAGE_CHAR_LIMIT),
    };
};

const sameMessage = (left, right) => (
    left?.role === right?.role &&
    left?.content === right?.content
);

const dedupeConsecutiveMessages = (messages = []) => {
    const output = [];
    for (const message of messages) {
        if (!message) continue;
        if (output.length && sameMessage(output[output.length - 1], message)) continue;
        output.push(message);
    }
    return output;
};

const parseSummaryLines = (summary) => {
    const text = normalizeText(summary);
    if (!text) return [];
    return text
        .replace(/^Previously:\s*/i, "")
        .split("\n")
        .map((line) => normalizeText(line))
        .filter(Boolean);
};

const buildMessageDigest = (message) => {
    if (!message?.content) return "";
    const label = message.role === "assistant" ? "Assistant" : "User";
    return `${label}: ${truncateText(message.content.replace(/\s+/g, " "), DIGEST_CHAR_LIMIT)}`;
};

const buildSummaryText = (lines = []) => {
    const trimmed = (Array.isArray(lines) ? lines : [])
        .map((line) => normalizeText(line))
        .filter(Boolean)
        .slice(-SUMMARY_LINE_LIMIT);
    if (!trimmed.length) return "";
    return truncateText(`Previously:\n${trimmed.join("\n")}`, SUMMARY_CHAR_LIMIT);
};

const normalizeChatMemory = (raw = {}) => {
    const recentMessages = dedupeConsecutiveMessages(
        (Array.isArray(raw?.recentMessages) ? raw.recentMessages : [])
            .map(normalizeMemoryMessage)
            .filter(Boolean),
    ).slice(-RECENT_MESSAGE_LIMIT);

    return {
        version: 1,
        summary: buildSummaryText(parseSummaryLines(raw?.summary)),
        recentMessages,
        updatedAt: typeof raw?.updatedAt === "string" ? raw.updatedAt : null,
    };
};

const getChatMemoryDbKey = (stateKey) => `${CHAT_MEMORY_PREFIX}${stateKey}`;

const resolveChatMemoryScope = (req) => {
    const scoped = getScopedStateKeyFromRequest(req);
    if (!scoped?.stateKey) return null;
    return {
        ...scoped,
        dbKey: getChatMemoryDbKey(scoped.stateKey),
    };
};

const loadScopedChatMemory = async (req) => {
    const scope = resolveChatMemoryScope(req);
    if (!scope) return null;

    const record = await loadAppState(scope.dbKey);
    return {
        ...scope,
        memory: normalizeChatMemory(record?.state),
        updatedAt: record?.updatedAt || null,
    };
};

const getLatestConversationSlice = (messages = [], limit = 2) => (
    dedupeConsecutiveMessages(
        (Array.isArray(messages) ? messages : [])
            .map(normalizeMemoryMessage)
            .filter(Boolean),
    ).slice(-Math.max(1, limit))
);

const mergeChatMemory = (existingMemory, messages = [], assistantReply = "") => {
    const current = normalizeChatMemory(existingMemory);
    const additions = getLatestConversationSlice(messages);
    const assistantMessage = normalizeMemoryMessage({
        role: "assistant",
        content: assistantReply,
    });
    if (assistantMessage) additions.push(assistantMessage);

    let combined = dedupeConsecutiveMessages([
        ...current.recentMessages,
        ...additions,
    ]);

    let summaryLines = parseSummaryLines(current.summary);
    if (combined.length > RECENT_MESSAGE_LIMIT) {
        const overflow = combined.slice(0, -RECENT_MESSAGE_LIMIT);
        summaryLines = [
            ...summaryLines,
            ...overflow.map(buildMessageDigest).filter(Boolean),
        ].slice(-SUMMARY_LINE_LIMIT);
        combined = combined.slice(-RECENT_MESSAGE_LIMIT);
    }

    return {
        version: 1,
        summary: buildSummaryText(summaryLines),
        recentMessages: combined,
        updatedAt: new Date().toISOString(),
    };
};

const saveScopedChatMemory = async (scope, memory) => {
    if (!scope?.dbKey) return null;
    const normalized = normalizeChatMemory(memory);
    const saved = await saveAppState(scope.dbKey, normalized);
    return {
        ...scope,
        memory: normalizeChatMemory(saved?.state),
        updatedAt: saved?.updatedAt || null,
    };
};

const shouldUseScopedChatMemory = (body = {}) => {
    if (!body || body.use_persistent_memory === false || body.memory === false) return false;
    const nonSystemMessages = (Array.isArray(body.messages) ? body.messages : [])
        .filter((message) => message?.role !== "system");
    return nonSystemMessages.length <= 2;
};

const buildMessagesWithScopedMemory = (messages = [], memoryState = {}) => {
    const original = Array.isArray(messages) ? messages : [];
    const systemMessages = original.filter((message) => message?.role === "system");
    const nonSystemMessages = original.filter((message) => message?.role !== "system");
    const memory = normalizeChatMemory(memoryState);

    let restoredRecent = Array.isArray(memory.recentMessages) ? [...memory.recentMessages] : [];
    const currentSlice = getLatestConversationSlice(nonSystemMessages, 2);
    while (restoredRecent.length && currentSlice.length && sameMessage(restoredRecent[restoredRecent.length - 1], currentSlice[0])) {
        restoredRecent = restoredRecent.slice(0, -1);
    }

    const restoredMessages = restoredRecent.map((message) => ({
        role: message.role,
        content: message.content,
    }));

    const summaryMessage = memory.summary
        ? [{
            role: "system",
            content: `${memory.summary}\nUse this persistent memory only when it is relevant to the current request.`,
        }]
        : [];

    return [
        ...systemMessages,
        ...summaryMessage,
        ...restoredMessages,
        ...nonSystemMessages,
    ];
};

module.exports = {
    CHAT_MEMORY_PREFIX,
    RECENT_MESSAGE_LIMIT,
    normalizeChatMemory,
    resolveChatMemoryScope,
    loadScopedChatMemory,
    saveScopedChatMemory,
    mergeChatMemory,
    shouldUseScopedChatMemory,
    buildMessagesWithScopedMemory,
};
