const ONE_MINUTE_MS = 60 * 1000;
const DEFAULT_TPM = 60000;
const DEFAULT_MAX_COMPLETION_TOKENS = 4096;
const MAX_COMPLETION_TOKENS_LIMIT = 8192;
const DEFAULT_REMOTE_HEADROOM = 2000;
const DEFAULT_CACHE_TTL_MS = 5 * 60 * 1000;
const DEFAULT_CACHE_MAX_ENTRIES = 100;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const safeNumber = (value, fallback) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const parseHeaderNumber = (headers, name) => {
    if (!headers) return null;
    const raw = typeof headers.get === "function"
        ? headers.get(name)
        : headers[name] ?? headers[name.toLowerCase()];
    if (raw === undefined || raw === null || raw === "") return null;
    const parsed = Number(raw);
    return Number.isFinite(parsed) ? parsed : null;
};

const parseRetryAfterMs = (headers) => {
    const retryAfter = parseHeaderNumber(headers, "retry-after");
    if (Number.isFinite(retryAfter) && retryAfter > 0) {
        return retryAfter * 1000;
    }
    return null;
};

const roughTokenEstimate = (value) => {
    const text = String(value ?? "");
    if (!text) return 0;
    return Math.max(1, Math.ceil(text.length / 4));
};

const estimateContentTokens = (content) => {
    if (typeof content === "string") return roughTokenEstimate(content);
    if (!Array.isArray(content)) return 0;

    let total = 0;
    for (const part of content) {
        if (!part) continue;
        if (part.type === "text") {
            total += roughTokenEstimate(part.text || "");
        } else if (part.type === "image_url") {
            total += 1200;
        } else {
            total += roughTokenEstimate(JSON.stringify(part));
        }
    }
    return total;
};

const estimateMessagesTokens = (messages = []) => {
    let total = 2;
    for (const message of Array.isArray(messages) ? messages : []) {
        if (!message) continue;
        total += 4;
        total += estimateContentTokens(message.content);
    }
    return total;
};

const estimatePayloadTokens = (payload = {}) => {
    const messageTokens = estimateMessagesTokens(payload.messages);
    const toolTokens = Array.isArray(payload.tools) && payload.tools.length
        ? roughTokenEstimate(JSON.stringify(payload.tools))
        : 0;
    const maxTokens = clampCompletionTokens(payload.max_tokens ?? payload.maxTokens);
    return messageTokens + toolTokens + maxTokens;
};

const clampCompletionTokens = (value, fallback = DEFAULT_MAX_COMPLETION_TOKENS) => {
    const parsed = safeNumber(value, fallback);
    return clamp(Math.round(parsed), 1, MAX_COMPLETION_TOKENS_LIMIT);
};

const extractUsage = (json = {}) => {
    const usage = json?.usage;
    if (usage && Number.isFinite(Number(usage.total_tokens))) {
        return {
            prompt_tokens: Number(usage.prompt_tokens) || 0,
            completion_tokens: Number(usage.completion_tokens) || 0,
            total_tokens: Number(usage.total_tokens) || 0,
        };
    }

    const promptTokens = Number(usage?.prompt_tokens) || 0;
    const completionTokens = Number(usage?.completion_tokens) || 0;
    if (promptTokens || completionTokens) {
        return {
            prompt_tokens: promptTokens,
            completion_tokens: completionTokens,
            total_tokens: promptTokens + completionTokens,
        };
    }

    return null;
};

const mergeUsage = (items = []) => {
    const total = items.reduce((acc, usage) => {
        if (!usage) return acc;
        acc.prompt_tokens += Number(usage.prompt_tokens) || 0;
        acc.completion_tokens += Number(usage.completion_tokens) || 0;
        acc.total_tokens += Number(usage.total_tokens) || 0;
        return acc;
    }, { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 });

    return total.total_tokens ? total : null;
};

const getBackoffDelayMs = (attempt, headers) => {
    const retryAfterMs = parseRetryAfterMs(headers);
    if (retryAfterMs) return retryAfterMs;

    const resetSeconds = parseHeaderNumber(headers, "x-ratelimit-reset-tokens-minute");
    if (Number.isFinite(resetSeconds) && resetSeconds > 0) {
        return Math.ceil(resetSeconds * 1000);
    }

    const base = Math.min(1000 * (2 ** attempt), 8000);
    return base + Math.floor(Math.random() * 250);
};

class SlidingWindowTokenLimiter {
    constructor({
        tokensPerMinute = DEFAULT_TPM,
        remoteHeadroom = DEFAULT_REMOTE_HEADROOM,
    } = {}) {
        this.tokensPerMinute = Math.max(1000, safeNumber(tokensPerMinute, DEFAULT_TPM));
        this.remoteHeadroom = Math.max(0, safeNumber(remoteHeadroom, DEFAULT_REMOTE_HEADROOM));
        this.window = [];
        this.remoteRemaining = null;
        this.remoteResetAt = 0;
        this.queue = Promise.resolve();
    }

    prune(now = Date.now()) {
        while (this.window.length && this.window[0].timestamp <= now - ONE_MINUTE_MS) {
            this.window.shift();
        }
        if (this.remoteResetAt && now >= this.remoteResetAt) {
            this.remoteRemaining = null;
            this.remoteResetAt = 0;
        }
    }

    usedTokens(now = Date.now()) {
        this.prune(now);
        return this.window.reduce((sum, entry) => sum + entry.tokens, 0);
    }

    computeRemoteDelay(estimatedTokens, now = Date.now()) {
        this.prune(now);
        if (!this.remoteResetAt || now >= this.remoteResetAt) return 0;

        const remaining = Number.isFinite(this.remoteRemaining)
            ? this.remoteRemaining
            : this.tokensPerMinute;
        const available = Math.max(0, remaining - this.remoteHeadroom);
        if (available <= 0 || estimatedTokens > available) {
            return Math.max(25, this.remoteResetAt - now);
        }
        return 0;
    }

    reserve(estimatedTokens) {
        const size = Math.max(1, Math.ceil(safeNumber(estimatedTokens, 1)));
        const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
        const work = async () => {
            while (true) {
                const now = Date.now();
                const remoteDelay = this.computeRemoteDelay(size, now);
                if (remoteDelay > 0) {
                    await sleep(remoteDelay);
                    continue;
                }

                const used = this.usedTokens(now);
                if (used + size <= this.tokensPerMinute) {
                    this.window.push({ id, timestamp: now, tokens: size });
                    return id;
                }

                const oldest = this.window[0];
                const waitMs = oldest
                    ? Math.max(25, oldest.timestamp + ONE_MINUTE_MS - now)
                    : 50;
                await sleep(waitMs);
            }
        };

        const pending = this.queue.then(work, work);
        this.queue = pending.then(() => undefined, () => undefined);
        return pending;
    }

    commit(id, actualTokens) {
        if (!id) return;
        const nextTokens = Math.max(1, Math.ceil(safeNumber(actualTokens, 1)));
        const entry = this.window.find((item) => item.id === id);
        if (entry) entry.tokens = nextTokens;
    }

    release(id) {
        if (!id) return;
        this.window = this.window.filter((entry) => entry.id !== id);
    }

    applyHeaders(headers) {
        const remaining = parseHeaderNumber(headers, "x-ratelimit-remaining-tokens-minute");
        const resetSeconds = parseHeaderNumber(headers, "x-ratelimit-reset-tokens-minute");
        if (Number.isFinite(remaining)) {
            this.remoteRemaining = Math.max(0, remaining);
        }
        if (Number.isFinite(resetSeconds) && resetSeconds > 0) {
            this.remoteResetAt = Date.now() + Math.ceil(resetSeconds * 1000);
        }
    }
}

class ExactResponseCache {
    constructor({
        ttlMs = DEFAULT_CACHE_TTL_MS,
        maxEntries = DEFAULT_CACHE_MAX_ENTRIES,
    } = {}) {
        this.ttlMs = Math.max(1000, safeNumber(ttlMs, DEFAULT_CACHE_TTL_MS));
        this.maxEntries = Math.max(1, safeNumber(maxEntries, DEFAULT_CACHE_MAX_ENTRIES));
        this.store = new Map();
    }

    makeKey(parts) {
        return JSON.stringify(parts);
    }

    get(key) {
        const entry = this.store.get(key);
        if (!entry) return null;
        if (entry.expiresAt <= Date.now()) {
            this.store.delete(key);
            return null;
        }
        return entry.value;
    }

    set(key, value) {
        this.prune();
        this.store.set(key, {
            value,
            expiresAt: Date.now() + this.ttlMs,
        });
    }

    prune() {
        const now = Date.now();
        for (const [key, entry] of this.store.entries()) {
            if (entry.expiresAt <= now) this.store.delete(key);
        }
        while (this.store.size >= this.maxEntries) {
            const firstKey = this.store.keys().next().value;
            if (!firstKey) break;
            this.store.delete(firstKey);
        }
    }
}

module.exports = {
    SlidingWindowTokenLimiter,
    ExactResponseCache,
    clampCompletionTokens,
    estimatePayloadTokens,
    extractUsage,
    getBackoffDelayMs,
    mergeUsage,
    parseHeaderNumber,
    sleep,
};
