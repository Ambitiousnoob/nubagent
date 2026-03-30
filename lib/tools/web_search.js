const { normalizeUrl, stripHtml } = require("../web");
const { getApiKeysFromEnv, getRotatingApiKey } = require("../api-key-rotation.cjs");
const {
    canonicalizeSourceUrl,
    extractQueryTerms,
    scoreTextForTerms,
    domainAuthorityBoost,
    getSourceDomain,
} = require("../rag");

const MAX_RESULTS = 30;
const REQUEST_TIMEOUT_MS = 8000;
const DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/";
const TAVILY_SEARCH_URL = "https://api.tavily.com/search";
const SERPER_SEARCH_URL = "https://google.serper.dev/search";
const JINA_SEARCH_URL = "https://s.jina.ai/";
const BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search";
const EXA_SEARCH_URL = "https://api.exa.ai/search";
const RESULT_LINK_RE = /<a\b[^>]*class=(?:"[^"]*\b(?:result__a|result-link)\b[^"]*"|'[^']*\b(?:result__a|result-link)\b[^']*')[^>]*href=(?:"([^"]+)"|'([^']+)')[^>]*>([\s\S]*?)<\/a>/gi;
const RESULT_SNIPPET_RE = /<(?:a|div|span)\b[^>]*class=(?:"[^"]*\b(?:result__snippet|result-snippet)\b[^"]*"|'[^']*\b(?:result__snippet|result-snippet)\b[^']*')[^>]*>([\s\S]*?)<\/(?:a|div|span)>/i;
const GOOGLE_ONLY_OPERATORS_RE = /\b(intitle:|inurl:|intext:|before:|after:|filetype:(?!pdf))/i;
const ANY_OPERATOR_RE = /\b(site:|filetype:|intitle:|inurl:|intext:|before:|after:)/i;

const normalizeText = (value) => (
    String(value || "")
        .replace(/\u0000/g, "")
        .replace(/\s+/g, " ")
        .trim()
);

const cleanHtmlText = (value) => normalizeText(stripHtml(String(value || "")));

const normalizeMatchText = (value) => (
    String(value || "")
        .toLowerCase()
        .normalize("NFKD")
        .replace(/[^\p{L}\p{N}\s]/gu, " ")
        .replace(/\s+/g, " ")
        .trim()
);

const countMatchedTerms = (text, terms) => {
    if (!text || !terms.length) return 0;
    const normalized = normalizeMatchText(text);
    let count = 0;
    for (const term of terms) {
        if (normalized.includes(term)) count += 1;
    }
    return count;
};

const extractQueryPhrases = (query, terms) => {
    const phrases = new Set();
    const normalizedQuery = normalizeMatchText(query);

    if (normalizedQuery && terms.length >= 2 && terms.length <= 8) {
        phrases.add(normalizedQuery);
    }

    const quoted = String(query || "").match(/"([^"]+)"/g) || [];
    for (const rawPhrase of quoted) {
        const normalized = normalizeMatchText(rawPhrase.slice(1, -1));
        if (normalized && normalized.split(/\s+/).length >= 2) {
            phrases.add(normalized);
        }
        if (phrases.size >= 4) break;
    }

    return [...phrases].slice(0, 4);
};

const queryAwareMatchBoost = (query, item, terms) => {
    if (!terms.length) return 0;

    const title = normalizeMatchText(item?.title || "");
    const description = normalizeMatchText(item?.description || "");
    const url = normalizeMatchText(item?.url || "");
    const phrases = extractQueryPhrases(query, terms);
    const titleCoverage = countMatchedTerms(title, terms) / terms.length;
    const descriptionCoverage = countMatchedTerms(description, terms) / terms.length;
    const urlCoverage = countMatchedTerms(url, terms) / terms.length;
    let phraseBoost = 0;

    for (const phrase of phrases) {
        if (title.includes(phrase)) phraseBoost += 1.9;
        else if (description.includes(phrase)) phraseBoost += 0.8;
        else if (url.includes(phrase)) phraseBoost += 0.35;
    }

    const completeTitleMatch = terms.length >= 2 && titleCoverage === 1 ? 1.1 : 0;

    return (
        phraseBoost +
        (titleCoverage * 1.6) +
        (descriptionCoverage * 0.85) +
        (urlCoverage * 0.35) +
        completeTitleMatch
    );
};

const getUrlPathDepth = (url) => {
    try {
        return new URL(canonicalizeSourceUrl(url)).pathname.split("/").filter(Boolean).length;
    } catch {
        return 0;
    }
};

const lowSignalPagePenalty = (item, terms) => {
    if (!terms.length) return 0;
    const titleMatches = countMatchedTerms(item?.title || "", terms);
    const descriptionMatches = countMatchedTerms(item?.description || "", terms);
    const pathDepth = getUrlPathDepth(item?.url || "");

    if (pathDepth <= 1 && titleMatches <= 1 && descriptionMatches <= 1) {
        return 0.55;
    }

    return 0;
};

const normalizeSourcePathForDedupe = (pathname = "") => {
    let next = String(pathname || "").replace(/\/{2,}/g, "/");
    next = next.replace(/\/index(?:\.[a-z0-9]+)?$/i, "/");
    next = next.replace(/\/(?:amp|amphtml)\/?$/i, "/");
    return next !== "/" ? next.replace(/\/+$/, "") || "/" : "/";
};

const normalizeTitleForDedupe = (title = "") => (
    normalizeMatchText(title)
        .split(/\s+/)
        .filter((term) => term.length > 1)
        .slice(0, 10)
        .join(" ")
);

const buildNearDuplicateKey = (url, title = "") => {
    try {
        const parsed = new URL(canonicalizeSourceUrl(url));
        if (parsed.search) return "";

        const host = parsed.hostname.replace(/^(?:www\.|m\.)/, "").toLowerCase();
        const path = normalizeSourcePathForDedupe(parsed.pathname);
        const titleKey = normalizeTitleForDedupe(title);

        if (!host || !titleKey) return "";
        return `${host}${path}::${titleKey}`;
    } catch {
        return "";
    }
};

const scoreUrlPreference = (url) => {
    try {
        const parsed = new URL(canonicalizeSourceUrl(url));
        let score = 0;
        if (parsed.protocol === "https:") score += 0.35;
        if (!parsed.search) score += 0.6;
        if (/^(?:www\.|m\.)/i.test(parsed.hostname)) score -= 0.05;
        if (/(^|\/)(amp|amphtml)(\/|$)/i.test(parsed.pathname)) score -= 0.5;
        score -= parsed.toString().length / 500;
        return score;
    } catch {
        return Number.NEGATIVE_INFINITY;
    }
};

const pickPreferredUrl = (current, candidate) => {
    const left = canonicalizeSourceUrl(current);
    const right = canonicalizeSourceUrl(candidate);
    if (!left) return right;
    if (!right) return left;
    if (left === right) return left;

    const leftScore = scoreUrlPreference(left);
    const rightScore = scoreUrlPreference(right);
    if (rightScore !== leftScore) {
        return rightScore > leftScore ? right : left;
    }

    return right.length < left.length ? right : left;
};

const domainCrowdingPenalty = (domainCounts, domain, item) => {
    const seen = domainCounts.get(domain) || 0;
    if (!domain || !seen) return 0;

    let penalty = 1.05 + ((seen - 1) * 0.8);
    if (getUrlPathDepth(item?.url || "") <= 1) {
        penalty += 0.2;
    }

    return penalty;
};

const decodeDuckDuckGoUrl = (value) => {
    const raw = String(value || "").trim();
    if (!raw) return "";

    try {
        const parsed = new URL(raw, DUCKDUCKGO_HTML_URL);
        const redirected = parsed.searchParams.get("uddg");
        if (redirected) {
            return normalizeUrl(decodeURIComponent(redirected)) || decodeURIComponent(redirected);
        }
        return normalizeUrl(parsed.toString()) || "";
    } catch {
        return "";
    }
};

const withTimeout = async (fn, timeoutMs = REQUEST_TIMEOUT_MS) => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
        return await fn(controller.signal);
    } finally {
        clearTimeout(timer);
    }
};

const hasDorkOperators = (query) => ANY_OPERATOR_RE.test(String(query || ""));

const needsGoogle = (query) => GOOGLE_ONLY_OPERATORS_RE.test(String(query || ""));

const getTavilyApiKey = () => getRotatingApiKey("tavily", "TAVILY_API_KEYS", "TAVILY_API_KEY");
const getSerperApiKeys = () => getApiKeysFromEnv("SERPER_API_KEYS", "SERPER_API_KEY");
const getSerperApiKey = () => getRotatingApiKey("serper", "SERPER_API_KEYS", "SERPER_API_KEY");
const getJinaApiKeys = () => getApiKeysFromEnv("JINA_API_KEYS", "JINA_API_KEY");
const getJinaApiKey = () => getRotatingApiKey("jina", "JINA_API_KEYS", "JINA_API_KEY");
const getBraveApiKeys = () => getApiKeysFromEnv("BRAVE_API_KEYS", "BRAVE_API_KEY");
const getBraveApiKey = () => getRotatingApiKey("brave", "BRAVE_API_KEYS", "BRAVE_API_KEY");
const resolveProviderApiKey = (explicitValue, fallback) => normalizeText(explicitValue) || fallback();

const parseDuckDuckGoResults = (html, limit = MAX_RESULTS) => {
    const results = [];
    const seen = new Set();
    let match;

    while ((match = RESULT_LINK_RE.exec(String(html || ""))) !== null && results.length < limit) {
        const url = decodeDuckDuckGoUrl(match[1] || match[2] || "");
        const title = cleanHtmlText(match[3]);
        if (!url || !title || seen.has(url)) continue;

        const snippetWindow = html.slice(match.index, Math.min(html.length, match.index + 2500));
        const description = cleanHtmlText(snippetWindow.match(RESULT_SNIPPET_RE)?.[1] || "");

        seen.add(url);
        results.push({
            title,
            url,
            description: description || "",
            source: "duckduckgo",
        });
    }

    return results;
};

const fetchDuckDuckGoResults = async (query, limit = MAX_RESULTS) => (
    withTimeout(async (signal) => {
        const body = new URLSearchParams({
            q: query,
            kl: "us-en",
        });

        const response = await fetch(DUCKDUCKGO_HTML_URL, {
            method: "POST",
            signal,
            headers: {
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "Accept": "text/html,application/xhtml+xml",
                "User-Agent": "nub-agent/1.0",
            },
            body: body.toString(),
        });

        if (!response.ok) {
            throw new Error(`DuckDuckGo returned status ${response.status}`);
        }

        const html = await response.text();
        if (/Unfortunately, bots use DuckDuckGo too\./i.test(html) || /anomaly-modal/i.test(html)) {
            throw new Error("DuckDuckGo returned bot challenge");
        }
        return parseDuckDuckGoResults(html, limit);
    })
);

const fetchTavilyResults = async (query, limit = MAX_RESULTS, explicitApiKey = "") => {
    const apiKey = resolveProviderApiKey(explicitApiKey, getTavilyApiKey);
    if (!apiKey) return [];

    return withTimeout(async (signal) => {
        const response = await fetch(TAVILY_SEARCH_URL, {
            method: "POST",
            signal,
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                api_key: apiKey,
                query,
                search_depth: "advanced",
                include_answer: true,
                include_raw_content: false,
                max_results: limit,
                topic: "general",
            }),
        });

        if (!response.ok) {
            throw new Error(`Tavily returned status ${response.status}`);
        }

        const data = await response.json();
        const results = (Array.isArray(data?.results) ? data.results : []).map((item) => ({
            title: normalizeText(item?.title),
            url: normalizeUrl(item?.url) || String(item?.url || "").trim(),
            description: normalizeText(item?.content || item?.description || ""),
            score: item?.score,
            source: "tavily",
        })).filter((item) => item.title && item.url);

        // Include Tavily's AI answer if available
        if (data?.answer) {
            results.unshift({
                title: "AI Answer",
                url: "",
                description: normalizeText(data.answer),
                score: 1.0,
                source: "tavily-answer",
            });
        }

        return results;
    });
};

const fetchSerperResults = async (query, limit = MAX_RESULTS, explicitApiKey = "") => {
    const apiKey = resolveProviderApiKey(explicitApiKey, getSerperApiKey);
    if (!apiKey) return [];

    return withTimeout(async (signal) => {
        const response = await fetch(SERPER_SEARCH_URL, {
            method: "POST",
            signal,
            headers: {
                "X-API-KEY": apiKey,
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                q: query,
                num: Math.min(limit, 10),
                gl: "us",
                hl: "en",
            }),
        });

        if (!response.ok) {
            throw new Error(`Serper returned status ${response.status}`);
        }

        const data = await response.json();
        const results = [];

        if (data?.answerBox?.answer || data?.answerBox?.snippet) {
            results.push({
                title: normalizeText(data.answerBox.title || "Answer"),
                url: normalizeUrl(data.answerBox.link) || String(data.answerBox.link || "").trim(),
                description: normalizeText(data.answerBox.answer || data.answerBox.snippet || ""),
                source: "serper-answerbox",
            });
        }

        for (const item of Array.isArray(data?.organic) ? data.organic : []) {
            results.push({
                title: normalizeText(item?.title),
                url: normalizeUrl(item?.link) || String(item?.link || "").trim(),
                description: normalizeText(item?.snippet || ""),
                date: normalizeText(item?.date),
                position: item?.position,
                source: "serper",
            });
            if (results.length >= limit) break;
        }

        // Include knowledge graph if available
        if (data?.knowledgeGraph) {
            const kg = data.knowledgeGraph;
            if (kg.description) {
                results.push({
                    title: normalizeText(kg.title || "Knowledge"),
                    url: normalizeUrl(kg.website) || "",
                    description: normalizeText(kg.description),
                    source: "serper-knowledge",
                });
            }
        }

        return results.filter((item) => item.title && item.url);
    });
};

const fetchJinaResults = async (query, limit = MAX_RESULTS, explicitApiKey = "") => {
    const apiKey = resolveProviderApiKey(explicitApiKey, getJinaApiKey);
    if (!apiKey) return [];

    return withTimeout(async (signal) => {
        const response = await fetch(`${JINA_SEARCH_URL}?q=${encodeURIComponent(query)}`, {
            method: "GET",
            signal,
            headers: {
                "Authorization": `Bearer ${apiKey}`,
                "X-Respond-With": "no-content",
                "Accept": "application/json",
            },
        });

        if (!response.ok) {
            throw new Error(`Jina search returned status ${response.status}`);
        }

        const data = await response.json();
        return (Array.isArray(data?.data) ? data.data : [])
            .slice(0, limit)
            .map((item) => ({
                title: normalizeText(item?.title),
                url: normalizeUrl(item?.url) || String(item?.url || "").trim(),
                description: normalizeText(item?.description || item?.content || ""),
                source: "jina",
            }))
            .filter((item) => item.title && item.url);
    });
};

const fetchBraveResults = async (query, limit = MAX_RESULTS, explicitApiKey = "") => {
    const apiKey = resolveProviderApiKey(explicitApiKey, getBraveApiKey);
    if (!apiKey) return [];

    return withTimeout(async (signal) => {
        const response = await fetch(`${BRAVE_SEARCH_URL}?q=${encodeURIComponent(query)}&count=${Math.min(limit, 20)}`, {
            method: "GET",
            signal,
            headers: {
                "Accept": "application/json",
                "X-Subscription-Token": apiKey,
            },
        });

        if (!response.ok) {
            throw new Error(`Brave search returned status ${response.status}`);
        }

        const data = await response.json();
        return (Array.isArray(data?.web?.results) ? data.web.results : [])
            .slice(0, limit)
            .map((item) => ({
                title: normalizeText(item?.title),
                url: normalizeUrl(item?.url) || String(item?.url || "").trim(),
                description: normalizeText(item?.description || ""),
                age: normalizeText(item?.age),
                source: "brave",
            }))
            .filter((item) => item.title && item.url);
    });
};

const getNumericScore = (value) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 0;
};

const mergeResultEntries = (existing, next) => {
    const existingProviders = Array.isArray(existing?.providers)
        ? existing.providers
        : [existing?.source].filter(Boolean);
    const nextProviders = Array.isArray(next?.providers)
        ? next.providers
        : [next?.source].filter(Boolean);
    const providers = [...new Set([...existingProviders, ...nextProviders].filter(Boolean))];
    const existingDescriptionLength = normalizeText(existing?.description || "").length;
    const nextDescriptionLength = normalizeText(next?.description || "").length;
    return {
        ...existing,
        ...next,
        title: normalizeText(next?.title || "").length > normalizeText(existing?.title || "").length ? next.title : existing.title,
        description: nextDescriptionLength > existingDescriptionLength ? next.description : existing.description,
        date: existing?.date || next?.date || "",
        age: existing?.age || next?.age || "",
        score: Math.max(getNumericScore(existing?.score), getNumericScore(next?.score)) || undefined,
        source: providers.length > 1 ? "multi" : (providers[0] || existing?.source || next?.source || ""),
        providers,
        providerCount: providers.length || 1,
    };
};

const mergeResults = (...sources) => {
    const mergedByKey = new Map();
    const canonicalUrlToKey = new Map();
    const nearDuplicateToKey = new Map();

    for (const source of sources) {
        for (const item of Array.isArray(source) ? source : []) {
            const canonicalUrl = canonicalizeSourceUrl(item?.url || "");
            if (!canonicalUrl || !item?.title) continue;

            const normalizedItem = canonicalUrl === item.url ? item : { ...item, url: canonicalUrl };
            const nearDuplicateKey = buildNearDuplicateKey(canonicalUrl, normalizedItem.title);
            const mergeKey = canonicalUrlToKey.get(canonicalUrl) || (nearDuplicateKey ? nearDuplicateToKey.get(nearDuplicateKey) : "") || canonicalUrl;
            const merged = mergeResultEntries(mergedByKey.get(mergeKey), normalizedItem);
            merged.url = pickPreferredUrl(mergedByKey.get(mergeKey)?.url, normalizedItem.url) || merged.url;
            merged.source = merged.providers.length > 1 ? "multi" : (merged.providers[0] || merged.source || "");
            mergedByKey.set(mergeKey, merged);
            canonicalUrlToKey.set(canonicalUrl, mergeKey);
            if (nearDuplicateKey) nearDuplicateToKey.set(nearDuplicateKey, mergeKey);
        }
    }

    return [...mergedByKey.values()];
};

const rankMergedResults = (query, items = [], limit = MAX_RESULTS) => {
    const list = Array.isArray(items) ? items : [];
    const terms = extractQueryTerms(query);
    const scored = list.map((item, position) => {
        const providers = Array.isArray(item?.providers) ? item.providers.filter(Boolean) : [item?.source].filter(Boolean);
        const blob = `${item?.title || ""} ${item?.description || ""} ${item?.url || ""}`;
        const lexical = scoreTextForTerms(blob, terms);
        const queryMatch = queryAwareMatchBoost(query, item, terms);
        const authority = domainAuthorityBoost(item?.url || "");
        const providerAgreement = Math.max(0, providers.length - 1) * 1.35;
        const providerScore = Math.max(0, Math.min(1.5, getNumericScore(item?.score)));
        const descriptionBoost = Math.min(0.35, normalizeText(item?.description || "").length / 320);
        const recency = Math.max(0, (list.length - position) / Math.max(list.length, 1)) * 0.2;
        const lowSignalPenalty = lowSignalPagePenalty(item, terms);
        return {
            item: {
                ...item,
                providers,
                providerCount: providers.length || 1,
            },
            position,
            domain: getSourceDomain(item?.url || "") || "__unknown__",
            baseScore: lexical + queryMatch + authority + providerAgreement + providerScore + descriptionBoost + recency - lowSignalPenalty,
        };
    });

    const remaining = [...scored];
    const ranked = [];
    const domainCounts = new Map();

    while (remaining.length && ranked.length < limit) {
        remaining.sort((a, b) => {
            const aPenalty = domainCrowdingPenalty(domainCounts, a.domain, a.item);
            const bPenalty = domainCrowdingPenalty(domainCounts, b.domain, b.item);
            const aScore = a.baseScore - aPenalty;
            const bScore = b.baseScore - bPenalty;
            if (bScore !== aScore) return bScore - aScore;
            return a.position - b.position;
        });

        const next = remaining.shift();
        ranked.push(next.item);
        domainCounts.set(next.domain, (domainCounts.get(next.domain) || 0) + 1);
    }

    return ranked;
};

const formatResults = (query, combined, settledResults) => {
    if (combined.length) {
        return JSON.stringify(combined.map((item, index) => ({
            rank: index + 1,
            title: item.title,
            url: item.url,
            description: item.description,
            ...(item.date ? { date: item.date } : {}),
            ...(item.age ? { age: item.age } : {}),
            ...(typeof item.score === "number" ? { score: item.score } : {}),
            ...(item.providerCount ? { providerCount: item.providerCount } : {}),
            ...(Array.isArray(item.providers) && item.providers.length ? { providers: item.providers } : {}),
            source: item.source,
        })));
    }

    const errors = (Array.isArray(settledResults) ? settledResults : [])
        .filter((item) => item?.status === "rejected")
        .map((item) => item.reason?.message || String(item.reason))
        .filter(Boolean);

    return errors.length
        ? `Error: search failed (${errors.join("; ")})`
        : `No results for "${query}"`;
};

/**
 * Tool definition for function calling
 */
const definition = {
    type: "function",
    function: {
        name: "web_search",
        strict: true,
        description: "Search the web with multiple backends. Supports Google dork operators: site:domain.com, filetype:pdf, intitle:keyword, inurl:keyword, intext:keyword, after:YYYY-MM-DD, before:YYYY-MM-DD, exact phrases, -exclude, OR. Auto-routes to Google when operators detected. Returns up to 30 ranked results with titles, URLs, descriptions, and dates.",
        parameters: {
            type: "object",
            properties: {
                query: {
                    type: "string",
                    description: "The search query. Include timezone/location for time-sensitive queries.",
                },
                maxResults: {
                    type: "number",
                    description: "Maximum number of results to return. Default: 30. Max: 30.",
                },
            },
            required: ["query"],
            additionalProperties: false,
        },
    },
};

/**
 * Web search handler function
 * @param {object} args - Search arguments
 * @param {string} args.query - Search query
 * @returns {Promise<string>} JSON string of results or error message
 */
const handler = async (args) => {
    const query = String(args.query || "").trim();
    if (!query) return "Error: query is required";
    const maxResults = Math.max(1, Math.min(Number(args.maxResults) || MAX_RESULTS, MAX_RESULTS));
    const providerApiKeys = args?.providerApiKeys && typeof args.providerApiKeys === "object"
        ? args.providerApiKeys
        : {};

    const usesOperators = hasDorkOperators(query);
    const requiresGoogle = needsGoogle(query);
    const hasSerper = Boolean(normalizeText(providerApiKeys.serper)) || getSerperApiKeys().length > 0;
    const hasBrave = Boolean(normalizeText(providerApiKeys.brave)) || getBraveApiKeys().length > 0;
    const hasJina = Boolean(normalizeText(providerApiKeys.jina)) || getJinaApiKeys().length > 0;

    let searches;

    // Google operators detected + Serper available
    if (requiresGoogle && hasSerper) {
        searches = await Promise.allSettled([
            fetchSerperResults(query, maxResults, providerApiKeys.serper),
        ]);
        const [serper] = searches;
        return formatResults(
            query,
            rankMergedResults(query, mergeResults(
                serper.status === "fulfilled" ? serper.value : [],
            ), maxResults),
            searches,
        );
    }

    // Dork operators + Serper available
    if (usesOperators && hasSerper) {
        searches = await Promise.allSettled([
            fetchSerperResults(query, maxResults, providerApiKeys.serper),
            fetchDuckDuckGoResults(query, maxResults),
        ]);
        const [serper, duckDuckGo] = searches;
        return formatResults(
            query,
            rankMergedResults(query, mergeResults(
                serper.status === "fulfilled" ? serper.value : [],
                duckDuckGo.status === "fulfilled" ? duckDuckGo.value : [],
            ), maxResults),
            searches,
        );
    }

    // Standard search with all available backends
    const promises = [
        fetchDuckDuckGoResults(query, maxResults),
        fetchTavilyResults(query, maxResults, providerApiKeys.tavily),
    ];

    if (hasSerper) {
        promises.push(fetchSerperResults(query, Math.min(maxResults, 10), providerApiKeys.serper));
    }

    if (hasJina) {
        promises.push(fetchJinaResults(query, Math.min(maxResults, 10), providerApiKeys.jina));
    }

    if (hasBrave) {
        promises.push(fetchBraveResults(query, Math.min(maxResults, 10), providerApiKeys.brave));
    }

    searches = await Promise.allSettled(promises);

    const [duckDuckGo, tavily, serper, jina, brave] = searches;
    return formatResults(
        query,
        rankMergedResults(query, mergeResults(
            duckDuckGo.status === "fulfilled" ? duckDuckGo.value : [],
            tavily.status === "fulfilled" ? tavily.value : [],
            serper?.status === "fulfilled" ? serper.value : [],
            jina?.status === "fulfilled" ? jina.value : [],
            brave?.status === "fulfilled" ? brave.value : [],
        ), maxResults),
        searches,
    );
};

module.exports = {
    definition,
    handler,
};
