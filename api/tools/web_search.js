const { normalizeUrl, stripHtml } = require("../../lib/web");

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
let tavilyApiKeyIndex = 0;
let serperApiKeyIndex = 0;
let jinaApiKeyIndex = 0;
let braveApiKeyIndex = 0;

const normalizeText = (value) => (
    String(value || "")
        .replace(/\u0000/g, "")
        .replace(/\s+/g, " ")
        .trim()
);

const cleanHtmlText = (value) => normalizeText(stripHtml(String(value || "")));

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

const getTavilyApiKey = () => {
    const raw = process.env.TAVILY_API_KEYS || process.env.TAVILY_API_KEY;
    const keys = String(raw || "").split(",").map((key) => key.trim()).filter(Boolean);
    if (!keys.length) return null;
    const nextKey = keys[tavilyApiKeyIndex % keys.length];
    tavilyApiKeyIndex = (tavilyApiKeyIndex + 1) % keys.length;
    return nextKey;
};

const getSerperApiKeys = () => (
    String(process.env.SERPER_API_KEYS || process.env.SERPER_API_KEY || "")
        .split(",")
        .map((key) => key.trim())
        .filter(Boolean)
);

const getSerperApiKey = () => {
    const keys = getSerperApiKeys();
    if (!keys.length) return null;
    const nextKey = keys[serperApiKeyIndex % keys.length];
    serperApiKeyIndex = (serperApiKeyIndex + 1) % keys.length;
    return nextKey;
};

const getJinaApiKeys = () => (
    String(process.env.JINA_API_KEYS || process.env.JINA_API_KEY || "")
        .split(",")
        .map((key) => key.trim())
        .filter(Boolean)
);

const getJinaApiKey = () => {
    const keys = getJinaApiKeys();
    if (!keys.length) return null;
    const nextKey = keys[jinaApiKeyIndex % keys.length];
    jinaApiKeyIndex = (jinaApiKeyIndex + 1) % keys.length;
    return nextKey;
};

const getBraveApiKeys = () => (
    String(process.env.BRAVE_API_KEYS || process.env.BRAVE_API_KEY || "")
        .split(",")
        .map((key) => key.trim())
        .filter(Boolean)
);

const getBraveApiKey = () => {
    const keys = getBraveApiKeys();
    if (!keys.length) return null;
    const nextKey = keys[braveApiKeyIndex % keys.length];
    braveApiKeyIndex = (braveApiKeyIndex + 1) % keys.length;
    return nextKey;
};

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
            description: description || "No description available",
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
        return parseDuckDuckGoResults(html, limit);
    })
);

const fetchTavilyResults = async (query, limit = MAX_RESULTS) => {
    const apiKey = getTavilyApiKey();
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
            description: normalizeText(item?.content || item?.description || "") || "No description available",
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

const fetchSerperResults = async (query, limit = MAX_RESULTS) => {
    const apiKey = getSerperApiKey();
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
                description: normalizeText(item?.snippet || "") || "No description available",
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

const fetchJinaResults = async (query, limit = MAX_RESULTS) => {
    const apiKey = getJinaApiKey();
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
                description: normalizeText(item?.description || item?.content || "") || "No description available",
                source: "jina",
            }))
            .filter((item) => item.title && item.url);
    });
};

const fetchBraveResults = async (query, limit = MAX_RESULTS) => {
    const apiKey = getBraveApiKey();
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
                description: normalizeText(item?.description || "") || "No description available",
                age: normalizeText(item?.age),
                source: "brave",
            }))
            .filter((item) => item.title && item.url);
    });
};

const mergeResults = (...sources) => {
    const merged = [];
    const seen = new Set();

    for (const source of sources) {
        for (const item of Array.isArray(source) ? source : []) {
            const key = `${item.url}::${item.title}`.toLowerCase();
            if (!item?.url || !item?.title || seen.has(key)) continue;
            seen.add(key);
            merged.push(item);
            if (merged.length >= MAX_RESULTS) return merged;
        }
    }

    return merged;
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

    const usesOperators = hasDorkOperators(query);
    const requiresGoogle = needsGoogle(query);
    const hasSerper = getSerperApiKeys().length > 0;
    const hasBrave = getBraveApiKeys().length > 0;

    let searches;

    // Google operators detected + Serper available
    if (requiresGoogle && hasSerper) {
        searches = await Promise.allSettled([
            fetchSerperResults(query, MAX_RESULTS),
        ]);
        const [serper] = searches;
        return formatResults(
            query,
            mergeResults(
                serper.status === "fulfilled" ? serper.value : [],
            ).slice(0, MAX_RESULTS),
            searches,
        );
    }

    // Dork operators + Serper available
    if (usesOperators && hasSerper) {
        searches = await Promise.allSettled([
            fetchSerperResults(query, MAX_RESULTS),
            fetchDuckDuckGoResults(query, MAX_RESULTS),
        ]);
        const [serper, duckDuckGo] = searches;
        return formatResults(
            query,
            mergeResults(
                serper.status === "fulfilled" ? serper.value : [],
                duckDuckGo.status === "fulfilled" ? duckDuckGo.value : [],
            ).slice(0, MAX_RESULTS),
            searches,
        );
    }

    // Standard search with all available backends
    const promises = [
        fetchDuckDuckGoResults(query, MAX_RESULTS),
        fetchTavilyResults(query, MAX_RESULTS),
    ];

    if (hasSerper) {
        promises.push(fetchSerperResults(query, 10));
    }

    if (getJinaApiKeys().length) {
        promises.push(fetchJinaResults(query, 10));
    }

    if (hasBrave) {
        promises.push(fetchBraveResults(query, 10));
    }

    searches = await Promise.allSettled(promises);

    const [duckDuckGo, tavily, serper, jina, brave] = searches;
    return formatResults(
        query,
        mergeResults(
            duckDuckGo.status === "fulfilled" ? duckDuckGo.value : [],
            tavily.status === "fulfilled" ? tavily.value : [],
            serper?.status === "fulfilled" ? serper.value : [],
            jina?.status === "fulfilled" ? jina.value : [],
            brave?.status === "fulfilled" ? brave.value : [],
        ).slice(0, MAX_RESULTS),
        searches,
    );
};

module.exports = {
    definition,
    handler,
};
