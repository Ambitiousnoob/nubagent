const { normalizeUrl, stripHtml } = require("../../lib/web");

const MAX_RESULTS = 30;
const REQUEST_TIMEOUT_MS = 8000;
const DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/";
const TAVILY_SEARCH_URL = "https://api.tavily.com/search";
const TAVILY_API_KEY = "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";
const RESULT_LINK_RE = /<a\b[^>]*class=(?:"[^"]*\b(?:result__a|result-link)\b[^"]*"|'[^']*\b(?:result__a|result-link)\b[^']*')[^>]*href=(?:"([^"]+)"|'([^']+)')[^>]*>([\s\S]*?)<\/a>/gi;
const RESULT_SNIPPET_RE = /<(?:a|div|span)\b[^>]*class=(?:"[^"]*\b(?:result__snippet|result-snippet)\b[^"]*"|'[^']*\b(?:result__snippet|result-snippet)\b[^']*')[^>]*>([\s\S]*?)<\/(?:a|div|span)>/i;

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
    if (!TAVILY_API_KEY) return [];

    return withTimeout(async (signal) => {
        const response = await fetch(TAVILY_SEARCH_URL, {
            method: "POST",
            signal,
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                api_key: TAVILY_API_KEY,
                query,
                search_depth: "basic",
                include_answer: false,
                max_results: limit,
            }),
        });

        if (!response.ok) {
            throw new Error(`Tavily returned status ${response.status}`);
        }

        const data = await response.json();
        return (Array.isArray(data?.results) ? data.results : []).map((item) => ({
            title: normalizeText(item?.title),
            url: normalizeUrl(item?.url) || String(item?.url || "").trim(),
            description: normalizeText(item?.content || item?.description || "") || "No description available",
            source: "tavily",
        })).filter((item) => item.title && item.url);
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

module.exports = {
    definition: {
        type: "function",
        function: {
            name: "web_search",
            strict: true,
            description: "Searches the web using DuckDuckGo plus supplemental web results. Supports operators like site:reddit.com and returns up to 30 results with titles, links, and snippets. Use for latest news, facts, product info, or anything not in training data.",
            parameters: {
                type: "object",
                properties: {
                    query: {
                        type: "string",
                        description: "The specific search query. If checking the date or time, include the timezone or location.",
                    },
                },
                required: ["query"],
                additionalProperties: false,
            },
        },
    },
    handler: async (args) => {
        const query = String(args.query || "").trim();
        if (!query) return "Error: query is required";

        const [duckDuckGo, tavily] = await Promise.allSettled([
            fetchDuckDuckGoResults(query, MAX_RESULTS),
            fetchTavilyResults(query, Math.min(18, MAX_RESULTS)),
        ]);

        const duckDuckGoResults = duckDuckGo.status === "fulfilled" ? duckDuckGo.value : [];
        const tavilyResults = tavily.status === "fulfilled" ? tavily.value : [];
        const combined = mergeResults(duckDuckGoResults, tavilyResults).slice(0, MAX_RESULTS);

        if (combined.length) {
            return JSON.stringify(combined.map((item, index) => ({
                rank: index + 1,
                title: item.title,
                url: item.url,
                description: item.description,
                source: item.source,
            })));
        }

        const errors = [
            duckDuckGo.status === "rejected" ? `duckduckgo: ${duckDuckGo.reason?.message || duckDuckGo.reason}` : "",
            tavily.status === "rejected" ? `tavily: ${tavily.reason?.message || tavily.reason}` : "",
        ].filter(Boolean);

        return errors.length
            ? `Error: search failed (${errors.join("; ")})`
            : `No results for "${query}"`;
    },
};
