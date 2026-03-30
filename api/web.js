const { readBody } = require("../lib/web");
const { handler: webSearchHandler } = require("./tools/web_search");
const { handler: webFetchHandler } = require("./tools/web_fetch");
const { rerankSourcesForQuery, rankSourcesWithRag, buildRagEvidenceBlock, RAG_FETCH_MAX_CHARS, RAG_EXCERPT_MAX_CHARS } = require("../src/lib/rag.js");

const writeCorsHeaders = (res) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
};

const sendJson = (res, status, payload) => {
    res.status(status);
    res.setHeader("Content-Type", "application/json; charset=utf-8");
    res.end(JSON.stringify(payload));
};

const FETCH_CONCURRENCY = 4;
const FETCH_TARGET = 12;

const runConcurrent = async (items, limit, worker) => {
    const values = Array.isArray(items) ? items : [];
    const concurrency = Math.max(1, limit || 1);
    const results = new Array(values.length);
    let cursor = 0;

    const runNext = async () => {
        while (cursor < values.length) {
            const current = cursor;
            cursor += 1;
            results[current] = await worker(values[current], current);
        }
    };

    await Promise.all(Array.from({ length: Math.min(concurrency, values.length) }, runNext));
    return results;
};

const stripFetchMeta = (content = "") => (
    String(content || "")
        .replace(/^<!--[\s\S]*?-->\s*/g, "")
        .trim()
);

/**
 * POST /api/web
 * Combined web research endpoint: search + fetch + RAG ranking
 */
const handleWebResearch = async (req, res) => {
    try {
        const body = await readBody(req);
        const query = String(body?.query || "").trim();
        const urls = Array.isArray(body?.urls) ? body.urls : [];
        const maxResults = Math.min(Number(body?.maxResults) || 20, 30);
        const fetchContent = Boolean(body?.fetchContent);
        const ragEnabled = Boolean(body?.rag);

        if (!query && !urls.length) {
            sendJson(res, 400, { error: "Provide a query string or URLs to fetch." });
            return;
        }

        const result = {
            ok: true,
            query,
            sources: [],
            fetched: [],
            evidence: [],
        };

        // Step 1: Search if query provided
        if (query) {
            const searchRaw = await webSearchHandler({ query });
            let searchResults = [];

            if (typeof searchRaw === "string" && !searchRaw.startsWith("Error:")) {
                try {
                    searchResults = JSON.parse(searchRaw);
                } catch {
                    searchResults = [];
                }
            }

            result.sources = Array.isArray(searchResults) ? searchResults.slice(0, maxResults) : [];

            // RAG re-ranking
            if (ragEnabled && result.sources.length) {
                result.sources = rankSourcesWithRag(query, result.sources);
            }
        }

        // Step 2: Fetch content if requested
        if (fetchContent && result.sources.length) {
            const sourcesToFetch = result.sources.slice(0, FETCH_TARGET);
            const fetchedEntries = await runConcurrent(sourcesToFetch, FETCH_CONCURRENCY, async (source) => {
                try {
                    const fetchRaw = await webFetchHandler({
                        url: source.url,
                        format: "markdown",
                        max_chars: RAG_FETCH_MAX_CHARS,
                    });

                    if (typeof fetchRaw === "string" && fetchRaw.startsWith("Error:")) {
                        return {
                            source,
                            content: "",
                            error: fetchRaw,
                        };
                    }

                    return {
                        source,
                        content: String(fetchRaw || ""),
                    };
                } catch (error) {
                    return {
                        source,
                        content: "",
                        error: error?.message || "Fetch failed",
                    };
                }
            });

            result.fetched = fetchedEntries.filter((e) => stripFetchMeta(e?.content || "").trim());

            // Step 3: Build RAG evidence blocks
            if (ragEnabled && result.fetched.length) {
                result.evidence = result.fetched.map((entry) =>
                    buildRagEvidenceBlock(entry, query)
                );
            }
        }

        // Step 4: Fetch specific URLs if provided
        if (urls.length) {
            const urlEntries = await runConcurrent(urls, FETCH_CONCURRENCY, async (url) => {
                try {
                    const fetchRaw = await webFetchHandler({
                        url,
                        format: "markdown",
                        max_chars: RAG_FETCH_MAX_CHARS,
                    });

                    return {
                        url,
                        content: String(fetchRaw || ""),
                        success: !String(fetchRaw || "").startsWith("Error:"),
                    };
                } catch (error) {
                    return {
                        url,
                        content: "",
                        success: false,
                        error: error?.message || "Fetch failed",
                    };
                }
            });

            result.urlContent = urlEntries;
        }

        sendJson(res, 200, result);
    } catch (error) {
        sendJson(res, 500, { error: error?.message || "Web research failed." });
    }
};

/**
 * GET /api/web
 * Endpoint metadata
 */
const handleMetadata = (res) => {
    sendJson(res, 200, {
        ok: true,
        endpoint: "/api/web",
        methods: ["GET", "POST"],
        description: "Combined web research endpoint with search, fetch, and RAG ranking",
        features: [
            "Multi-backend search (DuckDuckGo, Tavily, Serper, Jina, Brave)",
            "Content extraction with Jina/Firecrawl fallbacks",
            "RAG re-ranking for query-focused results",
            "Evidence block generation for synthesis",
        ],
        postBody: {
            query: "string (optional) - Search query",
            urls: "string[] (optional) - Specific URLs to fetch",
            maxResults: "number (optional) - Max search results (default: 20, max: 30)",
            fetchContent: "boolean (optional) - Fetch full content of search results",
            rag: "boolean (optional) - Enable RAG re-ranking and evidence blocks",
        },
    });
};

module.exports = async (req, res) => {
    writeCorsHeaders(res);

    if (req.method === "OPTIONS") {
        res.status(204).end();
        return;
    }

    if (req.method === "GET" || req.method === "HEAD") {
        handleMetadata(res);
        return;
    }

    if (req.method !== "POST") {
        sendJson(res, 405, { error: "Method not allowed" });
        return;
    }

    await handleWebResearch(req, res);
};
