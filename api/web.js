const {
  readBody,
  parseFetchToolPayload,
  stripFetchMeta,
} = require("../lib/web");
const { handler: webSearchHandler } = require("../lib/tools/web_search");
const { handler: webFetchHandler } = require("../lib/tools/web_fetch");
const {
  rerankSourcesForQuery,
  rankSourcesWithRag,
  selectSourcesForFetch,
  rankEvidenceEntriesForQuery,
  buildRagEvidenceBlock,
  RAG_FETCH_MAX_CHARS,
} = require("../lib/rag");

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

const hasMeaningfulDescription = (value = "") => {
  const normalized = String(value || "")
    .trim()
    .toLowerCase();
  return Boolean(normalized) && normalized !== "no description available";
};

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

  await Promise.all(
    Array.from({ length: Math.min(concurrency, values.length) }, runNext),
  );
  return results;
};

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
      const searchRaw = await webSearchHandler({ query, maxResults });
      let searchResults = [];

      if (typeof searchRaw === "string" && !searchRaw.startsWith("Error:")) {
        try {
          searchResults = JSON.parse(searchRaw);
        } catch {
          searchResults = [];
        }
      }

      result.sources = Array.isArray(searchResults)
        ? searchResults.slice(0, maxResults)
        : [];

      if (result.sources.length) {
        result.sources = ragEnabled
          ? rankSourcesWithRag(query, result.sources)
          : rerankSourcesForQuery(query, result.sources);
      }
    }

    // Step 2: Fetch content if requested
    if (fetchContent && result.sources.length) {
      const sourcesToFetch = selectSourcesForFetch(query, result.sources, {
        limit: FETCH_TARGET,
        perDomainLimit: 2,
      });
      const fetchedEntries = await runConcurrent(
        sourcesToFetch,
        FETCH_CONCURRENCY,
        async (source) => {
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

            const parsed = parseFetchToolPayload(fetchRaw);
            const enrichedSource = {
              ...source,
              url: parsed.finalUrl || source.url,
              title: parsed.title || source.title,
              description: parsed.description || source.description,
              date: parsed.publishedTime || source.date,
              via: parsed.via || source.via,
            };

            return {
              source: enrichedSource,
              content: parsed.content || "",
            };
          } catch (error) {
            return {
              source,
              content: "",
              error: error?.message || "Fetch failed",
            };
          }
        },
      );

      const successfulFetched = fetchedEntries.filter((entry) =>
        stripFetchMeta(entry?.content || "").trim(),
      );
      const usableEvidenceEntries = fetchedEntries.filter(
        (entry) =>
          stripFetchMeta(entry?.content || "").trim() ||
          hasMeaningfulDescription(entry?.source?.description),
      );
      result.fetched = query
        ? rankEvidenceEntriesForQuery(query, successfulFetched)
        : successfulFetched;
      result.snippetFallbacks = usableEvidenceEntries.filter(
        (entry) => !stripFetchMeta(entry?.content || "").trim(),
      ).length;

      // Step 3: Build RAG evidence blocks
      if (ragEnabled && usableEvidenceEntries.length) {
        result.evidence = rankEvidenceEntriesForQuery(
          query,
          usableEvidenceEntries,
        ).map((entry) => buildRagEvidenceBlock(entry, query));
      }
    }

    // Step 4: Fetch specific URLs if provided
    if (urls.length) {
      const urlEntries = await runConcurrent(
        urls,
        FETCH_CONCURRENCY,
        async (url) => {
          try {
            const fetchRaw = await webFetchHandler({
              url,
              format: "markdown",
              max_chars: RAG_FETCH_MAX_CHARS,
            });
            const parsed = parseFetchToolPayload(fetchRaw);

            return {
              url,
              finalUrl: parsed.finalUrl || url,
              title: parsed.title || "",
              description: parsed.description || "",
              publishedTime: parsed.publishedTime || "",
              via: parsed.via || "",
              content: parsed.content || "",
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
        },
      );

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
    description:
      "Combined web research endpoint with search, fetch, and RAG ranking",
    features: [
      "Multi-backend search (DuckDuckGo, Tavily, Serper, Jina, Brave)",
      "Content extraction with Jina and direct-fetch fallbacks",
      "RAG re-ranking for query-focused results",
      "Evidence block generation for synthesis",
    ],
    postBody: {
      query: "string (optional) - Search query",
      urls: "string[] (optional) - Specific URLs to fetch",
      maxResults:
        "number (optional) - Max search results (default: 20, max: 30)",
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
