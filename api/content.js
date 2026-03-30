/**
 * Content Endpoint
 * Consolidated content fetching endpoints: fetch, read, crawl
 *
 * Usage:
 * - /api/content?action=fetch (POST) - Simple URL fetch using web_fetch tool
 * - /api/content?action=read (POST) - Smart content extraction with modes
 * - /api/content?action=crawl (POST) - Multi-page crawling
 */

const { readBody, parseFetchToolPayload } = require('../lib/web');
const { handler: webFetchHandler } = require('../lib/tools/web_fetch');

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

module.exports = async (req, res) => {
    writeCorsHeaders(res);

    if (req.method === "OPTIONS") {
        res.status(204).end();
        return;
    }

    // Handle GET/HEAD for metadata
    if (req.method === "GET" || req.method === "HEAD") {
        sendJson(res, 200, {
            ok: true,
            endpoint: "/api/content",
            actions: {
                fetch: { method: "POST", description: "Simple URL fetch using web_fetch tool" },
                read: { method: "POST", description: "Smart content extraction with modes (article, full, outline)" },
                crawl: { method: "POST", description: "Multi-page crawling with depth control" },
            },
            example: {
                fetch: { action: "fetch", url: "https://example.com" },
                read: { action: "read", url: "https://example.com", mode: "article" },
                crawl: { action: "crawl", url: "https://example.com", maxPages: 6 },
            },
        });
        return;
    }

    if (req.method !== "POST") {
        sendJson(res, 405, { error: "Method not allowed" });
        return;
    }

    const action = req.query?.action || req.query.action;

    // Route to appropriate handler based on action
    switch (action) {
        case 'fetch':
            return handleFetch(req, res);
        case 'read':
            return handleRead(req, res);
        case 'crawl':
            return handleCrawl(req, res);
        default:
            sendJson(res, 400, {
                error: 'Unknown action. Use action=fetch, action=read, or action=crawl',
                validActions: ['fetch', 'read', 'crawl'],
            });
    }
};

/**
 * Fetch Handler - Simple URL fetch using web_fetch tool
 */
async function handleFetch(req, res) {
    try {
        const body = await readBody(req);
        const url = String(body?.url || "").trim();
        if (!url) {
            sendJson(res, 400, { error: "Provide a url." });
            return;
        }

        const raw = await webFetchHandler({
            url,
            format: body?.format,
            max_chars: body?.max_chars ?? body?.maxChars,
        });

        if (typeof raw === "string" && raw.startsWith("Error:")) {
            sendJson(res, 502, { error: raw });
            return;
        }

        const parsed = parseFetchToolPayload(raw);

        sendJson(res, 200, {
            ok: true,
            url,
            finalUrl: parsed.finalUrl || url,
            title: parsed.title || "",
            description: parsed.description || "",
            publishedTime: parsed.publishedTime || "",
            via: parsed.via || "",
            content: parsed.content || "",
        });
    } catch (error) {
        sendJson(res, 500, { error: error?.message || "Fetch request failed." });
    }
}

/**
 * Read Handler - Smart content extraction with modes
 */
async function handleRead(req, res) {
    const {
        clamp,
        normalizeUrl,
        extractTitle,
        extractMetaDescription,
        extractCanonical,
        stripHtml,
        extractLinks,
        extractHeadings,
        extractReadableHtml,
        fetchTextResource,
    } = require('../lib/web');

    const MAX_CHARS_LIMIT = 12000;
    const toFiniteNumber = (value, fallback) => {
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : fallback;
    };

    try {
        const body = await readBody(req);
        const sourceUrl = normalizeUrl(body?.url);

        if (!sourceUrl) {
            res.status(400).end(JSON.stringify({ error: "A valid absolute http(s) url is required." }));
            return;
        }

        const mode = ["article", "full", "outline"].includes(body.mode) ? body.mode : "article";
        const maxChars = clamp(toFiniteNumber(body.maxChars ?? body.maxLength, 3200), 200, MAX_CHARS_LIMIT);
        const timeoutMs = clamp(toFiniteNumber(body.timeoutMs, 12000), 2000, 30000);
        const includeLinks = body.includeLinks !== false;

        const resource = await fetchTextResource(sourceUrl, timeoutMs);
        const finalUrl = normalizeUrl(resource.finalUrl) || sourceUrl;
        let title = finalUrl;
        let description = "";
        let canonicalUrl = "";
        let headings = [];
        let links = [];
        let contentSource = resource.text.trim();

        if (resource.isHtml) {
            title = extractTitle(resource.text) || finalUrl;
            description = extractMetaDescription(resource.text);
            canonicalUrl = extractCanonical(resource.text, finalUrl) || "";
            headings = extractHeadings(resource.text);
            links = includeLinks ? extractLinks(resource.text, finalUrl).slice(0, 20) : [];

            if (mode === "outline") {
                contentSource = [
                    description ? `Description: ${description}` : "",
                    ...headings.map(item => `${"#".repeat(Math.min(item.level, 3))} ${item.text}`),
                ].filter(Boolean).join("\n");
            } else {
                const readableHtml = mode === "full" ? resource.text : extractReadableHtml(resource.text);
                contentSource = stripHtml(readableHtml);
            }
        } else if (/application\/json/i.test(resource.contentType)) {
            title = resource.titleHint || finalUrl.split("/").pop() || finalUrl;
            try {
                contentSource = JSON.stringify(JSON.parse(resource.text), null, 2);
            } catch {
                contentSource = resource.text.trim();
            }
        } else {
            title = resource.titleHint || finalUrl.split("/").pop() || finalUrl;
            contentSource = resource.text.trim();
        }

        const content = contentSource.slice(0, maxChars);

        res.status(200).end(JSON.stringify({
            sourceUrl,
            finalUrl,
            title,
            description,
            canonicalUrl,
            contentType: resource.contentType || "text/plain",
            mode,
            headings,
            links,
            content,
            contentTruncated: contentSource.length > maxChars,
            wordCount: contentSource ? contentSource.split(/\s+/).filter(Boolean).length : 0,
            via: resource.via || "direct",
        }));
    } catch (error) {
        res.status(500).end(JSON.stringify({ error: error?.message || "Reader request failed." }));
    }
}

/**
 * Crawl Handler - Multi-page crawling with depth control
 */
async function handleCrawl(req, res) {
    const {
        clamp,
        normalizeUrl,
        extractTitle,
        stripHtml,
        extractLinks,
        extractReadableHtml,
        fetchTextResource,
    } = require('../lib/web');

    const MAX_PAGES_LIMIT = 12;
    const MAX_DEPTH_LIMIT = 3;
    const MAX_CHARS_PER_PAGE_LIMIT = 5000;
    const MAX_PATTERN_COUNT = 12;
    const DEFAULT_TIMEOUT_MS = 12000;
    const toFiniteNumber = (value, fallback) => {
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : fallback;
    };

    const normalizePatterns = (patterns) => (
        Array.isArray(patterns)
            ? patterns.slice(0, MAX_PATTERN_COUNT).map(value => String(value).trim().toLowerCase()).filter(Boolean)
            : []
    );

    const matchesPatterns = (value, patterns) => {
        if (!patterns.length) return true;
        const normalizedValue = String(value).toLowerCase();
        return patterns.some(pattern => normalizedValue.includes(pattern));
    };

    try {
        const body = await readBody(req);
        const startUrl = normalizeUrl(body?.url);

        if (!startUrl) {
            res.status(400).end(JSON.stringify({ error: "A valid absolute http(s) url is required." }));
            return;
        }

        const maxPages = clamp(toFiniteNumber(body.maxPages, 6), 1, MAX_PAGES_LIMIT);
        const maxDepth = clamp(toFiniteNumber(body.maxDepth, 1), 0, MAX_DEPTH_LIMIT);
        const maxCharsPerPage = clamp(toFiniteNumber(body.maxCharsPerPage, 1800), 200, MAX_CHARS_PER_PAGE_LIMIT);
        const timeoutMs = clamp(toFiniteNumber(body.timeoutMs, DEFAULT_TIMEOUT_MS), 2000, 30000);
        const sameOrigin = body.sameOrigin !== false;
        const includePatterns = normalizePatterns(body.includePatterns);
        const excludePatterns = normalizePatterns(body.excludePatterns);
        const seedOrigin = new URL(startUrl).origin;

        const queue = [{ url: startUrl, depth: 0 }];
        const visited = new Set();
        const discovered = new Set([startUrl]);
        const pages = [];
        const errors = [];

        while (queue.length && pages.length < maxPages) {
            const current = queue.shift();
            if (!current || visited.has(current.url)) continue;
            visited.add(current.url);

            try {
                const page = await fetchTextResource(current.url, timeoutMs);
                const resolvedUrl = normalizeUrl(page.finalUrl) || current.url;
                const title = page.isHtml ? (extractTitle(page.text) || resolvedUrl) : (page.titleHint || resolvedUrl.split("/").pop() || resolvedUrl);
                const text = page.isHtml
                    ? stripHtml(extractReadableHtml(page.text))
                    : page.text.trim();
                const words = text ? text.split(/\s+/).filter(Boolean).length : 0;
                const links = page.isHtml ? extractLinks(page.text, resolvedUrl) : [];
                const queuedLinks = [];

                if (current.depth < maxDepth) {
                    for (const link of links) {
                        if (visited.has(link) || discovered.has(link)) continue;
                        if (sameOrigin && new URL(link).origin !== seedOrigin) continue;
                        if (includePatterns.length && !matchesPatterns(link, includePatterns)) continue;
                        if (excludePatterns.length && matchesPatterns(link, excludePatterns)) continue;

                        discovered.add(link);
                        queue.push({ url: link, depth: current.depth + 1 });
                        queuedLinks.push(link);
                    }
                }

                pages.push({
                    url: resolvedUrl,
                    title,
                    depth: current.depth,
                    words,
                    linksDiscovered: links.length,
                    queuedLinks: queuedLinks.slice(0, 10),
                    content: text.slice(0, maxCharsPerPage),
                    contentTruncated: text.length > maxCharsPerPage,
                    via: page.via || "direct",
                });
            } catch (error) {
                errors.push({
                    url: current.url,
                    depth: current.depth,
                    error: error?.name === "AbortError" ? `Timed out after ${timeoutMs}ms` : (error?.message || "Unknown fetch error"),
                });
            }
        }

        res.status(200).end(JSON.stringify({
            startUrl,
            maxPages,
            maxDepth,
            sameOrigin,
            discoveredCount: discovered.size,
            pages,
            errors,
        }));
    } catch (error) {
        res.status(500).end(JSON.stringify({ error: error?.message || "Crawler request failed." }));
    }
}
