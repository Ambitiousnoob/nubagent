const {
    clamp,
    normalizeUrl,
    extractTitle,
    stripHtml,
    extractLinks,
    extractReadableHtml,
    fetchTextResource,
    readBody,
} = require("../lib/web");

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

module.exports = async (req, res) => {
    res.setHeader("Content-Type", "application/json; charset=utf-8");

    if (req.method !== "POST") {
        res.status(405).end(JSON.stringify({ error: "Method not allowed" }));
        return;
    }

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
};
