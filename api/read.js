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
    readBody,
} = require("../lib/web");

const MAX_CHARS_LIMIT = 12000;
const toFiniteNumber = (value, fallback) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
};

module.exports = async (req, res) => {
    res.setHeader("Content-Type", "application/json; charset=utf-8");

    if (req.method !== "POST") {
        res.status(405).end(JSON.stringify({ error: "Method not allowed" }));
        return;
    }

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
};
