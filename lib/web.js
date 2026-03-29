const dns = require("node:dns").promises;
const net = require("node:net");

const READER_PROXY_ORIGIN = "https://r.jina.ai";
const PRIVATE_HOSTNAMES = new Set([
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "host.docker.internal",
    "metadata.google.internal",
]);

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const decodeEntities = (text = "") => text
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/&#(\d+);/g, (_, code) => String.fromCharCode(Number(code)));

const normalizeUrl = (value, base) => {
    try {
        const next = new URL(value, base);
        if (!/^https?:$/.test(next.protocol)) return null;
        if (next.username || next.password) return null;
        next.hash = "";
        return next.toString();
    } catch {
        return null;
    }
};

const extractTitle = (html = "") => decodeEntities(
    (html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1] || "").replace(/\s+/g, " ").trim()
);

const extractMetaContent = (html = "", names = []) => {
    const wanted = new Set(names.map(value => value.toLowerCase()));
    const metaPattern = /<meta\b[^>]*>/gi;
    let match;

    while ((match = metaPattern.exec(html)) !== null) {
        const tag = match[0];
        const nameMatch = tag.match(/\b(?:name|property)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/i);
        const contentMatch = tag.match(/\bcontent\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/i);
        const key = (nameMatch?.[1] || nameMatch?.[2] || nameMatch?.[3] || "").trim().toLowerCase();
        const content = decodeEntities((contentMatch?.[1] || contentMatch?.[2] || contentMatch?.[3] || "").trim());

        if (key && content && wanted.has(key)) return content;
    }

    return "";
};

const extractMetaDescription = (html = "") => (
    extractMetaContent(html, ["description", "og:description", "twitter:description"])
);

const extractCanonical = (html = "", baseUrl) => {
    const linkPattern = /<link\b[^>]*>/gi;
    let match;

    while ((match = linkPattern.exec(html)) !== null) {
        const tag = match[0];
        const relMatch = tag.match(/\brel\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/i);
        const hrefMatch = tag.match(/\bhref\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/i);
        const rel = (relMatch?.[1] || relMatch?.[2] || relMatch?.[3] || "").trim().toLowerCase();
        const href = (hrefMatch?.[1] || hrefMatch?.[2] || hrefMatch?.[3] || "").trim();
        if (rel.includes("canonical") && href) return normalizeUrl(href, baseUrl);
    }

    return null;
};

const extractBodyHtml = (html = "") => html.match(/<body[^>]*>([\s\S]*?)<\/body>/i)?.[1] || html;

const stripHtml = (html = "") => decodeEntities(
    html
        .replace(/<script[\s\S]*?<\/script>/gi, " ")
        .replace(/<style[\s\S]*?<\/style>/gi, " ")
        .replace(/<noscript[\s\S]*?<\/noscript>/gi, " ")
        .replace(/<svg[\s\S]*?<\/svg>/gi, " ")
        .replace(/<br\s*\/?>/gi, "\n")
        .replace(/<\/(p|div|section|article|main|li|ul|ol|table|tr|td|th|h1|h2|h3|h4|h5|h6|pre|blockquote)>/gi, "\n")
        .replace(/<[^>]+>/g, " ")
        .replace(/[ \t]+\n/g, "\n")
        .replace(/\n[ \t]+/g, "\n")
        .replace(/\n{3,}/g, "\n\n")
        .replace(/[ \t]{2,}/g, " ")
        .trim()
);

const extractLinks = (html = "", baseUrl) => {
    const links = new Set();
    const hrefPattern = /<a\b[^>]*href\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/gi;
    let match;
    while ((match = hrefPattern.exec(html)) !== null) {
        const rawHref = (match[1] || match[2] || match[3] || "").trim();
        const normalized = normalizeUrl(rawHref, baseUrl);
        if (normalized) links.add(normalized);
    }
    return [...links];
};

const extractHeadings = (html = "", max = 14) => {
    const headings = [];
    const headingPattern = /<h([1-6])[^>]*>([\s\S]*?)<\/h\1>/gi;
    let match;

    while ((match = headingPattern.exec(html)) !== null && headings.length < max) {
        const text = stripHtml(match[2]).replace(/\s+/g, " ").trim();
        if (!text) continue;
        headings.push({ level: Number(match[1]), text });
    }

    return headings;
};

const extractReadableHtml = (html = "") => {
    const body = extractBodyHtml(html);
    const candidates = [];
    const addCandidate = (label, fragment) => {
        const textLength = stripHtml(fragment).length;
        if (textLength > 120) candidates.push({ label, fragment, textLength });
    };

    addCandidate("body", body);

    for (const match of body.matchAll(/<(article|main)\b[^>]*>([\s\S]*?)<\/\1>/gi)) {
        addCandidate(match[1], match[0]);
    }

    const sectionPattern = /<(section|div)\b[^>]*(?:id|class)\s*=\s*(?:"([^"]*)"|'([^']*)')[^>]*>([\s\S]*?)<\/\1>/gi;
    for (const match of body.matchAll(sectionPattern)) {
        const attrs = `${match[2] || ""} ${match[3] || ""}`.toLowerCase();
        if (/(content|article|main|post|doc|docs|markdown|entry|page|body|guide)/.test(attrs)) {
            addCandidate(`${match[1]}:${attrs}`, match[0]);
        }
    }

    candidates.sort((a, b) => b.textLength - a.textLength);
    return candidates[0]?.fragment || body;
};

const isTextLikeContent = (contentType = "") => (
    !contentType ||
    /(text\/|application\/(json|xml|xhtml\+xml|javascript|ld\+json))/i.test(contentType)
);

const isBlockedHostname = (hostname = "") => {
    const normalized = hostname.toLowerCase();
    return (
        PRIVATE_HOSTNAMES.has(normalized) ||
        normalized.endsWith(".localhost") ||
        normalized.endsWith(".local")
    );
};

const isPrivateIp = (address = "") => {
    if (address.startsWith("::ffff:")) return isPrivateIp(address.slice(7));

    const family = net.isIP(address);
    if (family === 4) {
        const [a, b] = address.split(".").map(Number);
        return (
            a === 0 ||
            a === 10 ||
            a === 127 ||
            (a === 169 && b === 254) ||
            (a === 172 && b >= 16 && b <= 31) ||
            (a === 192 && b === 168) ||
            (a === 100 && b >= 64 && b <= 127) ||
            (a === 198 && (b === 18 || b === 19))
        );
    }

    if (family === 6) {
        const normalized = address.toLowerCase();
        return (
            normalized === "::" ||
            normalized === "::1" ||
            normalized.startsWith("fc") ||
            normalized.startsWith("fd") ||
            normalized.startsWith("fe8") ||
            normalized.startsWith("fe9") ||
            normalized.startsWith("fea") ||
            normalized.startsWith("feb")
        );
    }

    return false;
};

const assertPublicHttpUrl = async (urlString) => {
    const normalized = normalizeUrl(urlString);
    if (!normalized) throw new Error("A valid absolute http(s) url is required.");

    const parsed = new URL(normalized);
    if (isBlockedHostname(parsed.hostname)) throw new Error("Private or local hosts are blocked.");

    if (net.isIP(parsed.hostname)) {
        if (isPrivateIp(parsed.hostname)) throw new Error("Private or local IP addresses are blocked.");
        return normalized;
    }

    const records = await dns.lookup(parsed.hostname, { all: true, verbatim: true });
    if (!records.length) throw new Error("Unable to resolve hostname.");
    if (records.some(record => isPrivateIp(record.address))) {
        throw new Error("Private or local network targets are blocked.");
    }

    return normalized;
};

const parseReaderProxyText = (text = "") => {
    const normalizedText = String(text).replace(/\r\n/g, "\n").trim();
    const title = normalizedText.match(/^Title:\s*(.+)$/m)?.[1]?.trim() || "";
    const sourceUrl = normalizeUrl(normalizedText.match(/^URL Source:\s*(.+)$/m)?.[1]?.trim() || "") || "";
    const publishedTime = normalizedText.match(/^Published Time:\s*(.+)$/m)?.[1]?.trim() || "";
    const marker = normalizedText.match(/\nMarkdown Content:\n/i);
    let content = normalizedText;

    if (marker) {
        content = normalizedText.slice(marker.index + marker[0].length).trim();
    }

    return {
        title,
        sourceUrl,
        publishedTime,
        content: content.replace(/^Warning:.*$/gmi, "").trim() || normalizedText,
    };
};

const fetchDirectTextResource = async (normalized, timeoutMs = 12000, redirectCount = 0) => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);

    try {
        const response = await fetch(normalized, {
            redirect: "manual",
            signal: controller.signal,
            headers: {
                accept: "text/html,application/xhtml+xml,text/plain;q=0.9,application/json;q=0.9,application/xml;q=0.8,*/*;q=0.5",
            },
        });

        if (response.status >= 300 && response.status < 400) {
            if (redirectCount >= 5) throw new Error("Too many redirects.");
            const location = response.headers.get("location");
            if (!location) throw new Error(`Redirect ${response.status} without location.`);
            const redirectUrl = await assertPublicHttpUrl(new URL(location, normalized).toString());
            return fetchDirectTextResource(redirectUrl, timeoutMs, redirectCount + 1);
        }

        const contentType = response.headers.get("content-type") || "";
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        if (!isTextLikeContent(contentType)) throw new Error(`Unsupported content-type: ${contentType || "unknown"}`);

        return {
            finalUrl: normalized,
            contentType,
            text: await response.text(),
            isHtml: /html|xhtml/i.test(contentType),
            via: "direct",
        };
    } finally {
        clearTimeout(timeout);
    }
};

const fetchReaderProxyResource = async (normalized, timeoutMs = 12000) => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    const proxyUrl = `${READER_PROXY_ORIGIN}/http://${normalized}`;

    try {
        const response = await fetch(proxyUrl, {
            signal: controller.signal,
            headers: {
                accept: "text/plain,text/markdown;q=0.9,*/*;q=0.5",
                "x-no-cache": "true",
            },
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const parsed = parseReaderProxyText(await response.text());
        const text = parsed.content || "";
        if (!text.trim()) throw new Error("Empty response.");

        return {
            finalUrl: parsed.sourceUrl || normalized,
            contentType: response.headers.get("content-type") || "text/markdown; charset=utf-8",
            text,
            isHtml: false,
            via: "reader-proxy",
            titleHint: parsed.title,
            publishedTime: parsed.publishedTime,
        };
    } finally {
        clearTimeout(timeout);
    }
};

const fetchTextResource = async (url, timeoutMs = 12000) => {
    const normalized = await assertPublicHttpUrl(url);

    try {
        return await fetchDirectTextResource(normalized, timeoutMs);
    } catch (directError) {
        try {
            return await fetchReaderProxyResource(normalized, timeoutMs);
        } catch (fallbackError) {
            throw new Error(`${directError?.message || "fetch failed"}; fallback reader failed: ${fallbackError?.message || "unknown error"}`);
        }
    }
};

const readBody = async (req) => {
    if (req.body && typeof req.body === "object") return req.body;
    if (typeof req.body === "string" && req.body.trim()) return JSON.parse(req.body);

    const chunks = [];
    for await (const chunk of req) chunks.push(Buffer.from(chunk));
    const raw = Buffer.concat(chunks).toString("utf8");
    return raw ? JSON.parse(raw) : {};
};

module.exports = {
    clamp,
    normalizeUrl,
    assertPublicHttpUrl,
    extractTitle,
    extractMetaDescription,
    extractCanonical,
    stripHtml,
    extractLinks,
    extractHeadings,
    extractReadableHtml,
    fetchTextResource,
    readBody,
};
