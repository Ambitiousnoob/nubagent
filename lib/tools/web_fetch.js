const {
    assertPublicHttpUrl,
    normalizeUrl,
    extractTitle,
    extractMetaDescription,
    extractPublishedTime,
    extractCanonical,
    extractReadableHtml,
    extractLinks: extractPageLinks,
    stripHtml,
} = require("../web");
const { getApiKeysFromEnv, getRotatingApiKey } = require("../api-key-rotation.cjs");

const DEFAULT_MAX_CHARS = 20000;
const HARD_MAX_CHARS = 80000;
const TIMEOUT_MS = 15000;
const MAX_RETRIES = 2;
const MAX_REDIRECTS = 5;
const MAX_ERROR_DETAIL_CHARS = 240;

const JINA_READER_URL = "https://r.jina.ai/";

const getJinaApiKey = () => getRotatingApiKey("jina", "JINA_API_KEYS", "JINA_API_KEY");

const normalizeText = (value) => String(value ?? "").replace(/\u0000/g, "").trim();

const getErrorMessage = (error, timeoutMs = TIMEOUT_MS) => {
    if (!error) return "unknown error";
    if (error.name === "AbortError") return `timed out after ${timeoutMs}ms`;
    return normalizeText(error.message || String(error)) || "unknown error";
};

const withTimeoutSignal = async (worker, timeoutMs = TIMEOUT_MS) => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
        return await worker(controller.signal);
    } finally {
        clearTimeout(timer);
    }
};

const extractMarkdownTitle = (content) => {
    const match = content.match(/^#\s+(.+)$/m);
    return match ? match[1].trim() : "";
};

const collectUniqueLinks = (links = [], limit = 20) => {
    const unique = [];
    const seen = new Set();
    for (const link of links) {
        const normalized = normalizeUrl(link);
        if (!normalized || seen.has(normalized)) continue;
        seen.add(normalized);
        unique.push(normalized);
        if (unique.length >= limit) break;
    }
    return unique;
};

const extractLinksFromMarkdown = (content, baseUrl) => {
    const links = [];
    const text = String(content || "");

    for (const match of text.matchAll(/\[[^\]]*]\((https?:\/\/[^)\s]+)\)/gi)) {
        links.push(match[1]);
    }

    for (const match of text.matchAll(/\bhttps?:\/\/[^\s)<>"']+/gi)) {
        links.push(match[0].replace(/[),.;:!?]+$/, ""));
    }

    return collectUniqueLinks(
        links.map((link) => normalizeUrl(link, baseUrl)).filter(Boolean),
    );
};

const renderHtmlAsMarkdown = (html, baseUrl) => {
    const readableHtml = extractReadableHtml(html);
    const outboundLinks = collectUniqueLinks(extractPageLinks(readableHtml, baseUrl));
    const withMarkdownLinks = String(readableHtml || "").replace(
        /<a\b[^>]*href\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))[^>]*>([\s\S]*?)<\/a>/gi,
        (_, hrefA, hrefB, hrefC, inner) => {
            const href = normalizeUrl(hrefA || hrefB || hrefC || "", baseUrl);
            const label = normalizeText(stripHtml(inner));
            if (href && label) return `[${label}](${href})`;
            return href || label || "";
        },
    );

    const markdown = stripHtml(
        withMarkdownLinks
            .replace(/<pre\b[^>]*>([\s\S]*?)<\/pre>/gi, (_, inner) => `\n\n\`\`\`\n${stripHtml(inner)}\n\`\`\`\n\n`)
            .replace(/<code\b[^>]*>([\s\S]*?)<\/code>/gi, (_, inner) => `\`${normalizeText(stripHtml(inner))}\``)
            .replace(/<h([1-6])\b[^>]*>([\s\S]*?)<\/h\1>/gi, (_, level, inner) => {
                const heading = normalizeText(stripHtml(inner));
                if (!heading) return "\n\n";
                return `\n\n${"#".repeat(Math.min(Number(level), 6))} ${heading}\n\n`;
            })
            .replace(/<li\b[^>]*>([\s\S]*?)<\/li>/gi, (_, inner) => {
                const item = normalizeText(stripHtml(inner));
                return item ? `\n- ${item}` : "\n";
            })
            .replace(/<blockquote\b[^>]*>([\s\S]*?)<\/blockquote>/gi, (_, inner) => {
                const quote = normalizeText(stripHtml(inner));
                return quote ? `\n\n> ${quote.replace(/\n+/g, "\n> ")}\n\n` : "\n\n";
            })
            .replace(/<br\s*\/?>/gi, "\n")
            .replace(/<\/?(?:p|div|section|article|main|header|footer|aside|nav|ul|ol|table|tr|td|th)\b[^>]*>/gi, "\n"),
    )
        .replace(/[ \t]+\n/g, "\n")
        .replace(/\n[ \t]+/g, "\n")
        .replace(/\n{3,}/g, "\n\n")
        .trim();

    return {
        markdown,
        readableHtml,
        outboundLinks,
    };
};

const readErrorDetail = async (response) => {
    try {
        const contentType = response.headers.get("content-type") || "";
        if (/json/i.test(contentType)) {
            const data = await response.json();
            const detail = normalizeText(
                data?.error
                || data?.message
                || data?.detail
                || data?.details
                || data?.statusText,
            );
            return detail.slice(0, MAX_ERROR_DETAIL_CHARS);
        }

        const detail = normalizeText(await response.text());
        return detail.slice(0, MAX_ERROR_DETAIL_CHARS);
    } catch {
        return "";
    }
};

const isTextLikeContent = (contentType = "") => (
    !contentType ||
    /(text\/|application\/(json|xml|xhtml\+xml|javascript|ld\+json))/i.test(contentType)
);

const parseJinaResponse = (text) => {
    const lines = text.split("\n");
    const meta = {
        title: "",
        url: "",
        publishedTime: "",
    };
    const contentLines = [];
    let inContent = false;

    for (const line of lines) {
        if (!inContent) {
            if (line.startsWith("Title: ")) {
                meta.title = line.slice(7).trim();
            } else if (line.startsWith("URL Source: ")) {
                meta.url = line.slice(12).trim();
            } else if (line.startsWith("Published Time: ")) {
                meta.publishedTime = line.slice(16).trim();
            } else if (line.startsWith("Markdown Content:")) {
                inContent = true;
            }
        } else {
            contentLines.push(line);
        }
    }

    return {
        meta,
        content: contentLines.join("\n").trim(),
    };
};

const fetchWithJina = async (url, format, maxChars, signal) => {
    const jinaUrl = `${JINA_READER_URL}${url}`;
    const headers = {
        "Accept": format === "text" ? "text/plain" : "text/markdown",
        "User-Agent": "nub-agent/1.0",
        "X-Return-Format": format,
        "X-With-Generated-Alt": "true",
        "X-With-Links-Summary": "true",
    };

    const apiKey = getJinaApiKey();
    if (apiKey) {
        headers["Authorization"] = `Bearer ${apiKey}`;
    }

    const response = await fetch(jinaUrl, {
        method: "GET",
        signal,
        headers,
    });

    if (!response.ok) {
        const detail = await readErrorDetail(response);
        throw new Error(`Jina returned HTTP ${response.status}${detail ? `: ${detail}` : ""}`);
    }

    const raw = await response.text();
    const parsed = parseJinaResponse(raw);
    const content = parsed.content.slice(0, maxChars);

    return {
        title: parsed.meta.title,
        url: parsed.meta.url || url,
        content,
        fullLength: parsed.content.length,
        description: "",
        publishedTime: parsed.meta.publishedTime,
        via: "jina",
        outboundLinks: extractLinksFromMarkdown(content, parsed.meta.url || url),
    };
};

const fetchDirectly = async (url, format, maxChars, signal, redirectCount = 0) => {
    const normalized = await assertPublicHttpUrl(url);
    const response = await fetch(normalized, {
        method: "GET",
        redirect: "manual",
        signal,
        headers: {
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,application/json;q=0.9,application/xml;q=0.8,*/*;q=0.5",
            "User-Agent": "nub-agent/1.0",
        },
    });

    if (response.status >= 300 && response.status < 400) {
        if (redirectCount >= MAX_REDIRECTS) {
            throw new Error("Too many redirects");
        }

        const location = response.headers.get("location");
        if (!location) {
            throw new Error(`Redirect ${response.status} without location`);
        }

        return fetchDirectly(new URL(location, normalized).toString(), format, maxChars, signal, redirectCount + 1);
    }

    if (!response.ok) {
        const detail = await readErrorDetail(response);
        throw new Error(`Direct fetch returned HTTP ${response.status}${detail ? `: ${detail}` : ""}`);
    }

    const contentType = response.headers.get("content-type") || "";
    if (!isTextLikeContent(contentType)) {
        throw new Error(`Unsupported content-type: ${contentType || "unknown"}`);
    }

    const rawText = await response.text();
    const isHtml = /html|xhtml/i.test(contentType);
    const readable = isHtml ? renderHtmlAsMarkdown(rawText, normalized) : {
        markdown: normalizeText(rawText),
        readableHtml: "",
        outboundLinks: [],
    };
    const plainText = isHtml ? normalizeText(stripHtml(readable.readableHtml || rawText)) : normalizeText(rawText);
    const content = format === "html"
        ? rawText
        : (format === "markdown" ? readable.markdown : plainText);

    return {
        title: extractTitle(rawText) || "",
        url: extractCanonical(rawText, normalized) || normalized,
        content: content.slice(0, maxChars),
        fullLength: content.length,
        description: extractMetaDescription(rawText) || "",
        publishedTime: extractPublishedTime(rawText) || "",
        via: "direct",
        outboundLinks: isHtml ? readable.outboundLinks : extractLinksFromMarkdown(content, normalized),
    };
};

const fetchWithFallback = async (url, format, maxChars) => {
    const steps = [];

    if (format !== "html") {
        steps.push({
            name: "Jina",
            run: (signal) => fetchWithJina(url, format, maxChars, signal),
        });
    }

    steps.push({
        name: "Direct",
        run: (signal) => fetchDirectly(url, format, maxChars, signal),
    });

    const errors = [];
    for (const step of steps) {
        try {
            return await withTimeoutSignal((signal) => step.run(signal));
        } catch (error) {
            errors.push(`${step.name} (${getErrorMessage(error)})`);
        }
    }

    throw new Error(`All fetch methods failed: ${errors.join(", ")}`);
};

/**
 * Tool definition for function calling
 */
const definition = {
    type: "function",
    function: {
        name: "web_fetch",
        strict: true,
        description: "Fetches a webpage and returns its main content as clean Markdown. Handles JavaScript-heavy pages, paywalls, and anti-bot measures. Use for articles, documentation, research papers, or any URL needing content extraction.",
        parameters: {
            type: "object",
            properties: {
                url: {
                    type: "string",
                    description: "The exact HTTP/HTTPS URL to fetch. Must be publicly accessible.",
                },
                format: {
                    type: "string",
                    enum: ["markdown", "text", "html"],
                    description: "Output format. 'markdown' (default): clean MD with headings/links. 'text': plain text only. 'html': raw HTML.",
                },
                max_chars: {
                    type: "number",
                    description: "Maximum characters to return. Default: 20000. Max: 80000. Use lower for summaries, higher for full articles.",
                },
                extract_links: {
                    type: "boolean",
                    description: "If true, includes a summary of outbound links at the end.",
                },
            },
            required: ["url"],
            additionalProperties: false,
        },
    },
};

/**
 * Web fetch handler function
 * @param {object} args - Fetch arguments
 * @param {string} args.url - URL to fetch
 * @param {string} [args.format] - Output format (markdown, text, html)
 * @param {number} [args.max_chars] - Maximum characters to return
 * @param {boolean} [args.extract_links] - Whether to extract links
 * @returns {Promise<string>} Markdown content or error message
 */
const handler = async (args) => {
    const url = normalizeText(args.url);
    const format = args.format === "text" ? "text" : (args.format === "html" ? "html" : "markdown");
    const maxChars = Math.min(Number(args.max_chars) || DEFAULT_MAX_CHARS, HARD_MAX_CHARS);
    const includeLinks = Boolean(args.extract_links);

    // URL validation
    if (!url) {
        return "Error: URL is required";
    }

    if (!/^https?:\/\//i.test(url)) {
        return "Error: URL must start with http:// or https://";
    }

    try {
        // Validate URL format
        new URL(url);
    } catch {
        return "Error: Invalid URL format";
    }

    let lastError = null;
    let result = null;

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
        if (attempt > 0) {
            // Exponential backoff
            await new Promise((resolve) => setTimeout(resolve, 1000 * (attempt + 1)));
        }

        try {
            result = await fetchWithFallback(url, format, maxChars);
            break;
        } catch (error) {
            lastError = getErrorMessage(error);
        }
    }

    if (!result) {
        return `Error: web_fetch failed after ${MAX_RETRIES + 1} attempts. Last error: ${lastError}`;
    }

    // Build response
    const content = String(result.content || "");
    const fullLength = Number.isFinite(Number(result.fullLength)) ? Number(result.fullLength) : content.length;
    const estTokens = Math.round(fullLength / 4);
    const truncated = fullLength > content.length;

    const meta = [
        `<!-- Source: ${result.url} -->`,
        `<!-- Title: ${result.title || "Untitled"} -->`,
        `<!-- Format: ${format} | Characters: ${content.length} | Est. tokens: ~${estTokens}${truncated ? ` | Truncated from ${fullLength} chars` : ""} -->`,
        `<!-- Fetched via: ${result.via} -->`,
    ];

    if (result.description) {
        meta.push(`<!-- Description: ${result.description.slice(0, 200)} -->`);
    }

    if (result.publishedTime) {
        meta.push(`<!-- Published: ${result.publishedTime} -->`);
    }

    let finalContent = `${meta.join("\n")}\n\n`;

    if (format === "markdown" && result.title) {
        finalContent += `# ${result.title}\n\n`;
    }

    finalContent += content;

    if (includeLinks && format === "markdown") {
        const links = collectUniqueLinks(
            Array.isArray(result.outboundLinks) ? result.outboundLinks : [],
        );
        if (links.length) {
            finalContent += "\n\n---\n\n## Outbound Links\n\n";
            for (const link of links) {
                finalContent += `- ${link}\n`;
            }
        }
    }

    return finalContent;
};

module.exports = {
    definition,
    handler,
};
