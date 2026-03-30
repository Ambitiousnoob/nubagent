const { normalizeUrl, extractTitle, extractMetaDescription, extractCanonical, extractHeadings, extractLinks, fetchTextResource } = require("../../lib/web");

const DEFAULT_MAX_CHARS = 20000;
const HARD_MAX_CHARS = 80000;
const TIMEOUT_MS = 15000;
const MAX_RETRIES = 2;

const JINA_READER_URL = "https://r.jina.ai/";
const FIRECRAWL_URL = "https://api.firecrawl.dev/v1/scrape";

const getFirecrawlApiKey = () => {
    const raw = process.env.FIRECRAWL_API_KEY || process.env.FIRECRAWL_API_KEYS;
    const keys = String(raw || "").split(",").map((k) => k.trim()).filter(Boolean);
    return keys[0] || null;
};

const normalizeText = (value) => String(value ?? "").replace(/\u0000/g, "").trim();

const extractMarkdownTitle = (content) => {
    const match = content.match(/^#\s+(.+)$/m);
    return match ? match[1].trim() : "";
};

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

    const apiKey = process.env.JINA_API_KEY;
    if (apiKey) {
        headers["Authorization"] = `Bearer ${apiKey}`;
    }

    const response = await fetch(jinaUrl, {
        method: "GET",
        signal,
        headers,
    });

    if (!response.ok) {
        throw new Error(`Jina returned HTTP ${response.status}`);
    }

    const raw = await response.text();
    const parsed = parseJinaResponse(raw);

    return {
        title: parsed.meta.title,
        url: parsed.meta.url || url,
        content: parsed.content.slice(0, maxChars),
        description: "",
        publishedTime: parsed.meta.publishedTime,
        via: "jina",
    };
};

const fetchWithFirecrawl = async (url, format, maxChars, signal) => {
    const apiKey = getFirecrawlApiKey();
    if (!apiKey) throw new Error("Firecrawl API key not configured");

    const response = await fetch(FIRECRAWL_URL, {
        method: "POST",
        signal,
        headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
            url,
            formats: ["markdown"],
            onlyMainContent: true,
            waitFor: 0,
            timeout: TIMEOUT_MS,
        }),
    });

    if (!response.ok) {
        throw new Error(`Firecrawl returned HTTP ${response.status}`);
    }

    const data = await response.json();
    if (!data.success || !data.markdown) {
        throw new Error("Firecrawl returned no content");
    }

    return {
        title: extractMarkdownTitle(data.markdown) || data.metadata?.title || "",
        url: data.metadata?.sourceURL || url,
        content: data.markdown.slice(0, maxChars),
        description: data.metadata?.description || "",
        publishedTime: data.metadata?.date || "",
        via: "firecrawl",
    };
};

const fetchWithFallback = async (url, format, maxChars) => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

    try {
        // Try Jina first (most reliable)
        const jinaResult = await fetchWithJina(url, format, maxChars, controller.signal);
        clearTimeout(timer);
        return jinaResult;
    } catch (jinaError) {
        // Try Firecrawl if configured
        const firecrawlKey = getFirecrawlApiKey();
        if (firecrawlKey) {
            try {
                const firecrawlResult = await fetchWithFirecrawl(url, format, maxChars, controller.signal);
                clearTimeout(timer);
                return firecrawlResult;
            } catch (firecrawlError) {
                // Continue to direct fetch
            }
        }

        // Direct fetch with lib/web.js
        try {
            const directResult = await fetchTextResource(url, TIMEOUT_MS);
            clearTimeout(timer);

            let content = directResult.text;
            if (directResult.isHtml) {
                content = content; // Already processed by lib/web.js
            }

            return {
                title: directResult.titleHint || extractTitle(content) || "",
                url: directResult.finalUrl || url,
                content: content.slice(0, maxChars),
                description: extractMetaDescription(content) || "",
                publishedTime: directResult.publishedTime || "",
                via: "direct",
            };
        } catch (directError) {
            throw new Error(`All fetch methods failed: Jina (${jinaError.message}), Firecrawl (${firecrawlKey ? firecrawlError.message : "not configured"}), Direct (${directError.message})`);
        }
    } finally {
        clearTimeout(timer);
    }
};

module.exports = {
    definition: {
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
    },
    handler: async (args) => {
        const url = normalizeText(args.url);
        const format = args.format === "text" ? "text" : (args.format === "html" ? "html" : "markdown");
        const maxChars = Math.min(Number(args.max_chars) || DEFAULT_MAX_CHARS, HARD_MAX_CHARS);
        const extractLinks = Boolean(args.extract_links);

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
                lastError = error.message;
            }
        }

        if (!result) {
            return `Error: web_fetch failed after ${MAX_RETRIES + 1} attempts. Last error: ${lastError}`;
        }

        // Build response
        const content = result.content;
        const estTokens = Math.round(content.length / 4);
        const truncated = content.length >= maxChars;

        const meta = [
            `<!-- Source: ${result.url} -->`,
            `<!-- Title: ${result.title || "Untitled"} -->`,
            `<!-- Format: ${format} | Characters: ${content.length} | Est. tokens: ~${estTokens}${truncated ? ` | Truncated from ${result.content.length} chars` : ""} -->`,
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

        if (extractLinks && format === "markdown") {
            try {
                const rawHtml = await fetch(url, { method: "GET" }).then((r) => r.text()).catch(() => "");
                const links = extractLinks(rawHtml, url).slice(0, 20);
                if (links.length) {
                    finalContent += "\n\n---\n\n## Outbound Links\n\n";
                    for (const link of links) {
                        finalContent += `- ${link}\n`;
                    }
                }
            } catch {
                // Ignore link extraction errors
            }
        }

        return finalContent;
    },
};
