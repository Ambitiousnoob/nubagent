module.exports = {
    definition: {
        type: "function",
        function: {
            name: "web_fetch",
            strict: true,
            description: "Fetches a webpage and returns its main content in clean Markdown or plain text. Handles JavaScript-heavy pages. Use for articles, docs, or any URL.",
            parameters: {
                type: "object",
                properties: {
                    url: {
                        type: "string",
                        description: "The precise HTTP or HTTPS URL to fetch.",
                    },
                    format: {
                        type: "string",
                        enum: ["markdown", "text"],
                        description: "Output format. Defaults to 'markdown'.",
                    },
                    max_chars: {
                        type: "number",
                        description: "Max characters to return. Defaults to 20000. Increase for long docs, decrease for summaries.",
                    },
                },
                required: ["url"],
                additionalProperties: false,
            },
        },
    },
    handler: async (args) => {
        const url = String(args.url || "").trim();
        const format = args.format === "text" ? "text" : "markdown";
        const maxChars = Math.min(Number(args.max_chars) || 20000, 60000);

        if (!/^https?:\/\//i.test(url)) {
            return "Error: URL must start with http:// or https://";
        }

        const TIMEOUT_MS = 15000;
        const MAX_RETRIES = 2;
        const jinaUrl = `https://r.jina.ai/${url}`;
        const headers = {
            Accept: format === "text" ? "text/plain" : "text/markdown",
            "User-Agent": "NubAgent-WebReader/1.1",
            "X-Return-Format": format,
        };

        const attemptFetch = async () => {
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
            try {
                const res = await fetch(jinaUrl, {
                    method: "GET",
                    signal: controller.signal,
                    headers,
                });
                clearTimeout(timer);
                return res;
            } catch (e) {
                clearTimeout(timer);
                throw e;
            }
        };

        let lastError = null;

        for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
            if (attempt > 0) {
                await new Promise((r) => setTimeout(r, 1000 * attempt));
            }

            try {
                const res = await attemptFetch();

                if (!res.ok) {
                    lastError = `HTTP ${res.status}: ${res.statusText}`;
                    if (res.status >= 400 && res.status < 500) break;
                    continue;
                }

                const raw = await res.text();
                const content = raw.slice(0, maxChars).trim();

                if (!content) {
                    return "Error: Page fetched successfully but no readable content found.";
                }

                const estTokens = Math.round(content.length / 4);
                const truncated = raw.length > maxChars;
                const meta = [
                    `<!-- Source: ${url} -->`,
                    `<!-- Format: ${format} | Characters: ${content.length} | Est. tokens: ~${estTokens}${truncated ? ` | Truncated from ${raw.length} chars` : ""} -->`,
                ].join("\n");

                return `${meta}\n\n${content}`;
            } catch (e) {
                if (e.name === "AbortError") {
                    lastError = `Timed out after ${TIMEOUT_MS / 1000}s`;
                } else {
                    lastError = e.message;
                }
            }
        }

        return `Error: web_fetch failed after ${MAX_RETRIES + 1} attempts. Last error: ${lastError}`;
    },
};
