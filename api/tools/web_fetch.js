module.exports = {
    definition: {
        type: "function",
        function: {
            name: "web_fetch",
            strict: true,
            description: "Fetches a webpage and returns its main content perfectly formatted in clean Markdown. Use this to read specific articles, documentation, or websites. It can read heavily JavaScript-rendered pages.",
            parameters: {
                type: "object",
                properties: {
                    url: { type: "string", description: "The precise HTTP or HTTPS URL to read." },
                },
                required: ["url"],
                additionalProperties: false,
            },
        },
    },
    handler: async (args) => {
        const url = String(args.url || "").trim();
        if (!/^https?:\/\//i.test(url)) return "Error: url must start with http or https";

        try {
            const jinaUrl = `https://r.jina.ai/${url}`;
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), 10000);

            const res = await fetch(jinaUrl, {
                method: "GET",
                signal: controller.signal,
                headers: {
                    "Accept": "text/plain",
                    "User-Agent": "NubAgent-WebReader/1.0",
                },
            });

            clearTimeout(timer);

            if (!res.ok) {
                return `Error: Failed to fetch page. Status: ${res.status}`;
            }

            const markdownText = await res.text();
            const safeContent = markdownText.slice(0, 20000);

            if (!safeContent.trim()) {
                return "Error: Page was fetched but no readable content was found.";
            }

            return safeContent;
        } catch (e) {
            if (e.name === "AbortError") {
                return "Error: Fetch timed out after 10 seconds. The page was too slow or heavy.";
            }
            return `Error: web fetch failed (${e.message})`;
        }
    },
};
