const google = require("googlethis");

module.exports = {
    definition: {
        type: "function",
        function: {
            name: "web_search",
            strict: true,
            description: "CRITICAL: You MUST use this tool whenever you need real-time data, the current date or time, current events, or information beyond your training cutoff. Never reply saying you 'don't have access to real-time data'—use this tool instead.",
            parameters: {
                type: "object",
                properties: {
                    query: {
                        type: "string",
                        description: "The specific search query. If checking the date or time, include the timezone or location.",
                    },
                    count: {
                        type: "integer",
                        minimum: 1,
                        maximum: 5,
                        description: "Results to return (1-5)",
                    },
                },
                required: ["query"],
                additionalProperties: false,
            },
        },
    },
    handler: async (args) => {
        const query = String(args.query || "").trim();
        const count = Math.min(Math.max(parseInt(args.count, 10) || 3, 1), 5);

        if (!query) return "Error: query is required";

        // Try legacy Google search first, then fall back to SearxNG.
        try {
            const results = await google.search(query, { page: 0, safe: false, additional_params: { hl: "en" } });
            const items = Array.isArray(results?.results) ? results.results : Array.isArray(results) ? results : [];
            const top = items.slice(0, count).map((item, idx) => ({
                rank: idx + 1,
                title: item.title,
                url: item.url,
                description: item.description || item.snippet || "No description available",
            }));
            if (top.length) return JSON.stringify(top);
        } catch (err) {
            // Fall through to SearxNG fallback.
        }

        try {
            const searchUrl = new URL("https://opnxng.com/search");
            searchUrl.searchParams.append("q", query);
            searchUrl.searchParams.append("format", "json");

            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), 8000);

            const response = await fetch(searchUrl.toString(), {
                signal: controller.signal,
                headers: {
                    "Accept": "application/json",
                    "User-Agent": "NubAgent/1.0",
                },
            });

            clearTimeout(timer);

            if (!response.ok) {
                return `Error: Search engine returned status ${response.status}`;
            }

            const data = await response.json();
            const top = (data.results || []).slice(0, count).map((item, idx) => ({
                rank: idx + 1,
                title: item.title,
                url: item.url,
                description: item.content || item.snippet || "No description available",
            }));

            if (!top.length) return `No results for "${query}"`;
            return JSON.stringify(top);
        } catch (e) {
            return `Error: search failed (${e.message})`;
        }
    },
};
