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
        if (!query) return "Error: query is required";

        const apiKey = process.env.TAVILY_API_KEY || "tvly-YOUR_FREE_API_KEY_HERE";
        if (!apiKey || apiKey.includes("YOUR_FREE_API_KEY_HERE")) {
            return "Error: Tavily API key missing. Set TAVILY_API_KEY on the server.";
        }

        try {
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), 8000);

            const response = await fetch("https://api.tavily.com/search", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    api_key: apiKey,
                    query,
                    search_depth: "basic",
                    include_answer: false,
                    max_results: 5,
                }),
                signal: controller.signal,
            });

            clearTimeout(timer);

            if (!response.ok) {
                return `Error: Tavily API returned status ${response.status}`;
            }

            const data = await response.json();
            const top = (data.results || []).slice(0, 5).map((item, idx) => ({
                rank: idx + 1,
                title: item.title,
                url: item.url,
                description: item.content || item.description || "No description available",
            }));

            if (!top.length) return `No results for "${query}"`;
            return JSON.stringify(top);
        } catch (e) {
            return `Error: search failed (${e.message})`;
        }
    },
};
