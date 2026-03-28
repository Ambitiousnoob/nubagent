const requireSafeGoogleThis = () => {
    try {
        // eslint-disable-next-line global-require
        return require("googlethis");
    } catch (e) {
        console.error("googlethis load failed:", e);
        return null;
    }
};

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
                    count: { type: "integer", minimum: 1, maximum: 5, description: "Results to return (1-5)" },
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
        try {
            const googleThis = requireSafeGoogleThis();
            if (!googleThis) return "Error: web search library unavailable";
            const results = await googleThis.search(query, { page: 0, safe: false, parse_ads: false });
            const top = (results?.results || []).slice(0, count).map((item, idx) => ({
                rank: idx + 1,
                title: item.title,
                url: item.url,
                description: item.description,
            }));
            if (!top.length) return `No results for "${query}"`;
            return JSON.stringify(top);
        } catch (e) {
            return `Error: search failed (${e.message})`;
        }
    },
};
