module.exports = {
    definition: {
        type: "function",
        function: {
            name: "search_images",
            strict: true,
            description: "Searches the web for relevant images and returns up to 10 results with URLs and snippets. Use when visuals genuinely help (people, places, products, diagrams).",
            parameters: {
                type: "object",
                properties: {
                    query: { type: "string", description: "Image search query. Include details like subject, style, angle." },
                    count: { type: "integer", minimum: 1, maximum: 10, description: "Results to return (1-10)" },
                },
                required: ["query"],
                additionalProperties: false,
            },
        },
    },
    handler: async (args) => {
        const query = String(args.query || "").trim();
        const count = Math.min(Math.max(parseInt(args.count, 10) || 5, 1), 10);
        if (!query) return "Error: query is required";
        try {
            const searchUrl = new URL("https://opnxng.com/search");
            searchUrl.searchParams.append("q", query);
            searchUrl.searchParams.append("format", "json");
            searchUrl.searchParams.append("categories", "images");

            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), 8000);
            const res = await fetch(searchUrl.toString(), { signal: controller.signal, headers: { "User-Agent": "NubAgent/1.0" } });
            clearTimeout(timer);
            if (!res.ok) return `Error: image search returned ${res.status}`;
            const data = await res.json();
            const results = (data.results || []).slice(0, count).map((item, idx) => ({
                rank: idx + 1,
                title: item.title || item.source || "image",
                url: item.img_src || item.thumbnail || item.url,
                page: item.url,
                description: item.content || item.source || "",
            }));
            if (!results.length) return `No image results for "${query}"`;
            return JSON.stringify(results);
        } catch (e) {
            return `Error: image search failed (${e.message})`;
        }
    },
};
