module.exports = {
    definition: {
        type: "function",
        function: {
            name: "web_fetch",
            strict: true,
            description: "Fetch a URL (text/html) and return up to 3000 characters of readable text.",
            parameters: {
                type: "object",
                properties: {
                    url: { type: "string", description: "HTTP or HTTPS URL to fetch" },
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
            const controller = new AbortController();
            const timer = setTimeout(() => controller.abort(), 8000);
            const res = await fetch(url, { signal: controller.signal });
            clearTimeout(timer);
            const type = String(res.headers.get("content-type") || "").toLowerCase();
            if (!type.includes("text/")) return `Error: content-type not supported (${type || "unknown"})`;
            const text = await res.text();
            const trimmed = text.replace(/\s+/g, " ").slice(0, 3000);
            return trimmed || "Error: empty response";
        } catch (e) {
            return `Error: fetch failed (${e.message})`;
        }
    },
};
