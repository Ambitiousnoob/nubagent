module.exports = {
    definition: {
        type: "function",
        function: {
            name: "view_image",
            strict: true,
            description: "Fetch basic metadata for an image URL (content-type, size) to help verify an image. Use when asked to inspect a specific image link.",
            parameters: {
                type: "object",
                properties: {
                    url: { type: "string", description: "HTTP/HTTPS image URL to inspect" },
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
            let res = await fetch(url, { method: "HEAD", signal: controller.signal });
            if (!res.ok || !res.headers.get("content-type")) {
                res = await fetch(url, { method: "GET", signal: controller.signal });
            }
            clearTimeout(timer);
            if (!res.ok) return `Error: image fetch returned ${res.status}`;
            const type = res.headers.get("content-type") || "unknown";
            const length = res.headers.get("content-length");
            return JSON.stringify({
                url,
                contentType: type,
                sizeBytes: length ? Number(length) : null,
                reachable: true,
            });
        } catch (e) {
            return `Error: image inspection failed (${e.message})`;
        }
    },
};
