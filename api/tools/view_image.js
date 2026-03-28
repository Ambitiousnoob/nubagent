const IMAGE_TIMEOUT_MS = 15000;
const MAX_READER_CHARS = 6000;

const trimText = (value, max = MAX_READER_CHARS) => {
    const text = String(value || "").trim();
    if (!text) return "";
    return text.length > max ? `${text.slice(0, max)}...` : text;
};

const unique = (items = []) => [...new Set(items.filter(Boolean))];

const extractCaptions = (markdown = "") => {
    const text = String(markdown || "");
    const captions = [];

    for (const match of text.matchAll(/!\[([^\]]+)\]\((https?:\/\/[^)]+)\)/gi)) {
        const alt = trimText(match[1], 500);
        if (alt) captions.push(alt);
    }

    for (const match of text.matchAll(/(?:^|\n)Image\s+\d+\s*:\s*(.+?)(?=\n|$)/gi)) {
        const caption = trimText(match[1], 500);
        if (caption) captions.push(caption);
    }

    return unique(captions);
};

const extractSummary = (markdown = "") => {
    const normalized = String(markdown || "")
        .replace(/\r\n/g, "\n")
        .replace(/!\[[^\]]*\]\((https?:\/\/[^)]+)\)/gi, " ")
        .replace(/^\s*Title:\s.*$/gim, " ")
        .replace(/^\s*URL Source:\s.*$/gim, " ")
        .replace(/^\s*Markdown Content:\s*$/gim, " ")
        .replace(/\n{3,}/g, "\n\n")
        .trim();
    const lines = normalized.split(/\n+/).map((line) => line.trim()).filter(Boolean);
    return trimText(lines[0] || "", 700);
};

const fetchMetadata = async (url, signal) => {
    let res = await fetch(url, { method: "HEAD", signal });
    if (!res.ok || !res.headers.get("content-type")) {
        res = await fetch(url, { method: "GET", signal });
    }
    if (!res.ok) {
        throw new Error(`image fetch returned ${res.status}`);
    }
    return {
        contentType: res.headers.get("content-type") || "unknown",
        sizeBytes: res.headers.get("content-length") ? Number(res.headers.get("content-length")) : null,
        reachable: true,
    };
};

const fetchReaderAnalysis = async (url, signal) => {
    const readerUrl = `https://r.jina.ai/${url}`;
    const res = await fetch(readerUrl, {
        method: "GET",
        signal,
        headers: {
            Accept: "text/plain",
            "User-Agent": "NubAgent-ImageReader/2.0",
            "x-with-generated-alt": "true",
        },
    });

    if (!res.ok) {
        throw new Error(`reader returned ${res.status}`);
    }

    const markdown = await res.text();
    const trimmed = trimText(markdown);
    return {
        raw: trimmed,
        captions: extractCaptions(trimmed),
        summary: extractSummary(trimmed),
    };
};

module.exports = {
    definition: {
        type: "function",
        function: {
            name: "view_image",
            strict: true,
            description: "Analyze a specific image URL and describe what is in it, including text when available. Use this for screenshots, charts, memes, product photos, or any image the user wants inspected.",
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

        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), IMAGE_TIMEOUT_MS);

        try {
            const [metadataResult, analysisResult] = await Promise.allSettled([
                fetchMetadata(url, controller.signal),
                fetchReaderAnalysis(url, controller.signal),
            ]);

            const metadata = metadataResult.status === "fulfilled"
                ? metadataResult.value
                : {
                    contentType: "unknown",
                    sizeBytes: null,
                    reachable: false,
                    metadataError: metadataResult.reason?.message || "metadata fetch failed",
                };

            const analysis = analysisResult.status === "fulfilled"
                ? analysisResult.value
                : {
                    raw: "",
                    captions: [],
                    summary: "",
                    analysisError: analysisResult.reason?.message || "analysis fetch failed",
                };

            clearTimeout(timer);

            if (!metadata.reachable && !analysis.summary && !analysis.captions.length && !analysis.raw) {
                return `Error: image inspection failed (${analysis.analysisError || metadata.metadataError || "unreachable"})`;
            }

            const response = {
                url,
                ...metadata,
                summary: analysis.summary || null,
                captions: analysis.captions || [],
                readerOutput: analysis.raw || null,
            };

            if (analysis.analysisError) response.analysisError = analysis.analysisError;
            if (metadata.metadataError) response.metadataError = metadata.metadataError;

            return JSON.stringify(response);
        } catch (e) {
            clearTimeout(timer);
            if (e.name === "AbortError") {
                return `Error: image inspection timed out after ${IMAGE_TIMEOUT_MS / 1000}s`;
            }
            return `Error: image inspection failed (${e.message})`;
        }
    },
};
