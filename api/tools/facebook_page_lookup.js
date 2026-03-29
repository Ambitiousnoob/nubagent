const { normalizeUrl } = require("../../lib/web");
const { handler: webSearchHandler } = require("./web_search");

const APIFY_ACTOR = "apify~facebook-pages-scraper";
const APIFY_ENDPOINT = `https://api.apify.com/v2/acts/${APIFY_ACTOR}/run-sync-get-dataset-items`;
const REQUEST_TIMEOUT_MS = 25000;
const MAX_RESULTS = 5;
const MAX_TEXT_CHARS = 1200;
const MAX_POSTS = 3;
const RESERVED_FACEBOOK_SEGMENTS = new Set([
    "groups",
    "watch",
    "events",
    "event",
    "share",
    "sharer",
    "story.php",
    "photo.php",
    "photos",
    "posts",
    "reel",
    "reels",
    "videos",
    "permalink.php",
    "login",
    "plugins",
    "marketplace",
    "hashtag",
    "search",
    "help",
    "games",
    "messages",
    "business",
    "ads",
    "privacy",
    "about",
    "public",
]);

const normalizeText = (value) => (
    String(value ?? "")
        .replace(/\u0000/g, "")
        .replace(/\s+/g, " ")
        .trim()
);

const truncate = (value, max = MAX_TEXT_CHARS) => {
    const text = normalizeText(value);
    if (!text) return "";
    return text.length > max ? `${text.slice(0, max - 3)}...` : text;
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const toNumber = (value) => {
    if (value === null || value === undefined || value === "") return null;
    const numeric = Number(String(value).replace(/[^\d.-]/g, ""));
    return Number.isFinite(numeric) ? numeric : null;
};

const pickFirst = (source, keys) => {
    for (const key of keys) {
        const value = source?.[key];
        if (value === null || value === undefined) continue;
        if (typeof value === "string" && !value.trim()) continue;
        return value;
    }
    return null;
};

const compactObject = (value) => Object.fromEntries(
    Object.entries(value).filter(([, entry]) => {
        if (entry === null || entry === undefined) return false;
        if (typeof entry === "string") return Boolean(entry.trim());
        if (Array.isArray(entry)) return entry.length > 0;
        return true;
    })
);

const normalizeFacebookPageInput = (value) => {
    const raw = normalizeText(value);
    if (!raw) return "";

    if (/^https?:\/\//i.test(raw)) {
        const normalized = normalizeUrl(raw);
        if (!normalized) throw new Error("page must be a valid http(s) Facebook page URL or handle");
        const parsed = new URL(normalized);
        const host = parsed.hostname.toLowerCase();
        if (!(host === "facebook.com" || host.endsWith(".facebook.com"))) {
            throw new Error("page must point to a public facebook.com URL");
        }
        return normalized;
    }

    const handle = raw.replace(/^@/, "").replace(/^\/+|\/+$/g, "");
    if (!/^[A-Za-z0-9._-]+(?:\/[A-Za-z0-9._-]+)*$/.test(handle)) {
        throw new Error("page must be a Facebook page URL or page handle");
    }
    return `https://www.facebook.com/${handle}`;
};

const canonicalizeFacebookSearchUrl = (value) => {
    const normalized = normalizeUrl(value);
    if (!normalized) return "";

    try {
        const parsed = new URL(normalized);
        const host = parsed.hostname.toLowerCase();
        if (!(host === "facebook.com" || host.endsWith(".facebook.com"))) return "";

        if (parsed.pathname === "/profile.php") {
            const id = normalizeText(parsed.searchParams.get("id"));
            return id ? `https://www.facebook.com/profile.php?id=${encodeURIComponent(id)}` : "";
        }

        const segments = parsed.pathname.split("/").filter(Boolean);
        if (!segments.length) return "";

        const first = segments[0].toLowerCase();
        if (first === "pages" && segments.length >= 3) {
            return `https://www.facebook.com/pages/${segments[1]}/${segments[2]}`;
        }
        if (first === "people" && segments.length >= 3) {
            return `https://www.facebook.com/people/${segments[1]}/${segments[2]}`;
        }
        if (RESERVED_FACEBOOK_SEGMENTS.has(first)) return "";

        return `https://www.facebook.com/${segments[0]}`;
    } catch {
        return "";
    }
};

const resolveFacebookPageUrl = async (value) => {
    const raw = normalizeText(value);
    if (!raw) return "";

    if (/^https?:\/\//i.test(raw)) {
        return normalizeFacebookPageInput(raw);
    }

    const handleCandidate = raw.replace(/^@/, "").replace(/^\/+|\/+$/g, "");
    if (/^[A-Za-z0-9._-]+(?:\/[A-Za-z0-9._-]+)*$/.test(handleCandidate) && !/\s/.test(handleCandidate)) {
        return normalizeFacebookPageInput(handleCandidate);
    }

    const searchResult = await webSearchHandler({ query: `${raw} site:facebook.com` });
    if (typeof searchResult !== "string") {
        throw new Error("Facebook page resolution failed");
    }
    if (searchResult.startsWith("Error:")) {
        throw new Error(searchResult);
    }
    if (searchResult.startsWith("No results")) {
        throw new Error(`No public Facebook page found for "${raw}"`);
    }

    let items;
    try {
        items = JSON.parse(searchResult);
    } catch {
        throw new Error("Could not parse Facebook page search results");
    }

    const resolvedUrl = (Array.isArray(items) ? items : [])
        .map((item) => canonicalizeFacebookSearchUrl(item?.url))
        .find(Boolean);

    if (!resolvedUrl) {
        throw new Error(`No public Facebook page found for "${raw}"`);
    }

    return resolvedUrl;
};

const normalizePostEntry = (value) => {
    if (!value) return null;
    if (typeof value === "string") {
        const text = truncate(value, 500);
        return text ? { text } : null;
    }

    if (typeof value !== "object") return null;

    const text = truncate(pickFirst(value, [
        "text",
        "content",
        "message",
        "caption",
        "description",
        "body",
    ]), 500);
    const date = truncate(pickFirst(value, [
        "date",
        "time",
        "createdAt",
        "created_at",
        "timestamp",
        "publishedAt",
        "published_at",
    ]), 120);
    const url = normalizeUrl(pickFirst(value, ["url", "postUrl", "link"])) || "";

    const post = compactObject({ text, date, url });
    return Object.keys(post).length ? post : null;
};

const extractLatestPosts = (item) => {
    const candidates = [
        item?.posts,
        item?.latestPosts,
        item?.timelinePosts,
        item?.recentPosts,
    ];

    for (const candidate of candidates) {
        if (!Array.isArray(candidate) || !candidate.length) continue;
        const posts = candidate
            .map(normalizePostEntry)
            .filter(Boolean)
            .slice(0, MAX_POSTS);
        if (posts.length) return posts;
    }

    const singlePost = normalizePostEntry(pickFirst(item, ["post", "latestPost", "recentPost"]));
    return singlePost ? [singlePost] : [];
};

const summarizeActorItem = (item, pageUrl, rank) => {
    const url = normalizeUrl(pickFirst(item, [
        "url",
        "pageUrl",
        "pageURL",
        "facebookUrl",
        "profileUrl",
        "link",
    ])) || pageUrl;
    const website = normalizeUrl(pickFirst(item, ["website", "web", "externalUrl", "site"])) || "";
    const latestPosts = extractLatestPosts(item);

    return compactObject({
        rank,
        url,
        name: truncate(pickFirst(item, ["pageName", "name", "title", "pageTitle"]), 200),
        username: truncate(pickFirst(item, ["username", "userName", "handle", "pageHandle"]), 120),
        category: truncate(pickFirst(item, ["category", "categories", "pageCategory"]), 240),
        followers: toNumber(pickFirst(item, ["followers", "pageFollowers", "followCount"])),
        likes: toNumber(pickFirst(item, ["likes", "pageLikes", "likeCount"])),
        description: truncate(pickFirst(item, ["description", "about", "bio", "intro", "summary"]), 1200),
        website,
        phone: truncate(pickFirst(item, ["phone", "phoneNumber", "contactPhone"]), 120),
        email: truncate(pickFirst(item, ["email", "contactEmail"]), 200),
        address: truncate(pickFirst(item, ["address", "location", "contactAddress"]), 240),
        latest_posts: latestPosts,
        raw_keys: Object.keys(item || {}).slice(0, 12),
    });
};

const runActor = async (pageUrl) => {
    const token = normalizeText(process.env.APIFY_TOKEN || process.env.APIFY_API_TOKEN);
    if (!token) return "Error: APIFY_TOKEN is not configured";

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
        const response = await fetch(`${APIFY_ENDPOINT}?token=${encodeURIComponent(token)}`, {
            method: "POST",
            signal: controller.signal,
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "nub-agent/1.0",
            },
            body: JSON.stringify({
                startUrls: [{ url: pageUrl }],
            }),
        });

        const rawText = await response.text();
        if (!response.ok) {
            return `Error: Apify actor returned status ${response.status}${rawText ? ` (${truncate(rawText, 300)})` : ""}`;
        }

        try {
            return JSON.parse(rawText);
        } catch {
            return `Error: Apify actor returned invalid JSON (${truncate(rawText, 300)})`;
        }
    } catch (error) {
        if (error?.name === "AbortError") {
            return `Error: Apify actor timed out after ${REQUEST_TIMEOUT_MS / 1000}s. Use a narrower Facebook page URL or rerun later.`;
        }
        return `Error: Apify request failed (${error?.message || error})`;
    } finally {
        clearTimeout(timer);
    }
};

module.exports = {
    definition: {
        type: "function",
        function: {
            name: "facebook_page_lookup",
            strict: true,
            description: "Fetch public Facebook page data for a person, creator, brand, or organization page using Apify. Use when you need page facts, about text, contact info, or recent posts. Accepts a Facebook page URL, page handle, or a person/page name that should be resolved to a public Facebook page. This is for public pages, not private profiles.",
            parameters: {
                type: "object",
                properties: {
                    page: {
                        type: "string",
                        description: "Public Facebook page URL, page handle, or person/page name, for example https://www.facebook.com/zuck, zuck, or Prince Keith Andrei.",
                    },
                    limit: {
                        type: "integer",
                        minimum: 1,
                        maximum: 5,
                        description: "How many returned dataset items to include in the result summary (1-5).",
                    },
                },
                required: ["page"],
                additionalProperties: false,
            },
        },
    },
    handler: async (args) => {
        try {
            const pageUrl = await resolveFacebookPageUrl(args?.page);
            const limit = clamp(parseInt(args?.limit, 10) || 3, 1, MAX_RESULTS);
            if (!pageUrl) return "Error: page is required";

            const result = await runActor(pageUrl);
            if (typeof result === "string") return result;

            const items = (Array.isArray(result) ? result : []).slice(0, limit);
            if (!items.length) {
                return `No public Facebook page data found for "${pageUrl}"`;
            }

            return JSON.stringify({
                source: "apify",
                actor: APIFY_ACTOR,
                page: pageUrl,
                count: items.length,
                items: items.map((item, index) => summarizeActorItem(item, pageUrl, index + 1)),
            });
        } catch (error) {
            return `Error: facebook page lookup failed (${error?.message || error})`;
        }
    },
};
