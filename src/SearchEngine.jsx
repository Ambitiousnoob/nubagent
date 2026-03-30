import React, { useState, useRef, useEffect, useCallback } from "react";

const CHAT_API = "/api/chat";
const SEARCH_API = "/api/search";
const FETCH_API = "/api/fetch";
const MODEL_NAME = "nub-agent";
const SOURCE_TARGET = 60;
const FETCH_TARGET = 24;
const SEARCH_SWARM_SIZE = 3;
const SYNTHESIS_SWARM_SIZE = 3;
const FETCH_CONCURRENCY = 4;
const FETCH_MAX_CHARS = 900;
const MAX_UPLOAD_FILES = 6;
const MAX_TEXT_ATTACHMENT_CHARS = 12000;
const MAX_TOTAL_ATTACHMENT_CHARS = 32000;
const MAX_IMAGE_UPLOAD_BYTES = 8 * 1024 * 1024;
const TEXT_FILE_NAME_RE = /\.(txt|md|markdown|json|csv|js|mjs|cjs|ts|jsx|tsx|py|rb|go|rs|java|c|h|cpp|hpp|html|css|scss|sass|xml|yaml|yml|toml|ini|env|log)$/i;
const IMAGE_FILE_NAME_RE = /\.(png|jpe?g|gif|webp|bmp|svg)$/i;

const createId = () => `${Date.now()}-${Math.random().toString(16).slice(2)}`;

const getDomain = (url) => {
    try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return url; }
};

const getFavicon = (url) => {
    try { return `https://www.google.com/s2/favicons?domain=${new URL(url).origin}&sz=32`; } catch { return null; }
};

const extractSources = (text = "") => {
    const sources = [];
    const seen = new Set();

    try {
        const parsed = JSON.parse(text);
        const items = Array.isArray(parsed) ? parsed : (parsed?.results || parsed?.organic || []);
        for (const item of items) {
            if (!item?.url || seen.has(item.url)) continue;
            seen.add(item.url);
            sources.push({
                title: item.title || getDomain(item.url),
                url: item.url,
                description: item.description || item.snippet || "",
                date: item.date || null,
            });
        }
    } catch {
        for (const match of String(text || "").matchAll(/https?:\/\/[^\s"',>\]]+/g)) {
            const url = match[0].replace(/[.,;)]+$/, "");
            if (seen.has(url)) continue;
            seen.add(url);
            sources.push({ title: getDomain(url), url, description: "" });
        }
    }

    return sources;
};

const extractToolSources = (tools = []) => {
    const merged = [];
    const seen = new Set();

    for (const tool of Array.isArray(tools) ? tools : []) {
        const candidates = [tool?.args, tool?.note, tool?.result, tool?.content];
        for (const candidate of candidates) {
            const text = typeof candidate === "string" ? candidate : JSON.stringify(candidate || {});
            for (const source of extractSources(text)) {
                if (seen.has(source.url)) continue;
                seen.add(source.url);
                merged.push(source);
            }
        }
    }

    return merged;
};

const extractAnswerParts = (text = "") => {
    const clean = String(text || "").replace(/##\s*Sources?[\s\S]*$/i, "").trim();
    const lines = clean.split("\n");
    const heading = (lines[0] || "").replace(/^#+\s*/, "").trim();
    const body = lines.length > 1 ? lines.slice(1).join("\n").trim() : clean;
    return {
        heading: heading || "Answer",
        body,
    };
};

const extractCitationNumbers = (text = "") => ([
    ...new Set(
        [...String(text || "").matchAll(/\[(\d+)\]/g)]
            .map((match) => Number(match[1]))
            .filter((value) => Number.isInteger(value) && value > 0),
    ),
]);

const buildAttributedSources = (answerText = "", evidenceEntries = []) => {
    const successfulEntries = (Array.isArray(evidenceEntries) ? evidenceEntries : []).filter((entry) => stripFetchMeta(entry?.content || "").trim());
    const byCitation = new Map(
        successfulEntries
            .map((entry) => [Number(entry?.source?.citationIndex), entry?.source])
            .filter((entry) => Number.isInteger(entry[0]) && entry[0] > 0 && entry[1]?.url),
    );
    const citedSources = extractCitationNumbers(answerText)
        .map((citationIndex) => byCitation.get(citationIndex))
        .filter(Boolean);

    if (citedSources.length) return dedupeSources(citedSources, FETCH_TARGET);
    return dedupeSources(successfulEntries.map((entry) => entry.source).filter(Boolean), FETCH_TARGET);
};

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const normalizeTextBlock = (value) => String(value ?? "").replace(/\u0000/g, "").replace(/\r\n?/g, "\n").trim();

const truncateText = (value, max = MAX_TEXT_ATTACHMENT_CHARS) => {
    const text = String(value ?? "");
    if (text.length <= max) return text;
    const suffix = `\n...[truncated ${text.length - max} chars]`;
    if (max <= suffix.length + 8) return text.slice(0, max);
    return `${text.slice(0, max - suffix.length)}${suffix}`;
};

const formatBytes = (bytes = 0) => {
    if (!Number.isFinite(Number(bytes))) return "0 B";
    const value = Number(bytes);
    if (value < 1024) return `${value} B`;
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
    return `${(value / (1024 * 1024)).toFixed(1)} MB`;
};

const isProbablyTextFile = (file) => {
    const type = String(file?.type || "");
    return TEXT_FILE_NAME_RE.test(file?.name || "")
        || type.startsWith("text/")
        || /(json|javascript|typescript|xml|yaml|csv)/i.test(type);
};

const isImageFile = (file) => {
    const type = String(file?.type || "");
    return type.startsWith("image/") || IMAGE_FILE_NAME_RE.test(file?.name || "");
};

const readFileAsDataUrl = (file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error(`Failed to read ${file?.name || "file"}.`));
    reader.readAsDataURL(file);
});

const postJson = async (url, payload, signal) => {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        cache: "no-store",
        signal,
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        let detail = response.statusText;
        try {
            const parsed = await response.json();
            detail = parsed?.error || parsed?.message || detail;
        } catch {
            detail = await response.text().catch(() => detail);
        }
        throw new Error(`${response.status} ${detail}`);
    }

    return response.json();
};

const buildSwarmQueries = (query) => {
    const base = String(query || "").trim();
    const operatorHeavy = /\b(site:|filetype:|intitle:|inurl:|after:|before:)\b/i.test(base);
    const variants = operatorHeavy
        ? [
            base,
            `${base} analysis`,
            `${base} evidence`,
            `${base} expert commentary`,
            `${base} latest`,
        ]
        : [
            base,
            `${base} overview`,
            `${base} evidence`,
            `${base} latest`,
            `${base} expert analysis`,
        ];

    return [...new Set(variants.map((item) => item.trim()).filter(Boolean))].slice(0, SEARCH_SWARM_SIZE);
};

const dedupeSources = (items = [], limit = SOURCE_TARGET) => {
    const seen = new Set();
    const merged = [];
    for (const item of Array.isArray(items) ? items : []) {
        const url = String(item?.url || "").trim();
        if (!url || seen.has(url)) continue;
        seen.add(url);
        merged.push({
            title: item?.title || getDomain(url),
            url,
            description: item?.description || item?.snippet || "",
            date: item?.date || null,
            source: item?.source || null,
        });
        if (merged.length >= limit) break;
    }
    return merged;
};

const chunkArray = (items = [], size = 1) => {
    const result = [];
    const chunkSize = Math.max(1, size);
    for (let index = 0; index < items.length; index += chunkSize) {
        result.push(items.slice(index, index + chunkSize));
    }
    return result;
};

const stripFetchMeta = (content = "") => (
    String(content || "")
        .replace(/^<!--[\s\S]*?-->\s*/g, "")
        .trim()
);

const buildEvidenceBlock = (entry, index) => {
    const source = entry?.source || {};
    const excerpt = stripFetchMeta(entry?.content || "").replace(/\s+/g, " ").trim();
    const parts = [
        `[${index}] ${source.title || getDomain(source.url || "")}`,
        `URL: ${source.url || ""}`,
    ];

    if (source.description) parts.push(`Search snippet: ${source.description}`);
    if (excerpt) parts.push(`Fetched excerpt: ${excerpt}`);
    return parts.join("\n");
};

const buildAttachmentBadge = (attachment) => {
    if (!attachment) return "";
    const parts = [attachment.kind === "image" ? "Image" : "Text", formatBytes(attachment.size)];
    if (attachment.truncated) parts.push("truncated");
    return parts.join(" • ");
};

const buildAttachmentManifest = (attachments = []) => (
    attachments
        .map((attachment) => `- ${attachment.kind === "image" ? "Image" : "Text"}: ${attachment.name} (${formatBytes(attachment.size)})${attachment.truncated ? " [truncated]" : ""}`)
        .join("\n")
);

const buildTextAttachmentSections = (attachments = []) => (
    attachments
        .filter((attachment) => attachment?.kind === "text" && attachment?.textContent)
        .map((attachment) => (
            `--- File: ${attachment.name} (${formatBytes(attachment.size)})${attachment.truncated ? " [truncated]" : ""} ---\n${attachment.textContent}`
        ))
        .join("\n\n")
);

const buildAttachmentContext = (attachments = []) => {
    if (!attachments.length) return "";
    const sections = [`Attached files:\n${buildAttachmentManifest(attachments)}`];
    const textSections = buildTextAttachmentSections(attachments);
    if (textSections) sections.push(textSections);
    return sections.join("\n\n").trim();
};

const getAttachmentAnalysisPrompt = (query, attachments = []) => {
    const normalized = normalizeTextBlock(query);
    if (normalized) return normalized;
    return attachments.some((attachment) => attachment?.kind === "image")
        ? "Analyze the attached image files."
        : "Analyze the attached files.";
};

const getChatText = (payload = {}) => String(payload?.choices?.[0]?.message?.content || payload?.output_text || "").trim();

const buildSessionAttachments = (attachments = []) => (
    attachments.map((attachment) => ({
        id: attachment.id || createId(),
        name: attachment.name || "attachment",
        kind: attachment.kind === "image" ? "image" : "text",
        size: Number(attachment.size) || 0,
        truncated: Boolean(attachment.truncated),
        dataUrl: attachment.kind === "image" ? String(attachment.dataUrl || "") : "",
        textContent: attachment.kind === "text" ? String(attachment.textContent || "") : "",
    }))
);

const analyzeAttachments = async (query, attachments, signal) => {
    if (!attachments.length) return "";

    const imagePayloads = attachments
        .filter((attachment) => attachment?.kind === "image" && attachment?.dataUrl)
        .map((attachment) => attachment.dataUrl);
    const attachmentContext = buildAttachmentContext(attachments);
    const payload = {
        model: MODEL_NAME,
        stream: false,
        use_tools: false,
        messages: [
            {
                role: "system",
                content: "You analyze uploaded files for a research interface. Start with a single H1 title. If images are attached, identify what is visible. If text files are attached, summarize the relevant facts. Be concrete and concise.",
            },
            {
                role: "user",
                content: `User request: ${getAttachmentAnalysisPrompt(query, attachments)}${attachmentContext ? `\n\n${attachmentContext}` : ""}`,
            },
        ],
        ...(imagePayloads.length ? { images: imagePayloads } : {}),
    };

    const response = await postJson(CHAT_API, payload, signal);
    return getChatText(response);
};

const runConcurrent = async (items, limit, worker) => {
    const values = Array.isArray(items) ? items : [];
    const concurrency = Math.max(1, limit || 1);
    const results = new Array(values.length);
    let cursor = 0;

    const runNext = async () => {
        while (cursor < values.length) {
            const current = cursor;
            cursor += 1;
            results[current] = await worker(values[current], current);
        }
    };

    await Promise.all(Array.from({ length: Math.min(concurrency, values.length) }, runNext));
    return results;
};

const renderInlineMarkup = (str, sources = []) => {
    const parts = String(str || "").split(/(\*\*[^*]+\*\*|`[^`]+`|\[\d+\])/g);
    return parts.map((part, index) => {
        if (/^\*\*[^*]+\*\*$/.test(part)) return <strong key={index}>{part.slice(2, -2)}</strong>;
        if (/^`[^`]+`$/.test(part)) return <code key={index} className="ic">{part.slice(1, -1)}</code>;
        if (/^\[\d+\]$/.test(part)) {
            const sourceIndex = Number(part.slice(1, -1)) - 1;
            return (
                <a key={index} href={sources[sourceIndex]?.url || "#"} target="_blank" rel="noopener noreferrer" className="cite">
                    {part.slice(1, -1)}
                </a>
            );
        }
        return part;
    });
};

const extractHighlightPoints = (text = "", limit = 4) => {
    const bullets = String(text || "")
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => /^[-*]\s+/.test(line))
        .map((line) => line.replace(/^[-*]\s+/, "").trim())
        .filter(Boolean);
    if (bullets.length) return bullets.slice(0, limit);

    return String(text || "")
        .replace(/\n+/g, " ")
        .split(/(?<=[.!?])\s+/)
        .map((line) => line.trim())
        .filter(Boolean)
        .slice(0, limit);
};

const buildResultSummary = (text = "") => {
    const cleaned = String(text || "")
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => line && !/^[-*]\s+/.test(line) && !/^#{1,6}\s+/.test(line));
    return cleaned.join(" ");
};

const SOURCE_COLOR_CLASSES = [
    "blue", "sky", "red", "emerald", "purple", "amber", "stone", "cyan",
    "indigo", "rose", "teal", "lime", "violet", "pink", "fuchsia", "orange", "yellow", "slate",
];

const POINT_STYLES = [
    { className: "red", symbol: "!" },
    { className: "amber", symbol: "↓" },
    { className: "green", symbol: "✓" },
    { className: "blue", symbol: "•" },
];

const getSourceBadge = (source, index) => {
    const domain = getDomain(source?.url || "");
    if (/(\.gov|\.mil|\.edu)\b/i.test(domain) || /(nist|europa|who|un\.org|cisa|gov$)/i.test(domain)) {
        return { label: "Official", className: "official" };
    }
    if (index < 3) return { label: "Relevant", className: "relevant" };
    return null;
};

const getSourceLetterMark = (source) => {
    const domain = getDomain(source?.url || "");
    const segments = domain.split(".").filter(Boolean);
    const core = segments[0] || domain || "S";
    const letters = core.replace(/[^a-z0-9]/gi, "").slice(0, 2).toUpperCase();
    return letters || "S";
};

const getSourceCategory = (source) => {
    const domain = getDomain(source?.url || "");
    if (!domain) return "Web";
    if (/(\.gov|\.mil)\b/i.test(domain) || /(nist|nih|cisa|fda|who|un\.org|europa)/i.test(domain)) return "Government";
    if (/(\.edu)\b/i.test(domain) || /(arxiv|nature|science|springer|ieee|acm|pubmed|doi\.org|nejm|jamanetwork|cell|mit\.edu)/i.test(domain)) return "Research";
    if (/(reuters|apnews|bbc|euronews|nytimes|washingtonpost|theguardian|wired|technologyreview|cyberscoop|techcrunch|theverge)/i.test(domain)) return "News";
    if (/(google|cloudflare|microsoft|openai|anthropic|aws|ibm|meta|github|docs\.)/i.test(domain)) return "Vendor";
    return "Web";
};

const formatGeneratedAt = (value) => {
    const date = value ? new Date(value) : null;
    if (!date || Number.isNaN(date.getTime())) return "Just now";
    return date
        .toISOString()
        .replace("T", " ")
        .replace(/\.\d{3}Z$/, " UTC");
};

const buildCoverageRows = (query, researchMeta = {}, sources = []) => {
    const normalizedQuery = String(query || "").trim();
    const operatorMode = /\b(site:|filetype:|intitle:|inurl:|after:|before:)\b/i.test(normalizedQuery);
    const attachmentCount = Number(researchMeta?.attachments || 0);
    const queryCount = Number(researchMeta?.searchCount || 0);
    const rankedSites = Number(researchMeta?.rankedSites || 0);
    const fetchedSites = Number(researchMeta?.fetchedSites || 0);
    const synthesisWorkers = Number(researchMeta?.synthesisWorkers || 0);
    const rows = [
        {
            label: "Search mode",
            spec: normalizedQuery
                ? (operatorMode ? "Operator-guided web research with targeted query expansion" : "Natural-language web research with query expansion")
                : "Attachment-only analysis",
        },
        {
            label: "Query workers",
            spec: `${Math.max(queryCount, normalizedQuery ? 1 : 0)} search path${Math.max(queryCount, normalizedQuery ? 1 : 0) === 1 ? "" : "s"} executed`,
        },
        {
            label: "Sites ranked",
            spec: `${Math.max(rankedSites, sources.length)} unique site${Math.max(rankedSites, sources.length) === 1 ? "" : "s"} kept after dedupe`,
        },
        {
            label: "Pages fetched",
            spec: `${fetchedSites} page${fetchedSites === 1 ? "" : "s"} read for evidence extraction`,
        },
        {
            label: "Cited sources",
            spec: `${sources.length} source${sources.length === 1 ? "" : "s"} carried into the final answer`,
        },
        {
            label: "Synthesis",
            spec: `${Math.max(synthesisWorkers, 1)} summarization worker${Math.max(synthesisWorkers, 1) === 1 ? "" : "s"} merged into one final response`,
        },
    ];

    if (attachmentCount) {
        rows.push({
            label: "Attachments",
            spec: `${attachmentCount} uploaded file${attachmentCount === 1 ? "" : "s"} used as extra evidence`,
        });
    }

    rows.push({
        label: "Generated",
        spec: formatGeneratedAt(researchMeta?.generatedAt),
    });

    return rows;
};

function Markdown({ text, sources = [] }) {
    const lines = String(text || "").split("\n");
    const elements = [];
    let list = [];

    const flush = (key) => {
        if (!list.length) return;
        elements.push(
            <ul key={`ul${key}`} className="md-ul">
                {list.map((item, index) => <li key={index}>{renderInlineMarkup(item, sources)}</li>)}
            </ul>,
        );
        list = [];
    };

    lines.forEach((line, index) => {
        const heading = line.match(/^(#{1,3})\s(.*)/);
        if (heading) {
            flush(index);
            const level = heading[1].length;
            const content = renderInlineMarkup(heading[2], sources);
            if (level === 1) elements.push(<h1 key={index} className="md-h1">{content}</h1>);
            else if (level === 2) elements.push(<h2 key={index} className="md-h2">{content}</h2>);
            else elements.push(<h3 key={index} className="md-h3">{content}</h3>);
            return;
        }

        if (/^[-*] /.test(line)) {
            list.push(line.slice(2));
            return;
        }

        if (!line.trim()) {
            flush(index);
            elements.push(<br key={index} />);
            return;
        }

        flush(index);
        elements.push(<p key={index} className="md-p">{renderInlineMarkup(line, sources)}</p>);
    });

    flush("end");
    return <>{elements}</>;
}

function SearchingCard({
    queries,
    done,
    statusText = "",
    title = "Searching the web",
    doneTitle = "Searched the web",
    icon = "🌐",
}) {
    const [shown, setShown] = useState(0);
    const [dot, setDot] = useState(0);

    useEffect(() => {
        const timer = setInterval(() => setShown((count) => Math.min(count + 1, queries.length)), 550);
        return () => clearInterval(timer);
    }, [queries.length]);

    useEffect(() => {
        if (done) return undefined;
        const timer = setInterval(() => setDot((value) => (value + 1) % 3), 380);
        return () => clearInterval(timer);
    }, [done]);

    return (
        <div className={`scard ${done ? "scard--done" : ""}`}>
            <div className="scard__head">
                <span className="scard__icon">{icon}</span>
                <span className="scard__title">{done ? doneTitle : title}</span>
            </div>
            {queries.length > 0 && (
                <div className="scard__chips">
                    {queries.slice(0, shown).map((query, index) => (
                        <span key={index} className="qchip" style={{ animationDelay: `${index * 0.08}s` }}>
                            <span className="qchip__dot">🔍</span>
                            {query}
                        </span>
                    ))}
                </div>
            )}
            {!done && (
                <div className="scard__dots">
                    {[0, 1, 2].map((index) => <span key={index} className={`sdot ${dot === index ? "sdot--on" : ""}`}>●</span>)}
                    <span className="scard__searching">{statusText || "Searching..."}</span>
                </div>
            )}
            {done && statusText && <div className="scard__searching">{statusText}</div>}
        </div>
    );
}

const STEPS = [
    ["Identifying", "key search areas and topics"],
    ["Preparing", "autonomous research workflow"],
    ["Analyzing", "query for optimal results"],
    ["Selecting", "search strategies and operators"],
];

function PlanningCard({ done }) {
    const [visible, setVisible] = useState(0);

    useEffect(() => {
        let count = 0;
        const timer = setInterval(() => {
            count += 1;
            setVisible(count);
            if (count >= STEPS.length) clearInterval(timer);
        }, 650);
        return () => clearInterval(timer);
    }, []);

    return (
        <div className={`scard ${done ? "scard--done" : ""}`}>
            <div className="scard__head">
                <span className="scard__icon">💡</span>
                <div>
                    <div className="scard__title">Research Planning</div>
                    <div className="scard__sub">Analyzing requirements</div>
                </div>
            </div>
            <div className="scard__steps">
                {STEPS.map(([strongText, rest], index) => (
                    <div key={index} className={`pstep ${index < visible ? "pstep--done" : "pstep--wait"}`}>
                        <span className="pstep__icon">{index < visible ? "✓" : "○"}</span>
                        <span><strong>{strongText}</strong> <span className="pstep__rest">{rest}...</span></span>
                    </div>
                ))}
            </div>
        </div>
    );
}

function SourcePills({ sources }) {
    if (!sources.length) return null;
    return (
        <div className="pills">
            {sources.slice(0, 6).map((source, index) => (
                <a key={index} href={source.url} target="_blank" rel="noopener noreferrer" className="pill">
                    {getFavicon(source.url) && (
                        <img
                            src={getFavicon(source.url)}
                            className="pill__fav"
                            alt=""
                            onError={(event) => {
                                event.currentTarget.style.display = "none";
                            }}
                        />
                    )}
                    <span className="pill__n">{index + 1}</span>
                    <span className="pill__domain">{getDomain(source.url)}</span>
                </a>
            ))}
        </div>
    );
}

function LibertyTableCard({ icon, title, badge, children, note }) {
    return (
        <div className="la-card">
            <div className="la-card-header">
                <div className="la-card-icon">{icon}</div>
                <span className="la-card-title">{title}</span>
                {badge ? <span className="la-card-badge">{badge}</span> : null}
            </div>
            <div className="la-table-wrap">{children}</div>
            {note ? <div className="la-note">{note}</div> : null}
        </div>
    );
}

function LibertySourceTable({ sources = [] }) {
    const rows = sources.slice(0, 7);
    if (!rows.length) return null;

    return (
        <LibertyTableCard icon="⌘" title="Key Sources & Findings" badge={`${rows.length} entr${rows.length === 1 ? "y" : "ies"}`}>
            <table className="la-table">
                <thead>
                    <tr>
                        <th className="col-num">#</th>
                        <th className="col-source">Source</th>
                        <th className="col-discipline">Type</th>
                        <th className="col-findings">Main Findings</th>
                    </tr>
                </thead>
                <tbody>
                    {rows.map((source, index) => (
                        <tr key={source.url || `${index}`}>
                            <td className="col-num">{index + 1}</td>
                            <td className="col-source">
                                <div className="la-source-name">{source.title || getDomain(source.url)}</div>
                                <div className="la-source-ref">{getDomain(source.url)}</div>
                                <a href={source.url} target="_blank" rel="noopener noreferrer" className="la-source-url">
                                    ↗ {source.url}
                                </a>
                            </td>
                            <td className="col-discipline">
                                <span className="la-discipline-tag">{getSourceCategory(source)}</span>
                            </td>
                            <td className="col-findings">{source.description || "Used as cited evidence in the final answer."}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </LibertyTableCard>
    );
}

function LibertyCoverageTable({ query, researchMeta = {}, sources = [] }) {
    const rows = buildCoverageRows(query, researchMeta, sources);
    if (!rows.length) return null;

    return (
        <LibertyTableCard
            icon="◌"
            title="Scope & Coverage"
            badge={`${rows.length} criteria`}
            note="Coverage is derived from the actual search and fetch pipeline for this answer."
        >
            <table className="la-table">
                <thead>
                    <tr>
                        <th className="col-criterion">Criterion</th>
                        <th className="col-spec">Specification</th>
                    </tr>
                </thead>
                <tbody>
                    {rows.map((row) => (
                        <tr key={row.label}>
                            <td className="col-criterion"><span className="la-criterion-label">{row.label}</span></td>
                            <td className="col-spec">{row.spec}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </LibertyTableCard>
    );
}

function LibertyResultCard({ query, heading, body, sources = [], searchCount = 0, researchMeta = {} }) {
    const visibleSources = sources.slice(0, 20);
    const summary = buildResultSummary(body);
    const points = extractHighlightPoints(body, 4);
    const topDomains = [...new Set(visibleSources.map((source) => getDomain(source.url)).filter(Boolean))];

    return (
        <div className="la-result">
            <div className="la-header">
                <div className="la-logo">✳</div>
                <span className="la-brand">nub-agent</span>
                <span className="la-badge">{sources.length} Source{sources.length === 1 ? "" : "s"}</span>
            </div>

            <div className="la-body">
                <div className="la-query">Query</div>
                <div className="la-question">{renderInlineMarkup(query || heading, sources)}</div>
                <div className="la-divider" />

                <div className="la-summary-label">AI Summary</div>
                <div className="la-summary-title">{renderInlineMarkup(heading, sources)}</div>
                {summary ? <div className="la-summary-text">{renderInlineMarkup(summary, sources)}</div> : null}

                {points.length ? (
                    <div className="la-points">
                        {points.map((point, index) => {
                            const style = POINT_STYLES[index % POINT_STYLES.length];
                            return (
                                <div key={`${index}-${point.slice(0, 32)}`} className="la-point">
                                    <div className={`la-point-icon ${style.className}`}>{style.symbol}</div>
                                    <p>{renderInlineMarkup(point, sources)}</p>
                                </div>
                            );
                        })}
                    </div>
                ) : null}

                {(visibleSources.length || Object.keys(researchMeta || {}).length) ? (
                    <div className="la-tables">
                        <LibertySourceTable sources={visibleSources} />
                        <LibertyCoverageTable query={query} researchMeta={{ ...researchMeta, searchCount }} sources={visibleSources} />
                    </div>
                ) : null}
            </div>

            {visibleSources.length ? (
                <div className="la-footer">
                    <span>{sources.length} source{sources.length === 1 ? "" : "s"} from {Math.max(searchCount, 1)} search{Math.max(searchCount, 1) === 1 ? "" : "es"}</span>
                    <span>{topDomains.slice(0, 3).join(", ")}{topDomains.length > 3 ? ` +${topDomains.length - 3}` : ""}</span>
                </div>
            ) : null}
        </div>
    );
}

function AttachmentList({ attachments = [], onRemove, compact = false }) {
    if (!attachments.length) return null;

    return (
        <div className={`att-list ${compact ? "att-list--compact" : ""}`}>
            {attachments.map((attachment) => (
                <div key={attachment.id || attachment.name} className={`att-chip ${compact ? "att-chip--compact" : ""}`}>
                    {attachment.kind === "image" && attachment.dataUrl ? (
                        <img className="att-chip__thumb" src={attachment.dataUrl} alt={attachment.name} />
                    ) : (
                        <div className="att-chip__icon">{attachment.kind === "image" ? "🖼" : "📄"}</div>
                    )}
                    <div className="att-chip__meta">
                        <div className="att-chip__name">{attachment.name}</div>
                        <div className="att-chip__sub">{buildAttachmentBadge(attachment)}</div>
                    </div>
                    {typeof onRemove === "function" && (
                        <button type="button" className="att-chip__remove" onClick={() => onRemove(attachment.id)} title={`Remove ${attachment.name}`}>
                            ×
                        </button>
                    )}
                </div>
            ))}
        </div>
    );
}

function UserMsg({ text, attachments = [] }) {
    return (
        <div className="umsg">
            <div className="umsg__av"><span>U</span></div>
            <div className="umsg__body">
                <div className="umsg__name">You</div>
                {text ? <div className="umsg__text">{text}</div> : null}
                <AttachmentList attachments={attachments} />
            </div>
            <div className="umsg__acts">
                <button className="act-btn" title="Edit">✏</button>
                <button className="act-btn" title="Copy">⧉</button>
            </div>
        </div>
    );
}

function BotMsg({ msg, isLast, streaming, sessionQuery = "" }) {
    const {
        heading,
        body,
        sources = [],
        showPlanning,
        showSearching,
        queries = [],
        searchDone,
        statusText = "",
        activityTitle,
        activityDoneTitle,
        activityIcon,
        researchMeta,
    } = msg;
    const active = isLast && streaming;

    return (
        <div className="bmsg">
            {showPlanning && <PlanningCard done={searchDone} />}
            {showSearching && (
                <SearchingCard
                    queries={queries}
                    done={searchDone}
                    statusText={statusText}
                    title={activityTitle}
                    doneTitle={activityDoneTitle}
                    icon={activityIcon}
                />
            )}
            {(body || (active && !showSearching)) && (
                <div className="bmsg__ans">
                    <LibertyResultCard
                        query={sessionQuery}
                        heading={heading}
                        body={body || ""}
                        sources={sources}
                        searchCount={queries.length}
                        researchMeta={researchMeta}
                    />
                    {active && !body && <div className="bmsg__body"><span className="caret">▍</span></div>}
                </div>
            )}
        </div>
    );
}

function Landing({ onSearch, uploads = [], onOpenUpload, onRemoveUpload, uploadStatus = "" }) {
    const [query, setQuery] = useState("");
    const inputRef = useRef(null);

    useEffect(() => { inputRef.current?.focus(); }, []);

    const submit = (event) => {
        event?.preventDefault();
        if (query.trim() || uploads.length) onSearch(query.trim());
    };

    const examples = [
        "How does quantum computing threaten modern encryption?",
        'site:arxiv.org "retrieval augmented generation" after:2024-01-01',
        "Best open source LLMs benchmark 2025",
        "intitle:CVE Apache Log4j critical vulnerability",
    ];

    return (
        <div className="land">
            <div className="land__logo"><span className="land__star">✳</span><span>nub-agent</span></div>
            <h1 className="land__h1">What do you want to know?</h1>
            <p className="land__sub">AI research with real citations, dork operators, and deep web reading.</p>
            <form className="land__form" onSubmit={submit}>
                <button type="button" className="land__attach" onClick={onOpenUpload} title="Attach files">📎</button>
                <input
                    ref={inputRef}
                    className="land__in"
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    placeholder="Ask anything or use site: filetype: intitle: operators..."
                    autoComplete="off"
                />
                <button type="submit" className="land__btn" disabled={!query.trim() && !uploads.length}>
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <path d="M8 2L14 8L8 14M2 8H14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                </button>
            </form>
            {(uploads.length || uploadStatus) && (
                <div className="land__uploads">
                    <AttachmentList attachments={uploads} onRemove={onRemoveUpload} compact />
                    {uploadStatus ? <div className="upload-status">{uploadStatus}</div> : null}
                </div>
            )}
            <div className="land__exs">
                {examples.map((example) => <button key={example} className="land__ex" onClick={() => onSearch(example)}>{example}</button>)}
            </div>
        </div>
    );
}

export default function SearchEngine() {
    const [sessions, setSessions] = useState([]);
    const [activeId, setActiveId] = useState(null);
    const [streaming, setStreaming] = useState(false);
    const [input, setInput] = useState("");
    const [navActive, setNavActive] = useState("search");
    const [pendingUploads, setPendingUploads] = useState([]);
    const [uploadStatus, setUploadStatus] = useState("");

    const abortRef = useRef(null);
    const bottomRef = useRef(null);
    const inputRef = useRef(null);
    const fileInputRef = useRef(null);
    const uploadStatusTimerRef = useRef(null);

    const active = sessions.find((session) => session.id === activeId);
    const isLanding = !active && !streaming;

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [sessions, streaming]);

    const setTransientUploadStatus = useCallback((value) => {
        if (uploadStatusTimerRef.current) {
            clearTimeout(uploadStatusTimerRef.current);
            uploadStatusTimerRef.current = null;
        }
        setUploadStatus(value);
        if (value) {
            uploadStatusTimerRef.current = setTimeout(() => {
                setUploadStatus("");
                uploadStatusTimerRef.current = null;
            }, 3200);
        }
    }, []);

    useEffect(() => () => {
        if (uploadStatusTimerRef.current) clearTimeout(uploadStatusTimerRef.current);
    }, []);

    const patchLastBot = useCallback((sessionId, patch) => {
        setSessions((previous) => previous.map((session) => {
            if (session.id !== sessionId) return session;
            const messages = [...session.messages];
            let index = -1;
            for (let i = messages.length - 1; i >= 0; i -= 1) {
                if (messages[i].role === "bot") {
                    index = i;
                    break;
                }
            }
            if (index < 0) return session;
            messages[index] = {
                ...messages[index],
                ...(typeof patch === "function" ? patch(messages[index]) : patch),
            };
            return { ...session, messages };
        }));
    }, []);

    const revealAnswer = useCallback(async (sessionId, finalText, sources, signal, extraPatch = {}) => {
        const { heading, body } = extractAnswerParts(finalText);
        patchLastBot(sessionId, {
            heading,
            body: "",
            sources,
            showPlanning: false,
            showSearching: false,
            searchDone: true,
            ...extraPatch,
        });

        const chunks = body ? body.match(/.{1,34}(\s|$)/g) || [body] : [];
        let built = "";
        for (const chunk of chunks) {
            if (signal?.aborted) break;
            built += chunk;
            patchLastBot(sessionId, {
                heading,
                body: built.trimEnd(),
                sources,
                showPlanning: false,
                showSearching: false,
                searchDone: true,
                ...extraPatch,
            });
            await new Promise((resolve) => setTimeout(resolve, 18));
        }

        patchLastBot(sessionId, {
            heading,
            body: body || heading,
            sources,
            showPlanning: false,
            showSearching: false,
            searchDone: true,
            ...extraPatch,
        });
    }, [patchLastBot]);

    const handleFileUpload = useCallback(async (event) => {
        const files = Array.from(event.target.files || []);
        event.target.value = "";
        if (!files.length || streaming) return;

        const remainingSlots = Math.max(0, MAX_UPLOAD_FILES - pendingUploads.length);
        if (remainingSlots <= 0) {
            setTransientUploadStatus(`Attachment limit reached (${MAX_UPLOAD_FILES}). Remove one to add another.`);
            return;
        }

        const selected = files.slice(0, remainingSlots);
        let totalChars = pendingUploads
            .filter((attachment) => attachment?.kind === "text")
            .reduce((sum, attachment) => sum + (attachment?.textContent?.length || 0), 0);
        let skippedCount = Math.max(0, files.length - selected.length);
        let truncatedCount = 0;
        let unreadableCount = 0;
        let oversizedCount = 0;
        const nextUploads = [];

        for (const file of selected) {
            if (isImageFile(file)) {
                if ((file.size || 0) > MAX_IMAGE_UPLOAD_BYTES) {
                    oversizedCount += 1;
                    skippedCount += 1;
                    continue;
                }
                try {
                    const dataUrl = await readFileAsDataUrl(file);
                    if (!dataUrl) {
                        unreadableCount += 1;
                        continue;
                    }
                    nextUploads.push({
                        id: createId(),
                        name: file.name,
                        kind: "image",
                        size: file.size,
                        mimeType: file.type || "image/*",
                        dataUrl,
                        truncated: false,
                    });
                } catch {
                    unreadableCount += 1;
                }
                continue;
            }

            if (!isProbablyTextFile(file)) {
                skippedCount += 1;
                continue;
            }

            try {
                let text = normalizeTextBlock(await file.text());
                if (!text) {
                    skippedCount += 1;
                    continue;
                }

                const remainingBudget = MAX_TOTAL_ATTACHMENT_CHARS - totalChars;
                if (remainingBudget <= 0) {
                    skippedCount += 1;
                    continue;
                }

                let wasTruncated = false;
                if (text.length > MAX_TEXT_ATTACHMENT_CHARS) {
                    text = truncateText(text, MAX_TEXT_ATTACHMENT_CHARS);
                    wasTruncated = true;
                }
                if (text.length > remainingBudget) {
                    text = truncateText(text, remainingBudget);
                    wasTruncated = true;
                }
                if (!text.trim()) {
                    skippedCount += 1;
                    continue;
                }

                totalChars += text.length;
                if (wasTruncated) truncatedCount += 1;
                nextUploads.push({
                    id: createId(),
                    name: file.name,
                    kind: "text",
                    size: file.size,
                    mimeType: file.type || "text/plain",
                    textContent: text,
                    truncated: wasTruncated,
                });
            } catch {
                unreadableCount += 1;
            }
        }

        if (!nextUploads.length) {
            const reason = unreadableCount
                ? "Selected files could not be read."
                : skippedCount
                    ? "No supported files were added."
                    : "No readable content found.";
            setTransientUploadStatus(`Upload error: ${reason}`);
            return;
        }

        setPendingUploads((previous) => [...previous, ...nextUploads]);
        const notes = [];
        if (truncatedCount) notes.push(`${truncatedCount} truncated`);
        if (oversizedCount) notes.push(`${oversizedCount} too large`);
        if (skippedCount) notes.push(`${skippedCount} skipped`);
        if (unreadableCount) notes.push(`${unreadableCount} unreadable`);
        setTransientUploadStatus(
            `Attached ${nextUploads.length} file${nextUploads.length > 1 ? "s" : ""}${notes.length ? ` (${notes.join(", ")})` : ""}.`,
        );
    }, [pendingUploads, setTransientUploadStatus, streaming]);

    const removePendingUpload = useCallback((uploadId) => {
        setPendingUploads((previous) => previous.filter((attachment) => attachment?.id !== uploadId));
    }, []);

    const openFilePicker = useCallback(() => {
        if (streaming) return;
        fileInputRef.current?.click();
    }, [streaming]);

    const runSearch = useCallback(async (rawQuery, uploadsOverride = pendingUploads) => {
        const query = normalizeTextBlock(rawQuery);
        const attachments = buildSessionAttachments(uploadsOverride || []);
        if ((!query && !attachments.length) || streaming) return;

        const sessionId = createId();
        const complex = Boolean(query) && (query.length > 35 || /site:|filetype:|intitle:|inurl:|after:|before:/.test(query));
        const swarmQueries = query ? buildSwarmQueries(query) : attachments.map((attachment) => attachment.name).slice(0, 4);
        const displayQuery = query || `Analyze ${attachments.length} attached file${attachments.length > 1 ? "s" : ""}`;
        const userText = query || getAttachmentAnalysisPrompt("", attachments);
        const activityTitle = query ? "Searching the web" : "Inspecting uploads";
        const activityDoneTitle = query ? "Searched the web" : "Inspected uploads";
        const activityIcon = query ? "🌐" : "📎";

        const newSession = {
            id: sessionId,
            query: displayQuery,
            messages: [
                { id: createId(), role: "user", text: userText, attachments },
                {
                    id: createId(),
                    role: "bot",
                    heading: "",
                    body: "",
                    sources: [],
                    showPlanning: complex,
                    showSearching: true,
                    queries: swarmQueries,
                    searchDone: false,
                    statusText: query
                        ? `Launching ${swarmQueries.length} search workers...`
                        : `Reading ${attachments.length} uploaded file${attachments.length > 1 ? "s" : ""}...`,
                    activityTitle,
                    activityDoneTitle,
                    activityIcon,
                },
            ],
        };

        setSessions((previous) => [...previous, newSession]);
        setActiveId(sessionId);
        setStreaming(true);
        setInput("");
        setPendingUploads([]);
        setUploadStatus("");

        abortRef.current = new AbortController();
        const { signal } = abortRef.current;

        try {
            let attachmentDigest = "";
            if (attachments.length) {
                patchLastBot(sessionId, {
                    statusText: `Reading ${attachments.length} uploaded file${attachments.length > 1 ? "s" : ""}...`,
                });
                try {
                    attachmentDigest = await analyzeAttachments(query, attachments, signal);
                } catch (error) {
                    if (!query) throw error;
                    patchLastBot(sessionId, {
                        statusText: `Upload read failed (${error.message || "unknown error"}). Continuing with web research...`,
                    });
                }
            }

            if (!query) {
                if (!attachmentDigest) {
                    throw new Error("No readable attachment content was found.");
                }
                await revealAnswer(sessionId, attachmentDigest, [], signal, {
                    researchMeta: {
                        attachments: attachments.length,
                        generatedAt: new Date().toISOString(),
                        searchCount: 0,
                        rankedSites: 0,
                        fetchedSites: 0,
                        synthesisWorkers: 1,
                    },
                });
                return;
            }

            patchLastBot(sessionId, {
                statusText: `Running ${swarmQueries.length} search workers to collect up to ${SOURCE_TARGET} sites...`,
            });

            const searchSettled = await Promise.allSettled(
                swarmQueries.map((workerQuery) => postJson(SEARCH_API, { query: workerQuery }, signal)),
            );

            const mergedSources = dedupeSources(
                searchSettled.flatMap((item) => (
                    item.status === "fulfilled" ? item.value.results || [] : []
                )),
                SOURCE_TARGET,
            ).map((source, index) => ({
                ...source,
                citationIndex: index + 1,
            }));

            if (!mergedSources.length) {
                throw new Error("No searchable sources were found.");
            }

            patchLastBot(sessionId, {
                showPlanning: false,
                statusText: `Ranked ${mergedSources.length} sites. Starting ${FETCH_CONCURRENCY} fetch workers...`,
                sources: mergedSources,
            });

            let fetchedCount = 0;
            const evidenceEntries = await runConcurrent(mergedSources, FETCH_CONCURRENCY, async (source) => {
                try {
                    const fetchResult = await postJson(FETCH_API, {
                        url: source.url,
                        format: "text",
                        max_chars: FETCH_MAX_CHARS,
                    }, signal);

                    fetchedCount += 1;
                    patchLastBot(sessionId, {
                        statusText: `Fetched ${fetchedCount}/${mergedSources.length} sites. Building evidence graph...`,
                    });

                    return {
                        source,
                        content: fetchResult?.content || "",
                    };
                } catch (error) {
                    if (error?.name === "AbortError") throw error;
                    fetchedCount += 1;
                    patchLastBot(sessionId, {
                        statusText: `Fetched ${fetchedCount}/${mergedSources.length} sites. Some fetches were skipped.`,
                    });
                    return {
                        source,
                        content: "",
                        error: error?.message || "Fetch failed.",
                    };
                }
            });

            const fetchedEvidenceEntries = evidenceEntries.filter((entry) => stripFetchMeta(entry?.content || "").trim()).slice(0, FETCH_TARGET);
            if (!fetchedEvidenceEntries.length) {
                throw new Error("No readable source content was fetched.");
            }

            const chunkSize = Math.max(1, Math.ceil(fetchedEvidenceEntries.length / SYNTHESIS_SWARM_SIZE));
            const evidenceChunks = chunkArray(fetchedEvidenceEntries, chunkSize).slice(0, SYNTHESIS_SWARM_SIZE);

            patchLastBot(sessionId, {
                statusText: `Running ${evidenceChunks.length} synthesis workers over ${fetchedEvidenceEntries.length} fetched sites...`,
            });

            const draftSettled = await Promise.allSettled(
                evidenceChunks.map((chunk, index) => {
                    const evidenceBlock = chunk
                        .map((entry) => buildEvidenceBlock(entry, entry.source.citationIndex))
                        .join("\n\n---\n\n");

                    return postJson(CHAT_API, {
                        model: MODEL_NAME,
                        stream: false,
                        use_tools: false,
                        messages: [
                            {
                                role: "system",
                                content: `You are research worker ${index + 1}/${evidenceChunks.length}. Use only the supplied evidence. Pull out the most relevant facts for the user query. Preserve citation numbers like [12]. Do not invent new sources.`,
                            },
                            {
                                role: "user",
                                content: `User query: ${query}${attachmentDigest ? `\n\nUploaded file evidence:\n${truncateText(attachmentDigest, 3000)}` : ""}\n\nEvidence set:\n\n${evidenceBlock}\n\nProduce a concise evidence digest with bullet points and inline source numbers.`,
                            },
                        ],
                    }, signal);
                }),
            );

            const workerDrafts = draftSettled
                .map((item) => (
                    item.status === "fulfilled"
                        ? (item.value?.choices?.[0]?.message?.content || item.value?.output_text || "")
                        : ""
                ))
                .filter(Boolean);

            if (!workerDrafts.length) {
                throw new Error("Research workers did not produce a final digest.");
            }

            const sourceIndex = fetchedEvidenceEntries
                .map((entry) => {
                    const source = entry.source || {};
                    return `[${source.citationIndex}] ${source.title} — ${source.url}`;
                })
                .join("\n");

            patchLastBot(sessionId, {
                statusText: `Merging ${workerDrafts.length} research agents into the final cited answer...`,
            });

            const finalPayload = await postJson(CHAT_API, {
                model: MODEL_NAME,
                stream: false,
                use_tools: false,
                messages: [
                    {
                        role: "system",
                        content: `You are a synthesis model for a search swarm. Merge the worker drafts into one answer. Use citation numbers like [1], [2], [57] that refer to the provided source index. Start with a single H1 title. Then provide a concise but information-dense answer. End with a short section called ## Sources used listing the cited source numbers only.`,
                    },
                    {
                        role: "user",
                        content: `User query: ${query}${attachmentDigest ? `\n\nUploaded file evidence:\n${truncateText(attachmentDigest, 5000)}` : ""}\n\nSource index:\n${sourceIndex}\n\nWorker drafts:\n\n${workerDrafts.map((draft, index) => `### Worker ${index + 1}\n${draft}`).join("\n\n")}`,
                    },
                ],
            }, signal);

            const fullText = getChatText(finalPayload);
            await delay(250);
            const attributedSources = buildAttributedSources(fullText, fetchedEvidenceEntries);
            await revealAnswer(sessionId, fullText, attributedSources, signal, {
                researchMeta: {
                    attachments: attachments.length,
                    generatedAt: new Date().toISOString(),
                    searchCount: swarmQueries.length,
                    rankedSites: mergedSources.length,
                    fetchedSites: fetchedEvidenceEntries.length,
                    synthesisWorkers: evidenceChunks.length,
                },
            });
        } catch (error) {
            if (error.name === "AbortError") {
                patchLastBot(sessionId, {
                    heading: "Stopped",
                    body: "Generation stopped.",
                    showPlanning: false,
                    showSearching: false,
                    searchDone: true,
                });
            } else {
                patchLastBot(sessionId, {
                    heading: "Error",
                    body: error.message || "Search failed. Try again.",
                    showPlanning: false,
                    showSearching: false,
                    searchDone: true,
                });
            }
        } finally {
            setStreaming(false);
            abortRef.current = null;
            setTimeout(() => inputRef.current?.focus(), 80);
        }
    }, [patchLastBot, pendingUploads, revealAnswer, streaming]);

    const handleSubmit = (event) => {
        event?.preventDefault();
        if (input.trim() || pendingUploads.length) runSearch(input.trim(), pendingUploads);
    };

    const NAV = [
        { id: "search", icon: "⊙", label: "Search" },
        { id: "library", icon: "⊟", label: "Library" },
        { id: "discover", icon: "◫", label: "Discover" },
        { id: "watcher", icon: "⊞", label: "Watcher" },
        { id: "finance", icon: "⊠", label: "Finance" },
    ];

    const NAV_BOTTOM = [
        { id: "pro", icon: "♛", label: "Pro" },
        { id: "settings", icon: "⚙", label: "Settings" },
        { id: "account", icon: "◉", label: "Account" },
    ];

    return (
        <div className="se">
            <style>{`
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=DM+Mono&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0c0c0f;--bg2:#111116;--bg3:#18181e;--bgc:#141418;
  --sur:rgba(255,255,255,0.04);--surh:rgba(255,255,255,0.07);
  --bdr:rgba(255,255,255,0.08);
  --tx:#ededf0;--txd:#8888a0;--txm:#484858;
  --ac:#00c9a7;--acd:rgba(0,201,167,0.14);--ac2:#6d6dff;
  --red:#ff4444;
  --r:12px;--sw:56px;
  --font:'DM Sans',system-ui,sans-serif;--mono:'DM Mono',monospace;
}
html,body,#root{height:100%;background:var(--bg)}
.se{display:flex;height:100vh;overflow:hidden;font-family:var(--font);color:var(--tx);background:var(--bg)}
.sb{width:var(--sw);flex:0 0 var(--sw);background:var(--bg2);border-right:1px solid var(--bdr);display:flex;flex-direction:column;align-items:center;padding:14px 0;gap:2px;z-index:20}
.sb__logo{width:34px;height:34px;border-radius:9px;background:var(--ac);display:flex;align-items:center;justify-content:center;font-size:17px;color:#000;margin-bottom:14px;cursor:pointer;box-shadow:0 0 18px rgba(0,201,167,.28)}
.sb__new{width:34px;height:34px;border-radius:9px;border:1px solid var(--bdr);background:transparent;color:var(--txd);font-size:17px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all .2s;margin-bottom:6px}
.sb__new:hover{background:var(--surh);color:var(--tx)}
.sb__sp{flex:1}
.nb{width:38px;height:38px;border-radius:9px;border:none;background:transparent;color:var(--txm);font-size:16px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all .2s;position:relative}
.nb:hover{background:var(--sur);color:var(--txd)}
.nb--on{background:var(--sur);color:var(--tx)}
.nb--on::before{content:'';position:absolute;left:0;top:50%;transform:translateY(-50%);width:3px;height:18px;background:var(--ac);border-radius:0 2px 2px 0}
.mn{flex:1;display:flex;flex-direction:column;overflow:hidden;position:relative}
.tb{height:50px;border-bottom:1px solid var(--bdr);display:flex;align-items:center;padding:0 22px;gap:12px;background:var(--bg2);flex:0 0 auto}
.tb__q{font-size:14px;font-weight:500;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tb__b{padding:5px 12px;border-radius:8px;border:1px solid var(--bdr);background:transparent;color:var(--txd);font-family:var(--font);font-size:12px;cursor:pointer;transition:all .2s}
.tb__b:hover{background:var(--surh);color:var(--tx)}
.tb__up{padding:5px 13px;border-radius:8px;border:none;background:var(--ac);color:#000;font-family:var(--font);font-size:12px;font-weight:600;cursor:pointer;display:flex;align-items:center;gap:5px}
.chat{flex:1;overflow-y:auto;padding:28px 0 180px}
.chat::-webkit-scrollbar{width:4px}
.chat::-webkit-scrollbar-thumb{background:rgba(255,255,255,.07);border-radius:2px}
.chat__in{max-width:740px;margin:0 auto;padding:0 24px;display:flex;flex-direction:column;gap:24px}
.umsg{display:flex;align-items:flex-start;gap:11px}
.umsg__av{width:30px;height:30px;border-radius:50%;background:linear-gradient(135deg,#6d6dff,#00c9a7);display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:#fff;flex:0 0 30px}
.umsg__body{flex:1;min-width:0}
.umsg__name{font-size:11px;font-weight:600;color:var(--txm);margin-bottom:3px}
.umsg__text{font-size:15px;color:var(--tx);line-height:1.6}
.umsg__acts{display:flex;gap:3px;opacity:0;transition:opacity .2s;padding-top:3px}
.umsg:hover .umsg__acts{opacity:1}
.act-btn{width:26px;height:26px;border-radius:6px;border:none;background:transparent;color:var(--txm);font-size:12px;cursor:pointer;transition:all .15s}
.act-btn:hover{background:var(--sur);color:var(--txd)}
.scard{background:var(--bgc);border:1px solid var(--bdr);border-radius:var(--r);padding:15px;display:flex;flex-direction:column;gap:11px;animation:fsi .28s ease;margin-left:41px}
.scard--done{opacity:.6}
@keyframes fsi{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:translateY(0)}}
.scard__head{display:flex;align-items:center;gap:9px}
.scard__icon{font-size:15px;flex:0 0 auto}
.scard__title{font-size:13px;font-weight:600}
.scard__sub{font-size:11px;color:var(--txm);margin-top:1px}
.scard__chips{display:flex;flex-wrap:wrap;gap:5px}
.qchip{display:inline-flex;align-items:center;gap:4px;padding:3px 9px;border-radius:999px;background:var(--sur);border:1px solid var(--bdr);font-size:11px;color:var(--txd);animation:pi .22s ease both}
@keyframes pi{from{opacity:0;transform:scale(.9)}to{opacity:1;transform:scale(1)}}
.qchip__dot{font-size:9px;opacity:.5}
.scard__dots{display:flex;align-items:center;gap:3px}
.sdot{font-size:9px;color:var(--txm);transition:color .3s}
.sdot--on{color:var(--ac)}
.scard__searching{font-size:11px;color:var(--txd);margin-left:7px}
.scard__steps{display:flex;flex-direction:column;gap:7px}
.pstep{display:flex;align-items:flex-start;gap:7px;font-size:12px;transition:color .3s}
.pstep--wait{color:var(--txm)}
.pstep--done{color:var(--txd)}
.pstep__icon{font-size:11px;margin-top:1px;color:var(--ac);flex:0 0 auto;font-family:var(--mono)}
.pstep--wait .pstep__icon{color:var(--txm)}
.pstep__rest{color:var(--txm)}
.bmsg{display:flex;flex-direction:column;gap:11px;animation:fsi .3s ease}
.bmsg__ans{margin-left:41px;display:flex;flex-direction:column;gap:11px}
.bmsg__body{font-size:15px;line-height:1.78;color:var(--tx)}
.md-h1{font-size:19px;font-weight:700;margin:14px 0 5px;letter-spacing:-.02em}
.md-h2{font-size:16px;font-weight:600;margin:12px 0 4px}
.md-h3{font-size:14px;font-weight:600;margin:9px 0 3px;color:var(--txd)}
.md-p{margin:3px 0}
.md-ul{padding-left:18px;display:flex;flex-direction:column;gap:3px}
.ic{background:rgba(109,109,255,.12);border:1px solid rgba(109,109,255,.2);border-radius:4px;padding:1px 5px;font-family:var(--mono);font-size:12px;color:#a5a5ff}
.cite{display:inline-flex;align-items:center;justify-content:center;min-width:17px;height:17px;padding:0 3px;border-radius:4px;background:var(--acd);color:var(--ac);font-size:10px;font-weight:700;font-family:var(--mono);text-decoration:none;margin:0 2px;vertical-align:middle;border:1px solid rgba(0,201,167,.22)}
.cite:hover{background:rgba(0,201,167,.28)}
.caret{display:inline-block;color:var(--ac);animation:bl .65s step-end infinite;margin-left:2px}
@keyframes bl{0%,100%{opacity:1}50%{opacity:0}}
.pills{display:flex;flex-wrap:wrap;gap:5px;margin-top:3px}
.pill{display:inline-flex;align-items:center;gap:4px;padding:3px 9px;border-radius:999px;background:var(--sur);border:1px solid var(--bdr);font-size:11px;color:var(--txd);text-decoration:none;transition:all .2s}
.pill:hover{border-color:rgba(0,201,167,.3);color:var(--ac);background:var(--acd)}
.pill__fav{width:11px;height:11px;border-radius:2px}
.pill__n{font-size:9px;font-weight:700;font-family:var(--mono);color:var(--ac);background:var(--acd);border-radius:3px;padding:0 3px}
.pill__domain{max-width:110px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.la-result{
  --la-bg:#0c0a09;--la-card:rgba(255,255,255,0.03);--la-border:rgba(255,255,255,0.06);--la-border-hover:rgba(255,255,255,0.1);
  --la-text:#a8a29e;--la-text-dim:#57534e;--la-text-bright:#e7e5e4;--la-text-heading:#fafaf9;--la-accent:#f97316;
  --la-accent-bg:rgba(249,115,22,0.1);--la-accent-border:rgba(249,115,22,0.25);
  background:var(--la-bg);border:1px solid var(--la-border);border-radius:16px;overflow:hidden;color:var(--la-text)
}
.la-header{display:flex;align-items:center;gap:8px;padding:16px 20px;border-bottom:1px solid var(--la-border)}
.la-logo{width:22px;height:22px;background:var(--la-accent);border-radius:6px;display:flex;align-items:center;justify-content:center;color:#fff;font-size:12px;font-weight:700}
.la-brand{font-size:13px;font-weight:600;color:var(--la-text-bright);letter-spacing:-.01em}
.la-badge{font-size:10px;font-weight:600;color:var(--la-text-dim);background:var(--la-card);border:1px solid var(--la-border);padding:2px 8px;border-radius:20px;margin-left:auto}
.la-body{padding:20px}
.la-query{font-size:11px;font-weight:600;color:var(--la-text-dim);text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
.la-question{font-size:17px;font-weight:600;color:var(--la-text-heading);line-height:1.35;letter-spacing:-.02em;margin-bottom:16px}
.la-divider{height:1px;margin:0 0 16px;background:linear-gradient(90deg,transparent,var(--la-accent-border),transparent)}
.la-summary-label{font-size:11px;font-weight:600;color:var(--la-text-dim);text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px}
.la-summary-title{font-size:15px;font-weight:600;color:var(--la-text-heading);line-height:1.35;letter-spacing:-.01em;margin-bottom:10px}
.la-summary-text{font-size:13px;color:var(--la-text);line-height:1.7;margin-bottom:16px}
.la-summary-text strong{color:var(--la-text-bright);font-weight:500}
.la-points{display:flex;flex-direction:column;gap:8px;margin-bottom:20px}
.la-point{display:flex;gap:10px;padding:10px 12px;background:var(--la-card);border:1px solid var(--la-border);border-radius:10px}
.la-point-icon{width:18px;height:18px;border-radius:5px;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px;font-size:10px;font-weight:700}
.la-point-icon.red{background:rgba(239,68,68,0.12);color:#f87171}
.la-point-icon.amber{background:rgba(245,158,11,0.12);color:#fbbf24}
.la-point-icon.green{background:rgba(34,197,94,0.12);color:#4ade80}
.la-point-icon.blue{background:rgba(59,130,246,0.12);color:#60a5fa}
.la-point p{font-size:12px;color:var(--la-text);line-height:1.5}
.la-tables{display:flex;flex-direction:column;gap:20px;margin-top:20px}
.la-card{background:var(--la-bg);border:1px solid var(--la-border);border-radius:14px;overflow:hidden}
.la-card-header{display:flex;align-items:center;gap:8px;padding:14px 18px;border-bottom:1px solid var(--la-border)}
.la-card-icon{width:20px;height:20px;border-radius:5px;display:flex;align-items:center;justify-content:center;background:var(--la-accent-bg);border:1px solid var(--la-accent-border);color:var(--la-accent);font-size:10px;font-weight:700}
.la-card-title{font-size:13px;font-weight:600;color:var(--la-text-heading);letter-spacing:-.01em}
.la-card-badge{font-size:10px;font-weight:600;color:var(--la-text-dim);background:var(--la-card);border:1px solid var(--la-border);padding:2px 8px;border-radius:20px;margin-left:auto}
.la-table-wrap{overflow-x:auto}
.la-table{width:100%;border-collapse:collapse;font-size:12.5px}
.la-table thead th{text-align:left;padding:10px 16px;font-size:10px;font-weight:600;color:var(--la-text-dim);text-transform:uppercase;letter-spacing:.08em;background:rgba(255,255,255,0.02);border-bottom:1px solid var(--la-border);white-space:nowrap}
.la-table tbody td{padding:12px 16px;border-bottom:1px solid var(--la-border);vertical-align:top;color:var(--la-text);line-height:1.55}
.la-table tbody tr:last-child td{border-bottom:none}
.la-table tbody tr{transition:background 120ms}
.la-table tbody tr:hover{background:var(--la-card)}
.la-table .col-num{width:36px;text-align:center;color:var(--la-text-dim);font-size:11px;font-weight:500;font-variant-numeric:tabular-nums}
.la-table .col-source{min-width:200px}
.la-table .col-discipline{min-width:130px;white-space:nowrap}
.la-table .col-findings{min-width:280px}
.la-table .col-criterion{min-width:140px;white-space:nowrap}
.la-table .col-spec{min-width:300px}
.la-source-name{color:var(--la-text-bright);font-weight:500}
.la-source-ref{font-size:11px;color:var(--la-text-dim);margin-top:2px}
.la-source-url{display:inline-flex;align-items:center;gap:4px;font-size:10px;color:var(--la-accent);margin-top:4px;opacity:.85;text-decoration:none}
.la-source-url:hover{opacity:1}
.la-discipline-tag{display:inline-block;font-size:10px;font-weight:500;padding:2px 8px;border-radius:6px;white-space:nowrap;background:rgba(255,255,255,0.04);border:1px solid var(--la-border);color:var(--la-text)}
.la-criterion-label{color:var(--la-text-bright);font-weight:500;font-size:12px}
.la-note{padding:10px 18px;border-top:1px solid var(--la-border);font-size:11px;color:var(--la-text-dim);font-style:italic}
.la-sources-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.la-sources-title{font-size:11px;font-weight:600;color:var(--la-text-dim);text-transform:uppercase;letter-spacing:.08em}
.la-sources-count{font-size:11px;color:var(--la-text-dim)}
.la-sources{display:flex;flex-direction:column;gap:4px}
.la-source{display:flex;align-items:flex-start;gap:10px;padding:10px 12px;border-radius:10px;border:1px solid transparent;transition:all 120ms ease}
.la-source:hover{background:var(--la-card);border-color:var(--la-border-hover)}
.la-source-favicon{width:28px;height:28px;border-radius:7px;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px;font-size:10px;font-weight:700}
.la-source-favicon.blue{background:rgba(59,130,246,0.1);color:#60a5fa;border:1px solid rgba(59,130,246,0.15)}
.la-source-favicon.sky{background:rgba(14,165,233,0.1);color:#38bdf8;border:1px solid rgba(14,165,233,0.15)}
.la-source-favicon.red{background:rgba(239,68,68,0.1);color:#f87171;border:1px solid rgba(239,68,68,0.15)}
.la-source-favicon.emerald{background:rgba(16,185,129,0.1);color:#34d399;border:1px solid rgba(16,185,129,0.15)}
.la-source-favicon.purple{background:rgba(168,85,247,0.1);color:#c084fc;border:1px solid rgba(168,85,247,0.15)}
.la-source-favicon.amber{background:rgba(245,158,11,0.1);color:#fbbf24;border:1px solid rgba(245,158,11,0.15)}
.la-source-favicon.stone{background:rgba(168,162,158,0.08);color:#a8a29e;border:1px solid rgba(168,162,158,0.12)}
.la-source-favicon.cyan{background:rgba(6,182,212,0.1);color:#22d3ee;border:1px solid rgba(6,182,212,0.15)}
.la-source-favicon.indigo{background:rgba(99,102,241,0.1);color:#818cf8;border:1px solid rgba(99,102,241,0.15)}
.la-source-favicon.rose{background:rgba(244,63,94,0.1);color:#fb7185;border:1px solid rgba(244,63,94,0.15)}
.la-source-favicon.teal{background:rgba(20,184,166,0.1);color:#2dd4bf;border:1px solid rgba(20,184,166,0.15)}
.la-source-favicon.lime{background:rgba(132,204,22,0.1);color:#a3e635;border:1px solid rgba(132,204,22,0.15)}
.la-source-favicon.violet{background:rgba(139,92,246,0.1);color:#a78bfa;border:1px solid rgba(139,92,246,0.15)}
.la-source-favicon.pink{background:rgba(236,72,153,0.1);color:#f472b6;border:1px solid rgba(236,72,153,0.15)}
.la-source-favicon.fuchsia{background:rgba(192,38,211,0.1);color:#e879f9;border:1px solid rgba(192,38,211,0.15)}
.la-source-favicon.orange{background:rgba(249,115,22,0.1);color:#fb923c;border:1px solid rgba(249,115,22,0.15)}
.la-source-favicon.yellow{background:rgba(234,179,8,0.1);color:#facc15;border:1px solid rgba(234,179,8,0.15)}
.la-source-favicon.slate{background:rgba(100,116,139,0.1);color:#94a3b8;border:1px solid rgba(100,116,139,0.15)}
.la-source-content{flex:1;min-width:0}
.la-source-meta{display:flex;align-items:center;gap:6px;margin-bottom:3px}
.la-source-domain{font-size:10px;font-weight:500;color:var(--la-text-dim)}
.la-source-tag{font-size:10px;font-weight:600;padding:1px 6px;border-radius:4px}
.la-source-tag.relevant{color:#fb923c;background:var(--la-accent-bg);border:1px solid var(--la-accent-border)}
.la-source-tag.official{color:#4ade80;background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.2)}
.la-source-title{font-size:13px;font-weight:500;color:var(--la-text-bright);line-height:1.35;transition:color 120ms}
.la-source:hover .la-source-title{color:#fb923c}
.la-source-desc{font-size:12px;color:var(--la-text-dim);line-height:1.5;margin-top:3px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.la-source-arrow{flex-shrink:0;margin-top:4px;color:var(--la-text-dim);transition:color 120ms}
.la-source:hover .la-source-arrow{color:var(--la-text)}
.la-footer{padding:10px 20px;border-top:1px solid var(--la-border);display:flex;align-items:center;justify-content:space-between;font-size:11px;color:var(--la-text-dim);gap:12px;flex-wrap:wrap}
.att-list{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.att-list--compact{margin-top:0}
.att-chip{display:flex;align-items:center;gap:8px;min-width:0;max-width:100%;padding:8px 10px;border:1px solid var(--bdr);border-radius:10px;background:var(--sur)}
.att-chip--compact{padding:6px 8px}
.att-chip__thumb,.att-chip__icon{width:34px;height:34px;border-radius:8px;flex:0 0 34px}
.att-chip__thumb{object-fit:cover;background:var(--bg2)}
.att-chip__icon{display:flex;align-items:center;justify-content:center;background:var(--bg2);font-size:15px}
.att-chip__meta{min-width:0;display:flex;flex-direction:column;gap:2px}
.att-chip__name{font-size:12px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:280px}
.att-chip__sub{font-size:11px;color:var(--txm)}
.att-chip__remove{width:22px;height:22px;border:none;border-radius:6px;background:transparent;color:var(--txm);cursor:pointer;flex:0 0 auto}
.att-chip__remove:hover{background:var(--surh);color:var(--tx)}
.bot{position:absolute;bottom:0;left:0;right:0;padding:14px 22px calc(14px + env(safe-area-inset-bottom));background:linear-gradient(transparent,var(--bg) 32%);z-index:10}
.bot__wrap{max-width:740px;margin:0 auto;background:var(--bg3);border:1px solid var(--bdr);border-radius:16px;overflow:hidden;transition:border-color .2s,box-shadow .2s}
.bot__wrap:focus-within{border-color:rgba(0,201,167,.35);box-shadow:0 0 0 3px rgba(0,201,167,.06)}
.bot__uploads{padding:12px 12px 0}
.bot__row{display:flex;align-items:center;padding:4px 4px 4px 16px;gap:7px}
.bot__in{flex:1;border:none;background:transparent;font-family:var(--font);font-size:15px;color:var(--tx);outline:none;padding:9px 0}
.bot__in::placeholder{color:var(--txm)}
.bot__in:disabled{opacity:.5}
.send-btn{width:36px;height:36px;border-radius:10px;border:none;background:var(--ac);color:#000;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:opacity .2s;flex:0 0 auto}
.send-btn:hover{opacity:.85}
.send-btn:disabled{opacity:.3;cursor:not-allowed}
.stop-btn{width:36px;height:36px;border-radius:10px;border:none;background:var(--red);color:#fff;font-size:13px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex:0 0 auto;transition:opacity .2s}
.stop-btn:hover{opacity:.85}
.bot__bar{display:flex;align-items:center;padding:5px 9px;border-top:1px solid var(--bdr);gap:5px}
.bb{display:flex;align-items:center;gap:5px;padding:5px 10px;border-radius:7px;border:1px solid var(--bdr);background:transparent;color:var(--txd);font-family:var(--font);font-size:12px;cursor:pointer;transition:all .2s}
.bb:hover{background:var(--sur);color:var(--tx)}
.bb--on{background:var(--sur);color:var(--tx);border-color:rgba(255,255,255,.12)}
.bb__sp{flex:1}
.ib{width:30px;height:30px;border-radius:7px;border:none;background:transparent;color:var(--txm);font-size:14px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all .2s}
.ib:hover{background:var(--sur);color:var(--txd)}
.land{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100%;padding:40px 24px 190px;gap:18px;text-align:center}
.land__logo{display:flex;align-items:center;gap:7px;font-size:14px;font-weight:600;color:var(--txd);margin-bottom:6px}
.land__star{font-size:20px;color:var(--ac);filter:drop-shadow(0 0 10px var(--ac))}
.land__h1{font-size:clamp(24px,4vw,38px);font-weight:700;letter-spacing:-.03em;line-height:1.15}
.land__sub{font-size:14px;color:var(--txd);max-width:400px;line-height:1.65}
.land__form{width:100%;max-width:630px;background:var(--bg3);border:1px solid var(--bdr);border-radius:14px;display:flex;align-items:center;padding:4px 4px 4px 17px;gap:7px;margin-top:6px;transition:border-color .2s,box-shadow .2s}
.land__form:focus-within{border-color:rgba(0,201,167,.38);box-shadow:0 0 0 3px rgba(0,201,167,.06)}
.land__attach{width:36px;height:36px;border-radius:10px;border:1px solid var(--bdr);background:transparent;color:var(--txd);cursor:pointer;display:flex;align-items:center;justify-content:center;flex:0 0 auto}
.land__attach:hover{background:var(--sur);color:var(--tx)}
.land__in{flex:1;border:none;background:transparent;font-family:var(--font);font-size:15px;color:var(--tx);outline:none;padding:10px 0}
.land__in::placeholder{color:var(--txm)}
.land__btn{width:36px;height:36px;border-radius:10px;border:none;background:var(--ac);color:#000;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:opacity .2s;flex:0 0 auto}
.land__btn:disabled{opacity:.32;cursor:not-allowed}
.land__btn:hover:not(:disabled){opacity:.85}
.land__uploads{display:flex;flex-direction:column;gap:8px;width:100%;max-width:630px}
.upload-status{font-size:12px;color:var(--txd)}
.land__exs{display:flex;flex-direction:column;gap:5px;width:100%;max-width:630px;margin-top:2px}
.land__ex{padding:9px 15px;border-radius:9px;border:1px solid var(--bdr);background:var(--sur);color:var(--txd);font-family:var(--font);font-size:13px;cursor:pointer;transition:all .2s;text-align:left}
.land__ex:hover{border-color:rgba(0,201,167,.24);color:var(--tx);background:var(--acd)}
@media(max-width:600px){
  .tb{padding:0 14px}
  .chat__in{padding:0 14px}
  .bot{padding:10px 12px}
}
`}</style>

            <aside className="sb">
                <div className="sb__logo" onClick={() => { setActiveId(null); setInput(""); }}>✳</div>
                <button
                    className="sb__new"
                    onClick={() => {
                        setActiveId(null);
                        setInput("");
                        if (streaming) abortRef.current?.abort();
                    }}
                    title="New search"
                >
                    ＋
                </button>
                {NAV.map((item) => (
                    <button key={item.id} className={`nb ${navActive === item.id ? "nb--on" : ""}`} title={item.label} onClick={() => setNavActive(item.id)}>
                        {item.icon}
                    </button>
                ))}
                <div className="sb__sp" />
                {NAV_BOTTOM.map((item) => <button key={item.id} className="nb" title={item.label}>{item.icon}</button>)}
            </aside>

            <div className="mn">
                {!isLanding && (
                    <div className="tb">
                        <div className="tb__q">{active?.query || ""}</div>
                        <button className="tb__up">⬆ Upgrade to Pro</button>
                        <button className="tb__b">···</button>
                        <button className="tb__b">⬆ Share</button>
                    </div>
                )}

                <div className="chat">
                    {isLanding ? (
                        <Landing
                            onSearch={runSearch}
                            uploads={pendingUploads}
                            onOpenUpload={openFilePicker}
                            onRemoveUpload={removePendingUpload}
                            uploadStatus={uploadStatus}
                        />
                    ) : (
                        <div className="chat__in">
                            {active?.messages.map((message, index) => (
                                message.role === "user"
                                    ? <UserMsg key={message.id} text={message.text} attachments={message.attachments} />
                                    : <BotMsg key={message.id} msg={message} isLast={index === active.messages.length - 1} streaming={streaming} sessionQuery={active.query} />
                            ))}
                            <div ref={bottomRef} />
                        </div>
                    )}
                </div>

                {!isLanding && (
                    <div className="bot">
                        <div className="bot__wrap">
                            {(pendingUploads.length || uploadStatus) && (
                                <div className="bot__uploads">
                                    <AttachmentList attachments={pendingUploads} onRemove={removePendingUpload} compact />
                                    {uploadStatus ? <div className="upload-status">{uploadStatus}</div> : null}
                                </div>
                            )}
                            <form className="bot__row" onSubmit={handleSubmit}>
                                <input
                                    ref={inputRef}
                                    className="bot__in"
                                    value={input}
                                    onChange={(event) => setInput(event.target.value)}
                                    placeholder={pendingUploads.length ? "Ask about the uploaded files or continue research..." : "Ask a new question..."}
                                    disabled={streaming}
                                    autoComplete="off"
                                />
                                {streaming ? (
                                    <button type="button" className="stop-btn" onClick={() => abortRef.current?.abort()} title="Stop">■</button>
                                ) : (
                                    <button type="submit" className="send-btn" disabled={!input.trim() && !pendingUploads.length}>
                                        <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
                                            <path d="M7.5 2L13 7.5L7.5 13M1 7.5H13" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                                        </svg>
                                    </button>
                                )}
                            </form>
                            <div className="bot__bar">
                                <button className="bb bb--on">🔍 Search</button>
                                <button className="bb">🔧</button>
                                <button className="bb">🔔</button>
                                <div className="bb__sp" />
                                <button className="ib" title="Attach" onClick={openFilePicker} disabled={streaming}>📎</button>
                                {streaming ? (
                                    <button className="ib" style={{ color: "var(--red)" }} onClick={() => abortRef.current?.abort()}>■</button>
                                ) : (
                                    <button className="ib" title="Voice">🎙</button>
                                )}
                            </div>
                        </div>
                    </div>
                )}
                <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept=".txt,.md,.markdown,.json,.csv,.js,.mjs,.cjs,.ts,.jsx,.tsx,.py,.rb,.go,.rs,.java,.c,.h,.cpp,.hpp,.html,.css,.scss,.sass,.xml,.yaml,.yml,.toml,.ini,.env,.log,text/*,application/json,image/*"
                    style={{ display: "none" }}
                    onChange={handleFileUpload}
                />
            </div>
        </div>
    );
}
