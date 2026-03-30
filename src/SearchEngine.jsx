import React, { useState, useRef, useEffect, useCallback } from "react";
import Company, { prepareSearchQuery, finalizeResearchAnswer } from "./lib/index.js";
import { buildAttributedSourcesFromEvidence, findSourceForCitation, getDisplaySourceNumber } from "./lib/citations.js";
import { buildSessionShareUrl, getSavedSessions, promptToCopySessionUrl, saveSession } from "./lib/library.js";
import {
    PLANNING_STEPS,
    SUBAGENT_STAGE_LABELS,
    compileResearchPlan,
    countSubagentAssignments,
    formatSubagentStatus,
    summarizeDag,
    summarizeSubagents,
} from "./lib/researchOrchestration.js";
import { invokeResearchRuntime, sendResearchControl } from "./lib/researchClient.js";
import { withClientStateKey } from "./lib/clientStateKey.js";
import { useSettingsStore } from "./store/useSettingsStore.js";

const CHAT_API = "/api/chat";
const SEARCH_API = "/api/search";
const READ_API = "/api/read";
const MODEL_NAME = "nub-agent";
const SOURCE_TARGET = 60;
const FETCH_TARGET = 24;
const FETCH_PLAN_BUFFER = 8;
const SEARCH_SWARM_SIZE = 4;
const SEARCH_VARIANT_RESULT_TARGET = 18;
const SYNTHESIS_SWARM_SIZE = 3;
const FETCH_CONCURRENCY = 4;
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
    const requestPayload = withClientStateKey(payload || {});
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        cache: "no-store",
        signal,
        body: JSON.stringify(requestPayload),
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

const inferDepthPreference = (query) => {
    const normalized = normalizeTextBlock(query).toLowerCase();
    if (!normalized) return "balanced";
    if (/\b(quick|brief|fast|speed|high level|high-level|tl;dr)\b/.test(normalized)) return "speed";
    if (/\b(deep|deeper|thorough|exhaustive|comprehensive|detailed|deep dive)\b/.test(normalized)) return "deep";
    return "balanced";
};

const extractJsonObject = (text = "") => {
    const raw = String(text || "").trim();
    if (!raw) return null;

    const fencedMatch = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
    const candidate = fencedMatch?.[1] || raw;
    const start = candidate.indexOf("{");
    const end = candidate.lastIndexOf("}");
    if (start < 0 || end <= start) return null;

    try {
        return JSON.parse(candidate.slice(start, end + 1));
    } catch {
        return null;
    }
};

const clampScore = (value, fallback = 0.5) => {
    const resolved = Number(value);
    if (!Number.isFinite(resolved)) return fallback;
    return Math.max(0, Math.min(1, resolved));
};

const parseTribunalResponse = (text = "") => {
    const parsed = extractJsonObject(text);
    if (!parsed || typeof parsed !== "object") return null;
    const critics = parsed.critics && typeof parsed.critics === "object" ? parsed.critics : {};
    const targetedDimension = String(
        parsed.targeted_dimension
        || parsed.targetedDimension
        || parsed.focus
        || "",
    ).trim();

    return {
        internal_consistency: clampScore(
            critics.internal_consistency ?? critics.internalConsistency ?? parsed.internal_consistency ?? parsed.internalConsistency,
            0.72,
        ),
        coverage: clampScore(
            critics.coverage ?? parsed.coverage,
            0.72,
        ),
        user_goal_alignment: clampScore(
            critics.user_goal_alignment ?? critics.userGoalAlignment ?? parsed.user_goal_alignment ?? parsed.userGoalAlignment,
            0.74,
        ),
        targeted_dimension: targetedDimension || "coverage",
        rewrite_brief: String(parsed.rewrite_brief || parsed.rewriteBrief || parsed.guidance || "").trim(),
    };
};

const formatDimensionLabel = (value = "") => (
    String(value || "").replace(/_/g, " ").trim() || "coverage"
);

const buildResearchPlanBrief = (plan = {}) => {
    const lines = [
        `Framework: Research Framework v${plan.frameworkVersion || "3.0"}`,
        `Domain: ${plan.domain?.label || "General Research"}${plan.domain?.taxonomy ? ` (${plan.domain.taxonomy})` : ""}`,
        `Scope: ${plan.scope?.label || "Broad Research"}`,
        `Output mode: ${plan.outputMode?.label || "State-of-the-Field"}`,
        `Pareto mode: ${plan.pareto?.mode || "balanced"}`,
        `Search lanes: ${Math.max(1, Number(plan.searchQueries?.length) || 0)}`,
        `Counter-hypotheses: ${(plan.queryMatrix?.counterHypotheses || []).length}`,
        `Safety checks: ${plan.safety?.activeCount || 0}`,
    ];

    const ambiguousAxes = Array.isArray(plan.intentConfidence?.ambiguousAxes)
        ? plan.intentConfidence.ambiguousAxes.filter(Boolean)
        : [];
    if (ambiguousAxes.length) {
        lines.push(`Ambiguous axes: ${ambiguousAxes.join(", ")}`);
    }

    if (plan.continuity?.active && plan.continuity?.overlap) {
        lines.push(`Session continuity overlap: ${Math.round(plan.continuity.overlap * 100)}%`);
    }

    return lines.join("\n");
};

const buildOutputModeInstruction = (outputMode = {}) => {
    switch (outputMode?.id) {
    case "tutorial":
        return "Organize the response as a tutorial with setup, core concepts, and practical application guidance.";
    case "controversy_map":
        return "Organize the response as a controversy map with the dominant view, strongest counter-view, and unresolved tensions.";
    case "gap_analysis":
        return "Organize the response as a gap analysis with current consensus, missing evidence, and the highest-value next questions.";
    case "replication_crisis_report":
        return "Organize the response as a replication crisis report with reproducibility risks, failed replications, and robustness signals.";
    case "foundational_review":
        return "Organize the response as a foundational review with historical context, seminal ideas, and the current field position.";
    case "decision_brief":
        return "Organize the response as a decision brief with recommendation, tradeoffs, risk profile, confidence, and reversibility.";
    case "policy_recommendation":
        return "Organize the response as a policy recommendation with recommendation, stakeholder impact, risk profile, and implementation caveats.";
    case "engineering_action_plan":
        return "Organize the response as an engineering action plan with recommendation, rollout steps, risk profile, and reversibility.";
    default:
        return "Organize the response as a state-of-the-field briefing with consensus, disagreement, evidence quality, and practical takeaways.";
    }
};

const buildContinuityBlock = (continuity = {}) => (
    continuity?.active && continuity?.summary
        ? `Prior session continuity (${Math.round((continuity.overlap || 0) * 100)}% overlap):\n${continuity.summary}`
        : ""
);

const dedupeSources = (items = [], limit = SOURCE_TARGET) => {
    return Company.rag.mergeSourcesByCanonicalUrl(items, { limit });
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

const buildPersistedAttachments = (attachments = []) => (
    attachments.map((attachment) => ({
        id: attachment.id || createId(),
        name: attachment.name || "attachment",
        kind: attachment.kind === "image" ? "image" : "text",
        size: Number(attachment.size) || 0,
        mimeType: String(attachment.mimeType || ""),
        truncated: Boolean(attachment.truncated),
    }))
);

const hydrateSavedSession = (session) => {
    const query = String(session?.query || "").trim();
    const heading = String(session?.heading || query || "Saved Session").trim();
    const body = String(session?.body || "").trim();

    return {
        id: session?.id || createId(),
        query: query || heading,
        messages: [
            {
                id: createId(),
                role: "user",
                text: query || heading,
                attachments: buildSessionAttachments(session?.attachments || []),
            },
            {
                id: createId(),
                role: "bot",
                heading,
                body: body || heading,
                sources: Array.isArray(session?.sources) ? session.sources : [],
                showPlanning: false,
                showSearching: false,
                searchDone: true,
                researchMeta: session?.researchMeta,
            },
        ],
    };
};

const buildSavedSessionRecord = ({
    id,
    query,
    heading = "",
    body = "",
    sources = [],
    attachments = [],
    researchMeta = null,
} = {}) => ({
    id,
    query: String(query || ""),
    heading: String(heading || ""),
    body: String(body || ""),
    sources: Array.isArray(sources) ? sources : [],
    attachments: buildPersistedAttachments(attachments),
    researchMeta: researchMeta && typeof researchMeta === "object" ? researchMeta : null,
});

const clearSharedSessionUrl = () => {
    if (typeof window === "undefined") return;

    const url = new URL(window.location.href);
    if (!url.searchParams.has("session")) return;

    url.searchParams.delete("session");
    try {
        window.history.replaceState({}, "", url.toString());
    } catch {
        const fallback = `${url.pathname}${url.search}${url.hash}`;
        try {
            window.history.replaceState({}, "", fallback || "/");
        } catch {
            // Ignore history-sync failures in constrained environments like jsdom.
        }
    }
};

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
        research_mode: true,
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

const collectSuccessfulResults = async (items, target, limit, worker) => {
    const values = Array.isArray(items) ? items : [];
    const successTarget = Math.max(1, Number(target) || values.length || 1);
    const concurrency = Math.max(1, limit || 1);
    const results = [];
    let cursor = 0;

    const runNext = async () => {
        while (results.length < successTarget) {
            const current = cursor;
            if (current >= values.length) return;
            cursor += 1;

            const nextValue = await worker(values[current], current);
            if (nextValue) {
                results.push(nextValue);
            }
        }
    };

    await Promise.all(Array.from({ length: Math.min(concurrency, values.length) }, runNext));
    return results.slice(0, successTarget);
};

const renderInlineMarkup = (str, sources = []) => {
    const parts = String(str || "").split(/(\*\*[^*]+\*\*|`[^`]+`|\[\d+\])/g);
    return parts.map((part, index) => {
        if (/^\*\*[^*]+\*\*$/.test(part)) return <strong key={index}>{part.slice(2, -2)}</strong>;
        if (/^`[^`]+`$/.test(part)) return <code key={index} className="ic">{part.slice(1, -1)}</code>;
        if (/^\[\d+\]$/.test(part)) {
            const source = findSourceForCitation(part.slice(1, -1), sources);
            const label = part.slice(1, -1);
            if (!source?.url) {
                return <span key={index} className="cite">{label}</span>;
            }
            return (
                <a key={index} href={source.url} target="_blank" rel="noopener noreferrer" className="cite">
                    {label}
                </a>
            );
        }
        return part;
    });
};

const extractHighlightPoints = (text = "", limit = 4) => {
    return String(text || "")
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => /^[-*]\s+/.test(line))
        .map((line) => line.replace(/^[-*]\s+/, "").trim())
        .filter(Boolean)
        .slice(0, limit);
};

const buildResultSummary = (text = "") => {
    const cleaned = String(text || "")
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => line && !/^[-*]\s+/.test(line) && !/^#{1,6}\s+/.test(line));
    const summary = cleaned.join(" ");
    if (summary.length <= 560) return summary;
    return `${summary.slice(0, 557).trimEnd()}...`;
};

const stripRepeatedHeading = (text = "", heading = "") => {
    const lines = String(text || "").split("\n");
    const normalizedHeading = String(heading || "").replace(/^#+\s*/, "").trim().toLowerCase();

    while (lines.length) {
        const firstLine = String(lines[0] || "").trim();
        const normalizedFirstLine = firstLine.replace(/^#+\s*/, "").trim().toLowerCase();
        if (!normalizedFirstLine) {
            lines.shift();
            continue;
        }
        if (
            normalizedFirstLine === normalizedHeading
            || normalizedFirstLine === "research answer"
            || normalizedFirstLine === "answer"
        ) {
            lines.shift();
            continue;
        }
        break;
    }

    return lines.join("\n").trim();
};

const collectSearchProviderKeys = (getApiKey) => {
    if (typeof getApiKey !== "function") return {};

    const configured = {
        tavily: getApiKey("tavily"),
        serper: getApiKey("serper"),
        brave: getApiKey("brave"),
        jina: getApiKey("jina"),
    };

    return Object.fromEntries(
        Object.entries(configured).filter(([, value]) => String(value || "").trim()),
    );
};

const DEFAULT_RESEARCH_MODEL = "nvidia/nemotron-3-super-120b-a12b:free";
const OPENROUTER_ROUND_ROBIN_PRESET = "openrouter-round-robin";

const collectResearchProviderKeys = (getApiKey) => {
    if (typeof getApiKey !== "function") return {};

    const configured = {
        openrouter: getApiKey("openrouter"),
        google: getApiKey("google") || getApiKey("gemini"),
    };

    return Object.fromEntries(
        Object.entries(configured).filter(([, value]) => String(value || "").trim()),
    );
};

const resolveResearchModelSelection = (researchSelectedModel = "") => {
    if (String(researchSelectedModel || "").trim() === OPENROUTER_ROUND_ROBIN_PRESET) {
        return {
            researchProvider: "openrouter",
            researchModel: DEFAULT_RESEARCH_MODEL,
            researchModelChain: [],
        };
    }

    const entries = String(researchSelectedModel || "")
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean);
    const chain = entries.length ? [...new Set(entries)] : [DEFAULT_RESEARCH_MODEL];
    const model = chain[0] || DEFAULT_RESEARCH_MODEL;
    const normalizedModel = model.toLowerCase();
    const provider = normalizedModel.startsWith("gemini-") ? "google" : "openrouter";

    return {
        researchProvider: provider,
        researchModel: model,
        researchModelChain: chain,
    };
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

const getSourceSignals = (source = {}) => {
    const signals = [];
    const providerCount = Number(source?.providerCount || 0);
    const queryHitCount = Number(source?.queryHitCount || 0);

    if (providerCount > 1) {
        signals.push(`${providerCount} providers`);
    }
    if (queryHitCount > 1) {
        signals.push(`${queryHitCount} query variants`);
    }

    return signals;
};

const buildCoverageRows = (query, researchMeta = {}, sources = []) => {
    const normalizedQuery = String(query || "").trim();
    const operatorMode = /\b(site:|filetype:|intitle:|inurl:|after:|before:)\b/i.test(normalizedQuery);
    const attachmentCount = Number(researchMeta?.attachments || 0);
    const queryCount = Number(researchMeta?.searchCount || 0);
    const rankedSites = Number(researchMeta?.rankedSites || 0);
    const fetchedSites = Number(researchMeta?.fetchedSites || 0);
    const fetchPlanned = Number(researchMeta?.fetchPlanned || 0);
    const fetchAttempts = Number(researchMeta?.fetchAttempts || fetchedSites);
    const synthesisWorkers = Number(researchMeta?.synthesisWorkers || 0);
    const crossCheckedSources = sources.filter((source) => Number(source?.providerCount || 0) > 1).length;
    const multiQuerySources = sources.filter((source) => Number(source?.queryHitCount || 0) > 1).length;
    const subagents = Array.isArray(researchMeta?.subagents) ? researchMeta.subagents.filter(Boolean) : [];
    const dedicatedSubagentCount = countSubagentAssignments(subagents);
    const dagSummary = String(researchMeta?.dagSummary || summarizeDag(researchMeta?.dag || {})).trim();
    const configuredProviders = Array.isArray(researchMeta?.configuredProviders) ? researchMeta.configuredProviders.filter(Boolean) : [];
    const providersUsed = Array.isArray(researchMeta?.providersUsed) ? researchMeta.providersUsed.filter(Boolean) : [];
    const providerErrors = Array.isArray(researchMeta?.providerErrors) ? researchMeta.providerErrors.filter(Boolean) : [];
    const ambiguousAxes = Array.isArray(researchMeta?.intentConfidence?.ambiguousAxes)
        ? researchMeta.intentConfidence.ambiguousAxes.filter(Boolean)
        : [];
    const tribunal = researchMeta?.tribunal || null;
    const convergence = researchMeta?.convergence || null;
    const rows = [
        ...(researchMeta?.frameworkVersion
            ? [{
                label: "Framework",
                spec: `Research Framework v${researchMeta.frameworkVersion} with compiled DAG orchestration`,
            }]
            : []),
        {
            label: "Search mode",
            spec: normalizedQuery
                ? (operatorMode ? "Operator-guided web research with targeted query expansion" : "Natural-language web research with query expansion")
                : "Attachment-only analysis",
        },
        ...((researchMeta?.domain?.label || researchMeta?.scope?.label || researchMeta?.outputMode?.label)
            ? [{
                label: "Intent decomposition",
                spec: `Domain: ${researchMeta?.domain?.label || "General Research"}; scope: ${researchMeta?.scope?.label || "Broad Research"}; output: ${researchMeta?.outputMode?.label || "State-of-the-Field"}${ambiguousAxes.length ? `; low-confidence axes: ${ambiguousAxes.join(", ")}` : ""}`,
            }]
            : []),
        ...(researchMeta?.pareto?.mode
            ? [{
                label: "Pareto profile",
                spec: `${researchMeta.pareto.mode} mode${researchMeta?.pareto?.explanation ? `; ${researchMeta.pareto.explanation}` : ""}`,
            }]
            : []),
        ...(researchMeta?.continuity?.active
            ? [{
                label: "Session continuity",
                spec: `Linked prior session context into Phase 1 with ${Math.round((researchMeta.continuity.overlap || 0) * 100)}% overlap`,
            }]
            : []),
        ...(dagSummary
            ? [{
                label: "Pipeline DAG",
                spec: dagSummary,
            }]
            : []),
        {
            label: "Query workers",
            spec: `${Math.max(queryCount, normalizedQuery ? 1 : 0)} search path${Math.max(queryCount, normalizedQuery ? 1 : 0) === 1 ? "" : "s"} executed`,
        },
        {
            label: "Sites ranked",
            spec: `${Math.max(rankedSites, sources.length)} unique site${Math.max(rankedSites, sources.length) === 1 ? "" : "s"} kept after dedupe`,
        },
        ...((configuredProviders.length || providersUsed.length || providerErrors.length)
            ? [{
                label: "Provider status",
                spec: providersUsed.length
                    ? `${providersUsed.length} provider${providersUsed.length === 1 ? "" : "s"} returned usable results${configuredProviders.length ? `; configured: ${configuredProviders.join(", ")}` : ""}${providerErrors.length ? `; ${providerErrors.length} provider issue${providerErrors.length === 1 ? "" : "s"}: ${providerErrors.slice(0, 2).join(" | ")}` : ""}`
                    : configuredProviders.length
                        ? `Configured for this run: ${configuredProviders.join(", ")}${providerErrors.length ? `; ${providerErrors.length} provider issue${providerErrors.length === 1 ? "" : "s"}: ${providerErrors.slice(0, 3).join(" | ")}` : ""}`
                        : `${providerErrors.length} provider issue${providerErrors.length === 1 ? "" : "s"}: ${providerErrors.slice(0, 3).join(" | ")}`,
            }]
            : []),
        ...(fetchPlanned
            ? [{
                label: "Fetch plan",
                spec: `${fetchPlanned} candidate page${fetchPlanned === 1 ? "" : "s"} selected with domain diversity before fetch`,
            }]
            : []),
        ...(researchMeta?.ragLexical
            ? [{
                label: "Retrieval (RAG)",
                spec: "Sources re-ranked by query terms; page excerpts are query-focused before synthesis",
            }]
            : []),
        {
            label: "Readable pages",
            spec: `${fetchedSites} page${fetchedSites === 1 ? "" : "s"} kept after ${Math.max(fetchAttempts, fetchedSites)} fetch attempt${Math.max(fetchAttempts, fetchedSites) === 1 ? "" : "s"}`,
        },
        {
            label: "Cited sources",
            spec: `${sources.length} source${sources.length === 1 ? "" : "s"} carried into the final answer`,
        },
        ...((crossCheckedSources || multiQuerySources)
            ? [{
                label: "Consensus signals",
                spec: `${crossCheckedSources} source${crossCheckedSources === 1 ? "" : "s"} matched across providers; ${multiQuerySources} source${multiQuerySources === 1 ? "" : "s"} returned across multiple query variants`,
            }]
            : []),
        {
            label: "Synthesis",
            spec: `${Math.max(synthesisWorkers, 1)} synthesis worker${Math.max(synthesisWorkers, 1) === 1 ? "" : "s"} used across position mapping, debate, and narrative compilation`,
        },
    ];

    if (attachmentCount) {
        rows.push({
            label: "Attachments",
            spec: `${attachmentCount} uploaded file${attachmentCount === 1 ? "" : "s"} used as extra evidence`,
        });
    }

    if (subagents.length) {
        rows.push({
            label: "Dedicated subagents",
            spec: `${dedicatedSubagentCount} dedicated subagent${dedicatedSubagentCount === 1 ? "" : "s"}: ${summarizeSubagents(subagents)}`,
        });
    }

    if (researchMeta?.safety?.activeCount) {
        rows.push({
            label: "Safety & ethics",
            spec: `${researchMeta.safety.activeCount} active check${researchMeta.safety.activeCount === 1 ? "" : "s"} spanning dual-use, funding conflicts, predatory journals, and statistical manipulation`,
        });
    }

    if (tribunal) {
        rows.push({
            label: "Quality tribunal",
            spec: `${tribunal.refinement_cycles}/${tribunal.refinement_budget} cycle${tribunal.refinement_cycles === 1 ? "" : "s"}; targeted ${formatDimensionLabel(tribunal.targeted_dimension)}; critics: consistency ${tribunal.critics?.internal_consistency ?? "n/a"}, coverage ${tribunal.critics?.coverage ?? "n/a"}, alignment ${tribunal.critics?.user_goal_alignment ?? "n/a"}`,
        });
    }

    if (convergence) {
        rows.push({
            label: "Convergence",
            spec: `stability ${convergence.stability_score}; coverage delta ${convergence.evidence_coverage_delta}; residual uncertainty ${convergence.residual_uncertainty}; stop condition ${formatDimensionLabel(convergence.stop_condition)}`,
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

const STEPS = PLANNING_STEPS;

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
                    <div className="scard__sub">Compiling a dynamic research DAG</div>
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
                    <span className="pill__n">{getDisplaySourceNumber(source, index)}</span>
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
                            <td className="col-num">{getDisplaySourceNumber(source, index)}</td>
                            <td className="col-source">
                                <div className="la-source-name">{source.title || getDomain(source.url)}</div>
                                <div className="la-source-ref">{getDomain(source.url)}</div>
                                {getSourceSignals(source).length ? (
                                    <div className="la-source-ref">{getSourceSignals(source).join(" • ")}</div>
                                ) : null}
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

function SubagentOwnershipTable({ researchMeta = {} }) {
    const rows = (Array.isArray(researchMeta?.subagents) ? researchMeta.subagents : [])
        .filter((entry) => entry?.id && entry?.label && entry?.scope)
        .map((entry) => ({
            key: String(entry.id),
            label: SUBAGENT_STAGE_LABELS[entry.id] || entry.label,
            owner: entry.count > 1 ? `${entry.count}x ${entry.label}` : entry.label,
            focus: `${entry.scope}${entry.detail ? `; ${entry.detail}` : ""}`,
            equipment: (Array.isArray(entry.equipment) ? entry.equipment : []).filter(Boolean).slice(0, 6),
        }));
    if (!rows.length) return null;

    return (
        <LibertyTableCard
            icon="⚙"
            title="Subagent Ownership"
            badge={`${rows.length} owners`}
            note="Every micro-stage is owned by a dedicated subagent. Equipment shows the tools, memories, and artifacts each owner can use."
        >
            <div className="la-table-wrap">
                <table className="la-table la-table--subagents">
                    <thead>
                        <tr>
                            <th className="col-part">Part</th>
                            <th className="col-owner">Subagent</th>
                            <th className="col-focus">Focus</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows.map((row) => (
                            <tr key={row.key}>
                                <td className="col-part"><span className="la-criterion-label">{row.label}</span></td>
                                <td className="col-owner">
                                    <span className="subagent-owner__name">{row.owner}</span>
                                </td>
                                <td className="col-focus">
                                    <span className="subagent-owner__focus">{row.focus}</span>
                                    {row.equipment.length ? (
                                        <span className="subagent-owner__equipment">
                                            <span className="subagent-owner__equipment-label">Equipment</span>
                                            <span className="subagent-owner__equipment-items">
                                                {row.equipment.map((item) => (
                                                    <span key={`${row.key}-${item}`} className="subagent-owner__equipment-pill">{item}</span>
                                                ))}
                                            </span>
                                        </span>
                                    ) : null}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </LibertyTableCard>
    );
}

function LibertyResultCard({ query, heading, body, sources = [], searchCount = 0, researchMeta = {} }) {
    const visibleSources = sources.slice(0, 20);
    const cleanedBody = stripRepeatedHeading(body, heading);
    const summary = buildResultSummary(cleanedBody);
    const answerCopy = summary || cleanedBody;
    const topDomains = [...new Set(visibleSources.map((source) => getDomain(source.url)).filter(Boolean))];
    const modeLabel = researchMeta?.outputMode?.label || "State-of-the-Field";
    const paretoLabel = researchMeta?.pareto?.mode || "balanced";
    const convergenceLabel = researchMeta?.convergence?.stability_score
        ? `Stability ${researchMeta.convergence.stability_score}`
        : null;
    const frameworkLabel = researchMeta?.frameworkVersion ? `Framework v${researchMeta.frameworkVersion}` : null;
    const noGroundedSources = researchMeta?.sourceSelection?.mode === "no_grounded_sources"
        || (!visibleSources.length && Array.isArray(researchMeta?.providerErrors) && researchMeta.providerErrors.length);

    return (
        <div className="la-result">
            <div className="la-header">
                <div className="la-header-copy">
                    <span className="la-kicker">Research answer</span>
                    <div className="la-summary-title">{renderInlineMarkup(heading, sources)}</div>
                </div>
                <div className="la-header-badges">
                    <span className="la-badge">{modeLabel}</span>
                    {frameworkLabel ? <span className="la-badge">{frameworkLabel}</span> : null}
                    {convergenceLabel ? <span className="la-badge">{convergenceLabel}</span> : null}
                    <span className="la-badge">{sources.length} source{sources.length === 1 ? "" : "s"}</span>
                </div>
            </div>

            <div className="la-body">
                {query ? (
                    <div className="la-question-block">
                        <div className="la-query">Question</div>
                        <div className="la-question">{renderInlineMarkup(query, sources)}</div>
                    </div>
                ) : null}

                <div className="la-meta-row">
                    <span className="la-meta-pill">Mode: {modeLabel}</span>
                    <span className="la-meta-pill">Searches: {Math.max(searchCount, 1)}</span>
                    <span className="la-meta-pill">Depth: {paretoLabel}</span>
                    {convergenceLabel ? <span className="la-meta-pill">{convergenceLabel}</span> : null}
                </div>

                {noGroundedSources ? (
                    <div className="la-warning">
                        No grounded sources were retrieved for this run. Open <strong>Research process</strong> for provider details and next steps.
                    </div>
                ) : null}

                {answerCopy ? (
                    <>
                        <div className="la-summary-label">Answer</div>
                        <div className="la-summary-text">{renderInlineMarkup(answerCopy, sources)}</div>
                    </>
                ) : null}

                {visibleSources.length ? (
                    <div className="la-tables">
                        <LibertySourceTable sources={visibleSources} />
                    </div>
                ) : null}

                {(Object.keys(researchMeta || {}).length || (Array.isArray(researchMeta?.subagents) && researchMeta.subagents.length)) ? (
                    <details className="la-process-details">
                        <summary className="la-process-summary">
                            <span>Research process</span>
                            <span className="la-process-meta">
                                {Math.max(searchCount, 1)} search{Math.max(searchCount, 1) === 1 ? "" : "es"}
                                {sources.length ? ` • ${sources.length} source${sources.length === 1 ? "" : "s"}` : ""}
                            </span>
                        </summary>
                        <div className="la-tables la-tables--secondary">
                            <LibertyCoverageTable query={query} researchMeta={{ ...researchMeta, searchCount }} sources={visibleSources} />
                            <SubagentOwnershipTable researchMeta={researchMeta} />
                        </div>
                    </details>
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

function EditMsg({ message, onSave, onCancel }) {
    const [editText, setEditText] = useState(message.text);

    const handleSave = () => {
        onSave(message.id, editText);
    };

    return (
        <div className="umsg umsg--editing">
            <div className="umsg__meta-row">
                <div className="umsg__label">You</div>
                <div className="umsg__label umsg__label--muted">Editing</div>
            </div>
            <div className="umsg__bubble umsg__bubble--editing">
                <textarea
                    className="edit-msg__textarea"
                    value={editText}
                    onChange={(e) => setEditText(e.target.value)}
                    autoFocus
                />
                <div className="edit-msg__actions">
                    <button className="edit-msg__btn" onClick={handleSave}>Save</button>
                    <button className="edit-msg__btn" onClick={onCancel}>Cancel</button>
                </div>
            </div>
        </div>
    );
}

function UserMsg({ text, attachments = [], onEdit }) {
    const [isCopied, setIsCopied] = useState(false);

    const handleCopy = () => {
        if (navigator.clipboard && text) {
            navigator.clipboard.writeText(text).then(() => {
                setIsCopied(true);
                setTimeout(() => setIsCopied(false), 2000);
            });
        }
    };

    return (
        <div className="umsg">
            <div className="umsg__meta-row">
                <div className="umsg__label">You</div>
                <div className="umsg__label umsg__label--muted">
                    {attachments.length ? `${attachments.length} attachment${attachments.length === 1 ? "" : "s"}` : "Research prompt"}
                </div>
            </div>
            <div className="umsg__bubble">
                {text ? <div className="umsg__text">{text}</div> : null}
                <AttachmentList attachments={attachments} />
            </div>
            <div className="umsg__acts">
                {typeof onEdit === "function" ? <button className="act-btn" title="Edit" onClick={onEdit}>✏</button> : null}
                <button className="act-btn" title="Copy" onClick={handleCopy}>
                    {isCopied ? "Copied!" : "⧉"}
                </button>
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

function Landing({ onChooseExample }) {
    const examples = [
        {
            label: "Write & edit",
            query: "Help me draft a professional email",
        },
        {
            label: "Code",
            query: "Debug my Python function",
        },
        {
            label: "Analyze",
            query: "Summarize this document for me",
        },
        {
            label: "Research",
            query: "Explain quantum entanglement simply",
        },
    ];

    return (
        <div className="land">
            <h1 className="land__h1">How can I help you?</h1>
            <p className="land__sub">Ask me anything. I can write, analyze, code, research, and keep the source trail attached when it matters.</p>
            <div className="land__exs">
                {examples.map((example) => (
                    <button key={example.query} className="land__ex" onClick={() => onChooseExample(example.query)}>
                        <span className="land__ex-label">{example.label}</span>
                        <span className="land__ex-query">{example.query}</span>
                    </button>
                ))}
            </div>
        </div>
    );
}

export default function SearchEngine({ session = null, resetSignal = 0, onSessionLoaded, onSessionRouteChange } = {}) {
    const [sessions, setSessions] = useState([]);
    const [activeId, setActiveId] = useState(null);
    const [streaming, setStreaming] = useState(false);
    const [input, setInput] = useState("");
    const [pendingUploads, setPendingUploads] = useState([]);
    const [uploadStatus, setUploadStatus] = useState("");
    const [activeResearchRunId, setActiveResearchRunId] = useState("");
    const [steeringStatus, setSteeringStatus] = useState("");
    const [steeringBusy, setSteeringBusy] = useState(false);
    const [steeringArea, setSteeringArea] = useState("");
    const [steeringMode, setSteeringMode] = useState("gap_analysis");
    const [excludedPreprints, setExcludedPreprints] = useState(false);

    const abortRef = useRef(null);
    const bottomRef = useRef(null);
    const inputRef = useRef(null);
    const fileInputRef = useRef(null);
    const uploadStatusTimerRef = useRef(null);
    const steeringStatusTimerRef = useRef(null);
    const resetSignalRef = useRef(resetSignal);
    const sessionsRef = useRef(sessions);
    const getApiKey = useSettingsStore((state) => state.getApiKey);
    const researchSelectedModel = useSettingsStore((state) => state.researchSelectedModel || DEFAULT_RESEARCH_MODEL);

    const active = sessions.find((session) => session.id === activeId);
    const isLanding = !active && !streaming;

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [sessions, streaming]);

    useEffect(() => {
        const node = inputRef.current;
        if (!node) return;
        node.style.height = "0px";
        node.style.height = `${Math.min(node.scrollHeight, 240)}px`;
    }, [input, isLanding]);

    useEffect(() => {
        if (streaming) return undefined;
        const timer = setTimeout(() => {
            inputRef.current?.focus();
        }, 50);
        return () => clearTimeout(timer);
    }, [activeId, isLanding, streaming]);

    useEffect(() => {
        sessionsRef.current = sessions;
    }, [sessions]);

    const activateStoredSession = useCallback((savedSession) => {
        if (!savedSession?.id) return;

        abortRef.current?.abort();
        abortRef.current = null;
        const loadedSession = hydrateSavedSession(savedSession);
        const shareUrl = buildSessionShareUrl(loadedSession.id);
        setStreaming(false);
        setSessions((previous) => {
            const nextSessions = [
                ...previous.filter((current) => current.id !== loadedSession.id),
                loadedSession,
            ];
            sessionsRef.current = nextSessions;
            return nextSessions;
        });
        setActiveId(loadedSession.id);
        setInput("");
        setPendingUploads([]);
        setUploadStatus("");
        setSteeringStatus("");
        setSteeringArea("");
        setSteeringMode("gap_analysis");
        setExcludedPreprints(false);
        setActiveResearchRunId("");
        if (shareUrl) {
            try {
                window.history.replaceState({}, "", shareUrl);
            } catch {
                // Let the shell route callback own the canonical URL when history updates are unavailable.
            }
        }
        onSessionRouteChange?.(loadedSession.id, shareUrl || window.location.href);
    }, [onSessionRouteChange]);

    const resetComposer = useCallback(() => {
        abortRef.current?.abort();
        abortRef.current = null;
        setStreaming(false);
        setActiveId(null);
        setInput("");
        setPendingUploads([]);
        setUploadStatus("");
        setSteeringStatus("");
        setSteeringArea("");
        setSteeringMode("gap_analysis");
        setExcludedPreprints(false);
        setActiveResearchRunId("");
        clearSharedSessionUrl();
    }, []);

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

    const setTransientSteeringStatus = useCallback((value) => {
        if (steeringStatusTimerRef.current) {
            clearTimeout(steeringStatusTimerRef.current);
            steeringStatusTimerRef.current = null;
        }
        setSteeringStatus(value);
        if (value) {
            steeringStatusTimerRef.current = setTimeout(() => {
                setSteeringStatus("");
                steeringStatusTimerRef.current = null;
            }, 3600);
        }
    }, []);

    useEffect(() => () => {
        if (uploadStatusTimerRef.current) clearTimeout(uploadStatusTimerRef.current);
        if (steeringStatusTimerRef.current) clearTimeout(steeringStatusTimerRef.current);
        abortRef.current?.abort();
        abortRef.current = null;
    }, []);

    useEffect(() => {
        if (resetSignalRef.current === resetSignal) return;
        resetSignalRef.current = resetSignal;

        resetComposer();
    }, [resetComposer, resetSignal]);

    useEffect(() => {
        if (!session?.id) return;
        activateStoredSession(session);
        onSessionLoaded?.();
    }, [activateStoredSession, onSessionLoaded, session]);

    const patchLastBot = useCallback((sessionId, patch) => {
        setSessions((previous) => {
            const nextSessions = previous.map((session) => {
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
            });
            sessionsRef.current = nextSessions;
            return nextSessions;
        });
    }, []);

    const revealAnswer = useCallback(async (sessionId, sessionQuery, finalText, sources, signal, extraPatch = {}) => {
        const { heading, body } = extractAnswerParts(finalText);
        const currentSession = sessionsRef.current.find((item) => item.id === sessionId);
        const botMessages = Array.isArray(currentSession?.messages)
            ? currentSession.messages.filter((message) => message.role === "bot")
            : [];
        const existingSources = Array.isArray(botMessages[botMessages.length - 1]?.sources)
            ? botMessages[botMessages.length - 1].sources
            : [];
        const resolvedSources = Array.isArray(sources) && sources.length ? sources : existingSources;
        patchLastBot(sessionId, {
            heading,
            body: "",
            sources: resolvedSources,
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
                sources: resolvedSources,
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
            sources: resolvedSources,
            showPlanning: false,
            showSearching: false,
            searchDone: true,
            ...extraPatch,
        });

        // Auto-save session to library
        const savedSession = sessionsRef.current.find((item) => item.id === sessionId);
        const researchMeta = extraPatch?.researchMeta || {};
        const savedAttachments = savedSession?.messages
            ?.find((message) => message.role === "user")
            ?.attachments || [];
        saveSession(buildSavedSessionRecord({
            id: sessionId,
            query: savedSession?.query || sessionQuery || heading,
            heading,
            body: body || heading,
            sources: resolvedSources,
            attachments: savedAttachments,
            researchMeta,
        }));
    }, [patchLastBot]);

    const queueSteeringCommand = useCallback(async (control) => {
        if (!streaming || !activeResearchRunId || !control?.type) return;
        setSteeringBusy(true);

        try {
            await sendResearchControl(activeResearchRunId, control);
            if (control.type === "exclude_source" && normalizeTextBlock(control.sourceType || control.typeId).toLowerCase() === "preprint") {
                setExcludedPreprints(true);
            }
            if (control.type === "increase_depth") {
                setSteeringArea("");
            }
            setTransientSteeringStatus(`Queued ${control.type.replace(/_/g, " ")}.`);
        } catch (error) {
            setTransientSteeringStatus(error?.message || "Unable to queue steering control.");
        } finally {
            setSteeringBusy(false);
        }
    }, [activeResearchRunId, setTransientSteeringStatus, streaming]);

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
        const searchQuery = prepareSearchQuery(query);
        const attachments = buildSessionAttachments(uploadsOverride || []);
        if ((!query && !attachments.length) || streaming) return;

        const sessionId = createId();
        const depthPreference = inferDepthPreference(searchQuery || query);
        const compiledPlan = compileResearchPlan({
            query: searchQuery,
            attachments: attachments.length,
            savedSessions: getSavedSessions(),
            currentSessionId: sessionId,
            maxQueries: SEARCH_SWARM_SIZE,
            depthPreference,
        });
        const swarmQueries = query
            ? (compiledPlan.searchQueries.length ? compiledPlan.searchQueries : [searchQuery])
            : attachments.map((attachment) => attachment.name).slice(0, 4);
        const displayQuery = query || `Analyze ${attachments.length} attached file${attachments.length > 1 ? "s" : ""}`;
        const userText = query || getAttachmentAnalysisPrompt("", attachments);
        const activityTitle = query ? "Searching the web" : "Inspecting uploads";
        const activityDoneTitle = query ? "Searched the web" : "Inspected uploads";
        const activityIcon = query ? "🌐" : "📎";
        const researchMetaBase = {
            attachments: attachments.length,
            frameworkVersion: compiledPlan.frameworkVersion,
            domain: compiledPlan.domain,
            scope: compiledPlan.scope,
            outputMode: compiledPlan.outputMode,
            intentConfidence: compiledPlan.intentConfidence,
            pareto: compiledPlan.pareto,
            continuity: compiledPlan.continuity,
            safety: compiledPlan.safety,
            dag: compiledPlan.dag,
            dagSummary: summarizeDag(compiledPlan.dag),
            queryMatrix: compiledPlan.queryMatrix,
            refinementBudget: compiledPlan.refinementBudget,
            subagents: compiledPlan.subagents,
        };

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
                    showPlanning: Boolean(query),
                    showSearching: true,
                    queries: swarmQueries,
                    searchDone: false,
                    statusText: query
                        ? formatSubagentStatus("cognitiveCommandLayer", `compiled a ${compiledPlan.scope.label.toLowerCase()} DAG with ${swarmQueries.length} search lane${swarmQueries.length === 1 ? "" : "s"} and ${compiledPlan.refinementBudget} tribunal cycle${compiledPlan.refinementBudget === 1 ? "" : "s"} budgeted...`)
                        : formatSubagentStatus("statisticalClaimExtractor", `reading ${attachments.length} uploaded file${attachments.length > 1 ? "s" : ""}...`),
                    activityTitle,
                    activityDoneTitle,
                    activityIcon,
                },
            ],
        };

        setSessions((previous) => {
            const nextSessions = [...previous, newSession];
            sessionsRef.current = nextSessions;
            return nextSessions;
        });
        setActiveId(sessionId);
        setStreaming(true);
        setInput("");
        setPendingUploads([]);
        setUploadStatus("");
        setActiveResearchRunId("");
        setSteeringStatus("");
        setSteeringArea("");
        setSteeringMode("gap_analysis");
        setExcludedPreprints(false);
        saveSession(buildSavedSessionRecord({
            id: sessionId,
            query: displayQuery,
            body: query ? "Research in progress..." : "Attachment analysis in progress...",
            attachments,
            researchMeta: {
                generatedAt: new Date().toISOString(),
                status: "in_progress",
                searchCount: swarmQueries.length,
                ...researchMetaBase,
            },
        }));
        const shareUrl = buildSessionShareUrl(sessionId);
        if (shareUrl) {
            try {
                window.history.replaceState({}, "", shareUrl);
            } catch {
                // Let the shell route callback own the canonical URL when history updates are unavailable.
            }
        }
        onSessionRouteChange?.(sessionId, shareUrl || window.location.href);

        abortRef.current = new AbortController();
        const { signal } = abortRef.current;
        let runtimeRunId = "";
        let terminalSaved = false;
        let terminalMessage = "";

        try {
            const searchProviderKeys = collectSearchProviderKeys(getApiKey);
            const researchProviderKeys = collectResearchProviderKeys(getApiKey);
            const researchModelSelection = resolveResearchModelSelection(researchSelectedModel);
            const streamResearchEvent = (event) => {
                if (!event || typeof event !== "object") return;
                if (event.runId) {
                    runtimeRunId = String(event.runId);
                    setActiveResearchRunId(runtimeRunId);
                }

                if (event.type === "status") {
                    patchLastBot(sessionId, (current) => ({
                        statusText: event.detail || current.statusText,
                        researchMeta: {
                            ...(current.researchMeta || {}),
                            ...(event.researchMeta || {}),
                        },
                    }));
                    return;
                }

                if (event.type === "inventory") {
                    patchLastBot(sessionId, (current) => ({
                        statusText: formatSubagentStatus(
                            "tieredEpistemicFilter",
                            `retained ${event.counts?.core || 0} core, ${event.counts?.supporting || 0} supporting, and ${event.counts?.peripheral || 0} peripheral sources.`,
                        ),
                        showPlanning: false,
                        sources: Array.isArray(event.sources) && event.sources.length ? event.sources : current.sources,
                        researchMeta: {
                            ...(current.researchMeta || {}),
                            ...(event.researchMeta || {}),
                        },
                    }));
                    return;
                }

                if (event.type === "summary") {
                    patchLastBot(sessionId, (current) => ({
                        statusText: formatSubagentStatus(
                            "deepComprehensionEngine",
                            `extracting evidence from ${event.source?.title || "selected sources"}...`,
                        ),
                        sources: event.source?.url
                            ? [...current.sources, event.source].filter((source, index, array) => (
                                array.findIndex((item) => item?.url === source?.url) === index
                            ))
                            : current.sources,
                        researchMeta: {
                            ...(current.researchMeta || {}),
                            ...(event.researchMeta || {}),
                        },
                    }));
                    return;
                }

                if (event.type === "control_applied") {
                    setTransientSteeringStatus(event.statusText || "Steering command applied.");
                    patchLastBot(sessionId, (current) => ({
                        statusText: event.statusText || current.statusText,
                        researchMeta: {
                            ...(current.researchMeta || {}),
                            ...(event.researchMeta || {}),
                        },
                    }));
                    return;
                }

                if (event.type === "warning") {
                    patchLastBot(sessionId, (current) => ({
                        statusText: event.detail || current.statusText,
                        researchMeta: {
                            ...(current.researchMeta || {}),
                            ...(event.researchMeta || {}),
                        },
                    }));
                    return;
                }

                if (event.type !== "checkpoint") return;
                const checkpointId = String(event.checkpoint || "");
                const payload = event.payload || {};

                patchLastBot(sessionId, (current) => ({
                    statusText: checkpointId === "plan"
                        ? formatSubagentStatus("cognitiveCommandLayer", `compiled a ${compiledPlan.scope.label.toLowerCase()} DAG with ${swarmQueries.length} search lane${swarmQueries.length === 1 ? "" : "s"} and ${compiledPlan.refinementBudget} tribunal cycle${compiledPlan.refinementBudget === 1 ? "" : "s"} budgeted...`)
                        : checkpointId === "inventory"
                            ? formatSubagentStatus("tieredEpistemicFilter", "ranked the source mesh into core, supporting, and peripheral evidence.")
                            : checkpointId === "summaries"
                                ? formatSubagentStatus("deepComprehensionEngine", "completed extraction across fetched evidence, supplementary material, and linked repos.")
                                : checkpointId === "draft"
                                    ? formatSubagentStatus("dialecticalSynthesisEngine", "completed position mapping, thesis/antithesis debate, and synthesis mediation.")
                                    : checkpointId === "tribunal"
                                        ? formatSubagentStatus("internalConsistencyCritic", "completed verifier swarm scoring, critic review, and convergence checks.")
                                    : checkpointId === "decision"
                                        ? formatSubagentStatus("decisionIntelligenceLayer", "packaged the decision payload, risk profile, and reversibility guidance.")
                                    : checkpointId === "final"
                                        ? formatSubagentStatus("adaptiveDeliveryHub", "packaged the final report, exports, and postmortem artifacts.")
                                        : current.statusText,
                    showPlanning: current.showPlanning && checkpointId === "plan",
                    queries: Array.isArray(payload.searchQueries) && payload.searchQueries.length ? payload.searchQueries : current.queries,
                    sources: Array.isArray(payload.sources) && payload.sources.length
                        ? payload.sources
                        : current.sources,
                    researchMeta: {
                        ...(current.researchMeta || {}),
                        ...(event.researchMeta || {}),
                    },
                }));
            };

            const runtimeResult = await invokeResearchRuntime({
                action: "run",
                query: searchQuery,
                attachments,
                stream: true,
                responseType: "sse",
                depthPreference,
                forcedOutputMode: compiledPlan.outputMode.id,
                refinementBudget: compiledPlan.refinementBudget,
                maxQueries: SEARCH_SWARM_SIZE,
                researchProvider: researchModelSelection.researchProvider,
                researchModel: researchModelSelection.researchModel,
                ...(researchModelSelection.researchModelChain.length ? { researchModelChain: researchModelSelection.researchModelChain } : {}),
                researchRoundRobin: true,
                ...(Object.keys(searchProviderKeys).length ? { searchProviderKeys } : {}),
                ...(Object.keys(researchProviderKeys).length ? { researchProviderKeys } : {}),
            }, signal, streamResearchEvent);
            runtimeRunId = runtimeResult?.runId || runtimeRunId;

            const finalResult = runtimeResult?.final || {};
            const fullText = finalizeResearchAnswer(
                finalResult.markdown
                || `# ${finalResult.heading || "Research Answer"}\n\n${finalResult.body || ""}`,
            );
            const tribunal = finalResult.tribunal || null;
            const convergence = finalResult.convergence || null;
            const researchMeta = {
                generatedAt: new Date().toISOString(),
                ragLexical: true,
                tribunal: tribunal || undefined,
                convergence: convergence || undefined,
                sourceSelection: finalResult.sourceSelection || undefined,
                ...researchMetaBase,
                ...(runtimeResult?.researchMeta || {}),
            };

            await revealAnswer(
                sessionId,
                displayQuery,
                fullText,
                Array.isArray(finalResult.sources) ? finalResult.sources : [],
                signal,
                {
                    researchMeta,
                    statusText: formatSubagentStatus(
                        "adaptiveDeliveryHub",
                        `delivered ${(researchMeta.outputMode?.label || compiledPlan.outputMode.label).toLowerCase()}${tribunal?.refinement_cycles ? ` after ${tribunal.refinement_cycles} tribunal cycle${tribunal.refinement_cycles === 1 ? "" : "s"}` : ""}.`,
                    ),
                },
            );
            terminalSaved = true;
        } catch (error) {
            if (error.name === "AbortError") {
                terminalMessage = "Generation stopped.";
                patchLastBot(sessionId, {
                    heading: "Stopped",
                    body: terminalMessage,
                    showPlanning: false,
                    showSearching: false,
                    searchDone: true,
                });
                saveSession(buildSavedSessionRecord({
                    id: sessionId,
                    query: displayQuery,
                    heading: "Stopped",
                    body: terminalMessage,
                    attachments,
                    researchMeta: {
                        generatedAt: new Date().toISOString(),
                        status: "stopped",
                        runId: runtimeRunId || undefined,
                        ...researchMetaBase,
                    },
                }));
                terminalSaved = true;
            } else {
                terminalMessage = error.message || "Search failed. Try again.";
                patchLastBot(sessionId, {
                    heading: "Error",
                    body: terminalMessage,
                    showPlanning: false,
                    showSearching: false,
                    searchDone: true,
                });
                saveSession(buildSavedSessionRecord({
                    id: sessionId,
                    query: displayQuery,
                    heading: "Error",
                    body: terminalMessage,
                    attachments,
                    researchMeta: {
                        generatedAt: new Date().toISOString(),
                        status: "error",
                        runId: runtimeRunId || undefined,
                        ...researchMetaBase,
                    },
                }));
                terminalSaved = true;
            }
        } finally {
            if (!terminalSaved) {
                const fallbackMessage = terminalMessage || "Search failed before completion. Try again.";
                patchLastBot(sessionId, {
                    heading: "Error",
                    body: fallbackMessage,
                    showPlanning: false,
                    showSearching: false,
                    searchDone: true,
                });
                saveSession(buildSavedSessionRecord({
                    id: sessionId,
                    query: displayQuery,
                    heading: "Error",
                    body: fallbackMessage,
                    attachments,
                    researchMeta: {
                        generatedAt: new Date().toISOString(),
                        status: "error",
                        runId: runtimeRunId || undefined,
                        ...researchMetaBase,
                    },
                }));
            }
            setStreaming(false);
            setSteeringBusy(false);
            setActiveResearchRunId("");
            abortRef.current = null;
            setTimeout(() => inputRef.current?.focus(), 80);
        }
    }, [
        getApiKey,
        patchLastBot,
        pendingUploads,
        researchSelectedModel,
        revealAnswer,
        setTransientSteeringStatus,
        streaming,
    ]);

    const handleSubmit = (event) => {
        event?.preventDefault();
        if (input.trim() || pendingUploads.length) runSearch(input.trim(), pendingUploads);
    };

    const handleComposerKeyDown = (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            if (input.trim() || pendingUploads.length) runSearch(input.trim(), pendingUploads);
        }
    };

    const handleExampleSelect = (query) => {
        setInput(query);
        setTimeout(() => inputRef.current?.focus(), 40);
    };

    const handleShareCurrentSession = () => {
        const url = buildSessionShareUrl(active?.id) || window.location.href;
        if (!navigator.clipboard?.writeText) {
            promptToCopySessionUrl(url);
            return;
        }

        navigator.clipboard.writeText(url).then(() => {
            alert("Link copied to clipboard!");
        }).catch(() => {
            promptToCopySessionUrl(url);
        });
    };

    const cleanLatestBotMessage = active?.messages?.slice().reverse().find((message) => message.role === "bot") || null;
    const cleanLatestResearchMeta = cleanLatestBotMessage?.researchMeta || {};
    const cleanOutputModeLabel = cleanLatestResearchMeta?.outputMode?.label || "State-of-the-Field";
    const cleanSourceCount = cleanLatestBotMessage?.sources?.length || 0;
    const cleanStability = Number(cleanLatestResearchMeta?.convergence?.stability_score);
    const cleanStabilityLabel = Number.isFinite(cleanStability) ? `${Math.round(cleanStability * 100)}% stability` : "";
    const cleanDepthLabel = cleanLatestResearchMeta?.pareto?.mode ? `Depth: ${cleanLatestResearchMeta.pareto.mode}` : null;
    const hasComposerValue = Boolean(input.trim() || pendingUploads.length);

    return (
        <div className="research-view">
            <div className="research-view__scroll">
                <div className="research-view__inner">
                    {isLanding ? (
                        <Landing onChooseExample={handleExampleSelect} />
                    ) : (
                        <>
                            <div className="thread-head">
                                <div className="thread-head__copy">
                                    <div className="thread-head__eyebrow">{streaming ? "Research in progress" : "Research"}</div>
                                    <h1 className="thread-head__title">{active?.query || "Research session"}</h1>
                                    <div className="thread-head__meta">
                                        <span className="thread-head__pill">{cleanOutputModeLabel}</span>
                                        {cleanSourceCount ? <span className="thread-head__pill">{cleanSourceCount} source{cleanSourceCount === 1 ? "" : "s"}</span> : null}
                                        {cleanDepthLabel ? <span className="thread-head__pill">{cleanDepthLabel}</span> : null}
                                        {cleanStabilityLabel ? <span className="thread-head__pill">{cleanStabilityLabel}</span> : null}
                                    </div>
                                </div>
                                <button className="thread-head__action" onClick={handleShareCurrentSession}>Share</button>
                            </div>

                            <div className="chat chat--thread">
                                <div className="chat__in">
                                    {active?.messages.map((message, index) => (
                                        message.role === "user"
                                            ? <UserMsg key={message.id} text={message.text} attachments={message.attachments} />
                                            : <BotMsg key={message.id} msg={message} isLast={index === active.messages.length - 1} streaming={streaming} sessionQuery={active.query} />
                                    ))}
                                    <div ref={bottomRef} />
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </div>

            <div className="composer">
                <div className="composer__inner">
                    {(pendingUploads.length || uploadStatus) && (
                        <div className="composer__uploads">
                            <AttachmentList attachments={pendingUploads} onRemove={removePendingUpload} compact />
                            {uploadStatus ? <div className="upload-status">{uploadStatus}</div> : null}
                        </div>
                    )}

                    <form className="composer__box" onSubmit={handleSubmit}>
                        <textarea
                            ref={inputRef}
                            className="composer__textarea"
                            value={input}
                            onChange={(event) => setInput(event.target.value)}
                            onKeyDown={handleComposerKeyDown}
                            placeholder={pendingUploads.length ? "Ask about the uploaded files or continue research..." : "Message NubAgent"}
                            aria-label="Message NubAgent"
                            disabled={streaming}
                            rows={1}
                        />
                        <div className="composer__footer">
                            <div className="composer__tools">
                                <button type="button" className="composer__tool" onClick={openFilePicker} disabled={streaming}>Attach</button>
                                <span className="composer__tool composer__tool--static">{streaming ? "Research running" : "Search web"}</span>
                            </div>
                            {streaming ? (
                                <button type="button" className="composer__send composer__send--stop" onClick={() => abortRef.current?.abort()} title="Stop">■</button>
                            ) : (
                                <button type="submit" className="composer__send" disabled={!hasComposerValue} title="Send">↑</button>
                            )}
                        </div>
                    </form>

                    {streaming && (
                        <div className="composer__steering">
                            <div className="composer__steering-row">
                                <span className="composer__steering-label">Pace</span>
                                <button
                                    type="button"
                                    className="composer__chip"
                                    disabled={!activeResearchRunId || steeringBusy}
                                    onClick={() => queueSteeringCommand({ type: "prioritize_speed" })}
                                >
                                    Speed
                                </button>
                                <button
                                    type="button"
                                    className="composer__chip"
                                    disabled={!activeResearchRunId || steeringBusy}
                                    onClick={() => queueSteeringCommand({ type: "go_deeper" })}
                                >
                                    Go deeper
                                </button>
                                <button
                                    type="button"
                                    className={`composer__chip ${excludedPreprints ? "composer__chip--active" : ""}`}
                                    disabled={!activeResearchRunId || steeringBusy || excludedPreprints}
                                    onClick={() => queueSteeringCommand({ type: "exclude_source", sourceType: "preprint" })}
                                >
                                    Exclude preprints
                                </button>
                            </div>
                            <div className="composer__steering-row composer__steering-row--dense">
                                <span className="composer__steering-label">Output</span>
                                <select
                                    className="composer__select"
                                    value={steeringMode}
                                    disabled={!activeResearchRunId || steeringBusy}
                                    onChange={(event) => setSteeringMode(event.target.value)}
                                >
                                    <option value="state_of_the_field">State of the field</option>
                                    <option value="gap_analysis">Gap analysis</option>
                                    <option value="controversy_map">Controversy map</option>
                                    <option value="tutorial">Tutorial</option>
                                    <option value="decision_brief">Decision brief</option>
                                </select>
                                <button
                                    type="button"
                                    className="composer__chip"
                                    disabled={!activeResearchRunId || steeringBusy}
                                    onClick={() => queueSteeringCommand({ type: "force_mode", mode: steeringMode })}
                                >
                                    Apply mode
                                </button>
                                <input
                                    className="composer__focus"
                                    value={steeringArea}
                                    onChange={(event) => setSteeringArea(event.target.value)}
                                    placeholder="Deepen an area or hypothesis..."
                                    disabled={!activeResearchRunId || steeringBusy}
                                />
                                <button
                                    type="button"
                                    className="composer__chip"
                                    disabled={!activeResearchRunId || steeringBusy || !steeringArea.trim()}
                                    onClick={() => queueSteeringCommand({ type: "increase_depth", area: steeringArea.trim() })}
                                >
                                    Increase depth
                                </button>
                            </div>
                            <div className="composer__status">
                                {steeringStatus || (activeResearchRunId ? `Run ${String(activeResearchRunId).slice(-8)}` : "Waiting for runtime...")}
                            </div>
                        </div>
                    )}

                    <p className="composer__note">NubAgent can make mistakes. Verify important info.</p>
                </div>
            </div>

            <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".txt,.md,.markdown,.json,.csv,.js,.mjs,.cjs,.ts,.jsx,.tsx,.py,.rb,.go,.rs,.java,.c,.h,.cpp,.hpp,.html,.css,.scss,.sass,.xml,.yaml,.yml,.toml,.ini,.env,.log,text/*,application/json,image/*"
                style={{ display: "none" }}
                onChange={handleFileUpload}
            />
        </div>
    );
}
