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
import Library from "./Library.jsx";

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
    window.history.replaceState({}, "", url.toString());
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
        }));
    if (!rows.length) return null;

    return (
        <LibertyTableCard
            icon="⚙"
            title="Subagent Ownership"
            badge={`${rows.length} owners`}
            note="Every micro-stage is owned by a dedicated subagent for clear accountability."
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
    const summary = buildResultSummary(body);
    const points = extractHighlightPoints(body, 4);
    const topDomains = [...new Set(visibleSources.map((source) => getDomain(source.url)).filter(Boolean))];
    const modeLabel = researchMeta?.outputMode?.label || "State-of-the-Field";
    const paretoLabel = researchMeta?.pareto?.mode || "balanced";
    const convergenceLabel = researchMeta?.convergence?.stability_score
        ? `Stability ${researchMeta.convergence.stability_score}`
        : null;
    const frameworkLabel = researchMeta?.frameworkVersion ? `Framework v${researchMeta.frameworkVersion}` : null;

    return (
        <div className="la-result">
            <div className="la-header">
                <div className="la-header-main">
                    <div className="la-logo">✳</div>
                    <div className="la-header-copy">
                        <span className="la-brand">nub-agent</span>
                        <span className="la-kicker">Research synthesis</span>
                    </div>
                </div>
                <div className="la-header-badges">
                    {frameworkLabel ? <span className="la-badge">{frameworkLabel}</span> : null}
                    <span className="la-badge">{modeLabel}</span>
                    <span className="la-badge">{sources.length} Source{sources.length === 1 ? "" : "s"}</span>
                </div>
            </div>

            <div className="la-body">
                <div className="la-hero">
                    <div className="la-hero-copy">
                        <div className="la-query">Research Question</div>
                        <div className="la-question">{renderInlineMarkup(query || heading, sources)}</div>
                    </div>
                    <div className="la-hero-stats">
                        <div className="la-stat">
                            <span className="la-stat__label">Mode</span>
                            <strong>{modeLabel}</strong>
                        </div>
                        <div className="la-stat">
                            <span className="la-stat__label">Search Lanes</span>
                            <strong>{Math.max(searchCount, 1)}</strong>
                        </div>
                        <div className="la-stat">
                            <span className="la-stat__label">Pareto</span>
                            <strong>{paretoLabel}</strong>
                        </div>
                        {convergenceLabel ? (
                            <div className="la-stat">
                                <span className="la-stat__label">Convergence</span>
                                <strong>{convergenceLabel}</strong>
                            </div>
                        ) : null}
                    </div>
                </div>
                <div className="la-divider" />

                <div className="la-summary-label">Synthesized Answer</div>
                <div className="la-summary-title">{renderInlineMarkup(heading, sources)}</div>
                {summary ? <div className="la-summary-text">{renderInlineMarkup(summary, sources)}</div> : null}

                {topDomains.length ? (
                    <div className="la-domain-strip">
                        {topDomains.slice(0, 6).map((domain) => (
                            <span key={domain} className="la-domain-pill">{domain}</span>
                        ))}
                    </div>
                ) : null}

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
                        <SubagentOwnershipTable researchMeta={researchMeta} />
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

function EditMsg({ message, onSave, onCancel }) {
    const [editText, setEditText] = useState(message.text);

    const handleSave = () => {
        onSave(message.id, editText);
    };

    return (
        <div className="umsg">
            <div className="umsg__av"><span>U</span></div>
            <div className="umsg__body">
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
            <div className="umsg__av"><span>U</span></div>
            <div className="umsg__body umsg__card">
                <div className="umsg__head">
                    <div className="umsg__name">You</div>
                    <div className="umsg__meta">
                        {attachments.length ? `${attachments.length} attachment${attachments.length === 1 ? "" : "s"}` : "Research prompt"}
                    </div>
                </div>
                {text ? <div className="umsg__text">{text}</div> : null}
                <AttachmentList attachments={attachments} />
            </div>
            <div className="umsg__acts">
                <button className="act-btn" title="Edit" onClick={onEdit}>✏</button>
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

function Landing({ onSearch, uploads = [], onOpenUpload, onRemoveUpload, uploadStatus = "" }) {
    const [query, setQuery] = useState("");
    const inputRef = useRef(null);

    useEffect(() => { inputRef.current?.focus(); }, []);

    const submit = (event) => {
        event?.preventDefault();
        if (query.trim() || uploads.length) onSearch(query.trim());
    };

    const examples = [
        {
            label: "Threat analysis",
            detail: "Cross-check a fast-moving technical risk with cited evidence.",
            query: "How does quantum computing threaten modern encryption?",
        },
        {
            label: "Operator search",
            detail: "Use web operators to force a narrower evidence set.",
            query: 'site:arxiv.org "retrieval augmented generation" after:2024-01-01',
        },
        {
            label: "Market scan",
            detail: "Survey the current field and surface the strongest contenders.",
            query: "Best open source LLMs benchmark 2025",
        },
        {
            label: "Incident research",
            detail: "Investigate a vulnerability with prioritised source retrieval.",
            query: "intitle:CVE Apache Log4j critical vulnerability",
        },
    ];
    const capabilities = [
        "Compiled research DAG",
        "Verifier swarm and tribunal scoring",
        "Streaming checkpoints with live steering",
    ];

    return (
        <div className="land">
            <div className="land__eyebrow">Research Framework v3.1</div>
            <div className="land__logo"><span className="land__star">✳</span><span>nub-agent</span></div>
            <h1 className="land__h1">Research the web like an operator, not a chatbot.</h1>
            <p className="land__sub">Compiled search lanes, deep reading, verifier passes, and cited synthesis in one surface built for serious investigation.</p>
            <div className="land__capabilities">
                {capabilities.map((capability) => (
                    <span key={capability} className="land__cap">{capability}</span>
                ))}
            </div>
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
            <div className="land__panels">
                <div className="land__panel">
                    <div className="land__panel-head">
                        <div className="land__panel-eyebrow">Examples</div>
                        <div className="land__panel-title">Start from a strong brief</div>
                    </div>
                    <div className="land__exs">
                        {examples.map((example) => (
                            <button key={example.query} className="land__ex" onClick={() => onSearch(example.query)}>
                                <span className="land__ex-label">{example.label}</span>
                                <span className="land__ex-detail">{example.detail}</span>
                                <span className="land__ex-query">{example.query}</span>
                            </button>
                        ))}
                    </div>
                </div>
                <div className="land__panel land__panel--summary">
                    <div className="land__panel-head">
                        <div className="land__panel-eyebrow">Workflow</div>
                        <div className="land__panel-title">What the runtime does</div>
                    </div>
                    <div className="land__flow">
                        <div className="land__flow-step"><span>01</span><strong>Decompose intent into domain, scope, and output mode.</strong></div>
                        <div className="land__flow-step"><span>02</span><strong>Compile a dynamic search and evidence DAG.</strong></div>
                        <div className="land__flow-step"><span>03</span><strong>Synthesize, verify, and stream checkpoints as they complete.</strong></div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default function SearchEngine({ session = null, resetSignal = 0, onSessionLoaded } = {}) {
    const [sessions, setSessions] = useState([]);
    const [activeId, setActiveId] = useState(null);
    const [streaming, setStreaming] = useState(false);
    const [input, setInput] = useState("");
    const [pendingUploads, setPendingUploads] = useState([]);
    const [uploadStatus, setUploadStatus] = useState("");
    const [showLibrary, setShowLibrary] = useState(false);
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

    const active = sessions.find((session) => session.id === activeId);
    const isLanding = !active && !streaming;
    const isLibraryView = showLibrary;

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [sessions, streaming]);

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
        setShowLibrary(false);
        if (shareUrl) {
            window.history.replaceState({}, "", shareUrl);
        }
    }, []);

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
        setShowLibrary(false);
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
            sources,
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

        abortRef.current = new AbortController();
        const { signal } = abortRef.current;
        let runtimeRunId = "";

        try {
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
        } catch (error) {
            if (error.name === "AbortError") {
                patchLastBot(sessionId, {
                    heading: "Stopped",
                    body: "Generation stopped.",
                    showPlanning: false,
                    showSearching: false,
                    searchDone: true,
                });
                saveSession(buildSavedSessionRecord({
                    id: sessionId,
                    query: displayQuery,
                    heading: "Stopped",
                    body: "Generation stopped.",
                    attachments,
                    researchMeta: {
                        generatedAt: new Date().toISOString(),
                        status: "stopped",
                        runId: runtimeRunId || undefined,
                        ...researchMetaBase,
                    },
                }));
            } else {
                patchLastBot(sessionId, {
                    heading: "Error",
                    body: error.message || "Search failed. Try again.",
                    showPlanning: false,
                    showSearching: false,
                    searchDone: true,
                });
                saveSession(buildSavedSessionRecord({
                    id: sessionId,
                    query: displayQuery,
                    heading: "Error",
                    body: error.message || "Search failed. Try again.",
                    attachments,
                    researchMeta: {
                        generatedAt: new Date().toISOString(),
                        status: "error",
                        runId: runtimeRunId || undefined,
                        ...researchMetaBase,
                    },
                }));
            }
        } finally {
            setStreaming(false);
            setSteeringBusy(false);
            setActiveResearchRunId("");
            abortRef.current = null;
            setTimeout(() => inputRef.current?.focus(), 80);
        }
    }, [patchLastBot, pendingUploads, revealAnswer, setTransientSteeringStatus, streaming]);

    const handleSubmit = (event) => {
        event?.preventDefault();
        if (input.trim() || pendingUploads.length) runSearch(input.trim(), pendingUploads);
    };

    const latestBotMessage = active?.messages?.slice().reverse().find((message) => message.role === "bot") || null;
    const latestResearchMeta = latestBotMessage?.researchMeta || {};
    const activeOutputModeLabel = latestResearchMeta?.outputMode?.label || "State-of-the-Field";
    const activeSourceCount = latestBotMessage?.sources?.length || 0;
    const activeSubagentCount = countSubagentAssignments(latestResearchMeta?.subagents || {});
    const activeStability = Number(latestResearchMeta?.convergence?.stability_score);
    const activeStabilityLabel = Number.isFinite(activeStability) ? `${Math.round(activeStability * 100)}% stability` : "";
    const activeDagSummary = typeof latestResearchMeta?.dagSummary === "string" ? latestResearchMeta.dagSummary : "";
    const composerModeLabel = streaming ? "Live steering" : "Research composer";
    const composerTitle = streaming
        ? "Adjust depth, output mode, and source policy while the runtime is still working."
        : "Continue the active thread or launch a new cited search from this session.";

    return (
        <div className="se se--studio">
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
.edit-msg__textarea { width: 100%; padding: 10px; border-radius: 8px; border: 1px solid var(--bdr); background: var(--bg2); color: var(--tx); font-family: var(--font); font-size: 15px; margin-bottom: 8px; }
.edit-msg__actions { display: flex; gap: 8px; }
.edit-msg__btn { padding: 5px 12px; border-radius: 8px; border: none; background: var(--ac); color: #000; font-family: var(--font); font-size: 12px; font-weight: 600; cursor: pointer; }
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
.la-table--subagents .col-part{min-width:170px}
.la-table--subagents .col-owner{min-width:150px}
.la-table--subagents .col-focus{min-width:260px}
.subagent-owner__name{display:block;font-weight:600;color:var(--la-text-heading);letter-spacing:-.01em}
.subagent-owner__focus{display:block;font-size:11px;color:var(--la-text);margin-top:4px;line-height:1.4}
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
.bot__steer{display:flex;align-items:center;gap:6px;padding:10px 12px;border-top:1px solid var(--bdr);flex-wrap:wrap;background:rgba(255,255,255,0.02)}
.bot__steer-status{font-size:11px;color:var(--txd);margin-left:auto}
.bot__steer-input{min-width:140px;flex:1;border:1px solid var(--bdr);background:var(--bg2);color:var(--tx);border-radius:8px;padding:7px 10px;font-family:var(--font);font-size:12px;outline:none}
.bot__steer-input::placeholder{color:var(--txm)}
.bot__steer-input:disabled{opacity:.5}
.bot__steer-select{border:1px solid var(--bdr);background:var(--bg2);color:var(--txd);border-radius:8px;padding:7px 10px;font-family:var(--font);font-size:12px;outline:none}
.bb{display:flex;align-items:center;gap:5px;padding:5px 10px;border-radius:7px;border:1px solid var(--bdr);background:transparent;color:var(--txd);font-family:var(--font);font-size:12px;cursor:pointer;transition:all .2s}
.bb:hover{background:var(--sur);color:var(--tx)}
.bb--on{background:var(--sur);color:var(--tx);border-color:rgba(255,255,255,.12)}
.bb:disabled{opacity:.38;cursor:not-allowed}
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
            <style>{`
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');
.se--studio{
  --studio-bg:#081019;--studio-panel:rgba(13,20,29,0.82);--studio-panel-2:rgba(18,28,39,0.88);
  --studio-card:rgba(17,25,35,0.84);--studio-border:rgba(255,255,255,0.08);--studio-border-strong:rgba(255,158,76,0.2);
  --studio-text:#f6f0e4;--studio-muted:rgba(246,240,228,0.64);--studio-dim:rgba(246,240,228,0.44);
  --studio-accent:#ff9e4c;--studio-accent-soft:rgba(255,158,76,0.14);--studio-cool:#46d6c6;
  position:relative;background:
    radial-gradient(circle at 6% 0%, rgba(255,158,76,0.18), transparent 24%),
    radial-gradient(circle at 100% 12%, rgba(70,214,198,0.12), transparent 24%),
    linear-gradient(180deg,#081019 0%,#0e1822 48%,#14202c 100%);
}
.se--studio::before{content:'';position:absolute;inset:0;pointer-events:none;background-image:
  linear-gradient(rgba(255,255,255,0.025) 1px,transparent 1px),
  linear-gradient(90deg,rgba(255,255,255,0.025) 1px,transparent 1px);
  background-size:46px 46px;mask-image:radial-gradient(circle at center,rgba(0,0,0,.9),transparent 88%);opacity:.35}
.se--studio .sb{display:none}
.se--studio .mn{padding:20px 24px 22px;gap:0}
.se--studio .tb,
.se--studio .bot__wrap,
.se--studio .scard,
.se--studio .la-result{
  border-radius:24px;
  border:1px solid var(--studio-border);
  background:linear-gradient(160deg,var(--studio-panel),var(--studio-panel-2));
  box-shadow:0 20px 56px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.05);
  backdrop-filter:blur(20px);
}
.se--studio .tb{height:auto;min-height:78px;padding:16px 20px;gap:14px;margin-bottom:18px}
.se--studio .tb__meta{display:flex;flex-direction:column;gap:6px;min-width:0;flex:1}
.se--studio .tb__eyebrow{display:inline-flex;align-items:center;gap:8px;width:max-content;padding:6px 10px;border-radius:999px;background:var(--studio-accent-soft);border:1px solid rgba(255,158,76,.2);color:#ffd5ad;font-size:10px;font-weight:800;letter-spacing:.14em;text-transform:uppercase}
.se--studio .tb__q{font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;line-height:1.05;letter-spacing:-.04em;color:var(--studio-text);white-space:normal}
.se--studio .tb__b{padding:10px 14px;border-radius:14px;color:var(--studio-text);border-color:var(--studio-border);background:rgba(255,255,255,.04);font-size:12px;font-weight:700}
.se--studio .tb__b:hover{border-color:var(--studio-border-strong);background:var(--studio-accent-soft);color:#fff7ee}
.se--studio .chat{padding:0 0 220px}
.se--studio .chat__in{max-width:1100px;padding:0 8px;gap:28px}
.se--studio .umsg__av{width:38px;height:38px;flex-basis:38px;border-radius:14px;background:linear-gradient(135deg,#ff9e4c,#46d6c6)}
.se--studio .umsg__name{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--studio-dim)}
.se--studio .umsg__text,.se--studio .bmsg__body{font-family:'Manrope',system-ui,sans-serif;font-size:15px;line-height:1.78;color:var(--studio-text)}
.se--studio .scard{margin-left:50px;padding:18px 18px 16px;gap:14px}
.se--studio .scard__title{font-family:'Space Grotesk',sans-serif;font-size:14px;color:var(--studio-text)}
.se--studio .scard__sub,.se--studio .scard__searching,.se--studio .pstep__rest,.se--studio .qchip,.se--studio .sdot,.se--studio .land__sub,.se--studio .upload-status{color:var(--studio-muted)}
.se--studio .qchip{padding:7px 11px;border-radius:999px;background:rgba(255,255,255,.05);border-color:var(--studio-border)}
.se--studio .bmsg__ans{margin-left:50px;gap:14px}
.se--studio .la-result{overflow:hidden}
.se--studio .la-header{padding:18px 22px;background:rgba(255,255,255,.02)}
.se--studio .la-logo{border-radius:8px;background:var(--studio-accent);box-shadow:0 0 24px rgba(255,158,76,.28)}
.se--studio .la-brand,.se--studio .la-question,.se--studio .la-summary-title,.se--studio .la-card-title,.se--studio .la-source-name,.se--studio .la-criterion-label,.se--studio .subagent-owner__name{color:#fff8ee}
.se--studio .la-badge,.se--studio .la-card-badge{background:rgba(255,255,255,.05);border-color:var(--studio-border);color:var(--studio-muted)}
.se--studio .la-table thead th,.se--studio .la-query,.se--studio .la-summary-label,.se--studio .la-sources-title{color:var(--studio-dim)}
.se--studio .la-table tbody td,.se--studio .la-summary-text,.se--studio .la-point p,.se--studio .la-source-ref,.se--studio .la-discipline-tag,.se--studio .la-note,.se--studio .subagent-owner__focus{color:var(--studio-muted)}
.se--studio .la-source-url,.se--studio .la-card-icon,.se--studio .la-source:hover .la-source-title,.se--studio .cite,.se--studio .pill__n{color:var(--studio-accent)}
.se--studio .la-divider,.se--studio .la-table thead th,.se--studio .la-card-header,.se--studio .la-note{border-color:var(--studio-border)}
.se--studio .la-point,.se--studio .la-source:hover,.se--studio .pill:hover{background:rgba(255,255,255,.04)}
.se--studio .land{padding:32px 8px 210px;justify-content:flex-start}
.se--studio .land__logo{margin-top:18px;font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--studio-dim)}
.se--studio .land__star{width:42px;height:42px;border-radius:14px;display:grid;place-items:center;background:linear-gradient(135deg,#ff9e4c,#46d6c6);color:#091018;filter:none}
.se--studio .land__h1{max-width:760px;font-family:'Space Grotesk',sans-serif;font-size:clamp(42px,7vw,76px);line-height:.94;letter-spacing:-.06em}
.se--studio .land__sub{max-width:560px;font-size:16px}
.se--studio .land__form{max-width:840px;padding:8px 8px 8px 18px;border-radius:24px;background:linear-gradient(160deg,var(--studio-panel),rgba(20,31,43,.96));border:1px solid var(--studio-border);box-shadow:0 18px 44px rgba(0,0,0,.28)}
.se--studio .land__attach,.se--studio .land__btn,.se--studio .send-btn,.se--studio .stop-btn{width:44px;height:44px;border-radius:14px}
.se--studio .land__attach{border-color:var(--studio-border);color:var(--studio-text);background:rgba(255,255,255,.04)}
.se--studio .land__in,.se--studio .bot__in{font-family:'Manrope',system-ui,sans-serif;font-size:15px;color:var(--studio-text)}
.se--studio .land__in::placeholder,.se--studio .bot__in::placeholder,.se--studio .bot__steer-status{color:var(--studio-muted)}
.se--studio .land__btn,.se--studio .send-btn{background:var(--studio-accent);color:#091018;box-shadow:0 12px 28px rgba(255,158,76,.32)}
.se--studio .land__exs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;max-width:840px}
.se--studio .land__ex{padding:13px 15px;border-radius:16px;background:rgba(255,255,255,.04);border:1px solid var(--studio-border);color:var(--studio-text)}
.se--studio .land__ex:hover{background:var(--studio-accent-soft);border-color:var(--studio-border-strong);color:#fff8ee}
.se--studio .att-list{gap:10px}
.se--studio .att-chip{padding:10px 12px;border-radius:16px;background:rgba(255,255,255,.04);border-color:var(--studio-border)}
.se--studio .bot{padding:0 28px 24px;background:linear-gradient(180deg,transparent,rgba(8,16,25,.66) 22%,rgba(8,16,25,.96))}
.se--studio .bot__wrap{max-width:1080px}
.se--studio .bot__uploads{padding:14px 14px 0}
.se--studio .bot__row{padding:8px 8px 8px 18px;gap:10px}
.se--studio .bot__bar{padding:10px 12px;border-top:1px solid var(--studio-border)}
.se--studio .bot__steer{padding:12px 14px;border-top:1px solid var(--studio-border);background:rgba(255,255,255,.03)}
.se--studio .bot__steer-input,.se--studio .bot__steer-select,.se--studio .bb,.se--studio .ib{border-radius:12px;border-color:var(--studio-border);background:rgba(255,255,255,.04);color:var(--studio-text)}
.se--studio .bb--on,.se--studio .bb:hover,.se--studio .ib:hover{background:var(--studio-accent-soft);border-color:var(--studio-border-strong);color:#fff7ee}
.se--studio .stop-btn{background:#d65d5d;color:#fff}
@media(max-width:980px){
  .se--studio .mn{padding:14px 14px 18px}
  .se--studio .tb{padding:16px}
  .se--studio .chat__in{padding:0}
  .se--studio .bmsg__ans,.se--studio .scard{margin-left:0}
  .se--studio .land__exs{grid-template-columns:1fr}
}
@media(max-width:720px){
  .se--studio .tb{flex-direction:column;align-items:flex-start}
  .se--studio .tb__q{font-size:16px}
  .se--studio .land{padding:18px 0 220px}
  .se--studio .land__h1{font-size:clamp(34px,11vw,56px)}
  .se--studio .bot{padding:0 10px 12px}
  .se--studio .bot__row{padding-left:12px}
  .se--studio .bot__steer{padding:10px}
}
`}</style>
            <style>{`
.se--studio{height:100%;min-height:100%}
.se--studio .sb{display:none}
.se--studio .mn{min-height:0}
.se--studio .tb{
  display:grid;
  grid-template-columns:minmax(0,1fr) auto auto;
  align-items:center;
}
.se--studio .tb__summary{display:flex;flex-direction:column;gap:10px;min-width:0}
.se--studio .tb__meta,
.se--studio .tb__stats{display:flex;flex-wrap:wrap;gap:8px}
.se--studio .tb__side{display:flex;flex-direction:column;align-items:flex-end;gap:10px}
.se--studio .tb__hint{
  max-width:320px;color:var(--studio-muted);font-size:11px;line-height:1.55;text-align:right
}
.se--studio .tb__pill{
  display:inline-flex;align-items:center;justify-content:center;
  min-height:34px;padding:0 12px;border-radius:999px;
  background:rgba(255,255,255,.045);border:1px solid var(--studio-border);
  color:var(--studio-text);font-size:11px;font-weight:700;letter-spacing:.02em
}
.se--studio .tb__pill--accent{
  background:rgba(255,158,76,.12);border-color:rgba(255,158,76,.24);color:#fff1dd
}
.se--studio .chat__in{gap:32px}
.se--studio .umsg{align-items:stretch;max-width:820px;margin-left:auto}
.se--studio .umsg__body{
  border-radius:22px;border:1px solid rgba(255,255,255,.06);background:rgba(255,255,255,.04);
  padding:16px 18px 14px;box-shadow:0 18px 40px rgba(0,0,0,.18)
}
.se--studio .umsg__card{
  background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.06);
  border-radius:22px;
  padding:16px 18px 14px;
  box-shadow:0 18px 40px rgba(0,0,0,.18)
}
.se--studio .umsg__head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:8px}
.se--studio .umsg__meta{
  color:var(--studio-dim);font-size:10px;font-weight:700;letter-spacing:.12em;text-transform:uppercase
}
.se--studio .umsg__acts{padding-top:12px}
.se--studio .scard{box-shadow:0 18px 44px rgba(0,0,0,.2), inset 0 1px 0 rgba(255,255,255,.04)}
.se--studio .la-header{
  display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap
}
.se--studio .la-header-main{display:flex;align-items:center;gap:14px}
.se--studio .la-header-copy{display:flex;flex-direction:column;gap:3px}
.se--studio .la-kicker{
  color:var(--studio-dim);font-size:10px;font-weight:700;letter-spacing:.16em;text-transform:uppercase
}
.se--studio .la-header-badges{display:flex;flex-wrap:wrap;gap:8px}
.se--studio .la-hero{
  display:grid;grid-template-columns:minmax(0,1.7fr) minmax(260px,.9fr);gap:16px;margin-bottom:18px
}
.se--studio .la-hero-copy{
  padding:18px;border-radius:20px;
  background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.025));
  border:1px solid rgba(255,255,255,.05)
}
.se--studio .la-question{font-size:26px;line-height:1.05;letter-spacing:-.05em}
.se--studio .la-hero-stats{
  display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px
}
.se--studio .la-stat{
  padding:16px 15px;border-radius:18px;background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.05);display:flex;flex-direction:column;gap:8px;
  min-height:104px;justify-content:space-between
}
.se--studio .la-stat__label{
  color:var(--studio-dim);font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase
}
.se--studio .la-stat strong{
  color:#fff9f0;font-family:'Space Grotesk',sans-serif;font-size:16px;line-height:1.15
}
.se--studio .la-summary-title{
  font-family:'Space Grotesk',sans-serif;font-size:32px;line-height:1.02;letter-spacing:-.05em
}
.se--studio .la-summary-text{font-size:15px;line-height:1.8;max-width:72ch}
.se--studio .la-domain-strip{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 6px}
.se--studio .la-domain-pill{
  display:inline-flex;align-items:center;padding:6px 11px;border-radius:999px;
  background:rgba(70,214,198,.08);border:1px solid rgba(70,214,198,.18);
  color:#bdf7f0;font-size:11px;font-weight:700
}
.se--studio .la-points{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:20px}
.se--studio .la-point{
  min-height:100%;padding:16px 16px 15px;border-radius:18px;border:1px solid rgba(255,255,255,.05)
}
.se--studio .la-tables{display:grid;gap:16px;margin-top:22px}
.se--studio .la-card{
  border-radius:22px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);
  overflow:hidden
}
.se--studio .la-footer{padding-top:14px}
.se--studio .land{
  max-width:1180px;margin:0 auto;align-items:flex-start;text-align:left
}
.se--studio .land__eyebrow{
  display:inline-flex;align-items:center;justify-content:center;
  min-height:34px;padding:0 14px;border-radius:999px;
  background:rgba(255,158,76,.12);border:1px solid rgba(255,158,76,.22);
  color:#ffd8b5;font-size:10px;font-weight:800;letter-spacing:.16em;text-transform:uppercase
}
.se--studio .land__logo{gap:12px}
.se--studio .land__capabilities{display:flex;flex-wrap:wrap;gap:10px}
.se--studio .land__cap{
  display:inline-flex;align-items:center;padding:8px 12px;border-radius:999px;
  background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.06);
  color:var(--studio-text);font-size:12px;font-weight:600
}
.se--studio .land__form{width:100%}
.se--studio .land__panels{
  width:100%;display:grid;grid-template-columns:minmax(0,1.45fr) minmax(280px,.9fr);gap:16px
}
.se--studio .land__panel{
  padding:20px;border-radius:24px;background:linear-gradient(160deg,var(--studio-panel),rgba(18,28,39,.92));
  border:1px solid rgba(255,255,255,.06);box-shadow:0 18px 40px rgba(0,0,0,.2)
}
.se--studio .land__panel-head{display:flex;flex-direction:column;gap:6px;margin-bottom:16px}
.se--studio .land__panel-eyebrow{
  color:var(--studio-dim);font-size:10px;font-weight:800;letter-spacing:.16em;text-transform:uppercase
}
.se--studio .land__panel-title{
  color:#fff8ee;font-family:'Space Grotesk',sans-serif;font-size:24px;line-height:1.02;letter-spacing:-.04em
}
.se--studio .land__exs{max-width:none}
.se--studio .land__ex{
  min-height:132px;display:flex;flex-direction:column;align-items:flex-start;justify-content:flex-start;gap:10px;
  text-align:left
}
.se--studio .land__ex-label{
  color:#fff8ee;font-size:14px;font-weight:800;letter-spacing:-.02em
}
.se--studio .land__ex-detail{color:var(--studio-muted);font-size:12px;line-height:1.55}
.se--studio .land__ex-query{
  color:#ffd8b5;font-size:12px;line-height:1.55;font-family:var(--mono);word-break:break-word
}
.se--studio .land__flow{display:flex;flex-direction:column;gap:12px}
.se--studio .land__flow-step{
  display:flex;gap:12px;align-items:flex-start;padding:14px 0;border-top:1px solid rgba(255,255,255,.06)
}
.se--studio .land__flow-step:first-child{border-top:none;padding-top:0}
.se--studio .land__flow-step span{
  display:inline-flex;align-items:center;justify-content:center;
  width:32px;height:32px;border-radius:10px;background:rgba(255,158,76,.14);
  color:#ffd8b5;font-size:11px;font-weight:800;flex:0 0 32px
}
.se--studio .land__flow-step strong{color:#fff8ee;font-size:14px;line-height:1.55}
.se--studio .bot{padding:0 28px 22px}
.se--studio .bot__wrap{overflow:visible}
.se--studio .bot__titlebar{
  display:flex;align-items:flex-start;justify-content:space-between;gap:16px;
  padding:16px 18px 0
}
.se--studio .bot__titlecopy{display:flex;flex-direction:column;gap:6px}
.se--studio .bot__eyebrow{
  color:var(--studio-dim);font-size:10px;font-weight:800;letter-spacing:.16em;text-transform:uppercase
}
.se--studio .bot__title{color:#fff8ee;font-size:14px;line-height:1.55}
.se--studio .bot__runtime{
  display:inline-flex;align-items:center;justify-content:center;min-height:34px;
  padding:0 12px;border-radius:999px;background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.06);color:var(--studio-muted);font-size:11px;font-weight:700
}
.se--studio .bot__uploads{padding-top:12px}
.se--studio .bot__row{padding-top:12px}
.se--studio .bot__steer{
  display:grid;grid-template-columns:repeat(3,max-content) minmax(220px,1fr) auto;gap:12px;align-items:end
}
.se--studio .bot__steer-group{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.se--studio .bot__steer-group--grow{min-width:0}
.se--studio .bot__steer-label{
  color:var(--studio-dim);font-size:10px;font-weight:800;letter-spacing:.16em;text-transform:uppercase
}
.se--studio .bot__steer-status{
  min-height:42px;display:flex;align-items:center;justify-content:flex-end;text-align:right
}
@media(max-width:1100px){
  .se--studio .tb{grid-template-columns:minmax(0,1fr) auto;align-items:flex-start}
  .se--studio .tb__summary{grid-column:1 / -1}
  .se--studio .tb__side{grid-column:1 / -1;align-items:flex-start}
  .se--studio .tb__hint{text-align:left;max-width:none}
  .se--studio .la-hero{grid-template-columns:1fr}
  .se--studio .land__panels{grid-template-columns:1fr}
  .se--studio .bot__steer{grid-template-columns:1fr}
  .se--studio .bot__steer-status{justify-content:flex-start;text-align:left}
}
@media(max-width:720px){
  .se--studio .sb{display:none}
  .se--studio .mn{padding:12px 12px 14px}
  .se--studio .tb{grid-template-columns:1fr}
  .se--studio .tb__summary,.se--studio .tb__side{grid-column:auto}
  .se--studio .tb__b{width:100%}
  .se--studio .la-points{grid-template-columns:1fr}
  .se--studio .la-hero-stats{grid-template-columns:1fr 1fr}
  .se--studio .land__h1{max-width:11ch}
  .se--studio .bot__titlebar{flex-direction:column;align-items:flex-start}
}
@media(max-width:560px){
  .se--studio .la-question,.se--studio .la-summary-title{font-size:24px}
  .se--studio .la-hero-stats{grid-template-columns:1fr}
  .se--studio .land__exs{grid-template-columns:1fr}
}
`}</style>

            <aside className="sb">
                <div className="sb__logo" onClick={() => { resetComposer(); }}>✳</div>
                <button
                    className="sb__new"
                    onClick={() => {
                        resetComposer();
                    }}
                    title="New search"
                >
                    ＋
                </button>
                <button
                    className={`nb ${!isLibraryView ? "nb--on" : ""}`}
                    onClick={() => {
                        setShowLibrary(false);
                        setTimeout(() => inputRef.current?.focus(), 40);
                    }}
                    title="Research workspace"
                >
                    ⌕
                </button>
                <button
                    className={`nb ${isLibraryView ? "nb--on" : ""}`}
                    onClick={() => setShowLibrary(true)}
                    title="Saved sessions"
                >
                    ☰
                </button>
                <div className="sb__sp" />
                <div className="sb__status">{streaming ? "LIVE" : "READY"}</div>
            </aside>

            <div className="mn">
                {isLibraryView ? (
                    <Library
                        onBack={() => setShowLibrary(false)}
                        onViewSession={(savedSession) => {
                            activateStoredSession(savedSession);
                        }}
                        onNewSearch={() => {
                            resetComposer();
                            inputRef.current?.focus();
                        }}
                    />
                ) : (
                    <>
                        {!isLanding && (
                            <div className="tb">
                                <div className="tb__summary">
                                    <div className="tb__eyebrow">Research session</div>
                                    <div className="tb__q">{active?.query || ""}</div>
                                    <div className="tb__meta">
                                        <span className="tb__pill">{activeOutputModeLabel}</span>
                                        {latestResearchMeta?.pareto?.mode ? <span className="tb__pill">Pareto: {latestResearchMeta.pareto.mode}</span> : null}
                                        {activeSubagentCount ? <span className="tb__pill">{activeSubagentCount} subagents</span> : null}
                                        {activeSourceCount ? <span className="tb__pill">{activeSourceCount} cited source{activeSourceCount === 1 ? "" : "s"}</span> : null}
                                        {activeStabilityLabel ? <span className="tb__pill tb__pill--accent">{activeStabilityLabel}</span> : null}
                                    </div>
                                </div>
                                <div className="tb__side">
                                    {activeDagSummary ? <div className="tb__hint">{activeDagSummary}</div> : null}
                                    <button className="tb__b" onClick={() => {
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
                                    }}>Share session</button>
                                </div>
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
                                    <div className="bot__titlebar">
                                        <div className="bot__titlecopy">
                                            <div className="bot__eyebrow">{composerModeLabel}</div>
                                            <div className="bot__title">{composerTitle}</div>
                                        </div>
                                        <div className="bot__runtime">
                                            {latestResearchMeta?.frameworkVersion ? `Framework v${latestResearchMeta.frameworkVersion}` : "Research runtime"}
                                        </div>
                                    </div>
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
                                    {streaming && (
                                        <div className="bot__steer">
                                            <div className="bot__steer-group">
                                                <span className="bot__steer-label">Pace</span>
                                                <button
                                                    type="button"
                                                    className="bb"
                                                    disabled={!activeResearchRunId || steeringBusy}
                                                    onClick={() => queueSteeringCommand({ type: "prioritize_speed" })}
                                                >
                                                    ⚡ Speed
                                                </button>
                                                <button
                                                    type="button"
                                                    className="bb"
                                                    disabled={!activeResearchRunId || steeringBusy}
                                                    onClick={() => queueSteeringCommand({ type: "go_deeper" })}
                                                >
                                                    ⇣ Deep
                                                </button>
                                            </div>
                                            <div className="bot__steer-group">
                                                <span className="bot__steer-label">Evidence</span>
                                                <button
                                                    type="button"
                                                    className={`bb ${excludedPreprints ? "bb--on" : ""}`}
                                                    disabled={!activeResearchRunId || steeringBusy || excludedPreprints}
                                                    onClick={() => queueSteeringCommand({ type: "exclude_source", sourceType: "preprint" })}
                                                >
                                                    ⛔ Preprints
                                                </button>
                                            </div>
                                            <div className="bot__steer-group">
                                                <span className="bot__steer-label">Output</span>
                                                <select
                                                    className="bot__steer-select"
                                                    value={steeringMode}
                                                    disabled={!activeResearchRunId || steeringBusy}
                                                    onChange={(event) => setSteeringMode(event.target.value)}
                                                >
                                                    <option value="state_of_the_field">State</option>
                                                    <option value="gap_analysis">Gap</option>
                                                    <option value="controversy_map">Controversy</option>
                                                    <option value="tutorial">Tutorial</option>
                                                    <option value="decision_brief">Decision</option>
                                                </select>
                                                <button
                                                    type="button"
                                                    className="bb"
                                                    disabled={!activeResearchRunId || steeringBusy}
                                                    onClick={() => queueSteeringCommand({ type: "force_mode", mode: steeringMode })}
                                                >
                                                    ↺ Mode
                                                </button>
                                            </div>
                                            <div className="bot__steer-group bot__steer-group--grow">
                                                <span className="bot__steer-label">Focus</span>
                                                <input
                                                    className="bot__steer-input"
                                                    value={steeringArea}
                                                    onChange={(event) => setSteeringArea(event.target.value)}
                                                    placeholder="Deepen an area or hypothesis..."
                                                    disabled={!activeResearchRunId || steeringBusy}
                                                />
                                                <button
                                                    type="button"
                                                    className="bb"
                                                    disabled={!activeResearchRunId || steeringBusy || !steeringArea.trim()}
                                                    onClick={() => queueSteeringCommand({ type: "increase_depth", area: steeringArea.trim() })}
                                                >
                                                    + Depth
                                                </button>
                                            </div>
                                            <div className="bot__steer-status">
                                                {steeringStatus || (activeResearchRunId ? `Run ${String(activeResearchRunId).slice(-8)}` : "Waiting for runtime...")}
                                            </div>
                                        </div>
                                    )}
                                    <div className="bot__bar">
                                        <button className="bb bb--on" onClick={() => alert("Search mode: Active — web research with query expansion")}>🔍 Search</button>
                                        <div className="bb__sp" />
                                        <button className="ib" title="Attach" onClick={openFilePicker} disabled={streaming}>📎</button>
                                        {streaming && (
                                            <button className="ib" style={{ color: "var(--red)" }} onClick={() => abortRef.current?.abort()}>■</button>
                                        )}
                                    </div>
                                </div>
                            </div>
                        )}
                    </>
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
