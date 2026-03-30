const { handler: webSearchHandler } = require("../tools/web_search");
const { handler: webFetchHandler } = require("../tools/web_fetch");
const {
    canonicalizeSourceUrl,
    extractQueryTerms,
    scoreTextForTerms,
    domainAuthorityBoost,
    getSourceDomain,
    mergeSourcesByCanonicalUrl,
    selectSourcesForFetch,
    rankSourcesWithRag,
    rankEvidenceEntriesForQuery,
    buildRagEvidenceBlock,
    RAG_FETCH_MAX_CHARS,
} = require("../rag");
const { parseFetchToolPayload, stripFetchMeta } = require("../web");
const { runLiteHostChat } = require("../litehost-chat");
const {
    compileResearchPlan,
    applySteeringCommands,
    summarizeDag,
    RESEARCH_FRAMEWORK_VERSION,
} = require("./planner");
const {
    resolveResearchScope,
    createResearchRun,
    updateResearchRun,
    appendResearchCheckpoint,
    appendResearchIndex,
    updateResearchMemoryFromRun,
    consumeSteeringCommands,
} = require("./storage");

const FETCH_TARGET = 18;
const FETCH_CONCURRENCY = 4;
const SYNTHESIS_SWARM_SIZE = 3;
const DEFAULT_SEARCH_RESULT_LIMIT = 8;
const DEFAULT_TIMEOUT_MS = 12000;
const USER_AGENT = "nub-agent/1.0";

const normalizeText = (value) => String(value || "").replace(/\u0000/g, "").replace(/\r\n?/g, "\n").trim();

const clamp = (value, min, max, fallback = min) => {
    const number = Number(value);
    if (!Number.isFinite(number)) return fallback;
    return Math.min(max, Math.max(min, number));
};

const clamp01 = (value) => Math.max(0, Math.min(1, Number(value) || 0));

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const parseJsonObject = (value) => {
    if (value && typeof value === "object" && !Array.isArray(value)) return value;
    const raw = normalizeText(value);
    if (!raw) return null;

    const cleaned = raw
        .replace(/^```(?:json)?\s*/i, "")
        .replace(/\s*```$/i, "")
        .trim();

    try {
        return JSON.parse(cleaned);
    } catch {
        const match = cleaned.match(/\{[\s\S]*\}$/);
        if (!match) return null;
        try {
            return JSON.parse(match[0]);
        } catch {
            return null;
        }
    }
};

const toArray = (value) => (Array.isArray(value) ? value : []);

const toPositiveInteger = (value) => {
    const parsed = Number(value);
    return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
};

const extractCitationNumbers = (text = "") => {
    const citations = [];
    const seen = new Set();
    for (const match of String(text || "").matchAll(/\[(\d+)\]/g)) {
        const number = toPositiveInteger(match[1]);
        if (!number || seen.has(number)) continue;
        seen.add(number);
        citations.push(number);
    }
    return citations;
};

const extractAnswerParts = (text = "") => {
    const clean = String(text || "").replace(/##\s*Sources?[\s\S]*$/i, "").trim();
    const lines = clean.split("\n");
    const heading = (lines[0] || "").replace(/^#+\s*/, "").trim();
    const body = lines.length > 1 ? lines.slice(1).join("\n").trim() : clean;
    return {
        heading: heading || "Research Answer",
        body: body || heading || clean,
    };
};

const uniqueBy = (items = [], keyFn) => {
    const seen = new Set();
    const result = [];
    for (const item of items) {
        const key = keyFn(item);
        if (!key || seen.has(key)) continue;
        seen.add(key);
        result.push(item);
    }
    return result;
};

const buildCsv = (rows = []) => {
    if (!rows.length) return "";
    const columns = [...new Set(rows.flatMap((row) => Object.keys(row || {})))];
    const escape = (value) => {
        const text = String(value ?? "");
        if (/["\n,]/.test(text)) return `"${text.replace(/"/g, "\"\"")}"`;
        return text;
    };
    return [
        columns.join(","),
        ...rows.map((row) => columns.map((column) => escape(row?.[column])).join(",")),
    ].join("\n");
};

const jaccardSimilarity = (left = "", right = "") => {
    const a = new Set(extractQueryTerms(left, 64));
    const b = new Set(extractQueryTerms(right, 64));
    if (!a.size && !b.size) return 1;
    if (!a.size || !b.size) return 0;
    let intersection = 0;
    for (const item of a) {
        if (b.has(item)) intersection += 1;
    }
    const union = new Set([...a, ...b]).size;
    return union ? intersection / union : 0;
};

const withTimeout = async (worker, timeoutMs = DEFAULT_TIMEOUT_MS) => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        return await worker(controller.signal);
    } finally {
        clearTimeout(timer);
    }
};

const parseSearchResults = (value) => {
    if (Array.isArray(value)) return value;
    if (typeof value !== "string") return [];
    if (value.startsWith("Error:")) return [];
    try {
        return JSON.parse(value);
    } catch {
        return [];
    }
};

const parseFetchedPayload = (value) => {
    if (typeof value !== "string") return { content: "", title: "", finalUrl: "", publishedTime: "", via: "" };
    if (value.startsWith("Error:")) return { error: value };
    return parseFetchToolPayload(value);
};

const getQueryOverlap = (query, text) => {
    const terms = extractQueryTerms(query, 24);
    const score = scoreTextForTerms(text, terms);
    return clamp01(score / Math.max(terms.length * 1.2, 1));
};

const getDateAgeYears = (value) => {
    const timestamp = Date.parse(String(value || ""));
    if (!Number.isFinite(timestamp)) return null;
    const deltaYears = Math.max(0, (Date.now() - timestamp) / (365.25 * 24 * 60 * 60 * 1000));
    return deltaYears;
};

const computeFreshnessWeight = (value, halfLifeYears = 4) => {
    const ageYears = getDateAgeYears(value);
    if (ageYears == null) return 0.72;
    const weight = Math.exp((-Math.log(2) * ageYears) / Math.max(halfLifeYears, 0.5));
    return clamp01(Math.max(0.18, weight));
};

const detectEvidenceType = (source = {}, content = "") => {
    const haystack = `${source.title || ""}\n${source.description || ""}\n${content}`.toLowerCase();
    if (/\brandomized|rct|clinical trial\b/.test(haystack)) return "rct";
    if (/\bmeta-analysis|systematic review\b/.test(haystack)) return "meta_analysis";
    if (/\bcohort|observational|survey|case study\b/.test(haystack)) return "observational";
    if (/\bbenchmark|ablation|evaluation\b/.test(haystack)) return "benchmark";
    if (/\bopinion|commentary|editorial\b/.test(haystack)) return "opinion";
    return "general";
};

const evidenceTypeWeight = (type = "general") => ({
    meta_analysis: 1,
    rct: 0.94,
    observational: 0.72,
    benchmark: 0.72,
    general: 0.64,
    opinion: 0.34,
}[type] || 0.58);

const extractUrls = (text = "") => Array.from(
    new Set(
        (String(text || "").match(/\bhttps?:\/\/[^\s)<>"']+/gi) || [])
            .map((item) => item.replace(/[),.;:!?]+$/, "")),
    ),
);

const extractSupplementaryLinks = (text = "") => extractUrls(text).filter((url) => /supplement|appendix|appendices|supp\b/i.test(url));
const extractRepoLinks = (text = "") => extractUrls(text).filter((url) => /github\.com|gitlab\.com|bitbucket\.org|paperswithcode\.com/i.test(url));

const extractFundingSignals = (text = "") => {
    const matches = String(text || "").match(/\b(funded by|sponsored by|supported by|grant from|industry[- ]funded)\b/gi) || [];
    return Array.from(new Set(matches.map((item) => normalizeText(item))));
};

const extractStatClaims = (text = "") => {
    const normalized = String(text || "");
    const claims = [];
    const pValueMatch = normalized.match(/\bp\s*(?:=|<|>)\s*0?\.\d+/gi) || [];
    const ciMatch = normalized.match(/\b(?:95%?\s*ci|confidence interval)[^.\n]{0,120}/gi) || [];
    const sampleMatch = normalized.match(/\bn\s*=\s*\d+/gi) || [];
    const effectMatch = normalized.match(/\b(?:effect size|cohen'?s d|odds ratio|hazard ratio|relative risk|i\^?2)\b[^.\n]{0,120}/gi) || [];

    for (const value of [...pValueMatch, ...ciMatch, ...sampleMatch, ...effectMatch].slice(0, 12)) {
        claims.push({
            text: normalizeText(value),
        });
    }
    return claims;
};

const buildQuantitativeClaims = (entries = []) => {
    const claims = [];
    for (const entry of toArray(entries)) {
        const source = entry.source || {};
        const fragments = extractStatClaims(entry.content || source.description || "");
        for (const fragment of fragments) {
            claims.push({
                sourceTitle: source.title || source.url,
                sourceUrl: source.url,
                citationIndex: source.citationIndex,
                text: fragment.text,
            });
        }
    }
    return claims;
};

const estimateContradictionScore = (positionMaps = [], thesisText = "", antithesisText = "") => {
    const contradictionTerms = ["however", "but", "conflict", "contradict", "uncertain", "mixed"];
    const corpus = `${positionMaps.join("\n")}\n${thesisText}\n${antithesisText}`.toLowerCase();
    let hits = 0;
    for (const term of contradictionTerms) {
        if (corpus.includes(term)) hits += 1;
    }
    return clamp01(hits / 8);
};

const detectPlausibleStatisticalManipulation = (claims = []) => {
    const warnings = [];
    for (const claim of claims) {
        const text = String(claim?.text || "");
        if (/\bp\s*=\s*0\.05\b/i.test(text)) warnings.push("edge-threshold p-value");
        if (/\bn\s*=\s*[1-9]\d?\b/i.test(text)) warnings.push("small sample size");
        if (/\b100\.0{2,}\b/.test(text)) warnings.push("implausibly round numeric precision");
    }
    return Array.from(new Set(warnings));
};

const parseMetaAnalysisNumber = (text, patterns = []) => {
    for (const pattern of patterns) {
        const match = String(text || "").match(pattern);
        if (match) {
            const number = Number(match[1]);
            if (Number.isFinite(number)) return number;
        }
    }
    return null;
};

const buildCrossPaperStatSummary = (claims = []) => {
    const effectValues = [];
    for (const claim of claims) {
        const text = String(claim?.text || "");
        const effect = parseMetaAnalysisNumber(text, [
            /\bcohen'?s d[^-+0-9]*([-+]?\d+(?:\.\d+)?)/i,
            /\beffect size[^-+0-9]*([-+]?\d+(?:\.\d+)?)/i,
            /\bodds ratio[^0-9]*([0-9]+(?:\.\d+)?)/i,
            /\bhazard ratio[^0-9]*([0-9]+(?:\.\d+)?)/i,
            /\brelative risk[^0-9]*([0-9]+(?:\.\d+)?)/i,
        ]);
        if (effect != null) effectValues.push(effect);
    }

    if (!effectValues.length) {
        return {
            combined_effect_size: null,
            i_squared: null,
            model: "not_applicable",
            assumption_conflicts: detectPlausibleStatisticalManipulation(claims),
        };
    }

    const mean = effectValues.reduce((sum, value) => sum + value, 0) / effectValues.length;
    const variance = effectValues.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / Math.max(effectValues.length - 1, 1);
    const iSquared = clamp(Math.round((variance / Math.max(Math.abs(mean) + variance, 0.01)) * 100), 0, 100, 0);

    return {
        combined_effect_size: Number(mean.toFixed(2)),
        i_squared: iSquared,
        model: effectValues.length > 1 ? "random_effects" : "single_study",
        assumption_conflicts: detectPlausibleStatisticalManipulation(claims),
    };
};

const buildProbabilisticClaims = (claims = [], tribunal = {}, contradictionScore = 0.2) => {
    const internalConsistency = tribunal?.critics?.internal_consistency ?? 0.78;
    const coverage = tribunal?.critics?.coverage ?? 0.76;
    return claims.slice(0, 12).map((claim, index) => ({
        id: `claim-${index + 1}`,
        claim: claim?.text || "",
        confidence: Number(clamp01((internalConsistency * 0.5) + (coverage * 0.3) + 0.18 - (contradictionScore * 0.2)).toFixed(2)),
        evidence_weight: Number(clamp01(0.58 + (coverage * 0.34)).toFixed(2)),
        contradiction_score: Number(clamp01(contradictionScore).toFixed(2)),
        sensitivity: Number(clamp01(0.22 + (index / Math.max(claims.length, 1)) * 0.12).toFixed(2)),
    }));
};

const buildDecisionPayload = (plan, tribunal, convergence, uncertaintyClaims) => {
    const confidence = Number(clamp01(
        ((tribunal?.critics?.coverage ?? 0.72) * 0.3)
        + ((tribunal?.critics?.user_goal_alignment ?? 0.76) * 0.3)
        + ((tribunal?.critics?.internal_consistency ?? 0.78) * 0.2)
        + (1 - (convergence?.residual_uncertainty ?? 0.24)) * 0.2,
    ).toFixed(2));
    const outputMode = plan?.outputMode?.id || "state_of_the_field";
    const recommend = ["decision_brief", "policy_recommendation", "engineering_action_plan"].includes(outputMode);
    const riskScale = Number((convergence?.residual_uncertainty ?? 0.24).toFixed(2));

    return {
        decision: recommend ? `Proceed with the strongest supported option for ${plan.query}` : `Use the synthesis as a research brief for ${plan.query}`,
        expected_outcome: recommend
            ? "Decision guidance grounded in the highest-ranked evidence and explicit uncertainty."
            : "Research synthesis grounded in the highest-ranked evidence and explicit uncertainty.",
        risk_profile: {
            technical: Number(clamp01(riskScale + (plan?.domain?.id === "cs_ml" ? 0.06 : 0)).toFixed(2)),
            epistemic: Number(clamp01(riskScale + ((tribunal?.critics?.coverage ?? 0.76) < 0.8 ? 0.08 : 0)).toFixed(2)),
        },
        confidence,
        reversibility: confidence >= 0.82 ? "high" : confidence >= 0.64 ? "medium" : "low",
        top_uncertainties: uncertaintyClaims.slice(0, 3).map((claim) => claim.claim).filter(Boolean),
    };
};

const buildSlideDeckOutline = (heading, body, checkpoints = []) => {
    const sections = String(body || "")
        .split(/\n##+\s+/)
        .map((item) => normalizeText(item))
        .filter(Boolean)
        .slice(0, 8);
    return [
        `# ${heading || "Research Brief"}`,
        "",
        "1. Executive Summary",
        "2. Research Scope",
        "3. Source Inventory",
        "4. Evidence Map",
        "5. Dominant View",
        "6. Counter View",
        "7. Risks and Uncertainty",
        "8. Decision Guidance",
        "",
        ...sections.map((section, index) => `${index + 9}. ${section.slice(0, 96)}`),
        "",
        `Checkpoint count: ${toArray(checkpoints).length}`,
    ].join("\n");
};

const buildObsidianMarkdown = (finalResult, plan, runId) => {
    const tags = [
        "research",
        plan?.domain?.id || "general",
        plan?.scope?.id || "broad",
        plan?.outputMode?.id || "state-of-the-field",
    ].filter(Boolean);
    return [
        "---",
        `title: "${(finalResult.heading || "Research Brief").replace(/"/g, "'")}"`,
        `run_id: "${runId}"`,
        `framework_version: "${RESEARCH_FRAMEWORK_VERSION}"`,
        `tags: [${tags.map((tag) => `"${tag}"`).join(", ")}]`,
        "---",
        "",
        `# ${finalResult.heading || "Research Brief"}`,
        "",
        finalResult.body || "",
        "",
        `Backlink: [[Research Run ${runId}]]`,
    ].join("\n");
};

const buildFallbackNarrative = (query, evidenceEntries = [], plan) => {
    const ranked = toArray(evidenceEntries).slice(0, 6);
    const bullets = ranked.map((entry) => {
        const source = entry.source || {};
        const snippet = normalizeText((entry.content || source.description || "").split("\n").slice(0, 3).join(" ")).slice(0, 240);
        return `- ${source.title || source.url}: ${snippet || "Relevant evidence."} [${source.citationIndex}]`;
    });
    return [
        `# ${plan?.outputMode?.label || "Research Brief"}: ${query}`,
        "",
        "## Summary",
        bullets.length ? bullets.join("\n") : "- No readable evidence was collected.",
        "",
        "## Residual Uncertainty",
        "- Evidence quality and coverage remain limited where the source inventory is sparse.",
        "",
        "## Sources used",
        extractCitationNumbers(bullets.join("\n")).map((item) => `[${item}]`).join(" ") || "None",
    ].join("\n");
};

const buildAttributedSourcesFromEvidence = (answerText = "", evidenceEntries = [], limit = 24) => {
    const citedNumbers = extractCitationNumbers(answerText);
    const byCitation = new Map(
        toArray(evidenceEntries)
            .map((entry) => [toPositiveInteger(entry?.source?.citationIndex), entry?.source])
            .filter(([citationIndex, source]) => citationIndex && source?.url),
    );

    const citedSources = citedNumbers
        .map((citationIndex) => byCitation.get(citationIndex))
        .filter(Boolean);

    const fallback = toArray(evidenceEntries)
        .map((entry) => entry?.source)
        .filter(Boolean)
        .sort((left, right) => (left?.citationIndex || 0) - (right?.citationIndex || 0));

    return uniqueBy(citedSources.length ? citedSources : fallback, (source) => canonicalizeSourceUrl(source?.url || "") || `c:${source?.citationIndex || ""}`)
        .slice(0, limit);
};

const buildAttachmentEvidenceEntries = (attachments = [], startCitationIndex = 1) => {
    let citationIndex = startCitationIndex;
    return toArray(attachments).map((attachment) => {
        const kind = attachment?.kind === "image" ? "image" : "text";
        const content = kind === "text"
            ? normalizeText(attachment?.textContent || "").slice(0, RAG_FETCH_MAX_CHARS)
            : `Image attachment: ${normalizeText(attachment?.name || "attachment")}`;
        const entry = {
            source: {
                title: normalizeText(attachment?.name || `Attachment ${citationIndex}`),
                url: `attachment://${attachment?.id || citationIndex}`,
                description: kind === "image" ? "User-provided image attachment." : "User-provided text attachment.",
                source: "attachment",
                tier: "core",
                epistemicScore: 1,
                citationIndex,
            },
            content,
        };
        citationIndex += 1;
        return entry;
    }).filter((entry) => normalizeText(entry.content));
};

const classifyRuntimeError = (error) => {
    const message = normalizeText(error?.message || String(error));
    if (/timed out|429|503|temporarily|overloaded|rate/i.test(message)) return { class: "soft", message };
    if (/no searchable sources|no readable source content|unsupported|not configured/i.test(message)) return { class: "partial", message };
    return { class: "critical", message };
};

const withRetry = async (worker, retries = 2, baseDelayMs = 400) => {
    let lastError;
    for (let attempt = 0; attempt <= retries; attempt += 1) {
        try {
            return await worker(attempt);
        } catch (error) {
            lastError = error;
            const classified = classifyRuntimeError(error);
            if (classified.class !== "soft" || attempt >= retries) {
                throw error;
            }
            await sleep(baseDelayMs * (attempt + 1));
        }
    }
    throw lastError;
};

const runConcurrent = async (items, limit, worker) => {
    const values = Array.isArray(items) ? items : [];
    const concurrency = Math.max(1, limit || 1);
    const results = new Array(values.length);
    let cursor = 0;

    const next = async () => {
        while (cursor < values.length) {
            const current = cursor;
            cursor += 1;
            results[current] = await worker(values[current], current);
        }
    };

    await Promise.all(Array.from({ length: Math.min(concurrency, values.length) }, next));
    return results;
};

const collectSuccessfulResults = async (items, target, limit, worker) => {
    const values = Array.isArray(items) ? items : [];
    const successTarget = Math.max(1, Number(target) || values.length || 1);
    const results = [];
    let cursor = 0;

    const next = async () => {
        while (results.length < successTarget) {
            const current = cursor;
            if (current >= values.length) return;
            cursor += 1;
            const value = await worker(values[current], current);
            if (value) results.push(value);
        }
    };

    await Promise.all(Array.from({ length: Math.min(Math.max(1, limit || 1), values.length) }, next));
    return results.slice(0, successTarget);
};

const standardizeSearchResult = (item = {}, provider = "web") => {
    const url = canonicalizeSourceUrl(item.url || item.link || item.html_url || item.htmlUrl || item.doi_url || "");
    if (!url) return null;
    return {
        title: normalizeText(item.title || item.name || item.full_name || item.repository || url),
        url,
        description: normalizeText(item.description || item.snippet || item.summary || item.abstract || ""),
        date: item.date || item.published_at || item.updated_at || item.year || null,
        source: normalizeText(item.source || provider) || provider,
        providerCount: Number(item.providerCount || 0),
        providers: toArray(item.providers).filter(Boolean),
        queryVariant: normalizeText(item.queryVariant),
    };
};

const fetchJson = async (url, options = {}, timeoutMs = DEFAULT_TIMEOUT_MS) => withTimeout(async (signal) => {
    const response = await fetch(url, {
        ...options,
        signal,
        headers: {
            "User-Agent": USER_AGENT,
            ...(options.headers || {}),
        },
    });
    if (!response.ok) {
        const detail = normalizeText(await response.text().catch(() => ""));
        throw new Error(`HTTP ${response.status}${detail ? `: ${detail.slice(0, 180)}` : ""}`);
    }
    return response.json();
}, timeoutMs);

const searchGitHubProvider = async (query, limit = DEFAULT_SEARCH_RESULT_LIMIT) => {
    const normalizedQuery = normalizeText(query);
    if (!normalizedQuery) return [];
    const apiUrl = new URL("https://api.github.com/search/repositories");
    apiUrl.searchParams.set("q", normalizedQuery);
    apiUrl.searchParams.set("per_page", String(Math.min(limit, 10)));
    apiUrl.searchParams.set("sort", "stars");
    apiUrl.searchParams.set("order", "desc");

    const token = normalizeText(process.env.GITHUB_TOKEN);
    const data = await fetchJson(apiUrl.toString(), {
        headers: {
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
            Accept: "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    }, 9000).catch(() => null);

    const items = toArray(data?.items);
    return items.map((item) => standardizeSearchResult({
        title: item.full_name,
        url: item.html_url,
        description: item.description,
        date: item.updated_at,
        source: "github",
    }, "github")).filter(Boolean);
};

const searchConfiguredJsonProvider = async (providerName, query, envPrefix, limit = DEFAULT_SEARCH_RESULT_LIMIT) => {
    const baseUrl = normalizeText(process.env[`${envPrefix}_SEARCH_URL`]);
    if (!baseUrl) return [];
    const apiKey = normalizeText(process.env[`${envPrefix}_API_KEY`]);
    const apiKeyHeader = normalizeText(process.env[`${envPrefix}_API_KEY_HEADER`] || "Authorization");
    const apiKeyPrefix = normalizeText(process.env[`${envPrefix}_API_KEY_PREFIX`]);
    const queryParam = normalizeText(process.env[`${envPrefix}_QUERY_PARAM`] || "query");
    const limitParam = normalizeText(process.env[`${envPrefix}_LIMIT_PARAM`] || "limit");

    const url = new URL(baseUrl);
    url.searchParams.set(queryParam, query);
    url.searchParams.set(limitParam, String(Math.min(limit, 10)));

    const data = await fetchJson(url.toString(), {
        headers: apiKey ? {
            [apiKeyHeader]: apiKeyPrefix ? `${apiKeyPrefix} ${apiKey}` : apiKey,
        } : {},
    }, 9000).catch(() => null);

    const items = toArray(data?.results || data?.items || data?.data || data?.documents);
    return items.map((item) => standardizeSearchResult({
        title: item.title || item.name,
        url: item.url || item.link || item.html_url || item.id,
        description: item.description || item.abstract || item.summary || item.snippet,
        date: item.date || item.year || item.updated_at || item.published_at,
        source: providerName,
    }, providerName)).filter(Boolean);
};

const extractDoi = (value = "") => {
    const match = String(value || "").match(/\b10\.\d{4,9}\/[-._;()/:a-z0-9]+\b/i);
    return match ? match[0] : "";
};

const resolveUnpaywallLinks = async (sources = []) => {
    const email = normalizeText(process.env.UNPAYWALL_EMAIL);
    if (!email) return [];

    const results = await runConcurrent(sources.slice(0, 12), 4, async (source) => {
        const doi = extractDoi(source.url) || extractDoi(source.title) || extractDoi(source.description);
        if (!doi) return null;

        const url = new URL(`https://api.unpaywall.org/v2/${encodeURIComponent(doi)}`);
        url.searchParams.set("email", email);
        const data = await fetchJson(url.toString(), {}, 9000).catch(() => null);
        const oaLocation = data?.best_oa_location || data?.oa_locations?.[0] || null;
        if (!oaLocation?.url) return null;
        return {
            targetUrl: canonicalizeSourceUrl(source.url),
            openAccessUrl: canonicalizeSourceUrl(oaLocation.url),
            license: normalizeText(oaLocation.license || ""),
            version: normalizeText(oaLocation.version || ""),
            source: "unpaywall",
        };
    });

    return results.filter(Boolean);
};

const lookupRetractionSignals = async (sources = []) => {
    const baseUrl = normalizeText(process.env.RETRACTION_WATCH_SEARCH_URL);
    if (!baseUrl) return [];

    const results = await runConcurrent(sources.slice(0, 12), 3, async (source) => {
        const title = normalizeText(source.title);
        if (!title) return null;
        const url = new URL(baseUrl);
        url.searchParams.set("query", title);
        const data = await fetchJson(url.toString(), {}, 9000).catch(() => null);
        const match = toArray(data?.results || data?.items || data?.data)[0];
        if (!match) return null;
        return {
            targetUrl: canonicalizeSourceUrl(source.url),
            retracted: Boolean(match.retracted ?? true),
            reason: normalizeText(match.reason || match.retraction_reason || ""),
            source: "retraction_watch",
        };
    });

    return results.filter(Boolean);
};

const analyzeGitHubRepo = async (repoUrl = "") => {
    const match = normalizeText(repoUrl).match(/^https?:\/\/github\.com\/([^/]+)\/([^/#?]+)/i);
    if (!match) return null;

    const owner = match[1];
    const repo = match[2].replace(/\.git$/i, "");
    const token = normalizeText(process.env.GITHUB_TOKEN);
    const headers = {
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
    };

    const repoMeta = await fetchJson(`https://api.github.com/repos/${owner}/${repo}`, { headers }, 9000).catch(() => null);
    if (!repoMeta) return null;
    const readme = await fetchJson(`https://api.github.com/repos/${owner}/${repo}/readme`, { headers }, 9000).catch(() => null);
    const contents = await fetchJson(`https://api.github.com/repos/${owner}/${repo}/contents`, { headers }, 9000).catch(() => null);
    const topLevelNames = toArray(contents).map((item) => normalizeText(item?.name).toLowerCase()).filter(Boolean);

    const score = (
        (repoMeta.description ? 1 : 0)
        + (readme?.content ? 1 : 0)
        + (topLevelNames.some((name) => /requirements|package\.json|environment\.yml|pyproject|cargo\.toml/.test(name)) ? 1 : 0)
        + (topLevelNames.some((name) => /license/.test(name)) ? 1 : 0)
        + (repoMeta.homepage ? 1 : 0)
    );

    return {
        repoUrl: `https://github.com/${owner}/${repo}`,
        stars: Number(repoMeta.stargazers_count || 0),
        forks: Number(repoMeta.forks_count || 0),
        language: normalizeText(repoMeta.language),
        score: clamp(score, 0, 5, 0),
        rubric: {
            hasDescription: Boolean(repoMeta.description),
            hasReadme: Boolean(readme?.content),
            hasDependencyManifest: topLevelNames.some((name) => /requirements|package\.json|environment\.yml|pyproject|cargo\.toml/.test(name)),
            hasLicense: topLevelNames.some((name) => /license/.test(name)),
            hasHomepage: Boolean(repoMeta.homepage),
        },
    };
};

const buildRuntimeMeta = (state) => ({
    frameworkVersion: state.plan?.frameworkVersion || RESEARCH_FRAMEWORK_VERSION,
    domain: state.plan?.domain || null,
    scope: state.plan?.scope || null,
    outputMode: state.plan?.outputMode || null,
    intentConfidence: state.plan?.intentConfidence || null,
    pareto: state.plan?.pareto || null,
    continuity: state.plan?.continuity || null,
    safety: state.safety || state.plan?.safety || null,
    dag: state.plan?.dag || null,
    dagSummary: summarizeDag(state.plan?.dag || {}),
    queryMatrix: state.plan?.queryMatrix || null,
    refinementBudget: state.plan?.refinementBudget || 3,
    searchCount: toArray(state.plan?.searchQueries).length,
    rankedSites: toArray(state.filteredSources).length || toArray(state.searchInventory).length,
    fetchPlanned: Number(state.fetchPlan?.length || 0),
    fetchedSites: Number(state.evidenceEntries?.length || 0),
    fetchAttempts: Number(state.fetchAttempts || 0),
    synthesisWorkers: Number(state.synthesisWorkers || 0),
    tribunal: state.tribunal || null,
    convergence: state.convergence || null,
    uncertainty: state.uncertainty || [],
    subagents: state.plan?.subagents || [],
    runId: state.runId,
    status: state.status,
    checkpoints: toArray(state.checkpoints).length,
};

const createDefaultDeps = () => ({
    searchWeb: async (query, limit) => parseSearchResults(await webSearchHandler({ query, maxResults: limit })),
    fetchUrl: async (url, maxChars = RAG_FETCH_MAX_CHARS) => parseFetchedPayload(await webFetchHandler({
        url,
        format: "markdown",
        max_chars: maxChars,
    })),
    textModel: async ({ system, user }) => {
        const result = await runLiteHostChat({
            model: "nub-agent",
            stream: false,
            use_tools: false,
            research_mode: true,
            messages: [
                { role: "system", content: system },
                { role: "user", content: user },
            ],
        });
        return normalizeText(result?.reply?.content || "");
    },
});

const safePersist = async (worker, fallback = null) => {
    try {
        return await worker();
    } catch {
        return fallback;
    }
};

const parseExtractionPayload = (text, fallbackEntries = []) => {
    const parsed = parseJsonObject(text);
    if (!parsed) {
        return {
            summaries: fallbackEntries.map((entry) => normalizeText((entry.content || entry.source?.description || "").slice(0, 260))).filter(Boolean).slice(0, 6),
            claims: buildQuantitativeClaims(fallbackEntries),
            concepts: extractQueryTerms(fallbackEntries.map((entry) => `${entry.source?.title || ""} ${entry.source?.description || ""}`).join(" "), 16),
            methods: [],
            debate_axes: [],
            authors: [],
            repo_links: uniqueBy(fallbackEntries.flatMap((entry) => extractRepoLinks(entry.content || "")), (item) => item),
            supplementary_links: uniqueBy(fallbackEntries.flatMap((entry) => extractSupplementaryLinks(entry.content || "")), (item) => item),
            funding_signals: uniqueBy(fallbackEntries.flatMap((entry) => extractFundingSignals(entry.content || "")), (item) => item),
        };
    }

    return {
        summaries: toArray(parsed.summaries).map((item) => normalizeText(item?.text || item)).filter(Boolean),
        claims: toArray(parsed.claims).map((item) => ({
            text: normalizeText(item?.text || item),
            citationIndex: toPositiveInteger(item?.citationIndex),
        })).filter((item) => item.text),
        concepts: toArray(parsed.concepts).map((item) => normalizeText(item?.label || item)).filter(Boolean),
        methods: toArray(parsed.methods).map((item) => normalizeText(item?.label || item)).filter(Boolean),
        debate_axes: toArray(parsed.debate_axes || parsed.debateAxes).map((item) => normalizeText(item?.label || item)).filter(Boolean),
        authors: toArray(parsed.authors).map((item) => normalizeText(item?.name || item)).filter(Boolean),
        repo_links: uniqueBy(toArray(parsed.repo_links || parsed.repoLinks).map((item) => normalizeText(item?.url || item)).filter(Boolean), (item) => item),
        supplementary_links: uniqueBy(toArray(parsed.supplementary_links || parsed.supplementaryLinks).map((item) => normalizeText(item?.url || item)).filter(Boolean), (item) => item),
        funding_signals: uniqueBy(toArray(parsed.funding_signals || parsed.fundingSignals).map((item) => normalizeText(item?.label || item)).filter(Boolean), (item) => item),
    };
};

const buildConvergence = (draftHistory = [], tribunal = {}) => {
    const latest = draftHistory[draftHistory.length - 1] || "";
    const previous = draftHistory[draftHistory.length - 2] || "";
    const stability = previous ? jaccardSimilarity(previous, latest) : 0.92;
    const coverage = tribunal?.critics?.coverage ?? 0.76;
    const internalConsistency = tribunal?.critics?.internal_consistency ?? 0.78;
    const evidenceCoverageDelta = Number((1 - coverage).toFixed(2));
    const residualUncertainty = Number((Math.max(0.08, (1 - stability) + ((1 - internalConsistency) * 0.35))).toFixed(2));
    return {
        iterations: Math.max(1, Number(tribunal?.refinement_cycles || draftHistory.length || 1)),
        stability_score: Number(clamp01(stability).toFixed(2)),
        evidence_coverage_delta: evidenceCoverageDelta,
        residual_uncertainty: residualUncertainty,
        stop_condition: stability < 0.95 && evidenceCoverageDelta > 0.03 ? "residual_disagreement" : "stability_reached",
    };
};

const executeDag = async (nodes, state, runtimeApi) => {
    const statuses = new Map();
    const active = new Set();
    for (const node of nodes) statuses.set(node.id, "pending");

    while ([...statuses.values()].some((status) => status === "pending")) {
        const ready = nodes
            .filter((node) => statuses.get(node.id) === "pending")
            .filter((node) => node.dependsOn.every((dependency) => statuses.get(dependency) === "completed"))
            .sort((left, right) => right.priority - left.priority);

        if (!ready.length) {
            throw new Error("DAG scheduler deadlocked.");
        }

        const parallelBatch = ready.filter((node) => node.parallelizable);
        const sequentialNode = ready.find((node) => !node.parallelizable);
        const batch = parallelBatch.length ? parallelBatch : [sequentialNode];

        await Promise.all(batch.map(async (node) => {
            active.add(node.id);
            statuses.set(node.id, "running");

            const controls = await runtimeApi.consumeControls();
            if (controls.length) {
                state.plan = applySteeringCommands(state.plan, controls);
                await runtimeApi.emit({
                    phase: "Human-in-the-Loop Control",
                    title: "Steering Commands Applied",
                    statusText: `${controls.length} steering command(s) applied and the research DAG was recompiled.`,
                    payload: { commands: controls, dagSummary: summarizeDag(state.plan.dag || {}) },
                });
            }

            try {
                const update = await withRetry(() => node.run(state, runtimeApi), node.classification === "soft" ? 2 : 1);
                Object.assign(state, update || {});
                statuses.set(node.id, "completed");
            } catch (error) {
                const classified = classifyRuntimeError(error);
                await runtimeApi.emit({
                    phase: node.label,
                    title: classified.class === "critical" ? "Critical Failure" : "Degraded Execution",
                    statusText: `${node.label} ${classified.class === "partial" ? "continued in degraded mode" : "failed"}. ${classified.message}`,
                    payload: { error: classified.message, errorClass: classified.class },
                });
                if (classified.class === "partial") {
                    statuses.set(node.id, "completed");
                } else {
                    throw error;
                }
            } finally {
                active.delete(node.id);
            }
        }));
    }

    return state;
};

const runResearchRuntime = async (input = {}, options = {}) => {
    const deps = {
        ...createDefaultDeps(),
        ...(options.deps || {}),
    };
    const query = normalizeText(input.query || input.prompt);
    const attachments = toArray(input.attachments);
    if (!query && !attachments.length) {
        throw new Error("Provide a query or attachments.");
    }

    const scope = options.scope || resolveResearchScope(options.req);
    const memory = options.memory || { episodes: [] };
    let plan = compileResearchPlan({
        query,
        attachments: attachments.length,
        depthPreference: input.depthPreference,
        maxQueries: clamp(input.maxQueries, 2, 8, 6),
        episodes: toArray(memory.episodes),
        refinementBudget: input.refinementBudget,
    });
    if (toArray(input.commands).length) {
        plan = applySteeringCommands(plan, input.commands);
    }

    const runRecord = await safePersist(() => createResearchRun({
        runId: input.runId,
        scopeKey: scope.stateKey,
        query,
        attachments,
        plan,
        status: "running",
    }), {
        runId: input.runId || `rr_ephemeral_${Date.now()}`,
        scopeKey: scope.stateKey,
        query,
        attachments,
        plan,
        status: "running",
    });

    const state = {
        runId: runRecord.runId,
        scopeKey: scope.stateKey,
        query,
        attachments,
        plan,
        status: "running",
        checkpoints: [],
        warnings: [],
        searchInventory: [],
        filteredSources: [],
        fetchPlan: [],
        evidenceEntries: [],
        extractedEvidence: [],
        synthesisWorkers: 0,
        fetchAttempts: 0,
        output: null,
        final: null,
    };

    const onEvent = typeof options.onEvent === "function" ? options.onEvent : null;
    const runtimeApi = {
        consumeControls: async () => (typeof options.consumeControls === "function"
            ? options.consumeControls(state.runId)
            : consumeSteeringCommands(state.runId)),
        emit: async (checkpoint) => {
            const normalizedCheckpoint = {
                id: `cp_${Date.now()}_${state.checkpoints.length + 1}`,
                type: "checkpoint",
                createdAt: new Date().toISOString(),
                ...checkpoint,
            };
            state.checkpoints.push(normalizedCheckpoint);
            await safePersist(() => appendResearchCheckpoint(state.runId, normalizedCheckpoint));
            if (onEvent) {
                await onEvent({
                    type: "checkpoint",
                    runId: state.runId,
                    checkpoint: normalizedCheckpoint,
                    researchMeta: buildRuntimeMeta(state),
                });
            }
        },
    };

    await runtimeApi.emit({
        phase: "Cognitive Command Layer",
        title: "Research Plan Ready",
        statusText: `Compiled Research Framework v${plan.frameworkVersion} with ${plan.searchQueries.length} search lane(s).`,
        payload: {
            dagSummary: summarizeDag(plan.dag || {}),
            hypotheses: plan.hypotheses,
            searchQueries: plan.searchQueries,
        },
    });

    const nodes = [
        {
            id: "cognitiveCommandLayer",
            label: "Cognitive Command Layer",
            priority: 1,
            dependsOn: [],
            parallelizable: false,
            run: async () => ({
                status: "planning",
            }),
        },
        {
            id: "activeSafetyAndEthics",
            label: "Active Safety & Ethics",
            priority: 0.96,
            dependsOn: ["cognitiveCommandLayer"],
            parallelizable: true,
            run: async (currentState, api) => {
                const dualUse = Boolean(currentState.plan?.safety?.checks?.find((item) => item.id === "dual_use_flagging" && item.active));
                const requiresConfirmation = dualUse && input.confirm_dual_use !== true;
                const warnings = [...currentState.warnings];
                if (requiresConfirmation) {
                    warnings.push("Dual-use signal detected. Deep extraction ran in reduced depth until confirmed.");
                }
                const safety = {
                    ...currentState.plan.safety,
                    requiresConfirmation,
                    causalRiskChains: dualUse
                        ? [{ chain: "Technique -> misuse vector -> impact severity", score: 0.78 }]
                        : [],
                };
                await api.emit({
                    phase: "Active Safety & Ethics",
                    title: "Safety Checks Applied",
                    statusText: `${safety.activeCount} safety check(s) active${requiresConfirmation ? "; depth reduced pending confirmation" : ""}.`,
                    payload: safety,
                });
                return { safety, warnings };
            },
        },
        {
            id: "adversarialQueryForge",
            label: "Adversarial Query Forge",
            priority: 0.94,
            dependsOn: ["cognitiveCommandLayer"],
            parallelizable: false,
            run: async (currentState, api) => {
                await api.emit({
                    phase: "Adversarial Query Forge",
                    title: "Hypotheses Forged",
                    statusText: `Generated ${currentState.plan.hypotheses.length} hypothesis lane(s) with ${currentState.plan.queryMatrix.counterHypotheses.length} counter-hypothesis lane(s).`,
                    payload: {
                        hypotheses: currentState.plan.hypotheses,
                        queryVersions: currentState.plan.queryMatrix.versions,
                    },
                });
                return {};
            },
        },
        {
            id: "intelligentCrawlerMesh",
            label: "Intelligent Crawler Mesh",
            priority: 0.9,
            dependsOn: ["adversarialQueryForge"],
            parallelizable: true,
            run: async (currentState, api) => {
                if (!currentState.query) {
                    await api.emit({
                        phase: "Intelligent Crawler Mesh",
                        title: "Attachment-Only Run",
                        statusText: "No web retrieval was needed because the run is attachment-only.",
                        payload: { inventoryCount: 0, topSources: [] },
                    });
                    return { searchInventory: [], temporalTrend: {} };
                }
                const queries = currentState.plan.searchQueries;
                const queryResults = await Promise.all(queries.map(async (variant) => {
                    const [web, github, papersWithCode, ieee, acm, jstor] = await Promise.all([
                        deps.searchWeb(variant, DEFAULT_SEARCH_RESULT_LIMIT).catch(() => []),
                        searchGitHubProvider(variant, 4).catch(() => []),
                        searchConfiguredJsonProvider("papers_with_code", variant, "PAPERS_WITH_CODE", 4).catch(() => []),
                        searchConfiguredJsonProvider("ieee_xplore", variant, "IEEE_XPLORE", 4).catch(() => []),
                        searchConfiguredJsonProvider("acm_digital_library", variant, "ACM_DL", 4).catch(() => []),
                        searchConfiguredJsonProvider("jstor", variant, "JSTOR", 4).catch(() => []),
                    ]);
                    return [...web, ...github, ...papersWithCode, ...ieee, ...acm, ...jstor]
                        .map((source) => standardizeSearchResult({ ...source, queryVariant: variant }, source.source))
                        .filter(Boolean);
                }));

                const merged = mergeSourcesByCanonicalUrl(queryResults.flat(), { limit: 96 });
                const openAccessMatches = await resolveUnpaywallLinks(merged).catch(() => []);
                const openAccessMap = new Map(openAccessMatches.map((item) => [item.targetUrl, item]));
                const enriched = merged.map((source) => {
                    const oa = openAccessMap.get(canonicalizeSourceUrl(source.url));
                    return {
                        ...source,
                        openAccessUrl: oa?.openAccessUrl || "",
                        openAccessLicense: oa?.license || "",
                    };
                });
                const ranked = rankSourcesWithRag(currentState.query, enriched);
                const yearHistogram = {};
                for (const source of ranked) {
                    const year = String(source.date || "").slice(0, 4);
                    if (/^\d{4}$/.test(year)) yearHistogram[year] = (yearHistogram[year] || 0) + 1;
                }
                await api.emit({
                    phase: "Intelligent Crawler Mesh",
                    title: "Paper Inventory Ready",
                    statusText: `Collected ${ranked.length} candidate source(s) across web and specialist providers.`,
                    payload: {
                        inventoryCount: ranked.length,
                        topSources: ranked.slice(0, 8),
                        temporalTrend: yearHistogram,
                    },
                });
                return {
                    searchInventory: ranked,
                    temporalTrend: yearHistogram,
                };
            },
        },
        {
            id: "tieredEpistemicFilter",
            label: "Tiered Epistemic Filter",
            priority: 0.88,
            dependsOn: ["intelligentCrawlerMesh", "activeSafetyAndEthics"],
            parallelizable: true,
            run: async (currentState, api) => {
                if (!currentState.searchInventory.length) {
                    await api.emit({
                        phase: "Tiered Epistemic Filter",
                        title: "Attachment Evidence Passed Through",
                        statusText: "Tiered filtering skipped because no web inventory was collected.",
                        payload: { tiers: { core: 0, supporting: 0, peripheral: 0, discard: 0 }, inventory: [] },
                    });
                    return { filteredSources: [] };
                }
                const retractionMatches = await lookupRetractionSignals(currentState.searchInventory).catch(() => []);
                const retractionMap = new Map(retractionMatches.map((item) => [item.targetUrl, item]));
                const filteredSources = currentState.searchInventory.map((source, index) => {
                    const overlap = getQueryOverlap(currentState.query, `${source.title} ${source.description}`);
                    const freshness = computeFreshnessWeight(source.date, currentState.plan.domain.recencyHalfLifeYears);
                    const authority = clamp01(domainAuthorityBoost(source.url) / 2.25);
                    const retraction = retractionMap.get(canonicalizeSourceUrl(source.url));
                    const samplePenalty = /\bn\s*=\s*[1-9]\d?\b/i.test(source.description || "") ? 0.12 : 0;
                    const score = clamp01((overlap * 0.44) + (freshness * 0.18) + (authority * 0.22) + 0.16 - samplePenalty - (retraction?.retracted ? 0.4 : 0));
                    const tier = score >= 0.8 ? "core" : score >= 0.55 ? "supporting" : score >= 0.35 ? "peripheral" : "discard";
                    return {
                        ...source,
                        rank: index + 1,
                        tier,
                        epistemicScore: Number(score.toFixed(2)),
                        retractionReason: retraction?.reason || "",
                        retracted: Boolean(retraction?.retracted),
                    };
                });

                const kept = filteredSources.filter((source) => source.tier !== "discard");
                await api.emit({
                    phase: "Tiered Epistemic Filter",
                    title: "Evidence Gate Complete",
                    statusText: `${kept.length} source(s) survived the epistemic filter; ${filteredSources.length - kept.length} discarded or quarantined.`,
                    payload: {
                        tiers: {
                            core: kept.filter((item) => item.tier === "core").length,
                            supporting: kept.filter((item) => item.tier === "supporting").length,
                            peripheral: kept.filter((item) => item.tier === "peripheral").length,
                            discard: filteredSources.filter((item) => item.tier === "discard").length,
                        },
                        inventory: kept.slice(0, 12),
                    },
                });
                return { filteredSources };
            },
        },
        {
            id: "deepComprehensionEngine",
            label: "Deep Comprehension Engine",
            priority: 0.86,
            dependsOn: ["tieredEpistemicFilter", "activeSafetyAndEthics"],
            parallelizable: true,
            run: async (currentState, api) => {
                const candidateSources = currentState.filteredSources.filter((source) => source.tier === "core" || source.tier === "supporting");
                const fetchPlan = selectSourcesForFetch(currentState.query || "attachment evidence", candidateSources, {
                    limit: currentState.safety?.requiresConfirmation ? 6 : FETCH_TARGET,
                    perDomainLimit: 2,
                }).map((source, index) => ({
                    ...source,
                    citationIndex: index + 1,
                }));
                const attachmentEvidenceEntries = buildAttachmentEvidenceEntries(currentState.attachments, fetchPlan.length + 1);

                let fetchAttempts = 0;
                const evidenceEntries = await collectSuccessfulResults(fetchPlan, fetchPlan.length, FETCH_CONCURRENCY, async (source) => {
                    fetchAttempts += 1;
                    const targetUrl = source.openAccessUrl || source.url;
                    const fetched = await deps.fetchUrl(targetUrl, currentState.safety?.requiresConfirmation ? 12000 : RAG_FETCH_MAX_CHARS).catch(() => null);
                    if (!fetched || fetched.error) return null;
                    const content = stripFetchMeta(fetched.content || "").trim();
                    if (!content && !normalizeText(source.description)) return null;
                    return {
                        source: {
                            ...source,
                            url: canonicalizeSourceUrl(fetched.finalUrl || targetUrl) || targetUrl,
                        title: fetched.title || source.title,
                        description: source.description || fetched.description || "",
                        date: source.date || fetched.publishedTime || null,
                        via: fetched.via || source.source || "runtime",
                        citationIndex: source.citationIndex,
                        },
                        content: content || source.description || "",
                    };
                });

                const rankedEvidenceEntries = rankEvidenceEntriesForQuery(currentState.query || "attachment evidence", [...attachmentEvidenceEntries, ...evidenceEntries]);
                const evidenceChunks = [];
                const chunkSize = Math.max(1, Math.ceil(rankedEvidenceEntries.length / SYNTHESIS_SWARM_SIZE));
                for (let index = 0; index < rankedEvidenceEntries.length; index += chunkSize) {
                    evidenceChunks.push(rankedEvidenceEntries.slice(index, index + chunkSize));
                }

                const extractedChunks = await Promise.all(evidenceChunks.map(async (chunk, index) => {
                    const evidenceBlock = chunk.map((entry) => buildRagEvidenceBlock(entry, currentState.query)).join("\n\n---\n\n");
                    const prompt = [
                        `Research plan: ${currentState.plan.outputMode.label}, ${currentState.plan.scope.label}, ${currentState.plan.domain.label}.`,
                        `Return strict JSON with keys summaries, claims, concepts, methods, debate_axes, authors, repo_links, supplementary_links, funding_signals.`,
                        "Every claim should stay grounded in the supplied evidence only.",
                        "",
                        evidenceBlock,
                    ].join("\n\n");
                    const response = await deps.textModel({
                        system: "You are the Deep Comprehension Engine. Return JSON only.",
                        user: prompt,
                    }).catch(() => "");
                    return parseExtractionPayload(response, chunk);
                }));

                const repoLinks = uniqueBy(extractedChunks.flatMap((chunk) => chunk.repo_links).concat(
                    rankedEvidenceEntries.flatMap((entry) => extractRepoLinks(entry.content || "")),
                ), (item) => item).slice(0, 6);
                const supplementaryLinks = uniqueBy(extractedChunks.flatMap((chunk) => chunk.supplementary_links).concat(
                    rankedEvidenceEntries.flatMap((entry) => extractSupplementaryLinks(entry.content || "")),
                ), (item) => item).slice(0, 6);
                const repoAnalyses = (await Promise.all(repoLinks.map((url) => analyzeGitHubRepo(url).catch(() => null)))).filter(Boolean);
                const supplementaryMaterials = await Promise.all(supplementaryLinks.slice(0, 2).map(async (url) => {
                    const fetched = await deps.fetchUrl(url, 8000).catch(() => null);
                    return fetched && !fetched.error
                        ? { url, title: fetched.title || url, content: normalizeText(fetched.content || "").slice(0, 2400) }
                        : null;
                }));
                const claimPool = extractedChunks.flatMap((chunk) => chunk.claims).concat(buildQuantitativeClaims(rankedEvidenceEntries));
                const metaAnalysis = buildCrossPaperStatSummary(claimPool);
                const extractedEvidence = extractedChunks.map((chunk, index) => ({
                    chunkId: index + 1,
                    summaries: chunk.summaries,
                    concepts: chunk.concepts,
                    methods: chunk.methods,
                    debateAxes: chunk.debate_axes,
                }));

                await api.emit({
                    phase: "Deep Comprehension Engine",
                    title: "Paper Summaries Ready",
                    statusText: `Extracted structured evidence from ${rankedEvidenceEntries.length} fetched source(s).`,
                    payload: {
                        summaries: extractedChunks.flatMap((chunk) => chunk.summaries).slice(0, 8),
                        repoAnalyses,
                        supplementaryMaterials: supplementaryMaterials.filter(Boolean).map((item) => ({ url: item.url, title: item.title })),
                        metaAnalysis,
                    },
                });

                return {
                    fetchPlan,
                    fetchAttempts,
                    evidenceEntries: rankedEvidenceEntries,
                    extractedEvidence,
                    extractedChunks,
                    claimPool,
                    repoAnalyses,
                    supplementaryMaterials: supplementaryMaterials.filter(Boolean),
                    metaAnalysis,
                };
            },
        },
        {
            id: "dialecticalSynthesisEngine",
            label: "Dialectical Synthesis Engine",
            priority: 0.84,
            dependsOn: ["deepComprehensionEngine"],
            parallelizable: false,
            run: async (currentState, api) => {
                const evidenceChunks = [];
                const chunkSize = Math.max(1, Math.ceil(currentState.evidenceEntries.length / SYNTHESIS_SWARM_SIZE));
                for (let index = 0; index < currentState.evidenceEntries.length; index += chunkSize) {
                    evidenceChunks.push(currentState.evidenceEntries.slice(index, index + chunkSize));
                }

                const positionMaps = (await Promise.all(evidenceChunks.map(async (chunk, index) => {
                    const evidenceBlock = chunk.map((entry) => buildRagEvidenceBlock(entry, currentState.query)).join("\n\n---\n\n");
                    const response = await deps.textModel({
                        system: "You are the Position Mapping stage. Use only supplied evidence. Return markdown with sections for Dominant Position, Counter Position, Key Claims, Quantitative Signals, and Open Gaps.",
                        user: `Query: ${currentState.query}\nOutput mode: ${currentState.plan.outputMode.label}\n\n${evidenceBlock}`,
                    }).catch(() => "");
                    return normalizeText(response) || `### Position Map ${index + 1}\n- No structured position map generated.`;
                }))).filter(Boolean);

                const sourceIndex = currentState.evidenceEntries
                    .map((entry) => `[${entry.source.citationIndex}] ${entry.source.title} — ${entry.source.url}`)
                    .join("\n");
                const thesisText = await deps.textModel({
                    system: "You are ThesisAgent. Build the strongest case for the dominant view using only the supplied position maps and source index. Use [n] citations only.",
                    user: `Query: ${currentState.query}\n\nSource index:\n${sourceIndex}\n\nPosition maps:\n${positionMaps.join("\n\n")}`,
                }).catch(() => "");
                const antithesisText = await deps.textModel({
                    system: "You are AntithesisAgent. Build the strongest counter-case using the supplied position maps and source index. Use [n] citations only.",
                    user: `Query: ${currentState.query}\nCounter-hypotheses:\n${toArray(currentState.plan.queryMatrix?.counterHypotheses).map((item) => `- ${item}`).join("\n")}\n\nSource index:\n${sourceIndex}\n\nPosition maps:\n${positionMaps.join("\n\n")}`,
                }).catch(() => "");
                const draft = await deps.textModel({
                    system: `You are SynthesisMediator and NarrativeArchitect. Produce a ${currentState.plan.outputMode.label} report. Use [n] citations only. Start with a single H1 title. Include residual uncertainty and end with ## Sources used.`,
                    user: `Research plan: ${currentState.plan.scope.label} / ${currentState.plan.outputMode.label} / ${currentState.plan.domain.label}\n\nSource index:\n${sourceIndex}\n\nPosition maps:\n${positionMaps.join("\n\n")}\n\nThesis:\n${thesisText || "No thesis generated."}\n\nAntithesis:\n${antithesisText || "No antithesis generated."}`,
                }).catch(() => "");

                const resolvedDraft = normalizeText(draft) ? draft : buildFallbackNarrative(currentState.query, currentState.evidenceEntries, currentState.plan);

                await api.emit({
                    phase: "Dialectical Synthesis Engine",
                    title: "Draft Synthesis Ready",
                    statusText: `Built ${positionMaps.length} position map(s) and completed thesis/antithesis reconciliation.`,
                    payload: {
                        positionMaps: positionMaps.length,
                        thesisPreview: normalizeText(thesisText).slice(0, 320),
                        antithesisPreview: normalizeText(antithesisText).slice(0, 320),
                    },
                });

                return {
                    positionMaps,
                    thesisText,
                    antithesisText,
                    draftHistory: [resolvedDraft],
                    draft: resolvedDraft,
                    synthesisWorkers: evidenceChunks.length + 3,
                };
            },
        },
        {
            id: "recursiveSelfImprovementLoop",
            label: "Recursive Self-Improvement Loop",
            priority: 0.82,
            dependsOn: ["dialecticalSynthesisEngine"],
            parallelizable: false,
            run: async (currentState, api) => {
                let draft = currentState.draft;
                const draftHistory = [...toArray(currentState.draftHistory)];
                let tribunal = {
                    refinement_budget: currentState.plan.refinementBudget,
                    refinement_cycles: 1,
                    targeted_dimension: "coverage",
                    critics: {
                        internal_consistency: 0.78,
                        coverage: 0.76,
                        user_goal_alignment: 0.8,
                    },
                };

                const sourceIndex = currentState.evidenceEntries
                    .map((entry) => `[${entry.source.citationIndex}] ${entry.source.title} — ${entry.source.url}`)
                    .join("\n");

                for (let cycle = 1; cycle <= currentState.plan.refinementBudget; cycle += 1) {
                    const critiqueRaw = await deps.textModel({
                        system: "You are the Quality Tribunal. Return strict JSON with internal_consistency, coverage, user_goal_alignment, targeted_dimension, and rewrite_brief.",
                        user: `Query: ${currentState.query}\nOutput mode: ${currentState.plan.outputMode.label}\nCounter hypotheses: ${toArray(currentState.plan.queryMatrix?.counterHypotheses).join("; ")}\nSource index:\n${sourceIndex}\n\nDraft:\n${draft}`,
                    }).catch(() => "");
                    const critique = parseJsonObject(critiqueRaw) || {};
                    const internalConsistency = clamp01(critique.internal_consistency ?? critique.critics?.internal_consistency ?? 0.8);
                    const coverage = clamp01(critique.coverage ?? critique.critics?.coverage ?? 0.78);
                    const alignment = clamp01(critique.user_goal_alignment ?? critique.critics?.user_goal_alignment ?? 0.82);
                    const targetedDimension = normalizeText(critique.targeted_dimension || critique.targetedDimension || "coverage") || "coverage";
                    tribunal = {
                        refinement_budget: currentState.plan.refinementBudget,
                        refinement_cycles: cycle,
                        targeted_dimension: targetedDimension,
                        critics: {
                            internal_consistency: Number(internalConsistency.toFixed(2)),
                            coverage: Number(coverage.toFixed(2)),
                            user_goal_alignment: Number(alignment.toFixed(2)),
                        },
                    };

                    const citations = extractCitationNumbers(draft);
                    const supportedSet = new Set(currentState.evidenceEntries.map((entry) => entry.source.citationIndex));
                    const inlineConstraintViolations = citations.filter((citation) => !supportedSet.has(citation));
                    const needsRewrite = (
                        internalConsistency < 0.8
                        || coverage < 0.78
                        || alignment < 0.8
                        || inlineConstraintViolations.length > 0
                    );

                    if (!needsRewrite || cycle >= currentState.plan.refinementBudget) {
                        break;
                    }

                    await api.emit({
                        phase: "Recursive Self-Improvement Loop",
                        title: "Tribunal Rewrite Requested",
                        statusText: `Cycle ${cycle}/${currentState.plan.refinementBudget} targeted ${targetedDimension.replace(/_/g, " ")}.`,
                        payload: {
                            tribunal,
                            inlineConstraintViolations,
                        },
                    });

                    const rewritten = await deps.textModel({
                        system: `You are NarrativeArchitect revising a draft. Improve ${targetedDimension.replace(/_/g, " ")} without weakening the other critic scores. Keep only [n] citations that exist in the source index.`,
                        user: `Source index:\n${sourceIndex}\n\nRewrite brief: ${normalizeText(critique.rewrite_brief || critique.rewriteBrief || "Strengthen evidence coverage, tighten unsupported claims, and improve fit to the requested output mode.")}\n\nCurrent draft:\n${draft}`,
                    }).catch(() => "");
                    if (normalizeText(rewritten)) {
                        draft = rewritten;
                        draftHistory.push(rewritten);
                    } else {
                        break;
                    }
                }

                const contradictionScore = estimateContradictionScore(currentState.positionMaps, currentState.thesisText, currentState.antithesisText);
                const convergence = buildConvergence(draftHistory, tribunal);
                const uncertainty = buildProbabilisticClaims(currentState.claimPool, tribunal, contradictionScore);

                await api.emit({
                    phase: "Recursive Self-Improvement Loop",
                    title: "Tribunal Complete",
                    statusText: `Completed ${tribunal.refinement_cycles} tribunal cycle(s); targeted ${tribunal.targeted_dimension.replace(/_/g, " ")}.`,
                    payload: {
                        tribunal,
                        convergence,
                    },
                });

                return {
                    draft,
                    draftHistory,
                    tribunal,
                    convergence,
                    uncertainty,
                };
            },
        },
        {
            id: "decisionIntelligenceLayer",
            label: "Decision Intelligence Layer",
            priority: 0.8,
            dependsOn: ["recursiveSelfImprovementLoop"],
            parallelizable: false,
            run: async (currentState, api) => {
                const decision = buildDecisionPayload(currentState.plan, currentState.tribunal, currentState.convergence, currentState.uncertainty);
                await api.emit({
                    phase: "Decision Intelligence Layer",
                    title: "Decision Payload Ready",
                    statusText: `Generated ${currentState.plan.outputMode.label} decision support metadata.`,
                    payload: decision,
                });
                return { decision };
            },
        },
        {
            id: "adaptiveDeliveryHub",
            label: "Adaptive Delivery Hub",
            priority: 0.78,
            dependsOn: ["decisionIntelligenceLayer"],
            parallelizable: false,
            run: async (currentState, api) => {
                const finalMarkdown = currentState.draft || buildFallbackNarrative(currentState.query, currentState.evidenceEntries, currentState.plan);
                const { heading, body } = extractAnswerParts(finalMarkdown);
                const sources = buildAttributedSourcesFromEvidence(finalMarkdown, currentState.evidenceEntries, 24);
                const datasetRows = toArray(currentState.claimPool).slice(0, 80).map((claim, index) => ({
                    row: index + 1,
                    claim: normalizeText(claim?.text),
                    citationIndex: claim?.citationIndex || "",
                    sourceTitle: normalizeText(claim?.sourceTitle),
                    sourceUrl: normalizeText(claim?.sourceUrl),
                }));
                const postmortem = {
                    query_type: currentState.plan.scope.id,
                    domain: currentState.plan.domain.label,
                    phases_that_degraded_score: [
                        currentState.tribunal?.targeted_dimension
                            ? `Recursive Self-Improvement Loop - ${currentState.tribunal.targeted_dimension}`
                            : "",
                    ].filter(Boolean),
                    prompt_patches_applied: [
                        currentState.tribunal?.targeted_dimension
                            ? `targeted rewrite for ${currentState.tribunal.targeted_dimension}`
                            : "",
                    ].filter(Boolean),
                    final_score_delta: Number(((currentState.tribunal?.critics?.coverage ?? 0.76) * 10).toFixed(0)),
                };
                const final = {
                    heading,
                    body,
                    markdown: finalMarkdown,
                    sources,
                    claims: currentState.claimPool,
                    concepts: uniqueBy(currentState.extractedChunks.flatMap((chunk) => chunk.concepts || []), (item) => item),
                    abstractions: [
                        {
                            pattern: currentState.metaAnalysis?.combined_effect_size != null
                                ? `Combined effect size trends toward ${currentState.metaAnalysis.combined_effect_size}`
                                : "Evidence remains primarily qualitative.",
                            evidence: `I²: ${currentState.metaAnalysis?.i_squared ?? "n/a"}`,
                        },
                    ],
                    repoAnalyses: currentState.repoAnalyses,
                    supplementaryMaterials: currentState.supplementaryMaterials,
                    metaAnalysis: currentState.metaAnalysis,
                    decision: currentState.decision,
                    uncertainty: currentState.uncertainty,
                    tribunal: currentState.tribunal,
                    convergence: currentState.convergence,
                    postmortem,
                    exports: {
                        markdown: finalMarkdown,
                        obsidian: buildObsidianMarkdown({ heading, body }, currentState.plan, currentState.runId),
                        slides: buildSlideDeckOutline(heading, body, currentState.checkpoints),
                        dataset_csv: buildCsv(datasetRows),
                        dataset_json: JSON.stringify(datasetRows, null, 2),
                        research_api: {
                            get: `/api/research?action=get&runId=${encodeURIComponent(currentState.runId)}`,
                            export: `/api/research?action=export&runId=${encodeURIComponent(currentState.runId)}`,
                        },
                    },
                };

                await api.emit({
                    phase: "Adaptive Delivery Hub",
                    title: "Final Report Ready",
                    statusText: "Final report, exports, and research API links are available.",
                    payload: {
                        heading,
                        sources: sources.length,
                    },
                });

                return { final };
            },
        },
    ];

    try {
        await executeDag(nodes, state, runtimeApi);
        state.status = "complete";
        await safePersist(() => updateResearchRun(state.runId, {
            status: "complete",
            outputs: {
                tribunal: state.tribunal,
                convergence: state.convergence,
            },
            final: state.final,
            updatedAt: new Date().toISOString(),
        }));
        await safePersist(() => appendResearchIndex(state.scopeKey, {
            runId: state.runId,
            query: state.query,
            status: "complete",
            domain: state.plan?.domain?.label,
            outputMode: state.plan?.outputMode?.label,
            updatedAt: new Date().toISOString(),
        }));
        await safePersist(() => updateResearchMemoryFromRun(state.scopeKey, {
            runId: state.runId,
            query: state.query,
            plan: state.plan,
            final: state.final,
            updatedAt: new Date().toISOString(),
        }));

        const responsePayload = {
            ok: true,
            runId: state.runId,
            status: "complete",
            query: state.query,
            plan: state.plan,
            checkpoints: state.checkpoints,
            researchMeta: buildRuntimeMeta(state),
            final: state.final,
        };

        if (onEvent) {
            await onEvent({
                type: "final",
                runId: state.runId,
                result: responsePayload,
            });
        }

        return responsePayload;
    } catch (error) {
        state.status = "error";
        await safePersist(() => updateResearchRun(state.runId, {
            status: "error",
            error: normalizeText(error?.message || error),
            outputs: {
                tribunal: state.tribunal || null,
                convergence: state.convergence || null,
            },
            updatedAt: new Date().toISOString(),
        }));
        if (onEvent) {
            await onEvent({
                type: "error",
                runId: state.runId,
                error: normalizeText(error?.message || error),
                researchMeta: buildRuntimeMeta(state),
            });
        }
        throw error;
    }
};

module.exports = {
    runResearchRuntime,
    buildFallbackNarrative,
    buildConvergence,
    buildRuntimeMeta,
};
