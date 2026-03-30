const vm = require("node:vm");
const { handler: webFetchHandler } = require("../api/tools/web_fetch");
const { parseFetchToolPayload } = require("./web");
const { extractQueryTerms, buildRagEvidenceBlock } = require("./rag");
const {
    extractRepoLinks,
    getDomainSampleThreshold,
    inferSourceSampleSignals,
    buildSourceSafetySignals,
} = require("./research-sources");

const MAX_FULL_FETCH = 18;
const MAX_LIGHT_FETCH = 12;
const FETCH_CONCURRENCY = 4;
const REQUEST_TIMEOUT_MS = 9000;
const GITHUB_API_BASE_URL = "https://api.github.com";

const P_VALUE_RE = /\bp\s*([<=>])\s*(0?\.\d+(?:e-?\d+)?)\b/gi;
const CI_RE = /\b(?:95%\s*)?CI\s*[:=]?\s*\[?\s*(-?\d+(?:\.\d+)?)\s*[,–-]\s*(-?\d+(?:\.\d+)?)\s*\]?/gi;
const SAMPLE_SIZE_RE = /\b(?:n|N)\s*[:=]\s*(\d{1,7})\b|\b(?:sample size|cohort(?: size)?|dataset(?: size)?|trial(?: size)?)\s*[:=]?\s*(\d{1,7})\b|\b(\d{1,7})\s+(participants|patients|subjects|samples|records|observations|images|respondents|users|documents)\b/gi;
const EFFECT_SIZE_RE = /\b(Cohen'?s d|effect size|odds ratio|hazard ratio|risk ratio|relative risk|AUC|accuracy|F1|BLEU|ROUGE|HR|OR|RR)\s*[:=]?\s*(-?\d+(?:\.\d+)?)\b/gi;
const FUNDING_RE = /\b(funded by|funding|sponsored by|supported by|grant|conflict of interest|competing interests?)\b/i;
const PREDATORY_RE = /\b(predatory journal|questionable publisher|dubious journal|paper mill|pay[- ]to[- ]publish)\b/i;
const DUAL_USE_RE = /\b(biosecurity|surveillance|weapon|weapons|exploit|malware|pathogen|offensive cyber)\b/i;
const POSITIVE_SIGNAL_RE = /\b(significant(?:ly)?|improve(?:d|ment)?|outperform(?:s|ed)?|effective|benefit(?:s)?|higher|increase(?:d)?|reduces? error|state[- ]of[- ]the[- ]art)\b/i;
const NEGATIVE_SIGNAL_RE = /\b(no significant|null result|not effective|fails? to|worse|lower|decrease(?:d)?|did not improve|no improvement|harm(?:ful)?)\b/i;
const MIXED_SIGNAL_RE = /\b(mixed|inconsistent|however|but|conflicting|uncertain|counterevidence)\b/i;
const ROUND_NUMBER_RE = /\b(?:0\.\d*00+|100\.0+|50\.0+)\b/;
const EDGE_THRESHOLD_P_RE = /\bp\s*=\s*0\.0?(4[5-9]|5[0-5])\b/i;

const normalizeText = (value) => String(value ?? "").replace(/\u0000/g, "").replace(/\r\n?/g, "\n").trim();
const clamp = (value, min, max) => Math.min(max, Math.max(min, Number(value) || 0));
const unique = (values = []) => [...new Set((Array.isArray(values) ? values : []).filter(Boolean))];
const uniqueBy = (values = [], keySelector = (value) => value) => {
    const seen = new Set();
    const output = [];
    for (const value of Array.isArray(values) ? values : []) {
        const key = keySelector(value);
        if (!key || seen.has(key)) continue;
        seen.add(key);
        output.push(value);
    }
    return output;
};
const average = (values = []) => {
    const numeric = (Array.isArray(values) ? values : []).map(Number).filter(Number.isFinite);
    return numeric.length ? numeric.reduce((sum, value) => sum + value, 0) / numeric.length : null;
};
const median = (values = []) => {
    const numeric = (Array.isArray(values) ? values : []).map(Number).filter(Number.isFinite).sort((left, right) => left - right);
    if (!numeric.length) return null;
    const mid = Math.floor(numeric.length / 2);
    return numeric.length % 2 === 0 ? (numeric[mid - 1] + numeric[mid]) / 2 : numeric[mid];
};

const runConcurrent = async (items, limit, worker) => {
    const values = Array.isArray(items) ? items : [];
    const concurrency = Math.max(1, Number(limit) || 1);
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
    return results.filter(Boolean);
};

const extractLinks = (text = "") => (
    [...String(text || "").matchAll(/\bhttps?:\/\/[^\s)<>"']+/gi)]
        .map((match) => match[0].replace(/[),.;:!?]+$/, ""))
);

const extractSupplementaryLinks = (text = "") => unique(
    extractLinks(text).filter((url) => /\b(supplement|appendix|appendices|dataset|artifact|materials|zenodo|figshare|osf)\b/i.test(url)),
).slice(0, 8);

const inferStudyDesign = (text = "", source = {}) => {
    const corpus = `${source?.title || ""}\n${source?.description || ""}\n${text}`.toLowerCase();
    if (/\bmeta-analysis|systematic review\b/.test(corpus)) return "meta_analysis";
    if (/\brandomized|rct|controlled trial\b/.test(corpus)) return "rct";
    if (/\bcohort|observational|case control|longitudinal|survey\b/.test(corpus)) return "observational";
    if (/\bcase study|case report\b/.test(corpus)) return "case_study";
    if (/\bbenchmark|ablation|leaderboard\b/.test(corpus) || source?.kind === "repository") return "benchmark";
    return source?.kind === "paper" ? "paper" : "general";
};

const evidenceTypeWeight = (type = "general") => ({
    meta_analysis: 1,
    rct: 0.94,
    observational: 0.72,
    benchmark: 0.72,
    paper: 0.64,
    case_study: 0.48,
    general: 0.58,
    opinion: 0.34,
}[type] || 0.58);

const extractExcerpt = (text = "", index = 0, length = 100, radius = 140) => {
    const start = Math.max(0, Number(index) - radius);
    const end = Math.min(String(text || "").length, Number(index) + Number(length || 0) + radius);
    return normalizeText(String(text || "").slice(start, end));
};

const detectEffectMetricFamily = (metric = "") => {
    const normalized = String(metric || "").toLowerCase();
    if (/\b(cohen'?s d|effect size)\b/.test(normalized)) return "standardized_mean_difference";
    if (/\b(odds ratio|hazard ratio|risk ratio|relative risk|or|hr|rr)\b/.test(normalized)) return "ratio";
    if (/\b(auc|accuracy|f1|bleu|rouge)\b/.test(normalized)) return "bounded_score";
    return "generic_effect";
};

const normalizeEffectMetric = (metric = "", value) => {
    const metricFamily = detectEffectMetricFamily(metric);
    const numericValue = Number(value);
    if (!Number.isFinite(numericValue)) {
        return {
            metricFamily,
            normalizedValue: null,
            effectDirection: "unknown",
        };
    }

    let normalizedValue = numericValue;
    if (metricFamily === "ratio") {
        normalizedValue = numericValue > 0 ? Math.log(numericValue) : null;
    } else if (metricFamily === "bounded_score") {
        normalizedValue = numericValue > 1 ? numericValue / 100 : numericValue;
    }

    const effectDirection = normalizedValue == null
        ? "unknown"
        : normalizedValue > 0.02 ? "positive"
            : normalizedValue < -0.02 ? "negative"
                : "neutral";

    return {
        metricFamily,
        normalizedValue: Number.isFinite(normalizedValue) ? Number(normalizedValue.toFixed(4)) : null,
        effectDirection,
    };
};

const classifyStance = (text = "", source = {}) => {
    if (source?.contradictionHint) return source.contradictionHint;
    const corpus = `${source?.title || ""}\n${source?.description || ""}\n${text}`;
    if (MIXED_SIGNAL_RE.test(corpus)) return "mixed";
    if (NEGATIVE_SIGNAL_RE.test(corpus)) return "contrary";
    if (POSITIVE_SIGNAL_RE.test(corpus)) return "supportive";
    return "neutral";
};

const extractStatisticalClaims = (entry, plan = {}) => {
    const text = normalizeText(entry?.content || "");
    const source = entry?.source || {};
    const sourceSampleSignals = inferSourceSampleSignals(source, plan?.domain);
    const minimumRecommended = getDomainSampleThreshold(plan?.domain);
    const studyDesign = inferStudyDesign(text, source);
    const claims = [];
    if (!text) return claims;

    for (const match of text.matchAll(P_VALUE_RE)) {
        const pValue = Number(match[2]);
        claims.push({
            type: "p_value",
            metricFamily: "p_value",
            operator: match[1],
            value: pValue,
            pValue,
            nearThreshold: pValue >= 0.045 && pValue <= 0.055,
            studyDesign,
            studyWeight: evidenceTypeWeight(studyDesign),
            sampleSize: sourceSampleSignals.sampleSize,
            sourceUrl: source?.url || "",
            sourceTitle: source?.title || "",
            sourceKey: source?.url || source?.title || "",
            excerpt: extractExcerpt(text, match.index, match[0].length),
        });
    }

    for (const match of text.matchAll(CI_RE)) {
        const lower = Number(match[1]);
        const upper = Number(match[2]);
        claims.push({
            type: "confidence_interval",
            metricFamily: "confidence_interval",
            lower,
            upper,
            intervalWidth: Number((upper - lower).toFixed(4)),
            value: Number((((lower + upper) / 2)).toFixed(4)),
            studyDesign,
            studyWeight: evidenceTypeWeight(studyDesign),
            sampleSize: sourceSampleSignals.sampleSize,
            sourceUrl: source?.url || "",
            sourceTitle: source?.title || "",
            sourceKey: source?.url || source?.title || "",
            excerpt: extractExcerpt(text, match.index, match[0].length),
        });
    }

    for (const match of text.matchAll(SAMPLE_SIZE_RE)) {
        const sampleSize = Number(match[1] || match[2] || match[3] || 0);
        if (!sampleSize) continue;
        claims.push({
            type: "sample_size",
            metricFamily: "sample_size",
            value: sampleSize,
            sampleSize,
            minimumRecommended,
            adequacy: Number((sampleSize / minimumRecommended).toFixed(2)),
            sampleSizeFlag: sampleSize < minimumRecommended,
            underpowered: sampleSize < minimumRecommended,
            studyDesign,
            studyWeight: evidenceTypeWeight(studyDesign),
            sourceUrl: source?.url || "",
            sourceTitle: source?.title || "",
            sourceKey: source?.url || source?.title || "",
            excerpt: extractExcerpt(text, match.index, String(match[0] || "").length),
        });
    }

    for (const match of text.matchAll(EFFECT_SIZE_RE)) {
        const metric = normalizeText(match[1]);
        const numericValue = Number(match[2]);
        const normalizedEffect = normalizeEffectMetric(metric, numericValue);
        claims.push({
            type: "effect_size",
            metric,
            value: numericValue,
            metricFamily: normalizedEffect.metricFamily,
            normalizedValue: normalizedEffect.normalizedValue,
            effectDirection: normalizedEffect.effectDirection,
            studyDesign,
            studyWeight: evidenceTypeWeight(studyDesign),
            sampleSize: sourceSampleSignals.sampleSize,
            sampleSizeFlag: Boolean(sourceSampleSignals.sampleSizeFlag),
            minimumRecommended,
            sourceUrl: source?.url || "",
            sourceTitle: source?.title || "",
            sourceKey: source?.url || source?.title || "",
            excerpt: extractExcerpt(text, match.index, match[0].length),
        });
    }

    return claims;
};

const extractConceptsFromText = (text = "", query = "", source = {}, plan = {}) => {
    const ontologyTerms = Array.isArray(plan?.domain?.ontologyTerms) ? plan.domain.ontologyTerms : [];
    const queryTerms = extractQueryTerms(`${query} ${source?.title || ""} ${ontologyTerms.join(" ")}`, 18);
    const textTerms = extractQueryTerms(text, 28);
    const seeded = unique([...queryTerms, ...textTerms, ...ontologyTerms.map((term) => normalizeText(term).toLowerCase())]).slice(0, 18);

    return seeded.map((term) => ({
        name: term,
        claims: [source?.title].filter(Boolean),
        methods: [],
        weight: 1,
        sourceUrl: source?.url || "",
    }));
};

const scoreRepositoryReadme = (content = "") => {
    const text = String(content || "").toLowerCase();
    let score = 0;
    if (/install|setup|requirements|dependencies/.test(text)) score += 1;
    if (/usage|quickstart|example|examples/.test(text)) score += 1;
    if (/reproduce|reproducibility|replicate|training|evaluation/.test(text)) score += 1;
    if (/dataset|data|benchmark/.test(text)) score += 1;
    if (/license|citation|cite/.test(text)) score += 1;
    return Math.max(0, Math.min(5, score));
};

const withTimeout = async (worker, timeoutMs = REQUEST_TIMEOUT_MS) => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        return await worker(controller.signal);
    } finally {
        clearTimeout(timer);
    }
};

const readJson = async (url, options = {}) => withTimeout(async (signal) => {
    const response = await fetch(url, {
        ...options,
        signal,
        headers: {
            "User-Agent": "nub-agent/1.0",
            "Accept": "application/vnd.github+json, application/json",
            ...(options.headers || {}),
        },
    });
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
});

const parseGitHubRepoIdentity = (repoUrl = "") => {
    const match = String(repoUrl || "").match(/^https?:\/\/github\.com\/([^/]+)\/([^/#?]+)/i);
    if (!match) return null;
    return {
        owner: match[1],
        repo: match[2].replace(/\.git$/i, ""),
    };
};

const decodeGitHubContent = (payload = {}) => {
    const content = String(payload?.content || "");
    const encoding = String(payload?.encoding || "").toLowerCase();
    if (!content) return "";
    if (encoding === "base64") {
        try {
            return Buffer.from(content, "base64").toString("utf8");
        } catch {
            return "";
        }
    }
    return content;
};

const buildRepositoryRubric = ({ repoMeta = {}, readmeText = "", topLevelNames = [] } = {}) => {
    const topLevel = new Set((Array.isArray(topLevelNames) ? topLevelNames : []).map((item) => String(item || "").toLowerCase()));
    const manifestFiles = [
        "package.json",
        "pyproject.toml",
        "requirements.txt",
        "pipfile",
        "poetry.lock",
        "cargo.toml",
        "go.mod",
        "pom.xml",
        "build.gradle",
        "environment.yml",
    ].filter((file) => topLevel.has(file));
    const lockfiles = [
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "poetry.lock",
        "cargo.lock",
        "requirements.lock",
    ].filter((file) => topLevel.has(file));
    const testSignals = [
        "tests",
        "test",
        "__tests__",
        "tox.ini",
        "noxfile.py",
        "pytest.ini",
    ].filter((file) => topLevel.has(file));
    const ciSignals = [
        ".github",
        ".gitlab-ci.yml",
        "azure-pipelines.yml",
        "circle.yml",
        ".circleci",
    ].filter((file) => topLevel.has(file));
    const environmentSignals = [
        "dockerfile",
        "docker-compose.yml",
        ".devcontainer",
        "makefile",
        "justfile",
        ".env.example",
    ].filter((file) => topLevel.has(file));
    const artifactSignals = [
        "citation.cff",
        "license",
        "data",
        "dataset",
        "artifacts",
        "models",
    ].filter((file) => topLevel.has(file));
    const readmeScore = scoreRepositoryReadme(readmeText);
    const documentation = readmeScore / 5;
    const dependencies = manifestFiles.length ? (lockfiles.length ? 1 : 0.72) : 0;
    const automation = ciSignals.length ? 1 : (testSignals.length ? 0.68 : 0);
    const environment = environmentSignals.length ? 1 : (manifestFiles.length ? 0.52 : 0);
    const artifacts = Math.min(1, (artifactSignals.length ? 0.7 : 0) + (repoMeta?.license?.spdx_id ? 0.3 : 0));
    const reproducibilityScore = Number(clamp(documentation + dependencies + automation + environment + artifacts, 0, 5).toFixed(2));
    const blockedBy = [];
    if (!manifestFiles.length) blockedBy.push("missing_dependency_manifest");
    if (!readmeText) blockedBy.push("missing_readme");
    if (!ciSignals.length && !testSignals.length) blockedBy.push("missing_ci_or_tests");
    if (repoMeta?.archived) blockedBy.push("archived_repository");

    return {
        reproducibilityScore,
        readmeScore,
        manifestFiles,
        lockfiles,
        testSignals,
        ciSignals,
        environmentSignals,
        artifactSignals,
        blockedBy,
        rubric: {
            documentation: Number(documentation.toFixed(2)),
            dependencies: Number(dependencies.toFixed(2)),
            automation: Number(automation.toFixed(2)),
            environment: Number(environment.toFixed(2)),
            artifacts: Number(artifacts.toFixed(2)),
        },
    };
};

const analyzeGitHubRepository = async (repoUrl = "") => {
    const parsed = parseGitHubRepoIdentity(repoUrl);
    if (!parsed) return null;

    const headers = {};
    if (process.env.GITHUB_TOKEN) {
        headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
    }

    const [repoMeta, readmePayload, contentsPayload] = await Promise.all([
        readJson(`${GITHUB_API_BASE_URL}/repos/${parsed.owner}/${parsed.repo}`, { headers }).catch(() => null),
        readJson(`${GITHUB_API_BASE_URL}/repos/${parsed.owner}/${parsed.repo}/readme`, { headers }).catch(() => null),
        readJson(`${GITHUB_API_BASE_URL}/repos/${parsed.owner}/${parsed.repo}/contents`, { headers }).catch(() => null),
    ]);

    const readmeText = decodeGitHubContent(readmePayload);
    const topLevelNames = Array.isArray(contentsPayload)
        ? contentsPayload.map((item) => normalizeText(item?.name).toLowerCase()).filter(Boolean)
        : [];
    const rubric = buildRepositoryRubric({ repoMeta, readmeText, topLevelNames });

    return {
        url: repoUrl,
        provider: "github_api",
        repo: `${parsed.owner}/${parsed.repo}`,
        reproducibilityScore: rubric.reproducibilityScore,
        readmeScore: rubric.readmeScore,
        rubric: rubric.rubric,
        manifestFiles: rubric.manifestFiles,
        lockfiles: rubric.lockfiles,
        testSignals: rubric.testSignals,
        ciSignals: rubric.ciSignals,
        environmentSignals: rubric.environmentSignals,
        artifactSignals: rubric.artifactSignals,
        blockedBy: rubric.blockedBy,
        language: normalizeText(repoMeta?.language),
        stars: Number(repoMeta?.stargazers_count || 0),
        forks: Number(repoMeta?.forks_count || 0),
        archived: Boolean(repoMeta?.archived),
        openIssues: Number(repoMeta?.open_issues_count || 0),
        license: normalizeText(repoMeta?.license?.spdx_id),
        notes: normalizeText([
            rubric.manifestFiles.length ? `manifests: ${rubric.manifestFiles.join(", ")}` : "no dependency manifest",
            rubric.ciSignals.length ? "ci/test automation present" : "ci/test automation unclear",
            rubric.environmentSignals.length ? `env: ${rubric.environmentSignals.join(", ")}` : "no environment scaffolding",
        ].join("; ")),
    };
};

const analyzeRepositoryFallback = async (repoUrl = "") => {
    try {
        const raw = await webFetchHandler({
            url: repoUrl,
            format: "markdown",
            max_chars: 12000,
        });
        if (typeof raw !== "string" || raw.startsWith("Error:")) {
            return {
                url: repoUrl,
                provider: "web_fetch",
                reproducibilityScore: 0,
                blockedBy: ["repository_fetch_failed"],
                notes: "Repository fetch failed.",
            };
        }
        const parsed = parseFetchToolPayload(raw);
        const content = normalizeText(parsed.content);
        const readmeScore = scoreRepositoryReadme(content);
        const blockedBy = [];
        if (readmeScore < 2) blockedBy.push("low_readme_coverage");
        return {
            url: repoUrl,
            provider: "web_fetch",
            reproducibilityScore: Number(readmeScore.toFixed(2)),
            readmeScore,
            rubric: {
                documentation: Number((readmeScore / 5).toFixed(2)),
                dependencies: /\b(requirements|dependency|dependencies|install)\b/i.test(content) ? 0.5 : 0,
                automation: /\b(test|ci|workflow|github actions|docker)\b/i.test(content) ? 0.5 : 0,
                environment: /\b(docker|conda|venv|setup)\b/i.test(content) ? 0.5 : 0,
                artifacts: /\b(dataset|data|citation|license)\b/i.test(content) ? 0.5 : 0,
            },
            blockedBy,
            notes: content.slice(0, 600),
        };
    } catch {
        return {
            url: repoUrl,
            provider: "web_fetch",
            reproducibilityScore: 0,
            blockedBy: ["repository_analysis_failed"],
            notes: "Repository analysis failed.",
        };
    }
};

const analyzeRepositories = async (repoUrls = []) => {
    const uniqueRepos = unique(repoUrls).slice(0, 8);
    return runConcurrent(uniqueRepos, 3, async (repoUrl) => {
        const githubAnalysis = await analyzeGitHubRepository(repoUrl).catch(() => null);
        if (githubAnalysis) return githubAnalysis;
        return analyzeRepositoryFallback(repoUrl);
    });
};

const analyzeSupplementaryLinks = async (links = []) => {
    const selected = unique(links).slice(0, 4);
    const results = [];
    for (const link of selected) {
        try {
            const raw = await webFetchHandler({
                url: link,
                format: "markdown",
                max_chars: 8000,
            });
            if (typeof raw !== "string" || raw.startsWith("Error:")) continue;
            const parsed = parseFetchToolPayload(raw);
            results.push({
                url: link,
                title: parsed.title,
                excerpt: parsed.content.slice(0, 1200),
            });
        } catch {
            /* ignore supplementary failures */
        }
    }
    return results;
};

const buildEvidencePyramid = (entries = []) => (
    entries.slice(0, 24).map((entry) => {
        const source = entry?.source || {};
        const studyDesign = inferStudyDesign(entry?.content || "", source);
        return {
            sourceUrl: source.url || "",
            sourceTitle: source.title || "",
            level: studyDesign,
            weight: evidenceTypeWeight(studyDesign),
        };
    })
);

const buildSampleSizeProfile = (claims = [], plan = {}) => {
    const sampleClaims = claims.filter((claim) => claim.type === "sample_size" && Number.isFinite(claim.value));
    const values = sampleClaims.map((claim) => claim.value);
    const minimumRecommended = getDomainSampleThreshold(plan?.domain);
    const bySource = uniqueBy(sampleClaims, (claim) => claim.sourceUrl || claim.sourceTitle)
        .map((claim) => ({
            sourceUrl: claim.sourceUrl,
            sourceTitle: claim.sourceTitle,
            sampleSize: claim.value,
            underpowered: Boolean(claim.sampleSizeFlag),
        }))
        .slice(0, 20);

    return {
        minimumRecommended,
        count: values.length,
        min: values.length ? Math.min(...values) : null,
        max: values.length ? Math.max(...values) : null,
        median: values.length ? Number(median(values).toFixed(2)) : null,
        mean: values.length ? Number(average(values).toFixed(2)) : null,
        underpoweredCount: sampleClaims.filter((claim) => claim.sampleSizeFlag).length,
        underpoweredSources: bySource.filter((item) => item.underpowered),
        bySource,
    };
};

const buildPValueSummary = (claims = []) => {
    const pValues = claims
        .filter((claim) => claim.type === "p_value" && Number.isFinite(claim.pValue))
        .map((claim) => claim.pValue);
    return {
        count: pValues.length,
        significantCount: pValues.filter((value) => value < 0.05).length,
        nearThresholdCount: pValues.filter((value) => value >= 0.045 && value <= 0.055).length,
        median: pValues.length ? Number(median(pValues).toFixed(4)) : null,
        min: pValues.length ? Math.min(...pValues) : null,
        max: pValues.length ? Math.max(...pValues) : null,
    };
};

const buildContradictionArtifacts = (entries = [], claims = []) => {
    const stanceMap = entries.slice(0, 24).map((entry) => ({
        sourceUrl: entry?.source?.url || "",
        sourceTitle: entry?.source?.title || "",
        stance: classifyStance(entry?.content || "", entry?.source || {}),
        excerpt: normalizeText(String(entry?.content || "").slice(0, 260)),
    }));

    const effectClaims = claims.filter((claim) => claim.type === "effect_size" && Number.isFinite(claim.normalizedValue));
    const metricConflicts = [];
    const families = unique(effectClaims.map((claim) => claim.metricFamily));
    for (const family of families) {
        const familyClaims = effectClaims.filter((claim) => claim.metricFamily === family);
        const positive = familyClaims.filter((claim) => claim.effectDirection === "positive");
        const negative = familyClaims.filter((claim) => claim.effectDirection === "negative");
        if (positive.length && negative.length) {
            metricConflicts.push({
                metricFamily: family,
                positiveSources: unique(positive.map((claim) => claim.sourceTitle || claim.sourceUrl)).slice(0, 8),
                negativeSources: unique(negative.map((claim) => claim.sourceTitle || claim.sourceUrl)).slice(0, 8),
                count: familyClaims.length,
            });
        }
    }

    const supportiveCount = stanceMap.filter((item) => item.stance === "supportive").length;
    const contraryCount = stanceMap.filter((item) => item.stance === "contrary").length;
    const mixedCount = stanceMap.filter((item) => item.stance === "mixed").length;
    const contradictionCount = metricConflicts.length + ((supportiveCount && contraryCount) ? 1 : 0) + mixedCount;
    const falseConsensusRisk = Number(clamp(
        (metricConflicts.length * 0.18)
        + ((supportiveCount && contraryCount) ? 0.32 : 0)
        + (mixedCount * 0.08),
        0,
        1,
    ).toFixed(2));

    return {
        contradictionCount,
        falseConsensusRisk,
        stanceMap,
        metricConflicts,
        summary: {
            supportiveCount,
            contraryCount,
            mixedCount,
        },
    };
};

const buildMetaAnalysis = (claims = [], plan = {}) => {
    const effectClaims = claims.filter((claim) => claim.type === "effect_size" && Number.isFinite(claim.normalizedValue));
    const pValueSummary = buildPValueSummary(claims);
    const sampleProfile = buildSampleSizeProfile(claims, plan);
    if (!effectClaims.length) {
        return {
            combined_effect_size: null,
            i_squared: null,
            model: "not_enough_data",
            assumption_conflicts: pValueSummary.nearThresholdCount ? ["Borderline p-values without normalized effect sizes."] : [],
            family_summaries: [],
            effect_size_count: 0,
            p_value_summary: pValueSummary,
        };
    }

    const families = unique(effectClaims.map((claim) => claim.metricFamily));
    const familySummaries = families.map((family) => {
        const familyClaims = effectClaims.filter((claim) => claim.metricFamily === family);
        const weighted = familyClaims.map((claim) => ({
            value: claim.normalizedValue,
            weight: claim.sampleSize ? Math.sqrt(Math.max(1, claim.sampleSize)) : 1,
        })).filter((item) => Number.isFinite(item.value) && Number.isFinite(item.weight));
        const weightSum = weighted.reduce((sum, item) => sum + item.weight, 0) || 1;
        const combined = weighted.reduce((sum, item) => sum + (item.value * item.weight), 0) / weightSum;
        const qStatistic = weighted.reduce((sum, item) => sum + (item.weight * ((item.value - combined) ** 2)), 0);
        const degreesFreedom = Math.max(0, weighted.length - 1);
        const iSquared = qStatistic > 0 && degreesFreedom > 0
            ? clamp(((qStatistic - degreesFreedom) / qStatistic) * 100, 0, 100)
            : 0;
        const direction = combined > 0.02 ? "positive" : combined < -0.02 ? "negative" : "neutral";

        return {
            metricFamily: family,
            studyCount: familyClaims.length,
            combined_effect_size: Number(combined.toFixed(4)),
            i_squared: Number(iSquared.toFixed(2)),
            model: familyClaims.length > 1 ? (iSquared >= 40 ? "random_effects" : "fixed_effects") : "single_study",
            sample_weight_total: Number(weightSum.toFixed(2)),
            direction,
        };
    }).sort((left, right) => right.studyCount - left.studyCount);

    const primary = familySummaries[0];
    const assumptionConflicts = [];
    if (familySummaries.length > 1) assumptionConflicts.push("Multiple effect metric families required separate aggregation.");
    if ((primary?.i_squared || 0) >= 65) assumptionConflicts.push("High heterogeneity across normalized effect sizes.");
    if (sampleProfile.underpoweredCount > 0) assumptionConflicts.push(`${sampleProfile.underpoweredCount} source(s) fall below the field sample-size threshold.`);
    if (pValueSummary.nearThresholdCount > 0) assumptionConflicts.push("Borderline p-values increase statistical fragility.");

    return {
        combined_effect_size: primary?.combined_effect_size ?? null,
        i_squared: primary?.i_squared ?? null,
        model: primary?.model || "not_enough_data",
        assumption_conflicts: unique(assumptionConflicts),
        family_summaries: familySummaries,
        effect_size_count: effectClaims.length,
        evidence_direction: primary?.direction || "unknown",
        sample_weight_total: primary?.sample_weight_total ?? 0,
        p_value_summary: pValueSummary,
    };
};

const verifyStatisticsWithSandbox = (claims = [], plan = {}, sampleProfile = null, contradictions = null) => {
    const effectValues = claims
        .filter((claim) => claim.type === "effect_size" && Number.isFinite(claim.normalizedValue))
        .map((claim) => ({
            value: claim.normalizedValue,
            weight: claim.sampleSize ? Math.sqrt(Math.max(1, claim.sampleSize)) : 1,
        }));
    const pValues = claims
        .filter((claim) => claim.type === "p_value" && Number.isFinite(claim.pValue))
        .map((claim) => claim.pValue);
    const sampleSizes = claims
        .filter((claim) => claim.type === "sample_size" && Number.isFinite(claim.value))
        .map((claim) => claim.value);

    if (!effectValues.length && !pValues.length && !sampleSizes.length) {
        return {
            executed: false,
            engine: "node_vm",
            typedChecks: [],
            notes: [],
        };
    }

    const sandbox = {
        effects: effectValues,
        pValues,
        sampleSizes,
        result: null,
    };
    const context = vm.createContext(sandbox);
    const script = new vm.Script(`
        const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
        const median = (values) => {
            if (!values.length) return null;
            const sorted = [...values].sort((left, right) => left - right);
            const mid = Math.floor(sorted.length / 2);
            return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
        };

        const effectWeightSum = effects.reduce((sum, item) => sum + item.weight, 0);
        const weightedEffectMean = effectWeightSum
            ? effects.reduce((sum, item) => sum + (item.value * item.weight), 0) / effectWeightSum
            : null;
        const effectSigns = effects.map((item) => item.value > 0.02 ? 1 : item.value < -0.02 ? -1 : 0);
        result = {
            effects: effects.length ? {
                count: effects.length,
                weightedMean: weightedEffectMean,
                signFlipCount: effectSigns.filter((value) => value === 1).length && effectSigns.filter((value) => value === -1).length ? 1 : 0,
                min: Math.min(...effects.map((item) => item.value)),
                max: Math.max(...effects.map((item) => item.value)),
            } : null,
            pValues: pValues.length ? {
                count: pValues.length,
                significantCount: pValues.filter((value) => value < 0.05).length,
                median: median(pValues),
                nearThresholdCount: pValues.filter((value) => value >= 0.045 && value <= 0.055).length,
            } : null,
            sampleSizes: sampleSizes.length ? {
                count: sampleSizes.length,
                min: Math.min(...sampleSizes),
                max: Math.max(...sampleSizes),
                mean: mean(sampleSizes),
                median: median(sampleSizes),
            } : null,
        };
    `);
    script.runInContext(context, { timeout: 50 });

    const typedChecks = [];
    if (sandbox.result.effects) {
        typedChecks.push({
            type: "effect_consistency",
            status: sandbox.result.effects.signFlipCount ? "warn" : "pass",
            detail: `Weighted normalized effect mean ${Number(sandbox.result.effects.weightedMean || 0).toFixed(4)} across ${sandbox.result.effects.count} effect claim(s).`,
        });
    }
    if (sandbox.result.pValues) {
        typedChecks.push({
            type: "p_value_profile",
            status: sandbox.result.pValues.nearThresholdCount ? "warn" : "pass",
            detail: `${sandbox.result.pValues.significantCount}/${sandbox.result.pValues.count} p-value claim(s) are < 0.05.`,
        });
    }
    if (sandbox.result.sampleSizes) {
        const minimumRecommended = getDomainSampleThreshold(plan?.domain);
        typedChecks.push({
            type: "sample_size_profile",
            status: (sampleProfile?.underpoweredCount || 0) > 0 ? "warn" : "pass",
            detail: `Median sample size ${Number(sandbox.result.sampleSizes.median || 0).toFixed(2)} versus field threshold ${minimumRecommended}.`,
        });
    }
    if ((contradictions?.contradictionCount || 0) > 0) {
        typedChecks.push({
            type: "cross_paper_contradiction",
            status: "warn",
            detail: `${contradictions.contradictionCount} contradiction signal(s) detected across stance and metric families.`,
        });
    }

    return {
        executed: true,
        engine: "node_vm",
        typedChecks,
        summary: sandbox.result,
        notes: [
            sandbox.result.effects ? `effects:${sandbox.result.effects.count}` : "",
            sandbox.result.pValues ? `p_values:${sandbox.result.pValues.count}` : "",
            sandbox.result.sampleSizes ? `samples:${sandbox.result.sampleSizes.count}` : "",
        ].filter(Boolean),
    };
};

const buildDualUseChains = (text = "") => {
    const corpus = String(text || "").toLowerCase();
    const chains = [];
    if (/\bpathogen|biosecurity\b/.test(corpus)) {
        chains.push({ misuseVector: "biological misuse", impact: "high" });
    }
    if (/\bsurveillance\b/.test(corpus)) {
        chains.push({ misuseVector: "mass surveillance", impact: "medium" });
    }
    if (/\bmalware|exploit|offensive cyber\b/.test(corpus)) {
        chains.push({ misuseVector: "offensive cyber capability", impact: "high" });
    }
    if (/\bweapon|weapons\b/.test(corpus)) {
        chains.push({ misuseVector: "weapons optimization", impact: "high" });
    }
    return chains;
};

const buildSafetySignals = (entries = [], claims = [], contradictions = null) => {
    const fundingFindings = [];
    const predatoryFindings = [];
    const dualUseFindings = [];
    const manipulationFindings = [];
    const retractedFindings = [];

    for (const entry of entries) {
        const source = entry?.source || {};
        const text = `${source.title || ""}\n${source.description || ""}\n${entry.content || ""}`;
        const excerpt = normalizeText(String(entry?.content || source?.description || "").slice(0, 320));
        const sourceSafety = buildSourceSafetySignals(source);

        if (FUNDING_RE.test(text) || sourceSafety.summary.includes("funding_conflict")) {
            fundingFindings.push({
                type: "funding_conflict",
                severity: "medium",
                sourceTitle: source.title || "",
                sourceUrl: source.url || "",
                evidence: excerpt,
            });
        }
        if (PREDATORY_RE.test(text) || sourceSafety.summary.includes("predatory_candidate")) {
            predatoryFindings.push({
                type: "predatory_journal",
                severity: "high",
                sourceTitle: source.title || "",
                sourceUrl: source.url || "",
                evidence: excerpt,
            });
        }
        if (DUAL_USE_RE.test(text) || sourceSafety.summary.includes("dual_use")) {
            dualUseFindings.push({
                type: "dual_use",
                severity: "high",
                sourceTitle: source.title || "",
                sourceUrl: source.url || "",
                evidence: excerpt,
                causalChains: buildDualUseChains(text),
            });
        }
        if (source.isRetracted || sourceSafety.summary.includes("retracted")) {
            retractedFindings.push({
                type: "retracted",
                severity: "critical",
                sourceTitle: source.title || "",
                sourceUrl: source.url || "",
                evidence: normalizeText(source.retractionReason || excerpt),
            });
        }

        const pValueCount = (text.match(P_VALUE_RE) || []).length;
        const hasRoundNumbers = ROUND_NUMBER_RE.test(text);
        const hasEdgeThreshold = EDGE_THRESHOLD_P_RE.test(text);
        const sourceClaims = claims.filter((claim) => (claim.sourceUrl || claim.sourceTitle) === (source.url || source.title));
        const underpowered = sourceClaims.some((claim) => claim.sampleSizeFlag);
        if (pValueCount >= 4 || hasRoundNumbers || hasEdgeThreshold || underpowered) {
            manipulationFindings.push({
                type: "statistical_manipulation_candidate",
                severity: hasEdgeThreshold || hasRoundNumbers ? "high" : "medium",
                sourceTitle: source.title || "",
                sourceUrl: source.url || "",
                evidence: excerpt,
                signals: [
                    pValueCount >= 4 ? "dense_p_value_reporting" : "",
                    hasRoundNumbers ? "implausibly_round_numbers" : "",
                    hasEdgeThreshold ? "borderline_p_value" : "",
                    underpowered ? "underpowered_sample" : "",
                ].filter(Boolean),
            });
        }
    }

    const findings = [
        ...retractedFindings,
        ...predatoryFindings,
        ...dualUseFindings,
        ...fundingFindings,
        ...manipulationFindings,
    ];
    const contradictionPenalty = contradictions?.falseConsensusRisk || 0;
    const riskPropagationScore = Number(clamp(
        (dualUseFindings.length * 0.2)
        + (predatoryFindings.length * 0.16)
        + (retractedFindings.length * 0.24)
        + (manipulationFindings.length * 0.12)
        + contradictionPenalty,
        0,
        1,
    ).toFixed(2));

    return {
        activeCount: findings.length,
        fundingConflictCount: fundingFindings.length,
        predatoryCount: predatoryFindings.length,
        dualUseCount: dualUseFindings.length,
        manipulationCount: manipulationFindings.length,
        retractedCount: retractedFindings.length,
        contradictionCount: contradictions?.contradictionCount || 0,
        riskPropagationScore,
        findings: findings.slice(0, 40),
        fundingFindings,
        predatoryFindings,
        dualUseFindings,
        manipulationFindings,
        retractedFindings,
    };
};

const fetchTieredEvidence = async (sources = [], query = "") => {
    const fullSources = sources.filter((source) => source.fetchMode === "full").slice(0, MAX_FULL_FETCH);
    const lightSources = sources.filter((source) => source.fetchMode === "light").slice(0, MAX_LIGHT_FETCH);

    const fetchedFull = await runConcurrent(fullSources, FETCH_CONCURRENCY, async (source) => {
        try {
            const raw = await webFetchHandler({
                url: source.url,
                format: "markdown",
                max_chars: 24000,
            });
            if (typeof raw !== "string" || raw.startsWith("Error:")) return null;
            const parsed = parseFetchToolPayload(raw);
            const content = normalizeText(parsed.content);
            if (!content) return null;
            return {
                source: {
                    ...source,
                    title: parsed.title || source.title,
                    description: parsed.description || source.description,
                    url: parsed.finalUrl || source.url,
                },
                content,
                evidenceBlock: buildRagEvidenceBlock({
                    source: {
                        ...source,
                        title: parsed.title || source.title,
                        description: parsed.description || source.description,
                        url: parsed.finalUrl || source.url,
                    },
                    content,
                }, query),
            };
        } catch {
            return null;
        }
    });

    const fetchedLight = lightSources
        .map((source) => ({
            source,
            content: normalizeText(`${source.title}\n${source.description}`),
            evidenceBlock: buildRagEvidenceBlock({
                source,
                content: `${source.title}\n${source.description}`,
            }, query),
        }))
        .filter((entry) => entry.content);

    return [...fetchedFull, ...fetchedLight];
};

const extractResearchArtifacts = async ({ query, plan, evidenceEntries = [] }) => {
    const claims = evidenceEntries.flatMap((entry) => extractStatisticalClaims(entry, plan));
    const repoUrls = unique(evidenceEntries.flatMap((entry) => [
        ...(Array.isArray(entry?.source?.repoUrls) ? entry.source.repoUrls : []),
        ...extractRepoLinks(entry?.content || ""),
    ])).slice(0, 8);
    const supplementaryLinks = unique(evidenceEntries.flatMap((entry) => extractSupplementaryLinks(entry?.content || ""))).slice(0, 8);
    const repositories = await analyzeRepositories(repoUrls);
    const supplementary = await analyzeSupplementaryLinks(supplementaryLinks);
    const concepts = uniqueBy(
        evidenceEntries.flatMap((entry) => extractConceptsFromText(entry?.content || "", query, entry?.source, plan)),
        (item) => item.name,
    ).slice(0, 32);
    const sampleProfile = buildSampleSizeProfile(claims, plan);
    const contradictions = buildContradictionArtifacts(evidenceEntries, claims);
    const metaAnalysis = buildMetaAnalysis(claims, plan);
    const statisticalVerification = verifyStatisticsWithSandbox(claims, plan, sampleProfile, contradictions);
    const safety = buildSafetySignals(evidenceEntries, claims, contradictions);

    return {
        claims,
        repositories,
        supplementary,
        concepts,
        evidencePyramid: buildEvidencePyramid(evidenceEntries),
        sampleSizeProfile: sampleProfile,
        contradictions,
        metaAnalysis,
        statisticalVerification,
        safety,
        safetyFindings: safety.findings,
        quantitativeSummary: {
            sampleSizeProfile: sampleProfile,
            pValueSummary: metaAnalysis.p_value_summary,
            effectFamilyCount: metaAnalysis.family_summaries.length,
            contradictionRisk: contradictions.falseConsensusRisk,
        },
        codeArtifacts: {
            repositoryCount: repositories.length,
            reproducibilityAverage: repositories.length
                ? Number((repositories.reduce((sum, repo) => sum + Number(repo.reproducibilityScore || 0), 0) / repositories.length).toFixed(2))
                : 0,
            automationReadyCount: repositories.filter((repo) => Array.isArray(repo.ciSignals) && repo.ciSignals.length).length,
            dependencyReadyCount: repositories.filter((repo) => Array.isArray(repo.manifestFiles) && repo.manifestFiles.length).length,
        },
        query,
        domain: plan?.domain?.label || "",
    };
};

module.exports = {
    fetchTieredEvidence,
    extractResearchArtifacts,
};
