const { handler: webSearchHandler } = require("./tools/web_search");
const {
    canonicalizeSourceUrl,
    domainAuthorityBoost,
    getSourceDomain,
    rankSourcesWithRag,
    scoreTextForTerms,
    extractQueryTerms,
} = require("./rag");

const REQUEST_TIMEOUT_MS = 12000;
const OPENALEX_WORKS_URL = "https://api.openalex.org/works";
const CROSSREF_WORKS_URL = "https://api.crossref.org/works";
const GITHUB_REPO_SEARCH_URL = "https://api.github.com/search/repositories";
const IEEE_XPLORE_SEARCH_URL = "https://ieeexploreapi.ieee.org/api/v1/search/articles";
const PWC_SEARCH_URL = "https://paperswithcode.com/api/v1/papers/";
const MAX_FORWARD_CITATION_SEEDS = 6;
const SAMPLE_SIZE_RE = /\b(?:n|N)\s*[:=]\s*(\d{1,7})\b|\b(?:sample size|cohort(?: size)?|dataset(?: size)?|trial(?: size)?)\s*[:=]?\s*(\d{1,7})\b|\b(\d{1,7})\s+(participants|patients|subjects|samples|records|observations|images|respondents|users|documents)\b/gi;
const FUNDING_RE = /\b(funded by|funding|sponsored by|supported by|grant from|industry[- ]funded|conflict of interest|competing interests?)\b/i;
const PREDATORY_RE = /\b(predatory journal|questionable publisher|dubious journal|paper mill|pay[- ]to[- ]publish)\b/i;
const DUAL_USE_RE = /\b(biosecurity|surveillance|weapon|weapons|exploit|malware|pathogen|offensive cyber)\b/i;
const NEAR_THRESHOLD_P_RE = /\bp\s*(?:=|<|>)\s*0\.0?(4[5-9]|5[0-5])\b/i;
const POSITIVE_FINDING_RE = /\b(significant(?:ly)?|improve(?:d|ment)?|outperform(?:s|ed)?|effective|benefit(?:s)?|higher|increase(?:d)?|reduces? error|state[- ]of[- ]the[- ]art)\b/i;
const NEGATIVE_FINDING_RE = /\b(no significant|null result|not effective|fails? to|worse|lower|decrease(?:d)?|did not improve|no improvement|harm(?:ful)?)\b/i;
const MIXED_FINDING_RE = /\b(mixed|inconsistent|however|but|conflicting|uncertain|counterevidence)\b/i;
const SCHOLARLY_LANE_RE = /\b(scholar\.google\.com|semanticscholar\.org|openalex\.org|site:\.edu|site:edu|arxiv\.org|pubmed|jstor\.org|ssrn\.com|paperswithcode\.com|osf\.io)\b/i;
const SCHOLARLY_PROVIDER_RE = /\b(openalex(?:_forward_citation)?|crossref|ieee_xplore|acm|jstor|papers_with_code)\b/i;
const SCHOLARLY_HOSTS = new Set([
    "scholar.google.com",
    "semanticscholar.org",
    "openalex.org",
    "arxiv.org",
    "doi.org",
    "pubmed.ncbi.nlm.nih.gov",
    "jstor.org",
    "ssrn.com",
    "paperswithcode.com",
    "osf.io",
    "zenodo.org",
]);
const DOMAIN_SAMPLE_THRESHOLDS = Object.freeze({
    biomedical: 100,
    cs_ml: 30,
    social_science: 80,
    interdisciplinary: 60,
    general_research: 50,
});

const normalizeText = (value) => String(value ?? "").replace(/\u0000/g, "").replace(/\s+/g, " ").trim();
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

const getAcademicHostSignal = (host = "") => {
    const normalized = normalizeText(host).toLowerCase();
    if (!normalized) return false;
    return normalized.endsWith(".edu")
        || normalized.includes(".edu.")
        || normalized.includes(".ac.")
        || SCHOLARLY_HOSTS.has(normalized);
};

const getScholarlyRoutingSignals = (plan = {}) => {
    const matrix = plan?.queryMatrix || {};
    const scholarlyLanes = unique([
        ...(Array.isArray(matrix.scholarlyDiscoveryLanes) ? matrix.scholarlyDiscoveryLanes : []),
        ...((Array.isArray(plan?.searchQueries) ? plan.searchQueries : []).filter((value) => SCHOLARLY_LANE_RE.test(normalizeText(value)))),
    ].map((value) => normalizeText(value)).filter(Boolean));
    const providerBias = unique((Array.isArray(matrix.scholarlyProviderBias) ? matrix.scholarlyProviderBias : [])
        .map((value) => normalizeText(value).toLowerCase())
        .filter(Boolean));
    return {
        active: scholarlyLanes.length > 0 || Boolean(plan?.scholarlyHarvest?.active),
        scholarlyLanes,
        providerBias,
    };
};

const sourceUsesScholarlyLane = (source = {}) => {
    const candidates = unique([
        source?.queryVariant,
        ...(Array.isArray(source?.queryVariants) ? source.queryVariants : []),
        ...((((source?.metadata && typeof source.metadata === "object") ? source.metadata : {}).queryLanes) || []),
    ].map((value) => normalizeText(value)).filter(Boolean));
    return candidates.some((candidate) => SCHOLARLY_LANE_RE.test(candidate));
};

const getScholarlyBoost = (source = {}, plan = {}) => {
    const routing = getScholarlyRoutingSignals(plan);
    if (!routing.active) return 0;

    const providers = unique(String(source?.provider || "")
        .split("+")
        .map((value) => normalizeText(value).toLowerCase())
        .filter(Boolean));
    const providerBiasHit = providers.some((provider) => routing.providerBias.includes(provider));
    const providerHit = providers.some((provider) => SCHOLARLY_PROVIDER_RE.test(provider));
    const host = getSourceDomain(source?.url || "");
    const academicHostHit = getAcademicHostSignal(host);
    const scholarlyLaneHit = sourceUsesScholarlyLane(source);

    if (providerBiasHit) return 0.12;
    if (providerHit) return 0.09;
    if (academicHostHit) return 0.07;
    if (scholarlyLaneHit) return 0.05;
    return 0;
};

const appendConfiguredMailto = (url, envName) => {
    const contact = normalizeText(process.env[envName]);
    if (contact) {
        url.searchParams.set("mailto", contact);
    }
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
    const response = await fetch(url, { ...options, signal });
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
});

const normalizeDate = (value) => {
    const text = normalizeText(value);
    if (!text) return "";
    const parsed = new Date(text);
    return Number.isNaN(parsed.getTime()) ? text : parsed.toISOString();
};

const flattenAuthors = (authors = []) => unique(
    (Array.isArray(authors) ? authors : [])
        .map((author) => normalizeText(author?.name || author?.author?.display_name || author?.display_name))
        .filter(Boolean),
).slice(0, 12);

const extractDoi = (value = "") => {
    const match = String(value || "").match(/\b10\.\d{4,9}\/[-._;()/:A-Z0-9]+\b/i);
    return match ? match[0].replace(/[)>.,;]+$/, "") : "";
};

const extractRepoLinks = (value = "") => unique(
    [...String(value || "").matchAll(/https?:\/\/github\.com\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+/gi)]
        .map((match) => canonicalizeSourceUrl(match[0]))
        .filter(Boolean),
).slice(0, 6);

const getDomainSampleThreshold = (domain = {}) => DOMAIN_SAMPLE_THRESHOLDS[domain?.id] || DOMAIN_SAMPLE_THRESHOLDS.general_research;

const collectSourceText = (source = {}) => normalizeText([
    source?.title || source?.name || source?.displayName || "",
    source?.description || source?.abstract || source?.summary || "",
    source?.metadata && typeof source.metadata === "object" ? JSON.stringify(source.metadata) : "",
].join(" "));

const inferSourceSampleSignals = (source = {}, domain = {}) => {
    const values = [];
    const text = collectSourceText(source);
    for (const match of text.matchAll(SAMPLE_SIZE_RE)) {
        const resolved = Number(match[1] || match[2] || match[3] || 0);
        if (resolved > 0) values.push(resolved);
    }

    const metadataSample = Number(
        source?.metadata?.sampleSize
        || source?.metadata?.sample_size
        || source?.metadata?.participants
        || source?.metadata?.subjects
        || 0,
    );
    if (metadataSample > 0) values.push(metadataSample);

    const sampleSizeCandidates = unique(values.map((value) => Number(value)).filter((value) => value > 0))
        .sort((left, right) => right - left)
        .slice(0, 6);
    const sampleSize = sampleSizeCandidates[0] || null;
    const sampleSizeThreshold = getDomainSampleThreshold(domain);
    const sampleSizeAdequacy = sampleSize ? Number((sampleSize / sampleSizeThreshold).toFixed(2)) : null;
    return {
        sampleSize,
        sampleSizeCandidates,
        sampleSizeThreshold,
        sampleSizeAdequacy,
        sampleSizeFlag: Boolean(sampleSize && sampleSize < sampleSizeThreshold),
        sampleSizeConfidence: sampleSize
            ? Number((metadataSample > 0 ? 0.9 : Math.min(0.75, 0.45 + (sampleSizeCandidates.length * 0.08))).toFixed(2))
            : 0,
    };
};

const buildSourceSafetySignals = (source = {}) => {
    const text = collectSourceText(source);
    const flags = [];
    if (source?.isRetracted) {
        flags.push({
            id: "retracted",
            severity: "critical",
            label: "Retracted paper",
            detail: normalizeText(source?.retractionReason || "Retraction signal present."),
        });
    }
    if (FUNDING_RE.test(text)) {
        flags.push({
            id: "funding_conflict",
            severity: "medium",
            label: "Funding conflict signal",
            detail: "Funding or conflict-of-interest language detected.",
        });
    }
    if (PREDATORY_RE.test(text) || Boolean(source?.metadata?.predatory || source?.metadata?.isPredatory)) {
        flags.push({
            id: "predatory_candidate",
            severity: "high",
            label: "Predatory venue candidate",
            detail: "Predatory-journal or questionable-publisher signal detected.",
        });
    }
    if (DUAL_USE_RE.test(text)) {
        flags.push({
            id: "dual_use",
            severity: "high",
            label: "Dual-use risk",
            detail: "Dual-use or misuse-sensitive terminology detected.",
        });
    }
    if (NEAR_THRESHOLD_P_RE.test(text) || ((text.match(/\bp\s*(?:=|<|>)\s*0?\.\d+/gi) || []).length >= 4)) {
        flags.push({
            id: "statistical_manipulation_candidate",
            severity: "medium",
            label: "Statistical manipulation candidate",
            detail: "Borderline p-values or dense significance-only reporting detected.",
        });
    }

    const dedupedFlags = uniqueBy(flags, (flag) => flag.id);
    return {
        flags: dedupedFlags,
        summary: dedupedFlags.map((flag) => flag.id),
        hasCritical: dedupedFlags.some((flag) => flag.severity === "critical"),
    };
};

const classifySourceStance = (source = {}) => {
    const text = collectSourceText(source);
    if (MIXED_FINDING_RE.test(text)) return "mixed";
    if (NEGATIVE_FINDING_RE.test(text)) return "contrary";
    if (POSITIVE_FINDING_RE.test(text)) return "supportive";
    return "neutral";
};

const normalizeResearchSource = (source = {}) => {
    const url = canonicalizeSourceUrl(source.url || source.pdfUrl || source.fullTextUrl || source.htmlUrl || source.repoUrl || "");
    const title = normalizeText(source.title || source.name || source.displayName);
    if (!url || !title) return null;

    const normalized = {
        title,
        url,
        description: normalizeText(source.description || source.abstract || source.summary || ""),
        provider: normalizeText(source.provider || source.source || "research"),
        providerLabel: normalizeText(source.providerLabel || source.provider || source.source || "Research"),
        kind: normalizeText(source.kind || source.type || "web") || "web",
        doi: normalizeText(source.doi || extractDoi(source.url) || extractDoi(source.description)),
        authors: flattenAuthors(source.authors),
        publishedAt: normalizeDate(source.publishedAt || source.published || source.date),
        citationCount: Math.max(0, Number(source.citationCount || source.citedByCount || 0) || 0),
        isRetracted: Boolean(source.isRetracted),
        retractionReason: normalizeText(source.retractionReason),
        venue: normalizeText(source.venue || source.journal || source.publicationTitle || source.containerTitle || source.container_title || source.metadata?.venue),
        journal: normalizeText(source.journal || source.publicationTitle || source.containerTitle || source.container_title || source.metadata?.journal),
        publisher: normalizeText(source.publisher || source.metadata?.publisher),
        openAccessUrl: canonicalizeSourceUrl(source.openAccessUrl || source.fullTextUrl || source.pdfUrl || ""),
        pdfUrl: canonicalizeSourceUrl(source.pdfUrl || ""),
        repoUrls: unique([
            ...(Array.isArray(source.repoUrls) ? source.repoUrls : []),
            ...(source.repoUrl ? [source.repoUrl] : []),
            ...extractRepoLinks(source.description || ""),
        ].map((value) => canonicalizeSourceUrl(value)).filter(Boolean)).slice(0, 6),
        citedByApiUrl: normalizeText(source.citedByApiUrl),
        queryVariant: normalizeText(source.queryVariant),
        queryVariants: unique([
            ...(Array.isArray(source.queryVariants) ? source.queryVariants : []),
            source.queryVariant,
        ].map((value) => normalizeText(value)).filter(Boolean)).slice(0, 8),
        metadata: source.metadata && typeof source.metadata === "object" ? source.metadata : {},
    };

    const sampleSignals = inferSourceSampleSignals(normalized);
    const safetySignals = buildSourceSafetySignals(normalized);

    return {
        ...normalized,
        sampleSize: sampleSignals.sampleSize,
        sampleSizeCandidates: sampleSignals.sampleSizeCandidates,
        sampleSizeConfidence: sampleSignals.sampleSizeConfidence,
        safetyFlags: safetySignals.flags,
        safetySummary: safetySignals.summary,
        contradictionHint: classifySourceStance(normalized),
    };
};

const attachLaneMetadata = (result = {}, laneQuery = "", laneIndex = 0) => ({
    ...result,
    queryVariant: normalizeText(result?.queryVariant || laneQuery),
    queryVariants: unique([
        ...(Array.isArray(result?.queryVariants) ? result.queryVariants : []),
        result?.queryVariant,
        laneQuery,
    ].map((value) => normalizeText(value)).filter(Boolean)).slice(0, 8),
    metadata: {
        ...(result?.metadata && typeof result.metadata === "object" ? result.metadata : {}),
        laneIndex,
        queryLane: laneQuery,
        queryLanes: unique([
            ...((((result?.metadata && typeof result.metadata === "object") ? result.metadata : {}).queryLanes) || []),
            laneQuery,
        ].map((value) => normalizeText(value)).filter(Boolean)).slice(0, 8),
    },
});

const runProviderBatch = async ({
    laneQueries = [],
    providerTasks = [],
    laneLimit = 8,
    sequential = false,
    errorPrefix = "",
}) => {
    const merged = [];
    const providerErrors = [];
    const tasks = Array.isArray(providerTasks) ? providerTasks : [];
    const prefix = normalizeText(errorPrefix);

    if (sequential) {
        for (let laneIndex = 0; laneIndex < laneQueries.length; laneIndex += 1) {
            const laneQuery = laneQueries[laneIndex];
            for (const task of tasks) {
                try {
                    const results = await task.run(laneQuery, laneLimit);
                    merged.push(...(Array.isArray(results) ? results : []).map((result) => attachLaneMetadata(result, laneQuery, laneIndex)));
                } catch (error) {
                    const label = prefix ? `${prefix}:${task.name}` : task.name;
                    providerErrors.push(`${label} ${normalizeText(error?.message || error || "provider failed")}`.trim());
                }
            }
        }

        return {
            merged,
            providerErrors,
        };
    }

    const settled = await Promise.allSettled(
        laneQueries.flatMap((laneQuery, laneIndex) => tasks.map(async (task) => ({
            name: task.name,
            laneQuery,
            laneIndex,
            results: (await task.run(laneQuery, laneLimit)).map((result) => attachLaneMetadata(result, laneQuery, laneIndex)),
        }))),
    );

    for (const item of settled) {
        if (item.status === "fulfilled") {
            merged.push(...item.value.results);
        } else {
            providerErrors.push(normalizeText(item.reason?.message || item.reason || "provider failed"));
        }
    }

    return {
        merged,
        providerErrors,
    };
};

const fetchGenericWebResults = async (query, limit, providerApiKeys = {}) => {
    const raw = await webSearchHandler({ query, maxResults: limit, providerApiKeys });
    if (typeof raw !== "string") return [];
    if (raw.startsWith("Error:")) {
        throw new Error(raw.slice("Error:".length).trim() || "web_search failed");
    }
    if (/^No results\b/i.test(raw)) return [];
    let parsed = [];
    try {
        parsed = JSON.parse(raw);
    } catch {
        parsed = [];
    }
    return (Array.isArray(parsed) ? parsed : []).map((item) => normalizeResearchSource({
        ...item,
        provider: item.provider || item.source || "web_search",
        providerLabel: "Web Search",
        kind: "web",
    })).filter(Boolean);
};

const parseOpenAlexWork = (work) => normalizeResearchSource({
    title: work?.display_name,
    url: work?.primary_location?.landing_page_url || work?.ids?.doi || work?.doi,
    description: work?.abstract_inverted_index
        ? Object.keys(work.abstract_inverted_index).join(" ")
        : (work?.primary_location?.source?.display_name || ""),
    provider: "openalex",
    providerLabel: "OpenAlex",
    kind: "paper",
    doi: work?.doi || work?.ids?.doi,
    authors: (Array.isArray(work?.authorships) ? work.authorships : []).map((authorship) => ({
        name: authorship?.author?.display_name,
    })),
    publishedAt: work?.publication_date || work?.publication_year,
    citationCount: work?.cited_by_count,
    isRetracted: Boolean(work?.is_retracted),
    journal: work?.primary_location?.source?.display_name,
    publisher: work?.primary_location?.source?.host_organization_name,
    openAccessUrl: work?.primary_location?.pdf_url || work?.best_oa_location?.landing_page_url || work?.best_oa_location?.pdf_url,
    pdfUrl: work?.primary_location?.pdf_url || work?.best_oa_location?.pdf_url,
    repoUrls: extractRepoLinks(JSON.stringify(work?.locations || [])),
    citedByApiUrl: work?.cited_by_api_url,
    metadata: {
        openalexId: work?.id,
        type: work?.type,
        venue: work?.primary_location?.source?.display_name,
        publisher: work?.primary_location?.source?.host_organization_name,
        referencedWorks: Array.isArray(work?.referenced_works) ? work.referenced_works.slice(0, 24) : [],
    },
});

const searchOpenAlexWorks = async (query, limit) => {
    const url = new URL(OPENALEX_WORKS_URL);
    url.searchParams.set("search", query);
    url.searchParams.set("per-page", String(limit));
    appendConfiguredMailto(url, "OPENALEX_EMAIL");

    const data = await readJson(url.toString(), {
        headers: {
            "User-Agent": "nub-agent/1.0",
        },
    });

    return (Array.isArray(data?.results) ? data.results : [])
        .map(parseOpenAlexWork)
        .filter(Boolean);
};

const fetchForwardCitationsFromOpenAlex = async (source, limit) => {
    if (!source?.citedByApiUrl) return [];
    const url = new URL(source.citedByApiUrl);
    url.searchParams.set("per-page", String(limit));
    appendConfiguredMailto(url, "OPENALEX_EMAIL");

    const data = await readJson(url.toString(), {
        headers: {
            "User-Agent": "nub-agent/1.0",
        },
    });

    return (Array.isArray(data?.results) ? data.results : [])
        .map(parseOpenAlexWork)
        .map((item) => item ? {
            ...item,
            provider: "openalex_forward_citation",
            providerLabel: "OpenAlex Forward Citation",
            kind: "citation",
        } : null)
        .filter(Boolean);
};

const searchCrossrefWorks = async (query, limit) => {
    const url = new URL(CROSSREF_WORKS_URL);
    url.searchParams.set("query.bibliographic", query);
    url.searchParams.set("rows", String(limit));
    url.searchParams.set("sort", "relevance");
    url.searchParams.set("order", "desc");
    appendConfiguredMailto(url, "CROSSREF_EMAIL");

    const data = await readJson(url.toString(), {
        headers: {
            "User-Agent": "nub-agent/1.0",
        },
    });

    return (Array.isArray(data?.message?.items) ? data.message.items : [])
        .map((item) => normalizeResearchSource({
            title: Array.isArray(item?.title) ? item.title[0] : item?.title,
            url: Array.isArray(item?.link) && item.link[0]?.URL ? item.link[0].URL : item?.URL || item?.DOI ? `https://doi.org/${item.DOI}` : "",
            description: Array.isArray(item?.container-title) ? item["container-title"][0] : "",
            provider: "crossref",
            providerLabel: "Crossref",
            kind: "paper",
            doi: item?.DOI,
            journal: Array.isArray(item?.container-title) ? item["container-title"][0] : "",
            publisher: item?.publisher,
            authors: Array.isArray(item?.author) ? item.author.map((author) => ({
                name: [author?.given, author?.family].filter(Boolean).join(" "),
            })) : [],
            publishedAt: item?.created?.["date-time"] || item?.issued?.["date-parts"]?.[0]?.join("-"),
            citationCount: item?.["is-referenced-by-count"],
            metadata: {
                type: item?.type,
                publisher: item?.publisher,
                venue: Array.isArray(item?.container-title) ? item["container-title"][0] : "",
            },
        }))
        .filter(Boolean);
};

const searchGitHubRepositories = async (query, limit) => {
    const url = new URL(GITHUB_REPO_SEARCH_URL);
    url.searchParams.set("q", `${query} in:name,description,readme`);
    url.searchParams.set("sort", "stars");
    url.searchParams.set("order", "desc");
    url.searchParams.set("per_page", String(limit));

    const headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "nub-agent/1.0",
    };
    if (process.env.GITHUB_TOKEN) {
        headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
    }

    const data = await readJson(url.toString(), { headers });
    return (Array.isArray(data?.items) ? data.items : [])
        .map((item) => normalizeResearchSource({
            title: item?.full_name,
            url: item?.html_url,
            description: item?.description,
            provider: "github",
            providerLabel: "GitHub",
            kind: "repository",
            authors: [{ name: item?.owner?.login }],
            publishedAt: item?.updated_at,
            citationCount: item?.stargazers_count,
            repoUrl: item?.html_url,
            metadata: {
                language: item?.language,
                stars: item?.stargazers_count,
                forks: item?.forks_count,
                archived: Boolean(item?.archived),
                openIssues: item?.open_issues_count,
            },
        }))
        .filter(Boolean);
};

const searchPapersWithCode = async (query, limit) => {
    const url = new URL(PWC_SEARCH_URL);
    url.searchParams.set("search", query);
    url.searchParams.set("items_per_page", String(limit));

    const data = await readJson(url.toString(), {
        headers: {
            "User-Agent": "nub-agent/1.0",
            "Accept": "application/json",
        },
    });

    const items = Array.isArray(data?.results) ? data.results : Array.isArray(data?.data) ? data.data : [];
    return items.map((item) => normalizeResearchSource({
        title: item?.title,
        url: item?.url_abs || item?.paper_url || item?.url_pdf,
        description: item?.abstract || item?.summary,
        provider: "papers_with_code",
        providerLabel: "Papers With Code",
        kind: "paper",
        doi: item?.doi,
        authors: Array.isArray(item?.authors) ? item.authors.map((name) => ({ name })) : [],
        publishedAt: item?.published,
        journal: item?.conference || item?.journal,
        pdfUrl: item?.url_pdf,
        repoUrls: Array.isArray(item?.repositories) ? item.repositories.map((repo) => repo?.url) : [],
    })).filter(Boolean);
};

const searchIeeeXplore = async (query, limit) => {
    if (!process.env.IEEE_XPLORE_API_KEY) return [];
    const url = new URL(IEEE_XPLORE_SEARCH_URL);
    url.searchParams.set("apikey", process.env.IEEE_XPLORE_API_KEY);
    url.searchParams.set("format", "json");
    url.searchParams.set("max_records", String(limit));
    url.searchParams.set("sort_order", "desc");
    url.searchParams.set("sort_field", "relevance");
    url.searchParams.set("querytext", query);

    const data = await readJson(url.toString(), {
        headers: {
            "User-Agent": "nub-agent/1.0",
        },
    });

    return (Array.isArray(data?.articles) ? data.articles : [])
        .map((item) => normalizeResearchSource({
            title: item?.title,
            url: item?.html_url || item?.pdf_url || item?.doi ? `https://doi.org/${item.doi}` : "",
            description: item?.abstract,
            provider: "ieee_xplore",
            providerLabel: "IEEE Xplore",
            kind: "paper",
            doi: item?.doi,
            journal: item?.publication_title,
            authors: Array.isArray(item?.authors?.authors) ? item.authors.authors.map((author) => ({
                name: author?.full_name,
            })) : [],
            publishedAt: item?.publication_date,
            pdfUrl: item?.pdf_url,
            metadata: {
                publicationTitle: item?.publication_title,
                articleNumber: item?.article_number,
                publisher: item?.publisher,
            },
        }))
        .filter(Boolean);
};

const searchConfiguredScholarlyProvider = async (envPrefix, query, limit) => {
    const baseUrl = normalizeText(process.env[`${envPrefix}_SEARCH_API_URL`]);
    if (!baseUrl) return [];

    const url = new URL(baseUrl);
    url.searchParams.set("query", query);
    url.searchParams.set("limit", String(limit));

    const headers = {
        "Accept": "application/json",
        "User-Agent": "nub-agent/1.0",
    };
    const apiKey = normalizeText(process.env[`${envPrefix}_API_KEY`]);
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

    const data = await readJson(url.toString(), { headers });
    const items = Array.isArray(data?.results) ? data.results : Array.isArray(data?.items) ? data.items : [];
    return items.map((item) => normalizeResearchSource({
        title: item?.title,
        url: item?.url || item?.landing_page_url || item?.pdf_url,
        description: item?.description || item?.abstract || item?.summary,
        provider: envPrefix.toLowerCase(),
        providerLabel: envPrefix.replace(/_/g, " "),
        kind: item?.kind || item?.type || "paper",
        doi: item?.doi,
        journal: item?.journal || item?.publication_title,
        publisher: item?.publisher,
        authors: Array.isArray(item?.authors) ? item.authors : [],
        publishedAt: item?.published_at || item?.date,
        pdfUrl: item?.pdf_url,
    })).filter(Boolean);
};

const fetchUnpaywallLink = async (source) => {
    if (!source?.doi || !process.env.UNPAYWALL_EMAIL) return source;
    const url = new URL(`https://api.unpaywall.org/v2/${encodeURIComponent(source.doi)}`);
    url.searchParams.set("email", process.env.UNPAYWALL_EMAIL);

    try {
        const data = await readJson(url.toString(), {
            headers: {
                "User-Agent": "nub-agent/1.0",
            },
        });
        return {
            ...source,
            openAccessUrl: canonicalizeSourceUrl(data?.best_oa_location?.url || data?.best_oa_location?.url_for_pdf || source.openAccessUrl),
            pdfUrl: canonicalizeSourceUrl(data?.best_oa_location?.url_for_pdf || source.pdfUrl),
            metadata: {
                ...source.metadata,
                isOa: Boolean(data?.is_oa),
                oaStatus: normalizeText(data?.oa_status),
            },
        };
    } catch {
        return source;
    }
};

const applyRetractionSignals = async (sources = []) => {
    const baseUrl = normalizeText(process.env.RETRACTION_WATCH_API_URL);
    if (!baseUrl) return sources;

    const enriched = [];
    for (const source of sources) {
        if (!source?.doi && !source?.title) {
            enriched.push(source);
            continue;
        }

        try {
            const url = new URL(baseUrl);
            if (source.doi) url.searchParams.set("doi", source.doi);
            else url.searchParams.set("title", source.title);
            const data = await readJson(url.toString(), {
                headers: {
                    "Accept": "application/json",
                    "User-Agent": "nub-agent/1.0",
                },
            });
            const record = Array.isArray(data?.results) ? data.results[0] : data?.result || data;
            enriched.push({
                ...source,
                isRetracted: Boolean(source.isRetracted || record?.retracted || record?.is_retracted),
                retractionReason: normalizeText(record?.reason || record?.retraction_reason || source.retractionReason),
            });
        } catch {
            enriched.push(source);
        }
    }

    return enriched;
};

const dedupeSources = (sources = [], limit = 120) => {
    const seen = new Map();
    const output = [];

    for (const source of sources) {
        const normalized = normalizeResearchSource(source);
        if (!normalized) continue;
        const key = normalized.url || `${normalized.title}:${normalized.provider}`;
        if (seen.has(key)) {
            const index = seen.get(key);
            const mergedFlags = uniqueBy([
                ...(Array.isArray(output[index].safetyFlags) ? output[index].safetyFlags : []),
                ...(Array.isArray(normalized.safetyFlags) ? normalized.safetyFlags : []),
            ], (flag) => flag.id);
            output[index] = {
                ...output[index],
                provider: unique([output[index].provider, normalized.provider]).join("+"),
                providerCount: Math.max(1, Number(output[index].providerCount || 1)) + 1,
                repoUrls: unique([...(output[index].repoUrls || []), ...(normalized.repoUrls || [])]).slice(0, 6),
                citationCount: Math.max(Number(output[index].citationCount || 0), Number(normalized.citationCount || 0)),
                authors: unique([...(output[index].authors || []), ...(normalized.authors || [])]).slice(0, 12),
                doi: output[index].doi || normalized.doi,
                openAccessUrl: output[index].openAccessUrl || normalized.openAccessUrl,
                pdfUrl: output[index].pdfUrl || normalized.pdfUrl,
                isRetracted: output[index].isRetracted || normalized.isRetracted,
                retractionReason: output[index].retractionReason || normalized.retractionReason,
                venue: output[index].venue || normalized.venue,
                journal: output[index].journal || normalized.journal,
                publisher: output[index].publisher || normalized.publisher,
                providerLabel: unique([output[index].providerLabel, normalized.providerLabel]).join(" / "),
                queryVariant: output[index].queryVariant || normalized.queryVariant,
                queryVariants: unique([
                    ...(output[index].queryVariants || []),
                    ...(normalized.queryVariants || []),
                    output[index].queryVariant,
                    normalized.queryVariant,
                ].map((value) => normalizeText(value)).filter(Boolean)).slice(0, 8),
                metadata: {
                    ...(output[index].metadata || {}),
                    ...(normalized.metadata || {}),
                    queryLanes: unique([
                        ...((output[index].metadata || {}).queryLanes || []),
                        ...((normalized.metadata || {}).queryLanes || []),
                        ...(output[index].queryVariants || []),
                        ...(normalized.queryVariants || []),
                    ].map((value) => normalizeText(value)).filter(Boolean)).slice(0, 8),
                },
                sampleSize: Math.max(Number(output[index].sampleSize || 0), Number(normalized.sampleSize || 0)) || null,
                sampleSizeCandidates: unique([...(output[index].sampleSizeCandidates || []), ...(normalized.sampleSizeCandidates || [])])
                    .map((value) => Number(value))
                    .filter((value) => value > 0)
                    .sort((left, right) => right - left)
                    .slice(0, 6),
                sampleSizeConfidence: Math.max(Number(output[index].sampleSizeConfidence || 0), Number(normalized.sampleSizeConfidence || 0)),
                safetyFlags: mergedFlags,
                safetySummary: mergedFlags.map((flag) => flag.id),
                contradictionHint: output[index].contradictionHint === "mixed"
                    ? "mixed"
                    : (normalized.contradictionHint === "mixed"
                        ? "mixed"
                        : output[index].contradictionHint || normalized.contradictionHint),
            };
            continue;
        }
        seen.set(key, output.length);
        output.push({
            ...normalized,
            providerCount: 1,
        });
        if (output.length >= limit) break;
    }

    return output;
};

const buildAuthorNetwork = (sources = []) => {
    const authorCounts = new Map();
    const coauthorEdges = new Map();

    for (const source of sources) {
        const authors = unique((Array.isArray(source?.authors) ? source.authors : []).map(normalizeText).filter(Boolean)).slice(0, 8);
        for (const author of authors) {
            authorCounts.set(author, (authorCounts.get(author) || 0) + 1);
        }
        for (let index = 0; index < authors.length; index += 1) {
            for (let inner = index + 1; inner < authors.length; inner += 1) {
                const edgeKey = [authors[index], authors[inner]].sort().join("::");
                coauthorEdges.set(edgeKey, (coauthorEdges.get(edgeKey) || 0) + 1);
            }
        }
    }

    return {
        topAuthors: [...authorCounts.entries()]
            .sort((left, right) => right[1] - left[1])
            .slice(0, 12)
            .map(([name, appearances]) => ({ name, appearances })),
        edges: [...coauthorEdges.entries()]
            .sort((left, right) => right[1] - left[1])
            .slice(0, 20)
            .map(([edge, weight]) => {
                const [left, right] = edge.split("::");
                return { left, right, weight };
            }),
    };
};

const buildTemporalTrends = (sources = []) => {
    const buckets = new Map();
    for (const source of sources) {
        const year = String(source?.publishedAt || "").slice(0, 4);
        if (!/^\d{4}$/.test(year)) continue;
        buckets.set(year, (buckets.get(year) || 0) + 1);
    }

    const points = [...buckets.entries()]
        .sort((left, right) => Number(left[0]) - Number(right[0]))
        .map(([year, count]) => ({ year, count }));
    const recent = points.slice(-3).reduce((sum, point) => sum + point.count, 0);
    const prior = points.slice(-6, -3).reduce((sum, point) => sum + point.count, 0);

    return {
        points,
        direction: recent > prior ? "emerging" : recent < prior ? "saturating" : "steady",
    };
};

const applyFieldRecencyDecay = (source, domain) => {
    const halfLife = Math.max(1, Number(domain?.recencyHalfLifeYears) || 4);
    const publishedAt = normalizeDate(source?.publishedAt);
    if (!publishedAt) return 0.92;
    const ageYears = Math.max(0, (Date.now() - new Date(publishedAt).getTime()) / (365.25 * 24 * 60 * 60 * 1000));
    if (!Number.isFinite(ageYears)) return 0.92;
    return Math.max(0.3, Math.pow(0.5, ageYears / halfLife));
};

const buildCounterevidenceMap = (sources = []) => {
    const byHint = {
        supportive: [],
        contrary: [],
        mixed: [],
        neutral: [],
    };
    for (const source of Array.isArray(sources) ? sources : []) {
        const hint = source?.contradictionHint || "neutral";
        if (!byHint[hint]) byHint[hint] = [];
        byHint[hint].push({
            title: source.title,
            url: source.url,
            tier: source.tier,
            epistemicScore: source.epistemicScore,
        });
    }

    const conflictClusters = [];
    if (byHint.supportive.length && byHint.contrary.length) {
        conflictClusters.push({
            type: "stance_conflict",
            supportive: byHint.supportive.slice(0, 6),
            contrary: byHint.contrary.slice(0, 6),
        });
    }
    if (byHint.mixed.length) {
        conflictClusters.push({
            type: "mixed_findings",
            sources: byHint.mixed.slice(0, 8),
        });
    }

    return {
        stanceCounts: Object.fromEntries(Object.entries(byHint).map(([key, value]) => [key, value.length])),
        conflictClusters,
    };
};

const scoreTieredSources = (query, sources = [], plan = {}) => {
    const ranked = rankSourcesWithRag(query, sources);
    const terms = extractQueryTerms(query, 32);

    return ranked.map((source, index) => {
        const sampleSignals = inferSourceSampleSignals(source, plan?.domain);
        const safetySignals = buildSourceSafetySignals({
            ...source,
            safetyFlags: source?.safetyFlags,
        });
        const lexical = scoreTextForTerms([source.title, source.description, source.url].join(" "), terms);
        const authority = domainAuthorityBoost(source.url);
        const providerBoost = Math.min(0.2, (Number(source.providerCount || 1) - 1) * 0.06);
        const citationBoost = Math.min(0.24, Math.log10((Number(source.citationCount || 0) || 0) + 1) * 0.08);
        const freshnessFactor = applyFieldRecencyDecay(source, plan.domain);
        const retractionPenalty = source.isRetracted ? 0.55 : 0;
        const repoBoost = (Array.isArray(source.repoUrls) && source.repoUrls.length) ? 0.06 : 0;
        const scholarlyBoost = getScholarlyBoost(source, plan);
        const sampleSizePenalty = sampleSignals.sampleSize
            ? clamp((1 - Math.min(1, sampleSignals.sampleSizeAdequacy || 0)) * 0.16, 0, 0.16)
            : (source.kind === "paper" && ["biomedical", "social_science", "interdisciplinary"].includes(plan?.domain?.id) ? 0.03 : 0);
        const sampleSizeBoost = sampleSignals.sampleSizeAdequacy && sampleSignals.sampleSizeAdequacy > 1
            ? Math.min(0.08, (sampleSignals.sampleSizeAdequacy - 1) * 0.04)
            : 0;
        const predatoryPenalty = safetySignals.summary.includes("predatory_candidate") ? 0.3 : 0;
        const fundingPenalty = safetySignals.summary.includes("funding_conflict") && plan?.scope?.id === "contested_topic" ? 0.05 : 0;
        const manipulationPenalty = safetySignals.summary.includes("statistical_manipulation_candidate") ? 0.08 : 0;
        const artifactAdjustment = clamp(
            Number(plan?.sourceAdjustments?.[source.url]?.scoreDelta || 0)
            + Number(plan?.sourceQuality?.byUrl?.[source.url]?.scoreDelta || 0),
            -0.2,
            0.2,
        );
        const normalizedRank = 1 - (index / Math.max(1, ranked.length));
        const quarantineReason = source.isRetracted
            ? `Retracted paper${source.retractionReason ? `: ${source.retractionReason}` : ""}`
            : (safetySignals.summary.includes("predatory_candidate") ? "Predatory venue signal" : "");
        const rawScore = (
            (normalizedRank * 0.42)
            + Math.min(0.28, lexical * 0.02)
            + Math.min(0.18, authority * 0.06)
            + providerBoost
            + citationBoost
            + repoBoost
            + scholarlyBoost
            + sampleSizeBoost
            + artifactAdjustment
        ) * freshnessFactor - retractionPenalty - predatoryPenalty - fundingPenalty - manipulationPenalty - sampleSizePenalty;
        const score = quarantineReason ? 0 : Math.max(0, Math.min(1, Number(rawScore.toFixed(2))));
        const tier = quarantineReason
            ? "discard"
            : score >= 0.8 ? "core" : score >= 0.55 ? "supporting" : score >= 0.35 ? "peripheral" : "discard";
        const weightMultiplier = tier === "core" ? 2 : tier === "supporting" ? 1 : tier === "peripheral" ? 0.5 : 0;
        const verdictRationale = [
            `rank:${normalizedRank.toFixed(2)}`,
            `freshness:${freshnessFactor.toFixed(2)}`,
            `citations:${Number(source.citationCount || 0)}`,
            scholarlyBoost ? `scholarly:${scholarlyBoost.toFixed(2)}` : "",
            sampleSignals.sampleSize ? `sample:${sampleSignals.sampleSize}/${sampleSignals.sampleSizeThreshold}` : "sample:unknown",
            safetySignals.summary.length ? `flags:${safetySignals.summary.join("|")}` : "flags:none",
            quarantineReason ? `quarantine:${quarantineReason}` : "",
        ].filter(Boolean);

        return {
            ...source,
            epistemicScore: score,
            tier,
            fetchMode: tier === "core" || tier === "supporting" ? "full" : tier === "peripheral" ? "light" : "skip",
            freshnessFactor: Number(freshnessFactor.toFixed(2)),
            sampleSize: sampleSignals.sampleSize,
            sampleSizeCandidates: sampleSignals.sampleSizeCandidates,
            sampleSizeThreshold: sampleSignals.sampleSizeThreshold,
            sampleSizeAdequacy: sampleSignals.sampleSizeAdequacy,
            sampleSizeFlag: sampleSignals.sampleSizeFlag,
            sampleSizeConfidence: sampleSignals.sampleSizeConfidence,
            safetyFlags: safetySignals.flags,
            safetySummary: safetySignals.summary,
            contradictionHint: source.contradictionHint || classifySourceStance(source),
            quarantineReason,
            auditStatus: quarantineReason ? "quarantined" : "active",
            weightMultiplier,
            verdictRationale,
            scoreComponents: {
                normalizedRank: Number(normalizedRank.toFixed(2)),
                lexical: Number(lexical.toFixed(2)),
                authority: Number(authority.toFixed(2)),
                providerBoost: Number(providerBoost.toFixed(2)),
                citationBoost: Number(citationBoost.toFixed(2)),
                repoBoost: Number(repoBoost.toFixed(2)),
                scholarlyBoost: Number(scholarlyBoost.toFixed(2)),
                sampleSizeBoost: Number(sampleSizeBoost.toFixed(2)),
                sampleSizePenalty: Number(sampleSizePenalty.toFixed(2)),
                fundingPenalty: Number(fundingPenalty.toFixed(2)),
                manipulationPenalty: Number(manipulationPenalty.toFixed(2)),
                artifactAdjustment: Number(artifactAdjustment.toFixed(2)),
                predatoryPenalty: Number(predatoryPenalty.toFixed(2)),
                retractionPenalty: Number(retractionPenalty.toFixed(2)),
            },
        };
    });
};

const searchResearchSources = async ({
    query,
    plan,
    maxResults = 24,
    providerApiKeys = {},
}) => {
    const limit = clamp(maxResults, 4, 48);
    const scholarlyRouting = getScholarlyRoutingSignals(plan);
    const laneQueries = unique([
        normalizeText(query),
        ...((Array.isArray(plan?.searchQueries) ? plan.searchQueries : []).map((value) => normalizeText(value))),
        ...scholarlyRouting.scholarlyLanes,
        ...((Array.isArray(plan?.queryMatrix?.versions) ? plan.queryMatrix.versions.map((version) => version?.query) : []).map((value) => normalizeText(value))),
    ].filter(Boolean)).slice(0, plan?.depthPreference === "deep" ? (scholarlyRouting.active ? 7 : 6) : plan?.depthPreference === "speed" ? (scholarlyRouting.active ? 3 : 2) : (scholarlyRouting.active ? 5 : 4));
    const laneLimit = Math.max(4, Math.ceil(limit / Math.max(1, laneQueries.length)) + 2);
    const providerTasks = [
        { name: "web_search", enabled: true, run: (laneQuery, taskLimit) => fetchGenericWebResults(laneQuery, taskLimit, providerApiKeys) },
        { name: "openalex", enabled: true, run: (laneQuery, taskLimit) => searchOpenAlexWorks(laneQuery, Math.min(taskLimit, 16)) },
        { name: "crossref", enabled: true, run: (laneQuery, taskLimit) => searchCrossrefWorks(laneQuery, Math.min(taskLimit, 16)) },
        { name: "github", enabled: plan?.domain?.id === "cs_ml" || /\b(code|implementation|repo|repository|github|benchmark)\b/i.test(query), run: (laneQuery, taskLimit) => searchGitHubRepositories(laneQuery, Math.min(taskLimit, 10)) },
        { name: "papers_with_code", enabled: plan?.domain?.id === "cs_ml", run: (laneQuery, taskLimit) => searchPapersWithCode(laneQuery, Math.min(taskLimit, 10)) },
        { name: "ieee_xplore", enabled: Boolean(process.env.IEEE_XPLORE_API_KEY), run: (laneQuery, taskLimit) => searchIeeeXplore(laneQuery, Math.min(taskLimit, 12)) },
        { name: "acm", enabled: Boolean(process.env.ACM_SEARCH_API_URL), run: (laneQuery, taskLimit) => searchConfiguredScholarlyProvider("ACM", laneQuery, Math.min(taskLimit, 12)) },
        { name: "jstor", enabled: Boolean(process.env.JSTOR_SEARCH_API_URL), run: (laneQuery, taskLimit) => searchConfiguredScholarlyProvider("JSTOR", laneQuery, Math.min(taskLimit, 12)) },
    ];
    const enabledProviderTasks = providerTasks.filter((task) => task.enabled);
    const initialBatch = await runProviderBatch({
        laneQueries,
        providerTasks: enabledProviderTasks,
        laneLimit,
    });

    const providerErrors = [...initialBatch.providerErrors];
    let merged = [...initialBatch.merged];
    if (!merged.length) {
        const recoveryBatch = await runProviderBatch({
            laneQueries: unique([normalizeText(query), ...laneQueries]).filter(Boolean).slice(0, 2),
            providerTasks: enabledProviderTasks.filter((task) => ["openalex", "crossref", "web_search"].includes(task.name)),
            laneLimit: Math.min(6, laneLimit),
            sequential: true,
            errorPrefix: "recovery",
        });
        merged = recoveryBatch.merged;
        providerErrors.push(...recoveryBatch.providerErrors);
    }

    const deduped = dedupeSources(merged, limit * 3);
    const oaResolved = await Promise.all(deduped.map((source) => fetchUnpaywallLink(source)));
    const retractionResolved = await applyRetractionSignals(oaResolved);
    const initialRanked = scoreTieredSources(query, retractionResolved, plan);

    const forwardCitationSeeds = initialRanked
        .filter((source) => source.provider.startsWith("openalex") && source.tier !== "discard" && source.citedByApiUrl)
        .slice(0, MAX_FORWARD_CITATION_SEEDS);
    const forwardCitations = await Promise.allSettled(
        forwardCitationSeeds.map((source) => fetchForwardCitationsFromOpenAlex(source, Math.min(6, limit / 2))),
    );

    const forwardCitationResults = forwardCitations.flatMap((item) => (
        item.status === "fulfilled" ? item.value : []
    ));
    const sourceSet = scoreTieredSources(query, dedupeSources([...retractionResolved, ...forwardCitationResults], limit * 4), plan)
        .slice(0, limit * 2);

    const authorNetwork = buildAuthorNetwork(sourceSet);
    const temporalTrends = buildTemporalTrends(sourceSet);
    const counterevidence = buildCounterevidenceMap(sourceSet);
    const tierCounts = sourceSet.reduce((accumulator, source) => {
        accumulator[source.tier] = (accumulator[source.tier] || 0) + 1;
        return accumulator;
    }, { core: 0, supporting: 0, peripheral: 0, discard: 0 });
    const quarantined = sourceSet
        .filter((source) => source.quarantineReason)
        .map((source) => ({
            title: source.title,
            url: source.url,
            reason: source.quarantineReason,
        }));
    const sampleSizeTelemetry = {
        threshold: getDomainSampleThreshold(plan?.domain),
        flagged: sourceSet.filter((source) => source.sampleSizeFlag).length,
        known: sourceSet.filter((source) => Number(source.sampleSize) > 0).length,
    };
    const safetyTelemetry = {
        fundingConflictCount: sourceSet.filter((source) => source.safetySummary?.includes("funding_conflict")).length,
        predatoryCandidateCount: sourceSet.filter((source) => source.safetySummary?.includes("predatory_candidate")).length,
        dualUseCount: sourceSet.filter((source) => source.safetySummary?.includes("dual_use")).length,
        manipulationCount: sourceSet.filter((source) => source.safetySummary?.includes("statistical_manipulation_candidate")).length,
        retractedCount: sourceSet.filter((source) => source.safetySummary?.includes("retracted")).length,
    };

    return {
        sources: sourceSet,
        citationIntelligence: {
            forwardCitations: forwardCitationResults.slice(0, limit),
            seedCount: forwardCitationSeeds.length,
        },
        authorNetwork,
        temporalTrends,
        counterevidence,
        providerErrors,
        providersUsed: unique(sourceSet.map((source) => source.provider)),
        searchLanes: laneQueries.map((laneQuery, laneIndex) => ({
            laneIndex,
            query: laneQuery,
            laneType: scholarlyRouting.scholarlyLanes.includes(normalizeText(laneQuery)) ? "scholarly" : "general",
        })),
        auditLog: {
            tierCounts,
            quarantined,
            sampleSizeTelemetry,
            safetyTelemetry,
            scholarlyRouting: {
                active: scholarlyRouting.active,
                laneCount: scholarlyRouting.scholarlyLanes.length,
                providerBias: scholarlyRouting.providerBias,
            },
        },
    };
};

module.exports = {
    getDomainSampleThreshold,
    inferSourceSampleSignals,
    buildSourceSafetySignals,
    extractDoi,
    extractRepoLinks,
    normalizeResearchSource,
    scoreTieredSources,
    searchResearchSources,
    buildAuthorNetwork,
    buildTemporalTrends,
};
