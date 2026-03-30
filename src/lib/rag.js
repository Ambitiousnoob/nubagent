/**
 * Retrieval: lexical re-ranking and query-focused passage selection (client-side).
 */

const STOP = new Set([
  "a",
  "an",
  "the",
  "and",
  "or",
  "but",
  "if",
  "in",
  "on",
  "at",
  "to",
  "for",
  "of",
  "as",
  "is",
  "was",
  "are",
  "were",
  "be",
  "been",
  "being",
  "have",
  "has",
  "had",
  "do",
  "does",
  "did",
  "will",
  "would",
  "could",
  "should",
  "may",
  "might",
  "must",
  "can",
  "with",
  "from",
  "by",
  "about",
  "into",
  "through",
  "during",
  "before",
  "after",
  "above",
  "below",
  "between",
  "under",
  "again",
  "then",
  "once",
  "here",
  "there",
  "when",
  "where",
  "why",
  "how",
  "all",
  "each",
  "every",
  "both",
  "few",
  "more",
  "most",
  "other",
  "some",
  "such",
  "no",
  "nor",
  "not",
  "only",
  "own",
  "same",
  "so",
  "than",
  "too",
  "very",
  "just",
  "that",
  "this",
  "these",
  "those",
  "what",
  "which",
  "who",
  "whom",
  "it",
  "its",
  "they",
  "them",
  "their",
  "we",
  "our",
  "you",
  "your",
  "he",
  "she",
  "his",
  "her",
  "my",
  "me",
  "i",
  "find",
  "get",
  "use",
  "using",
]);
const TRACKING_QUERY_PARAMS = new Set([
  "utm_source",
  "utm_medium",
  "utm_campaign",
  "utm_term",
  "utm_content",
  "utm_id",
  "gclid",
  "fbclid",
  "msclkid",
  "ref",
  "ref_src",
  "source",
  "mc_cid",
  "mc_eid",
  "_hsenc",
  "_hsmi",
  "srsltid",
  "ved",
  "ei",
  "usg",
]);

export function canonicalizeSourceUrl(url) {
  const raw = String(url || "").trim();
  if (!raw) return "";

  try {
    const parsed = new URL(raw);
    parsed.hash = "";
    parsed.hostname = parsed.hostname.toLowerCase();

    if (
      (parsed.protocol === "https:" && parsed.port === "443") ||
      (parsed.protocol === "http:" && parsed.port === "80")
    ) {
      parsed.port = "";
    }

    for (const key of [...parsed.searchParams.keys()]) {
      if (TRACKING_QUERY_PARAMS.has(key.toLowerCase())) {
        parsed.searchParams.delete(key);
      }
    }

    if (typeof parsed.searchParams.sort === "function") {
      parsed.searchParams.sort();
    }

    let pathname = parsed.pathname.replace(/\/{2,}/g, "/");
    pathname = pathname.replace(/\/index(?:\.[a-z0-9]+)?$/i, "/");
    parsed.pathname =
      pathname !== "/" ? pathname.replace(/\/+$/, "") || "/" : "/";
    parsed.search = parsed.searchParams.toString()
      ? `?${parsed.searchParams.toString()}`
      : "";
    return parsed.toString();
  } catch {
    return raw;
  }
}

const toUniqueList = (values = []) => [
  ...new Set(
    (Array.isArray(values) ? values : [values])
      .map((value) => String(value || "").trim())
      .filter(Boolean),
  ),
];

const getNumericScore = (value) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
};

const normalizeMatchText = (value) =>
  String(value || "")
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();

const pickLongerText = (current, candidate) => {
  const left = String(current || "").trim();
  const right = String(candidate || "").trim();
  if (!right) return left;
  if (!left) return right;
  return right.length > left.length ? right : left;
};

const normalizeSourcePathForDedupe = (pathname = "") => {
  let next = String(pathname || "").replace(/\/{2,}/g, "/");
  next = next.replace(/\/index(?:\.[a-z0-9]+)?$/i, "/");
  next = next.replace(/\/(?:amp|amphtml)\/?$/i, "/");
  return next !== "/" ? next.replace(/\/+$/, "") || "/" : "/";
};

const normalizeTitleForDedupe = (title = "") =>
  normalizeMatchText(title)
    .split(/\s+/)
    .filter((term) => term.length > 1 && !STOP.has(term))
    .slice(0, 10)
    .join(" ");

const buildNearDuplicateKey = (url, title = "") => {
  try {
    const parsed = new URL(canonicalizeSourceUrl(url));
    if (parsed.search) return "";

    const host = parsed.hostname.replace(/^(?:www\.|m\.)/, "").toLowerCase();
    const path = normalizeSourcePathForDedupe(parsed.pathname);
    const titleKey = normalizeTitleForDedupe(title);

    if (!host || !titleKey) return "";
    return `${host}${path}::${titleKey}`;
  } catch {
    return "";
  }
};

const scoreUrlPreference = (url) => {
  try {
    const parsed = new URL(canonicalizeSourceUrl(url));
    let score = 0;
    if (parsed.protocol === "https:") score += 0.35;
    if (!parsed.search) score += 0.6;
    if (/^(?:www\.|m\.)/i.test(parsed.hostname)) score -= 0.05;
    if (/(^|\/)(amp|amphtml)(\/|$)/i.test(parsed.pathname)) score -= 0.5;
    score -= parsed.toString().length / 500;
    return score;
  } catch {
    return Number.NEGATIVE_INFINITY;
  }
};

const pickPreferredUrl = (current, candidate) => {
  const left = canonicalizeSourceUrl(current);
  const right = canonicalizeSourceUrl(candidate);
  if (!left) return right;
  if (!right) return left;
  if (left === right) return left;

  const leftScore = scoreUrlPreference(left);
  const rightScore = scoreUrlPreference(right);
  if (rightScore !== leftScore) {
    return rightScore > leftScore ? right : left;
  }

  return right.length < left.length ? right : left;
};

export function mergeSourcesByCanonicalUrl(sources = [], options = {}) {
  const list = Array.isArray(sources) ? sources : [];
  const limit = Math.max(1, Number(options.limit) || list.length || 1);
  const mergedByKey = new Map();
  const canonicalUrlToKey = new Map();
  const nearDuplicateToKey = new Map();

  list.forEach((item, position) => {
    const canonicalUrl = canonicalizeSourceUrl(item?.url || "");
    if (!canonicalUrl) return;

    const providers = toUniqueList([
      ...(Array.isArray(item?.providers) ? item.providers : []),
      item?.source,
    ]);
    const queryVariants = toUniqueList([
      ...(Array.isArray(item?.queryVariants) ? item.queryVariants : []),
      item?.queryVariant,
    ]);
    const nearDuplicateKey = buildNearDuplicateKey(
      canonicalUrl,
      item?.title || "",
    );
    const mergeKey =
      canonicalUrlToKey.get(canonicalUrl) ||
      (nearDuplicateKey ? nearDuplicateToKey.get(nearDuplicateKey) : "") ||
      canonicalUrl;
    const existing = mergedByKey.get(mergeKey);
    const next = {
      ...item,
      title: item?.title || getDomain(canonicalUrl),
      url: canonicalUrl,
      description: item?.description || item?.snippet || "",
      date: item?.date || null,
      source: item?.source || null,
      providers,
      providerCount: Math.max(
        Number(item?.providerCount) || 0,
        providers.length || (item?.source ? 1 : 0),
        1,
      ),
      queryVariants,
      queryHitCount: Math.max(
        Number(item?.queryHitCount) || 0,
        queryVariants.length || (item?.queryVariant ? 1 : 0),
        1,
      ),
      rank: Number.isFinite(Number(item?.rank))
        ? Number(item.rank)
        : position + 1,
      _firstSeen: position,
    };

    if (!existing) {
      mergedByKey.set(mergeKey, next);
      canonicalUrlToKey.set(canonicalUrl, mergeKey);
      if (nearDuplicateKey) nearDuplicateToKey.set(nearDuplicateKey, mergeKey);
      return;
    }

    existing.url = pickPreferredUrl(existing.url, next.url);
    existing.title = pickLongerText(existing.title, next.title);
    existing.description = pickLongerText(
      existing.description,
      next.description,
    );
    existing.date = existing.date || next.date || null;
    existing.age = existing.age || next.age || "";
    existing.score =
      Math.max(getNumericScore(existing.score), getNumericScore(next.score)) ||
      undefined;
    existing.providers = toUniqueList([
      ...existing.providers,
      ...next.providers,
    ]);
    existing.providerCount = Math.max(
      existing.providerCount,
      next.providerCount,
      existing.providers.length || 1,
    );
    existing.queryVariants = toUniqueList([
      ...existing.queryVariants,
      ...next.queryVariants,
    ]);
    existing.queryHitCount = Math.max(
      existing.queryHitCount,
      next.queryHitCount,
      existing.queryVariants.length || 1,
    );
    existing.source =
      existing.providers.length > 1
        ? "multi"
        : existing.providers[0] || existing.source || next.source || null;
    existing.rank = Math.min(existing.rank, next.rank);
    existing._firstSeen = Math.min(existing._firstSeen, next._firstSeen);
    canonicalUrlToKey.set(canonicalUrl, mergeKey);
    if (nearDuplicateKey) nearDuplicateToKey.set(nearDuplicateKey, mergeKey);
  });

  return [...mergedByKey.values()]
    .sort((left, right) => left._firstSeen - right._firstSeen)
    .slice(0, limit)
    .map(({ _firstSeen, ...item }) => item);
}

export function extractQueryTerms(query, maxTerms = 32) {
  return String(query || "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOP.has(t))
    .slice(0, maxTerms);
}

export function scoreTextForTerms(text, terms) {
  if (!text || !terms.length) return 0;
  const lower = text.toLowerCase();
  let score = 0;
  for (const term of terms) {
    if (!lower.includes(term)) continue;
    const occurrences = lower.split(term).length - 1;
    score += 1 + Math.min(4, occurrences) * 0.35;
  }
  return score;
}

const countMatchedTerms = (text, terms) => {
  if (!text || !terms.length) return 0;
  const normalized = normalizeMatchText(text);
  let count = 0;
  for (const term of terms) {
    if (normalized.includes(term)) count += 1;
  }
  return count;
};

const extractQueryPhrases = (query, terms) => {
  const phrases = new Set();
  const normalizedQuery = normalizeMatchText(query);

  if (normalizedQuery && terms.length >= 2 && terms.length <= 8) {
    phrases.add(normalizedQuery);
  }

  const quoted = String(query || "").match(/"([^"]+)"/g) || [];
  for (const rawPhrase of quoted) {
    const normalized = normalizeMatchText(rawPhrase.slice(1, -1));
    if (normalized && normalized.split(/\s+/).length >= 2) {
      phrases.add(normalized);
    }
    if (phrases.size >= 4) break;
  }

  return [...phrases].slice(0, 4);
};

const queryAwareMatchBoost = (query, source, terms) => {
  if (!terms.length) return 0;

  const title = normalizeMatchText(source?.title || "");
  const description = normalizeMatchText(source?.description || "");
  const url = normalizeMatchText(source?.url || "");
  const phrases = extractQueryPhrases(query, terms);
  const titleCoverage = countMatchedTerms(title, terms) / terms.length;
  const descriptionCoverage =
    countMatchedTerms(description, terms) / terms.length;
  const urlCoverage = countMatchedTerms(url, terms) / terms.length;
  let phraseBoost = 0;

  for (const phrase of phrases) {
    if (title.includes(phrase)) phraseBoost += 1.9;
    else if (description.includes(phrase)) phraseBoost += 0.8;
    else if (url.includes(phrase)) phraseBoost += 0.35;
  }

  const completeTitleMatch = terms.length >= 2 && titleCoverage === 1 ? 1.1 : 0;
  const focusedTitleBoost =
    terms.length && title.startsWith(terms[0]) ? 0.2 : 0;

  return (
    phraseBoost +
    titleCoverage * 1.6 +
    descriptionCoverage * 0.85 +
    urlCoverage * 0.35 +
    completeTitleMatch +
    focusedTitleBoost
  );
};

const getUrlPathDepth = (url) => {
  try {
    return new URL(canonicalizeSourceUrl(url)).pathname
      .split("/")
      .filter(Boolean).length;
  } catch {
    return 0;
  }
};

const lowSignalPagePenalty = (source, terms) => {
  if (!terms.length) return 0;
  const titleMatches = countMatchedTerms(source?.title || "", terms);
  const descriptionMatches = countMatchedTerms(
    source?.description || "",
    terms,
  );
  const pathDepth = getUrlPathDepth(source?.url || "");

  if (pathDepth <= 1 && titleMatches <= 1 && descriptionMatches <= 1) {
    return 0.55;
  }

  return 0;
};

export function domainAuthorityBoost(url) {
  try {
    const host = new URL(canonicalizeSourceUrl(url)).hostname
      .replace(/^www\./, "")
      .toLowerCase();
    if (/\.(gov|edu)(\.|$)/.test(host)) return 2.25;
    if (
      /arxiv\.org|doi\.org|pubmed|pmc\.ncbi|nih\.gov|nature\.com|science\.org|springer|ieee\.org|acm\.org/.test(
        host,
      )
    ) {
      return 1.75;
    }
    if (/docs\.|developer\.|developers\.|support\.|learn\./.test(host))
      return 0.45;
    if (/wikipedia\.org$/.test(host)) return 0.35;
  } catch {
    /* ignore */
  }
  return 0;
}

function classifyQueryIntent(query) {
  const q = String(query || "");
  return {
    timeSensitive: queryNeedsFreshness(q),
    academic:
      /\b(rrl|review of related literature|literature review|systematic review|meta-analysis|peer[- ]reviewed|journal|study|studies|research|academic|paper|papers)\b/i.test(
        q,
      ),
    docs: /\b(api|sdk|docs?|documentation|reference|guide|tutorial|example|examples|integration|install|setup|migration|troubleshoot(?:ing)?)\b/i.test(
      q,
    ),
  };
}

function classifySourceCategory(url) {
  const host = getSourceDomain(url);
  if (!host) return "web";
  if (
    /\.(gov|mil)(\.|$)/.test(host) ||
    /(nist|nih|cisa|fda|who|un\.org|europa)/.test(host)
  )
    return "official";
  if (
    /\.(edu)(\.|$)/.test(host) ||
    /(arxiv|doi\.org|pubmed|pmc\.ncbi|nature\.com|science\.org|springer|ieee\.org|acm\.org)/.test(
      host,
    )
  )
    return "research";
  if (
    /(docs\.|developer\.|developers\.|support\.|learn\.|github\.com|npmjs\.com|readthedocs)/.test(
      host,
    )
  )
    return "docs";
  if (
    /(reuters|apnews|bbc|nytimes|washingtonpost|theguardian|wsj|bloomberg|techcrunch|theverge|wired|axios)/.test(
      host,
    )
  )
    return "news";
  return "web";
}

function intentBoostForSource(url, intent) {
  const category = classifySourceCategory(url);
  let boost = 0;

  if (category === "official") boost += 0.5;
  if (category === "research") boost += 0.35;

  if (intent.academic) {
    if (category === "research") boost += 2.1;
    else if (category === "official") boost += 1.1;
    else if (category === "news") boost -= 0.25;
  }

  if (intent.docs) {
    if (category === "docs") boost += 2.0;
    else if (category === "official") boost += 0.5;
  }

  if (intent.timeSensitive) {
    if (category === "news") boost += 1.45;
    else if (category === "official") boost += 1.15;
  }

  return boost;
}

function queryNeedsFreshness(query) {
  return /\b(latest|recent|today|current|new|newest|breaking|updated?|this week|this month|this year|202\d)\b/i.test(
    String(query || ""),
  );
}

function parseRelativeAgeToDays(value) {
  const match = String(value || "")
    .toLowerCase()
    .match(/(\d+)\s*(minute|min|hour|day|week|month|year)s?\s+ago/);
  if (!match) return null;

  const amount = Number(match[1]);
  const unit = match[2];
  if (!Number.isFinite(amount)) return null;
  if (unit.startsWith("minute") || unit === "min") return amount / 1440;
  if (unit.startsWith("hour")) return amount / 24;
  if (unit.startsWith("day")) return amount;
  if (unit.startsWith("week")) return amount * 7;
  if (unit.startsWith("month")) return amount * 30;
  if (unit.startsWith("year")) return amount * 365;
  return null;
}

function estimateSourceAgeDays(source) {
  const candidates = [source?.age, source?.date, source?.publishedTime];

  for (const candidate of candidates) {
    const relative = parseRelativeAgeToDays(candidate);
    if (relative != null) return relative;

    const timestamp = Date.parse(String(candidate || ""));
    if (!Number.isNaN(timestamp)) {
      return Math.max(0, (Date.now() - timestamp) / (1000 * 60 * 60 * 24));
    }
  }

  return null;
}

function freshnessBoost(query, source) {
  if (!queryNeedsFreshness(query)) return 0;

  const ageDays = estimateSourceAgeDays(source);
  if (ageDays == null) return 0;
  if (ageDays <= 7) return 1.25;
  if (ageDays <= 30) return 0.85;
  if (ageDays <= 180) return 0.35;
  if (ageDays <= 730) return 0;
  return -0.35;
}

const domainCrowdingPenalty = (domainCounts, domain, source) => {
  const seen = domainCounts.get(domain) || 0;
  if (!domain || !seen) return 0;

  let penalty = 1.05 + (seen - 1) * 0.8;
  if (getUrlPathDepth(source?.url || "") <= 1) {
    penalty += 0.2;
  }

  return penalty;
};

export function rerankSourcesForQuery(query, sources = []) {
  const terms = extractQueryTerms(query);
  const intent = classifyQueryIntent(query);
  const list = Array.isArray(sources) ? sources : [];
  const scored = list.map((source, position) => {
    const normalizedUrl = canonicalizeSourceUrl(source?.url || "");
    const normalizedSource =
      normalizedUrl && normalizedUrl !== source?.url
        ? { ...source, url: normalizedUrl }
        : source;
    const lexical =
      scoreTextForTerms(normalizedSource?.title || "", terms) * 1.35 +
      scoreTextForTerms(normalizedSource?.description || "", terms) * 0.9 +
      scoreTextForTerms(normalizedSource?.url || "", terms) * 0.45;
    const queryMatch = queryAwareMatchBoost(query, normalizedSource, terms);
    const authority = domainAuthorityBoost(normalizedSource?.url || "");
    const intentBoost = intentBoostForSource(
      normalizedSource?.url || "",
      intent,
    );
    const providerScore = Math.max(
      0,
      Math.min(1.5, Number(normalizedSource?.score) || 0),
    );
    const providerAgreement =
      Math.max(
        0,
        Math.min(1.6, (Number(normalizedSource?.providerCount) || 1) - 1),
      ) * 0.8;
    const queryAgreement =
      Math.max(
        0,
        Math.min(1.8, (Number(normalizedSource?.queryHitCount) || 1) - 1),
      ) * 0.85;
    const descriptionBoost = Math.min(
      0.35,
      String(normalizedSource?.description || "").length / 320,
    );
    const freshness = freshnessBoost(query, normalizedSource);
    const recency =
      Math.max(0, (list.length - position) / Math.max(list.length, 1)) * 0.25;
    const lowSignalPenalty = lowSignalPagePenalty(normalizedSource, terms);
    return {
      source: normalizedSource,
      _baseScore:
        lexical +
        queryMatch +
        authority +
        intentBoost +
        providerScore +
        providerAgreement +
        queryAgreement +
        descriptionBoost +
        freshness +
        recency -
        lowSignalPenalty,
      _position: position,
      _domain: getSourceDomain(normalizedSource?.url || ""),
    };
  });

  const remaining = [...scored];
  const ranked = [];
  const domainCounts = new Map();

  while (remaining.length) {
    remaining.sort((a, b) => {
      const aPenalty = domainCrowdingPenalty(domainCounts, a._domain, a.source);
      const bPenalty = domainCrowdingPenalty(domainCounts, b._domain, b.source);
      const aScore = a._baseScore - aPenalty;
      const bScore = b._baseScore - bPenalty;

      if (bScore !== aScore) return bScore - aScore;
      return a._position - b._position;
    });

    const next = remaining.shift();
    ranked.push(next);
    if (next?._domain) {
      domainCounts.set(next._domain, (domainCounts.get(next._domain) || 0) + 1);
    }
  }

  return ranked.map((row) => row.source);
}

export function getSourceDomain(url) {
  try {
    return new URL(canonicalizeSourceUrl(url)).hostname
      .replace(/^www\./, "")
      .toLowerCase();
  } catch {
    return "";
  }
}

export function selectSourcesForFetch(searchQuery, sources = [], options = {}) {
  const limit = Math.max(1, Number(options.limit) || 12);
  const perDomainLimit = Math.max(1, Number(options.perDomainLimit) || 2);
  const ranked = rerankSourcesForQuery(searchQuery, sources);
  const selected = [];
  const selectedUrls = new Set();
  const domainCounts = new Map();

  const trySelect = (source, allowance) => {
    const url = canonicalizeSourceUrl(source?.url || "");
    if (!url || selectedUrls.has(url)) return false;

    const domainKey = getSourceDomain(url) || `unknown:${selected.length}`;
    const existingCount = domainCounts.get(domainKey) || 0;
    if (existingCount >= allowance) return false;

    selected.push(source);
    selectedUrls.add(url);
    domainCounts.set(domainKey, existingCount + 1);
    return true;
  };

  for (
    let allowance = 1;
    allowance <= perDomainLimit && selected.length < limit;
    allowance += 1
  ) {
    for (const source of ranked) {
      trySelect(source, allowance);
      if (selected.length >= limit) break;
    }
  }

  if (selected.length < limit) {
    for (const source of ranked) {
      const url = canonicalizeSourceUrl(source?.url || "");
      if (!url || selectedUrls.has(url)) continue;
      selected.push(source);
      selectedUrls.add(url);
      if (selected.length >= limit) break;
    }
  }

  return selected;
}

export function splitIntoUnits(text, maxUnitChars = 1100) {
  const t = String(text || "")
    .replace(/\s+/g, " ")
    .trim();
  if (!t) return [];
  const sentences = t.split(/(?<=[.!?])\s+/).filter((s) => s.trim());
  if (!sentences.length) return [t.slice(0, maxUnitChars)];
  const units = [];
  let buf = "";
  for (const sent of sentences) {
    const next = buf ? `${buf} ${sent}` : sent;
    if (next.length > maxUnitChars && buf) {
      units.push(buf.trim());
      buf = sent;
    } else {
      buf = next;
    }
  }
  if (buf.trim()) units.push(buf.trim());
  return units.length ? units : [t.slice(0, maxUnitChars)];
}

export function selectRelevantExcerpt(query, fullText, options = {}) {
  const maxChars = options.maxChars ?? 1600;
  const maxUnits = options.maxUnits ?? 8;
  const raw = String(fullText || "")
    .replace(/\s+/g, " ")
    .trim();
  if (!raw) return "";

  const terms = extractQueryTerms(query);
  if (!terms.length) return raw.slice(0, maxChars);

  const units = splitIntoUnits(raw, 1000);
  const scored = units.map((unit, index) => ({
    unit,
    index,
    score: scoreTextForTerms(unit, terms),
  }));
  scored.sort((a, b) => b.score - a.score || a.index - b.index);

  const bestScore = scored[0]?.score ?? 0;
  const out = [];
  let total = 0;
  const used = new Set();

  for (const row of scored) {
    if (out.length >= maxUnits) break;
    if (used.has(row.unit)) continue;
    if (bestScore > 0 && row.score === 0) continue;
    used.add(row.unit);
    const sep = total ? "\n\n" : "";
    if (total + sep.length + row.unit.length > maxChars) {
      const room = maxChars - total - sep.length;
      if (room > 80) {
        out.push(`${row.unit.slice(0, room).trim()}…`);
      }
      break;
    }
    out.push(row.unit);
    total += sep.length + row.unit.length;
  }

  let joined = out.join("\n\n").trim();
  if (!joined) joined = raw.slice(0, maxChars);
  if (joined.length > maxChars)
    joined = `${joined.slice(0, maxChars - 1).trim()}…`;
  return joined;
}

/** Fetch budget per URL before passage selection (Jina/read pipeline). */
export const RAG_FETCH_MAX_CHARS = 5200;

/** Max chars per source after query-focused extraction (keeps swarm prompts bounded). */
export const RAG_EXCERPT_MAX_CHARS = 1600;

const stripFetchMeta = (content = "") =>
  String(content || "")
    .replace(/^(?:<!--[\s\S]*?-->\s*)+/g, "")
    .trim();

const hasMeaningfulDescription = (value = "") => {
  const normalized = String(value || "")
    .trim()
    .toLowerCase();
  return Boolean(normalized) && normalized !== "no description available";
};

/**
 * After dedupe, re-rank and assign citation indices [1..n].
 */
export function rankSourcesWithRag(searchQuery, dedupedSources) {
  return rerankSourcesForQuery(searchQuery, dedupedSources).map(
    (source, index) => ({
      ...source,
      citationIndex: index + 1,
    }),
  );
}

function getDomain(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url || "";
  }
}

export function rankEvidenceEntriesForQuery(query, entries = [], options = {}) {
  const list = Array.isArray(entries) ? entries : [];
  if (!list.length) return [];

  const terms = extractQueryTerms(query);
  const maxPerDomain = Math.max(1, Number(options.maxPerDomain) || 2);
  const scored = list.map((entry, position) => {
    const source = entry?.source || {};
    const raw = stripFetchMeta(entry?.content || "")
      .replace(/\s+/g, " ")
      .trim();
    const excerpt = selectRelevantExcerpt(query, raw, {
      maxChars: RAG_EXCERPT_MAX_CHARS,
      maxUnits: 6,
    });
    const sourceBlob =
      `${source?.title || ""} ${source?.description || ""} ${source?.url || ""}`.trim();
    const sourceScore = scoreTextForTerms(sourceBlob, terms);
    const excerptScore = scoreTextForTerms(excerpt || raw, terms);
    const authority = domainAuthorityBoost(source?.url || "");
    const carryForward =
      Math.max(0, (list.length - position) / Math.max(list.length, 1)) * 0.35;

    return {
      entry,
      position,
      domain: getSourceDomain(source?.url || "") || "__unknown__",
      score: excerptScore * 1.9 + sourceScore * 0.9 + authority + carryForward,
    };
  });

  scored.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return a.position - b.position;
  });

  const domainCounts = new Map();
  const preferred = [];
  const overflow = [];

  for (const row of scored) {
    const seen = domainCounts.get(row.domain) || 0;
    if (seen < maxPerDomain) {
      domainCounts.set(row.domain, seen + 1);
      preferred.push(row.entry);
    } else {
      overflow.push(row.entry);
    }
  }

  return [...preferred, ...overflow];
}

/**
 * One evidence block for a synthesis worker (replaces ad-hoc string building).
 * @param {{ source: object, content?: string }} entry
 * @param {string} searchQuery
 */
export function buildRagEvidenceBlock(entry, searchQuery) {
  const source = entry?.source || {};
  const index = source.citationIndex ?? 0;
  const raw = stripFetchMeta(entry?.content || "")
    .replace(/\s+/g, " ")
    .trim();
  const excerpt = selectRelevantExcerpt(searchQuery, raw, {
    maxChars: RAG_EXCERPT_MAX_CHARS,
  });
  const parts = [
    `[${index}] ${source.title || getDomain(source.url || "")}`,
    `URL: ${source.url || ""}`,
  ];
  if (hasMeaningfulDescription(source.description))
    parts.push(`Search snippet: ${source.description}`);
  if (excerpt) parts.push(`Fetched excerpt (query-focused): ${excerpt}`);
  return parts.join("\n");
}
