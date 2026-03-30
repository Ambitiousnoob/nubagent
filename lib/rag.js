/**
 * Retrieval: lexical re-ranking and query-focused passage selection (server-side CommonJS version).
 */

const STOP = new Set([
  "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for", "of", "as", "is", "was",
  "are", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
  "could", "should", "may", "might", "must", "can", "with", "from", "by", "about", "into", "through",
  "during", "before", "after", "above", "below", "between", "under", "again", "then", "once", "here",
  "there", "when", "where", "why", "how", "all", "each", "every", "both", "few", "more", "most", "other",
  "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just", "that",
  "this", "these", "those", "what", "which", "who", "whom", "it", "its", "they", "them", "their", "we",
  "our", "you", "your", "he", "she", "his", "her", "my", "me", "i", "find", "get", "use", "using",
]);

function extractQueryTerms(query, maxTerms = 32) {
  return String(query || "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOP.has(t))
    .slice(0, maxTerms);
}

function scoreTextForTerms(text, terms) {
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

function domainAuthorityBoost(url) {
  try {
    const host = new URL(url).hostname.replace(/^www\./, "").toLowerCase();
    if (/\.(gov|edu)(\.|$)/.test(host)) return 2.25;
    if (/arxiv\.org|doi\.org|pubmed|pmc\.ncbi|nih\.gov|nature\.com|science\.org|springer|ieee\.org|acm\.org/.test(host)) {
      return 1.75;
    }
    if (/wikipedia\.org$/.test(host)) return 0.35;
  } catch {
    /* ignore */
  }
  return 0;
}

function rerankSourcesForQuery(query, sources = []) {
  const terms = extractQueryTerms(query);
  const list = Array.isArray(sources) ? sources : [];
  const scored = list.map((source, position) => {
    const blob = `${source?.title || ""} ${source?.description || ""} ${source?.url || ""}`;
    const lexical = scoreTextForTerms(blob, terms);
    const boost = domainAuthorityBoost(source?.url || "");
    const recency = Math.max(0, (list.length - position) / list.length) * 0.4;
    return {
      source,
      _score: lexical + boost + recency,
      _position: position,
    };
  });
  scored.sort((a, b) => {
    if (b._score !== a._score) return b._score - a._score;
    return a._position - b._position;
  });
  return scored.map((row) => row.source);
}

function splitIntoUnits(text, maxUnitChars = 1100) {
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

function selectRelevantExcerpt(query, fullText, options = {}) {
  const maxChars = options.maxChars ?? 1600;
  const maxUnits = options.maxUnits ?? 8;
  const raw = String(fullText || "").replace(/\s+/g, " ").trim();
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
        total = maxChars;
      }
      break;
    }
    out.push(row.unit);
    total += sep.length + row.unit.length;
  }

  let joined = out.join("\n\n").trim();
  if (!joined) joined = raw.slice(0, maxChars);
  if (joined.length > maxChars) joined = `${joined.slice(0, maxChars - 1).trim()}…`;
  return joined;
}

/** Fetch budget per URL before passage selection (Jina/read pipeline). */
const RAG_FETCH_MAX_CHARS = 5200;

/** Max chars per source after query-focused extraction (keeps swarm prompts bounded). */
const RAG_EXCERPT_MAX_CHARS = 1600;

const stripFetchMeta = (content = "") => (
  String(content || "")
    .replace(/^<!--[\s\S]*?-->\s*/g, "")
    .trim()
);

/**
 * After dedupe, re-rank and assign citation indices [1..n].
 */
function rankSourcesWithRag(searchQuery, dedupedSources) {
  return rerankSourcesForQuery(searchQuery, dedupedSources).map((source, index) => ({
    ...source,
    citationIndex: index + 1,
  }));
}

function getDomain(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url || "";
  }
}

/**
 * One evidence block for a synthesis worker (replaces ad-hoc string building).
 * @param {{ source: object, content?: string }} entry
 * @param {string} searchQuery
 */
function buildRagEvidenceBlock(entry, searchQuery) {
  const source = entry?.source || {};
  const index = source.citationIndex ?? 0;
  const raw = stripFetchMeta(entry?.content || "").replace(/\s+/g, " ").trim();
  const excerpt = selectRelevantExcerpt(searchQuery, raw, { maxChars: RAG_EXCERPT_MAX_CHARS });
  const parts = [
    `[${index}] ${source.title || getDomain(source.url || "")}`,
    `URL: ${source.url || ""}`,
  ];
  if (source.description) parts.push(`Search snippet: ${source.description}`);
  if (excerpt) parts.push(`Fetched excerpt (query-focused): ${excerpt}`);
  return parts.join("\n");
}

module.exports = {
  extractQueryTerms,
  scoreTextForTerms,
  domainAuthorityBoost,
  rerankSourcesForQuery,
  splitIntoUnits,
  selectRelevantExcerpt,
  rankSourcesWithRag,
  buildRagEvidenceBlock,
  RAG_FETCH_MAX_CHARS,
  RAG_EXCERPT_MAX_CHARS,
};
