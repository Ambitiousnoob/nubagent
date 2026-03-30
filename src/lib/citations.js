import { canonicalizeSourceUrl } from "./rag.js";

const stripFetchMeta = (content = "") =>
  String(content || "")
    .replace(/^<!--[\s\S]*?-->\s*/g, "")
    .trim();

const hasMeaningfulDescription = (value = "") => {
  const normalized = String(value || "")
    .trim()
    .toLowerCase();
  return Boolean(normalized) && normalized !== "no description available";
};

const toPositiveInteger = (value) => {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
};

const dedupeSourcesByUrl = (items = [], limit = Infinity) => {
  const seen = new Set();
  const merged = [];

  for (const source of Array.isArray(items) ? items : []) {
    if (!source) continue;
    const citationIndex = toPositiveInteger(source?.citationIndex);
    const key =
      canonicalizeSourceUrl(source?.url || "") ||
      `citation:${citationIndex || merged.length + 1}`;
    if (seen.has(key)) continue;
    seen.add(key);
    merged.push(source);
    if (merged.length >= limit) break;
  }

  return merged;
};

export function extractCitationNumbers(text = "") {
  const seen = new Set();
  const citations = [];

  for (const match of String(text || "").matchAll(/\[(\d+)\]/g)) {
    const citationIndex = toPositiveInteger(match[1]);
    if (!citationIndex || seen.has(citationIndex)) continue;
    seen.add(citationIndex);
    citations.push(citationIndex);
  }

  return citations;
}

export function getDisplaySourceNumber(source, fallbackIndex = 0) {
  return toPositiveInteger(source?.citationIndex) || fallbackIndex + 1;
}

export function findSourceForCitation(citation, sources = []) {
  const citationIndex = toPositiveInteger(citation);
  if (!citationIndex) return null;

  const list = Array.isArray(sources) ? sources : [];
  return (
    list.find(
      (source) => toPositiveInteger(source?.citationIndex) === citationIndex,
    ) ||
    list[citationIndex - 1] ||
    null
  );
}

export function buildAttributedSourcesFromEvidence(
  answerText = "",
  evidenceEntries = [],
  limit = 24,
) {
  const successfulEntries = (
    Array.isArray(evidenceEntries) ? evidenceEntries : []
  ).filter(
    (entry) =>
      stripFetchMeta(entry?.content || "") ||
      hasMeaningfulDescription(entry?.source?.description),
  );

  const byCitation = new Map(
    successfulEntries
      .map((entry) => [
        toPositiveInteger(entry?.source?.citationIndex),
        entry?.source,
      ])
      .filter(([citationIndex, source]) => citationIndex && source?.url),
  );

  const citedSources = extractCitationNumbers(answerText)
    .sort((left, right) => left - right)
    .map((citationIndex) => byCitation.get(citationIndex))
    .filter(Boolean);

  const fallbackSources = successfulEntries
    .map((entry) => entry?.source)
    .filter(Boolean)
    .sort(
      (left, right) =>
        getDisplaySourceNumber(left) - getDisplaySourceNumber(right),
    );

  return dedupeSourcesByUrl(
    citedSources.length ? citedSources : fallbackSources,
    limit,
  ).sort(
    (left, right) =>
      getDisplaySourceNumber(left) - getDisplaySourceNumber(right),
  );
}
