/**
 * Barrel: one import for the whole research stack.
 *
 *   import Company, { prepareSearchQuery, finalizeResearchAnswer } from './lib/index.js';
 *   import * as Liberty from './lib/index.js';
 */
export {
  Company,
  prepareSearchQuery,
  finalizeResearchAnswer,
} from "./company.js";

export { default } from "./company.js";

export {
  RESEARCH_SUMMARY_SYSTEM_ADDENDUM,
  MERGE_SUMMARY_ADDENDUM,
  expandLiteratureQuery,
} from "./researchSummaryPrompt.js";

export { sanitizeSummaryText } from "./sanitizeSummaryText.js";

export {
  canonicalizeSourceUrl,
  mergeSourcesByCanonicalUrl,
  extractQueryTerms,
  scoreTextForTerms,
  domainAuthorityBoost,
  rerankSourcesForQuery,
  getSourceDomain,
  selectSourcesForFetch,
  splitIntoUnits,
  selectRelevantExcerpt,
  rankSourcesWithRag,
  rankEvidenceEntriesForQuery,
  buildRagEvidenceBlock,
  RAG_FETCH_MAX_CHARS,
  RAG_EXCERPT_MAX_CHARS,
} from "./rag.js";
