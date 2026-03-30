/**
 * Single facade: research query → retrieval → answer cleanup.
 * Import one module in UI and API helpers: `import { Company } from './lib/company.js'`
 */
import {
  sanitizeSummaryText,
} from "./sanitizeSummaryText.js";
import {
  RESEARCH_SUMMARY_SYSTEM_ADDENDUM,
  MERGE_SUMMARY_ADDENDUM,
  expandLiteratureQuery,
} from "./researchSummaryPrompt.js";
import {
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
} from "./rag.js";

/** Normalize user input for search + synthesis (RRL expansion, trim). */
export function prepareSearchQuery(rawQuery) {
  return expandLiteratureQuery(String(rawQuery ?? "").trim());
}

/** Post-process model output for the Liberty / AI Summary card. */
export function finalizeResearchAnswer(modelText) {
  return sanitizeSummaryText(modelText);
}

export const Company = {
  identity: {
    name: "Liberty Research",
    assistant: "nub-agent",
  },

  /** End-to-end helpers */
  prepareSearchQuery,
  finalizeResearchAnswer,

  /** Prompt snippets for /api/chat system messages */
  prompts: {
    RESEARCH_SUMMARY_SYSTEM_ADDENDUM,
    MERGE_SUMMARY_ADDENDUM,
  },

  /** Query expansion */
  query: {
    expandLiteratureQuery,
    extractQueryTerms,
  },

  /** Post-synthesis cleanup */
  sanitize: {
    sanitizeSummaryText,
  },

  /** RAG: rank URLs, compress pages to relevant spans, build evidence blocks */
  rag: {
    scoreTextForTerms,
    domainAuthorityBoost,
    rerankSourcesForQuery,
    splitIntoUnits,
    selectRelevantExcerpt,
    rankSourcesWithRag,
    buildRagEvidenceBlock,
    RAG_FETCH_MAX_CHARS,
    RAG_EXCERPT_MAX_CHARS,
  },
};

export default Company;
