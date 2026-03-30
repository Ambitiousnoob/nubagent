/**
 * Append this to the system (or final synthesis) prompt for web-research summaries.
 * Keeps tone academic and blocks identity / index-number leaks.
 */
export const RESEARCH_SUMMARY_SYSTEM_ADDENDUM = [
  "You are writing a research synthesis for the user, not a chat reply.",
  "Do not introduce yourself, your model name, your builder, or any meta line about who created you.",
  "Start with the substantive title or first sentence of the answer; no preamble about the assistant.",
  'Do not append source index numbers, chunk IDs, or comma-separated reference numbers (e.g. "4, 9, 11") in the prose.',
  'If citations are required, use author–year in parentheses or refer to "the sources below" — never bare numeric lists tied to retrieval indices.',
  'Write in clear paragraphs suitable for a Review of Related Literature when the query asks for RRL, "related literature", or similar.',
].join(" ");

/**
 * Short instruction for the merge / reduce step that combines worker outputs.
 */
export const MERGE_SUMMARY_ADDENDUM =
  "Merge into one coherent summary. Remove duplicate points. Do not include tool or worker labels. Apply the same rules as the research summary addendum: no self-introduction and no inline numeric source indices.";

/**
 * Expands terse academic query tokens so retrieval matches intent.
 */
export function expandLiteratureQuery(query) {
  const q = String(query ?? "").trim();
  if (!q) return q;

  const rrl =
    /\brrl\b/i.test(q) ||
    (/\brelated\s+literature\b/i.test(q) && !/\breview\s+of\b/i.test(q));

  if (rrl) {
    const expanded = q.replace(/\brrl\b/gi, "Review of Related Literature");
    return `${expanded} (prioritize peer-reviewed and academic sources; synthesize themes for an RRL section)`;
  }

  return q;
}
