/**
 * Expand terse academic tokens (e.g. RRL) for better retrieval.
 */
export function expandLiteratureQuery(query) {
  const q = String(query ?? "").trim();
  if (!q) return q;

  const hasRrl = /\brrl\b/i.test(q);
  if (hasRrl) {
    const expanded = q.replace(/\brrl\b/gi, "Review of Related Literature");
    return `${expanded} (prioritize academic and peer-reviewed sources; synthesize themes for an RRL section)`;
  }

  return q;
}
