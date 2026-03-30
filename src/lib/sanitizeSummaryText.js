/**
 * Post-process final research answer before display (strips identity leaks and index bleed).
 */
export function sanitizeSummaryText(text) {
  if (typeof text !== "string") return text;
  let s = text.replace(/\r\n/g, "\n");

  const introRes = [
    /^I am nub-agent,?\s+and\s+Ambitiousnoob built me\.?\s*/gim,
    /^I'm nub-agent,?\s+and\s+.+built me\.?\s*/gim,
    /^nub-agent\s+here\.?\s*/gim,
  ];
  for (const re of introRes) {
    s = s.replace(re, "");
  }

  s = s.replace(/\s+\d{1,2}(?:\s*,\s*\d{1,2})+(?=\s*[.!?])/g, "");
  s = s.replace(/(")\s+\d{1,2}(?:\s*,\s*\d{1,2})+(?=\s*[.!?])/g, "$1");

  s = s.replace(/[ \t]{2,}/g, " ");
  s = s.replace(/\n{3,}/g, "\n\n");
  return s.trim();
}
