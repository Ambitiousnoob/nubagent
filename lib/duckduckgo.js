const DUCKDUCKGO_HTML_ENDPOINT = "https://duckduckgo.com/html/";
const DEFAULT_RESULT_LIMIT = 5;

function decodeHtmlEntities(value) {
  return value
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">");
}

function stripHtml(value) {
  return decodeHtmlEntities(value)
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeDuckDuckGoResultUrl(value) {
  try {
    const resolved = new URL(value, "https://duckduckgo.com");
    const redirected = resolved.searchParams.get("uddg");

    return redirected ? decodeURIComponent(redirected) : resolved.toString();
  } catch {
    return value;
  }
}

export function extractDuckDuckGoResults(
  html,
  { limit = DEFAULT_RESULT_LIMIT } = {},
) {
  const results = [];
  const anchorPattern =
    /<a[^>]*class="[^"]*\bresult__a\b[^"]*"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi;
  let match;

  while ((match = anchorPattern.exec(html)) && results.length < limit) {
    const title = stripHtml(match[2]);
    const url = normalizeDuckDuckGoResultUrl(match[1]);
    const nextAnchorIndex = html.indexOf(
      'class="result__a"',
      anchorPattern.lastIndex,
    );
    const nearbyHtml = html.slice(
      anchorPattern.lastIndex,
      nextAnchorIndex === -1 ? anchorPattern.lastIndex + 2000 : nextAnchorIndex,
    );
    const snippetMatch = nearbyHtml.match(
      /class="[^"]*\bresult__snippet\b[^"]*"[^>]*>([\s\S]*?)<\/(?:a|div)>/i,
    );
    const snippet = snippetMatch ? stripHtml(snippetMatch[1]) : "";

    if (!title || !url) {
      continue;
    }

    results.push({
      title,
      url,
      snippet,
    });
  }

  return results;
}

export async function searchDuckDuckGo(
  query,
  { limit = DEFAULT_RESULT_LIMIT } = {},
) {
  const url = new URL(DUCKDUCKGO_HTML_ENDPOINT);
  url.searchParams.set("q", query);

  const response = await fetch(url, {
    headers: {
      Accept: "text/html,application/xhtml+xml",
      "User-Agent": "NubAgent/1.0 (+https://github.com/Ambitiousnoob/nubagent)",
    },
  });

  if (!response.ok) {
    throw new Error(`DuckDuckGo search ${response.status}`);
  }

  const html = await response.text();
  return extractDuckDuckGoResults(html, { limit });
}

export function buildDuckDuckGoGroundedPrompt(prompt, results) {
  if (!Array.isArray(results) || results.length === 0) {
    return prompt;
  }

  const resultText = results
    .map((result, index) => {
      const lines = [`[${index + 1}] ${result.title}`, `URL: ${result.url}`];

      if (result.snippet) {
        lines.push(`Snippet: ${result.snippet}`);
      }

      return lines.join("\n");
    })
    .join("\n\n");

  return `${prompt}

Use the following DuckDuckGo search results as optional fresh web context. Prefer them when they help answer time-sensitive questions, and mention the source URLs in plain text when relevant.

DuckDuckGo search results:
${resultText}`;
}
