const parseRequestPathname = (value) => {
  const raw = String(value || "").trim();
  if (!raw) return "";
  try {
    return new URL(raw, "http://localhost").pathname;
  } catch {
    return raw.split("?")[0] || "";
  }
};

const resolveUtilsAction = (req) => {
  const explicitAction = String(req?.query?.action || "").trim().toLowerCase();
  if (explicitAction) return explicitAction;

  const pathCandidates = [
    req?.url,
    req?.originalUrl,
    req?.path,
    req?.headers?.["x-matched-path"],
    req?.headers?.["x-invoke-path"],
  ]
    .map(parseRequestPathname)
    .filter(Boolean);

  const aliasMap = new Map([
    ["/api/health", "health"],
    ["/api/analytics", "analytics"],
    ["/api/export", "export"],
    ["/api/messenger", "messenger"],
  ]);

  for (const pathname of pathCandidates) {
    const match = aliasMap.get(pathname);
    if (match) return match;
  }

  return "";
};

module.exports = {
  parseRequestPathname,
  resolveUtilsAction,
};
