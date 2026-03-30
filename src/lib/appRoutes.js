export const DEFAULT_APP_VIEW = "chat";

export const APP_VIEW_PATHS = Object.freeze({
  chat: "/",
  library: "/library",
  docs: "/docs",
});

export function normalizeAppView(value = "", fallback = DEFAULT_APP_VIEW) {
  const normalized = String(value || "")
    .trim()
    .toLowerCase();
  return Object.prototype.hasOwnProperty.call(APP_VIEW_PATHS, normalized)
    ? normalized
    : fallback;
}

export function normalizeAppPathname(value = "") {
  const raw = String(value || "").trim();
  if (!raw) return APP_VIEW_PATHS.chat;

  let pathname = raw;
  try {
    pathname = new URL(raw, "https://nubagent.local").pathname;
  } catch {
    pathname = raw;
  }

  pathname = pathname.replace(/\/{2,}/g, "/");
  if (!pathname.startsWith("/")) pathname = `/${pathname}`;
  if (pathname.length > 1) pathname = pathname.replace(/\/+$/, "") || "/";
  return pathname || APP_VIEW_PATHS.chat;
}

export function getAppViewFromPathname(pathname = "") {
  const normalizedPathname = normalizeAppPathname(pathname);
  if (normalizedPathname === "/chat" || normalizedPathname === "/research") {
    return DEFAULT_APP_VIEW;
  }
  const match = Object.entries(APP_VIEW_PATHS).find(
    ([, candidatePath]) => candidatePath === normalizedPathname,
  );
  return match ? match[0] : DEFAULT_APP_VIEW;
}

export function getAppViewFromLocation(
  locationLike = typeof window !== "undefined" ? window.location : null,
) {
  return getAppViewFromPathname(locationLike?.pathname || APP_VIEW_PATHS.chat);
}

export function buildAppViewHref(
  view = DEFAULT_APP_VIEW,
  { sessionId = "" } = {},
) {
  const normalizedView = normalizeAppView(view);
  const href = new URL(
    APP_VIEW_PATHS[normalizedView],
    "https://nubagent.local",
  );
  const normalizedSessionId = String(sessionId || "").trim();

  if (normalizedView === "chat" && normalizedSessionId) {
    href.searchParams.set("session", normalizedSessionId);
  }

  return `${href.pathname}${href.search}`;
}

export function buildAppViewUrl(
  view = DEFAULT_APP_VIEW,
  {
    sessionId = "",
    locationLike = typeof window !== "undefined" ? window.location : null,
  } = {},
) {
  if (!locationLike?.origin) return "";
  return new URL(
    buildAppViewHref(view, { sessionId }),
    locationLike.origin,
  ).toString();
}

export function writeAppViewToHistory(
  view = DEFAULT_APP_VIEW,
  {
    sessionId = "",
    replace = false,
    historyLike = typeof window !== "undefined" ? window.history : null,
    locationLike = typeof window !== "undefined" ? window.location : null,
  } = {},
) {
  const nextUrl = buildAppViewUrl(view, { sessionId, locationLike });
  if (!nextUrl || !historyLike) return "";

  if (replace) {
    historyLike.replaceState({}, "", nextUrl);
  } else if (nextUrl !== locationLike?.href) {
    historyLike.pushState({}, "", nextUrl);
  }

  return nextUrl;
}

export default {
  APP_VIEW_PATHS,
  DEFAULT_APP_VIEW,
  normalizeAppView,
  normalizeAppPathname,
  getAppViewFromPathname,
  getAppViewFromLocation,
  buildAppViewHref,
  buildAppViewUrl,
  writeAppViewToHistory,
};
