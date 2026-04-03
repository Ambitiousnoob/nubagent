export const DEFAULT_LOCATION_CAPTURE_TTL_MINUTES = 15;
export const LOCATION_CAPTURE_PATH = "/api/location-capture";

function normalizeText(value) {
  return typeof value === "string" ? value.trim() : "";
}

function normalizeBaseUrl(baseUrl) {
  return normalizeText(baseUrl).replace(/\/+$/, "");
}

function formatMinutes(minutes) {
  return `${minutes} minute${minutes === 1 ? "" : "s"}`;
}

export function resolveLocationCaptureBaseUrl({ baseUrl, config } = {}) {
  return normalizeBaseUrl(config?.appBaseUrl || baseUrl);
}

export function buildLocationCaptureUrl({ baseUrl, token } = {}) {
  const normalizedBaseUrl = normalizeBaseUrl(baseUrl);
  const normalizedToken = normalizeText(token);

  if (!normalizedBaseUrl || !normalizedToken) {
    return "";
  }

  return `${normalizedBaseUrl}${LOCATION_CAPTURE_PATH}?token=${encodeURIComponent(normalizedToken)}`;
}

export function buildLocationCaptureReply({
  url,
  expiresInMinutes = DEFAULT_LOCATION_CAPTURE_TTL_MINUTES,
  leadIn = "",
} = {}) {
  const normalizedUrl = normalizeText(url);

  if (!normalizedUrl) {
    return "I could not create a location capture link right now. Try again in a moment.";
  }

  const ttlMinutes =
    Number.isFinite(expiresInMinutes) && expiresInMinutes > 0
      ? Math.floor(expiresInMinutes)
      : DEFAULT_LOCATION_CAPTURE_TTL_MINUTES;
  const sections = [];
  const normalizedLeadIn = normalizeText(leadIn);

  if (normalizedLeadIn) {
    sections.push(normalizedLeadIn);
  }

  sections.push(
    "Open this secure link and allow location access. I will save your current location automatically for nearby and Maps-grounded prompts:",
  );
  sections.push(normalizedUrl);
  sections.push(
    `The link expires in ${formatMinutes(ttlMinutes)}. If you already had a saved location, this will replace it. Use !location clear if you want me to forget it later.`,
  );

  return sections.join("\n\n");
}
