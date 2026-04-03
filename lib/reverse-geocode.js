function hasLocationContext(location) {
  return (
    Number.isFinite(location?.latitude) &&
    Number.isFinite(location?.longitude)
  );
}

function normalizeText(value) {
  return typeof value === "string" ? value.trim() : "";
}

export function buildGoogleMapsLocationUrl(location) {
  if (!hasLocationContext(location)) {
    return "";
  }

  const query = encodeURIComponent(`${location.latitude},${location.longitude}`);
  return `https://www.google.com/maps/search/?api=1&query=${query}`;
}

function buildCoordinatesLabel(location) {
  return `Latitude: ${location.latitude}\nLongitude: ${location.longitude}`;
}

function isPreciseGoogleGeocode(result) {
  const types = Array.isArray(result?.types) ? result.types : [];

  return types.some((type) =>
    ["street_address", "premise", "subpremise", "route"].includes(type),
  );
}

async function reverseGeocodeWithGoogle({
  location,
  apiKey,
  fetchImpl,
}) {
  if (!apiKey || !hasLocationContext(location)) {
    return null;
  }

  const url = new URL("https://maps.googleapis.com/maps/api/geocode/json");
  url.searchParams.set("latlng", `${location.latitude},${location.longitude}`);
  url.searchParams.set("key", apiKey);

  const response = await fetchImpl(url, {
    headers: {
      Accept: "application/json",
    },
  });
  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(
      payload?.error_message ||
        payload?.status ||
        `Google reverse geocoding failed with ${response.status}.`,
    );
  }

  if (payload?.status === "ZERO_RESULTS") {
    return null;
  }

  if (payload?.status && payload.status !== "OK") {
    throw new Error(
      payload?.error_message ||
        `Google reverse geocoding failed with ${payload.status}.`,
    );
  }

  const result = Array.isArray(payload?.results) ? payload.results[0] : null;
  const formattedAddress = normalizeText(result?.formatted_address);

  if (!formattedAddress) {
    return null;
  }

  return {
    formattedAddress,
    precise: isPreciseGoogleGeocode(result),
    provider: "google",
  };
}

async function reverseGeocodeWithNominatim({ location, fetchImpl }) {
  if (!hasLocationContext(location)) {
    return null;
  }

  const url = new URL("https://nominatim.openstreetmap.org/reverse");
  url.searchParams.set("format", "jsonv2");
  url.searchParams.set("lat", String(location.latitude));
  url.searchParams.set("lon", String(location.longitude));
  url.searchParams.set("zoom", "18");
  url.searchParams.set("addressdetails", "1");

  const response = await fetchImpl(url, {
    headers: {
      Accept: "application/json",
      "User-Agent": "NubAgent/1.0 reverse-geocode",
    },
  });
  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(
      payload?.error || `Nominatim reverse geocoding failed with ${response.status}.`,
    );
  }

  const formattedAddress =
    normalizeText(payload?.display_name) ||
    normalizeText(payload?.name);

  if (!formattedAddress) {
    return null;
  }

  return {
    formattedAddress,
    precise: true,
    provider: "nominatim",
  };
}

export async function reverseGeocodeLocation({
  location,
  config = {},
  fetchImpl = fetch,
}) {
  if (!hasLocationContext(location)) {
    return null;
  }

  const googleResult = await reverseGeocodeWithGoogle({
    location,
    apiKey: config.googleGeocodingApiKey,
    fetchImpl,
  }).catch(() => null);

  if (googleResult) {
    return googleResult;
  }

  const nominatimResult = await reverseGeocodeWithNominatim({
    location,
    fetchImpl,
  }).catch(() => null);

  if (nominatimResult) {
    return nominatimResult;
  }

  return null;
}

export function buildReverseGeocodeReply({ location, resolvedAddress }) {
  const mapUrl = buildGoogleMapsLocationUrl(location);

  if (!hasLocationContext(location)) {
    return "I do not have a saved location for an address lookup yet.";
  }

  if (!normalizeText(resolvedAddress?.formattedAddress)) {
    return `I could not resolve a specific address from your saved location right now.\n\nCoordinates:\n${buildCoordinatesLabel(location)}\n\nOpen in Google Maps:\n${mapUrl}`;
  }

  const leadIn = resolvedAddress.precise
    ? "Nearest resolved address:"
    : "Approximate resolved address:";

  return `${leadIn}\n${resolvedAddress.formattedAddress}\n\nCoordinates:\n${buildCoordinatesLabel(location)}\n\nOpen in Google Maps:\n${mapUrl}`;
}
