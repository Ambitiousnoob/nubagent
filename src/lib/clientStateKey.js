const STORAGE_KEY = "nubagent-client-state-key";
const STATE_KEY_RE = /^[A-Za-z0-9:_-]{8,191}$/;

const normalizeText = (value) => String(value ?? "").trim();

const createToken = () => {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID().replace(/-/g, "");
  }

  return `${Date.now().toString(16)}${Math.random().toString(16).slice(2)}${Math.random()
    .toString(16)
    .slice(2)}`;
};

const buildStateKey = () => `browser:${createToken().slice(0, 48)}`;

export function getClientStateKey(storageLike = typeof window !== "undefined" ? window.localStorage : null) {
  try {
    const stored = normalizeText(storageLike?.getItem?.(STORAGE_KEY));
    if (STATE_KEY_RE.test(stored)) return stored;
  } catch {
    // Ignore storage access failures and fall back to an in-memory key.
  }

  const generated = buildStateKey();

  try {
    storageLike?.setItem?.(STORAGE_KEY, generated);
  } catch {
    // Ignore storage access failures and return the generated key for this request.
  }

  return generated;
}

export function withClientStateKey(payload = {}, storageLike) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return payload;
  if (normalizeText(payload.stateKey)) return payload;
  return {
    ...payload,
    stateKey: getClientStateKey(storageLike),
  };
}

export const CLIENT_STATE_KEY_STORAGE_KEY = STORAGE_KEY;

export default {
  CLIENT_STATE_KEY_STORAGE_KEY,
  getClientStateKey,
  withClientStateKey,
};
