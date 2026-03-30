const normalizeApiKeys = (value) => (
    String(value || "")
        .split(",")
        .map((key) => key.trim())
        .filter(Boolean)
);

const dedupe = (items) => [...new Set(items)];

const rotationState = new Map();

const getApiKeysFromEnv = (...envNames) => dedupe(
    envNames.flatMap((envName) => normalizeApiKeys(process.env[envName])),
);

const getRotatingApiKey = (bucket, ...envNames) => {
    const keys = getApiKeysFromEnv(...envNames);
    if (!keys.length) return null;

    const index = rotationState.get(bucket) || 0;
    const nextKey = keys[index % keys.length];
    rotationState.set(bucket, (index + 1) % keys.length);
    return nextKey;
};

module.exports = {
    getApiKeysFromEnv,
    getRotatingApiKey,
};
