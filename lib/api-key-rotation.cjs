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

const getRotatingValue = (bucket, values = []) => {
    const items = dedupe(
        (Array.isArray(values) ? values : [values])
            .flatMap((value) => normalizeApiKeys(value)),
    );
    if (!items.length) return null;

    const index = rotationState.get(bucket) || 0;
    const nextValue = items[index % items.length];
    rotationState.set(bucket, (index + 1) % items.length);
    return nextValue;
};

const getRotatingApiKey = (bucket, ...envNames) => {
    return getRotatingValue(bucket, getApiKeysFromEnv(...envNames));
};

module.exports = {
    getApiKeysFromEnv,
    getRotatingApiKey,
    getRotatingValue,
};
