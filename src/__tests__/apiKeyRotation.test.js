import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

describe("api key rotation", () => {
  let getApiKeysFromEnv;
  let getRotatingApiKey;
  let getRotatingValue;

  const resetEnv = () => {
    delete process.env.TEST_API_KEYS;
    delete process.env.TEST_API_KEY;
  };

  beforeEach(() => {
    resetEnv();
    vi.resetModules();
    ({
      getApiKeysFromEnv,
      getRotatingApiKey,
      getRotatingValue,
    } = require("../../lib/api-key-rotation.cjs"));
  });

  afterEach(() => {
    resetEnv();
    vi.resetModules();
  });

  it("merges plural and singular env vars into one deduped key list", () => {
    process.env.TEST_API_KEYS = "alpha,beta";
    process.env.TEST_API_KEY = "beta,gamma";

    expect(getApiKeysFromEnv("TEST_API_KEYS", "TEST_API_KEY")).toEqual([
      "alpha",
      "beta",
      "gamma",
    ]);
  });

  it("rotates comma-separated keys from the singular env var", () => {
    process.env.TEST_API_KEY = "alpha,beta,gamma";

    expect(
      getRotatingApiKey("test-provider", "TEST_API_KEYS", "TEST_API_KEY"),
    ).toBe("alpha");
    expect(
      getRotatingApiKey("test-provider", "TEST_API_KEYS", "TEST_API_KEY"),
    ).toBe("beta");
    expect(
      getRotatingApiKey("test-provider", "TEST_API_KEYS", "TEST_API_KEY"),
    ).toBe("gamma");
    expect(
      getRotatingApiKey("test-provider", "TEST_API_KEYS", "TEST_API_KEY"),
    ).toBe("alpha");
  });

  it("rotates request-scoped values without relying on env vars", () => {
    expect(
      getRotatingValue("openrouter-request", ["key-a,key-b", "key-c"]),
    ).toBe("key-a");
    expect(
      getRotatingValue("openrouter-request", ["key-a,key-b", "key-c"]),
    ).toBe("key-b");
    expect(
      getRotatingValue("openrouter-request", ["key-a,key-b", "key-c"]),
    ).toBe("key-c");
    expect(
      getRotatingValue("openrouter-request", ["key-a,key-b", "key-c"]),
    ).toBe("key-a");
  });
});
