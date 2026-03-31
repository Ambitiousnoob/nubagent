import test from "node:test";
import assert from "node:assert/strict";

import {
  getRuntimeConfig,
  hasMessagingConfig,
  parseGeminiApiKeys,
  parseGeminiModels,
} from "../lib/config.js";

const ORIGINAL_ENV = { ...process.env };

function resetEnv(nextEnv = {}) {
  process.env = {
    ...ORIGINAL_ENV,
    ...nextEnv,
  };
}

test("config requires POSTGRES_URL for message processing", () => {
  resetEnv({
    GEMINI_API_KEY: "gemini",
    GEMINI_CHAT_MODEL: "gemini-3-flash-preview",
    PAGE_ACCESS_TOKEN: "page-token",
    VERIFY_TOKEN: "verify-token",
  });

  const config = getRuntimeConfig();

  assert.equal(hasMessagingConfig(config), false);
  assert.deepEqual(config.missingMessagingKeys, ["POSTGRES_URL"]);
});

test("config normalizes Gemini model and thinking level", () => {
  resetEnv({
    GEMINI_API_KEY: " gemini-1 , gemini-2 , , gemini-3 ",
    PAGE_ACCESS_TOKEN: "page-token",
    VERIFY_TOKEN: "verify-token",
    POSTGRES_URL: "postgres://example",
    GEMINI_CHAT_MODEL: "models/gemini-3-flash-preview, models/gemini-2.5-flash",
    GEMINI_CHAT_THINKING_LEVEL: "medium",
  });

  const config = getRuntimeConfig();

  assert.deepEqual(config.geminiApiKeys, ["gemini-1", "gemini-2", "gemini-3"]);
  assert.deepEqual(config.geminiModels, [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
  ]);
  assert.equal(config.geminiModel, "gemini-3-flash-preview");
  assert.equal(config.geminiThinkingLevel, "MEDIUM");
});

test("config requires GEMINI_CHAT_MODEL when it is not set", () => {
  resetEnv({
    GEMINI_API_KEY: "gemini",
    PAGE_ACCESS_TOKEN: "page-token",
    VERIFY_TOKEN: "verify-token",
    POSTGRES_URL: "postgres://example",
  });

  const config = getRuntimeConfig();

  assert.equal(hasMessagingConfig(config), false);
  assert.deepEqual(config.missingMessagingKeys, ["GEMINI_CHAT_MODEL"]);
});

test("parseGeminiApiKeys keeps only non-empty comma-separated keys", () => {
  assert.deepEqual(parseGeminiApiKeys(" key1, ,key2 , key3 "), [
    "key1",
    "key2",
    "key3",
  ]);
});

test("parseGeminiModels keeps only non-empty comma-separated models", () => {
  assert.deepEqual(
    parseGeminiModels(" models/model1, ,model2 , models/model3 "),
    ["model1", "model2", "model3"],
  );
});

test.after(() => {
  process.env = ORIGINAL_ENV;
});
