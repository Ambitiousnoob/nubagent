import test from "node:test";
import assert from "node:assert/strict";

import {
  buildRequestBody,
  generateGeminiReply,
  getNextGeminiApiKey,
  getOrderedGeminiModels,
  resetGeminiApiKeyRoundRobinForTesting,
  shouldFallbackToNextGeminiModel,
} from "../lib/gemini.js";

const BASE_CONFIG = {
  geminiModel: "gemini-3-flash-preview",
  geminiModels: ["gemini-3-flash-preview", "gemini-2.5-flash"],
  geminiThinkingLevel: "LOW",
  geminiApiKeys: ["key-1", "key-2"],
  systemPrompt: "You are helpful.",
};

test("buildRequestBody appends the latest user prompt", () => {
  const body = buildRequestBody({
    prompt: "What is new?",
    history: [{ role: "user", parts: [{ text: "Hello" }] }],
    config: BASE_CONFIG,
  });

  assert.deepEqual(body.contents, [
    { role: "user", parts: [{ text: "Hello" }] },
    { role: "user", parts: [{ text: "What is new?" }] },
  ]);
  assert.deepEqual(body.systemInstruction, {
    parts: [{ text: "You are helpful." }],
  });
  assert.deepEqual(body.generationConfig, {
    thinkingConfig: { thinkingLevel: "LOW" },
  });
});

test("buildRequestBody omits thinking config for older models", () => {
  const body = buildRequestBody({
    prompt: "Hello",
    history: [],
    config: {
      ...BASE_CONFIG,
      geminiModel: "gemini-2.5-flash",
    },
  });

  assert.equal(body.generationConfig, undefined);
});

test("getNextGeminiApiKey rotates through configured keys", () => {
  resetGeminiApiKeyRoundRobinForTesting();

  assert.equal(getNextGeminiApiKey(BASE_CONFIG), "key-1");
  assert.equal(getNextGeminiApiKey(BASE_CONFIG), "key-2");
  assert.equal(getNextGeminiApiKey(BASE_CONFIG), "key-1");
});

test("getOrderedGeminiModels returns the configured priority list", () => {
  assert.deepEqual(getOrderedGeminiModels(BASE_CONFIG), [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
  ]);
});

test("shouldFallbackToNextGeminiModel recognizes overload-style failures", () => {
  assert.equal(
    shouldFallbackToNextGeminiModel({
      httpStatus: 503,
      apiStatus: "UNAVAILABLE",
      apiMessage: "The model is overloaded.",
    }),
    true,
  );
  assert.equal(
    shouldFallbackToNextGeminiModel({
      httpStatus: 400,
      apiStatus: "INVALID_ARGUMENT",
      apiMessage: "Malformed request.",
    }),
    false,
  );
});

test("generateGeminiReply uses round-robin Gemini keys across attempts", async () => {
  resetGeminiApiKeyRoundRobinForTesting();

  const originalFetch = globalThis.fetch;
  const usedUrls = [];

  globalThis.fetch = async (url) => {
    usedUrls.push(String(url));
    return {
      ok: true,
      async json() {
        return {
          candidates: [
            {
              content: {
                parts: [{ text: "hello" }],
              },
            },
          ],
        };
      },
    };
  };

  try {
    await generateGeminiReply({
      prompt: "one",
      history: [],
      config: BASE_CONFIG,
    });
    await generateGeminiReply({
      prompt: "two",
      history: [],
      config: BASE_CONFIG,
    });

    assert.match(usedUrls[0], /key=key-1/);
    assert.match(usedUrls[1], /key=key-2/);
  } finally {
    globalThis.fetch = originalFetch;
    resetGeminiApiKeyRoundRobinForTesting();
  }
});

test("generateGeminiReply falls back to the next model on high-demand failures", async () => {
  resetGeminiApiKeyRoundRobinForTesting();

  const originalFetch = globalThis.fetch;
  const usedUrls = [];
  let callCount = 0;

  globalThis.fetch = async (url) => {
    usedUrls.push(String(url));
    callCount += 1;

    if (callCount === 1) {
      return {
        ok: false,
        status: 503,
        async json() {
          return {
            error: {
              status: "UNAVAILABLE",
              message: "The service is temporarily overloaded.",
            },
          };
        },
      };
    }

    return {
      ok: true,
      async json() {
        return {
          candidates: [
            {
              content: {
                parts: [{ text: "fallback answer" }],
              },
            },
          ],
        };
      },
    };
  };

  try {
    const reply = await generateGeminiReply({
      prompt: "hello",
      history: [],
      config: BASE_CONFIG,
    });

    assert.equal(reply, "fallback answer");
    assert.match(usedUrls[0], /models\/gemini-3-flash-preview:generateContent/);
    assert.match(usedUrls[1], /models\/gemini-2\.5-flash:generateContent/);
    assert.match(usedUrls[0], /key=key-1/);
    assert.match(usedUrls[1], /key=key-2/);
  } finally {
    globalThis.fetch = originalFetch;
    resetGeminiApiKeyRoundRobinForTesting();
  }
});
