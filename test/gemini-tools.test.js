import test from "node:test";
import assert from "node:assert/strict";

import { buildRequestBody } from "../lib/gemini.js";

function buildConfig(overrides = {}) {
  return {
    geminiModel: "gemini-2.5-flash",
    activeGeminiModel: "gemini-2.5-flash",
    geminiEnableCodeExecution: true,
    geminiEnableGoogleMaps: true,
    geminiEnableGoogleSearch: false,
    geminiEnableUrlContext: false,
    geminiGoogleMapsLocation: {
      latitude: 40.758,
      longitude: -73.9855,
    },
    geminiThinkingLevel: "",
    geminiCachedContent: { fallback: "", perModel: {} },
    systemPrompt: "",
    optionalInstruction: "",
    ...overrides,
  };
}

test("maps prompts attach google maps without code execution", () => {
  const body = buildRequestBody({
    prompt: "Find coffee shops near me that are open now.",
    history: [],
    config: buildConfig(),
  });

  assert.deepEqual(body.tools, [{ googleMaps: {} }]);
  assert.deepEqual(body.toolConfig, {
    retrievalConfig: {
      latLng: {
        latitude: 40.758,
        longitude: -73.9855,
      },
    },
  });
});

test("math prompts attach code execution without google maps config", () => {
  const body = buildRequestBody({
    prompt: "What is 123 * 456?",
    history: [],
    config: buildConfig(),
  });

  assert.deepEqual(body.tools, [{ code_execution: {} }]);
  assert.equal(body.toolConfig, undefined);
});

test("mixed geo and math prompts prefer google maps to avoid an invalid tool combination", () => {
  const body = buildRequestBody({
    prompt:
      "Find restaurants near me within 10 minutes and tell me whether 17 * 19 is 323.",
    history: [],
    config: buildConfig(),
  });

  assert.deepEqual(body.tools, [{ googleMaps: {} }]);
  assert.equal(
    body.tools.some((tool) => Object.hasOwn(tool, "code_execution")),
    false,
  );
});
