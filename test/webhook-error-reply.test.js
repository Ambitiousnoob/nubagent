import test from "node:test";
import assert from "node:assert/strict";

import { buildFailureReply } from "../api/webhook.js";

const GENERIC_REPLY =
  "I hit an upstream error while talking to Gemini. Please try again in a moment.";

test("returns the upstream Gemini API message for Gemini-stage failures", () => {
  const error = new Error("Gemini API 400: API key not valid.");
  error.stage = "gemini";
  error.apiMessage = "API key not valid.";

  assert.equal(buildFailureReply(error), "API key not valid.");
});

test("returns Gemini-authored non-api messages when they are already user-facing", () => {
  const error = new Error("Gemini returned no text response.");
  error.stage = "gemini:image_context";

  assert.equal(buildFailureReply(error), "Gemini returned no text response.");
});

test("keeps the generic fallback for non-Gemini failures", () => {
  const error = new Error("database timeout");
  error.stage = "db:load_history";

  assert.equal(buildFailureReply(error), GENERIC_REPLY);
});

test("keeps the generic fallback for Gemini-stage transport errors without an upstream message", () => {
  const error = new TypeError("fetch failed");
  error.stage = "gemini";

  assert.equal(buildFailureReply(error), GENERIC_REPLY);
});
