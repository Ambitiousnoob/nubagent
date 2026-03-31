import test from "node:test";
import assert from "node:assert/strict";

import {
  buildMessengerEndpoint,
  splitTextIntoChunks,
} from "../lib/messenger.js";

test("buildMessengerEndpoint prefers PAGE_ID when provided", () => {
  const endpoint = buildMessengerEndpoint({
    pageId: "12345",
    graphApiVersion: "v23.0",
  });

  assert.equal(endpoint, "https://graph.facebook.com/v23.0/12345/messages");
});

test("splitTextIntoChunks preserves shorter messages", () => {
  assert.deepEqual(splitTextIntoChunks("hello"), ["hello"]);
});

test("splitTextIntoChunks breaks oversized messages into multiple parts", () => {
  const longText = "word ".repeat(500);
  const parts = splitTextIntoChunks(longText);

  assert.equal(parts.length > 1, true);
  assert.equal(
    parts.every((part) => part.length <= 1800),
    true,
  );
});
