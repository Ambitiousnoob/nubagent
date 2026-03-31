import test from "node:test";
import assert from "node:assert/strict";

import {
  createConversationStore,
  mapRowsToConversationHistory,
} from "../lib/history.js";

function createMockPool() {
  const queries = [];
  const insertedEventIds = new Set();
  let nextId = 1;

  return {
    queries,
    async query(text, values = []) {
      queries.push({ text, values });

      if (text.includes("CREATE TABLE")) {
        return { rowCount: 0, rows: [] };
      }

      if (text.includes("CREATE INDEX")) {
        return { rowCount: 0, rows: [] };
      }

      if (text.includes("INSERT INTO messenger_messages") && values[2]) {
        if (insertedEventIds.has(values[2])) {
          return { rowCount: 0, rows: [] };
        }

        insertedEventIds.add(values[2]);
        return { rowCount: 1, rows: [{ id: nextId++ }] };
      }

      if (text.includes("INSERT INTO messenger_messages")) {
        return { rowCount: 1, rows: [{ id: nextId++ }] };
      }

      if (text.includes("SELECT id, role, content")) {
        return {
          rowCount: 2,
          rows: [
            { id: 3, role: "model", content: "Latest reply" },
            { id: 2, role: "user", content: "Earlier question" },
          ],
        };
      }

      return { rowCount: 0, rows: [] };
    },
  };
}

test("mapRowsToConversationHistory returns oldest-to-newest Gemini turns", () => {
  const history = mapRowsToConversationHistory([
    { role: "model", content: "Newest" },
    { role: "user", content: "Oldest" },
  ]);

  assert.deepEqual(history, [
    { role: "user", parts: [{ text: "Oldest" }] },
    { role: "model", parts: [{ text: "Newest" }] },
  ]);
});

test("conversation store deduplicates inbound events by source_event_id", async () => {
  const pool = createMockPool();
  const store = createConversationStore({ pool });

  const first = await store.saveInboundTurn({
    senderId: "psid-1",
    text: "hello",
    sourceEventId: "mid-1",
  });
  const duplicate = await store.saveInboundTurn({
    senderId: "psid-1",
    text: "hello",
    sourceEventId: "mid-1",
  });

  assert.deepEqual(first, { inserted: true, messageId: 1 });
  assert.deepEqual(duplicate, { inserted: false, messageId: null });
});

test("conversation store returns prior history excluding the current message id", async () => {
  const pool = createMockPool();
  const store = createConversationStore({ pool });
  const history = await store.getConversationHistory("psid-1", {
    excludeMessageId: 7,
  });

  assert.deepEqual(history, [
    { role: "user", parts: [{ text: "Earlier question" }] },
    { role: "model", parts: [{ text: "Latest reply" }] },
  ]);

  const selectQuery = pool.queries.find((entry) =>
    entry.text.includes("SELECT id, role, content"),
  );

  assert.deepEqual(selectQuery.values, ["psid-1", 7, 11]);
});
