import test from "node:test";
import assert from "node:assert/strict";

import { handleCommand, parseCommand } from "../lib/commands.js";

test("parseCommand recognizes slash commands and get started payload", () => {
  assert.deepEqual(parseCommand("/summary"), {
    name: "summary",
    args: "",
    raw: "/summary",
  });

  assert.deepEqual(parseCommand("Postback payload: NUBAGENT_GET_STARTED"), {
    name: "help",
    args: "",
    raw: "Postback payload: NUBAGENT_GET_STARTED",
  });
});

test("handleCommand saves explicit memory through /memory add", async () => {
  const saved = [];
  const result = await handleCommand({
    command: parseCommand("/memory add I prefer concise replies"),
    senderId: "user-1",
    store: {
      async saveMemory(memory) {
        saved.push(memory);
        return {
          content: memory.content,
          kind: memory.kind,
        };
      },
    },
  });

  assert.equal(result.reply, "Saved memory: I prefer concise replies");
  assert.deepEqual(saved, [
    {
      senderId: "user-1",
      content: "I prefer concise replies",
      kind: "preference",
      source: "command",
    },
  ]);
});

test("handleCommand deletes all memory through /forget all", async () => {
  const result = await handleCommand({
    command: parseCommand("/forget all"),
    senderId: "user-2",
    store: {
      async deleteMemory() {
        return {
          deletedCount: 2,
          deletedItems: ["You live in Lagos.", "You prefer spicy food."],
        };
      },
    },
  });

  assert.equal(result.reply, "Deleted 2 saved memories.");
});
