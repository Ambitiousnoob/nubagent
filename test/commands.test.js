import test from "node:test";
import assert from "node:assert/strict";

import { handleCommand, parseCommand } from "../lib/commands.js";

test("parseCommand recognizes bang commands and get started payload", () => {
  assert.deepEqual(parseCommand("!summary"), {
    name: "summary",
    args: "",
    raw: "!summary",
  });
  assert.deepEqual(parseCommand("!credits"), {
    name: "credits",
    args: "",
    raw: "!credits",
  });
  assert.equal(parseCommand("/summary"), null);
  assert.equal(parseCommand("summary"), null);
  assert.equal(parseCommand("help"), null);
  assert.equal(parseCommand("location"), null);

  assert.deepEqual(parseCommand("Postback payload: NUBAGENT_GET_STARTED"), {
    name: "help",
    args: "",
    raw: "Postback payload: NUBAGENT_GET_STARTED",
  });
});

test("handleCommand saves explicit memory through !memory add", async () => {
  const saved = [];
  const result = await handleCommand({
    command: parseCommand("!memory add I prefer concise replies"),
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

test("handleCommand deletes all memory through !forget all", async () => {
  const result = await handleCommand({
    command: parseCommand("!forget all"),
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

test("handleCommand returns project credits through !credits", async () => {
  const result = await handleCommand({
    command: parseCommand("!credits"),
    senderId: "user-credits",
    store: {},
  });

  assert.match(result.reply, /^Credits:/);
  assert.match(result.reply, /Built with nubagent/);
  assert.match(
    result.reply,
    /https:\/\/github\.com\/ambitiousnoob\/nubagent/,
  );
});

test("handleCommand returns a location capture link", async () => {
  const result = await handleCommand({
    command: parseCommand("!location"),
    senderId: "user-3",
    store: {
      async createLocationCaptureToken() {
        return {
          token: "capture-token-123",
        };
      },
    },
    config: {
      locationCaptureTtlMinutes: 15,
    },
    baseUrl: "https://example.test",
  });

  assert.match(result.reply, /Open this secure link and allow location access/);
  assert.match(
    result.reply,
    /https:\/\/example\.test\/api\/location-capture\?token=capture-token-123/,
  );
  assert.match(result.reply, /The link expires in 15 minutes/);
});

test("handleCommand shows the saved location when requested explicitly", async () => {
  const result = await handleCommand({
    command: parseCommand("!location show"),
    senderId: "user-3",
    store: {
      async getLatestLocation() {
        return {
          latitude: 6.5244,
          longitude: 3.3792,
        };
      },
    },
  });

  assert.match(result.reply, /Saved location:/);
  assert.match(result.reply, /Latitude: 6.5244/);
  assert.match(result.reply, /Longitude: 3.3792/);
});
