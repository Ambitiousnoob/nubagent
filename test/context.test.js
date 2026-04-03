import test from "node:test";
import assert from "node:assert/strict";

import {
  buildPromptWithPersistentContext,
  extractMemoryCandidates,
} from "../lib/context.js";

test("extractMemoryCandidates finds stable user facts and preferences", () => {
  const memories = extractMemoryCandidates(
    "My name is Ayo. I live in Lagos. My favorite language is JavaScript.",
  );

  assert.deepEqual(memories, [
    { content: "Your name is Ayo.", kind: "fact" },
    { content: "You live in Lagos.", kind: "fact" },
    {
      content: "Your favorite language is JavaScript.",
      kind: "preference",
    },
  ]);
});

test("buildPromptWithPersistentContext places summary, memory, image context, and user message in order", () => {
  const prompt = buildPromptWithPersistentContext({
    prompt: "What should I cook tonight?",
    imageContext: "A kitchen counter with tomatoes, onions, and rice.",
    conversationSummary: "The user wants simple dinner ideas and short replies.",
    memoryEntries: [
      { content: "You prefer spicy food.", kind: "preference" },
      { content: "You live in Lagos.", kind: "fact" },
    ],
    sharedLocation: {
      latitude: 6.5244,
      longitude: 3.3792,
    },
    maxContextChars: 1200,
  });

  assert.match(prompt, /^Conversation summary:/);
  assert.match(prompt, /Remembered user details:/);
  assert.match(prompt, /Latest shared location:/);
  assert.match(prompt, /Image context:/);
  assert.match(prompt, /User message:\nWhat should I cook tonight\?/);
});
