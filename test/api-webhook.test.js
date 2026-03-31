import test from "node:test";
import assert from "node:assert/strict";
import { Readable } from "node:stream";

import {
  createWebhookHandler,
  extractInboundPrompt,
  extractSourceEventId,
  verifySignature,
} from "../api/webhook.js";

function createResponseCapture() {
  return {
    statusCode: 0,
    headers: {},
    body: "",
    setHeader(name, value) {
      this.headers[name] = value;
    },
    end(payload) {
      this.body = payload;
    },
  };
}

function createJsonRequest(body, headers = {}) {
  const stream = Readable.from([JSON.stringify(body)]);
  stream.method = "POST";
  stream.url = "/api/webhook";
  stream.headers = {
    host: "example.com",
    ...headers,
  };
  return stream;
}

function createConfig(overrides = {}) {
  return {
    geminiApiKeys: ["gemini-1", "gemini-2"],
    pageAccessToken: "page-token",
    pageId: "12345",
    postgresUrl: "postgres://database",
    verifyToken: "verify-token",
    facebookAppSecret: "",
    graphApiVersion: "v23.0",
    geminiModel: "gemini-3-flash-preview",
    geminiThinkingLevel: "LOW",
    systemPrompt: "You are helpful.",
    missingVerificationKeys: [],
    missingMessagingKeys: [],
    ...overrides,
  };
}

test("extractInboundPrompt prefers text and handles postbacks", () => {
  assert.equal(extractInboundPrompt({ message: { text: " hello " } }), "hello");
  assert.equal(
    extractInboundPrompt({ postback: { payload: "OPEN_HELP" } }),
    "Postback payload: OPEN_HELP",
  );
});

test("extractSourceEventId reads message and postback ids", () => {
  assert.equal(extractSourceEventId({ message: { mid: "m-1" } }), "m-1");
  assert.equal(extractSourceEventId({ postback: { mid: "p-1" } }), "p-1");
  assert.equal(extractSourceEventId({}), null);
});

test("verifySignature allows unsigned requests when no app secret is configured", () => {
  assert.equal(verifySignature(Buffer.from("body"), undefined, ""), true);
});

test("webhook GET verification returns the hub challenge", async () => {
  const handler = createWebhookHandler({
    configLoader: () => createConfig(),
  });
  const req = {
    method: "GET",
    url: "/api/webhook?hub.mode=subscribe&hub.verify_token=verify-token&hub.challenge=12345",
    headers: { host: "example.com" },
  };
  const res = createResponseCapture();

  await handler(req, res);

  assert.equal(res.statusCode, 200);
  assert.equal(res.body, "12345");
});

test("webhook processes a happy-path text event and persists the reply", async () => {
  const sentActions = [];
  const sentMessages = [];
  const storeCalls = [];

  const handler = createWebhookHandler({
    configLoader: () => createConfig(),
    conversationStoreFactory: async () => ({
      async saveInboundTurn(payload) {
        storeCalls.push(["saveInboundTurn", payload]);
        return { inserted: true, messageId: 99 };
      },
      async getConversationHistory(senderId, options) {
        storeCalls.push(["getConversationHistory", { senderId, options }]);
        return [{ role: "user", parts: [{ text: "Earlier" }] }];
      },
      async saveModelTurn(payload) {
        storeCalls.push(["saveModelTurn", payload]);
      },
    }),
    geminiReply: async ({ prompt, history }) => {
      assert.equal(prompt, "What is new?");
      assert.deepEqual(history, [
        { role: "user", parts: [{ text: "Earlier" }] },
      ]);
      return "Fresh answer";
    },
    sendAction: async (senderId, action) => {
      sentActions.push([senderId, action]);
    },
    sendTextMessageImpl: async (senderId, text) => {
      sentMessages.push([senderId, text]);
    },
  });

  const req = createJsonRequest({
    object: "page",
    entry: [
      {
        messaging: [
          {
            sender: { id: "psid-1" },
            message: {
              mid: "mid-1",
              text: "What is new?",
            },
          },
        ],
      },
    ],
  });
  const res = createResponseCapture();

  await handler(req, res);

  assert.equal(res.statusCode, 200);
  assert.equal(res.body, "EVENT_RECEIVED");
  assert.deepEqual(sentActions, [
    ["psid-1", "mark_seen"],
    ["psid-1", "typing_on"],
    ["psid-1", "typing_off"],
  ]);
  assert.deepEqual(sentMessages, [["psid-1", "Fresh answer"]]);
  assert.deepEqual(storeCalls, [
    [
      "saveInboundTurn",
      { senderId: "psid-1", text: "What is new?", sourceEventId: "mid-1" },
    ],
    [
      "getConversationHistory",
      { senderId: "psid-1", options: { excludeMessageId: 99 } },
    ],
    ["saveModelTurn", { senderId: "psid-1", text: "Fresh answer" }],
  ]);
});

test("webhook skips duplicate inbound events", async () => {
  const sentMessages = [];

  const handler = createWebhookHandler({
    configLoader: () => createConfig(),
    conversationStoreFactory: async () => ({
      async saveInboundTurn() {
        return { inserted: false, messageId: null };
      },
    }),
    sendTextMessageImpl: async (senderId, text) => {
      sentMessages.push([senderId, text]);
    },
    logger: {
      info() {},
      error() {},
    },
  });

  const req = createJsonRequest({
    object: "page",
    entry: [
      {
        messaging: [
          {
            sender: { id: "psid-1" },
            message: {
              mid: "mid-1",
              text: "What is new?",
            },
          },
        ],
      },
    ],
  });
  const res = createResponseCapture();

  await handler(req, res);

  assert.equal(res.statusCode, 200);
  assert.deepEqual(sentMessages, []);
});

test("webhook returns attachment fallback without calling Gemini", async () => {
  const sentMessages = [];
  let geminiCalled = false;

  const handler = createWebhookHandler({
    configLoader: () => createConfig(),
    conversationStoreFactory: async () => ({
      async saveInboundTurn() {
        throw new Error("store should not be used for attachments");
      },
    }),
    geminiReply: async () => {
      geminiCalled = true;
      return "should not happen";
    },
    sendTextMessageImpl: async (senderId, text) => {
      sentMessages.push([senderId, text]);
    },
  });

  const req = createJsonRequest({
    object: "page",
    entry: [
      {
        messaging: [
          {
            sender: { id: "psid-1" },
            message: {
              attachments: [{ type: "image" }],
            },
          },
        ],
      },
    ],
  });
  const res = createResponseCapture();

  await handler(req, res);

  assert.equal(res.statusCode, 200);
  assert.equal(geminiCalled, false);
  assert.deepEqual(sentMessages, [
    [
      "psid-1",
      "I can reply to text messages right now. Send text and I will forward it to Gemini.",
    ],
  ]);
});

test("webhook rejects invalid signatures when app secret is configured", async () => {
  const handler = createWebhookHandler({
    configLoader: () =>
      createConfig({
        facebookAppSecret: "secret",
      }),
  });
  const req = createJsonRequest(
    {
      object: "page",
      entry: [],
    },
    {
      "x-hub-signature-256": "sha256=invalid",
    },
  );
  const res = createResponseCapture();

  await handler(req, res);

  assert.equal(res.statusCode, 403);
  assert.match(res.body, /Invalid webhook signature/);
});

test("webhook returns 500 when required env is missing", async () => {
  const handler = createWebhookHandler({
    configLoader: () =>
      createConfig({
        missingMessagingKeys: ["POSTGRES_URL"],
      }),
  });
  const req = createJsonRequest({
    object: "page",
    entry: [],
  });
  const res = createResponseCapture();

  await handler(req, res);

  assert.equal(res.statusCode, 500);
  assert.match(res.body, /POSTGRES_URL/);
});
