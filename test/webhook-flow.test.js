import test from "node:test";
import assert from "node:assert/strict";
import { Readable } from "node:stream";

import { createWebhookHandler } from "../api/webhook.js";

function buildConfig(overrides = {}) {
  return {
    pageAccessToken: "page-token",
    verifyToken: "verify-token",
    postgresUrl: "postgres://example/db",
    geminiApiKeys: ["gemini-key"],
    geminiModels: ["gemini-2.5-flash"],
    geminiModel: "gemini-2.5-flash",
    geminiEnableGoogleSearch: false,
    geminiEnableCodeExecution: true,
    geminiEnableGoogleMaps: true,
    geminiEnableUrlContext: true,
    geminiThinkingLevel: "",
    geminiCachedContent: { fallback: "", byModel: {} },
    optionalInstruction: "",
    systemPrompt: "",
    missingMessagingKeys: [],
    missingVerificationKeys: [],
    reliabilityRetryLimit: 0,
    reliabilityRetryBaseMs: 1,
    reliabilityRetentionDays: 14,
    healthLookbackHours: 24,
    memoryMaxItems: 6,
    memoryMaxChars: 700,
    summaryMaxChars: 600,
    promptContextMaxChars: 1400,
    ...overrides,
  };
}

function createRequest(payload) {
  const req = Readable.from([Buffer.from(JSON.stringify(payload))]);
  req.method = "POST";
  req.url = "/api/webhook";
  req.headers = {
    host: "example.test",
  };
  return req;
}

function createResponse() {
  return {
    statusCode: 0,
    headers: {},
    body: "",
    setHeader(name, value) {
      this.headers[name] = value;
    },
    end(payload = "") {
      this.body += payload;
    },
  };
}

function createPayload(message, attachments = []) {
  return {
    object: "page",
    entry: [
      {
        messaging: [
          {
            sender: { id: "user-123" },
            message: {
              mid: "mid.1",
              text: message,
              attachments,
            },
          },
        ],
      },
    ],
  };
}

test("webhook handles commands without calling Gemini", async () => {
  const sentMessages = [];
  let geminiCalls = 0;
  const eventUpdates = [];
  const handler = createWebhookHandler({
    configLoader: buildConfig,
    conversationStoreFactory: async () => ({
      async beginEventProcessing() {
        return { inserted: true, tracked: true };
      },
      async updateEventProcessing(update) {
        eventUpdates.push(update);
      },
    }),
    geminiReply: async () => {
      geminiCalls += 1;
      return "should not run";
    },
    sendAction: async () => {},
    sendTextMessageImpl: async (_senderId, text) => {
      sentMessages.push(text);
    },
    profileRepair: () => null,
    waitUntilImpl: () => {},
    logger: {
      info() {},
      warn() {},
      error() {},
    },
  });
  const req = createRequest(createPayload("/help"));
  const res = createResponse();

  await handler(req, res);

  assert.equal(res.statusCode, 200);
  assert.equal(res.body, "EVENT_RECEIVED");
  assert.equal(geminiCalls, 0);
  assert.equal(sentMessages.length, 1);
  assert.match(sentMessages[0], /\/help - show commands and capabilities/);
  assert.equal(eventUpdates.at(-1)?.status, "completed");
  assert.equal(eventUpdates.at(-1)?.stage, "command");
});

test("webhook skips duplicate tracked events", async () => {
  const sentMessages = [];
  let geminiCalls = 0;
  const handler = createWebhookHandler({
    configLoader: buildConfig,
    conversationStoreFactory: async () => ({
      async beginEventProcessing() {
        return { inserted: false, tracked: true };
      },
    }),
    geminiReply: async () => {
      geminiCalls += 1;
      return "should not run";
    },
    sendAction: async () => {},
    sendTextMessageImpl: async (_senderId, text) => {
      sentMessages.push(text);
    },
    profileRepair: () => null,
    waitUntilImpl: () => {},
    logger: {
      info() {},
      warn() {},
      error() {},
    },
  });

  await handler(createRequest(createPayload("hello")), createResponse());

  assert.equal(geminiCalls, 0);
  assert.deepEqual(sentMessages, []);
});

test("webhook does not send a second fallback when model turn persistence fails after reply delivery", async () => {
  const sentMessages = [];
  const failedOutbound = [];
  const eventUpdates = [];
  const handler = createWebhookHandler({
    configLoader: buildConfig,
    conversationStoreFactory: async () => ({
      async beginEventProcessing() {
        return { inserted: true, tracked: true };
      },
      async updateEventProcessing(update) {
        eventUpdates.push(update);
      },
      async saveInboundTurn() {
        return { inserted: true, messageId: 1 };
      },
      async getConversationHistory() {
        return [];
      },
      async getConversationSummary() {
        return "";
      },
      async findRelevantMemory() {
        return [];
      },
      async getLatestLocation() {
        return null;
      },
      async getLatestImageContext() {
        return null;
      },
      async saveModelTurn() {
        throw new Error("database unavailable");
      },
      async recordFailedOutbound(record) {
        failedOutbound.push(record);
      },
    }),
    geminiReply: async () => "Hello from Gemini",
    sendAction: async () => {},
    sendTextMessageImpl: async (_senderId, text) => {
      sentMessages.push(text);
    },
    profileRepair: () => null,
    waitUntilImpl: () => {},
    logger: {
      info() {},
      warn() {},
      error() {},
    },
  });

  await handler(createRequest(createPayload("hello")), createResponse());

  assert.deepEqual(sentMessages, ["Hello from Gemini"]);
  assert.equal(failedOutbound.length, 0);
  assert.equal(eventUpdates.at(-1)?.status, "failed");
  assert.equal(eventUpdates.at(-1)?.replySent, true);
  assert.equal(eventUpdates.at(-1)?.modelTurnSaved, false);
});

test("webhook saves location attachments and replies with location readiness text", async () => {
  const sentMessages = [];
  const savedLocations = [];
  const handler = createWebhookHandler({
    configLoader: buildConfig,
    conversationStoreFactory: async () => ({
      async beginEventProcessing() {
        return { inserted: true, tracked: true };
      },
      async updateEventProcessing() {},
      async saveLatestLocation(location) {
        savedLocations.push(location);
        return {
          latitude: location.latitude,
          longitude: location.longitude,
        };
      },
    }),
    geminiReply: async () => "should not run",
    sendAction: async () => {},
    sendTextMessageImpl: async (_senderId, text) => {
      sentMessages.push(text);
    },
    profileRepair: () => null,
    waitUntilImpl: () => {},
    logger: {
      info() {},
      warn() {},
      error() {},
    },
  });

  await handler(
    createRequest(
      createPayload("", [
        {
          type: "location",
          payload: {
            coordinates: {
              lat: 6.5244,
              long: 3.3792,
            },
          },
        },
      ]),
    ),
    createResponse(),
  );

  assert.deepEqual(savedLocations, [
    {
      senderId: "user-123",
      latitude: 6.5244,
      longitude: 3.3792,
      sourceEventId: "mid.1",
    },
  ]);
  assert.equal(sentMessages.length, 1);
  assert.match(sentMessages[0], /I saved your location/);
  assert.match(sentMessages[0], /Latitude: 6.5244/);
});

test("webhook answers direct self-location prompts locally when a location is saved", async () => {
  const sentMessages = [];
  let geminiCalls = 0;
  const handler = createWebhookHandler({
    configLoader: buildConfig,
    conversationStoreFactory: async () => ({
      async beginEventProcessing() {
        return { inserted: true, tracked: true };
      },
      async updateEventProcessing() {},
      async saveInboundTurn() {
        return { inserted: true, messageId: 1 };
      },
      async getConversationHistory() {
        return [];
      },
      async getConversationSummary() {
        return "";
      },
      async findRelevantMemory() {
        return [];
      },
      async getLatestLocation() {
        return {
          latitude: 6.5244,
          longitude: 3.3792,
        };
      },
      async saveModelTurn() {},
    }),
    geminiReply: async () => {
      geminiCalls += 1;
      return "should not run";
    },
    sendAction: async () => {},
    sendTextMessageImpl: async (_senderId, text) => {
      sentMessages.push(text);
    },
    profileRepair: () => null,
    waitUntilImpl: () => {},
    logger: {
      info() {},
      warn() {},
      error() {},
    },
  });

  await handler(
    createRequest(createPayload("What is my location?")),
    createResponse(),
  );

  assert.equal(geminiCalls, 0);
  assert.equal(sentMessages.length, 1);
  assert.match(sentMessages[0], /The latest location pin you shared is:/);
  assert.match(sentMessages[0], /Latitude: 6.5244/);
});

test("webhook explains when no saved location exists for a direct self-location prompt", async () => {
  const locationRequests = [];
  let geminiCalls = 0;
  const handler = createWebhookHandler({
    configLoader: buildConfig,
    conversationStoreFactory: async () => ({
      async beginEventProcessing() {
        return { inserted: true, tracked: true };
      },
      async updateEventProcessing() {},
      async saveInboundTurn() {
        return { inserted: true, messageId: 1 };
      },
      async getConversationHistory() {
        return [];
      },
      async getConversationSummary() {
        return "";
      },
      async findRelevantMemory() {
        return [];
      },
      async getLatestLocation() {
        return null;
      },
      async saveModelTurn() {},
    }),
    geminiReply: async () => {
      geminiCalls += 1;
      return "should not run";
    },
    sendAction: async () => {},
    sendLocationRequestImpl: async (_senderId, text) => {
      locationRequests.push(text);
    },
    profileRepair: () => null,
    waitUntilImpl: () => {},
    logger: {
      info() {},
      warn() {},
      error() {},
    },
  });

  await handler(
    createRequest(createPayload("What is my location?")),
    createResponse(),
  );

  assert.equal(geminiCalls, 0);
  assert.equal(locationRequests.length, 1);
  assert.match(
    locationRequests[0],
    /Tap the Messenger location button below/,
  );
});

test("webhook uses the configured default location when no saved location exists", async () => {
  const sentMessages = [];
  let geminiCalls = 0;
  const handler = createWebhookHandler({
    configLoader: () =>
      buildConfig({
        geminiGoogleMapsLocation: {
          latitude: 40.758,
          longitude: -73.9855,
        },
      }),
    conversationStoreFactory: async () => ({
      async beginEventProcessing() {
        return { inserted: true, tracked: true };
      },
      async updateEventProcessing() {},
      async saveInboundTurn() {
        return { inserted: true, messageId: 1 };
      },
      async getConversationHistory() {
        return [];
      },
      async getConversationSummary() {
        return "";
      },
      async findRelevantMemory() {
        return [];
      },
      async getLatestLocation() {
        return null;
      },
      async saveModelTurn() {},
    }),
    geminiReply: async () => {
      geminiCalls += 1;
      return "should not run";
    },
    sendAction: async () => {},
    sendTextMessageImpl: async (_senderId, text) => {
      sentMessages.push(text);
    },
    profileRepair: () => null,
    waitUntilImpl: () => {},
    logger: {
      info() {},
      warn() {},
      error() {},
    },
  });

  await handler(
    createRequest(createPayload("What is my location?")),
    createResponse(),
  );

  assert.equal(geminiCalls, 0);
  assert.equal(sentMessages.length, 1);
  assert.match(sentMessages[0], /configured default location context/);
  assert.match(sentMessages[0], /Latitude: 40.758/);
});
