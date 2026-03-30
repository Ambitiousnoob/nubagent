import { describe, expect, it, vi } from "vitest";
import { loadCommonJsModule } from "../../src/__tests__/loadCommonJsModule.js";

const createRes = () => {
  const res = {
    headers: {},
    statusCode: 200,
    body: "",
    setHeader: vi.fn((key, value) => {
      res.headers[key] = value;
    }),
    status: vi.fn((code) => {
      res.statusCode = code;
      return res;
    }),
    write: vi.fn((value) => {
      res.body += value;
      return true;
    }),
    end: vi.fn((value = "") => {
      res.body += value;
      return res;
    }),
  };
  return res;
};

const buildHandler = ({
  body = {},
  scopedMemory = null,
  shouldUseMemory = false,
  runLiteHostChat = vi.fn(async () => ({
    body: { id: "chatcmpl-test" },
    reply: { content: "Delegated answer" },
  })),
} = {}) => {
  const mocks = {
    "../lib/web": {
      readBody: vi.fn(async () => body),
    },
    "../lib/litehost-chat": {
      metadataPayload: vi.fn(() => ({ ok: true })),
      wantsStream: vi.fn(() => false),
      runLiteHostChat,
      createChatResponsePayload: vi.fn((payload, reply) => ({
        id: payload?.id || "chatcmpl-test",
        choices: [{ message: { content: reply?.content || "" } }],
      })),
      normalizeChatBody: vi.fn((value) => value),
    },
    "../lib/chat-memory": {
      loadScopedChatMemory: vi.fn(async () => scopedMemory),
      saveScopedChatMemory: vi.fn(async () => scopedMemory),
      mergeChatMemory: vi.fn((memory) => memory || {}),
      shouldUseScopedChatMemory: vi.fn(() => shouldUseMemory),
      buildMessagesWithScopedMemory: vi.fn((messages) => messages),
    },
    "../lib/api-key-memory": {
      saveApiKeyMemoryEntries: vi.fn(async () => {}),
      searchApiKeyMemoryDetailed: vi.fn(async () => ({ results: [], meta: null })),
      formatApiKeyMemoryContext: vi.fn(() => ""),
    },
  };

  const handler = loadCommonJsModule(
    "/root/.bot/.downloads/nubagent/api/chat.js",
    mocks,
  );

  return {
    handler,
    mocks,
    runLiteHostChat,
  };
};

describe("/api/chat", () => {
  it("forwards scoped delegation context and research overrides to the main chat runtime", async () => {
    const runLiteHostChat = vi.fn(async () => ({
      body: { id: "chatcmpl-main" },
      reply: { content: "Main lane response" },
    }));
    const { handler } = buildHandler({
      body: {
        messages: [
          {
            role: "user",
            content: "Research this and delegate if needed.",
          },
        ],
        searchProviderKeys: {
          tavily: "tvly-test-key-1234567890",
        },
        researchProvider: "openrouter",
        researchModel: "nvidia/nemotron-3-super-120b-a12b:free",
        researchModelChain: [
          "nvidia/nemotron-3-super-120b-a12b:free",
          "meta-llama/llama-3.3-70b-instruct:free",
        ],
        researchRoundRobin: true,
        researchProviderKeys: {
          openrouter: "sk-or-v1-1234567890abcdefghijklmnop",
        },
      },
      scopedMemory: {
        stateKey: "scope:test-chat",
        scope: "state_key",
        dbKey: "db:test-chat",
        memory: {},
      },
      runLiteHostChat,
    });
    const req = {
      method: "POST",
      headers: {},
      url: "/api/chat",
    };
    const res = createRes();

    await handler(req, res);

    expect(runLiteHostChat).toHaveBeenCalledWith(expect.objectContaining({
      messages: [
        {
          role: "user",
          content: "Research this and delegate if needed.",
        },
      ],
      delegationContext: {
        scopeKey: "scope:test-chat",
        scope: "state_key",
      },
      searchProviderKeys: {
        tavily: "tvly-test-key-1234567890",
      },
      researchProvider: "openrouter",
      researchModel: "nvidia/nemotron-3-super-120b-a12b:free",
      researchModelChain: [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "meta-llama/llama-3.3-70b-instruct:free",
      ],
      researchRoundRobin: true,
      researchProviderKeys: {
        openrouter: "sk-or-v1-1234567890abcdefghijklmnop",
      },
    }));
    expect(res.statusCode).toBe(200);
    expect(JSON.parse(res.body)).toMatchObject({
      id: "chatcmpl-main",
      choices: [{ message: { content: "Main lane response" } }],
    });
  });
});
