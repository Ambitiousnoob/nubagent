import { beforeEach, describe, expect, it, vi } from "vitest";
import { loadCommonJsModule } from "../../src/__tests__/loadCommonJsModule.js";

const buildLiteHostModule = ({
  delegateResearchHandler = vi.fn(async () => JSON.stringify({ delegated: true })),
} = {}) => ({
  module: loadCommonJsModule(
    "/root/.bot/.downloads/nubagent/lib/litehost-chat.js",
    {
    "./api-key-rotation.cjs": {
      getApiKeysFromEnv: vi.fn(() => []),
      getRotatingApiKey: vi.fn(() => null),
      getRotatingValue: vi.fn((bucket, values) => (
        Array.isArray(values) ? values[0] || null : values || null
      )),
    },
    "./ai-control": {
      clampCompletionTokens: vi.fn((value, fallback = 4096) => {
        const numeric = Number(value);
        return Number.isFinite(numeric) ? numeric : fallback;
      }),
      getBackoffDelayMs: vi.fn(() => 0),
      sleep: vi.fn(async () => {}),
    },
    "./web": {
      assertPublicHttpUrl: vi.fn(async (url) => url),
    },
    "@google/genai": {
      GoogleGenAI: class GoogleGenAI {},
    },
    "./tools/calculate": {
      definition: { function: { name: "calculate", description: "", parameters: {} } },
      handler: vi.fn(async () => "0"),
    },
    "./tools/web_search": {
      definition: { function: { name: "web_search", description: "", parameters: {} } },
      handler: vi.fn(async () => "[]"),
    },
    "./tools/web_fetch": {
      definition: { function: { name: "web_fetch", description: "", parameters: {} } },
      handler: vi.fn(async () => ""),
    },
    "./tools/search_images": {
      definition: { function: { name: "search_images", description: "", parameters: {} } },
      handler: vi.fn(async () => "[]"),
    },
    "./tools/view_image": {
      definition: { function: { name: "view_image", description: "", parameters: {} } },
      handler: vi.fn(async () => ""),
    },
    "./tools/delegate_research": {
      definition: {
        type: "function",
        function: {
          name: "delegate_research",
          description: "Delegate grounded research to the orchestration subagents",
          parameters: {
            type: "object",
            properties: {
              query: { type: "string" },
            },
            required: ["query"],
            additionalProperties: false,
          },
        },
      },
      handler: delegateResearchHandler,
    },
  },
  ),
  delegateResearchHandler,
});

const createJsonResponse = (payload) => ({
  ok: true,
  status: 200,
  headers: {
    get(name) {
      if (String(name || "").toLowerCase() === "content-type") {
        return "application/json; charset=utf-8";
      }
      return null;
    },
  },
  json: async () => payload,
  text: async () => JSON.stringify(payload),
});

describe("litehost chat provider routing", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.clearAllMocks();
    global.fetch = vi.fn();
  });

  it("keeps the main chat default on Gemini Flash Lite when no research override is supplied", () => {
    const { module: liteHost } = buildLiteHostModule();
    const { normalizeChatBody } = liteHost;

    const normalized = normalizeChatBody({
      providerApiKeys: {
        google: "google-test-key-1234567890",
      },
      messages: [{ role: "user", content: "Hello from the main agent." }],
    });

    expect(normalized.provider).toBe("gemini");
    expect(normalized.model).toBe("gemini-2.5-flash-lite");
  });

  it("defaults research OpenRouter requests to Nemotron when no explicit model is provided", () => {
    const { module: liteHost } = buildLiteHostModule();
    const { normalizeChatBody } = liteHost;

    const normalized = normalizeChatBody({
      provider: "openrouter",
      providerApiKeys: {
        openrouter: "sk-or-v1-1234567890abcdefghijklmnop",
      },
      research_mode: true,
      messages: [{ role: "user", content: "How should the verifier swarm route models?" }],
    });

    expect(normalized.provider).toBe("openrouter");
    expect(normalized.model).toBe("nvidia/nemotron-3-super-120b-a12b:free");
  });

  it("routes OpenRouter research calls through the chat completions endpoint", async () => {
    global.fetch.mockResolvedValue(createJsonResponse({
      choices: [
        {
          message: {
            content: "OpenRouter reply",
          },
        },
      ],
      usage: {
        prompt_tokens: 12,
        completion_tokens: 4,
      },
    }));

    const { module: liteHost } = buildLiteHostModule();
    const { runLiteHostChat } = liteHost;
    const result = await runLiteHostChat({
      provider: "openrouter",
      model: "nvidia/nemotron-3-super-120b-a12b:free",
      providerApiKeys: {
        openrouter: "sk-or-v1-1234567890abcdefghijklmnop",
      },
      use_tools: false,
      research_mode: true,
      messages: [{ role: "user", content: "Summarize the verifier swarm." }],
    });

    expect(global.fetch).toHaveBeenCalledWith(
      "https://openrouter.ai/api/v1/chat/completions",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          Authorization: "Bearer sk-or-v1-1234567890abcdefghijklmnop",
        }),
      }),
    );
    expect(result.reply).toMatchObject({
      content: "OpenRouter reply",
      usage: {
        prompt_tokens: 12,
        completion_tokens: 4,
      },
    });
  });

  it("lets the main model call delegate_research and continue from the delegated result", async () => {
    const delegateResearchHandler = vi.fn(async (args, context) => JSON.stringify({
      delegated: true,
      runId: "research-run-1",
      query: args.query,
      heading: "Research Answer",
      outputMode: {
        id: "state_of_the_field",
        label: "State of the Field",
      },
      answer: "Delegated research answer with sources.",
      sources: [{ citationIndex: 1, title: "Quantum Paper", url: "https://example.com/quantum", tier: "core" }],
      scopeKey: context?.delegationContext?.scopeKey || "",
    }));

    global.fetch
      .mockResolvedValueOnce(createJsonResponse({
        choices: [
          {
            message: {
              content: "",
              tool_calls: [
                {
                  id: "tool-1",
                  function: {
                    name: "delegate_research",
                    arguments: JSON.stringify({
                      query: "How does quantum computing threaten modern encryption?",
                    }),
                  },
                },
              ],
            },
          },
        ],
      }))
      .mockResolvedValueOnce(createJsonResponse({
        choices: [
          {
            message: {
              content: "Final answer based on delegated research.",
            },
          },
        ],
      }));

    const { module: liteHost } = buildLiteHostModule({ delegateResearchHandler });
    const { runLiteHostChat } = liteHost;
    const result = await runLiteHostChat({
      provider: "openrouter",
      model: "nvidia/nemotron-3-super-120b-a12b:free",
      providerApiKeys: {
        openrouter: "sk-or-v1-1234567890abcdefghijklmnop",
      },
      use_tools: true,
      messages: [{ role: "user", content: "Do the full research and then answer." }],
      delegationContext: {
        scopeKey: "scope:test-chat",
        scope: "state_key",
      },
      searchProviderKeys: {
        tavily: "tvly-test-key-1234567890",
      },
      researchProvider: "openrouter",
      researchModel: "nvidia/nemotron-3-super-120b-a12b:free",
      researchProviderKeys: {
        openrouter: "sk-or-v1-1234567890abcdefghijklmnop",
      },
    });

    expect(delegateResearchHandler).toHaveBeenCalledWith(
      expect.objectContaining({
        query: "How does quantum computing threaten modern encryption?",
      }),
      expect.objectContaining({
        delegationContext: expect.objectContaining({
          scopeKey: "scope:test-chat",
        }),
        requestBody: expect.objectContaining({
          searchProviderKeys: {
            tavily: "tvly-test-key-1234567890",
          },
          researchProvider: "openrouter",
        }),
      }),
    );
    expect(result.reply.content).toBe("Final answer based on delegated research.");
    expect(result.reply.delegatedResearch).toMatchObject({
      runId: "research-run-1",
      outputMode: {
        id: "state_of_the_field",
        label: "State of the Field",
      },
      noGroundedSources: false,
      sourceCount: 1,
      sources: [
        expect.objectContaining({
          title: "Quantum Paper",
          url: "https://example.com/quantum",
          tier: "core",
        }),
      ],
    });
    expect(result.reply.toolsUsed).toEqual(expect.arrayContaining([
      expect.objectContaining({
        name: "delegate_research",
        runId: "research-run-1",
        sourceCount: 1,
        outputMode: "state_of_the_field",
      }),
    ]));
    expect(liteHost.createChatResponsePayload({ use_tools: true }, result.reply)).toMatchObject({
      tools_used: expect.arrayContaining([
        expect.objectContaining({
          name: "delegate_research",
          runId: "research-run-1",
          sourceCount: 1,
        }),
      ]),
      sources: [
        expect.objectContaining({
          title: "Quantum Paper",
          url: "https://example.com/quantum",
          tier: "core",
        }),
      ],
      researchMeta: {
        outputMode: {
          id: "state_of_the_field",
          label: "State of the Field",
        },
      },
    });
  });
});
