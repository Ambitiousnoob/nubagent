import { afterEach, describe, expect, it, vi } from "vitest";
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

const CHAT_HANDLER_PATH = "/root/.bot/.downloads/nubagent/api/chat.js";

afterEach(() => {
  delete process.env.GEMINI_API_KEY;
  delete process.env.GEMINI_CHAT_MODEL;
  delete process.env.GEMINI_CHAT_THINKING_LEVEL;
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("/api/chat", () => {
  it("returns chat metadata on GET", async () => {
    const handler = loadCommonJsModule(CHAT_HANDLER_PATH);
    const req = {
      method: "GET",
      headers: {},
      url: "/api/chat",
    };
    const res = createRes();

    await handler(req, res);

    expect(res.statusCode).toBe(200);
    expect(JSON.parse(res.body)).toMatchObject({
      ok: true,
      endpoint: "/api/chat",
      provider: "google-gemini",
      mode: "simple-chatbot",
      streaming: false,
      agentic: false,
    });
  });

  it("maps plain chat messages to Gemini and returns a simple completion payload", async () => {
    process.env.GEMINI_API_KEY = "gemini-test-key";
    const fetchMock = vi.fn(async (_url, options = {}) => ({
      ok: true,
      status: 200,
      json: async () => ({
        candidates: [
          {
            content: {
              parts: [{ text: "Simple Gemini reply" }],
            },
            finishReason: "STOP",
          },
        ],
        usageMetadata: {
          promptTokenCount: 11,
          candidatesTokenCount: 7,
          totalTokenCount: 18,
        },
      }),
    }));
    vi.stubGlobal("fetch", fetchMock);

    const handler = loadCommonJsModule(CHAT_HANDLER_PATH);
    const req = {
      method: "POST",
      headers: {},
      url: "/api/chat",
      body: {
        system: "Be concise.",
        messages: [
          { role: "user", content: "Say hello." },
          { role: "assistant", content: "Previous answer." },
          { role: "user", content: [{ type: "text", text: "Try again." }] },
        ],
      },
    };
    const res = createRes();

    await handler(req, res);

    expect(res.statusCode).toBe(200);
    expect(fetchMock).toHaveBeenCalledWith(
      "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({
          "Content-Type": "application/json",
          "x-goog-api-key": "gemini-test-key",
        }),
      }),
    );

    const geminiPayload = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(geminiPayload).toEqual({
      contents: [
        { role: "user", parts: [{ text: "Say hello." }] },
        { role: "model", parts: [{ text: "Previous answer." }] },
        { role: "user", parts: [{ text: "Try again." }] },
      ],
      systemInstruction: {
        parts: [{ text: "Be concise." }],
      },
      generationConfig: {
        thinkingConfig: {
          thinkingLevel: "low",
        },
      },
    });

    expect(JSON.parse(res.body)).toMatchObject({
      ok: true,
      model: "gemini-3-flash-preview",
      provider: "google-gemini",
      output_text: "Simple Gemini reply",
      choices: [
        {
          message: {
            role: "assistant",
            content: "Simple Gemini reply",
          },
          finish_reason: "stop",
        },
      ],
      usage: {
        prompt_tokens: 11,
        completion_tokens: 7,
        total_tokens: 18,
      },
    });
  });

  it("rejects unsupported streaming requests", async () => {
    process.env.GEMINI_API_KEY = "gemini-test-key";
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const handler = loadCommonJsModule(CHAT_HANDLER_PATH);
    const req = {
      method: "POST",
      headers: {},
      url: "/api/chat",
      body: {
        messages: [{ role: "user", content: "Hello" }],
        stream: true,
      },
    };
    const res = createRes();

    await handler(req, res);

    expect(res.statusCode).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(JSON.parse(res.body)).toEqual({
      error: "Streaming is not supported by this simple chat endpoint.",
    });
  });
});
