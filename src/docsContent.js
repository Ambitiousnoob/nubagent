export const API_REFERENCE_SECTIONS = [
  {
    id: "chat-api",
    title: "Simple Chat API",
    paths: ["/api/chat"],
    methods: ["GET", "POST"],
    summary:
      "Single public endpoint that accepts plain chat messages and returns a Gemini Flash completion.",
    keyPoints: [
      "GET returns metadata about the live endpoint.",
      "POST accepts `message` or `messages` and returns a non-streaming JSON completion.",
      "The backend is intentionally non-agentic: no tools, no search, no delegated runtime.",
    ],
    requestShape: `{
  "system": "Be concise.",
  "messages": [
    { "role": "user", "content": "Say hello in one sentence." }
  ],
  "temperature": 0.7,
  "max_tokens": 256,
  "thinking_level": "low"
}`,
    responseShape: `{
  "ok": true,
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "gemini-3-flash-preview",
  "provider": "google-gemini",
  "output_text": "Hello. How can I help?",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello. How can I help?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 9,
    "total_tokens": 21
  }
}`,
    implementationFiles: ["api/chat.js", "lib/gemini-chat.js", "lib/web.js"],
  },
];

export const API_CREATION_STEPS = [
  {
    title: "Keep the public boundary narrow",
    body: "Only `/api/chat` is public. New externally visible behavior should fit there unless you are intentionally expanding the API surface.",
    bullets: [
      "Prefer extending `api/chat.js` over adding a new endpoint.",
      "Keep provider-specific logic in `lib/gemini-chat.js`.",
      "Keep request parsing and shared guards in `lib/web.js`.",
    ],
  },
  {
    title: "Treat the request contract as text-only chat",
    body: "The endpoint is a simple chatbot, not an orchestration engine.",
    bullets: [
      "Accept `message` or `messages` with `system`, `user`, and `assistant` roles.",
      "Reject unsupported modes such as streaming instead of silently emulating them.",
      "Return a stable JSON shape with `choices`, `output_text`, and `usage`.",
    ],
  },
  {
    title: "Keep deployment wiring obvious",
    body: "The runtime should remain easy to inspect from `vercel.json` down to the Gemini adapter.",
    bullets: [
      "Only `/api/chat` should rewrite into the serverless backend.",
      "Document environment variables in `.env.example` and `README.md` together.",
      "When the contract changes, update the rendered docs and the README in the same commit.",
    ],
  },
  {
    title: "Verify the change with focused checks",
    body: "Most regressions here show up in request normalization, provider mapping, or docs drift.",
    bullets: [
      "Run `npm run lint` after touching API or docs-site code.",
      "Run `npm run build` before shipping docs-site changes.",
      "Exercise `GET /api/chat` and `POST /api/chat` manually when the contract changes.",
    ],
  },
];

export const API_CREATION_CHECKLIST = [
  "Confirm the change belongs inside `/api/chat`.",
  "Keep GET metadata accurate.",
  "Preserve the non-streaming JSON response contract.",
  "Update `README.md` and rendered docs together.",
  "Keep `vercel.json` aligned with the single-route backend.",
  "Run lint, build, and a manual API check before pushing.",
];

export const BASIC_ENDPOINT_EXAMPLE = `const { readBody } = require("../lib/web");
const {
  metadataPayload,
  normalizeChatBody,
  runGeminiChat,
  createChatResponsePayload,
} = require("../lib/gemini-chat");

module.exports = async (req, res) => {
  if (req.method === "GET") {
    return writeJson(res, 200, metadataPayload());
  }

  if (req.method !== "POST") {
    return writeJson(res, 405, { error: "Method not allowed" });
  }

  const body = await readBody(req, { maxBytes: 64 * 1024 });
  const normalized = normalizeChatBody(body);
  const result = await runGeminiChat(normalized);
  return writeJson(res, 200, createChatResponsePayload(result));
};`;

export const ALIAS_ENDPOINT_EXAMPLE = `// vercel.json
{
  "rewrites": [
    { "source": "/api/chat", "destination": "/api/chat.js" },
    { "source": "/(.*)", "destination": "/" }
  ]
}

// Keep provider details behind the single public route.
module.exports = async (req, res) => {
  if (req.method === "GET") {
    return writeJson(res, 200, metadataPayload());
  }

  const body = await readBody(req, { maxBytes: 64 * 1024 });
  const result = await runGeminiChat(normalizeChatBody(body));
  return writeJson(res, 200, createChatResponsePayload(result));
};`;

export const OPERATIONS_SECTIONS = [
  {
    title: "Runtime behavior",
    body: "The endpoint is designed to be predictable for API consumers.",
    bullets: [
      "JSON only: GET metadata and POST chat completions.",
      "Non-streaming responses by design.",
      "Body parsing is capped to keep accidental or hostile payloads in check.",
    ],
  },
  {
    title: "Gemini adapter",
    body: "The provider mapping stays in one place so request normalization and provider behavior do not drift apart.",
    bullets: [
      "`lib/gemini-chat.js` maps chat messages to Gemini `generateContent` payloads.",
      "Assistant turns are sent as Gemini `model` role messages.",
      "Usage metadata is normalized back into a simple completion response.",
    ],
  },
  {
    title: "Docs-only frontend",
    body: "The shipped frontend is a documentation site, not a live chat client.",
    bullets: [
      "The docs explain the single public route and its ownership.",
      "Legacy app routes are still canonicalized back to `/`.",
      "Docs and API contract changes should ship together.",
    ],
  },
];

export const ENVIRONMENT_VARIABLES = [
  {
    name: "GEMINI_API_KEY",
    required: "Yes",
    defaultValue: "None",
    description: "Server-side Gemini API key required for `/api/chat`.",
  },
  {
    name: "GEMINI_CHAT_MODEL",
    required: "No",
    defaultValue: "gemini-3-flash-preview",
    description: "Optional model override for the chat endpoint.",
  },
  {
    name: "GEMINI_CHAT_THINKING_LEVEL",
    required: "No",
    defaultValue: "low",
    description: "Default thinking level used when the request does not provide one.",
  },
  {
    name: "CHAT_BODY_LIMIT_BYTES",
    required: "No",
    defaultValue: "65536",
    description: "Recommended request-body cap for `/api/chat` payloads.",
  },
];

export const MIGRATION_NOTES = [
  {
    title: "Single public route",
    body: "The public backend surface has been collapsed to `/api/chat`. Former top-level API routes are gone.",
  },
  {
    title: "No agentic runtime",
    body: "The endpoint no longer performs search, tool execution, delegation, or memory orchestration before answering.",
  },
  {
    title: "Docs first",
    body: "The frontend remains a docs-only shell so the product story stays aligned with the current API shape.",
  },
];
