# nub-agent API Documentation

Current public API surface for nub-agent.

## Base URLs

| Environment | URL |
|-------------|-----|
| Production | `https://your-domain.vercel.app/api` |
| Local (Vercel Dev) | `http://localhost:3000/api` |

## Public Endpoint

Only one public serverless endpoint remains:

- `GET /api/chat`
- `POST /api/chat`

All former public routes such as `/api/search`, `/api/web`, `/api/memory`, `/api/state`, `/api/content`, `/api/research`, and `/api/utils` have been removed.

## `GET /api/chat`

Returns metadata about the simple chat runtime.

**Response**

```json
{
  "ok": true,
  "endpoint": "/api/chat",
  "provider": "google-gemini",
  "model": "gemini-3-flash-preview",
  "mode": "simple-chatbot",
  "streaming": false,
  "agentic": false
}
```

## `POST /api/chat`

Primary completion endpoint. Accepts chat messages and forwards them to Gemini Flash.

**Request body**

```typescript
{
  messages: Array<{
    role: "system" | "user" | "assistant";
    content: string | Array<{ type: "text"; text: string }>;
  }>;
  message?: string;
  system?: string;
  max_tokens?: number;
  temperature?: number;
  thinking_level?: "minimal" | "low" | "medium" | "high";
}
```

**Example**

```bash
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      { "role": "user", "content": "Say hello in one sentence." }
    ],
    "system": "Be concise."
  }'
```

**Response**

```json
{
  "ok": true,
  "model": "gemini-3-flash-preview",
  "provider": "google-gemini",
  "output_text": "Hello. How can I help?",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Hello. How can I help?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 9,
    "total_tokens": 21
  }
}
```

## Notes

- The endpoint is non-streaming and non-agentic.
- `GEMINI_API_KEY` must be configured on the server.
- `GEMINI_CHAT_MODEL` and `GEMINI_CHAT_THINKING_LEVEL` can override the default model and thinking level.
