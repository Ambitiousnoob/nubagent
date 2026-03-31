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

Returns metadata about the chat runtime, including model/provider details and the tool surface available to the assistant.

**Response**

```json
{
  "model": "nub-agent",
  "provider": "google-gemini",
  "tools": ["calculate", "web_search", "web_fetch", "search_images", "view_image"],
  "streaming": true,
  "research_mode": true
}
```

## `POST /api/chat`

Primary completion endpoint. Accepts messages, optional streaming, and chat-runtime options.

**Request body**

```typescript
{
  messages: Array<{
    role: "system" | "user" | "assistant";
    content: string | Array<{
      type: "text" | "image_url";
      text?: string;
      image_url?: { url: string };
    }>;
  }>;
  model?: string;
  stream?: boolean;
  max_tokens?: number;
  temperature?: number;
  research_mode?: boolean;
  use_tools?: boolean;
  save_persistent_memory?: boolean;
}
```

**Example**

```bash
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      { "role": "user", "content": "What changed in the latest Vercel docs?" }
    ],
    "stream": false
  }'
```

**Response**

```json
{
  "ok": true,
  "model": "nub-agent",
  "output_text": "The recent changes include ...",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The recent changes include ..."
    },
    "finish_reason": "stop"
  }],
  "agentic": true,
  "tools_used": [],
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 128,
    "total_tokens": 173
  }
}
```

**Streaming response**

```text
data: {"choices":[{"index":0,"delta":{"content":"The"}}]}
data: {"choices":[{"index":0,"delta":{"content":" latest"}}]}
data: [DONE]
```

## Notes

- Scoped memory still uses `X-API-Key`, `Authorization: Bearer <key>`, or `X-State-Key` when supported by the chat runtime.
- Tool execution remains internal to `/api/chat`; search/fetch helpers are no longer exposed as top-level public endpoints.
