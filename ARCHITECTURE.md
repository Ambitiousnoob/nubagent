# NubAgent Architecture

Current architecture as of March 31, 2026.

## Product Shape

- Frontend: documentation-only React site.
- Public backend: one serverless endpoint, `GET /api/chat` and `POST /api/chat`.
- Model provider: Gemini via `lib/gemini-chat.js`.
- Runtime mode: simple non-agentic chat completion.

## Request Flow

```text
Browser / API client
  -> Vercel rewrite
  -> api/chat.js
  -> lib/web.js (JSON parsing + body cap)
  -> lib/gemini-chat.js (request normalization + Gemini call)
  -> Gemini generateContent
  -> normalized JSON completion response
```

## Core Files

| File | Responsibility |
|------|----------------|
| `api/chat.js` | Public HTTP contract, method handling, CORS, error mapping |
| `lib/gemini-chat.js` | Chat normalization, Gemini request construction, response normalization |
| `lib/web.js` | Shared request parsing utilities, including request-body size limits |
| `vercel.json` | Single-route backend rewrite |
| `src/Docs.jsx` | Rendered docs site for the current API |
| `src/lib/docsContent.js` | Docs-site content for the API contract and operational notes |

## API Contract

### `GET /api/chat`

Returns metadata about the live endpoint:

```json
{
  "ok": true,
  "endpoint": "/api/chat",
  "mode": "simple-chatbot",
  "provider": "google-gemini",
  "model": "gemini-3-flash-preview",
  "streaming": false,
  "agentic": false
}
```

### `POST /api/chat`

Accepts:

- `message` as a string shortcut, or
- `messages` as an array of `{ role, content }`
- optional `system`
- optional `temperature`
- optional `max_tokens`
- optional `thinking_level`

Returns:

- `output_text`
- `choices[0].message.content`
- `usage`

The endpoint does not expose streaming, tool calls, search, memory, or delegated research behavior.

## Environment

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `GEMINI_API_KEY` | Yes | - | Server-side Gemini API key |
| `GEMINI_CHAT_MODEL` | No | `gemini-3-flash-preview` | Optional model override |
| `GEMINI_CHAT_THINKING_LEVEL` | No | `low` | Default thinking level |
| `CHAT_BODY_LIMIT_BYTES` | No | `65536` | Recommended request body cap |

## Operational Rules

- Keep `/api/chat` as the only public serverless route.
- Keep provider-specific logic in `lib/gemini-chat.js`.
- Keep docs changes in sync across `API.md`, `README.md`, `src/Docs.jsx`, and `src/lib/docsContent.js`.
- Verify with targeted API tests, `npm run lint`, and `npm run build` before pushing.
