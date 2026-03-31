# NubAgent Quick Reference

## Public Endpoint

| Route | Methods | Purpose |
|-------|---------|---------|
| `/api/chat` | `GET`, `POST` | Simple Gemini-backed chat endpoint |

## Request Shape

```json
{
  "system": "Be concise.",
  "messages": [
    { "role": "user", "content": "Say hello." }
  ],
  "temperature": 0.7,
  "max_tokens": 256,
  "thinking_level": "low"
}
```

## Response Shape

```json
{
  "ok": true,
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
}
```

## Environment Variables

| Variable | Required | Default |
|----------|----------|---------|
| `GEMINI_API_KEY` | Yes | - |
| `GEMINI_CHAT_MODEL` | No | `gemini-3-flash-preview` |
| `GEMINI_CHAT_THINKING_LEVEL` | No | `low` |
| `CHAT_BODY_LIMIT_BYTES` | No | `65536` |

## Important Files

| File | Purpose |
|------|---------|
| `api/chat.js` | Public API handler |
| `lib/gemini-chat.js` | Gemini request/response adapter |
| `lib/web.js` | Shared body parsing and size guards |
| `API.md` | API contract |
| `src/Docs.jsx` | Rendered docs page |

## Verification Commands

```bash
npx vitest run tests/api/chat.test.js
npm run lint
npm run build
```
