# nub-agent

`nub-agent` is a Vite + React chat app with a Vercel serverless backend for tool-using AI workflows.

The frontend exposes a branded assistant experience. The backend routes requests to Google Gemini, executes a small toolset, and returns normalized responses for chat, web reading, crawling, and scoped memory APIs. When configured, a Cerebras Qwen helper reranks API-key memory matches before the primary model answers.

## Features

- Branded assistant surface exposed as `nub-agent`
- React chat UI with conversation history, session memory, and live activity logs
- Terminal-style agent activity view with tool call tracing
- Sandboxed live preview for generated HTML UI snippets
- Image upload support with in-browser OCR via `tesseract.js`
- Tool-enabled `/api/chat` route with:
  - `calculate`
  - `web_search`
  - `web_fetch`
  - `search_images`
  - `view_image`
- Standalone reader and crawler endpoints:
  - `/api/memory`
  - `/api/read`
  - `/api/crawl`
- Regex validation script for catching invalid regex literals before deploy

## Stack

- Frontend: React 17, Vite
- Backend: Vercel Serverless Functions
- Model provider: Google Gemini (primary) with optional Cerebras Qwen memory reranking
- OCR: `tesseract.js`

## Repository Layout

```text
src/              React app and UI components
api/              Vercel serverless functions
api/tools/        Tool handlers used by /api/chat
lib/              Shared web/reader helpers
scripts/          Repo utilities such as regex validation
vercel.json       Vercel routing and build config
```

## Requirements

- Node.js 18+
- npm
- `GEMINI_API_KEY` or `GEMINI_API_KEYS` (single key or comma-separated list)
- `DATABASE_URL`
- Vercel CLI if you want to deploy from the terminal

## Local Development

Frontend only:

```bash
npm install
export GEMINI_API_KEY="your-key-here"
npm run dev
```

Full stack with Vercel routes:

```bash
npx -y vercel dev
```

Build for production:

```bash
npm run build
```

Validate regex literals across `api/`, `src/`, and `lib/`:

```bash
npm run lint:regex
```

## Environment

Required:

- `GEMINI_API_KEY` or `GEMINI_API_KEYS` - backend key(s) used by `api/chat.js`
- `DATABASE_URL` - MySQL/TiDB connection string used for app state and scoped memory

Optional:

- `CEREBRAS_API_KEY` - enables Cerebras-assisted API-key memory reranking
- `CEREBRAS_MEMORY_MODEL` - overrides the retrieval helper model (defaults to `qwen-3-235b-a22b-instruct-2507`)
- `SERPER_API_KEY` or `SERPER_API_KEYS` - enables Google-backed search for dork operators in `web_search`
- `PAGE_ACCESS_TOKEN` - Facebook Page access token used by `api/messenger.js`
- `VERIFY_TOKEN` or `MESSENGER_VERIFY_TOKEN` - token used by Facebook webhook verification
- `FB_GRAPH_API` - overrides the Graph API origin/version. Defaults to `https://graph.facebook.com/v21.0`
- `MESSENGER_SYSTEM_PROMPT` - custom system instruction for Messenger replies
- `MESSENGER_MAX_MESSAGE_CHARS` - max characters per outbound Messenger text chunk

Notes:

- The frontend is static; keep secrets on the server side only.
- Vercel reads environment variables from project settings in production.

## API

### `GET /api/chat`

Returns metadata for the chat endpoint, including the public model label and enabled tools.

### `POST /api/chat`

Primary chat endpoint. It expects a non-empty `messages` array.

For local API testing, use `vercel dev` and target the local Vercel server.

Example:

```bash
curl -sS -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      { "role": "user", "content": "What is the current date in UTC?" }
    ],
    "stream": false
  }'
```

Response shape:

```json
{
  "ok": true,
  "model": "nub-agent",
  "output_text": "...",
  "choices": [
    {
      "message": {
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "agentic": true,
  "tools_used": [
    {
      "name": "web_search",
      "args": "{\"query\":\"current date UTC\"}"
    }
  ]
}
```

Behavior notes:

- Tool execution is enabled by default.
- Streaming is supported for direct completions.
- Tool turns run in non-stream mode for deterministic tool handling.
- If you send `X-API-Key` or `Authorization: Bearer <key>`, the server stores memory rows for that key in the database and searches them for related context on later requests.
- For lower token usage, reuse the same memory key and send only the latest user turn; the backend searches the key's memory table and injects only the relevant matches.
- When `CEREBRAS_API_KEY` is configured, the backend uses a Qwen helper on Cerebras to rerank the candidate memory rows before injecting them.

### `GET /api/memory`

Returns metadata for the API-key memory endpoint.

### `POST /api/memory`

Supports `insert` and `search` actions for API-key-scoped memory.

Insert example:

```bash
curl -sS -X POST http://localhost:3000/api/memory \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-memory-key-123" \
  -d '{
    "action": "insert",
    "entries": [
      { "role": "user", "content": "My favorite database is TiDB." },
      { "role": "assistant", "content": "Understood. Favorite database is TiDB." }
    ]
  }'
```

Search example:

```bash
curl -sS -X POST http://localhost:3000/api/memory \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-memory-key-123" \
  -d '{
    "action": "search",
    "query": "what is my favorite database",
    "limit": 4,
    "include_context": true
  }'
```

### `GET /api/state`

Loads persisted app state for a memory scope.

Send one of:

- `X-API-Key: <your-memory-key>` to isolate memory by API key
- `Authorization: Bearer <your-memory-key>` as an alternative
- `X-State-Key: <generated-browser-key>` for anonymous browser scope

### `POST /api/state`

Saves persisted app state for the selected memory scope.

Example:

```bash
curl -sS -X POST http://localhost:3000/api/state \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-memory-key-123" \
  -d '{
    "state": {
      "version": 1,
      "conversations": [],
      "currentConversationId": "",
      "primaryModelId": "nub-agent",
      "fallbackModelIds": []
    }
  }'
```

### `POST /api/read`

Reads a single URL and returns cleaned content plus metadata.

Example:

```bash
curl -sS -X POST http://localhost:3000/api/read \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "mode": "article",
    "maxChars": 3200
  }'
```

### `POST /api/crawl`

Crawls a site breadth-first within bounded limits and returns extracted text for each visited page.

Example:

```bash
curl -sS -X POST http://localhost:3000/api/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/docs",
    "maxPages": 4,
    "maxDepth": 1,
    "sameOrigin": true
  }'
```

### `GET /api/messenger`

Facebook Messenger webhook verification endpoint. Facebook sends `hub.mode`,
`hub.verify_token`, and `hub.challenge`; the route returns the challenge when
the verify token matches `VERIFY_TOKEN` or `MESSENGER_VERIFY_TOKEN`.

### `POST /api/messenger`

Facebook Messenger webhook receiver. For incoming message and postback events,
the route calls the same LiteHost AI runtime used by `/api/chat` and sends the
reply back through the Facebook Graph API using `PAGE_ACCESS_TOKEN`.

Example local verification request:

```bash
curl -sS "http://localhost:3000/api/messenger?hub.mode=subscribe&hub.verify_token=litehost_verify_2024&hub.challenge=12345"
```

## Tooling Inside `/api/chat`

The chat route registers five server-side tools:

- `calculate` - evaluates simple math expressions
- `web_search` - runs web search queries and supports Google dork operators when Serper is configured
- `web_fetch` - fetches a URL and returns cleaned markdown or text
- `search_images` - returns image search results
- `view_image` - checks an image URL and returns basic metadata

## Frontend Notes

- The activity panel shows live dispatch/tool events in a terminal-style layout.
- When the model returns a fenced `html` block, the UI renders it in a sandboxed `iframe` preview.
- Uploaded images are OCR-processed client-side before being included in context.

## Security Notes

- `lib/web.js` blocks private and local network targets for reader/crawler fetches.
- Generated HTML previews are sandboxed with `allow-scripts` and isolated from the main app.
- Do not commit provider keys or deployment secrets.

## Deploying to Vercel

This repo is already configured for Vercel through `vercel.json`.

Deploy:

```bash
npx -y vercel --prod
```

Make sure `GEMINI_API_KEY` or `GEMINI_API_KEYS` is configured in the target Vercel project before deploying.

## Current Gaps

- No automated test suite yet
- No formal lint/format pipeline beyond regex validation
- Some tool providers are configured directly in code and should be reviewed before wider distribution
