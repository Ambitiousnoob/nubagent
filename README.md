# gemini-3-flash

`gemini-3-flash` is a Vite + React chat app with a Vercel serverless backend for tool-using AI workflows.

The frontend exposes a branded assistant experience. The backend routes requests to Google Gemini, executes a small toolset, and returns normalized responses for chat, web reading, and crawling.

## Features

- Branded assistant surface exposed as `gemini-3-flash`
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
  - `/api/read`
  - `/api/crawl`
- Regex validation script for catching invalid regex literals before deploy

## Stack

- Frontend: React 17, Vite
- Backend: Vercel Serverless Functions
- Model provider: Google Gemini
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
- `GEMINI_API_KEYS` (comma-separated list)
- Vercel CLI if you want to deploy from the terminal

## Local Development

Frontend only:

```bash
npm install
export GEMINI_API_KEYS="your-keys-here"
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

- `GEMINI_API_KEYS` - backend keys used by `api/chat.js`

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
  "model": "gemini-3-flash",
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

## Tooling Inside `/api/chat`

The chat route registers five server-side tools:

- `calculate` - evaluates simple math expressions
- `web_search` - runs web search queries and returns result summaries
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

Make sure `GEMINI_API_KEYS` is configured in the target Vercel project before deploying.

## Current Gaps

- No automated test suite yet
- No formal lint/format pipeline beyond regex validation
- Some tool providers are configured directly in code and should be reviewed before wider distribution
