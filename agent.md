# Agent Memory

This file is a working memory dump for the current `nubagent` repo state.
It intentionally omits secrets and raw API keys.

## Identity

- Product name: `nub-agent`
- Public developer name: `Ambitiousnoob`
- Primary backend provider: Google Gemini
- Default runtime model: `gemini-2.5-flash-lite`
- Model fallback chain in `lib/litehost-chat.js`:
  - `gemini-2.5-flash-lite`
  - `gemini-2.5-flash`
  - `gemini-3-flash-preview`

## Current Repo State

- Repo root: this directory
- Branch: `master`
- Current HEAD when this file was written: `19d4ac0`
- Active frontend entrypoint:
  - `src/main.jsx` -> `src/App.jsx` -> `src/SearchEngine.jsx`
- Legacy UI still exists but is not the active entrypoint:
  - `src/ai.jsx`

## Product Shape Right Now

The active app is no longer a generic chat-first UI. It is a search-first research interface.

- Main UI file: `src/SearchEngine.jsx`
- Styling is embedded inside that file
- Search UX includes:
  - landing search box
  - upload support
  - research planning card
  - searching status card
  - Liberty-style result card
  - Liberty-style result tables

## Latest SearchEngine Changes

Recent committed UI state includes:

- search-first interface is active
- upload flow is wired
- user label is `You`, not a hardcoded personal name
- final answer cards use Liberty-style presentation
- result tables now render:
  - `Key Sources & Findings`
  - `Scope & Coverage`

The result tables are built from actual run metadata, not static placeholders.

## Search Pipeline

The active research flow in `src/SearchEngine.jsx` works like this:

1. Build query variants
2. Call `/api/search`
3. Rank and dedupe sources
4. Call `/api/fetch` on a bounded subset
5. Run synthesis workers through `/api/chat`
6. Merge to one cited final answer

Current search tuning constants in `src/SearchEngine.jsx`:

- `SOURCE_TARGET = 60`
- `FETCH_TARGET = 24`
- `SEARCH_SWARM_SIZE = 3`
- `SYNTHESIS_SWARM_SIZE = 3`
- `FETCH_CONCURRENCY = 4`
- `FETCH_MAX_CHARS = 900`

This is still parallelized research logic, but reduced from the earlier heavier swarm behavior.

## API Surface

Top-level Vercel functions currently deployed from `api/*.js`:

- `api/chat.js`
- `api/crawl.js`
- `api/fetch.js`
- `api/memory.js`
- `api/messenger.js`
- `api/read.js`
- `api/search.js`
- `api/state.js`

Internal tool modules remain under `api/tools/*.js`, but `vercel.json` now uses `api/*.js` for builds so tool modules are not counted as separate Vercel functions.

## Tooling

Current chat/tool layer includes:

- `calculate`
- `web_search`
- `web_fetch`
- `search_images`
- `view_image`

### web_search

`api/tools/web_search.js` merges multiple providers depending on query shape:

- DuckDuckGo HTML
- Tavily
- Serper
- Jina Search

Routing behavior:

- Google-only operators (`intitle:`, `inurl:`, `intext:`, `before:`, `after:`, non-pdf `filetype:`) prefer Serper
- `site:` / `filetype:` use Serper + DuckDuckGo
- plain queries use broader merged search

Key handling notes:

- Gemini keys support comma-separated rotation
- Tavily keys support comma-separated rotation
- Serper keys support comma-separated rotation
- Jina keys support comma-separated rotation

Important debt:

- `api/tools/web_search.js` still contains a hardcoded default Tavily fallback constant.
- If strict secret hygiene matters, remove that fallback and rely only on env vars.

### web_fetch

`api/tools/web_fetch.js` uses Jina Reader:

- endpoint pattern: `https://r.jina.ai/<url>`
- supports `markdown` or `text`
- retries
- timeout
- truncation metadata comment block

## Persistence and Memory

Two DB-backed persistence paths exist:

### 1. App state

- Files:
  - `api/state.js`
  - `lib/db.js`
  - `lib/state-scope.js`
- Stores scoped frontend state such as conversations and selected model state

### 2. API-key memory

- Files:
  - `api/memory.js`
  - `lib/api-key-memory.js`
  - `lib/chat-memory.js`
- Uses `X-API-Key` or `Authorization: Bearer <key>`
- Same key -> same memory namespace
- Different keys -> isolated memories
- Backend can search relevant memory rows and inject only matching context
- Optional Cerebras reranking exists for memory retrieval when `CEREBRAS_API_KEY` is configured

## Messenger

Messenger webhook support still exists:

- file: `api/messenger.js`
- uses Facebook webhook verification + send API env vars
- branding should align to `nub-agent`

## Important Backend Fixes Already Landed

These are part of current history:

- `13bfa03` — fixed malformed Jina Reader fallback URL
- `5c32b44` — normalized `/api/chat` memory inputs
- `7c9c646` — fixed Vercel Hobby function-count issue by narrowing `vercel.json`
- `19d4ac0` — added Liberty-style result tables

## Build / Deploy Notes

- Build command: `npm run build`
- Regex validation: `npm run lint:regex`
- Current `vercel.json` still uses the legacy `builds` key
- Because of that, Vercel warns that Project Settings build config does not apply

Last known direct production alias:

- `https://src-litehost.vercel.app`

Note:

- The table UI commit `19d4ac0` was pushed to GitHub.
- If production has not been redeployed after that commit, GitHub may be ahead of the live site.

## Environment Variables

Required or effectively required:

- `GEMINI_API_KEY` or `GEMINI_API_KEYS`
- `DATABASE_URL`

Optional:

- `CEREBRAS_API_KEY`
- `CEREBRAS_MEMORY_MODEL`
- `SERPER_API_KEY` or `SERPER_API_KEYS`
- `JINA_API_KEY` or `JINA_API_KEYS`
- `TAVILY_API_KEY` or `TAVILY_API_KEYS`
- Messenger envs:
  - `PAGE_ACCESS_TOKEN`
  - `VERIFY_TOKEN` or `MESSENGER_VERIFY_TOKEN`
  - `FB_GRAPH_API`
  - `MESSENGER_SYSTEM_PROMPT`
  - `MESSENGER_MAX_MESSAGE_CHARS`

## Known Technical Debt

- React is still `17`
- Vite is still `4`
- `src/SearchEngine.jsx` is doing a lot in one file: UI, orchestration, upload handling, answer rendering, and styling
- `src/ai.jsx` still exists and contains a lot of older logic that is no longer the active entrypoint
- `vercel.json` still uses `builds`, which causes the Vercel warning
- Tavily fallback secret handling should be cleaned up

## Practical Next Steps

If resuming work later, the highest-value cleanup items are:

1. Split `src/SearchEngine.jsx` into smaller components
2. Remove hardcoded Tavily fallback usage
3. Decide whether `src/ai.jsx` should be kept or retired
4. Replace legacy `vercel.json` `builds` usage with the current Vercel config style
5. Redeploy production after UI-only GitHub pushes

