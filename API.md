# nub-agent API Documentation

Complete API reference for nub-agent, a tool-using AI research assistant powered by Google Gemini.

## Base URLs

| Environment | URL |
|-------------|-----|
| Production | `https://your-domain.vercel.app/api` |
| Local (Vercel Dev) | `http://localhost:3000/api` |

## Authentication

Most endpoints are public. For scoped memory features, include one of:

- `X-API-Key: <your-key>` - API key for memory scoping
- `Authorization: Bearer <your-key>` - Alternative header format
- `X-State-Key: <key>` - Anonymous browser state scoping

---

## Endpoints

### `GET /api/chat`

Returns metadata about the chat endpoint including model info and available tools.

**Response:**
```json
{
  "model": "nub-agent",
  "provider": "google-gemini",
  "tools": ["calculate", "web_search", "web_fetch", "search_images", "view_image"],
  "streaming": true,
  "research_mode": true
}
```

---

### `POST /api/chat`

Primary chat endpoint. Processes messages with tool execution and returns AI responses.

**Request Body:**
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
  model?: string;              // Default: "nub-agent"
  stream?: boolean;            // Enable SSE streaming
  max_tokens?: number;         // Completion token limit
  temperature?: number;        // 0-2, default: 0.7
  research_mode?: boolean;     // Enable research synthesis mode
  use_tools?: boolean;         // Enable tool execution (default: true)
  save_persistent_memory?: boolean;  // Save to scoped memory
}
```

**Example:**
```bash
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is quantum computing?"}
    ],
    "stream": false
  }'
```

**Response (non-streaming):**
```json
{
  "ok": true,
  "model": "nub-agent",
  "output_text": "Quantum computing is a type of computing...",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Quantum computing is a type of computing..."
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

**Response (streaming):**
```
data: {"choices":[{"index":0,"delta":{"content":"Quantum"}}]}
data: {"choices":[{"index":0,"delta":{"content":" computing"}}]}
data: [DONE]
```

**Research Mode:**
When `research_mode: true`, the response is optimized for research synthesis:
- No self-introduction or identity statements
- Structured output with citations
- Optimized for RRL (Review of Related Literature) queries

---

### `GET /api/web`

Web research endpoint metadata.

**Response:**
```json
{
  "ok": true,
  "endpoint": "/api/web",
  "methods": ["GET", "POST"],
  "description": "Combined web research endpoint with search, fetch, and RAG ranking",
  "features": [
    "Multi-backend search (DuckDuckGo, Tavily, Serper, Jina, Brave)",
    "Content extraction with Jina/Firecrawl fallbacks",
    "RAG re-ranking for query-focused results",
    "Evidence block generation for synthesis"
  ]
}
```

---

### `POST /api/web`

Combined web research endpoint. Performs search, content fetching, and RAG ranking in a single request.

**Request Body:**
```typescript
{
  query?: string;           // Search query (optional if URLs provided)
  urls?: string[];          // Specific URLs to fetch (optional)
  maxResults?: number;      // Max search results (default: 20, max: 30)
  fetchContent?: boolean;   // Fetch full content of search results
  rag?: boolean;            // Enable RAG re-ranking and evidence blocks
}
```

**Example - Search only:**
```bash
curl -X POST http://localhost:3000/api/web \
  -H "Content-Type: application/json" \
  -d '{
    "query": "quantum computing breakthroughs 2025",
    "maxResults": 15
  }'
```

**Example - Search with content fetching:**
```bash
curl -X POST http://localhost:3000/api/web \
  -H "Content-Type: application/json" \
  -d '{
    "query": "RAG systems evaluation",
    "fetchContent": true,
    "rag": true
  }'
```

**Example - Fetch specific URLs:**
```bash
curl -X POST http://localhost:3000/api/web \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://arxiv.org/abs/2401.12345",
      "https://example.com/research"
    ]
  }'
```

**Response:**
```json
{
  "ok": true,
  "query": "quantum computing breakthroughs 2025",
  "sources": [
    {
      "rank": 1,
      "title": "Quantum Computing Breakthrough...",
      "url": "https://example.com/quantum",
      "description": "Researchers achieve...",
      "source": "tavily",
      "citationIndex": 1
    }
  ],
  "fetched": [
    {
      "source": {...},
      "content": "# Article Title\n\nFull markdown content..."
    }
  ],
  "evidence": [
    "[1] Example Domain\nURL: https://example.com\nFetched excerpt (query-focused): ..."
  ],
  "urlContent": [
    {
      "url": "https://arxiv.org/abs/2401.12345",
      "content": "# Paper Title\n\n...",
      "success": true
    }
  ]
}
```

**Features:**
- **Multi-backend search**: Queries DuckDuckGo, Tavily, Serper, Jina, and Brave simultaneously
- **RAG re-ranking**: Re-ranks sources by query relevance and domain authority
- **Evidence blocks**: Generates structured evidence for synthesis
- **Concurrent fetching**: Fetches up to 12 pages with 4 concurrent workers

---

### `GET /api/search`

Search endpoint metadata.

**Response:**
```json
{
  "endpoint": "/api/search",
  "method": "POST",
  "description": "Web search with query expansion and source ranking"
}
```

---

### `POST /api/search`

Performs web search with optional dork operators and returns ranked results.

**Request Body:**
```typescript
{
  query: string;        // Search query (supports dork operators)
  limit?: number;       // Max results (default: 20)
  fresh?: boolean;      // Prefer recent results
}
```

**Example:**
```bash
curl -X POST http://localhost:3000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "site:arxiv.org retrieval augmented generation after:2024-01-01",
    "limit": 10
  }'
```

**Response:**
```json
{
  "ok": true,
  "results": [
    {
      "title": "A Survey of RAG Systems",
      "url": "https://arxiv.org/abs/2401.12345",
      "description": "This paper surveys retrieval-augmented generation...",
      "snippet": "RAG systems combine retrieval with generation...",
      "date": "2024-01-15"
    }
  ],
  "query": "site:arxiv.org retrieval augmented generation after:2024-01-01",
  "total": 10
}
```

**Supported Dork Operators:**
- `site:` - Restrict to specific domain
- `filetype:` - Restrict to file type
- `intitle:` - Search in title only
- `inurl:` - Search in URL only
- `after:` / `before:` - Date range filtering

---

### `GET /api/fetch`

Fetch endpoint metadata.

---

### `POST /api/fetch`

Fetches and extracts content from a URL using Jina AI reader.

**Request Body:**
```typescript
{
  url: string;           // Target URL (must be public)
  format?: "text" | "markdown" | "html";  // Output format
  max_chars?: number;    // Max characters to extract
}
```

**Example:**
```bash
curl -X POST http://localhost:3000/api/fetch \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/article",
    "format": "markdown",
    "max_chars": 5000
  }'
```

**Response:**
```json
{
  "ok": true,
  "url": "https://example.com/article",
  "title": "Example Article Title",
  "content": "# Example Article\n\nThis is the extracted content...",
  "publishedTime": "2024-01-15T10:30:00Z",
  "via": "reader-proxy"
}
```

**Security:**
- Private IPs and local networks are blocked
- Timeout: 12 seconds
- Redirects: Max 5

---

### `GET /api/memory`

Memory endpoint metadata.

**Response:**
```json
{
  "endpoint": "/api/memory",
  "methods": ["POST"],
  "actions": ["insert", "search"],
  "auth": "X-API-Key or Authorization header required"
}
```

---

### `POST /api/memory`

Manages API-key-scoped persistent memory.

**Request Body (Insert):**
```typescript
{
  action: "insert";
  entries: Array<{
    role: "user" | "assistant";
    content: string;
  }>;
}
```

**Request Body (Search):**
```typescript
{
  action: "search";
  query: string;
  limit?: number;         // Max results (default: 4)
  include_context?: boolean;  // Include formatted context
}
```

**Insert Example:**
```bash
curl -X POST http://localhost:3000/api/memory \
  -H "Content-Type: application/json" \
  -H "X-API-Key: my-key-123" \
  -d '{
    "action": "insert",
    "entries": [
      {"role": "user", "content": "I prefer TypeScript over JavaScript"},
      {"role": "assistant", "content": "Noted: you prefer TypeScript."}
    ]
  }'
```

**Search Example:**
```bash
curl -X POST http://localhost:3000/api/memory \
  -H "Content-Type: application/json" \
  -H "X-API-Key: my-key-123" \
  -d '{
    "action": "search",
    "query": "programming language preference",
    "limit": 4,
    "include_context": true
  }'
```

**Search Response:**
```json
{
  "ok": true,
  "results": [
    {
      "content": "I prefer TypeScript over JavaScript",
      "role": "user",
      "score": 0.92,
      "created_at": "2024-01-15T10:00:00Z"
    }
  ],
  "context": "User context:\n- I prefer TypeScript over JavaScript",
  "meta": {
    "strategy": "semantic",
    "provider": "cerebras",
    "model": "qwen-3-235b-a22b-instruct-2507"
  }
}
```

---

### `GET /api/state`

Loads persisted application state for a memory scope.

**Headers:**
- `X-API-Key: <key>` - API-key scoped state
- `X-State-Key: <key>` - Anonymous browser state

**Response:**
```json
{
  "ok": true,
  "state": {
    "version": 1,
    "conversations": [],
    "currentConversationId": "",
    "primaryModelId": "nub-agent",
    "fallbackModelIds": []
  }
}
```

---

### `POST /api/state`

Saves application state for a memory scope.

**Request Body:**
```typescript
{
  state: {
    version: number;
    conversations: Array<any>;
    currentConversationId: string;
    primaryModelId: string;
    fallbackModelIds: string[];
  };
}
```

---

### `GET /api/read`

Reader endpoint metadata.

---

### `POST /api/read`

Reads a single URL and returns cleaned content with metadata.

**Request Body:**
```typescript
{
  url: string;
  mode?: "article" | "full" | "text";  // Extraction mode
  maxChars?: number;                    // Max characters
}
```

**Example:**
```bash
curl -X POST http://localhost:3000/api/read \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/docs",
    "mode": "article",
    "maxChars": 3200
  }'
```

---

### `GET /api/crawl`

Crawler endpoint metadata.

---

### `POST /api/crawl`

Crawls a site breadth-first within bounded limits.

**Request Body:**
```typescript
{
  url: string;
  maxPages?: number;      // Max pages to crawl (default: 10)
  maxDepth?: number;      // Max link depth (default: 2)
  sameOrigin?: boolean;   // Stay on same domain (default: true)
  patterns?: string[];    // URL patterns to include
  excludePatterns?: string[];  // URL patterns to exclude
}
```

**Example:**
```bash
curl -X POST http://localhost:3000/api/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/docs",
    "maxPages": 5,
    "maxDepth": 1,
    "sameOrigin": true
  }'
```

**Response:**
```json
{
  "ok": true,
  "pages": [
    {
      "url": "https://example.com/docs",
      "title": "Documentation",
      "content": "# Documentation\n\nWelcome to the docs...",
      "extractedAt": "2024-01-15T10:00:00Z"
    }
  ],
  "total": 5,
  "root": "https://example.com/docs"
}
```

---

### `GET /api/messenger`

Facebook Messenger webhook verification.

**Query Parameters:**
- `hub.mode` - Should be "subscribe"
- `hub.verify_token` - Verification token
- `hub.challenge` - Challenge string to return

**Example:**
```bash
curl "http://localhost:3000/api/messenger?hub.mode=subscribe&hub.verify_token=my-token&hub.challenge=12345"
```

**Response:** `12345` (the challenge value)

---

### `POST /api/messenger`

Facebook Messenger webhook receiver for incoming messages.

**Headers:**
- `X-Hub-Signature-256` - Facebook signature verification

**Request Body:** Facebook Messenger event payload

**Behavior:**
- Processes incoming messages and postbacks
- Calls LiteHost AI runtime for responses
- Sends replies via Facebook Graph API

---

## Tools

Tools are automatically invoked by the AI when appropriate.

### `calculate`

Evaluates mathematical expressions.

**Usage:** Automatic when math expressions detected

---

### `web_search`

Performs web searches with multiple backends and automatic routing based on query operators.

**Backends:**
| Backend | API Key Required | Features |
|---------|------------------|----------|
| DuckDuckGo | No | Default, privacy-focused |
| Tavily | `TAVILY_API_KEY` | AI-ready results with answer |
| Serper (Google) | `SERPER_API_KEY` | Full dork operator support |
| Jina | `JINA_API_KEY` | Search enrichment |
| Brave | `BRAVE_API_KEY` | Privacy-focused alternative |

**Automatic Routing:**
- Queries with Google dork operators (`site:`, `filetype:`, `intitle:`, etc.) automatically route to Serper when configured
- Standard queries use all available backends with result merging

**Supported Dork Operators:**
- `site:domain.com` - Restrict to domain
- `filetype:pdf` - Restrict to file type
- `intitle:keyword` - Search in title
- `inurl:keyword` - Search in URL
- `intext:keyword` - Search in body text
- `after:YYYY-MM-DD` / `before:YYYY-MM-DD` - Date range
- `"exact phrase"` - Exact match
- `-exclude` - Exclude terms
- `OR` - Logical OR

**Example with operators:**
```bash
curl -X POST http://localhost:3000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "site:arxiv.org \"transformer\" after:2024-01-01 filetype:pdf"
  }'
```

---

### `web_fetch`

Fetches and extracts content from URLs with intelligent fallbacks.

**Fetch Strategy (in order):**
1. **Jina AI Reader** - Most reliable, handles anti-bot
2. **Firecrawl** - If `FIRECRAWL_API_KEY` configured
3. **Direct fetch** - Fallback with lib/web.js

**Features:**
- HTML to markdown conversion
- Main content extraction (removes nav, ads, footers)
- Metadata extraction (title, description, published time)
- JavaScript-heavy page support
- Anti-bot bypass
- Automatic retry with exponential backoff
- Link extraction (optional)

**Output Format:**
```markdown
<!-- Source: https://example.com/article -->
<!-- Title: Article Title -->
<!-- Format: markdown | Characters: 5432 | Est. tokens: ~1358 -->
<!-- Fetched via: jina -->
<!-- Description: Article description... -->
<!-- Published: 2024-01-15T10:00:00Z -->

# Article Title

Main content here...
```

**Example with link extraction:**
```bash
curl -X POST http://localhost:3000/api/fetch \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/article",
    "format": "markdown",
    "max_chars": 10000,
    "extract_links": true
  }'
```

---

### `search_images`

Searches for images using available backends.

---

### `view_image`

Analyzes image URLs and returns metadata.

**Features:**
- OCR for images with text
- Content-type validation
- Accessibility analysis

---

## Models

### Current Model

| Property | Value |
|----------|-------|
| **Model ID** | `gemini-2.5-flash-lite` |
| **Public Name** | `nub-agent` |
| **Provider** | Google Gemini |
| **Developer** | Ambitiousnoob |

### Model Aliases

The following model names resolve to `gemini-2.5-flash-lite`:

- `nub-agent`
- `gemini-3-flash`
- `polly`
- `polly-*`

### Fallback Chain

If the primary model fails, requests fall back to:

1. `gemini-2.5-flash-lite` (primary)
2. `gemini-2.5-flash`
3. `gemini-3-flash-preview`

### Thinking Configuration

| Model | Thinking Config |
|-------|-----------------|
| `gemini-2.5-flash` | `thinkingBudget: 0` |
| `gemini-2.5-flash-lite` | `thinkingBudget: 0` |
| `gemini-3-flash-preview` | `thinkingLevel: "minimal"` |

---

## Rate Limiting

- Controlled by mutex queue
- Avoids parallel API calls unless needed
- Per-key memory isolation

---

## Error Responses

### Standard Error Format
```json
{
  "error": "Error message description"
}
```

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (invalid JSON, missing fields) |
| 401 | Unauthorized (invalid API key) |
| 403 | Forbidden (private network blocked) |
| 405 | Method not allowed |
| 500 | Internal server error |

---

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` or `GEMINI_API_KEYS` | Google Gemini API key(s) |
| `DATABASE_URL` | MySQL/TiDB connection string |

### Optional

#### Web Search & Fetch

| Variable | Description |
|----------|-------------|
| `TAVILY_API_KEY` / `TAVILY_API_KEYS` | Tavily AI search backend |
| `SERPER_API_KEY` / `SERPER_API_KEYS` | Google Search via Serper (dork operators) |
| `JINA_API_KEY` / `JINA_API_KEYS` | Jina AI reader & search enrichment |
| `BRAVE_API_KEY` / `BRAVE_API_KEYS` | Brave Search API |
| `FIRECRAWL_API_KEY` | Firecrawl web scraper (enhanced fetch) |

#### AI & Memory

| Variable | Description |
|----------|-------------|
| `CEREBRAS_API_KEY` | Enables Cerebras memory reranking |
| `CEREBRAS_MEMORY_MODEL` | Override retrieval model |

#### Facebook Messenger

| Variable | Description |
|----------|-------------|
| `PAGE_ACCESS_TOKEN` | Facebook Page access token |
| `VERIFY_TOKEN` / `MESSENGER_VERIFY_TOKEN` | Messenger webhook verification |
| `FB_GRAPH_API` | Override Graph API origin (default: `https://graph.facebook.com/v21.0`) |
| `MESSENGER_SYSTEM_PROMPT` | Custom system instruction for Messenger |
| `MESSENGER_MAX_MESSAGE_CHARS` | Max characters per outbound message |

---

## Code Examples

### JavaScript/TypeScript

```typescript
// Chat request
const response = await fetch('/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    messages: [{ role: 'user', content: 'Hello!' }],
    stream: false,
  }),
});

const data = await response.json();
console.log(data.output_text);
```

### Python

```python
import requests

response = requests.post(
    'https://your-domain.vercel.app/api/chat',
    json={
        'messages': [{'role': 'user', 'content': 'Hello!'}],
        'stream': False,
    }
)

data = response.json()
print(data['output_text'])
```

### cURL with Streaming

```bash
curl -N -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

---

## Best Practices

1. **Use scoped memory** - Include `X-API-Key` for personalized responses
2. **Batch operations** - Combine related queries in single messages
3. **Handle errors** - Always check response status and error field
4. **Respect limits** - Keep messages under token limits
5. **Use research mode** - Set `research_mode: true` for academic queries

---

## Support

- GitHub: [Ambitiousnoob/nubagent](https://github.com/Ambitiousnoob/nubagent)
- Documentation: See `README.md` and `AGENTS.md`
