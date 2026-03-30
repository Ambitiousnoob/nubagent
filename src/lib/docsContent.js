export const API_REFERENCE_SECTIONS = [
  {
    id: 'chat-api',
    title: 'Chat Runtime',
    paths: ['/api/chat'],
    methods: ['GET', 'POST'],
    summary: 'Primary answer-generation endpoint. Runs model orchestration, tool use, streaming, and scoped memory retrieval.',
    keyPoints: [
      'Use GET for capability metadata and POST for completions.',
      'Supports tool execution, research mode, streamed or non-streamed output, and scoped memory.',
      'This is the endpoint to hit when you want NubAgent to think, search, fetch, and answer in one loop.',
    ],
    requestShape: `{
  "messages": [{ "role": "user", "content": "What changed in the latest Vercel build cache docs?" }],
  "model": "nub-agent",
  "stream": false,
  "research_mode": true,
  "use_tools": true
}`,
    responseShape: `{
  "ok": true,
  "model": "nub-agent",
  "output_text": "...",
  "tools_used": ["web_search", "web_fetch"],
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
}`,
    implementationFiles: ['api/chat.js', 'lib/litehost-chat.js'],
  },
  {
    id: 'web-api',
    title: 'Combined Web Research',
    paths: ['/api/web'],
    methods: ['GET', 'POST'],
    summary: 'One-shot web research endpoint that combines search, fetch, ranking, and evidence shaping.',
    keyPoints: [
      'Use this when you want ranked sources plus optional fetched content in one request.',
      'Supports query-only, URL-only, and mixed modes.',
      'Good fit for background research workflows and non-chat integrations.',
    ],
    requestShape: `{
  "query": "best practices for verifying fetched claims in RAG systems",
  "maxResults": 12,
  "fetchContent": true,
  "rag": true
}`,
    responseShape: `{
  "ok": true,
  "query": "...",
  "sources": [{ "title": "...", "url": "...", "citationIndex": 1 }],
  "fetched": [{ "source": { "title": "..." }, "content": "# ..." }],
  "evidence": ["[1] ..."]
}`,
    implementationFiles: ['api/web.js', 'api/search.js', 'api/content.js'],
  },
  {
    id: 'search-api',
    title: 'Search Backend',
    paths: ['/api/search'],
    methods: ['GET', 'POST'],
    summary: 'Provider-routing search endpoint for query fan-out, ranking inputs, and source discovery.',
    keyPoints: [
      'Routes across DuckDuckGo, Tavily, Serper, Jina, and Brave depending on query shape and key availability.',
      'Returns normalized source records used by the live research pipeline.',
      'Useful if you want search-only behavior without fetch or synthesis.',
    ],
    requestShape: `{
  "query": "OpenAI responses API streaming docs",
  "maxResults": 10
}`,
    responseShape: `{
  "ok": true,
  "endpoint": "/api/search",
  "results": [{ "title": "...", "url": "...", "description": "...", "source": "serper" }]
}`,
    implementationFiles: ['api/search.js', 'api/tools/web_search.js'],
  },
  {
    id: 'content-aliases',
    title: 'Content Retrieval Aliases',
    paths: ['/api/content', '/api/fetch', '/api/read', '/api/crawl'],
    methods: ['GET', 'POST'],
    summary: 'Shared content endpoint plus aliases for raw fetch, smart read, and crawl-style extraction.',
    keyPoints: [
      'Use /api/fetch for direct page retrieval, /api/read for smart extraction, and /api/crawl for multi-page crawling.',
      'All three aliases route into the same backend boundary.',
      'This is the right place to add new retrieval modes or shared fetch controls.',
    ],
    requestShape: `{
  "url": "https://developers.openai.com",
  "mode": "smart",
  "maxChars": 12000
}`,
    responseShape: `{
  "ok": true,
  "content": "# ...",
  "url": "https://developers.openai.com",
  "title": "..."
}`,
    implementationFiles: ['api/content.js', 'api/tools/web_fetch.js', 'vercel.json'],
  },
  {
    id: 'memory-api',
    title: 'Memory Retrieval And Persistence',
    paths: ['/api/memory', '/api/state'],
    methods: ['GET', 'POST'],
    summary: 'State and memory endpoints for scoped persistence, retrieval, and API-key-based memory search.',
    keyPoints: [
      'Use /api/memory for memory operations and /api/state for browser or anonymous state scopes.',
      'Designed to preserve context without leaking data across scopes.',
      'Affects answer continuity, memory search, and persistence correctness.',
    ],
    requestShape: `{
  "query": "what stack did I mention before?",
  "limit": 5
}`,
    responseShape: `{
  "ok": true,
  "results": [{ "role": "user", "content": "..." }],
  "scope": "api_key"
}`,
    implementationFiles: ['api/memory.js', 'api/state.js', 'lib/api-key-memory.js', 'lib/state-scope.js'],
  },
  {
    id: 'utility-aliases',
    title: 'Utility Aliases',
    paths: ['/api/health', '/api/analytics', '/api/export', '/api/messenger', '/api/utils'],
    methods: ['GET', 'POST'],
    summary: 'Shared utility boundary for operational status, analytics, export, and Messenger webhook behavior.',
    keyPoints: [
      'The public aliases route into one shared utility handler.',
      'Useful when you need health checks, export flows, or Messenger webhook support.',
      'When adding a new alias here, update both the resolver logic and Vercel rewrites.',
    ],
    requestShape: `GET /api/health
GET /api/messenger?hub.mode=subscribe&hub.verify_token=...&hub.challenge=...
POST /api/export`,
    responseShape: `{
  "status": "healthy"
}

or

EVENT_RECEIVED`,
    implementationFiles: ['api/utils.js', 'api/utils-handler.cjs', 'lib/utils-routing.cjs', 'vercel.json'],
  },
];

export const API_CREATION_STEPS = [
  {
    title: 'Pick the boundary first',
    body: 'Decide whether the feature deserves its own top-level endpoint or should be an alias inside a shared boundary.',
    bullets: [
      'Use a dedicated file like `api/chat.js` or `api/search.js` when the contract is its own product surface.',
      'Use `api/content.js` for fetch/read/crawl-style retrieval modes that share parsing behavior.',
      'Use `api/utils-handler.cjs` plus `api/utils.js` when the feature belongs with utility aliases such as health, export, analytics, or Messenger.',
    ],
  },
  {
    title: 'Implement both metadata and action paths',
    body: 'Every public endpoint should answer GET with clear metadata and POST with the real behavior.',
    bullets: [
      'GET should describe the endpoint, available methods, and high-level capabilities.',
      'POST should validate the body, return explicit error codes, and normalize output.',
      'Shared alias endpoints should resolve the alias before branching into behavior-specific logic.',
    ],
  },
  {
    title: 'Wire the route in Vercel',
    body: 'Keep the runtime file and the public path aligned in `vercel.json`.',
    bullets: [
      'Add or update rewrites for every public alias.',
      'If the implementation is a wrapper plus a shared handler, make sure only the public wrapper owns the public path.',
      'Do not create filename collisions that Vercel treats as the same route.',
    ],
  },
  {
    title: 'Document and test the contract',
    body: 'Add the contract to the docs and verify at least one happy path and one safe failure path.',
    bullets: [
      'Update `API.md` with request and response examples.',
      'Update `README.md` or web-facing docs when the endpoint is part of the product story.',
      'Add targeted tests for routing helpers, alias resolution, or payload handling where regressions are likely.',
    ],
  },
];

export const API_CREATION_CHECKLIST = [
  'Choose the public path and the owning backend file.',
  'Add GET metadata for discoverability.',
  'Add POST validation and stable JSON or text responses.',
  'Register rewrites in `vercel.json`.',
  'Update `API.md` and any web-facing docs.',
  'Add targeted verification commands before shipping.',
];

export const BASIC_ENDPOINT_EXAMPLE = `const writeJson = (res, status, data) => {
  res.status(status);
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(data));
};

module.exports = async (req, res) => {
  if (req.method === "GET") {
    return writeJson(res, 200, {
      ok: true,
      endpoint: "/api/example",
      methods: ["GET", "POST"],
    });
  }

  if (req.method !== "POST") {
    return writeJson(res, 405, { error: "Method not allowed" });
  }

  const body = await readBody(req);
  return writeJson(res, 200, { ok: true, echo: body });
};`;

export const ALIAS_ENDPOINT_EXAMPLE = `// vercel.json
{
  "source": "/api/export",
  "destination": "/api/utils.js"
}

// api/utils-handler.cjs
const action = resolveUtilsAction(req);
switch (action) {
  case "export":
    return handleExport(req, res);
  default:
    return handleMetadata(res);
}`;

export const SUBAGENT_GROUPS = [
  {
    id: 'orchestration',
    title: 'Orchestration And Release',
    summary: 'These agents decide scope, sequencing, verification gates, and release readiness across the whole product.',
    agents: [
      { name: 'nub_scope_chief', scope: 'Defines the smallest coherent work slice and scope boundary before changes begin.' },
      { name: 'nub_product_guardian', scope: 'Keeps changes aligned with the product goal of being a trustworthy information finder.' },
      { name: 'nub_program_manager', scope: 'Sequences multi-owner work and keeps dependencies crisp.' },
      { name: 'nub_system_architect', scope: 'Confirms placement and boundaries when a change crosses surfaces.' },
      { name: 'nub_answer_verification_orchestrator', scope: 'Enforces fetched-evidence or docs verification before any user-facing answer is finalized.' },
      { name: 'nub_release_ops', scope: 'Owns deploy readiness, environment assumptions, and rollout risk.' },
    ],
  },
  {
    id: 'research',
    title: 'Research And Verification',
    summary: 'These agents help NubAgent choose, verify, and triage information before it becomes answer content.',
    agents: [
      { name: 'nub_competitive_analyst', scope: 'Compares external options such as models, providers, libraries, or products.' },
      { name: 'nub_data_researcher', scope: 'Handles quantitative evidence, metrics, datasets, and evidence-backed measurement questions.' },
      { name: 'nub_docs_researcher', scope: 'Verifies external API or framework behavior from primary documentation.' },
      { name: 'nub_fetched_info_verifier', scope: 'Checks whether fetched pages, excerpts, and citation targets actually support the facts being claimed.' },
      { name: 'nub_research_analyst', scope: 'Owns broader technical investigations when no narrower decision-support agent fits.' },
      { name: 'nub_search_specialist', scope: 'Finds the highest-signal files or external references before deeper analysis starts.' },
    ],
  },
  {
    id: 'search-surface',
    title: 'Search Product Surface',
    summary: 'These owners cover the live search-first UX from intake through answer rendering and saved research sessions.',
    agents: [
      { name: 'nub_search_intake_owner', scope: 'Owns query entry, upload validation, and input-side SearchEngine behavior.' },
      { name: 'nub_attachment_ingest_owner', scope: 'Handles attachment parsing, upload digestion, and ingestion edge cases.' },
      { name: 'nub_research_pipeline_owner', scope: 'Owns planning, fan-out, ranking, fetch orchestration, evidence shaping, and synthesis flow.' },
      { name: 'nub_answer_presentation_owner', scope: 'Owns result cards, citations, tables, and answer rendering.' },
      { name: 'nub_library_owner', scope: 'Owns saved sessions, export and history behavior, and the library surface.' },
      { name: 'nub_state_owner', scope: 'Owns persisted UI state, namespaces, and remote or local state coordination.' },
    ],
  },
  {
    id: 'shell',
    title: 'Shell And Settings',
    summary: 'These owners cover the application shell and shared interactive surfaces around the core research flow.',
    agents: [
      { name: 'nub_app_shell_owner', scope: 'Owns App layout, top-level navigation, view switching, and shell behavior.' },
      { name: 'nub_settings_owner', scope: 'Owns settings UI, provider preferences, and settings validation.' },
      { name: 'nub_shared_ui_owner', scope: 'Owns base UI primitives, theme providers, modal infrastructure, and shared styling tokens.' },
    ],
  },
  {
    id: 'backend',
    title: 'Backends And Tools',
    summary: 'These owners cover public API boundaries, route contracts, provider logic, retrieval behavior, and persistence.',
    agents: [
      { name: 'nub_chat_backend_owner', scope: 'Owns `/api/chat`, model orchestration, and chat backend contracts.' },
      { name: 'nub_search_backend_owner', scope: 'Owns `/api/search` and search-provider routing behavior.' },
      { name: 'nub_content_backend_owner', scope: 'Owns `/api/content` and the `/api/fetch`, `/api/read`, and `/api/crawl` aliases.' },
      { name: 'nub_search_tool_owner', scope: 'Owns search tool implementation details and provider-specific logic.' },
      { name: 'nub_fetch_tool_owner', scope: 'Owns fetch and read tooling plus page extraction behavior.' },
      { name: 'nub_image_tool_owner', scope: 'Owns image search, image view, and image-related tool behavior.' },
      { name: 'nub_memory_owner', scope: 'Owns memory APIs, retrieval, namespace isolation, and persistence.' },
      { name: 'nub_api_boundary_owner', scope: 'Owns cross-endpoint request and response normalization plus boundary consistency.' },
    ],
  },
  {
    id: 'quality',
    title: 'Quality And Docs',
    summary: 'These owners validate the result, keep repo docs aligned, and protect operator trust.',
    agents: [
      { name: 'nub_test_engineer', scope: 'Owns regression coverage, verification commands, and test-focused validation.' },
      { name: 'nub_code_reviewer', scope: 'Owns correctness, regression, maintainability, and evidence-path risk review.' },
      { name: 'nub_docs_owner', scope: 'Owns README, API docs, architecture notes, and operator-facing guidance.' },
      { name: 'nub_docs_sync', scope: 'Owns keeping docs aligned after behavior or contract changes.' },
    ],
  },
];
