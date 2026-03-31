# NubAgent Development Plan

Current plan as of March 31, 2026.

## Target State

Keep NubAgent small and explicit:

- one public API route: `/api/chat`
- one provider adapter: Gemini
- one frontend surface: documentation only
- one stable contract: non-streaming JSON chat completions

## Current State

- `/api/chat` is a simple Gemini-backed chatbot.
- The old multi-endpoint public API surface is gone.
- The frontend is a docs-only shell that explains the current contract.
- Request parsing includes a body-size cap and explicit validation errors.

## Shipped

- Replaced the old agentic `/api/chat` flow with a direct Gemini chat path.
- Added `lib/gemini-chat.js` as the provider adapter.
- Updated `API.md`, `README.md`, `.env.example`, and rendered docs for the new contract.
- Removed stale documentation that described the older research platform as the current product.

## Next Priorities

1. Add authentication or quota controls if the endpoint will be public on the internet.
2. Add more contract tests around provider failures and invalid request shapes.
3. Decide whether the docs-only frontend should stay in-repo or move to a separate docs site.
4. Add lightweight observability around error rates and provider latency.

## Change Rules

- Do not add new top-level public API routes casually.
- Keep Gemini-specific logic behind `lib/gemini-chat.js`.
- Keep docs and contract changes in the same commit.
- Run targeted tests, lint, and build before pushing.
