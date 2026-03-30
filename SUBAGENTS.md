# NubAgent Subagent Map

This repo uses dedicated subagent ownership down to the smallest meaningful part.
If a request touches code, docs, tests, or release notes, assign each touched slice to a named subagent before editing.

These owners exist to make NubAgent a better information finder first:

- improve trustworthy search coverage before adding cosmetic complexity
- improve evidence extraction and citation traceability before adding answer polish
- prefer changes that make uncertainty, source quality, and verification more explicit to the user

## Core Rules

- Every meaningful work slice gets exactly one dedicated subagent owner.
- A slice may be as small as a helper function, a validation branch, a table column, a doc paragraph, or a single endpoint path.
- If two micro-slices are tightly coupled, they may share one owner, but no touched part is left unowned.
- Prefer project-scoped `nub_*.toml` agents in `.codex/agents/` when both prefixed and generic aliases exist.
- If no existing agent cleanly owns a new slice, add a new `.codex/agents/*.toml` definition before implementation.

## Management Layer

Use these agents to define scope and sequencing before implementation:

- `.codex/agents/nub_scope_chief.toml`
- `.codex/agents/nub_product_guardian.toml`
- `.codex/agents/nub_program_manager.toml`
- `.codex/agents/nub_system_architect.toml`
- `.codex/agents/nub_release_ops.toml`

## Active Search UI Ownership

Use these as the default owners for the live search-first product surface:

- `.codex/agents/nub_search_intake_owner.toml`
  Scope: query entry, upload validation, attachment context assembly, input-side `src/SearchEngine.jsx` work.
- `.codex/agents/nub_attachment_ingest_owner.toml`
  Scope: attachment parsing, upload digestion, and content ingestion edge cases.
- `.codex/agents/nub_research_pipeline_owner.toml`
  Scope: query expansion, search fan-out, ranking, fetch concurrency, evidence chunking, synthesis orchestration, research metadata.
- `.codex/agents/nub_answer_presentation_owner.toml`
  Scope: result cards, citations, tables, answer rendering, presentation-side `src/SearchEngine.jsx` work.
- `.codex/agents/nub_library_owner.toml`
  Scope: saved sessions, library surface, export/history experience.
- `.codex/agents/nub_state_owner.toml`
  Scope: persisted UI state, state namespaces, remote/local state coordination.

## Support UI Ownership

- `.codex/agents/nub_app_shell_owner.toml`
  Scope: `src/App.jsx`, top-level view switching, shell layout, and PWA prompt behavior.
- `.codex/agents/nub_settings_owner.toml`
  Scope: settings UI, API key and model preferences, theme selection, and settings-side validation.
- `.codex/agents/nub_shared_ui_owner.toml`
  Scope: shared UI primitives, theme providers, toast or modal infrastructure, and base styling tokens.

## Backend And Tool Ownership

- `.codex/agents/nub_chat_backend_owner.toml`
  Scope: `/api/chat`, model orchestration, chat backend contracts.
- `.codex/agents/nub_search_backend_owner.toml`
  Scope: `/api/search`, search-provider routing, search backend behavior.
- `.codex/agents/nub_content_backend_owner.toml`
  Scope: `/api/content` plus the `/api/fetch`, `/api/read`, and `/api/crawl` aliases, backend content retrieval paths.
- `.codex/agents/nub_search_tool_owner.toml`
  Scope: search tool implementations and provider-specific logic.
- `.codex/agents/nub_fetch_tool_owner.toml`
  Scope: fetch/read tooling and page extraction behavior.
- `.codex/agents/nub_image_tool_owner.toml`
  Scope: image search/view flows and image-related tool behavior.
- `.codex/agents/nub_memory_owner.toml`
  Scope: memory APIs, namespace isolation, retrieval, persistence.
- `.codex/agents/nub_api_boundary_owner.toml`
  Scope: cross-endpoint contracts, request/response normalization, boundary consistency.

## Quality, Docs, And Release Ownership

- `.codex/agents/nub_test_engineer.toml`
  Scope: tests, verification commands, regression coverage.
- `.codex/agents/nub_code_reviewer.toml`
  Scope: defect review, correctness, regressions, code quality, and evidence-path risk.
- `.codex/agents/nub_docs_owner.toml`
  Scope: README, API docs, architecture notes, operator guidance.
- `.codex/agents/nub_docs_sync.toml`
  Scope: keeping docs aligned when behavior changes.
- `.codex/agents/nub_release_ops.toml`
  Scope: deploy readiness, environment assumptions, rollout risk.

## SearchEngine Micro-Scope Baseline

For `src/SearchEngine.jsx`, default to these micro-owners:

- Input capture, upload limits, attachment chip behavior: `nub_search_intake_owner`
- Attachment parsing and evidence preparation: `nub_attachment_ingest_owner`
- Search orchestration, ranking, fetch, synthesis, answer metadata: `nub_research_pipeline_owner`
- Liberty cards, citations, tables, rendered answer copy: `nub_answer_presentation_owner`
- Session save/load behavior: `nub_library_owner` or `nub_state_owner`, depending on persistence path

Even when a request changes only a tiny piece of `src/SearchEngine.jsx`, route it to the matching micro-owner instead of a generic frontend agent.

## Micro-Scope Assignment Protocol

Before implementation, record a handoff note with:

- owner
- exact file or module scope
- concrete deliverable
- dependencies
- validation expected

Do not allow overlapping edits unless the orchestrator explicitly coordinates the overlap.

## Default Handoff Order

1. Scope and product intent: management layer agents define the target behavior.
2. Architecture and placement: the system architect confirms boundaries when the change crosses surfaces.
3. Micro-assignment: each touched part gets a dedicated owner from `.codex/agents/`.
4. Implementation: owners edit only within their bounded scope.
5. Verification: test and review agents validate the merged result.
6. Docs and release: update docs and deploy notes when behavior changes.
