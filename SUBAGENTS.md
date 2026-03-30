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
- `.codex/agents/nub_answer_verification_orchestrator.toml`
- `.codex/agents/nub_release_ops.toml`

## Decision Support Ownership

- `.codex/agents/nub_competitive_analyst.toml`
  Scope: criteria-based comparison of models, providers, libraries, products, and implementation options before committing to an approach.
- `.codex/agents/nub_data_researcher.toml`
  Scope: dataset, metric, pipeline, and quantitative evidence research used to support product, architecture, and operational decisions.
- `.codex/agents/nub_docs_researcher.toml`
  Scope: documentation-backed verification of external APIs, framework behavior, version differences, defaults, and migration caveats.
- `.codex/agents/nub_fetched_info_verifier.toml`
  Scope: verification that fetched pages, excerpts, evidence blocks, and citation targets actually support the facts NubAgent plans to synthesize or present.
- `.codex/agents/nub_research_analyst.toml`
  Scope: broader technical investigations, design questions, and implementation-approach research when no narrower decision-support owner is a better fit.
- `.codex/agents/nub_search_specialist.toml`
  Scope: fast discovery and ranked high-signal search hits across the codebase or external sources before deeper analysis begins.

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

- `.codex/agents/nub_answer_verification_orchestrator.toml`
  Scope: enforcing that fetched-evidence or docs verification happens before any user-facing answer is finalized.
- `.codex/agents/nub_fetched_info_verifier.toml`
  Scope: fetched-evidence validation, claim-to-source support checks, contradiction surfacing, and weak-evidence filtering before synthesis or user-facing answer claims.
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

Use `nub_docs_researcher` for external documentation verification.
Use `nub_docs_owner` and `nub_docs_sync` for editing NubAgent's own repo docs.

## SearchEngine Micro-Scope Baseline

For `src/SearchEngine.jsx`, default to these micro-owners:

- Input capture, upload limits, attachment chip behavior: `nub_search_intake_owner`
- Attachment parsing and evidence preparation: `nub_attachment_ingest_owner`
- Search orchestration, ranking, fetch, synthesis, answer metadata: `nub_research_pipeline_owner`
- Liberty cards, citations, tables, rendered answer copy: `nub_answer_presentation_owner`
- Session save/load behavior: `nub_library_owner` or `nub_state_owner`, depending on persistence path

Even when a request changes only a tiny piece of `src/SearchEngine.jsx`, route it to the matching micro-owner instead of a generic frontend agent.

When the touched path includes fetched excerpts, evidence blocks, or citation support that need factual validation before answer text is trusted, add `nub_fetched_info_verifier` as the verification owner after the fetch or pipeline owner finishes.

When the task includes producing or approving a user-facing answer from fetched information, add `nub_answer_verification_orchestrator` to enforce that the verification pass happens before answer finalization.

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
5. Answer verification gate: `nub_answer_verification_orchestrator` ensures fetched-evidence or docs verification runs before user-facing answer finalization.
6. Verification: test and review agents validate the merged result.
7. Docs and release: update docs and deploy notes when behavior changes.
