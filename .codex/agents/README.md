# NubAgent Project-Scoped Agents

This directory contains the executable project-scoped Codex agents for nubagent.

The canonical roster uses only `nub_*` agent files.
Those owners are tuned for NubAgent's core job: being a fast, trustworthy information finder with strong source coverage, evidence quality, and citation clarity.

Use `SUBAGENTS.md` as the routing map.
That file defines the smallest practical owner for each slice of the research pipeline, including the micro-split stages inside `src/SearchEngine.jsx`.

Working rule:

- start with the smallest matching `nub_*` owner
- escalate to the management layer for broad or cross-cutting work
- finish with `nub_test_engineer`, `nub_code_reviewer`, `nub_docs_sync`, or `nub_release_ops` when the change requires them

The older generic manager roster has been retired on purpose.
If a new repo area appears and no current owner cleanly fits it, update `SUBAGENTS.md` and add a new `nub_*.toml` file here instead of recreating generic buckets.
