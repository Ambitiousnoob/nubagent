# NubAgent Project-Scoped Agents

This directory contains the executable project-scoped Codex agents for nubagent.

The canonical roster uses only `nub_*` agent files.
Those owners are tuned for NubAgent's core job: being a fast, trustworthy information finder with strong source coverage, evidence quality, and citation clarity.

Use `SUBAGENTS.md` as the routing map.
That file defines the smallest practical owner for each slice of the research pipeline, including the micro-split stages inside `src/SearchEngine.jsx`.

Working rule:

- start with the smallest matching `nub_*` owner
- use `nub_competitive_analyst` when the first task is choosing between external options rather than editing code
- use `nub_data_researcher` when the first task is gathering quantitative evidence or dataset-backed decision support
- use `nub_docs_researcher` when the first task is verifying external API or framework behavior from primary docs rather than editing repo docs
- use `nub_research_analyst` for broader technical investigations that are not primarily option comparison, quantitative evidence work, or direct docs verification
- use `nub_search_specialist` for fast discovery and triage when the immediate need is finding the highest-signal files or external references before deeper work starts
- escalate to the management layer for broad or cross-cutting work
- finish with `nub_test_engineer`, `nub_code_reviewer`, `nub_docs_sync`, or `nub_release_ops` when the change requires them

The older generic manager roster has been retired on purpose.
If a new repo area appears and no current owner cleanly fits it, update `SUBAGENTS.md` and add a new `nub_*.toml` file here instead of recreating generic buckets.
