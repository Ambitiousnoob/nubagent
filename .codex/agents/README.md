# NubAgent Project-Scoped Agents

This directory contains the executable project-scoped Codex agents for nubagent.

Use `SUBAGENTS.md` as the canonical routing map.
That file explains which agent owns each smallest practical slice, including the split stages inside `src/SearchEngine.jsx`.

Working rule:

- start with the smallest matching owner
- escalate to the management agents for broad or cross-cutting work
- finish with `verification_manager` and `docs_release_manager` when the change requires them

If a new repo area appears and no current agent cleanly owns it, update `SUBAGENTS.md` and add a new `.toml` file here instead of falling back to a generic bucket.
