---
title: Consolidate redundant scripts, skills, agents, and CLAUDE.md sections
kind: infra
tags: []
created_at: '2026-04-22T03:00:25.000Z'
has_clean_result: false
sagan_id: 3851d010-9055-4f85-85f1-989384abf41c
sagan_number: 72
priority: normal
---
## Summary

Audit surfaced redundancies across Claude Code setup and project scripts. Consolidate in phases to reduce cognitive overhead without changing behavior.

## Redundancies

### Scripts
`scripts/pod.py` is a dispatcher, but standalone backends (`sync_env.sh`, `sync_env_keys.sh`, `sync_pods.sh`, `sync_datasets.py`, `sync_models.py`, `pull_results.py`, `cleanup_pod.py`) are also documented as direct entry points in CLAUDE.md.

### Agents
`.claude/agents/manager.md.deprecated` — deadweight (superseded by `research-pm.md`).

### Skills (user-level)
- `humanizer`, `avoid-ai-writing` — duplicates of `ai-critic-loop`.
- `code-refactoring` — subsumed by `cleanup` + `refactor` + `deep-clean`.
- `simplify`, `review` — plugin-owned; out of scope.

### CLAUDE.md
- Reproducibility Card table duplicated; `templates/experiment_report.md` already has it.
- Pod/sync commands documented in three sections.
- Pod name table also in `.claude/agents/manager.md`.

### Skills layering
`issue`, `experiment-runner`, `auto-experiment-runner`, `experiment-proposer` — boundaries undocumented.

### Review agents
`critic`, `reviewer`, `code-reviewer` — not duplicates, but roles not obvious.

### Rules
- `.claude/rules/external-repos.md` is reference, belongs in `docs/`.
- `.claude/rules/research-project-structure.md` duplicates agent-roles table.

## Plan

- **Phase 0:** Delete `manager.md.deprecated`, user skills `humanizer`/`avoid-ai-writing`/`code-refactoring`; update `deep-clean` reference.
- **Phase 1:** Mark standalone sync scripts as `INTERNAL — backend for scripts/pod.py`; route CLAUDE.md Pre-Launch Protocol through `pod.py`.
- **Phase 2:** CLAUDE.md dedup — replace inline Reproducibility Card with pointer; consolidate pod/sync docs; remove pod table from `manager.md`.
- **Phase 3:** Scope & boundaries banners on experiment skills; remove duplicated content from `issue`.
- **Phase 4:** Role banners on `critic.md` / `reviewer.md` / `code-reviewer.md`.
- **Phase 5:** Move `external-repos.md` to `docs/`; strip agent-roles table from `research-project-structure.md`.

## Non-goals

- No behavioral changes.
- Not merging review agents.
- Not inlining sync scripts into `pod.py`.

## Test plan

- [ ] `python scripts/pod.py config --check` still passes
- [ ] No broken internal links after doc moves
- [ ] `grep -r` for each deleted name returns only history/transcripts
