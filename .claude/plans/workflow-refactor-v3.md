# Workflow refactor v3 — adversarial-planner approved

3 stages → 3 PRs. Total: ~+446 / -91 LoC.

## Architectural deltas vs reverted 5aa6a4f
1. Event-driven workflow (no cron, no audit-routing recurring step)
2. Extend `gh_project.py` in place; no parallel scripts
3. 3-step body-promote (no SHA check, no atomicity theater)
4. `LABEL_TO_COLUMN` table is single source of truth, referenced by prod + tests
5. Prose-only narratives matching #237; reuse existing comma-form `/clean-results`
6. Original body preserved as a comment (not duplicated inline)
7. 8 columns (Done variants folded; Clean Results column dropped)
8. Per-issue concurrency `cancel-in-progress: true` collapses burst events

## Stage 1 — Project board sync
**This PR.** Event-driven `on: issues: [opened, reopened, labeled, unlabeled]`. Per-issue concurrency. PROJECT_TOKEN secret already configured.

Files:
- `.github/workflows/project-sync.yml` — event-driven router with workflow_dispatch fallback
- `scripts/gh_project.py` — adds `set-status-from-labels`, `snapshot`, `migrate-options` subcommands + `LABEL_TO_COLUMN` table
- `tests/test_label_to_column_coverage.py` — 5 tests asserting routing covers all live status:* labels
- `.claude/skills/mentor-prep/SKILL.md` — defaults to `--label clean-results` (column dropped)

8 columns: Proposed, Plan Review, Approved, In Flight, Awaiting Promotion, Sign-off, Blocked, Done.

## Stage 2 — Inline clean-results (next PR)
3-step body-promote: preserve original as comment → edit body → add label.
One-shot migration script for 7 in-flight legacy issues.
Deletes 3 of 5 `set-status` callsites (the 2 that drive label changes stay).

## Stage 3 — Narrative consolidation (final PR)
Reuse existing `/clean-results <N1>,<N2>,<N3>` comma-form.
Add `Source-issues:` and `Supersedes:` lines to template.
Reference exemplar: #237.

## Sequencing
Stage 1 → Stage 2 → Stage 3.
Stage 1 makes the 3 redundant `set-status` callsites at `analyzer.md:118`, `clean-results SKILL.md:188`, `checklist.md:126` redundant. Stage 2 deletes them.
The 2 remaining callsites (`issue/SKILL.md:243` clarify→In-Progress, `issue/SKILL.md:753` Done variant) STAY — they trigger label changes that the workflow then routes.
