---
title: 'workflow-fix: root-draft observer misses untracked strays under scripts/'
kind: infra
tags:
- wf-fix
created_at: '2026-08-05T07:52:05Z'
has_clean_result: false
origin_prompt: 'Orchestrator observation on #2054 (2026-08-05): an untracked scripts/issue1482_blind_read_api.py
  reddened the repo-root workflow_lint oracle while origin/main lints clean; the #1341
  root_draft_pass is scoped to top-level root *.py and never saw it.'
workflow: v1
---
ARCHIVED as a false-premise filing (self-caught within minutes of filing, before
any session was dispatched — the filer reported "infra dispatch cap (5) full,
NOT dispatching", so zero pipeline time was burned).

The filing claimed the watcher's root-draft observer is "scoped to TOP-LEVEL
root *.py" and therefore blind to an untracked stray under `scripts/`. That is
FALSE. Verified in the code, not the docstring:

- `ROOT_DRAFT_PATHSPEC: tuple[str, ...] = ("*.py",)`
  (`scripts/autonomous_session_watch.py:6487`) — a git PATHSPEC glob, which
  matches at ANY depth, not just the root.
- `git --no-optional-locks status --porcelain -- '*.py'` at the repo root
  returns `scripts/issue1482_blind_read_api.py` today: the pass sees it.
- `.claude/cache/root-draft-events.jsonl` ALREADY carries a row naming
  `issue1482_blind_read_api.py` — it has been escalating.

My error: I read the docstring prose ("stale untracked `*.py` drafts at the
SHARED repo root") as a path SCOPE and filed without grepping
`ROOT_DRAFT_PATHSPEC`. The workflow-fix filer duties require binding the claim
to the code at compose time; the `verified-at-filing` line I wrote verified the
STRAY's untracked status (correct) but never the observer's scope (the actual
load-bearing claim). Recording it so the pattern is visible.

The residual is real but already tracked and is NOT this: the observer escalates
correctly and nobody acts on it. Open task #1761 ("daily-held: untracked
scripts/issue823_single_split_protocol") is exactly that held decision — an
untracked `scripts/*.py` escalated daily for ~12 days, parked because
committing-or-deleting another session's draft is a user call. The #1482 stray
is a second instance of the same held class, not a new observability gap.
