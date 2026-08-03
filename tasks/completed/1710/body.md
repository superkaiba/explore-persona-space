---
title: 'daily-fix: repo-root guard FP on ssh/note payloads'
kind: infra
tags:
- wf-fix
- wf-fix-fp:257796b1a49b
- daily-auto-filed
created_at: '2026-07-26T07:08:34Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 2): Four blocked calls in two
  sessions carried no repo-root mutation at all: two ssh commands targeting a pod
  repo, a post-marker whose note prose merely quoted a destructive git phrase, and
  a /daily fingerprint helper whose Python string literal quoted the same phrase,
  each costing a turn plus a rewrite.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the `/daily` 2026-07-25 problem sweep. `guard_repo_root_branch.sh`
blocked four calls that carried no repo-root mutation at all — including one issued by
this very `/daily` run, which is the fourth occurrence and was observed live.

## Goal

Skip the repo-root branch guard when the matched git verb sits inside a remote `ssh`
payload or inside a quoted `--note` / `--body` / heredoc argument, rather than
substring-matching the whole command line.

## Workflow gap

Four false-positive blocks, none of which would have moved the shared repo-root tree:

1. **#1689 (`5c5a89e8`) @ 2026-07-26T06:06:15Z** — `ssh pod-1689 '…git checkout -b
   issue-1689…'`, targeting the POD's repo.
2. **#1689 @ 06:06:21Z** — `ssh pod-1689 "…git reset --hard origin/issue-1689…"`, also
   pod-side. (Double-quoted, so outside the #1413 single-quoted allowance.)
3. **#1689 @ 06:13:46Z** — a `task.py post-marker --note "…"` whose prose merely
   *contained* the phrase `git reset --hard to origin/issue-1689 R8 HEAD`. **No git
   mutation of any kind was in the command.** Forced a rewrite to
   `--file /tmp/run-launched-v5.md`.
4. **`/daily` 2026-07-25 (this run) @ 2026-07-26T06:5xZ** — a
   `uv run python -c "…"` fingerprint helper whose **Python string literal** quoted the
   same destructive phrase as part of a bug description. Blocked; the entire compound
   was refused (the guard's own message notes earlier clauses do not run either), and
   the work had to be re-done via a `Write` + file-based script.

Three of the four are the same shape: the guarded verb appears in TEXT the command
carries (a marker note, a Python string, a remote payload) rather than in a git
invocation the local shell will run.

- **The guard already has this concept, partially.** Its own diagnostic documents a
  `#1566` masked shape for single-quoted `task.py` arguments, a `#1413` allowance for
  single-quoted multi-statement `ssh` payloads, a `#1538` waiver for grep pattern
  clauses, and a `#1463` allowance for `gcloud … --command='…'`. Each was added
  reactively for one shape. Occurrences 2, 3 and 4 fall in the residual: a
  double-quoted `ssh` payload, a double-quoted `--note`, and a Python string literal
  in a `-c` argument. The remedy is the same generalization the existing waivers
  approximate, not a fifth special case.
- **Confidence (emitter):** high that the four blocks were false positives; **medium**
  on the remedy — this is a security guard, and a general "it's inside a quoted
  argument" waiver is exactly the kind of relaxation that can open a real hole. The
  planner should weigh a narrower fix (e.g. extend the `--note`/`--body` masking to
  double quotes, and the `ssh` allowance to double-quoted payloads) against the general
  one, and MUST add pin tests for the shapes it does NOT intend to waive.
- verified-at-filing: the guard's current waiver set is quoted from its own live
  block message captured during this run (the `#1566` / `#1413` / `#1538` / `#1463`
  clauses), i.e. read from the running hook, not from the file's prose or from recall.
  Occurrences 1–3 are quoted from #1689's transcript (3 of the 5 `is_error: true`
  `tool_result` firing events in that session, counted as `tool_result` blocks with
  `is_error == true`, deduplicated per tool call). Occurrence 4 was observed directly
  by the filer. Landed-fix history check `git log --oneline --since='7 days ago' --
  scripts/guard_repo_root_branch.sh` → no commits in the window (the wave's
  guard work was on `guard_piped_git_push.sh`, #1675 `3e93592de4`). (2026-07-25)

## Proposed change (planner's call — narrow vs general)

```
  narrow (lower risk):
+ extend the existing #1566 note/title/prompt masking to DOUBLE-quoted arguments
+ extend the existing #1413 ssh-payload allowance to double-quoted payloads
+ add a masking clause for python/-c string literals

  general (higher leverage, needs careful pin tests):
+ tokenize the command and evaluate the guarded verb only when it is a COMMAND
+ position in a clause the local shell will execute — never inside a quoted
+ argument, a heredoc body, or a remote payload.
```

## Scope / surfaces

- Primary target: `scripts/guard_repo_root_branch.sh`.
- `tests/test_guard_repo_root_branch.py` — pin BOTH directions: the four false-positive
  shapes above must pass, and the genuine repo-root mutation shapes (bare
  `git checkout -b`, `git reset --hard`, `git clean -f`, root `merge`/`rebase`/
  `cherry-pick`/`revert`/`am`) must still block.
- **Note the paired-test sync hazard:** editing a hook without its pin test is the
  #1560 vintage-skew class that false-blocked four gates today; keep them in one
  commit.

## Constraints / invariants

- This is a SECURITY guard protecting the shared repo root from concurrent-session
  clobber (incidents 2026-06-01, #815, #841, #1090, #1193, #1234). Widening a waiver
  is the risky direction — every new waiver needs a negative pin test proving the
  guarded shape still blocks.
- Fail CLOSED on ambiguity: if the parser cannot confidently establish that the verb is
  inside a quoted/remote payload, it must still block.
- `scripts/workflow_lint.py --check-references` / `--check-asks` pass; ruff passes;
  `tests/test_guard_repo_root_branch.py` green.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/guard_repo_root_branch.sh
- fingerprint: 257796b1a49b
- Source: `/daily` 2026-07-25 transcript sweep, session `5c5a89e8` (#1689) @
  06:06:15Z / 06:06:21Z / 06:13:46Z, plus a live occurrence in the `/daily` run itself.
