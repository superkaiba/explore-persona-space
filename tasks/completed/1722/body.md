---
title: 'daily-fix: Step 10d durable markers record wrong SHA and emp'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5b77e91f3661
- daily-auto-filed
created_at: '2026-07-27T07:16:08Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 2): epm:merged recorded a sibling
  task''s merge SHA because the recipe reads git log -1 origin/main, a broken nested
  command substitution left a marker field silently empty, and the canonical Step-0
  read raises IndexError on a marker with an empty note'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 2). Surfaced by 4 independent
miner group(s) over the 2026-07-26 session transcripts.

## Goal

Give Step 10d an authoritative merge-SHA derivation, require the `--file` form for
note-bearing markers whose prose embeds command substitutions, and make the canonical
Step-0 task-state read total on an empty marker note.

## Workflow gap

- **Bug observed:** Three ways the Step 10d / Step 0 recipes produced wrong or empty durable records on 2026-07-26 — an `epm:merged v1` recording a SIBLING task's merge SHA, a broken nested command substitution leaving a durable marker field silently blank, and the canonical Step-0 events read raising `IndexError` on a marker with an empty note (2 firings in 90 seconds, one cancelling a parallel `CronCreate`).
- **Why it is a workflow gap:** Step 10d tells the session to post `epm:merged v1` "with the list of merge SHAs" but supplies NO derivation recipe, so each session improvises one — and one improvisation read the shared `origin/main` tip, which concurrent sessions move; separately the committed one-liner at SKILL.md L6155 indexes `[0]` into a possibly-empty `splitlines()`.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'MERGE_SHA\|merge_sha\|mergeCommit' .claude/skills/issue/SKILL.md` → 0 hits (absence-of-recipe: the Step 10d success bullet at L11273 reads "post `epm:merged v1` with the list of merge SHAs plus `merge_form:` … and `merge_attempts:`" and names no derivation); `grep -rn 'mergeCommit' .claude/ scripts/` → 0 hits repo-wide outside worktrees (the authoritative `gh pr view --json mergeCommit` call appears nowhere in the workflow surface). `grep -n 'splitlines()\[0\]' .claude/skills/issue/SKILL.md` → 1 hit, L6155: `print(e["ts"], e["kind"], (e.get("note") or "").splitlines()[0][:140])` (the miners cited L6154; the live line is 6155). `grep -n 'nested \$(\|heredoc' .claude/skills/issue/SKILL.md` → 0 hits bearing on `--note` composition. `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md` → 12 commits, none touching any of the three sites (2026-07-26)

## Evidence

- Session `06447a89`, 09:18:56Z: the session derived the merge SHA with `git -C "$REPO_ROOT" log -1 --format=%H origin/main` after `sync_repo_root.py`; because sibling sessions merge concurrently, the tip at that instant was task #1692's merge commit. The durable marker recorded `SUBJECT: task #1692: epm:merged — Merge landed.` under `MERGE_SHA=8426f64796…` for task #1691. The session noticed only because the subject line named the wrong task, and posted a v2 correction: `"**merge_sha: 262c69ade1…** (correcting v1 which quoted a sibling task #1692's merge commit that happened to be at origin/main tip …)"`. The correction used `gh pr view <PR> --json mergeCommit`, which was available all along.
- Same session, same command: the SHA substitution was embedded as `$(git -C \"$REPO_ROOT\" log -1 --format=%H origin/main)` inside an already-double-quoted `--note "…"`. The backslash-escaped quotes reached git literally — `"fatal: cannot change to '\"/home/thomasjiralerspong/explore-persona-space\"': No such file or directory"` (1 firing event) — so the substitution produced empty output and the marker's "verdict sha-bound to " field shipped blank. A `fatal:` line inside a Bash result is not a nonzero exit here, so nothing flagged it.
- Session `6b3fca14`, 07:07:44Z and 07:08:39Z: the Step-0 events read raised `"IndexError: list index out of range"` twice in the session's first 90 seconds (2 firing events), and the second firing cancelled a parallel `CronCreate` for the `/issue-tick 1693` backstop — `"<tool_use_error>Cancelled: parallel tool call Bash(uv run python scripts/task.py view 1693 …) errored</tool_use_error>"` — which had to be re-issued.
- Session `a2c4bae3`, 14:53:56Z: the same expression failed on that session's FIRST tool call — `"IndexError: list index out of range … === recent events (last 15) ===" (empty)` — so the session started blind to its own event history and had to re-query.
- Measured cost is small per firing (~1 min plus a re-issued cron arm; one re-query) but the failure sites are the session's first state read and its last durable provenance record. On a less observant run the sibling's SHA would have stood as #1691's permanent merge record.

## Proposed change

- `.claude/skills/issue/SKILL.md` Step 10d success bullet (L11273): add the authoritative derivation — `gh pr view <PR> --json mergeCommit -q .mergeCommit.oid` — and state that a tip read (`git log -1 origin/main`) is NOT valid, because concurrent sessions move the shared tip between the merge and the read. This is an ADDITION, not a replacement: no derivation recipe exists at that site today.
- Same site: require the marker's own subject/task id to be cross-checked against the derived SHA's commit subject before posting, so a foreign SHA is caught at post time rather than by eye.
- `.claude/skills/issue/SKILL.md` Step 10d and the CLAUDE.md § Task Workflow API `post-marker` guidance: for any note-bearing marker whose prose quotes git verbs or embeds a command substitution, prefer `post-marker --file <path>` (the flag is `--file`, mutually exclusive with `--note`) — building the note body in a file with every `$( )` resolved into a shell variable FIRST. Never nest a command substitution inside an already-quoted `--note "…"`. The `--file` form additionally sidesteps the repo-root guard's prose matching on git verbs.
- `.claude/skills/issue/SKILL.md` L6155: replace `(e.get("note") or "").splitlines()[0][:140]` with the total form `((e.get("note") or "").splitlines() or [""])[0][:140]`, and apply the same total form to the Step-0 task-state read recipe so the copied one-liner is safe wherever it is pasted.
- Grep the workflow surface for any sibling copy of the `splitlines()[0]` idiom before landing — the live SKILL.md carries exactly one, but the recipe is copied by hand into session shell and may have been mirrored into another spec.
- Add a pin test that the Step-0 enumerator one-liner handles an event whose `note` is empty or whitespace-only without raising.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- `CLAUDE.md` (§ Task Workflow API `post-marker` guidance — the `--file` preference for command-substitution-bearing notes)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `uv run python scripts/workflow_lint.py` passes (no-flags); ruff clean on touched files.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- sha-verify (filing-time, #1467): `06447a89` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- fingerprint: 5b77e91f3661

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: PENDING

/daily 2026-07-26 route-2 filing. Miner refs: E-P9, E-P10, D-P4, B-P14.
