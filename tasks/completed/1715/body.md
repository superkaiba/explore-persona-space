---
title: 'daily-fix: draft-PR recipe uses stale local main and omits -'
kind: infra
tags:
- wf-fix
- wf-fix-fp:70197c634340
- daily-auto-filed
created_at: '2026-07-27T07:14:10Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 2): the Step-4 draft-PR pre-check
  counts main..issue-N against the stale local main while worktrees are cut from fetched
  origin/main, gh pr create omits --title so it exits 1 non-interactively, and the
  site carries no never-pipe reminder'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 2). Surfaced by 5 independent
miner group(s) over the 2026-07-26 session transcripts.

## Goal

Fix the Step-4 draft-PR recipe in `.claude/skills/issue/SKILL.md` so its aheadness
pre-check compares against fetched `origin/main`, its `gh pr create` example carries
`--title`, and the site warns against piping the command.

## Workflow gap

- **Bug observed:** the Step-4 draft-PR block gates on `rev-list --count main..issue-<N>`
  against the repo root's STALE LOCAL `main` (worktrees are cut from fetched `origin/main`),
  so the guard reads a nonzero count on a branch with zero own commits and lets a doomed
  `gh pr create` fire — which itself exits 1 because the recipe omits `--title`, which
  non-interactive `gh` requires; and the site carries no "never pipe this" reminder, so
  sessions keep re-adding `| tail` and tripping `guard_piped_git_push.sh`.
- **Why it is a workflow gap:** all three are properties of the copy-paste recipe itself —
  a session that follows SKILL.md verbatim reproduces all three deterministically, while a
  session that improvises the `origin/main..` form does not.
- **Confidence (emitter):** high
- verified-at-filing: `Read .claude/skills/issue/SKILL.md` L2016-2026 (the named target
  block) — L2021 `if [ "$(git -C "$REPO_ROOT" rev-list --count main..issue-<N>)" -gt 0 ]; then`,
  L2022 `gh pr create --draft --head issue-<N> --body "Closes task #<N>."`.
  Per-target presence confirmed: `grep -n 'rev-list --count main\.\.' .claude/skills/issue/SKILL.md`
  → **1 hit, L2021** (the only bare `main..` aheadness guard in the file);
  `grep -c '\-\-title' <that block>` → **0** (absence-of-flag claim; the 0-hit IS the
  evidence); no `pipe`/`tail` warning anywhere in L2016-2026.
  Supporting facts verified: `grep -n 'origin/main\|base-local' scripts/new_worktree.sh` →
  L23-28 confirm new branches are cut from `refs/remotes/origin/main` (fetched), with
  `--base-local` the explicit opt-out — so local `main` is the WRONG base by construction;
  `grep -n 'gh pr' .claude/hooks/guard_piped_git_push.sh` → the hook's own BLOCKED message
  at L363 names `gh pr merge|create` explicitly. Landed-fix check:
  `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md` shows no commit
  touching the draft-PR recipe. (2026-07-26)

## Evidence

- Session `8571eca6`, 2026-07-26T10:51:16Z → 10:51:36Z: followed the recipe verbatim on
  `issue-1698`. The guard read `AHEAD=2` on a branch with zero own commits, so the
  `gh pr create` inside fired and died twice over —
  `Exit code 1 | AHEAD=2 | must provide --title and --body (or --fill …) when not running
  interactively`, then, after supplying a title,
  `pull request create failed: GraphQL: Head sha can't be blank, Base sha can't be blank,
  No commits between main and issue-1698` — i.e. exactly the error the guard exists to
  prevent. Cost: 2 wasted tool calls + 1 diagnostic call ≈ 1 turn. Session `06447a89` used
  the `origin/main..issue-1691` form instead and had no such failure; the two sessions
  diverged only because one followed the SKILL recipe verbatim.
- Session `8380a48c`, 2026-07-26T09:30:14Z: same shape on `issue-1696` with the diagnostic
  spelled out — `Exit code 1  must provide --title and --body (or --fill or fill-first or
  --fillverbose) when not running interactively`, followed by
  `rev-list count (main..issue-1696): 1` / `rev-list count (origin/main..issue-1696): 0`.
  The branch was byte-identical to `origin/main`; only the stale local ref made the guard
  pass. Cost: one failed `gh` call + a full diagnostic round (~1 min), self-recovered.
- Session `a2c4bae3` at 2026-07-26T15:19:54Z and session `0793d486` at 2026-07-26T11:09:04Z:
  the `--title` half alone, both from the verbatim recipe —
  `Exit code 1 | must provide \`--title\` and \`--body\` (or \`--fill\` or \`fill-first\`
  or \`--fillverbose\`) when not running interactively`. Counting: 2 tool_result FIRING
  events (text is the invoked `gh` command's own output, deduped per tool call, one per
  session). Each cost a wasted turn and forced re-opening the PR later; the guard's `else`
  branch never printed because the command died first.
- Session `67cf175e`, 2026-07-26T15:53:29Z: same `--title` failure, retried by hand with a
  written title. Four sessions, one recipe.
- Session `564d9a53`, 2026-07-26T07:31:38Z: the pipe half. The session added `| tail -3` to
  the recipe's `gh pr create` and `guard_piped_git_push.sh` correctly BLOCKED it:
  `"PreToolUse:Bash hook error: [.claude/hooks/guard_piped_git_push.sh]: BLOCKED: piping
  \`git push\` / \`git merge\` / \`git commit\` / \`gh pr merge|create\` through a filter
  masks the non-zero exit code…"`. Two aggravating details: SKILL.md gives the recipe
  UNPIPED, so the pipe is a composition habit the site does not warn against; and the
  piped call sat inside an `if` branch that was FALSE (`issue-1695 has no commits ahead of
  main yet`), yet the text-based PreToolUse guard blocked the whole compound call, taking
  an unrelated `session_progress_report.py` title refresh with it. Cost: one blocked call
  + one retry (~5 s).

## Proposed change

Rewrite `.claude/skills/issue/SKILL.md` L2016-2026 as:

```bash
# Base ref is FETCHED origin/main — new_worktree.sh cuts branches from
# refs/remotes/origin/main, and the repo root's local `main` routinely lags it.
# NEVER pipe this block — guard_piped_git_push.sh blocks a piped `gh pr create`
# (CLAUDE.md § Concurrent repo-root committers): a pipe masks the exit code.
timeout --kill-after=30s 120s git -C "$REPO_ROOT" fetch origin main --quiet || true
if [ "$(git -C "$REPO_ROOT" rev-list --count origin/main..issue-<N>)" -gt 0 ]; then
  gh pr create --draft --head issue-<N> \
    --title "issue-<N>: <task title>" \
    --body "Closes task #<N>."
else
  echo "issue-<N> has no commits ahead of origin/main yet; skipping draft PR (open it after the implementer commits)."
fi
```

- Change the aheadness test from `main..issue-<N>` to `origin/main..issue-<N>`, preceded by
  a bounded `git fetch origin main` (the same base ref every Step-10d guard already uses).
- Add `--title "issue-<N>: <task title>"` to the `gh pr create` example (`--fill` is an
  acceptable alternative but loses the issue-<N> title convention).
- Add the inline "never pipe this" comment naming `guard_piped_git_push.sh` at the recipe
  site, so a composer does not re-add a filter.
- Update the `else`-branch echo to say `origin/main` so the message matches the test.
- Grep the SKILL for any other bare `main..` / `main --` refs inside guards while making
  this change and bring them onto the same base ref (compose-time grep found only L2021,
  but the planner should re-verify at plan time in case another lands in between).

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- none

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `uv run python scripts/workflow_lint.py` passes (no-flags); ruff clean on touched files.
- The `fetch` must stay bounded and non-fatal (`timeout … || true`) — a network hiccup at
  Step 4 must not halt the pipeline; a stale-but-present `origin/main` still beats local
  `main`.
- Do not remove the aheadness guard itself; the `No commits between main and issue-<N>`
  failure it prevents is real, it was just checking the wrong ref.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 70197c634340

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: PENDING

/daily 2026-07-26 route-2 filing. Miner refs: E-P1, G-P5, B-P3, J-P4, I-P9.
