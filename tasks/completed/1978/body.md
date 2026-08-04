---
title: 'daily-fix: extract Step 10d guards to checked-in script'
kind: infra
tags:
- wf-fix
- wf-fix-fp:420fca5d2e09
- daily-auto-filed
created_at: '2026-08-01T07:10:35Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-31 problem sweep (route 2): Step 10d Guard 1-5 shell
  blocks are hand-retyped per session — a composed command carried the typo `--path-format:
  absolutre` + a stale issue-1713 /tmp template name (#1867); no scripts/step10d_guards.sh
  exists.'
workflow: v1
---
# daily-fix: extract Step 10d guards to checked-in script

## Overview / Motivation

Auto-filed by the /daily 2026-07-31 problem sweep (CONSOLIDATED L14; miner-8:P12). Source: session e42b6301 (#1867) — the session's hand-retyped Step 10d guard block carried the typo `--path-format: absolutre` plus a stale issue-1713 /tmp template name (probed by the miner from the composed command; the guard still ran clean via the fallback). Harmless in this instance, but the Step 10d Guard 1–5 shell blocks are multi-line recipes re-typed/adapted per session — exactly how typos and stale names creep into a merge-gating path.

## Goal

Extract the Step 10d Guard 1–5 shell blocks from `.claude/skills/issue/SKILL.md` into a checked-in `scripts/step10d_guards.sh <issue>` that sessions invoke, rather than re-typing the recipes per session.

## Workflow gap

- **Bug observed:** A session's hand-composed Step 10d guard command carried a typo (`--path-format: absolutre`) and a stale issue-1713-era /tmp template name; the composition only worked via the guard's fallback path.
- **Why it is a workflow gap:** The Guard 1–5 recipes live only as long multi-line shell prose inside `.claude/skills/issue/SKILL.md` (Guard 1 retry/pin logic ~lines 10660-10760, Guard 3 spec-freshness ~10858-10890, Guard 4 LOST-UPDATE refusal ~10957-10981, Guard 5 merge-hold ~10997+, plus repeated `REPO_ROOT=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")` preludes at 2064/9488/10564/12124). Every session re-types or adapts them, so each execution is a fresh transcription with fresh typo/staleness risk on the path that certifies merges to main. A checked-in script gives one tested implementation, one place to fix, and pin-testable behavior.
- **Confidence (emitter):** medium
- verified-at-filing: `ls scripts/step10d_guards.sh` → No such file or directory (absence claim — the 0-hit result is the evidence that no extracted script exists). `grep -n "Guard 1\|Guard 4\|Guard 5\|path-format" .claude/skills/issue/SKILL.md` → 25+ hits confirming the guard blocks live as SKILL.md shell prose (Guard 1 retry at 10664, Guard 4 refusal at 10957, Guard 5 at 10997; presence claim, spot context read). `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md` → 5+ commits (rule-23 diagnosis, Step 10d landing confirmation `a66935574c`, choom routing, rate/ETA duty, humanize verify) — the landing-confirmation commit extends Step 10d prose but extracts no script; no landed fix (2026-08-01 compose time). The `absolutre` typo itself was in the SESSION's composed command, not in SKILL.md (SKILL.md's own preludes spell `--path-format=absolute` correctly) — the gap is per-session retyping, not a SKILL.md typo.

## Proposed change (candidate diff sketch — refine in planning)

```
NEW scripts/step10d_guards.sh:
+ #!/usr/bin/env bash
+ # Step 10d Guards 1-5, extracted from .claude/skills/issue/SKILL.md.
+ # Usage: bash scripts/step10d_guards.sh <issue> [--guard N] [--worktree <path>]
+ # Prints one machine-parseable verdict line per guard
+ # (GUARD<k>: certified|not-certified|refused|no-op + reason) and exits
+ # nonzero on any blocking verdict; sessions branch on the printed states
+ # exactly as the current prose directs.

.claude/skills/issue/SKILL.md Step 10d:
- <the five multi-line shell blocks>
+ Run `bash scripts/step10d_guards.sh <N>` and branch on its verdict lines;
+ keep the per-guard SEMANTICS prose (what each state means, recovery
+ routes: merge-failed handling, artifact-confirmed merge, LOST-UPDATE
+ recipe) — only the retype-prone shell moves into the script.

NEW tests/test_step10d_guards.py: pin the verdict-line grammar + the
Guard 1 no-foreign-tasks-paths and Guard 4 refusal predicates on fixture
repos.
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 10d guard blocks) + NEW `scripts/step10d_guards.sh`
- Secondary: `tests/test_step10d_guards.py` (new pin tests), `scripts/workflow_lint.py` if a check references the inline blocks
- Grep before editing: `grep -rn 'Guard 1\|Guard 4\|merge-hold-candidate' .claude/skills/ scripts/` and account for every consumer of the guard prose (issue-v2 SKILL defers by section name — verify); list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- BEHAVIOR-PRESERVING extraction: every guard's predicate, retry, and refusal semantics stay byte-equivalent in effect (Guard 1's three-dot diff + bounded retry, Guard 3's spec-freshness split, Guard 4's LOST-UPDATE refusal, Guard 5's merge-hold) — this is likely a LARGE, merge-gating change; the planner should stage it (extract 1-2 guards first) and may flag `architectural: true` if it judges the Step 10d contract a public surface.
- The script must be safe under concurrent sessions (read-only probes; no repo-root mutations beyond what the current prose performs).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates.

## Provenance

- sha-verify (filing-time, #1467): `e42b6301` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- fingerprint: 420fca5d2e09

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: (driver-computed; tag authoritative)

Origin: CONSOLIDATED L14 (miner-8:P12), /daily 2026-07-31 — "Step 10d Guard 1–5 shell blocks are hand-retyped per session — typo `--path-format: absolutre` + a stale issue-1713 /tmp template name observed" (session e42b6301 / #1867).
