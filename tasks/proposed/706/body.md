---
title: /daily held-items route to PM-tracked tasks + PM digest line (stop the rot,
  add independent review)
kind: infra
tags:
- daily-held
created_at: '2026-06-28T07:13:55Z'
has_clean_result: false
origin_prompt: /daily 2026-06-27 held backlog
---
## Overview / Motivation

Filed from /daily 2026-06-27 held backlog: the /daily nightly retrospective routes held items to a gitignored log + one Telegram line where Thomas never looks, so held judgment-call and can't-verify items rot, and self-applied fixes get no independent review.

## Goal

Every /daily held item becomes a tracked task the PM surfaces + re-surfaces in the session Thomas actually uses; and the self-applied bucket stops being the only un-reviewed path.

## Problem (from /daily 2026-06-27)

The /daily nightly retrospective (cron 23:27 PT) runs reliably and already routes too-big fixes through /issue --auto (e.g. #699), BUT its output lands where Thomas does not look — a gitignored `logs/daily/<date>.md` marked `visible:false` + one my-goat Telegram digest line. Consequences: (a) HELD judgment-call + can't-verify items rot (06-27 alone held 6 judgment + 3 can't-verify); (b) the self-applied fixes get no independent review ("grades its own homework").

## Proposed change

Replace /daily's apply-vs-hold BINARY with a THREE-ROUTE classifier in `.claude/skills/daily/SKILL.md`:

1. Trivial mechanical (doc edits, string replaces, lint annotations — no behavior change) → self-apply + commit, as today (cheap, lint-gated, one git-revert away).
2. ANY behavior/logic change, including high-blast-radius REVERSIBLE fixes → file kind:infra via file_infra_task.py → /issue --auto (independent planner / adversarial critic ensemble / Claude+Codex code-review / test-verdict). Kills "grades its own homework" AND "only fixes small stuff": big reversible fixes stop being held and flow through review. (/daily already does this for the very biggest, e.g. #699 — make it the RULE for all behavior changes, not an ad-hoc judgment call.)
3. Genuine judgment call ONLY (the carve-out: destructive/irreversible, spends-money, external side-effects, result-interpretation, architectural/public-contract) → file a TRACKED proposed task tagged `daily-held needs-human`. No longer a dead-end log/Telegram line.

PM integration (load-bearing) in `.claude/agents/research-pm.md`:

- the `needs-human` tag must make the PM STATUS pass SURFACE the task to Thomas and RE-SURFACE per the ADHD-aware rule, WITHOUT auto-dispatching it (the PM's standing infra-auto-dispatch rule MUST skip `needs-human` tasks — else a judgment-call task gets auto-run, defeating the purpose).
- the PM STATUS readout gains a one-line digest: "/daily last night: applied N, filed M (→/issue), held J (needs you)".

## Scope / target files

- `.claude/skills/daily/SKILL.md`
- `.claude/agents/research-pm.md`
- possibly `.claude/skills/pm/SKILL.md`
- possibly `.claude/workflow.yaml` (register the daily-held / needs-human tag semantics)

## Constraints

- Workflow-surface only.
- Lint gate green (`workflow_lint.py --check-asks` + `--check-references`).
- Don't break the existing nightly sweep or the working big-fix→/issue path.
- Add a test pinning that PM auto-dispatch SKIPS needs-human tasks (that invariant is the whole point).

## Provenance
- workflow_fix_target: .claude/skills/daily/SKILL.md, .claude/agents/research-pm.md
- fingerprint: daily-held-pm-route
