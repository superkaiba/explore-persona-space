---
title: 'daily-fix: thread --env-pin into Step 6b composition'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7b0deff20abc
- daily-auto-filed
created_at: '2026-07-26T07:06:53Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 2): The #1669 env-pin channel
  (--env-pin on dispatch_issue.py launch, merged d44a52d2fc) is inert unless launch
  composers pass the flag; the /issue Step 6b operational-dispatch block and the experimenter-brief
  conventions never mention it, so the #1586 incident class (failover pod loses the
  declared WandB project) recurs.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the `/daily` 2026-07-25 Step C parked-workflow-fix-candidate routing pass
(`.claude/rules/workflow-fix-on-bug.md` § Recursion guard escape valve). The candidate was
parked on task #1669 at 2026-07-25T11:11:11Z because that session ran under the
`workflow_fix_target` recursion guard and could not auto-route its own candidate.

## Goal

Add a Step 6b composition rule + experimenter-brief bullet that threads
`--env-pin KEY=VALUE` into the `dispatch_issue.py launch` command when the plan's
Reproducibility Card declares a non-default `WANDB_PROJECT` (or other
`ENV_PIN_ALLOWED_KEYS` value).

## Workflow gap

- **Bug observed:** the #1669 env-pin channel (`--env-pin KEY=VALUE` on
  `dispatch_issue.py launch`) is inert unless launch composers pass the flag. The
  `/issue` Step 6b operational-dispatch block and the experimenter-brief conventions
  never mention it, so the #1586 incident class (a failover pod loses the declared
  WandB project) recurs for any run whose project pin lives only in the plan's
  Reproducibility Card.
- **Why it is a workflow gap:** Step 6b is where launch commands are composed. A flag
  that exists only in `--help` is not part of the composition contract — the same
  adoption pattern as `--boot-disk-gb`, which DOES have a Step 6b / plan-row rule
  ("the flags are what arm the gate").
- **Confidence (emitter):** high
- verified-at-filing: `grep -c 'env-pin' .claude/skills/issue/SKILL.md` → **0 hits**
  (absence confirmed in the named target). Repo-wide semantic probe
  `grep -rln 'env-pin\|ENV_PIN_ALLOWED_KEYS' scripts/ src/ .claude/` → the channel is
  live in `scripts/dispatch_issue.py` (`_parse_env_pins`, line 1295),
  `scripts/workflow_lint.py`, `src/explore_persona_space/backends/base.py`
  (`ENV_PIN_ALLOWED_KEYS`, line 100) and `backends/gcp.py`, with **no** mention in any
  `.claude/skills/**` composition surface. Landed-fix history check
  `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md` → 8 commits,
  none touching env-pin composition. SHA check:
  `git rev-parse --verify --quiet 'd44a52d2fcd^{commit}'` resolves to
  `d44a52d2fcd132943ac2cb2b3da8133ad6edaa7f` — "task #1669: failover re-provision carries
  launch env pins (#1445)", 2026-07-25 04:09:40 -0700. (2026-07-25)

## Proposed change (candidate diff sketch — refine in planning)

```
+ Step 6b (operational dispatch): when the plan's Reproducibility pod row /
+ env pins declare WANDB_PROJECT (or another ENV_PIN_ALLOWED_KEYS key), pass
+ `--env-pin WANDB_PROJECT=<value>` on `dispatch_issue.py launch` (repeatable;
+ workload-cmd lanes only) — the pin persists to the handle sidecar and the
+ failover reconstructors re-export it on the fresh pod (#1669/#1586).
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 6b operational-dispatch block).
- Also consider the experimenter-brief conventions (`.claude/agents/experimenter.md`) —
  grep the workflow surface for the `--boot-disk-gb` adoption rule and mirror its shape
  and placement, since that is the named precedent.
- Read `scripts/dispatch_issue.py::_parse_env_pins` and
  `backends/base.ENV_PIN_ALLOWED_KEYS` for the exact allowlist + repeatability semantics
  before writing the rule; do not restate the allowlist inline if it can drift.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` and `--check-references` pass; ruff on touched
  files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 7b0deff20abc

Parked candidate (verbatim), from task #1669 `events.jsonl` @ 2026-07-25T11:11:11Z:

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md
bug_observed: The #1669 env-pin channel (--env-pin KEY=VALUE on dispatch_issue.py launch, merged d44a52d2fcd) is inert unless launch composers pass the flag; the /issue Step 6b operational-dispatch block and the experimenter-brief conventions never mention it, so the #1586 incident class (failover pod loses the declared WandB project) recurs for any run whose project pin is only in the plan's Reproducibility Card.
why_workflow_gap: Step 6b is where launch commands are composed; a flag that exists only in --help is not part of the composition contract (same adoption pattern as --boot-disk-gb, which HAS a Step 6b/plan-row rule: "the flags are what arm the gate").
proposed_change: Add one Step 6b composition rule + one experimenter-brief bullet: when the plan's Reproducibility Card declares a non-default WANDB_PROJECT (or other ENV_PIN_ALLOWED_KEYS value), thread `--env-pin KEY=VALUE` into the dispatch_issue.py launch command so the pin persists into the handle sidecar and survives failover re-provision.
confidence: high
related_task: #1669
<!-- /workflow-fix-candidate -->
