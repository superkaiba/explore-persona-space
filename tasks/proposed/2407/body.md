---
title: Step 6b canonical launch snippet is provision-only — add the required workload
  leg + time-budget guidance
kind: infra
tags:
- workflow-fix
- issue-skill
created_at: '2026-08-20T01:22:16Z'
has_clean_result: false
origin_prompt: '#2205 round-1 code review: the 10-step-6.md Step 6b snippet is provision-only
  (dies rc=2 verbatim; circular c46 remediation pointer)'
workflow: v1
---
## Goal

Fix the canonical Step 6b launch snippet in `.claude/skills/issue/steps/10-step-6.md` (lines ~731-733): it is provision-only — `dispatch_issue.py launch --issue <N> --intent "$INTENT" --repo-branch "issue-<N>" ${BACKEND:+--backend "$BACKEND"}` carries no `--workload-cmd`/`--hydra` and no `--time-budget-hours` — so copied verbatim it dies rc=2 at dispatch (`launch requires exactly one of --workload-cmd / --hydra`, #588), and once #2205's c46 arm lands, every plan embedding it verbatim draws the new plan-time WARN whose remediation text says to "copy the SKILL.md Step 6b launch snippet" — a circular pointer to the offending shape. Extend the snippet with the required workload leg (a `--workload-cmd '...'` placeholder or `--hydra k=v` alternative, commented) and the `--time-budget-hours` guidance the c46 drift arms expect on SLURM-reachable lanes.

## Why (incident)

Surfaced by the #2205 round-1 code review (2026-08-20): the reviewer confirmed the snippet is pre-existing-on-trunk provision-only, already fatal if copied verbatim, and inconsistent with the c46 WARN remediation pointer that names it as the compliant shape. The c46 test fixture `test_c46_placeholder_tokens_never_warn` historically modeled this snippet WITHOUT a workload leg — #2205 had to add one to the fixture, confirming the canonical snippet lags the CLI's own runtime requirement.

## Acceptance

- The Step 6b snippet in `.claude/skills/issue/steps/10-step-6.md` carries exactly one workload leg (a `--workload-cmd` placeholder with a one-line comment naming the `--hydra` alternative) and `--time-budget-hours` guidance consistent with the c46 drift arms (`verify_plan.py` `_c46_drift_arms`).
- The snippet dry-parses clean under c46 including the #2205 exactly-one-of arm (verify by embedding it in a synthetic plan and running `verify_plan.py --plan-file`).
- Any surrounding prose that describes the snippet's flag set is updated to match; no behavior change to `dispatch_issue.py` itself.

## Provenance

Surfaced in the #2205 round-1 Claude code-review verdict (session cmt0rstzvmuuoxw0u2g5m28sk, 2026-08-20); filed per .claude/rules/workflow-fix-on-bug.md (workflow-surface gap: .claude/skills/issue/steps/10-step-6.md canonical launch snippet).
