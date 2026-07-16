---
title: 'workflow-fix: gotchas entry — rank-space bootstrap tail-mass gating (float-space
  CI coverage epsilon-fragile)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:92554ba1f495
created_at: '2026-07-16T17:36:07Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate: yes failure-lesson from #825 r11 rev4 (epm:failure-lesson
  v11): strict float-space CI coverage (lo < point < hi) is epsilon-fragile — interpolated
  bootstrap percentiles land 1e-8..1.6e-7 above the point on collapsed small-n distributions;
  gate on rank-space bootstrap tail mass (> alpha/2 strictly each side of a same-GEMM
  identity-resample anchor) instead.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes` failure-lesson
raised on task #825 (emitting agent: experiment-implementer; orchestrator routed per
`/issue` Step 7 failure-lesson capture action 3).

## Goal

Add a gotcha entry to `.claude/rules/gotchas.md` documenting the epsilon-fragility of
float-space bootstrap-CI gating and the rank-space tail-mass replacement.

## Workflow gap

- **Bug observed:** a mechanized parity gate using strict float-space CI coverage
  (`lo < point < hi`) mis-classified degenerate bootstrap cells as gating because
  interpolated percentiles land 1e-8..1.6e-7 above the point estimate, causing
  deterministic pipeline aborts (#825 r11 rev 3, code-review v26 FAIL at t=24/25/29;
  n_gating=25, n_fail=3).
- **Why it is a workflow gap:** `.claude/rules/gotchas.md` is the codebase-wide catalog of
  numerical/statistical traps; this class (float-space CI-coverage tests, and any
  >=alpha/2 tail test against a point computed by DIFFERENT arithmetic — the
  identity-resample tie cluster, mass ~n!/n^n, float-jitters at ~1e-7) recurs in any
  bootstrap-gated pipeline and is documented nowhere in the rules surface.
- **Confidence (emitter):** high (root_cause_confirmed: yes; fix validated by replay on the
  real archived table + 12/12 unit tests; code-review v27 PASS)
- verified-at-filing: `grep -n "CI coverage\|tail.mass\|rank.space\|percentile" .claude/rules/gotchas.md` → 0 hits (2026-07-16); `grep -c bootstrap .claude/rules/gotchas.md` → 3 (none about CI-coverage gating — eyeballed); `git log --oneline --since='7 days ago' -- .claude/rules/gotchas.md` → 5 commits, none matching (EXDEV, hf-xet, artifact-reuse, bf16, pid-namespace).

## Proposed change (candidate diff sketch — refine in planning)

```
+ **Float-space bootstrap-CI gating is epsilon-fragile — gate in rank space.** A strict
+ coverage test (`lo < point < hi`) mis-gates collapsed small-n bootstrap cells because
+ interpolated percentiles land 1e-8..1.6e-7 ABOVE the point; any >=alpha/2 tail test
+ against a point computed by DIFFERENT arithmetic inherits the fragility (identity-
+ resample tie cluster, mass ~n!/n^n, jitters ~1e-7 vs a float64/centered point). Gate:
+ strictly MORE than alpha/2 of finite draws strictly on EACH side of the identity-
+ resample anchor computed through the SAME vectorized expression as the draws (bitwise
+ tie-exact); alpha = the CI's own level. Annotate per-node tail fractions; fail loud on
+ nodes lacking them. Worked example: scripts/issue825_turndyn_fit.py _gc_verdict
+ (#825 r11 rev 4, commits 5f1ffd1dea+9ed2930464; agent-memory
+ feedback_rank_space_bootstrap_tail_gating.md).
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'lo < point < hi\|CI coverage' .claude/ CLAUDE.md scripts/`) and update every
  hit; list them in the plan. (The superseded agent-memory entry
  `feedback_small_cell_bootstrap_ci_degeneracy.md` is already annotated — do not re-edit.)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 92554ba1f495

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: fit_gc (issue825_turndyn_fit.py G-C parity verdict)
lesson: Strict float-space CI coverage (lo < point < hi) is itself epsilon-fragile: interpolated bootstrap percentiles land 1e-8..1.6e-7 ABOVE the point on collapsed small-n distributions (and any >= alpha/2 tail test against a point computed by DIFFERENT arithmetic inherits the same fragility — the identity-resample tie cluster float-jitters to a random side of a float64/centered point at ~1e-7). Gate on rank-space bootstrap tail mass instead: strictly more than alpha/2 of the finite draws strictly on each side of the identity-resample anchor computed through the SAME vectorized expression as the draws (bitwise tie-exact), alpha = the CI's own level.
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
supersedes: the 2026-07-16T15:54:13Z lesson (v10) on gating by strict CI coverage
<!-- /epm:failure-lesson -->
