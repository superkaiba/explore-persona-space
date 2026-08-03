---
title: 'workflow-fix: SLURM rsync lane strands committed eval_results inputs (extra-sync
  knob + lane-aware carry-over gate)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0ab6ae7a33a0
created_at: '2026-07-29T18:53:49Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised on #1689 (2026-07-29, fellows job 15188
  FAILED 1:0): the SLURM lane''s RSYNC_INCLUDE_PATHS/RSYNC_EXCLUDE_PATTERNS materialization
  excludes eval_results/, stranding plan-cited git-committed reference JSONs that
  verify_carryover_inputs had certified git-reachable; add a per-dispatch extra-sync-paths
  knob and make the gate lane-aware.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1689 (emitting agent: issue-orchestrator).

## Goal

Add a per-dispatch extra-sync-paths knob to the SLURM rsync lane and make
`verify_carryover_inputs` lane-aware, so plan-cited git-committed
`eval_results/` reference inputs actually reach rsync-materialized instances.

## Workflow gap

- **Bug observed:** fellows/SLURM lane rsync excludes `eval_results/`, so
  plan-cited git-committed reference inputs (parent ladder/percell JSONs)
  never reach the instance; `verify_carryover_inputs` PASSed them as
  git-reachable (the clone-lane contract) and the run crashed at first read
  (#1689 job 15188).
- **Why it is a workflow gap:** the carry-over gate (#1469, /issue Step 6a.5
  stanza 2) certifies inputs against `origin/issue-<N>` git-reachability —
  correct for the CLONE lanes (GCE `git clone`, RunPod bootstrap fetch) but
  vacuous for the SLURM lane, whose `build_rsync_command` copies only
  `RSYNC_INCLUDE_PATHS` (`./pyproject.toml ./uv.lock ./src ./scripts
  ./configs ./external/open-instruct ./tests ./data/sft`) with
  `eval_results/` in `RSYNC_EXCLUDE_PATTERNS`. A workload whose plan
  legitimately consumes a PRIOR round's committed `eval_results/issue_<N>/`
  references (a parity gate against published values, rung-eligibility
  indexes, per-pair n-asserts — the #1689 derived-vs-free battery does all
  three) passes every pre-dispatch gate and dies at first read on the
  instance (2026-07-29: fellows job 15188 FAILED 1:0, one full launch cycle
  burned; remediation was a hand rsync of
  `eval_results/issue_1689/{ladder,percell,analyzer}` into the shared-VAST
  scratch, which future FRESH scratch dirs would silently need repeated).
  This is the rsync-lane sibling of the #734/#1434 clone-lane class the gate
  was built for.
- **Confidence (emitter):** high (live incident, root cause read directly
  off the launch rsync argv + the crash traceback).
- verified-at-filing: `grep -rln "RSYNC_INCLUDE_PATHS" src/ scripts/ tests/`
  → 2 hits: `src/explore_persona_space/backends/slurm.py` (7 in-file
  occurrences — the constant + `build_rsync_command` consumer) and
  `scripts/issue1609_acceptance.py` (an acceptance harness echoing the
  constant; update only if the knob changes the argv shape it pins).
  Per-target: slurm.py presence CONFIRMED (the include/exclude assembly at
  ~L672-720); `scripts/verify_carryover_inputs.py` exists and its verdicts
  are lane-blind (no lane/backend parameter in its CLI — presence of the gap
  confirmed by reading its `--ref`-only contract) (2026-07-29).

## Proposed change (candidate diff sketch — refine in planning)

```
# backends/slurm.py
+ EXTRA_SYNC_PATHS_KEY = "extra_sync_paths"   # spec.extra channel
  def build_rsync_command(..., include_paths=RSYNC_INCLUDE_PATHS, ...):
+     # launch threads spec.extra["extra_sync_paths"] (dot-anchored,
+     # validated repo-relative) into include_paths ahead of the excludes
# scripts/dispatch_issue.py
+ --extra-sync-path (repeatable) → spec.extra["extra_sync_paths"]
# scripts/verify_carryover_inputs.py
+ lane-aware verdict: when the dispatch target is an rsync-materialized
+ lane (or --lane rsync passed), an in-ref eval_results/ citation that is
+ NOT covered by RSYNC_INCLUDE_PATHS ∪ extra_sync_paths downgrades from
+ PASS to FAIL(recoverable: add --extra-sync-path)
```

Planner notes: keep the knob additive (default argv byte-identical);
`--delete` interaction — staged extra paths must not be deleted on
subsequent launches that omit the knob (rsync excludes already protect
excluded trees from `--delete`; verify for included-then-omitted paths);
consider the same knob surfacing in plan §9 staging language + the /issue
Step 6a.5 gate prose. Pin with a `build_rsync_command` argv test + a
lane-aware verifier test.

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/slurm.py, scripts/verify_carryover_inputs.py`
- Secondary: `scripts/dispatch_issue.py` (flag threading),
  `scripts/issue1609_acceptance.py` (argv pin, only if shape changes),
  `.claude/skills/issue/SKILL.md` Step 6a.5 prose (one clause naming the
  rsync-lane contract).
- Grep the workflow surface for the pattern before editing
  (`grep -rn "RSYNC_INCLUDE_PATHS\|verify_carryover_inputs" src/ scripts/ tests/ .claude/`)
  and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard,
  `.claude/rules/workflow-fix-on-bug.md` § Recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/slurm.py, scripts/verify_carryover_inputs.py
- fingerprint: 0ab6ae7a33a0

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/backends/slurm.py, scripts/verify_carryover_inputs.py
bug_observed: fellows/SLURM lane rsync excludes eval_results/, so plan-cited git-committed reference inputs (parent ladder/percell JSONs) never reach the instance; verify_carryover_inputs PASSed them as git-reachable (the clone-lane contract) and the run crashed at first read (#1689 job 15188)
why_workflow_gap: the carry-over gate certifies git-reachability, the clone-lane materialization contract; the SLURM lane materializes via a fixed RSYNC_INCLUDE_PATHS set with eval_results/ excluded and exposes no per-dispatch input-staging knob, so a gate-green launch strands its own declared inputs
proposed_change: add a per-dispatch extra-sync-paths knob to the SLURM rsync lane and make verify_carryover_inputs lane-aware: an in-ref eval_results citation on an rsync-lane dispatch requires the path in the sync set
diff_sketch: |
  + spec.extra["extra_sync_paths"] -> build_rsync_command include_paths
  + dispatch_issue.py --extra-sync-path (repeatable)
  + verify_carryover_inputs: rsync-lane in-ref citation not in sync set -> FAIL(recoverable)
confidence: high
related_task: #1689
<!-- /workflow-fix-candidate -->
