---
title: 'workflow-fix: persistent-wedge bound on the poller zombie-override veto'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9e0a27a11c17
created_at: '2026-07-29T22:23:52Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate from #1768 orchestrator: veto suppression on
  _apply_zombie_override has no persistence bound — a 16h alive-but-wedged download
  kept the verdict running through posted gpu-idle escalation (fp 9e0a27a11c17)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a candidate raised on task #1768 (emitting agent: orchestrator, post-incident forensics on the 2026-07-29 ~16h download wedge).

## Goal

Add a persistent-wedge bound to `scripts/poll_pipeline.py`'s zombie-override veto stack: after N consecutive veto-suppressed ticks with the gpu-idle escalation posted and the main log frozen far beyond the stall window, the veto yields (a `stalled` verdict, or a distinct actionable wedge verdict) instead of resetting the streak forever.

## Workflow gap

- **Bug observed:** #1768 p2 unit `syc-pers-ft-po-s42` wedged inside a ~15 GB HF full-FT-checkpoint download (#931 family; workers futex-waiting, ~191 threads) from ~05:50Z. All 8 H100s sat at 0% with the main log frozen; the poller posted `[gpu-idle-advisory]` (07:04:13Z) and `[gpu-idle-escalation]` (07:32:54Z) — yet the poll VERDICT stayed `running` on every tick for ~16h (~$375 idle burn) until a manual SIGKILL of the workers (pids 96722/96725) surfaced the failure. The wedged workers were ALIVE and held ≥1 GiB GPU allocations, host-side dead-in-/proc (PID namespace).
- **Why it is a workflow gap:** the veto stack on `_apply_zombie_override` equates "alive, allocation-holding process" with "progressing workload." A network-download futex wedge keeps the workers alive + allocation-holding indefinitely, so the suppressing veto resets the zombie streak every tick with NO persistence bound — the running→stalled override (whose entire purpose is verdict-flipping on this class) becomes unreachable no matter how long the wedge lasts. The just-landed #1752 fix (commit `44c5804814`) is the MARKER-channel sibling (Nth-repeat escalation switches note KIND) and explicitly changes no verdict — the verdict channel remains unbounded.
- **Confidence (emitter):** medium — the #813/#864/#1216 false-positive history means N/K must be chosen conservatively; the spawned session's planner decides with the file + veto forensics open.
- `unverified hypothesis — verify at plan time:` WHICH veto term suppressed. The wedge shape (workers alive holding UVM allocations, host-unresolvable → `total > 0 AND resolvable == 0 AND alloc_holders > 0`) matches the #864/#1216 namespace veto, and a futex-waiting downloader accrues ~no CPU (so the #951 material-compute veto should have stayed inert) — but per-tick veto forensics were not captured before the workers were killed. Either way the gap is the same: no persistence bound on veto suppression.
- verified-at-filing: `grep -c 'persistent.wedge\|wedge_streak' scripts/poll_pipeline.py` → 0 hits (absence claim — the 0-hit in-target result is the evidence) (2026-07-29). Landed-fix history: `git log --oneline --since='7 days ago' -- scripts/poll_pipeline.py` → `44c5804814` (#1752, escalation-repeat KIND-switch — marker channel only, "NOTHING is stopped on either form"; read + judged distinct from this verdict-channel gap), `c341f3bd59`, `c2863252b5` (unrelated).

## Proposed change (candidate diff sketch — refine in planning)

```
+ scripts/poll_pipeline.py, in/near _apply_zombie_override:
+ - track veto_suppressed_streak per phase (state key, surviving run-scope
+   resets like gpu_idle_escalation_counts, #1752)
+ - when streak >= EPM_POLL_WEDGE_VETO_YIELD_TICKS (default ~8-12 ticks,
+   i.e. hours) AND gpu-idle escalation has been posted for this phase AND
+   main-log staleness >> stall window: the veto YIELDS — the override
+   proceeds (verdict stalled, stall_reason=persistent_wedge_veto_yield),
+   or at minimum a distinct verdict-adjacent wedge signal the
+   orchestrator/watcher treat as actionable
+ - kill switch EPM_POLL_WEDGE_VETO_YIELD=0; every degraded read fails
+   toward today's behavior (veto holds)
```

## Scope / surfaces

- Primary target: `scripts/poll_pipeline.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'zombie_streak\|namespace.*veto\|_apply_zombie_override' .claude/ CLAUDE.md scripts/`) and update every doc hit (the module docstring's veto contract, `.claude/rules/background-automation.md` if it names the veto); list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The #813/#826/#951/#1216 false-positive protections stay intact for the non-persistent case; the yield fires only on the long-persistence + escalation-posted + frozen-log conjunction.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/poll_pipeline.py
- fingerprint: 9e0a27a11c17

<!-- workflow-fix-candidate v1 -->
target_file: scripts/poll_pipeline.py
bug_observed: #1768 p2 unit wedged in a 15GB HF checkpoint download ~16h with all 8 GPUs idle and gpu-idle escalation posted while the poll verdict stayed running every tick — the veto reset the zombie streak indefinitely
why_workflow_gap: the zombie-override veto stack equates alive allocation-holding processes with progressing workloads; a futex download wedge keeps workers alive forever, so veto suppression has no persistence bound and the running->stalled override is unreachable
proposed_change: add a persistent-wedge override so the zombie-GPU namespace veto yields to a stalled or actionable wedge verdict after N consecutive suppressed ticks with gpu-idle escalation posted and a frozen main log
diff_sketch: |
  + per-phase veto_suppressed_streak state (survives run-scope resets);
  + at streak >= EPM_POLL_WEDGE_VETO_YIELD_TICKS with escalation posted +
  + frozen main log, the veto yields (stalled, stall_reason=persistent_wedge_veto_yield);
  + kill switch EPM_POLL_WEDGE_VETO_YIELD=0, degraded reads fail toward veto-holds
confidence: medium
related_task: #1768
<!-- /workflow-fix-candidate -->
