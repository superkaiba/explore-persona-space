---
title: 'workflow-fix: gotchas entry — chained waves on a detached-spawn launcher fan
  out concurrently'
kind: infra
tags:
- wf-fix
- wf-fix-fp:65a89f7ab41b
created_at: '2026-07-28T16:18:20Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate failure-lesson from #1738 experimenter (moosefs wedge;
  launcher setsid-detach chain trap)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a gotcha_candidate failure-lesson raised on task #1738 (emitting agent: experimenter).

## Goal

Add a gotchas.md entry: a setsid-detaching fan-out launcher exits after its spawn loop, so &&-chained sequential "waves" fan out CONCURRENTLY; verify a wait/poll loop in the launcher SOURCE before chaining, and treat "the launcher waits on its pids" brief claims as paraphrase until source-verified. Doubled concurrent FUSE load on a RunPod MooseFS mount is a plausible wedge trigger (#779 signature).

## Workflow gap

- **Bug observed:** #1738 waves 2-4 were &&-chained on the premise the launcher waits on shard pids; the launcher setsid-detaches and exits, 16 shards fanned onto 8 GPUs concurrently, and the pod's MooseFS mount wedged (statfs hang, request_wait_answer, GPUs 0 MiB) — launch dead, pod swap required.
- **Why it is a workflow gap:** gotchas.md § vLLM/MooseFS entries carry the wedge signature but not the chained-detached-launcher trigger class; nothing on the workflow surface tells a brief-composer or experimenter to source-verify wait semantics before chaining waves.
- **Confidence (emitter):** high (root_cause_confirmed: yes)
- verified-at-filing: `grep -rn 'setsid' .claude/rules/gotchas.md` → hits describe the detached VM-phase launch shape only; no chained-wave/wait-semantics entry (2026-07-28); per-target: .claude/rules/gotchas.md 0 hits for 'chain' near 'launcher'.

## Proposed change (candidate diff sketch — refine in planning)

+ gotchas.md new entry under the MooseFS/launcher family:
+ "Chained waves on a detached-spawn launcher: a fan-out launcher that setsid-detaches shards exits after its spawn loop — `&&`-chaining sequential waves runs them CONCURRENTLY (16 shards on 8 GPUs, #1738), and the doubled FUSE load can wedge the MooseFS mount (#779 signature). Verify a wait/poll loop in the launcher SOURCE before chaining; gate waves on shard-pid death or dispatch one wave per launch."

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'setsid' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 65a89f7ab41b

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/gotchas.md
bug_observed: "&&-chained launcher waves fanned out concurrently (launcher setsid-detaches and exits after spawn loop); 16-shard concurrent FUSE load wedged the pod MooseFS mount (#1738 waves 2-4)"
why_workflow_gap: gotchas.md carries the MooseFS wedge signature but not the chained-detached-launcher trigger class or the source-verify-wait-semantics rule
proposed_change: add the chained-waves-on-detached-launcher gotcha entry (verify wait loop in source; gate waves on shard-pid death; one wave per dispatch)
diff_sketch: |
  + gotchas.md: "Chained waves on a detached-spawn launcher ... (#1738/#779)" entry
confidence: high
related_task: #1738
<!-- /workflow-fix-candidate -->
