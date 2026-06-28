---
title: 'workflow-fix: gotchas.md — per-cell vLLM rendezvous-port EADDRINUSE trap'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9a0b8e84075f
created_at: '2026-06-28T19:45:02Z'
has_clean_result: false
origin_prompt: 'prose follow-up from #664 deep-dive: gotchas.md lacks the sequential-per-cell
  vLLM rendezvous-port EADDRINUSE collision trap (#664 shard-7 p2, distinct from existing
  teardown/orphan/deadlock entries)'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on
task #664 (emitting agent: general-purpose deep-dive diagnostic, PM-session
routed). #664's shard-7 p2 eval died with a `torch.distributed.DistNetworkError
... EADDRINUSE port 37609` because a prior per-cell vLLM EngineCore on that GPU
had not released the rendezvous port before the next per-cell engine tried to
bind it. gotchas.md already documents three vLLM traps (teardown-OOM,
crash-orphan, generate()-deadlock) but NOT this per-cell rendezvous-port
collision.

## Goal

Add a `.claude/rules/gotchas.md` entry documenting the sequential-per-cell vLLM
rendezvous-port EADDRINUSE collision trap, with the prevention (reap the engine
/ `destroy_process_group()` between same-GPU cells, or randomize/free the
rendezvous port before each cell's engine init) and the diagnostic signature.

## Workflow gap

- **Bug observed:** Sequential per-cell vLLM engines on the same GPU shard
  collide on the torch.distributed rendezvous port (EADDRINUSE) when a prior
  per-cell worker has not released it — #664 shard-7 p2 died with
  `DistNetworkError EADDRINUSE port 37609`.
- **Why it is a workflow gap:** This is a distinct trap from the three vLLM
  entries already in gotchas.md (teardown-OOM, crash-orphan EngineCore, and the
  large-batch generate() deadlock). The next sequential-per-cell fleet job will
  re-hit it with no documented prevention. `.claude/rules/gotchas.md` is the
  canonical home for this class of guidance.
- **Confidence (emitter):** low (borderline: the actual code fix likely lives in
  the per-cell engine-lifecycle of an experiment script such as
  `scripts/issue664_eval.py`, which is OUT of the workflow surface — but the
  GUIDANCE note belongs in the shared gotchas rule, which is in-scope. The
  spawned session's planner makes the call and may scope the note to "document
  the trap + point at the engine-lifecycle helper" rather than touching
  experiment code.)

## Proposed change (candidate diff sketch — refine in planning)

```
+ - **Sequential per-cell vLLM engines on the SAME GPU can collide on the
+   torch.distributed rendezvous port → EADDRINUSE.** When a per-cell loop
+   creates and tears down a vLLM engine per cell on one GPU shard, the next
+   cell's EngineCore can fail to bind the rendezvous port if the prior worker
+   has not released it: `torch.distributed.DistNetworkError ... EADDRINUSE
+   port <P>`. Distinct from the teardown-OOM / crash-orphan / generate()-hang
+   traps above. RULE: between sequential same-GPU cells, ensure
+   destroy_process_group() + engine reap completes (see _reap_vllm_engine),
+   OR set a fresh/randomized MASTER_PORT (rendezvous port) per cell so a
+   lingering bind cannot block the next. Recovery: a single scoped re-run of
+   the failed cell/shard clears it (as #664 shard-7 did). (Incident #664 r18
+   p2, 2026-06-28.)
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep before editing: `grep -niE 'eaddrinuse|rendezvous|master_port|destroy_process_group|vllm' .claude/rules/gotchas.md` and place the new entry alongside the existing vLLM-teardown cluster (do not duplicate them).
- If the planner judges a code-side prevention belongs in an experiment script (e.g. a per-cell MASTER_PORT randomization helper), that is OUT of the workflow surface — document the trap + point at where the fix lives; do not edit experiment code in this task.

## Constraints / invariants

- Workflow-surface only — `.claude/rules/gotchas.md` is in scope; experiment scripts under `scripts/` (except the workflow-helper list) are NOT.
- `scripts/workflow_lint.py --check-asks` passes; if any helper script is touched, ruff passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 9a0b8e84075f

Surfaced prose follow-up (verbatim from the #664 deep-dive):

> `.claude/rules/gotchas.md` § vLLM teardown / rerun discipline — the
> EADDRINUSE-on-per-cell-vLLM-restart collision (port not released between
> sequential cells on the same GPU shard) is a *distinct* trap from the
> documented "vLLM generate() hang" and the "zombie CUDA allocation" entries
> #664 already hit at r8. A one-line addition ("sequential per-cell vLLM engines
> on the same GPU can collide on the torch.distributed rendezvous port →
> EADDRINUSE; ensure destroy_process_group()/engine reap between cells or
> randomize the port") would save the next fleet job a rerun. Target:
> `.claude/rules/gotchas.md`.
