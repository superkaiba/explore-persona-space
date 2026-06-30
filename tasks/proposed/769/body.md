---
title: 'workflow-fix: add CUDA-allocator-fragmentation OOM gotcha + workload-cmd hot-fix
  recipe (#761 r3)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d9e873e78c09
created_at: '2026-06-30T18:29:55Z'
has_clean_result: false
origin_prompt: 'epm:failure-lesson v1 on #761 round-3 relaunch — gotcha_candidate:
  yes, root_cause_confirmed: yes, owning_agent: experimenter. workload-cmd-only fix
  (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True + --batch-probes 16→8) for Qwen-2.5-7B
  teacher-forced capture CUDA-allocator fragmentation OOM at lm_head; belongs in .claude/rules/gotchas.md
  alongside the existing CVD/parallel-cell/vLLM GPU traps.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an `epm:failure-lesson v1`
captured on task #761 round-3 relaunch (gotcha_candidate: yes,
root_cause_confirmed: yes, owning_agent: experimenter).

## Goal

Add a CUDA-allocator-fragmentation OOM gotcha to `.claude/rules/gotchas.md`
documenting the workload-cmd-only hot-fix recipe
(`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + halve `--batch-probes`)
for multi-layer teacher-forced activation capture on 7B-class models.

## Workflow gap

- **Bug observed:** A Qwen-2.5-7B teacher-forced answer-side capture (#761
  line) OOM'd on a GCP A100-80 at the `lm_head` after ~6000 successful
  forwards. The crash signature was unambiguous PyTorch CUDA-allocator
  fragmentation (39.5 GB live + 26.9 GB reserved-but-unallocated, no
  contiguous 12.54 GB slot) — NOT a true headroom shortfall. The error
  message itself recommended `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  but neither `.claude/rules/gotchas.md` nor any agent memory captured the
  recipe before the round-3 relaunch surfaced it.
- **Why it is a workflow gap:** gotchas.md is the canonical "loads when
  you touch training / eval / orchestrate code" rule per CLAUDE.md, and it
  ALREADY documents related GPU traps (the `+gpu_id=N` CVD clobber, the
  parallel-cell co-location OOM, the vLLM teardown / EngineCore zombie /
  rendezvous / hang traps). A capture-side allocator-fragmentation OOM
  belongs in the same set — it is a recurring class of NON-code-bug GPU
  failure that requires a workload-cmd-only fix, NOT a code change AND
  NOT an implementer bounce. Without the entry, a future experimenter
  hits the same crash and re-derives the fix from the PyTorch error
  message (or worse, mis-classifies it as a code bug and bounces to
  experiment-implementer).
- **Confidence (emitter):** high.

## Proposed change (candidate diff sketch — refine in planning)

Add a new bullet to `.claude/rules/gotchas.md` in the GPU/CUDA cluster
(near the existing CVD-clobber and parallel-cell OOM entries) along the
lines of:

```
- **CUDA-allocator fragmentation OOM on long teacher-forced multi-layer
  captures — workload-cmd hot-fix, NOT a code bug.** A multi-layer
  activation capture on a 7B-class model (e.g. Qwen-2.5-7B,
  `BatchedAnswerSpanCapture`-style hook) runs healthy through thousands
  of forwards then OOMs at the `lm_head` linear with the signature
  `Tried to allocate <N> GiB. GPU 0 has a total capacity of 79.25 GiB
  of which <small> GiB is free. ... <X> GiB is reserved by PyTorch but
  unallocated.` (reserved-but-unallocated > the request size).
  The PyTorch CUDA allocator has fragmented: enough total free, no
  contiguous block. The error message itself recommends the fix.
  RULE: hot-fix with TWO workload-cmd knobs, NO script edit:
  1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (env var,
     switches to expandable-segments allocator which defragments).
  2. Halve the per-forward batch (`--batch-probes 16→8` or equivalent;
     halves the `(B, T, H)` per-layer capture buffer + lm_head
     intermediate). Throughput hit is ~2× wall-clock per phase, right
     trade vs OOM. Both via `--workload-cmd` on the dispatcher OR the
     RunPod launcher wrapper. NEVER bounce to experiment-implementer
     for a code change — the capture script is correct; the allocator
     behavior is the issue. Cross-ref: long-form recipe at
     `.claude/agent-memory/experimenter/feedback_cuda_oom_expandable_segments.md`.
     (Incident #761 r3 relaunch, 2026-06-30: GCP A100-80 OOM at
     ~6000 forwards into a 150-capture sweep; manual GCP→RunPod
     failover with the hot-fix in workload-cmd.)
```

The planner should grep for any near-duplicate existing bullet and
collapse / merge if appropriate (the CVD-clobber bullet is the nearest
sibling but is about a different class — multi-process CVD pinning).

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing:
  `grep -rln 'expandable_segments\|PYTORCH_CUDA_ALLOC_CONF\|CUDA-allocator fragmentation' .claude/ CLAUDE.md`
  (the agent-memory entry just landed, so expect 1 hit there).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files
  passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent
  with the rule file. (This fix shouldn't need either — gotchas.md only.)
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of
  its own subagents' workflow-fix candidates (recursion guard, per
  `.claude/rules/workflow-fix-on-bug.md` § Recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: d9e873e78c09

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/gotchas.md
bug_observed: Qwen-2.5-7B teacher-forced capture OOM on A100-80 at lm_head after ~6000 forwards — PyTorch allocator fragmentation, no contiguous 12.54 GB slot, fix is workload-cmd-only
why_workflow_gap: gotchas.md is the canonical "loads when you touch training/eval/orchestrate code" rule and already documents related GPU traps (CVD clobber, parallel-cell OOM, vLLM teardown/zombie/rendezvous/hang). A capture-side allocator-fragmentation OOM with a workload-cmd-only hot-fix recipe belongs in the same set; absent the entry, the next experimenter re-derives it from the PyTorch error message and risks mis-classifying as a code bug.
proposed_change: add CUDA-allocator fragmentation OOM gotcha + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True + halve batch hot-fix recipe
diff_sketch: |
  see "Proposed change" section above (one new bullet near the CVD/parallel-cell GPU cluster, cross-referencing the just-landed experimenter agent-memory file feedback_cuda_oom_expandable_segments.md)
confidence: high
related_task: #761
<!-- /workflow-fix-candidate -->
