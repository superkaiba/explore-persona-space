---
title: 'workflow-fix: document VM earlyoom silent-SIGTERM + stream-reduce recipe in
  gotchas.md'
kind: infra
tags:
- wf-fix
- wf-fix-fp:vm-earlyoom-bulk-tensor-load-no-traceback
created_at: '2026-06-30T09:59:31Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate from #658 r5: earlyoom killed bulk tensor load
  (33GB RSS) at fleet 9.4% mem avail with no Python traceback'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised by the experiment-implementer on task #658 (round-5 fix after an earlyoom SIGTERM during the r_B build's all-at-once tensor load).

## Goal

Document in `.claude/rules/gotchas.md` that the EPS VM runs `earlyoom` (Ubuntu's early-OOM daemon) which SIGTERMs the highest-`badness` process when free memory drops below 10% — and this triggers on bulk in-memory tensor loads (per-rollout activations, per-step checkpoints) without any Python traceback. Recipe: stream-reduce when loading N tensors where N × tensor_size > 10GB on a shared VM.

## Workflow gap

- **Bug observed:** task #658 9a-ter round-4 — the fit's r_B build loaded all 12000 PV activation tensors (each ~3.2MB = ~38GB total) into memory at once. At peak fleet memory pressure (9.4% avail / 128GB total VM), earlyoom selected our process by `badness 1138` and SIGTERMed it. No Python traceback (earlyoom doesn't give Python a chance). The log just stops mid-progress.
- **Why workflow gap:** `.claude/rules/gotchas.md` documents fleet-VM disk-full + uv-cache + various HF Hub traps but says NOTHING about (a) the existence of earlyoom on this VM, (b) the 10%-of-128GB trigger threshold, (c) the silent-no-traceback failure mode, (d) the stream-reduce recipe for bulk tensor loads. Every workflow that loads N×3MB tensors at scale will re-hit this.
- **Confidence:** high (this is a confirmed live incident; the fix is generalizable to all HF-backed bulk-tensor-load workflows).

## Proposed change

Add a single bullet to `.claude/rules/gotchas.md`'s VM/runtime section:

```
- **VM `earlyoom` SIGTERMs bulk in-memory tensor loads silently (no Python traceback).**
  The EPS VM runs Ubuntu `earlyoom` which sends SIGTERM to the highest-`badness`
  process when free memory drops below 10% (~12.8GB of 128GB total). Scripts
  loading bulk tensor sets (per-rollout activations, per-step checkpoints) into
  memory at once trip this under fleet pressure; the process dies WITHOUT a
  Python traceback (earlyoom interrupts before the SIGTERM handler runs).
  Recipe for bulk tensor reductions (e.g. diff-of-means, layer averages,
  PCA): stream-reduce — iterate per-rollout, accumulate sums + counts in a
  fixed-size buffer (L × D × 4 bytes ≈ 460KB for typical activations); peak
  RSS stays O(one-tensor) regardless of N. Verify with `journalctl --since X
  | grep earlyoom` when a workload dies silently. Incident: #658.
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Verify no existing reference: `grep -rln "earlyoom\|VmRSS\|SIGTERM.*memory\|10%.*memory" .claude/ CLAUDE.md scripts/`

## Constraints / invariants

- Workflow-surface only — documentation addition.
- `scripts/workflow_lint.py` passes.
- Recursion guard: this session is NOT a workflow-fix session (no `workflow_fix_target:` Provenance line).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: vm-earlyoom-bulk-tensor-load-no-traceback

<!-- workflow-fix-candidate v1 -->
target_file: .claude/rules/gotchas.md
bug_observed: task #658 r5 — the fit's r_B build loaded ~12000 PV activation tensors (~38GB total) into memory at once; earlyoom SIGTERMed the process at VmRSS 33.1GB under fleet pressure (mem avail 9.4%) with no Python traceback; the log just stops mid-progress.
why_workflow_gap: gotchas.md documents disk-full + HF Hub traps but nothing about earlyoom on the VM, its 10%-trigger, its silent-no-traceback failure mode, or the stream-reduce recipe for bulk tensor loads. Every load-many-tensors workflow re-introduces this.
proposed_change: add a gotchas.md bullet naming earlyoom's 10%-of-128GB SIGTERM threshold, the no-traceback failure mode, and the stream-reduce recipe (per-tensor accumulator into a fixed L×D buffer; verify via journalctl earlyoom).
diff_sketch: |
  + - **VM `earlyoom` SIGTERMs bulk in-memory tensor loads silently (no Python
  +   traceback).** Ubuntu earlyoom on this VM SIGTERMs the highest-badness process
  +   when free memory drops below 10% (~12.8GB of 128GB). Bulk tensor loads
  +   (per-rollout activations) trip this under fleet pressure with no Python
  +   traceback. Recipe: stream-reduce (per-tensor accumulator into L×D×4B buffer;
  +   peak RSS O(one-tensor)). Verify via `journalctl --since X | grep earlyoom`.
  +   Incident #658.
confidence: high
related_task: #658
<!-- /workflow-fix-candidate -->
