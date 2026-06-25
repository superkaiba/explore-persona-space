---
name: Merge-per-probe accumulation hits RunPod EDQUOT
description: A full-precision merged checkpoint per probed dose-ckpt / per read cell silently exhausts the ~130GB MooseFS per-pod quota mid-run; delete-on-read-return + re-merge-on-demand.
type: feedback
---

Any per-pod / per-node workload that materializes a full-precision merged
checkpoint (~15 GB for a 7B `merge_and_unload().save_pretrained()`) PER probed
dose checkpoint OR PER read cell must delete each merge the instant the read
consuming it returns. Accumulating them silently hits the RunPod MooseFS
~130 GB per-pod quota (`OSError errno=122 EDQUOT`) mid-run, often with NO
traceback (the process just exits after the next write EDQUOTs).

**Why:** #653 round 3 (epm:failure v5, 2026-06-24) — `phase_select_checkpoint`
called `_merge_adapter_for_read(ckpt, cell)` per probed dose checkpoint and never
deleted the previous one; 8+ merges = 192 GB on a 130 GB quota → silent death.
The same class re-bites one phase later if dx/install/ablation each leave their
selected-cell merge resident (12 cells x 15 GB = 180 GB).

**How to apply:**
- Wrap the read in `try/finally` and delete the just-probed merge in the
  `finally` (strict-immediate-delete → <=1 merge resident during a probe loop).
- Re-create the ONE eventually-needed merge (the selected checkpoint) ON DEMAND
  downstream — the resolver should re-derive it from the checkpoint DIR, never
  store/read a now-deleted merge path. Store `selected_model_path: None` and
  resolve from `selected_checkpoint_dir`.
- Per-cell cleanup at the END of each read phase's per-cell iteration
  (dx/install/ablation), guarded `if mode != CPU_STUB`; the next phase
  re-merges on demand (<=1 merge resident across the whole N-cell phase).
- NO-OP for full-FT cells: they read the FT checkpoint dir DIRECTLY (no merge is
  created) and that dir must never be deleted.
- Resume-skip: a re-entered selection/read phase MUST skip cells whose output
  manifest already exists (no re-probe, no re-merge) and sweep stale merges a
  crashed run left under them — else a relaunch redoes (and re-explodes) the
  completed cells.
- Plan-time: the planner's §9 compute table has no transient-disk dimension;
  surface a workflow-fix-candidate so merge-heavy read phases project peak
  transient disk = max_concurrent_merges x per_merge_GB against the ~130 GB
  per-pod quota.
