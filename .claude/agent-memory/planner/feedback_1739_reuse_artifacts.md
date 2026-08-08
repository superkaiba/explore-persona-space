---
name: 1739 ctxmap reuse artifacts
description: "#1739 reuse base: capture stores carry context_end+prefix_end+t1 per-rollout summaries (extraction-point variants need NO re-capture); maps (linear+mlp+kernel) at issue1739_ctxmap/analysis_tensors/maps/; proven staging = issue1739_leg2.sh; measured walls: wide-nomlp transfer ~23 s/unit, nlmap transfer 28.81 s/unit, map_fit 2651 s/key"
type: feedback
---

For any follow-up round on #1739 (context→answer map line), the reuse base is:

- **Capture stores** (HF `issue1739_ctxmap/capture_store/<b>_{extraction,labeling}.tar`,
  3 behaviors × 2): each carries per-rollout summaries in THREE kinds —
  `prefix_end`, `context_end`, `t1` (28 layers × 4 shards = 112 files/kind). So any
  extraction-POINT variant (final-context r_B etc.) is a diff-of-means re-read, NOT a
  re-capture. Kind aliases: `store_io.py` maps `context_end`↔`context_k`, `t1`↔`answer_k_t1`.
- **Maps** (linear `.npz` + nonlinear `__mlp.pt`/`__kernel.pt`, both variants × U rungs):
  `issue1739_ctxmap/analysis_tensors/maps/`. `_load_nl_map` refuses wrong `map_seed` (pin 0).
- **Proven staging + fit dispatch:** `scripts/issue1739_leg2.sh` (scoped via
  `EPM_I1739_BEHAVIORS`) → `scripts/issue1739_fits.py --transfer ...` (gap2 script is the
  worked single-corner example). E1 direction = `_load_rb_e1` over the extraction store;
  e2/e2p directions come from the LABELED store + per-rollout DV (`matched_pair_split_weights`)
  — natpv streaming (963k tar) is only needed for the natural rungs, hall+syc only.
- **Measured walls:** wide-nomlp transfer ≈23 s/unit (evil OOD ≈10 h / 1556 units); nlmap
  transfer 28.81 s/unit + map_fit 2651.32 s/key + readout groups {250: 4.9 s, 2500: 46 s,
  8000: 140 s} (`eval_results/issue_1739/nonlinear_map/fanout_runbook.md`); arm5 MLP-bearing
  unit ≈12× a core unit (`issue1739_transfer_roster_pilot.py`, n_train=6468/n_eval=1982/d=3584).

**Why:** rounds on this issue recur; re-deriving the store-kind fact alone cost several
plan-time reads, and mis-assuming a re-capture would have added a phantom GPU phase.
**How to apply:** verify prefixes still resolve (scoped `list_repo_tree`) then cite these
as `Source: #1739 <round>` — the inherit fast-path.
