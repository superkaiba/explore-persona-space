---
title: 'Finish #511''s dropped pieces: CPU un-descope (wass2 + ridge N=500 + loc-epoch
  2/3/5 + g_logprob) on VM + full-response Rao-Blackwellized JS predictor on GPU'
kind: analysis
tags: []
created_at: '2026-06-08T23:16:05Z'
has_clean_result: false
parent_id: 511
---
# Goal

Finish the #511 predictor comparison by running the two pieces the autonomous
compute-deviation descope dropped, so "which persona-distance metric best predicts
marker-leakage transfer" is answered with (a) the full activation-space cloud-aware
ridge and (b) an output-distribution baseline computed with the **canonical
full-response** JS estimator (not the deprecated single-next-token proxy).

Parent: #511 (itself parent #502). #511 plateaued the headline gauss_kl L22 cell but
dropped wass2, ridge N=500, loc-epoch robustness, the g_logprob target, and the JS
baseline under a CPU wall-time blowup (52× the plan §9 estimate; full eigh on the
2N×2N joint Gram). This task runs the feasible remainder: the CPU pieces **as-is** on
the VM, and the JS baseline **upgraded to the full-response estimator** on a GPU.
Search-best (full ~1500-cell grid) stays deferred — it needs the truncated-eig /
GPU-port optimization, not an as-is run.

## Component 1 — CPU un-descope (VM, 0 GPU, ~1-1.5 days)

Restore the descoped scope in `scripts/issue511_probe_count_sweep.py` and re-run
`--mode full` on the VM (29 cores):

- `CLOUD_METRICS_KEEP` → add `wass2` (restore {gauss_kl, mmd, wass2}).
- Ridge cells (L19-L24 cloud-aware): restore **N=500 and R=10** (currently capped
  N≤350, R=5).
- Checkpoints: run loc-epoch **{1,2,3,5}** (`DEFAULT_EPOCHS_TIME_PERMITTING`) for the
  robustness sweep, not just loc_ep1.
- **Distance-matrix caching across epochs.** The distance matrices are
  target-independent, so loc_ep{2,3,5} should cost regression refits only, not
  re-eigh. Verify the driver caches; if it recomputes per epoch it's ~4× and needs
  the cache added (this is what keeps Component 1 to ~1 day instead of ~4 days).
- Secondary target: add the **g_logprob** ΔG variant alongside the count-style ΔG so
  the plateau is verified within the DV family.
- **Out of scope:** search-best per-N (full grid) — deferred to a separate
  truncated-eig / GPU-port task.
- Deliverable: extend the #511 bakeoff table — wass2 vs gauss_kl vs mmd vs cosine
  across the full N grid and across loc_ep{1,2,3,5}; does the ridge hold and does the
  plateau survive on the other checkpoints (#502/#511's open checkpoint-fragility
  question).

## Component 2 — Full-response JS baseline (GPU pod, base Qwen-2.5-7B)

Replace #511's dropped single-next-token JS with the canonical **sequence-level
Rao-Blackwellized JS** (the next-token version is DEPRECATED per
`.claude/rules/persona-distance-metrics.md`).

- Reuse `scripts/issue444_persona_distance_topic.py` (`sample_responses`,
  `js_vs_reference`, `_js_from_logprobs`) — the full-response estimator (arXiv
  2504.10637): sample R≈8 responses temp=1 ≤256 tok under BOTH personas,
  teacher-force each through both conditioned models, exact full-vocab per-position
  JS, length-normalized, averaged over positions + samples. Headline = base-2 JS,
  polarity-aligned `M_js = 1 − JS`; also report both KL directions (asymmetry is
  diagnostic).
- Personas: #502/#511's **16 transforms** (system-prompt personas). Build the 16×16
  JS distance matrix on the base model.
- Probes: design choice for the planner — #502's mixed-distribution probe set
  (apples-to-apples with the activation metrics) vs the canonical Betley
  preregistered paraphrases. Lean toward #502's probes for a clean predictor
  comparison.
- Regress (length-partial Spearman + LOCO CV R², same as #511) against #474 ΔG at
  loc_ep{1,2,3,5}. Compare CV R² to **gauss_kl L22** (headline) and **cosine**: does
  an output-distribution metric predict marker leakage as well as activation
  geometry?
- GPU: 1× H100 (`eval` intent), base-model forward passes only, no training.

## Reuse vs new

- **Reuse:** #511 sweep driver + the bakeoff metric/regression code; #444's
  full-response JS estimator; #502 cached activations + probes; #474 ΔG targets
  (on disk).
- **New:** the wass2 / N=500 / loc-epoch / g_logprob un-descope wiring (small);
  wiring #444's JS estimator to emit a 16×16 matrix over #502's personas + the
  regression against #474 ΔG.

## Routing note

Component 2 is a new measurement → run through `/adversarial-planner` (via `/issue`).
Component 1 is a mechanical scope-restore re-run. The planner may split into two
phases (CPU VM phase, GPU pod phase) or two tasks if the consistency-checker flags
the bundling.
