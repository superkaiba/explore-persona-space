---
title: 'Preview: test leakage-theory gate-chain assumptions A3.6–A3.10 on #537''s
  existing contrastive adapters (forward-pass, no retrain), A3.7 both ways, vs G'
kind: experiment
tags: []
created_at: '2026-06-25T08:04:47Z'
has_clean_result: false
parent_id: 660
origin_prompt: 'Create a new issue for this. Run it in the background with happy coder
  — the #537 gate-chain preview (A3.6–A3.10, A3.7 both ways, vs G) under #660.'
goal: 'Test the five trained-model gate-chain assumptions of the leakage-predictor
  theory (A3.6 read-out stability, A3.7 source write ŵ∥δ, A3.8 rank-one/scalar gate,
  A3.9 base-similarity-predicts-realized-gate, A3.10 base-gate-predicts-post-FT-gate)
  on Qwen2.5-7B using #537''s EXISTING contrastive LoRA adapters with NO new fine-tuning,
  via a forward-pass activation-extraction sweep benchmarked against #537''s measured
  leakage matrix G, with A3.7 measured BOTH ways on the contrastive adapters (cos(ŵ,δ^contra)
  and cos(ŵ,δ^pos) decomposition + frac_ctx partial + shuffled-δ null) so the result
  also says whether the positive-only fleet arm is needed; a fast forward-pass preview/de-risk
  for program #660, explicitly excluding the clean positive-only A3.7 identification,
  the non-saturated/dose-controlled final gate numbers, and the held-out end-to-end
  predictor (those need the fleet retrain).'
relates_to:
- leak-predictor
---
## Goal

Test the five trained-model gate-chain assumptions of the leakage-predictor theory (A3.6 read-out stability, A3.7 source write ŵ∥δ, A3.8 rank-one/scalar gate, A3.9 base-similarity-predicts-realized-gate, A3.10 base-gate-predicts-post-FT-gate) on Qwen2.5-7B using #537's EXISTING contrastive LoRA adapters with NO new fine-tuning, via a forward-pass activation-extraction sweep benchmarked against #537's measured leakage matrix G, with A3.7 measured BOTH ways on the contrastive adapters (cos(ŵ,δ^contra) and cos(ŵ,δ^pos) decomposition + frac_ctx partial + shuffled-δ null) so the result also says whether the positive-only fleet arm is needed; a fast forward-pass preview/de-risk for program #660, explicitly excluding the clean positive-only A3.7 identification, the non-saturated/dose-controlled final gate numbers, and the held-out end-to-end predictor (those need the fleet retrain).


## Provenance

Originating user prompt (chat, 2026-06-25): "Create a new issue for this. Run it
in the background with happy coder" — "this" = the #537 gate-chain preview
(A3.6–A3.10, A3.7 both ways, vs G) under program #660, scoped across a chat thread
that (a) re-read #651's geometry, (b) confirmed #537's adapters span the
(behavior × context) grid the theory's leakage object needs, (c) confirmed
`δ^contra = t⁺ − t⁻` is computable on #537's frozen training JSONLs, and (d)
scoped this as the cheap forward-pass preview vs the #660 fleet retrain.

## Background — why this is cheap and what it answers

Program #660 splits its 11 assumption tests into base-only (A3.1–A3.5, A3.5a —
Phase 0/1, #658, running, no adapters) and trained-model (A3.6–A3.10 — needs
adapters). The trained set is what the #660 plan currently re-trains a fleet for.
But #537 already trained ~80 (behavior × train-context) contrastive LoRA adapters
and measured their cross-context leakage matrix `G[behavior, train-ctx →
eval-ctx]`. The gate-chain tests are forward-pass reads on those adapters plus CPU
linear algebra — no retrain. #651 already established (i) the extraction pipeline
+ nulls on the same model, and (ii) that em / sycophancy / fact are NON-saturated
on these adapters (graded reads), so their gate signal is readable; marker is
saturated-ish and is excluded / caveated.

**A3.7 is measured BOTH ways on the same contrastive adapters** (holding the
realized write `ŵ` fixed, varying the displacement target):
- `cos(ŵ, δ^contra)` with `δ^contra = t⁺ − t⁻` — the objective-consistent
  contrastive displacement;
- `cos(ŵ, δ^pos)` with `δ^pos = t⁺ − v0(C)` — the positive-only displacement, as a
  **decomposition** of the contrastive write into its positive-aligned vs
  negative-induced parts.
Comparing the two alignments directly measures **whether the contrastive
negatives rotate the realized write direction** — i.e. whether the positive-only
fleet arm would even change the A3.7 verdict (a cheap test of whether the retrain
arm is necessary).

This is a **fast preview / de-risk for #660**, forward-pass + CPU only. It is
explicitly NOT the clean-final layer: the decoupled positive-only A3.7
identification (needs a positive-only-TRAINED adapter), the non-saturated /
dose-controlled final gate numbers, and the held-out end-to-end predictor `L̂`
remain for the fleet-retrain phase (#660 Phase 2+).

## Design

**Extraction (GPU, forward-pass only; reuse #651's `activation_shift` pipeline):**
- Sources: the non-saturated behaviors (em, sycophancy, fact) × #537's 16 train
  contexts = ~48 source adapters (marker excluded as saturated; optionally
  included with a saturation caveat).
- For each source adapter, read `v⁺(C')` (trained) and `v0(C')` (base) across the
  16 #537 contexts as **targets** C' (the target grid); capture layer 14 primary,
  7/21 supplement (per #651/#658). Persist per-cell shift tensors to HF
  (analysis_tensors).
- Base-side: `v0(C)`, `c_C` (prompt-side), `t⁺` / `t⁻` (teacher-force #537's
  frozen positive / contrastive-negative training rows through the base model),
  `Σ_c` over a background corpus. Reuse #658's base store where it already covers
  these; re-extract what it does not.

**Per-assumption tests (CPU on the cached store, benchmarked against #537's `G`):**
- **A3.6** read-out stability: re-extract `r⁺_{B'}`; judge on the PARTIALLED-OUT
  change `r_B'ᵀ(v⁺−v0)` vs `(E⁺−E0)` with `E0` partialled out (C10), not the level.
- **A3.7** source write: `cos(ŵ, δ^contra)` AND `cos(ŵ, δ^pos)` + the R3-1
  context-offset partial `frac_ctx = ‖v0(C)−v0(C_neg)‖/‖δ^contra‖` + the R3-2
  shuffled-δ null (`cos(ŵ, δ_of_a_different_behavior)`).
- **A3.8** rank-one / scalar gate: per source, are the target `Δv(C')` scalar
  multiples of `ŵ`? (#651 did the 1-target slice; this is the multi-target test.)
- **A3.9** base-similarity gate (headline): does a base-model key-query similarity
  (`c_C` key default; `ψ(t_{C,B})` answer-profile key as an ablation) predict the
  realized ACTIVATION gate `ĝ^real(C') = ŵᵀΔv(C') / ŵᵀŵ` (B1 — score on the
  activation gate, not the marker log-prob)?
- **A3.10** base-gate validity: does the base-model gate predict the
  post-fine-tuning gate (at fixed base metric M⁰; R3-3)?

**Net-new code (B3, the pacing item):** the whitened gate
`g_C(C') = c_CᵀΣc⁻¹c_{C'} / c_CᵀΣc⁻¹c_C`, the `Σ_c` estimator + regularized
inverse `(Σc+λI)⁻¹`, and the cosine-limit reduction unit test (the gate must
reduce to `cos(c_C,c_C')` in the `Σc=I` / equal-norm / `δ∥r_B` limit) — NO A3.9 /
A3.10 number is trusted until that unit test passes. A3.6/A3.7/A3.8/A3.10 are
cheap algebra reusing #651 machinery.

## Scope / caveats (carried into the clean-result)

- Forward-pass + CPU only; NO fine-tuning. Preview-grade, not clean-final.
- Restricted to NON-saturated behaviors (em / sycophancy / fact); marker excluded
  / caveated (saturation hides the gate — #448 / #474 / #651).
- A3.7 = the contrastive-objective-consistent + decomposition read; the CLEAN
  positive-only identification (decoupled from the negative set) needs the
  positive-only-trained fleet arm (#660 Phase 2).
- η (install dose) is uncontrolled in #537 → within-source A3.8 is clean,
  cross-source dose comparisons are confounded (reported, not corrected here).
- Ground truth = #537's existing `G` (no new judge runs for the realized leakage).

## Reuse (artifact-reuse fitness)

- Reuse #537's ~80 contrastive LoRA adapters (`adapters/i537_*`, HF) — same base
  model + recipe; the single new variable is the GEOMETRY READ + gate analysis.
- Reuse #537's frozen training JSONLs (positives + contrastive negatives) for
  `t⁺` / `t⁻`.
- Reuse #651's extraction pipeline (`activation_shift`, sign-flip / row-shuffle
  nulls, layer set) and its saturation map (em/syc/fact non-saturated).
- Reuse #658's base store where it covers `v0` / `c_C` / `Σc`.

## Resource estimate

~2–4 GPU-h forward-pass (4×H100, ~2–3 h wallclock) for the extraction sweep;
per-assumption tests are CPU minutes; the whitened-gate net-new code + unit test
is the main implementation effort. Well under the Step 2c auto-approve cap.
