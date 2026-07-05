---
title: 'Map-change substrate-dependence: generic vs behavior-eliciting vs mix queries'
kind: experiment
tags: []
created_at: '2026-07-01T18:56:55Z'
has_clean_result: false
parent_id: 667
origin_prompt: 'Spec + run the clean map-change substrate-dependence experiment: 4
  behaviors x 3 query substrates (generic UltraChat vs behavior-eliciting vs mix),
  reuse #537 adapters, 50-context battery, question-averaged c_C, PCA-48, M0 vs M+
  (Delta/floor + chain-rho). Run in background (happy coder, autonomous). Resume pod-667;
  if not up in 5 min, provision a new pod. SAVE+UPLOAD the pre and post trained maps
  AND ALL context+answer activations pre/post finetuning UNREDUCED (per-token, per-question,
  all layers, base+trained) for followups. Ensure the pod does not run out of space.'
goal: 'Determine whether the finetuning-induced context->answer map change (M0 base
  vs M+ trained, per #537 behavior) depends on the query substrate, by fitting + comparing
  M0 vs M+ across three substrates (generic UltraChat, behavior-eliciting probes,
  mix) on the shared 50-context battery, reporting floor-normalized function-change
  and chain-rho per substrate.'
relates_to:
- leak-predictor
---
## Goal

Determine whether the finetuning-induced context->answer map change (M0 base vs M+ trained, per #537 behavior) depends on the query substrate, by fitting + comparing M0 vs M+ across three substrates (generic UltraChat, behavior-eliciting probes, mix) on the shared 50-context battery, reporting floor-normalized function-change and chain-rho per substrate.

## Design

- **Behaviors (4)**: em, fact, sycophancy, marker. **REUSE the existing #537 LoRA adapters — do NOT retrain.** One canonical adapter per behavior (planner picks + justifies; suggest the `default`-context install). M0 = base Qwen-2.5-7B-Instruct, M⁺ = adapter-applied.
- **Contexts**: the 50-context battery (`data/issue594/battery.json`), shared across all three substrates (so this is directly comparable to the base-map + the contexts/questions/completions dashboards).
- **Query substrate = the single manipulated variable**:
  1. **Generic** — 48 UltraChat questions (`data/issue594/probes_ultrachat.json`).
  2. **Behavior-eliciting** — each behavior's #537 eval probe pool (marker `pool_marker_eval_32`; fact 30 recall+OOD; sycophancy 25 wrong-claims; em Betley-8).
  3. **Mix** — matched-size combination (planner specifies, e.g. half generic + half behavior-eliciting).
- **Map fit** M: c_C → v_A per (behavior, substrate, layer), following the base-map recipe EXACTLY: **question-AVERAGED** c_C (last-input-token, mean over that substrate's questions) and v_A (mean-answer, mean over that substrate's questions), **top-48 PCA** target, closed-form ridge (+ the overfit-upper-bound MLP), LOCO-CV, all 28 layers.
- **DVs**: function-change **Δ/floor** (floor-normalized map-output change projected on r_B; marker via `W_U[※]`) AND **chain-ρ** (r_Bᵀ M(c) vs leakage E) under M0 vs M⁺. IMPORTANT: on the generic substrate the behavior often won't fire → E≈0 → chain-ρ is uninformative there; report Δ/floor as the valid read and flag chain-ρ as N/A where E has no variance.
- **r_B**: em/sycophancy diff-of-means (#658); fact re-extracted; marker `W_U[※]`.

## Deliverables — SAVE + UPLOAD (explicit user requirement)

1. **The fitted maps M0 AND M⁺** per (behavior, substrate, layer) — factored form (weights + PCA basis + input/output norms + λ), verified-exact-reconstruction, uploaded to the HF data repo. (Reuse this session's `scripts/issue667_save_maps.py` approach.)
2. **ALL context and answer activations, pre AND post finetuning, UNREDUCED.** Per (behavior, substrate, context, question), save the FULL per-token residual-stream activations over BOTH the context span AND the answer span, at ALL 28 layers, for BOTH the base model (pre) and the adapter-applied model (post). **Do NOT reduce to means/summaries** — keep per-token + per-question so followup analyses (per-token, per-question) are possible later. fp16 is acceptable. ALSO save the reduced per-question summaries (c_C last-input-token + v_A mean-answer, per question, all layers, base+trained) as the direct map-fit inputs.

## Disk safety (explicit user requirement) — the unreduced activations are LARGE (order a few TB)

- **Size the full-activation footprint in the plan.** Rough estimate: full per-token × 28 layers × per-question × 4 behaviors × 3 substrates × (base+trained), fp16 ≈ **~2–3 TB**.
- **The pod local disk (~130 GB MooseFS quota) MUST NOT fill.** Stream-upload each (behavior, substrate, context, question) cell's activations to the HF data repo the MOMENT it is computed, then DELETE the local copy — NEVER accumulate the full set on the pod. Monitor `df` / the EDQUOT quota; writes must fail loud (never silently drop) on quota exhaustion.
- **Check HF storage headroom BEFORE the run** (account ≈ 4.3 TB / 10 TB ceiling; +2–3 TB must stay under the ceiling). If it would exceed the ceiling: shard / route large LFS to the private overflow repo, or (last resort) descope the saved layer set — but the user's DEFAULT is **keep unreduced**, so surface any descope decision rather than silently reducing.
- Bulk per-cell `upload_folder` commits (not per-file), staying under the 256-commits/hr cap; per-cell resume-skip so a mid-run pod death re-runs only incomplete cells.

## Pod handling (explicit user requirement)

- **FIRST try to resume pod-667** (`uv run python scripts/pod.py resume --issue 667`). If it does NOT come up within ~5 minutes (SUPPLY_CONSTRAINT / host unavailable), **PROVISION A NEW 8×H100 pod** (`pod.py provision --issue <N> --gpu-type H100 --gpu-count 8`) and proceed. Do not block indefinitely on the resume.

## Reuse (don't reinvent)

- Base-map generic base R + base c_C/v_A already exist in the #594 store — reuse for the generic-substrate M0 side.
- The #537 adapters exist on HF — reuse (no training).
- The map-fit + floor-normalized-Δ machinery exists (`scripts/issue722_fit_M.py`, `scripts/issue667_save_maps.py`, `src/explore_persona_space/analysis/vectorized_mlp_skill.py`) — reuse; add per-substrate looping + the unreduced-activation dump.
- **Measure one cell first** to lock the wall-clock/storage estimate before the full run (per this session's repeated estimate misses).
