---
title: Context×answer summary-recipe sweep under LOFO on UltraChat probes with OOD-probe
  generalization
kind: experiment
tags: []
created_at: '2026-07-03T08:50:07Z'
has_clean_result: false
parent_id: 810
origin_prompt: 'Run the following experiment:

  - LOFO evaluation

  - ultrachat probes for activation collection

  - evaluation on OOD ultrachat probes

  - For both context vector and answer vector try:

  -- mean pool over all tokens (per layer)

  -- mean pool over all tokens at all layers

  -- max pool over all tokens (per layer)

  -- max pool over all tokens at all layers

  -- mean of max pool over layers

  -- max of mean pool over layers

  -- newline before assistant starts answering (after chat template) for context vector,
  newline before user would start answering (after chat template) --> per layer and
  also mean pool over layers, max pool over layers

  - For answer vector:

  -- im_end token

  -- token before im_end

  Try all combinations of these context and answer vectors (at all layers when this
  makes sense) and figure out the best one. Help me to plan this experiment and give
  an estimate as to GPU and wallclock time'
goal: On base Qwen2.5-7B-Instruct over the 50-context battery, determine which (context-vector
  recipe × answer-vector recipe) combination best supports (a) the linear context→answer
  map, (b) direct behavior read-out, and (c) map-mediated (chain) read-out, under
  leave-one-family-out (LOFO) evaluation with generic UltraChat probes for activation
  collection and a disjoint OOD UltraChat probe pool testing probe-set generalization
  on both the input and target sides — naming the best combination against selection-symmetric
  nulls.
relates_to:
- leak-predictor
---
## Goal

On base Qwen2.5-7B-Instruct over the 50-context battery, determine which (context-vector recipe × answer-vector recipe) combination best supports (a) the linear context→answer map, (b) direct behavior read-out, and (c) map-mediated (chain) read-out, under leave-one-family-out (LOFO) evaluation with generic UltraChat probes for activation collection and a disjoint OOD UltraChat probe pool testing probe-set generalization on both the input and target sides — naming the best combination against selection-symmetric nulls.

## Overview / Motivation

Every prior summary-recipe choice in the leakage-predictor line was made piecemeal under the weakest evaluation regime: the context read was locked in #658 against ONE alternative (reconstruction DV only, LOCO, Betley probes); the answer-side sweep #810 was Betley/LOCO-only and its 2026-07-03 LOFO re-read collapsed read-out ρ 0.909→0.285 under family folds; #812's pooled-vs-unpooled bound is LOCO/Betley-only. No experiment has (a) compared context-vector recipes as predictors, (b) swept context×answer COMBINATIONS, (c) combined UltraChat probes with LOFO, or (d) tested OOD-probe generalization of any summary. This closes all four gaps in one shot.

**User decisions (locked in chat 2026-07-03):** three DVs — map reconstruction + oracle read-out + chain (reconstruction) read-out · matched-layer pairing only · OOD probes on BOTH axes (input-side and target-side).

## Design

- **Model:** base `Qwen/Qwen2.5-7B-Instruct` only, nothing trained, all 28 layers.
- **Contexts:** the 50-context battery (`data/issue594/battery.json`); LOFO = 7 leave-one-family-out folds on its `family` field (persona 14 / wildchat 10 / icl 8 / rephrase 6 / format 5 / behavior 5 / default 2).
- **Probes:** set A = existing 48-probe UltraChat pool (`data/issue594/probes_ultrachat.json`); set B = NEW disjoint 48-probe UltraChat pool, identical filters + Betley-length matching, set-A exclusion applied BEFORE matching (mod `scripts/issue594_build_probes_ultrachat.py`: `--exclude-probes-file` + `--out`).
- **E0 targets (reused, no new judging):** `eval_results/issue_812/graded_e0_{highm,lowm}.json` — context-level `graded_mean`, 7 behaviors (deception EXCLUDED, failed #812 reliability preflight); #812 reliability ceilings reused; context ids join the battery directly.

### Summary recipes

Context side — 9 families, 90 cells: mean over prompt tokens per layer (28) · mean over tokens×layers (1) · max over prompt tokens per layer (28) · max over tokens×layers (1) · mean-over-layers of per-layer max (1) · max-over-layers of per-layer mean (1) · assistant-header newline (last input token) per layer (28) + layer-mean (1) + layer-max (1).

Answer side — 11 families, 146 cells: the same 6 mean/max families over ANSWER-CONTENT tokens (58) · `uh_nl` = final `\n` of a teacher-forced `<|im_end|>\n<|im_start|>user\n` tail, per layer (28) + layer-mean (1) + layer-max (1) · `<|im_end|>` per layer (28) · last content token (token before `<|im_end|>`) per layer (28).

Map cells, matched-layer: 3×5×28 + 84×6 + 6×140 + 36 = **1,800**.

### Pinned decisions

- **Chain read-out:** the answer-side oracle read-out is ridge-fit on TRUE set-A answer summaries (in the map's train-fold PCA basis) on train families, applied UNCHANGED to M(c) on held-out families — the chain-vs-oracle gap is then map-induced signal loss. Never refit on reconstructions.
- **Max probe-averaging = `probe_avg_max`** (token-max per probe fp32, then probe-mean); recorded in the store manifest.
- **Context span** = full templated input incl. probe turn + assistant header (context-only is degenerate for the `default` family and probe-invariant). **Answer span** = answer-content tokens only; `im_end` / last-content / `uh_nl` are boundary families, never inside pools.
- **fp16 overflow guard:** reductions fp32 on GPU; max-pool families persisted bf16; fail-loud `|x| < 6e4` assert on fp16-bound tensors.
- **#813's unreduced store NOT reused** (full_trained dead weight, lacks the `uh_nl` tail, set B needs fresh capture anyway). The #658 UltraChat position store (`issue658_theory_assumptions/answer_position_sweep_genre-generalization-ultrachat/`) is the EQUIVALENCE GATE: new extractor's probe-mean `im_end` + last-content vectors must match it (fp16 tol) on 2–3 contexts before the full run.
- **Set-A completions reused** from `issue658_theory_assumptions/raw_completions_genre-generalization-ultrachat/` (no regeneration); set-B generation matches set-A recipe exactly (vLLM greedy, max_new_tokens 512, same chat templating).

### Scoring matrix (all fits on set-A summaries, LOFO 7-fold)

F1 = ridge map per (c-cell × a-cell), train-fold PCA target basis (k ≤ min(48, n_train−2), per-fold scoring mandatory). F2c/F2a = scalar ridge read-outs per cell × behavior.

| Read | DV | Eval input → target | Regime | Null |
|---|---|---|---|---|
| R1 | recon skill | A-ctx → A-ans | in-probe | perm-refit, selection-symmetric max over 1,800 |
| R2 | recon skill | B-ctx → A-ans | input-OOD | same null fits scored at this regime |
| R3 | recon skill | A-ctx → B-ans | target-OOD | same; ceiling diagnostic = skill of Y_A predicting Y_B |
| R4 | recon skill | B-ctx → B-ans | both-OOD | diagnostic only |
| R5/R6 | oracle ρ | A-ctx / A-ans → E0 | in-probe | stored-prediction perm-ρ, selection-symmetric |
| R7/R8 | oracle ρ | B-ctx / B-ans → E0 | input-OOD | stored-prediction perm-ρ |
| R9 | chain ρ | A-ctx → M(c) → E0 | in-probe | stored-prediction perm-ρ over 1,800 |
| R10 | chain ρ | B-ctx → M(c) → E0 | input-OOD | stored-prediction perm-ρ |

ρ reads pool held-out predictions across folds, one Spearman over ≤50 joined contexts (handles the n=2 `default` fold; ≥4 joined contexts required per read); skill reads score per fold, variance-weighted. Any "best combination" headline uses the per-draw max-over-cells inherited band (`.claude/rules/selection-symmetric-nulls.md`).

## Script inventory (reuse map)

1. Mod `scripts/issue594_build_probes_ultrachat.py`: `--exclude-probes-file` (drop set-A prompt_ids/casefolded text pre-match) + `--out`; assert 48 matches, 0 overlap, decile band vs set A's meta.
2. New gen script (GPU, vLLM, own process): 2,400 set-B greedy completions; reuses `issue658_extract_base_store.py` gen patterns; per-context checkpoint+resume; one `upload_folder`.
3. New extraction script (GPU, HF, own process): extends `scripts/issue810_extract_positions.py` (batched left-pad forward, GPU-side gather; boundary append 2→5 tokens with fed-id asserts 151645, 198, 151644, runtime-asserted `user` id, 198); in-forward fp32 reductions (8 base families/side); per-probe fp16/bf16 persist (50 files × 2 sets ≈ 7.7 GB); the 12 layer-pooled cells derived per-probe at fit time.
4. New LOFO fit driver (CPU): builds 90×146 summary matrices per probe set; batched fits reusing `scripts/issue810_adhoc_lofo_heatmaps.py` primitives (`_group_fold_ridge_predict`, `_recon_fold_predict`, `skill_over_mean_r2_lofo`, `_gram_top_k_pca`) + `issue658_fit_predictors` ridge/PRESS primitives; X-caches once per (c-cell × fold)=630, target PCA once per (a-cell × fold)=1,022, all Y-solves batched; persists pooled held-out predictions.
5. New nulls/figures script (CPU): DV-1 perm-refit null adapted from `scripts/issue810_batched_null.py` LOCO→LOFO, batched fp32, scored at ALL regimes in-pass (~2.4 PFLOP → ~3–5 h; a serial version is ~15+ h — banned per `.claude/rules/vectorize-many-cell-fits.md`); DV-2/3 nulls re-correlate STORED predictions; per-draw×per-cell matrices persisted to `analysis_tensors/`; family×family heatmaps per DV×regime, OOD deltas, chain-vs-oracle gap.

Nothing imported from the unmerged issue-812/issue-813 worktrees (built-but-stranded rule).

## Compute + routing

1. VM: probe-B build (CPU minutes).
2. ONE GPU provision (single A100-40/H100, GCP auto ladder): gen-B → extract set A → extract set B (equivalence gate first) → upload verify → RELEASE GPU.
3. `cpu-mid` lane (e2-standard-8): fits → nulls → figures, per-cell-block checkpoints + resume. The CPU phase must NOT hold the GPU pod.

**Estimates:** GPU ~1–1.5 GPU-h (book 2 with one retry); CPU ~4–6 h detached; ~6–8 h pure compute, ~1 day end-to-end. Storage: 7.7 GB per-probe summaries + ~10 MB completions JSON + ~200 MB null matrices to HF; eval JSONs to git.

## Key risks

fp16 max-pool overflow (fp32 reduce + bf16 persist + assert) · `uh_nl` is genuinely new code (5-token append, runtime-asserted ids; re-tokenization drift accepted as in #810) · LOFO fold imbalance (per-fold PCA cap n_train−2; pooled-ρ / variance-weighted-skill scoring) · probe-pool-B length-matching drift after exclusion (assert decile band; fallback second-seed re-match, documented deviation) · E0 join gaps (≥4 joined contexts per read) · vLLM and HF capture in separate processes (teardown gotcha).

## Verification

- Pre-run: position-store equivalence gate; probe-B disjointness + decile asserts; battery/probe/completion sha pins resolve on HF.
- Post-fit: batched-vs-serial oracle equivalence on 2–3 cells (the `issue810_batched_null` contract); the R1 mean×mean cell should land near the parent line's ~0.7–0.8 anchor (pipeline sanity).

## Provenance

Planned interactively with the user 2026-07-03 (plan file: `~/.claude/plans/run-the-following-experiment-stateful-sedgewick.md`, user-approved). Design decisions (three DVs, matched-layer, both OOD axes) are user-locked; the adversarial-planner should treat them as fixed scope, not open questions.
