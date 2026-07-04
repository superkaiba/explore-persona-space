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

On base Qwen2.5-7B-Instruct over the 50-context battery, determine which (context-vector recipe × answer-vector recipe) combination best supports (a) the linear context→answer map, (b) direct behavior read-out, and (c) map-mediated (chain) read-out, under leave-one-family-out (LOFO) evaluation with generic UltraChat probes for activation collection and a disjoint OOD UltraChat probe pool testing probe-set generalization on both the input and target sides — including chat-template-token summaries, with-vs-without-template pooling variants, last-k context content tokens, and first-10/last-10 answer-token positions as map targets — naming the best combination against selection-symmetric nulls.

## Overview / Motivation

Every prior summary-recipe choice in the leakage-predictor line was made piecemeal under the weakest evaluation regime: the context read was locked in #658 against ONE alternative (reconstruction DV only, LOCO, Betley probes); the answer-side sweep #810 was Betley/LOCO-only and its 2026-07-03 LOFO re-read collapsed read-out ρ 0.909→0.285 under family folds; #812's pooled-vs-unpooled bound is LOCO/Betley-only. No experiment has (a) compared context-vector recipes as predictors, (b) swept context×answer COMBINATIONS, (c) combined UltraChat probes with LOFO, or (d) tested OOD-probe generalization of any summary. This closes all four gaps in one shot. The template-token families are motivated by the boundary reads already being the line's strongest single positions (#810: turn-newline 0.735 vs whole-answer mean 0.800 reconstruction skill) — consistent with template tokens acting as aggregation sites.

**User decisions (locked in chat 2026-07-03):** three DVs — map reconstruction + oracle read-out + chain (reconstruction) read-out · matched-layer pairing only · OOD probes on BOTH axes (input-side and target-side) · EXTENDED scope (user, 2026-07-03 second pass): chat-template tokens separately + template-block mean/max pools on both sides; with-vs-without-chat-template pooling variants on both sides; last-k context content tokens as summaries; first-10/last-10 answer-token activations as reconstruction targets.

**Assumptions from the scope extension (user AFK at ask-time; defaults chosen, adjustable at relaunch):** (i) context-side template tokens = the trailing assistant-header block only (uniform across all 50 contexts), not per-family system/user headers; (ii) last-k = last 1–8 content tokens; (iii) the 20 answer positions serve as map TARGETS and as read-out PREDICTORS; (iv) WITH-template answer pools in both variants (content+im_end+\n, and content+full 5-token boundary block).

## Design

- **Model:** base `Qwen/Qwen2.5-7B-Instruct` only, nothing trained, all 28 layers.
- **Contexts:** the 50-context battery (`data/issue594/battery.json`); LOFO = 7 leave-one-family-out folds on its `family` field (persona 14 / wildchat 10 / icl 8 / rephrase 6 / format 5 / behavior 5 / default 2).
- **Probes:** set A = existing 48-probe UltraChat pool (`data/issue594/probes_ultrachat.json`); set B = NEW disjoint 48-probe UltraChat pool, identical filters + Betley-length matching, set-A exclusion applied BEFORE matching (mod `scripts/issue594_build_probes_ultrachat.py`: `--exclude-probes-file` + `--out`).
- **E0 targets (reused, no new judging):** `eval_results/issue_812/graded_e0_{highm,lowm}.json` — context-level `graded_mean`, 7 behaviors (deception EXCLUDED, failed #812 reliability preflight); #812 reliability ceilings reused; context ids join the battery directly.

### Context-side summary recipes (19 per-layer families + 10 layer-pooled singles = 542 cells)

1. Mean over ALL prompt tokens (WITH template), per layer (28); max likewise (28); the 4 layer-pooled variants: mean over tokens×layers, max over tokens×layers, mean-over-layers of per-layer max, max-over-layers of per-layer mean (4).
2. Content-only (WITHOUT template) mean per layer (28) and max per layer (28), over system+user text tokens with all template/special tokens excluded (the probe turn's content tokens ARE included — keeps the `default` family non-degenerate); + their 4 layer-pooled variants (4).
3. Assistant-header newline `ah_nl` (last input token), per layer (28) + layer-mean (1) + layer-max (1).
4. Each remaining trailing template-block token separately, per layer: user-turn `<|im_end|>`, its `\n`, `<|im_start|>`, `assistant` (4×28).
5. Mean and max over the 5-token trailing template block, per layer (2×28).
6. Last-k content tokens before the template suffix, k = 1..8 end-aligned single positions, per layer (8×28).

### Answer-side summary recipes (16 per-layer families + 10 layer-pooled singles = 458 cells)

1. Mean over answer-CONTENT tokens per layer (28); max likewise (28); the 4 layer-pooled variants (4).
2. Boundary/template tokens separately, per layer: `<|im_end|>` (28), token-before-`<|im_end|>` = last content token (28), the `\n` after `<|im_end|>` (28), `<|im_start|>` of the appended user header (28), `user` token (28), `uh_nl` = the final `\n` of the teacher-forced `<|im_end|>\n<|im_start|>user\n` tail (28) + uh_nl layer-mean (1) + layer-max (1).
3. Template-block pools, per layer: mean and max over the 3-token user header (2×28); mean and max over the full 5-token boundary block (2×28).
4. WITH-template answer pools, per layer: mean and max over content+`<|im_end|>`+`\n` (2×28); mean and max over content+the full 5-token block (2×28); + 4 layer-pooled variants of the maximal one (4).

### Answer-side position TARGETS (20 per-layer families)

First 10 answer-token activations (start-aligned positions 0..9) and last 10 (end-aligned 1..10), per layer, probe-averaged — used as (a) map-reconstruction targets from every context summary and (b) oracle read-out predictors → E0. Mirrors #810's position machinery; the #658 UltraChat position store is the equivalence cross-check.

**Map cells (matched-layer):** per-layer-c × per-layer-a-targets 19×36×28 = 19,152 · per-layer-c × pooled-a 532×10 = 5,320 · pooled-c × per-layer-a 10×1,008 = 10,080 · pooled×pooled 100 → **~34,700 cells**.

### Pinned decisions

- **Chain read-out:** the answer-side oracle read-out is ridge-fit on TRUE set-A answer summaries (in the map's train-fold PCA basis) on train families, applied UNCHANGED to M(c) on held-out families — the chain-vs-oracle gap is map-induced signal loss. Never refit on reconstructions.
- **Max probe-averaging = `probe_avg_max`** (token-max per probe fp32, then probe-mean); recorded in the store manifest.
- **fp16 overflow guard:** reductions fp32 on GPU; max-pool families persisted bf16; fail-loud `|x| < 6e4` assert on fp16-bound tensors.
- **#813's unreduced store NOT reused** (full_trained dead weight, lacks the `uh_nl` tail, set B needs fresh capture anyway). The #658 UltraChat position store (`issue658_theory_assumptions/answer_position_sweep_genre-generalization-ultrachat/`) is the EQUIVALENCE GATE: new extractor's probe-mean `im_end` + last-content + tail/head position vectors must match it (fp16 tol) on 2–3 contexts before the full run.
- **Set-A completions reused** from `issue658_theory_assumptions/raw_completions_genre-generalization-ultrachat/` (no regeneration); set-B generation matches set-A recipe exactly (vLLM greedy, max_new_tokens 512, same chat templating).

### Scoring matrix (all fits on set-A summaries, LOFO 7-fold)

F1 = ridge map per (c-cell × a-target-cell), train-fold PCA target basis (k ≤ min(48, n_train−2), per-fold scoring mandatory). F2c/F2a = scalar ridge read-outs per cell × behavior (answer-side predictors now include the 20 position families).

| Read | DV | Eval input → target | Regime | Null |
|---|---|---|---|---|
| R1 | recon skill | A-ctx → A-ans | in-probe | perm-refit, selection-symmetric max over all cells |
| R2 | recon skill | B-ctx → A-ans | input-OOD | same null fits scored at this regime |
| R3 | recon skill | A-ctx → B-ans | target-OOD | same; ceiling diagnostic = skill of Y_A predicting Y_B |
| R4 | recon skill | B-ctx → B-ans | both-OOD | diagnostic only |
| R5/R6 | oracle ρ | A-ctx / A-ans → E0 | in-probe | stored-prediction perm-ρ, selection-symmetric |
| R7/R8 | oracle ρ | B-ctx / B-ans → E0 | input-OOD | stored-prediction perm-ρ |
| R9 | chain ρ | A-ctx → M(c) → E0 | in-probe | stored-prediction perm-ρ |
| R10 | chain ρ | B-ctx → M(c) → E0 | input-OOD | stored-prediction perm-ρ |

ρ reads pool held-out predictions across folds, one Spearman over ≤50 joined contexts (handles the n=2 `default` fold; ≥4 joined contexts required per read); skill reads score per fold, variance-weighted. Any "best combination" headline uses the per-draw max-over-cells inherited band (`.claude/rules/selection-symmetric-nulls.md`).

## Script inventory (reuse map)

1. Mod `scripts/issue594_build_probes_ultrachat.py`: `--exclude-probes-file` (drop set-A prompt_ids/casefolded text pre-match) + `--out`; assert 48 matches, 0 overlap, decile band vs set A's meta.
2. New gen script (GPU, vLLM, own process): 2,400 set-B greedy completions; reuses `issue658_extract_base_store.py` gen patterns; per-context checkpoint+resume; one `upload_folder`.
3. New extraction script (GPU, HF, own process): extends `scripts/issue810_extract_positions.py` (batched left-pad forward, GPU-side gather; boundary append 2→5 tokens with fed-id asserts 151645, 198, 151644, runtime-asserted `user` id, 198); in-forward fp32 reductions for ALL per-layer families (context 19, answer 16, positions 20); per-probe fp16/bf16 persist (~50 GB across both probe sets); layer-pooled cells derived per-probe at fit time.
4. New LOFO fit driver: builds the summary matrices per probe set; batched fits reusing `scripts/issue810_adhoc_lofo_heatmaps.py` primitives (`_group_fold_ridge_predict`, `_recon_fold_predict`, `skill_over_mean_r2_lofo`, `_gram_top_k_pca`) + `issue658_fit_predictors` ridge/PRESS primitives; X-caches once per (c-cell × fold), target PCA once per (a-cell × fold), all Y-solves batched; persists pooled held-out predictions. **Runs on the GPU (torch batched, TF32) before the pod is released** — at ~35K cells the CPU lane no longer fits the wall-time budget.
5. New nulls/figures script: DV-1 perm-refit null adapted from `scripts/issue810_batched_null.py` LOCO→LOFO, batched on GPU (~46 PFLOP → ~1–2 h TF32; serial or CPU-fp64 is banned per `.claude/rules/vectorize-many-cell-fits.md`), scored at ALL regimes in-pass, never persisting per-draw weights; DV-2/3 nulls re-correlate STORED predictions (CPU-fine); per-draw×per-cell matrices persisted to `analysis_tensors/`; family×family heatmaps per DV×regime, OOD deltas, chain-vs-oracle gap. Figures + aggregation run post-release on `cpu-mid`.

Nothing imported from the unmerged issue-812/issue-813 worktrees (built-but-stranded rule).

## Compute + routing

1. VM: probe-B build (CPU minutes).
2. ONE GPU provision (single A100-40/H100, GCP auto ladder): gen-B → extract set A → extract set B (equivalence gate first) → **batched fits + DV-1 null battery on-GPU** → upload verify → RELEASE GPU.
3. `cpu-mid` lane (e2-standard-8): stored-prediction ρ nulls, aggregation, figures.

**Estimates:** GPU ~2.5–3.5 GPU-h (extraction ~1 h + fits/nulls ~1–2 h; book 4 with one retry); CPU ~1–2 h; wall-clock ~4–6 h pure compute, ~1 day end-to-end. Storage: ~50 GB per-probe summaries + ~10 MB completions JSON + null matrices to HF; eval JSONs to git.

## Key risks

fp16 max-pool overflow (fp32 reduce + bf16 persist + assert) · `uh_nl`/user-header reads are genuinely new code (5-token append, runtime-asserted ids; re-tokenization drift accepted as in #810) · LOFO fold imbalance (per-fold PCA cap n_train−2; pooled-ρ / variance-weighted-skill scoring) · probe-pool-B length-matching drift after exclusion (assert decile band; fallback second-seed re-match, documented deviation) · E0 join gaps (≥4 joined contexts per read) · vLLM and HF capture in separate processes (teardown gotcha) · at ~35K cells the multiple-comparisons burden is severe — the selection-symmetric max-inherited band is the ONLY honest headline read; per-cell p-values are not reportable uncorrected · short answers may have <10 content tokens (head/tail positions overlap — dedupe by absolute position, keep schema slots, mask duplicates in fits as #812 did).

## Verification

- Pre-run: position-store equivalence gate; probe-B disjointness + decile asserts; battery/probe/completion sha pins resolve on HF.
- Post-fit: batched-vs-serial oracle equivalence on 2–3 cells (the `issue810_batched_null` contract); the R1 mean×mean cell should land near the parent line's ~0.7–0.8 anchor (pipeline sanity).

## Provenance

Planned interactively with the user 2026-07-03 (plan file: `~/.claude/plans/run-the-following-experiment-stateful-sedgewick.md`, user-approved; scope EXTENDED by user same day: template tokens, with/without-template pools, last-k context tokens, first/last-10 position targets). DVs, fold scheme, probe design, and the extension items are user-locked scope; the four AFK-default assumptions (trailing-block-only, k=8, positions-as-predictors-too, both with-template pool variants) are adjustable at relaunch.

## Compute discipline (BINDING — user directive 2026-07-03: "vectorize and parallelize as much as possible")

- **Extraction:** one batched teacher-forced forward per probe-batch (start at batch 8, raise until A100-40 HBM headroom is consumed — probe batches are independent, so the batch axis is free parallelism); ALL per-layer reductions (means, maxes, position gathers, block pools — every family) computed GPU-side inside the same forward, fp32, before a single PCIe transfer per batch. No per-family re-forwards, no per-token host transfers. Set-B generation is one vLLM continuous-batching pass (chunks ≤500), never sequential `generate()`.
- **Fits:** the full ~34.7K-cell map battery + all read-outs run as batched tensor ops on the GPU (TF32) before pod release: X-side caches (standardization + Gram eigendecomposition) computed ONCE per (context-cell × fold) and shared across ALL answer targets; train-fold target PCAs batched across answer cells; all Y-dependent ridge solves stacked into large batched GEMM/eigh calls chunked by a mem-aware cap (the `resolve_chunk_cap` pattern). A per-cell Python loop over fits is BANNED (`.claude/rules/vectorize-many-cell-fits.md`); the batched path must pass `assert_matches_reference`-style bit-equivalence vs a 2–3-cell serial oracle before the full battery runs.
- **Nulls:** all 1,000 permutation draws × all cells × ALL FOUR regimes scored in one batched pass (draw axis stacked into the GEMMs; per-draw weights never persisted); stored-prediction ρ nulls as single `(1000, 50) @ (50, n_cells)` GEMMs. Serial per-draw or per-cell null loops are the #722/#778/#823 failure class — banned.
- **Process-level parallelism:** GPU phases (gen-B → extract → fits/nulls) are sequential by design on ONE GPU (each is minutes-to-an-hour; a second GPU would idle-burn more than it saves — the #778 width-right-sizing rule). CPU-side aggregation/figures overlap the final uploads where dependencies allow. Any phase found running a serial inner loop past ~15 min wall is a STOP-and-vectorize, not a wait.
