---
title: Qwen2.5-7B-Instruct's context-to-answer map outscores Qwen3.5-9B's at matched
  LMSYS data, an edge from a peak at its middle captured layer rather than a uniform
  deficit (MODERATE confidence)
kind: experiment
tags:
- followup-auto
created_at: '2026-08-16T18:05:01Z'
has_clean_result: true
parent_id: 1491
origin_prompt: can you make an issue to just compare the performance for the model
  we've been using vs qwen3.5-9B on same data (on-policy generations) -- lm SYS --
  and compare the R^2 of both. start issue in background. 5k generations and 10k generations
  as 2 data points for each model is enough
workflow: v1
backend: runpod
goal: 'Compare context-to-answer map quality between Qwen2.5-7B-Instruct and Qwen3.5-9B
  (thinking disabled) on matched LMSYS prompts with each model''s own on-policy generations:
  per-layer ridge maps fit at 5k and 10k training generations per model, compared
  on held-out R2 (plus ceiling-normalized secondary, identity+bias/kNN companions,
  shuffled-pairing null).'
relates_to:
- spec-context-as-vector
---
# Qwen2.5-7B-Instruct's context-to-answer map outscores Qwen3.5-9B's at matched LMSYS data, an edge from a peak at its middle captured layer rather than a uniform deficit (MODERATE confidence)
<!-- clean-result-v4 -->
**Methodology:** [docs/methodology/issue_2330.md @ a65f5c9117](https://github.com/superkaiba/explore-persona-space/blob/a65f5c911729bf3c3e1e2491e8891f75dfcad25a/docs/methodology/issue_2330.md) · [gist mirror](https://gist.github.com/superkaiba/c715b157f1f82b2d0e3dea11446424ca)


## Takeaways

- **−0.044 raw R² difference (p = 0.002, N = 1,000):** at 10,000 matched LMSYS training prompts, Qwen2.5-7B-Instruct's held-out map R² is 0.705 vs 0.661 for Qwen3.5-9B — below the plan's 0.05 materiality bar as a point estimate, though the interval reaches past that bar (bounds in Methodology).
- **−0.038 ceiling-normalized (0.763 vs 0.724, p = 0.002):** response-sampling noise does not explain the gap, and the 9B stays far above the 0.60 normalized level that would falsify cross-family stability of map quality.
- **−0.015 best-of-sweep point gap:** the 9B peaks at layer 18 (uncorrected best-of-31 read), but only the 9B was densely swept; the 7B comparator is its best of three captured layers.
- **−0.037 at the 2,048-token cap (p = 0.002, N = 1,000):** full regeneration cut the 9B's test truncation from 20.7% to 6.3%; its R² rose just 0.006.
- **Every fitted map clears its best floor by at least 0.60:** shuffled-pairing nulls sit at −0.017 to −0.022 and the best identity-family baselines at 0.079 (7B) and 0.018 (9B); retrieval acc@1 runs 64.4-73.5% against 0.1% chance but does not follow the R² ordering.
- **+0.024/+0.026 R² from 5,000 to 10,000 training prompts** (near-parallel, so the gap is not data hunger); on the WildChat transfer fold the 10k gap widens to −0.067 (point estimates, N = 998).

## Goal

Compare context-to-answer map quality between Qwen2.5-7B-Instruct (the model every map result in this project is built on) and Qwen3.5-9B with thinking disabled, on matched LMSYS prompts with each model's own on-policy generations: per-layer ridge maps fit at 5,000 and 10,000 training generations per model, compared on held-out R², with a ceiling-normalized secondary read, identity-family and retrieval companions, and a shuffled-pairing null.

**This experiment in context:** the context-to-answer map — a linear map from the last prompt-token activation to the model's own answer-activation profile — is this project's core measurement object ([#779](https://eps.superkaiba.com/tasks/779) established the LMSYS on-policy fit line; [#1491](https://eps.superkaiba.com/tasks/1491) established the scale ladder, the two-draw reliability ceiling, and showed a 14B-to-32B organization effect of 0.076 on this same measure within one family — same rig, directly comparable, and the source of the reused 7B store). [#2329](https://eps.superkaiba.com/tasks/2329) covers Qwen3.5-9B integration; the fit guards come from [#825](https://eps.superkaiba.com/tasks/825); single-draw targets were validated as adequate in [#1073](https://eps.superkaiba.com/tasks/1073).

**Broader narrative:** if map quality is stable across model bundles, the leakage-prediction machinery built on the incumbent transfers to newer models. This is the first cross-family point on that question: the answer is "close but not identical, and the layer profile differs more than the peak numbers do." Any difference here is attributable to the model bundle at the operational capture positions (weights, width, depth, tokenizer, attention pattern, engine stack jointly) — no single mechanism is identified.

## Methodology

**Design:** 2 models × 2 training sizes (5,000 and 10,000 LMSYS prompts; the 5,000 are a prefix of the 10,000; prompt ids identical across models), completions on-policy per model. The dependent variable is held-out variance-weighted test R² of a per-layer ridge map from the last rendered-prompt-token activation to the response-token-mean activation, scored on a shared 1,000-prompt LMSYS test split; primary layers were fixed in the plan as each model's middle capture layer (7B: layer 19 of 28; 9B: index 22 of 32; depth fractions 0.679 and 0.688). Contrasts use a paired bootstrap (1,000 draws, seed 42, one shared resample matrix); companions per cell: two-draw reliability ceiling (regenerations at seeds 43/44), ceiling-normalized R², per-context cosine, k-nearest-neighbor retrieval, identity-family floors, train-mean floor, shuffled-pairing null refit, and a WildChat transfer fold. A follow-up round extended the same paired bootstrap to every cross-model layer pairing and to each model's best captured layer (per-layer ceilings for the normalized read). A second follow-up round added a dense 31-layer sweep of the 9B — fresh capture of every block output over the same banked generations, ridge refit per layer, a locate-the-peak diagnostic with no floors, ceilings, or transfer fold — and a cap-robustness battery: both models regenerated end to end at a 2,048-token cap and re-fit through the full main-run battery.

The 7B side reuses banked captures produced by the same driver in an earlier run: vLLM generation (temperature 1.0, top_p 0.95, n = 1, max 1,024 tokens, seed 42) over the same pinned LMSYS prompt manifest, fp32 teacher-forced capture at layers 14/19/26, stored per split on the HF data repo. The 9B side regenerated and captured fresh in an isolated environment (vLLM 0.27.1, transformers ≥ 5.2.0) because the repo environment does not recognize the `qwen3_5` architecture; thinking was disabled via the chat template (every rendered prompt ends with an empty think block, token-pinned at the gate). Pre-run gates all passed: template pin; full-corpus length scan (1 WildChat prompt over the 7,104-token budget, dropped from both models — WildChat N = 998); capture-port parity against banked 7B rows (worst per-vector cosine 0.99955); a forward-hook vs hidden-state-index probe on the 9B; and an anchor refit that reproduced the 7B 25,000-row layer-19 R² of 0.7250873 through the new pipeline with deviation exactly 0.

**Training:** **N/A — no model training.**

**Evaluation:**

The dependent variable is held-out R² of a per-layer linear (ridge) map from the last rendered-prompt-token activation to the model's own response-token-mean activation — the parent ladder's measure, kept identical so both models and the 25,000-row anchor read on one scale. Both models are scored on the same 1,000 held-out LMSYS prompts, each predicting its own on-policy generations, plus a 998-prompt WildChat transfer fold. Preprocessing: one pinned prompt manifest, per-model chat-template rendering (9B thinking disabled), fp32 capture, variance-weighted pooling across hidden dimensions.

| Parameter | Value | Source |
|---|---|---|
| Decoding (both models) | temperature 1.0, top_p 0.95, n = 1, max_tokens 1,024, seed 42, vLLM | driver constants `issue1491_ladder_generate_capture.py:137-150`; recipe identity with the banked 7B targets (plan §11) |
| 9B rendering / runtime | `enable_thinking=False` threaded into every chat-template call (rendered think-suffix token ids pinned by the template gate); isolated venv: vLLM 0.27.1 (exact install pin), transformers 5.15.0 (reconstructed — the pin's own floor is ≥ 5.5.3 and 5.15.0 was the newest release at the 2026-08-17 venv build; the launcher's version probe printed the realized value only to the pod log, which was not persisted) | committed launcher `run_records/launch_issue_2330_p1.sh` (venv pin + version probe); `run_records/run_meta.json` `template_pin` (suffix token ids) |
| Ceiling draws | seeds 43 and 44 over the 1,000 test prompts, per model | plan §6, parent-ladder convention |
| Capture | fp32; context vector = last rendered-prompt-token activation; answer profile = response-token mean; layers 7B {14, 19, 26} of 28, 9B {16, 22, 30} of 32 (depth-fraction matched; 9B index 16 is a full-attention block output, 22/30 are linear-attention block outputs) | plan §11; driver |
| Ridge fit | streaming fp64 primal; λ grid logspace(−3, 8, 23), validation-selected (400-prompt split); selected λ: 7B 3,162 at all layers, 9B 1,000 at indices 16/22 and 3,162 at 30; no grid-edge hits, no extensions | `issue1491_ladder_fits.py:127`; fits JSONs |
| Bootstrap | paired, 1,000 draws, seed 42, one shared resample matrix; ceilings treated as fixed per-model scalars | `contrasts.json` meta |
| Retrieval companion | k-nearest-neighbor acc@k at k = 1, 5, 10, 50; cosine and euclidean; pool = the 1,000 held-out test targets; chance = k/1,000 (0.1% at k = 1) | fits JSONs `knn_retrieval` blocks, via `analysis/mapping_baselines.knn_retrieval` |
| Splits (realized) | train 10,000 / 5,000 (nested), val 400, test 1,000, WildChat 998 | `split_ids.json` (sha256 789ae56b34); per-fit count pins |
| n_train / d (recorded per fit) | 7B (d = 3,584): 2.79 at 10k, 1.40 at 5k; 9B (d = 4,096): 2.44 and 1.22 | fits `n_vs_d` blocks |
| Contrast CI bounds | 10k primary: raw −0.0544 to −0.0320; normalized −0.0501 to −0.0256. 5k: raw −0.0573 to −0.0340. Within-model 10k−5k: 7B +0.0219 to +0.0253, 9B +0.0239 to +0.0292 (95%, paired bootstrap) | `contrasts.json` |
| Revision-round recomputes | common-uncapped intersection read (N = 791), intrusion-exclusion sensitivity (N = 951), per-prompt cosine wins, response-length medians | `eval_results/issue_2330/revision_recomputes.json`; script `issue2330_revision_recomputes.py` |
| Cross-layer contrast round | same paired bootstrap over the shared draws, every cross-model layer pairing + each model's best captured layer; per-(model, layer) ceilings; best-layer 10k bounds raw −0.0491 to −0.0243, normalized −0.0426 to −0.0153; 5k raw −0.0462 to −0.0216; the 5k grid matches the 10k grid in sign throughout; depth-matched cells match `contrasts.json` to 1e-9 | `eval_results/issue_2330/crosslayer/crosslayer_contrasts.json` @ `a87d5598b4`; script `issue2330_crosslayer_contrasts.py` |
| Dense 9B layer sweep | every block output, layers 0-30, fresh fp32 capture over the banked 9B generations (2-way sharded per split, matching the banked generation store); same ridge recipe, splits, and λ grid; per-layer test R² + selected λ only (locate-the-peak diagnostic); streaming fits checkpointed per unit (`EPM_N1M_STREAM_CKPT_EVERY=1`, 62 units ~194 s) | `dense_sweep/matched_fits_q35_dense.json`; captures `issue2330_matched/qwen35_9b_dense/` |
| 2,048-token-cap robustness battery | both models regenerated at max_tokens 2,048 (same prompts, seed, decoding otherwise identical); train/val/test re-captured and re-fit; WildChat fold + two-draw ceilings inherited from the original stores (regeneration scope excluded them), so those two reads are never quoted as 2,048-cap quantities | cap2048 fits JSONs, `store_prefix_override` blocks |
| cap2048 contrast bounds | 10k raw −0.0476 to −0.0263, normalized −0.0426 to −0.0194 (inherited ceilings); 5k raw −0.0505 to −0.0293 (95%, paired bootstrap, shared seed-42 resample matrix) | `cap2048/contrasts_cap2048.json` |

Cap-hit fractions were recorded per model and split, and instead of the default over-2% regeneration trigger the plan inherited the parent ladder's two-arm truncation-restriction control (a test-restricted read plus an untruncated refit), preserving recipe identity with the banked 7B targets. The 9B's responses also run longer on the identical prompts (median 1,154 vs 714 characters), so the response-token-mean target averages a longer window for the 9B — a bundle difference adjacent to, and broader than, the cap-hit asymmetry. Language-intrusion audit (both arms, test split): 45 of 1,000 prompts themselves contain CJK characters; among the 955 remaining, completions containing CJK number 29 (3.04%) for the 7B and 28 (2.93%) for the 9B — arm-symmetric. A judge-rate-style zeroed recount does not apply to an R² outcome, but an exclusion sensitivity does: dropping the 49 prompts whose completion carries CJK in either arm moves the 10k primary difference from −0.044 to −0.046 (0.709 vs 0.663, N = 951), so intrusion does not account for the gap. Intruded rows are identifiable by `ci` index in the linked raw-completion files (regex: CJK unified ideographs + extension A, compatibility ideographs, kana, hangul).

I acknowledge the fired conciseness WARNs (Takeaways bullet-length, per-result prose band, total-prose budget): the run plus three follow-up rounds carry five contrasts and five control families across ten result sections, plus the nine open review-concern caveats enumerated in the footer, and each result block stays inside the hard caps. The bootstrap-delta companion panel is deliberately linked rather than embedded to keep one figure per result, and the hero figure's sidecar categorical-slot WARN reflects two series drawn across four tick positions per axis — all four cells render.

**Data extraction:** fits stream the capture stores from the HF data repo with per-split count pins against the committed id lists and a matched-id assert across models. The contrast stage re-verified every committed cell × layer R² from the stored per-context predictions (maximum absolute deviation 2.0e-7) and test-id identity before bootstrapping. Raw-output spot check (5 shared test prompts, drawn with seed 42): all 10 completions are coherent, on-topic, and answer in the prompt's language; one roleplay prompt hit the token cap in both models; 0 of 5 sampled prompt pairs were flagged.

**Sample training/evaluation data + completions:** the rows below are a random sample (5 of the 1,000 shared test prompts, seed 42 — not cherry-picked), shown as ~12-15-word excerpts sanitized for context hygiene (LMSYS is real user text; `NAME_1`-style tokens are LMSYS's own upstream PII redaction, present in the stored prompts). Full rows: [7B raw completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/815ff6d976c686af8672b27cfdfb1ce6b419c02c/issue1491_scale_ladder/scale7_refit/test_1000/raw_completions) (pinned) and [9B raw completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/197f1dd57fe07be48deef116cb8ab142ed24ce35/issue2330_matched/qwen35_9b/test_1000/raw_completions) (pinned), keyed by the `ci` field.

<details>
<summary>5 sampled test rows (prompt + both models' completions, truncated excerpts)</summary>

| Row | Shared prompt (excerpt) | Qwen2.5-7B completion (excerpt, finish reason) | Qwen3.5-9B completion (excerpt, finish reason) |
|---|---|---|---|
| ci=654 | Eu tinha 2 cachorros, o Pete e o repete, o Pete morreu, [truncated] | Parece que você tinha dois cachorros, o Pete e o Repete. Se o Pete morreu, [truncated] (stop) | Quem ficou foi o **Rete** (ou **Repete**, dependendo de como o nome dele foi escrito [truncated] (stop) |
| ci=114 | You are NAME_1, a renowned detective working alongside the brilliant NAME_2. You [truncated] | **NAME_2:** [Opening the letter from Inspector NAME_3, you and NAME_1 exchange glances.] Inspector NAME_3 has [truncated] (length) | **Context:** *The rain taps a rhythmic code against the window of our shared office at [truncated] (length) |
| ci=25 | o que é uma cidade-estado? | Uma cidade-estado é um tipo de entidade política que consiste em uma única cidade (ou [truncated] (stop) | Uma **cidade-estado** é um tipo de estado cujo limite político coincide com o de uma [truncated] (stop) |
| ci=759 | Show me how to use awk to extract file name from a [truncated] | Certainly! The `awk` command can be used to extract various parts of a file path, [truncated] (stop) | To extract the file name from a path using `awk`, the most efficient approach is [truncated] (stop) |
| ci=281 | Find the answers for the question words - 'who, what, when, where, [truncated] | Here is the output for the sentence "NAME_4 is the brother of NAME_1." 1. NAME_4 [truncated] (stop) | 1. NAME_4 - who - NAME_4, NAME_4 - what - brother, NAME_4 - when - [truncated] (stop) |

</details>

## Results

### The primary contrast resolves to incumbent-higher: −0.044 raw, −0.038 ceiling-normalized

Held-out test R² per model × training-size cell at the plan-fixed middle layers on the shared 1,000-prompt LMSYS test split, raw (with two-draw ceilings and shuffled-pairing nulls) and divided by each model's ceiling.

![Held-out map R2 per cell, raw and ceiling-normalized](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8713bfdb2dd587a8dda4104c28db6ddd1d80c6a1/figures/issue_2330/hero_r2_raw_and_normalized.png)

> **Figure.** *The incumbent is higher in all four cells, raw and ceiling-normalized.* Context-to-answer map R² at matched training data (7B layer 19 of 28, 9B index 22 of 32; N = 1,000 test prompts). Dashed: two-draw reliability ceilings (0.924 / 0.913); dotted: shuffled-pairing nulls. Right panel: the same values divided by each model's ceiling.

| Contrast (primary layers) | Raw Δ | Normalized Δ | p (two-sided) |
|---|---|---|---|
| 9B − 7B at 10k (primary) | −0.0436 | −0.0382 | 0.002 |
| 9B − 7B at 5k | −0.0465 | −0.0417 | 0.002 |

Both intervals sit wholly below zero (bounds in the Methodology table), so the plan's three-way classification resolves to incumbent-higher. The gap is below the plan's 0.05 materiality threshold only as a point estimate — the interval reaches past that threshold at its lower end. Normalization moves it little; the 9B's normalized 0.724 is far above the 0.60 cross-family falsification level.

Why this test: both models score on one shared test split, so the contrast is a paired bootstrap over the 1,000 shared prompts (each draw stays paired); the error bars are per-cell intervals from the same draws.

Per-draw bootstrap distributions: [boot-delta panel](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8713bfdb2dd587a8dda4104c28db6ddd1d80c6a1/figures/issue_2330/boot_delta_distributions.png). The intrusion audit is arm-symmetric and excluding intruded rows slightly widens the difference (Methodology); the per-context strip in the next result is the per-unit view behind these bars.

### Per-context prediction quality shifts lower for the 9B as a distribution (median cosine 0.922 vs 0.954 at 10k), with a heavier low tail

Per-test-prompt cosine between the predicted and realized answer profile — the per-unit view behind the aggregate bars — 1,000 points per cell at the primary layers, medians marked.

![Per-context cosine between predicted and realized answer profile, 1000 points per cell](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8713bfdb2dd587a8dda4104c28db6ddd1d80c6a1/figures/issue_2330/percontext_cosine_strip.png)

> **Figure.** *The 9B cloud sits lower as a whole, with a heavier low-cosine tail and no separate collapsed mode visible.* Per-context cosine(predicted, realized) for all four cells (N = 1,000 each; diamonds mark medians).

Medians recomputed from the stored predictions: 0.950 and 0.954 for the 7B at 5k/10k vs 0.915 and 0.922 for the 9B; 10th percentiles 0.896/0.904 vs 0.846/0.857. The shift is distributional rather than prompt-uniform: at 10k the 9B still has the higher cosine on 185 of the 1,000 prompts, and 30 of its rows fall below cosine 0.8 against 5 for the 7B — a generally lower cloud with a heavier tail, not a separate collapsed mode.

### The gap is the 7B's mid-stack peak (0.705 vs 0.612/0.619 at its flanks), not a uniform 9B deficit

Held-out R² at all three depth-fraction-matched capture layers per model (10k fits solid, 5k dashed), plotted against depth as a fraction of each model's stack.

![Per-layer map R2 at depth-matched capture layers for both models](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8713bfdb2dd587a8dda4104c28db6ddd1d80c6a1/figures/issue_2330/per_layer_r2.png)

> **Figure.** *Only the middle depth-matched pair favors the 7B.* The 7B (blue) peaks sharply at depth 0.68 (0.705 at 10k); the 9B (orange) is flat and declines with depth (0.668 / 0.661 / 0.635). The star marks the 9B's shallowest capture point, a full-attention block output.

A follow-up round bootstrapped every cross-model layer pairing: at 10k the 9B wins all six against 7B layers 14 and 26 (+0.016 to +0.056, p ≤ 0.010) and loses all three against layer 19 (−0.037 to −0.070, p = 0.002). Each model's best captured layer still favors the 7B (9B − 7B: −0.037 raw, −0.030 ceiling-normalized, p = 0.002). Within-model, the 7B peaks sharply (+0.093, layer 19 over 14); the 9B's top two layers are near-tied (+0.006, index 16 over 22, p = 0.032).

This read covers only three captured depths; the dense sweep below finds a 9B peak between them, at layer 18. Whether the flat profile reflects the 9B's hybrid attention pattern or any other bundle component is not identified.

Per-unit exemption: the per-context strip above is this contrast's per-unit view; the full grid is in the pinned cross-layer JSON (footer).

### A dense sweep of all 31 9B layers finds its peak at layer 18, off the captured grid, narrowing the best-layer point gap to −0.015

Held-out test R² at every Qwen3.5-9B block output (layers 0-30), fit at both training sizes over the same banked generations, with the depth-matched pick (index 22) and the realized peak (layer 18) marked and the 7B's layer-19 10k fit as a reference level.

![Dense per-layer map R2 for Qwen3.5-9B with the layer-18 peak and layer-22 pick marked](https://raw.githubusercontent.com/superkaiba/explore-persona-space/5a9c85994f4a64a3919cd0e8fcce04078d34f2ec/figures/issue_2330/dense_layer_profile.png)

> **Figure.** *Only the 9B was densely swept; its profile peaks at layer 18, off the captured grid.* Dense per-layer held-out R² (N = 1,000; solid 10k, dashed 5k fit). The blue dashed reference is the 7B's best of three captured layers (layer 19, 10k); the selected layer-18 maximum is an uncorrected best-of-31 read sitting 0.015 below it.

The peak sits at layer 18 at both sizes (0.690 at 10k, 0.669 at 5k), so the depth-matched index 22 understates the 9B by 0.028 and its best captured layer (index 16) by 0.022. Against the 7B's 0.705 the gap narrows to −0.015 — an uncorrected best-of-31 point read that flatters the 9B: only the 9B was swept densely, and the 7B's own true peak may sit off its three captured depths.

The fresh capture reproduces the main run's index-16 and index-22 values within 0.0004 (max 0.00035). The profile is this result's per-unit view (31 points per series); selected per-layer regularization is in the committed JSON, and the sweep's 62 fits ran serially, not batched (the open `dense-fit-loop-unbatched` concern, ~194 s).

### The 9B's three-to-four-times-higher truncation rate does not appear to explain the gap: −0.040 on shared uncapped prompts

Fraction of generations ending at the 1,024-token cap, per model and split (kept for recipe identity with the banked 7B targets; the plan swapped the default over-2% regeneration trigger for a two-arm restriction control).

![Generation cap-hit fraction per model and split](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8713bfdb2dd587a8dda4104c28db6ddd1d80c6a1/figures/issue_2330/cap_hit_fractions.png)

> **Figure.** *The 9B truncates roughly 3-4x more often under the identical decoding recipe.* Cap-hit fraction: Qwen3.5-9B 20.7% test (207/1,000), 20.8% val (83/400), 25.8% train (2,583/10,000) vs Qwen2.5-7B 5.6%, 6.3%, 7.9%.

The cleanest control scores both models on the same 791 test prompts left uncapped in both arms: 0.700 vs 0.660, difference −0.040. The per-arm controls agree: each model's own untruncated prompts give 0.701 (7B, N = 944) vs 0.660 (9B, N = 793), and refitting with capped train and validation rows removed gives 0.701 vs 0.657, near the full-read −0.044. Caveats: the per-arm reads compare non-identical prompt subsets; the untruncated refits train on 9,214 vs 7,417 rows, a mismatch worth roughly 0.008 against the 9B at the measured data-scaling slope; and the reliability ceilings were not recomputed on the restricted subsets.

Per-unit exemption: the outcome is binary per prompt, so the plotted fractions are its complete summary; per-prompt finish reasons are in the pinned raw-completion files.

### The gap survives a doubled generation cap: −0.037 at 10k on fully regenerated completions

Primary-layer test R² per cell on completions regenerated at a 2,048-token cap, next to the original 1,024-token cells (left), and the fraction of generations ending at the cap per model and split at both caps (right).

![Map R2 at both generation caps and cap-hit fractions per model and split](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fb4b88f39591c904f7963da386a28260eff2e355/figures/issue_2330/cap2048_comparison.png)

> **Figure.** *Doubling the cap barely moves either model's map quality.* Left: open markers are the 1,024-token cells, filled the 2,048-token regenerations (N = 1,000; the 7B pairs coincide). Right: cap-hit percentage per model and split at both caps. Ceiling and WildChat reads were inherited from the original 1,024-cap stores, not regenerated.

Regenerated end to end at the doubled cap (same prompts, seed, decoding), the 9B's test truncation falls from 20.7% to 6.3% — roughly the 7B's original rate — yet its R² rises only 0.006 at both sizes, and the 7B moves under 0.001. The paired contrast reads −0.037 at 10k and −0.040 at 5k (both p = 0.002, N = 1,000), near the original −0.044 and −0.047.

The 9B still answers longer (median 1,116 vs 723 characters), so the longer-target-window difference persists. A CJK recount on the regenerated test pools is near-symmetric across arms (7B 34 vs 9B 24 intruded of 955 eligible); excluding rows intruded in either arm leaves the 10k difference at −0.037. Per-context predictions and targets for these cells: the pinned `preds_cap2048` npz (footer).

### Both models gain about +0.025 from 5k to 10k, so the gap is not data hunger

Held-out R² at the primary layers versus training-set size, with the 7B's 25,000-row anchor from the parent ladder (reproduced through this pipeline with deviation exactly 0).

![Map R2 versus training set size for both models with the 7B 25k anchor](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8713bfdb2dd587a8dda4104c28db6ddd1d80c6a1/figures/issue_2330/r2_vs_n.png)

> **Figure.** *Doubling training data moves both models by about +0.025.* Held-out R² vs training-set size (7B +0.024, 9B +0.026, raw); the anchor diamond marks the 7B 25k fit (0.725). Lines are two-point segments, not fitted curves.

The near-parallel gains (+0.024 for the 7B, +0.026 for the 9B, raw) say the gap is not a data-hunger artifact at these training sizes, and the 7B's 25,000-row anchor shows the incumbent's curve still rising past the sizes tested here. The four LMSYS cells here are the same four fits shown point-by-point in the per-context cosine strip above, which is their per-unit view.

### Every fitted map clears its best floor by at least 0.60

Baseline floor R² per cell at the primary layers: shuffled-pairing nulls, train-mean, and the three identity-family baselines (the fitted maps appear in the hero figure).

![Baseline floor R2 per cell for five floor families, all at or below near-zero, far under the fitted maps](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8713bfdb2dd587a8dda4104c28db6ddd1d80c6a1/figures/issue_2330/floors_panel.png)

> **Figure.** *No floor family comes close to the fitted maps.* Baseline held-out R² per cell (primary layers, N = 1,000): shuffled-pairing nulls and train-mean sit near zero, per-dimension scaled identity barely above zero, identity-plus-bias and identity-copy far below zero.

At 10k the shuffled-pairing nulls read −0.022 (7B) and −0.018 (9B), train-mean −0.02, and the best identity-family baseline (per-dimension scaled identity) 0.079 and 0.018; identity-plus-learned-bias reads −0.89 and −2.07 (input and output dimensions match within each model). The smallest margin between any fitted map and its best floor is 0.60 (the 7B at 5k), so the maps capture structure no identity-family or null baseline reproduces.

Per-unit exemption: each floor is a per-cell baseline scalar; the fitted maps' per-unit view is the per-context strip above, and the per-context targets the floors are computed from are in the pinned npz (footer).

### Retrieval acc@1 runs 64.4-73.5% against 0.1% chance but does not track the R² ordering

Nearest-neighbor retrieval per cell: the fraction of held-out prompts whose predicted answer profile lands closest to its own realized profile among the 1,000 test targets (cosine shown; euclidean values in the fits JSONs).

![Retrieval accuracy at one per cell against a near-zero chance level](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8713bfdb2dd587a8dda4104c28db6ddd1d80c6a1/figures/issue_2330/retrieval_acc1.png)

> **Figure.** *All four cells retrieve the right answer profile on the first try roughly 70% of the time.* Cosine retrieval acc@1 per cell among the 1,000 held-out test targets; chance (dotted) is 0.1%.

Across cells and both metrics acc@1 spans 64.4-73.5%. Cosine favors the 9B at 5k (70.0% vs 68.3%) and is near-tied at 10k (72.5% vs 73.1% for the 7B), while euclidean favors the 7B at both sizes (73.5% vs 70.7% at 10k). Retrieval and R² dissociating this way suggests part of the R² gap reflects activation geometry or scale rather than weaker answer discrimination.

Per-unit exemption: acc@k is binary per prompt, so the plotted per-cell fractions are its complete summary; the per-context predictions behind it are in the pinned npz (footer).

### The gap widens to −0.067 on the WildChat transfer fold, where the 9B's depth pattern inverts

Held-out R² of the same 10k fits scored in-distribution (LMSYS test) versus on the WildChat transfer fold (998 prompts; fit never sees WildChat), at the primary layers.

![In-distribution versus WildChat transfer R2 per cell](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8713bfdb2dd587a8dda4104c28db6ddd1d80c6a1/figures/issue_2330/wc_transfer_vs_indist.png)

> **Figure.** *Every cell drops on transfer, the 9B cells most at the plotted primary layers.* Fold-labeled reads: LMSYS in-distribution (green) vs WildChat transfer (red), per cell at the primary layers only.

At 10k the transfer read is 0.598 (7B) vs 0.531 (9B) — a 0.067 gap, point estimates only (no bootstrap was run on this fold). The depth pattern inverts for the 9B on transfer: its transfer R² rises with depth (0.517 / 0.531 / 0.556) while the 7B again peaks mid-stack (0.478 / 0.598 / 0.503), so at the shallow depth-matched pair the 9B transfers better (0.517 vs 0.478); per-layer, the 9B's largest transfer drop is at its shallow layer, not the middle. In-distribution numbers carry no generalization claim; every headline above is labeled with its fold.

Per-unit exemption: the per-context strip above is the in-distribution per-unit view; the transfer fold's per-unit evidence (per-context WildChat predictions and targets) is in the same pinned npz (footer).

---

**Repro:** code on branch `issue-2330` @ `0ca8b47888` (generation/capture + fits) with hot-fix `81c0f49d64` (HF 409 retry); contrasts @ `41149663f6`; revision-round recomputes + relabeled figures @ `8713bfdb2d`; cross-layer grid contrasts @ `a87d5598b4`; fu1 dense-sweep + cap2048 round @ `df2abd8a47` (fits, cap-hit aggregates) and `fb4b88f395` (cap2048 contrasts, intrusion recount, fu1 figures; wiring `f31bed0121`), dense-profile figure relabel @ `5a9c85994f` — scripts at the branch tip: [split ids](https://github.com/superkaiba/explore-persona-space/blob/8713bfdb2dd587a8dda4104c28db6ddd1d80c6a1/scripts/issue2330_split_ids.py), [generation/capture](https://github.com/superkaiba/explore-persona-space/blob/8713bfdb2dd587a8dda4104c28db6ddd1d80c6a1/scripts/issue2330_qwen35_generate_capture.py), [matched fits](https://github.com/superkaiba/explore-persona-space/blob/8713bfdb2dd587a8dda4104c28db6ddd1d80c6a1/scripts/issue2330_matched_fits.py), [contrasts](https://github.com/superkaiba/explore-persona-space/blob/8713bfdb2dd587a8dda4104c28db6ddd1d80c6a1/scripts/issue2330_contrasts.py), [figures](https://github.com/superkaiba/explore-persona-space/blob/8713bfdb2dd587a8dda4104c28db6ddd1d80c6a1/scripts/issue2330_figures.py), [revision recomputes](https://github.com/superkaiba/explore-persona-space/blob/8713bfdb2dd587a8dda4104c28db6ddd1d80c6a1/scripts/issue2330_revision_recomputes.py), [cross-layer contrasts](https://github.com/superkaiba/explore-persona-space/blob/a87d5598b47f58da2b3d31874f3ef9895c9a3bdb/scripts/issue2330_crosslayer_contrasts.py). Committed artifacts (same branch; [pinned tree](https://github.com/superkaiba/explore-persona-space/tree/8713bfdb2dd587a8dda4104c28db6ddd1d80c6a1/eval_results/issue_2330)): `eval_results/issue_2330/matched_fits_{q25_n5k,q25_n10k,q35_n5k,q35_n10k}.json`, `contrasts.json`, `split_ids.json`, `cap_hit/` (6 files), `revision_recomputes.json`, the cross-layer grid follow-up's `crosslayer/crosslayer_contrasts.json` @ [`a87d5598b4`](https://github.com/superkaiba/explore-persona-space/blob/a87d5598b47f58da2b3d31874f3ef9895c9a3bdb/eval_results/issue_2330/crosslayer/crosslayer_contrasts.json), and the fu1 round's ([pinned tree at fb4b88f395](https://github.com/superkaiba/explore-persona-space/tree/fb4b88f39591c904f7963da386a28260eff2e355/eval_results/issue_2330)) `dense_sweep/matched_fits_q35_dense.json`, `cap2048/matched_fits_{q25_n5k,q25_n10k,q35_n5k,q35_n10k}_cap2048.json`, `cap2048/contrasts_cap2048.json`, `cap2048/intrusion_cap2048.json`, and `cap_hit_cap2048/` (6 files) — fu1 scripts: [dense + cap2048 fits wiring](https://github.com/superkaiba/explore-persona-space/blob/fb4b88f39591c904f7963da386a28260eff2e355/scripts/issue2330_matched_fits.py), [cap2048 intrusion recount](https://github.com/superkaiba/explore-persona-space/blob/fb4b88f39591c904f7963da386a28260eff2e355/scripts/issue2330_cap2048_intrusion.py), [figures incl. fu1 panels](https://github.com/superkaiba/explore-persona-space/blob/fb4b88f39591c904f7963da386a28260eff2e355/scripts/issue2330_figures.py). fu1 provenance: pod-2330-fu1 (2× H200, ~8 h wall, ~16 GPU-h); one mid-round crash-fix — two legs sharing a scratch directory poisoned 3 dense train-capture chunks, deleted and re-captured (repair verified: 20 of 20 chunks healthy), workflow gap filed as [#2350](https://eps.superkaiba.com/tasks/2350). fu1 HF prefixes, pinned at data-repo revision `b99d86de23` (all four verified resolvable at that revision via a live tree listing): `issue2330_matched/qwen35_9b_dense/` (31-layer captures, all splits), `issue2330_matched/qwen35_9b_cap2048/` + `issue2330_matched/q25_cap2048/` (2,048-cap generations + captures), `issue2330_matched/analysis_tensors/preds_cap2048/` (4 npz; [pinned tree](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/b99d86de23240b160be4d0094db8a71b51fb98c6/issue2330_matched)). HF data repo `superkaiba1/explore-persona-space-data` (listings verified live at write time, revision `197f1dd57f`): `issue2330_matched/qwen35_9b/{train_10k,val_400,test_1000,wc_test_1k,ceiling_draws/seed43,ceiling_draws/seed44}/` @ `197f1dd57f`, each with `final_token_capture/` + `raw_completions/` @ `197f1dd57f`, and `issue2330_matched/analysis_tensors/preds/` @ `197f1dd57f` (4 npz, per-context predictions + targets; [pinned tree](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/197f1dd57fe07be48deef116cb8ab142ed24ce35/issue2330_matched)). Models: `Qwen/Qwen2.5-7B-Instruct` (banked captures) and `Qwen/Qwen3.5-9B` @ HF sha `c202236235`, thinking disabled. Reused store from [#1491](https://eps.superkaiba.com/tasks/1491): `issue1491_scale_ladder/scale7_refit` @ `815ff6d` — fit: same driver, decoding recipe, seed, and prompt manifest as the 9B side; matched prompt ids by construction. Compute: ≈ 4-5 GPU-h (pod-2330, 2× H200; generation+capture ~1.5 h wall, fits ~0.4 h on 1 GPU) + ≈ 16 GPU-h for the fu1 round; contrasts/figures CPU. Caveats carried from the nine open review concerns: `n-vs-d-marker-overstatement` — n_train/d is recorded per fit, not asserted on the fit path (all recorded ratios 1.22-2.79, above 1); `cap-hit-artifact-provenance-stale` — the committed cap-hit JSONs embed pod-absolute source paths and a producer-commit field predating their final producer (counts independently re-verified in review: 56/1,000, 25/400, 786/10,000 on the 7B; the in-repo `cap_hit/` copies are the portable record); `p4-provenance-unchecked` — the contrast stage verified per-cell R² parity (max 2.0e-7) and test-id identity but not split-id sha or anchor state across cells; `gpu-phase-resume-contract` — an all-skipped generation resume would bypass the run's terminal gates (rig-level gap; this run generated fresh with nonzero counts, and the cap-hit aggregates were recounted from raw finish_reason in review and again in the revision round — counts match); `q25-store-revision-unpinned` — the 7B store streamed at the repo's default Hub revision with the pin recorded as metadata (the anchor refit binds layer-19 train content exactly; flanking layers and auxiliary splits are bound by counts and id sets), and the fu1 cap2048 7B regeneration + fits likewise streamed the default Hub revision; `p1-smoke-shard-gate-unpinned` — the smoke-shard gate records a pass for any successful no-upload run and a later failing rerun does not retire a prior passing sentinel (rig-level gap; the production gates ran and passed in this run); `cap-hit-production-path-guarded` — the cap-hit aggregator/fits/figures path defaults do not connect across roots, so the realized run wired them by explicit flags (the committed `cap_hit/` copies plus the fits' truncation blocks are the durable record); `blocker-invariant-pytests-missing` (persisted at review round 5) — the rig's blocker-closing invariants (sentinel fingerprint, cap-hit missing-field rejection, count pins) are covered by in-script selftests rather than committed pytests (rig test debt; no bearing on the results); `dense-fit-loop-unbatched` — the fu1 dense sweep's 62 per-layer ridge fits ran as a serial loop rather than a batched solve (accepted residual; ~194 s total). Closed during review: `named-assemble-helper-deviation` — the plan-named assembly helper was substituted by a local equivalent-or-stronger wrapper without a deviation-list entry (reviewer-judged equivalent). Additional caveats: single-draw targets per model (ceiling-normalized read addresses this); generation/capture engine stacks differ across arms (banked 7B: vLLM 0.11.0 / transformers 4.57.6; 9B: vLLM 0.27.1 / transformers ≥ 5.2.0 — inseparable from the model swap); different tokenizers (matched prompts, not matched token counts) and hidden dimensions (3,584 vs 4,096); LMSYS carries no train/test near-duplicate filter (inherited from the parent manifest); data tier 1 (real user prompts).

**Context:** originating prompt (user, 2026-08-16, verbatim): `can you make an issue to just compare the performance for the model we've been using vs qwen3.5-9B on same data (on-policy generations) -- lm SYS -- and compare the R^2 of both. start issue in background. 5k generations and 10k generations as 2 data points for each model is enough`. Task created 2026-08-16; run completed 2026-08-17 (pod-2330, 5 environment crash-fix rounds during gate bring-up, zero experiment-code changes; one mid-run HF 409 hot-fix). Interpretation revised 2026-08-17 after critique round 1 (truncation-control intersection read, intrusion-exclusion sensitivity, distributional-shift calibration, figure relabels); zero-GPU cross-layer contrast follow-up (full layer grid + best-layer contrast) folded 2026-08-17; cheap-band follow-up round `fu1-dense-cap2048` (proposer-initiated via the automatic sub-20-GPU-h band, no user prompt; dense 9B layer sweep + 2,048-cap regeneration battery, ~16 GPU-h) folded 2026-08-17. Lineage: reuses the [#1491](https://eps.superkaiba.com/tasks/1491) scale-ladder manifest and 7B store; first Qwen3.5-family measurement in this project.


