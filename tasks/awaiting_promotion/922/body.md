---
title: Answer-time activation trajectories are forecastable from the prompt alone
  through 32 tokens with a rolled per-layer affine map (HIGH confidence)
kind: experiment
tags:
- followup-auto
created_at: '2026-07-03T10:21:41Z'
has_clean_result: true
parent_id: 841
origin_prompt: Run an issue in the background which is 841 but predicting the next
  token activation from previous token activation, one mapping per layer, trying the
  different options
goal: 'On Qwen-2.5-7B-Instruct, test whether the future (answer-time) activation trajectory
  can be predicted WITHOUT generating, comparing per-layer transition structures —
  (a) a global next-token map rolled from the last prompt token, (b1) an additively
  context-conditioned map (global operator + context drift), (b2) an operator-valued
  map where the context RESHAPES the transition itself (A(c) via FiLM-diagonal / low-rank
  delta / gated mixture of K operators, capacity-matched to b1), and (c) a direct
  per-horizon map c -> h_{t+k} (no recursion; #779''s answer-profile map = its horizon-mean
  nested baseline) — over the identity/ridge/MLP ladder with no token-identity inputs;
  evaluate single-step Delta R^2, rollout/horizon skill vs true on-policy answer activations,
  and behavior read-out vs #779''s direct context->trait predictor.'
relates_to:
- spec-context-as-vector
---
# Answer-time activation trajectories are forecastable from the prompt alone through 32 tokens with a rolled per-layer affine map (HIGH confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_922.md](https://github.com/superkaiba/explore-persona-space/blob/8cf01cb8bd28ababe86d06e42ec25a6feaf9e23f/docs/methodology/issue_922.md) · [gist](https://gist.github.com/superkaiba/8583bdcafcf63dc2b61e4c7cb6d5793f)

## Takeaways

- A per-layer affine map rolled from the last prompt token forecasts answer-time activations at 32 tokens with skill 0.38–0.42 over the frozen-state null — no token generated.
- Off-corpus the roll keeps skill 0.35–0.44 and beats direct per-horizon maps by 0.11–0.26 at all six read-out blocks (in-corpus they tie); the rolled dynamics are what generalizes, and after the pairing repair this holds for all three traits with exact provenance.
- Rolling forward adds nothing for trait read-out: the rolled read sits at or below the frozen prompt-end read in all six trait-by-mode cells.
- Context reshapes the transition operator (+0.003–0.012 over the additive twin at all six layers), but no conditioned or nonlinear map beats the global affine one; conditioned rolls diverge.
- Every closed-form iterated map contracts (spectral radius 0.98–0.99) and every gradient-fit conditioned operator expands (1.29–2.13); the map's fixed point is a norm-shrunken mean answer state (cosine 0.985–0.999), and its thin slow shell (8–18 modes at |λ| > 0.95) concentrates trait directions and context identity 30–125× above a random-subspace null.
- Scope: 624 planned transfer windows shrank to 432 (many-shot exemplars unrecoverable). A repair round regenerated properly paired completions for the 288 mismatched-question windows; they match the evil-only exact-provenance companion on every criterion, and the mismatch had shifted horizon-32 roll skill by at most 0.05 without changing any read.

## Goal

- **This experiment in context:** Tests whether the answer-time residual-stream trajectory can be forecast from the prompt-end state alone — no token generated — and which per-layer transition structure carries the forecast: a global rolled map, additive or operator-valued context conditioning, or direct per-horizon regression. It extends the depth-axis affine-ladder result of [#841](https://eps.superkaiba.com/tasks/841) to the position axis, and uses [#779](https://eps.superkaiba.com/tasks/779)'s context-to-trait predictor and answer-profile map as read-out reference and nested baseline.
- **Broader narrative:** Whether a context's downstream behavioral influence is carried by a compact, forecastable activation-state object; if so, behavior reads and context screening can run at prompt time, before generation.

## Methodology

**Design:** One model (Qwen-2.5-7B-Instruct, frozen throughout). Per-layer next-position maps are fit on 5,000 real LMSYS chat contexts (4,000 fit / 500 validation / 500 test, split by context, seed 42) and evaluated in-corpus (held-out test contexts) and out-of-corpus (432 persona-condition windows). The manipulated variable is the transition structure at matched data and recipe: a global per-layer affine map whose only input is the current state; an additively conditioned map (global operator plus a context-dependent drift term); three operator-valued conditioned maps in which the context vector `c = h_(l,T)` (the same layer's last-prompt-token state) reshapes the transition itself — a diagonal FiLM gate, a low-rank delta, and a gated 2-operator mixture — each capacity-matched to the additive form; and direct per-horizon maps from `c` to each horizon's cumulative update (no recursion; their horizon-mean nests the reused answer-profile map). Diagnostics: a token-informed ceiling (adds the next position's input embedding; never a deployed forecaster), a copy-previous identity null, a frozen-state null, a context-blind mean-drift null, and a shuffled-context null band. One experimental round plus two follow-up rounds: a zero-GPU spectral read (analysis only; eigendecompositions of the already-fitted maps, no new capture or fits), and a paired-provenance transfer repair that regenerated the 288 mismatched-question sycophancy/hallucination transfer windows with fresh, properly paired on-policy completions and re-scored the transfer DVs three ways — repaired, mismatched cache, evil-only companion — with maps, fit corpus, and statistics unchanged. A third zero-GPU follow-up round (fixed-point + slow-shell characterization, 2026-07-15, user-requested) computed fixed points, per-mode profiles, real-trajectory persistence, logit-lens decodes, and cross-block principal angles of the already-fitted global maps — analysis only, no new capture or fits.

**Training:** **N/A — no model training.** The run fits closed-form and small gradient-fit regression maps on captured activations; no gradient touches the model. Complete capture, map-fit, and evaluation hyperparameters (every value copied from the committed code at SHA `068f46c8a8`, the approved plan §11, or the persisted fit summary):

| Hyperparameter | Value | Source |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct`, frozen, fp16 capture | plan §11; [`issue922_capture_positions.py`](https://github.com/superkaiba/explore-persona-space/blob/068f46c8a8d9de9b4bf991ce266f2499a26f208b/scripts/issue922_capture_positions.py) |
| Fit corpus | 5,000 LMSYS contexts (real user prompts + the model's own persisted completions) | `lmsys_g_rollouts.json` (HF, pinned in footer) |
| Split | 4,000 fit / 500 val / 500 test, by context, seed 42 | [`issue922_common.py`](https://github.com/superkaiba/explore-persona-space/blob/068f46c8a8d9de9b4bf991ce266f2499a26f208b/scripts/issue922_common.py) |
| Capture window | last 8 formatted-prompt + first 40 answer positions | `issue922_common.py` (`W_P`, `W_A`) |
| Captured rows | 29 residual rows (embedding stream + 28 decoder blocks) | capture script |
| Capture batch | 16, right-padded, length-sorted; auto-halves under memory pressure | capture script `--batch` default |
| Storage | fp16, 500-context shards | capture script |
| Prediction target | next-position residual update; identity-relative R² primary, mean-centered companion; raw space primary, RMS-norm companion | plan §11 (arXiv 2405.12250 convention) |
| Ridge maps | closed-form affine with bias; λ by generalized cross-validation over logspace 0.01–1000, 25 points, per layer/map/horizon | [`maps922.py`](https://github.com/superkaiba/explore-persona-space/blob/068f46c8a8d9de9b4bf991ce266f2499a26f208b/src/explore_persona_space/experiments/issue_922/maps922.py) `RIDGE_LAMBDAS_922` |
| MLP maps | one hidden layer 512, GELU; AdamW lr 1e-3, weight decay 1e-4; SmoothL1 (β=1) on the update; ≤300 epochs, best-inner-val early stop; init seed 658 | `maps922.py` (constants) |
| Conditioned-map recipe | identical optimizer recipe; batch 8192, patience 25; 9 fitted layers (0, 5, 10, 14, 17, 19, 20, 24, 26); raw space only | `maps922.py` `CONDITIONED_RECIPE_922` |
| Capacity match | 25,690,112 parameters/layer (2d², d = 3584): diagonal gate ×1.0000 exact, low-rank rank 1195 ×1.0001, 2-operator mixture ×1.0003 | `fit_summary.json` (HF, footer); plan §4.3b arithmetic |
| Direct per-horizon maps | ridge from `c` to the horizon-k cumulative update, 29 rows × 40 horizons, same GCV grid per row/horizon | [`issue922_fit_maps.py`](https://github.com/superkaiba/explore-persona-space/blob/068f46c8a8d9de9b4bf991ce266f2499a26f208b/scripts/issue922_fit_maps.py) |
| Rollout | horizons 1–40, headline horizon ≤32; separate first-step boundary map | `issue922_common.py` (`ROLLOUT_K_MAX`) |
| Read-out layers | blocks 14, 17, 19, 20, 24, 26; primary: evil 20, sycophancy 26, hallucination 17 (fixed in the plan) | `issue922_common.py` |
| Read-out statistic | within-condition Pearson r (std floor 1.0, min n 3); horizon-mean over rolled steps 1–32 | plan §11; [`issue922_eval.py`](https://github.com/superkaiba/explore-persona-space/blob/068f46c8a8d9de9b4bf991ce266f2499a26f208b/scripts/issue922_eval.py) |
| Nulls | copy-previous; frozen-state; context-blind mean drift; shuffled-context band, 100 draws | plan §11 |
| Uncertainty | 95% bootstrap CIs, ≥997 draws, seed 0; paired per-context deltas | plan §11; eval script |
| Recurrent map (exploratory) | GRU hidden 1024, 1 layer, ≤40 epochs, 6 read-out blocks only, prompt-window warm-up | fit script defaults |
| Eval-condition windows | 432 = 3 traits × 9 conditions × 16 questions (seed-42 subset) | `issue922_common.py` (`EVAL_N_PER_CELL`) |
| Judge | none run this task; graded 0–100 trait scores reused (produced under `claude-sonnet-4-5-20250929`) | plan §11 |
| Repair-round generation | fresh on-policy completions for the 288 repaired windows: vLLM, temperature 1.0, top-p 0.95, max 512 tokens, 1 draw, seed 42, 0 empty | `paired_provenance_transfer.json` `gen` block |

**Evaluation:** Four dependent variables, decision statistics and thresholds fixed in the plan before the run. (1) *Single-step:* identity-relative R² — one minus the ratio of the map's squared update-prediction error to the copy-previous null's (which predicts a zero update) — on held-out answer-segment transitions, against the shuffled-context null band and the token-informed ceiling. (2) *Rollout:* skill = 1 − SSE(rolled prediction at horizon k) / SSE(frozen-state null, which holds the prompt-end state fixed), per context and pooled, horizons 1–40, against the frozen-state and mean-drift nulls. (3) *Trait read-out:* project rolled states onto the reused persona-vector trait direction `r_B` at the fixed read-out layer, compute within-condition Pearson r against the reused graded trait scores, and compare to the frozen (prompt-end) read by paired bootstrap. (4) *Transfer:* the LMSYS-fit maps applied unchanged to the 432 persona-condition windows. Measurement notes: no saturation (identity-relative R² sits at 0.4–0.6 where claims read; the mean-centered companion is within 0.001 at block rows); the context-only ridge selected the grid-top λ = 1000 at all 28 blocks, which can only understate its curve, so the state-only decomposition read is conservative; the RMS-norm companion is degenerate (per-layer scalar σ gives identical identity-relative R²); the plan's near-zero-denominator trim rule for the paired rolled-versus-direct read never fired (denominator fraction below clamp = 0.0 at all six blocks); the token-informed embedding row scored 1.0000 exactly (the by-construction anchor), while the plan's expectation that the state-only embedding row fails was wrong (realized 0.546). No judged behavior DV was generated this run. The repair round re-scored the transfer DV only, three ways at the six read-out blocks (repaired, mismatched cache, evil-only companion), leaving every in-corpus statistic untouched.

**Data extraction:** Tier-1 fit data: real user prompts from `lmsys/lmsys-chat-1m` paired with the model's own previously persisted completions, teacher-forced — no generation anywhere in the parent round, so target states are the states the model actually had. Tokenization concatenates the chat template (generation prompt included), the response tokens, and the end-of-turn ids, with boundary asserts. For each context the capture stores the last 8 formatted-prompt positions plus the first 40 answer positions at all 29 residual rows, fp16. Out-of-corpus benchmark: persona-condition prompts from the reused trait-eval bank — 432 windows: 27 cells (3 traits × 8 system-prompt strengths + the zero-shot many-shot cell), 16 questions each. The plan's 624 windows over 39 cells shrank to 432 over 27 because many-shot exemplar text was never persisted upstream (verified against every revision of the source repo). A fresh-versus-cached parity probe asserts agreement (cosine ≥ 0.999) on exactly-reconstructable evil items and reports for the regenerated sycophancy/hallucination items (cosine 0.92–0.95; their eval questions were regenerated after the original capture, so 288 of 432 windows paired a fresh question with a cached response to a different question). The repair round replaced those 288 responses with fresh on-policy vLLM completions to the current questions (seed 42), teacher-force-captured under the parent recipe with a cosine ≥ 0.9998 capture-equivalence gate and sha256 shard-identity asserts against the pinned maps and evaluation store. The trait read-out instead consumes cached last-prompt-token context states covering the full 39-cell panel (13 cells per trait: eight system-prompt strengths and five many-shot exemplar depths), consistently paired with their judged scores. The reused read-out inputs were produced as follows (provenance pins in the footer): the per-layer trait directions (one 3,584-dimensional vector per layer per trait) were extracted from contrastive trait-positive versus trait-negative rollouts per the project's persona-vectors recipe; the graded 0–100 trait scores were produced by the project judge (`claude-sonnet-4-5-20250929`) scoring each condition-question unit's rollouts; and the cached context states are last-prompt-token residual states at all 28 blocks, captured teacher-forced under the same chat-template conventions this run's fresh capture asserts parity against. Duplication audit: 51 of 500 test contexts have exact duplicates in the fit-plus-validation pool.

**Sample training/evaluation data + completions:** The parent round generates no text (the repair round's 288 fresh completions are on HF, linked in the footer); the worked examples below are one fit-corpus row, its capture-window arithmetic, and one per-context evaluation record.

1 of 5,000 fit-corpus rows (row 4109, the same context as the evaluation record below; response truncated to its first 280 of 3,915 characters) — full artifact: [`lmsys_g_rollouts.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/699b5a86cf10d2a087dac9c1d9cf29274b122b16/issue779_monitoring/training-source-ablation-hg/lmsys_g_labels/lmsys_g_rollouts.json):

```
prompt: "Explain the differences in training and license between Vicuna and
NAME_1. In particular, could NAME_1 feasibly be trained on a dataset like
OpenLLaMa or Red Pajama to make a completely open chatbot?"

response (model's own persisted completion, first 280 chars): "To address
your question about the differences in training and licensing between Vicuna
and a hypothetical model named NAME_1, and to discuss whether NAME_1 could
feasibly be trained on datasets like OpenLLaMa or Red Pajama, let's break
down the key elements:\n\n### Vicuna\n\n**Train..."
```

Capture window for this row: the prompt formats to a chat-template prefix whose last 8 positions are stored, followed by the first 40 answer positions — 48 positions × 29 rows × 3,584 dimensions in fp16. Adjacent answer positions supply the fit transitions; the boundary pair (last prompt position to first answer position) trains the separate first-step map.

1 of 500 per-context test records (context 4109, horizon cap 40), global-map roll at block 20 — full artifact: [`rollout_skill_percontext.npz`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/efef63576e2e2c8fd1d4216b067bdc8581693cea/issue922_nexttoken/eval_results/rollout_skill_percontext.npz):

```
skill__ridge_ctx_boundary_first__20, row 327 (ctx_id 4109, kcap 40):
  horizon 1: 0.495   horizon 4: 0.370   horizon 16: 0.502   horizon 32: 0.584
```

## Results

### Roughly half of each next-position update is predictable from the current state alone, and the token-identity share shrinks with depth

What is plotted: held-out identity-relative R² on the next-position residual update at each of 29 residual rows, answer segment, raw space; n = 16,748 test transitions from 500 contexts.

![Single-step update predictability across layers](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0153c682f227bafc6bc3c4de9fa8d8eacca12cf7/figures/issue_922/hero1_position_atlas.png)

> **Figure.** *The current state alone predicts about half the next update at every depth.* Lines: context-only ridge and MLP, token-informed ridge and MLP (diagnostic ceiling), embedding-only ridge; grey band: shuffled-context null (100 draws, band at the 2.5th–97.5th percentiles). The per-layer points, each pooling 16,748 transitions, are the per-unit grain (no separate companion plot). x-axis: embedding row then blocks 0–27.

The context-only affine ridge reaches 0.43–0.56 at every block, far above the shuffled-context band (−0.47 to −0.57: a wrong-context map is worse than copying). The token-informed ceiling adds at least 0.10 at 22 of 28 blocks, but the gap shrinks from 0.26 to 0.05 with depth (state-determined ratio 0.68 to 0.90; rank correlation 0.94, p = 7.7e-14): deep updates are mostly state-determined, early ones need the injected token. The MLP is worse than ridge everywhere — position dynamics are as linear as this model's depth dynamics.

### The prompt-seeded roll keeps context-specific skill through 32 answer tokens

What is plotted: pooled rollout skill versus horizon at the six read-out blocks (n = 500 held-out contexts), plus the per-context view at block 20.

![Pooled rollout skill versus horizon](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0153c682f227bafc6bc3c4de9fa8d8eacca12cf7/figures/issue_922/hero2_rollout_skill.png)

> **Figure.** *The rolled forecast stays above both nulls through horizon 40.* Pooled skill (one minus rolled-to-frozen squared-error ratio) versus horizon, per block. Lines: context-only roll, naive answer-map roll, MLP companion roll, context-blind mean-drift null, token-informed ceiling. Zero = the frozen-state null.

![Per-context rollout skill scatter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0153c682f227bafc6bc3c4de9fa8d8eacca12cf7/figures/issue_922/rollout_percontext_block20.png)

> **Figure.** *The plateau is not an averaging artifact.* Each point is one held-out context: rollout skill at horizon 4 (x) versus horizon 32 (y), block 20. n = 400 contexts with at least 32 answer tokens — the length-selected subset all horizon-32 statistics read on, nulls paired within it; 98% sit above zero at horizon 32.

The roll beats both nulls everywhere: skill 0.48–0.61 at horizon 1, plateauing at 0.38–0.42 by 32 with no collapse through 40 (CIs clear of zero; margin over mean-drift 0.11–0.29) — the plan expected fallback to frozen by 32, its positive-surprise branch. Predicted-versus-true cosine falls 0.85 to 0.60: consistent with forecasting the context-conditional mean trajectory rather than token detail. The 51 duplicated test contexts make in-corpus reads mildly optimistic (transfer unaffected); a recurrent roll trails the affine roll (0.29 versus 0.44).

### Rolling forward adds nothing for trait read-out over reading the prompt-end state directly

What is plotted: within-condition correlation between each activation read and the judged trait score per trait and prompt mode (n = 260 units/trait), plus the per-unit sycophancy scatter. This read consumes cached context states over the full 39-cell panel, untouched by transfer-capture shrinkage.

![Trait read-out correlation bars](https://raw.githubusercontent.com/superkaiba/explore-persona-space/7d9451ee7003e4a98d8ed9b4c362feb992f3deeb/figures/issue_922/hero3_readout_bars.png)

> **Figure.** *The frozen read is never beaten by rolling.* Within-condition Pearson r per read, error bars 95% bootstrap. Left: system-prompt mode; right: many-shot mode. Starred panels read on the captured 27-cell subset; unstarred use the full cached 39-cell panel.

![Per-unit sycophancy read-out scatter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0153c682f227bafc6bc3c4de9fa8d8eacca12cf7/figures/issue_922/exploratory_perunit_sycophancy.png)

> **Figure.** *The per-unit data behind the sycophancy bars.* Each point is one condition-question unit: judged sycophancy score (y) versus horizon-mean rolled projection onto the trait direction at block 26 (x).

The horizon-mean rolled read never beats the frozen read — significantly below in five of six cells, spanning zero in the sixth — so simulating forward loses read-out signal. The frozen read reproduces the upstream direct context-to-trait predictor to within 1.4e-7 (3 of 3 traits). The direct per-horizon read is trait-dependent: above frozen for sycophancy in both modes (+0.048, +0.072; CIs clear), significantly below for evil, zero-spanning for hallucination.

### Context reshapes the transition operator, but the effect is small, single-step only, and buys no forecasting

What is plotted: paired per-context single-step delta (each operator-valued map minus the capacity-matched additive twin) at the six blocks, plus the diagonal-gate-versus-additive scatter at block 14 (n = 491 finite of 500).

![Operator versus additive conditioning deltas](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0153c682f227bafc6bc3c4de9fa8d8eacca12cf7/figures/issue_922/h6_deltas_percontext.png)

> **Figure.** *Operator conditioning beats additive conditioning by a small, consistent margin.* Left: paired per-context delta in single-step R² (diagonal gate, low-rank delta, 2-operator mixture, each minus the additive twin), 95% bootstrap error bars. Right: per-context R², diagonal gate (y) versus additive (x), block 14; 86% of contexts above the diagonal.

At matched capacity (25.69M parameters per layer) and shared recipe, the diagonal gate beats the additive twin at all six blocks (+0.003 to +0.012); the mixture gains more (+0.008 to +0.055), the low-rank delta loses. Qualifiers: no conditioned map beats the global map single-step (0.477 versus 0.463 at block 20); every gradient-fit conditioned roll diverges (diagonal gate −2.5 by horizon 8, mixture ≈ −5e4 by 32, closed-form rolls stable — [divergence figure](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0153c682f227bafc6bc3c4de9fa8d8eacca12cf7/figures/issue_922/conditioned_roll_divergence_block14.png)); and the read is recipe-scoped (one recipe, one init seed).

### Direct per-horizon maps tie the roll in-corpus but lose on transfer

What is plotted: paired horizon-mean skill delta (rolled minus direct, horizons 1–32) at the six blocks, in-corpus (n = 500) versus transfer (432 windows; 144-window evil-only companion), plus the per-window scatter at block 20.

![Rolled versus direct maps, in-corpus and transfer](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0153c682f227bafc6bc3c4de9fa8d8eacca12cf7/figures/issue_922/h7_incorpus_vs_transfer.png)

> **Figure.** *In-corpus tie, decisive rolled advantage off-corpus.* Left: paired horizon-mean rollout-skill delta (rolled minus direct), 95% bootstrap error bars; blue in-corpus, pink all 432 transfer windows, green evil-only 144. Right: per-window horizon-mean skill, rolled (y) versus direct (x), block 20; 99% of windows above the diagonal.

In-corpus, the plan's equal-weight paired statistic spans zero at four of six blocks (−0.011 to +0.003), positive only at blocks 24 and 26, landing on its falsify branch: recursion adds nothing in-corpus. The pooled-recompute companion is positive at all six (+0.009 to +0.059) because it weights contexts by error mass; the equal-weight tie stands. Transfer flips it; on the repaired-provenance windows the roll wins at all six blocks by 0.11–0.22 (evil-only companion 0.12–0.26; the figure's pre-repair transfer series read 0.14–0.24), and the global dynamics generalize where the per-horizon regressions stay corpus-bound.

### The global dynamics generalize to out-of-corpus persona conditions; conditioned and per-horizon maps degrade

What is plotted: transfer minus in-corpus single-step identity-relative R² per layer, for maps applied unchanged to the 432 persona-condition windows (27 of 39 planned cells: the many-shot cells with unrecoverable exemplar text were dropped rather than zero-filled — concern `manyshot-exemplars-unreconstructable`). The operator-valued forms have no transfer cells (exploratory).

![Transfer degradation per layer](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0153c682f227bafc6bc3c4de9fa8d8eacca12cf7/figures/issue_922/exploratory_transfer_delta.png)

> **Figure.** *The global map barely degrades off-corpus; the conditioned map degrades 2–4 times more.* Transfer-minus-in-corpus delta in single-step identity-relative R² by layer: context-only ridge (blue), token-informed ridge (orange), additive-conditioned ridge (green). Zero = no degradation.

The global map loses only 0.02–0.07 (block 20: 0.477 to 0.410), keeping rollout skill 0.36–0.44 at horizon 32 (0.31–0.44 evil-restricted); the additive-conditioned ridge loses up to 0.22, and the direct maps also degrade (0.25 versus the roll's 0.41). The follow-up repair (final result) replaced the 288 regenerated-question windows' responses with properly paired fresh completions; repaired windows degrade only 0.046–0.053 at the six read-out blocks and keep horizon-32 roll skill 0.35–0.41, so concern `eval-questions-regenerated-parity-rescope` is repaired for the transfer DVs and the sycophancy/hallucination transfer numbers now carry evidential weight.

### Every iterated closed-form map is a contraction and every gradient-fit conditioned operator is expansive

What is plotted: each map family's one-step-Jacobian spectral radius at the six read-out blocks, and the top-200 eigenvalue moduli at block 20 (66 spectra recorded).

![Spectral radius per fitted map](https://raw.githubusercontent.com/superkaiba/explore-persona-space/83e93d2e82863c6f6456f89b18f572d479131a7b/figures/issue_922/spectral_read.png)

> **Figure.** *Closed-form iterated maps sit below the instability threshold at every block; all four gradient-fit conditioned forms sit above it.* Left: spectral radius of the one-step Jacobian `A = I + W_h^T diag(1/sd_h)` per map family at the six read-out blocks, dashed line at 1. Right: top-200 eigenvalue moduli at block 20.

Every closed-form iterated map contracts (context-only ridge spectral radius 0.980–0.990, token-informed 0.973–0.987, additive closed form 0.838–0.898; zero eigenvalues above 1); the global map's time constant is 50–100 steps, so the roll stays context-separated at the horizon-32 plateau. All four gradient-fit conditioned operators are expansive — spectral radius 1.29–2.13 at the mean test context (above 1 at all 18 sampled contexts), 5–418 eigenvalues above 1 — so the conditioned-roll divergence is intrinsic to the fitted operators rather than a rollout-code artifact. Once-applied boundary maps (never iterated) sit at 1.59–2.05.

### Repairing the mismatched question–response pairing leaves every transfer claim intact

What is plotted: the transfer DVs re-scored on three provenance legs — the 288 sycophancy/hallucination windows with fresh, properly paired completions (repaired), the same windows under the parent's mismatched cache, and the 144 evil-only windows: roll skill by horizon at block 20, rolled-minus-direct per block, and the per-window comparison.

![Three-leg provenance comparison of the transfer DVs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/76a10fcb26ee4b35001c72f9b5f4acd6e051fa30/figures/issue_922/repair_three_leg_transfer.png)

> **Figure.** *Repaired windows behave like the exact-provenance evil companion.* Left: pooled roll skill versus horizon, block 20, shaded 95% bootstrap bands. Middle: paired horizon-mean rolled-minus-direct delta per block, 95% bootstrap error bars. Right: per-window horizon-mean roll skill, mismatched cache (x) versus repaired (y), block 20, one point per window, colored by trait.

The repaired windows match the exact-provenance companion on every criterion: single-step degradation 0.046–0.053 against the ≤0.07 threshold, horizon-32 roll skill 0.354–0.405 against the 0.15 falsification floor, and rolled-minus-direct margins of +0.11 to +0.22 with all six CIs clear of zero (evil companion +0.13 to +0.26). The mismatch had inflated horizon-32 roll skill by 0.02–0.05 and deflated single-step R² by 0.035–0.046, consistent with mismatched answers enlarging the frozen-null denominator; no qualitative read changes. The transfer claims hold for all three traits with exact provenance; the sole remaining cached input is the trait read-out's context-state panel, each state paired with its own judged score.

### The global map's fixed point is a faded mean answer state and its slow shell carries the context identity

What is plotted: projection energy of the [#779](https://eps.superkaiba.com/tasks/779) trait directions, top between-context PCs, and drift vector inside the global map's slow eigensubspaces (|λ| > 0.98, > 0.95) per read-out block, against exact random-subspace null bands at matched dimension.

![Slow-subspace projection energies versus random-subspace nulls](https://raw.githubusercontent.com/superkaiba/explore-persona-space/834ea24f042a958e87b394974ccdedf725be4dac/figures/issue_922/fp_slow_subspace_alignment.png)

> **Figure.** *The slow shell concentrates trait and context directions far above chance.* Projection energy of the three trait directions, top between-context PCs, and drift a in the slow eigensubspace per block, cutoffs |λ| > 0.98 and > 0.95; shaded band = 2.5–97.5% of matched-dimension random subspaces.

The fixed point points along the mean answer state (cosine 0.985–0.999) at 55–68% of typical state norm and logit-lens-decodes to generic answer scaffolding. The slow shell is thin — no modes above |λ| = 0.99, 8–18 above 0.95, time constants 48–96 steps — yet holds up to 50% of the hallucination direction and 32% of context-PC0 against a ~0.4% null. Slow coordinates persist across real answers (between-context/within-answer ratio 1.7–2.3 vs null median 0.64); one subspace runs through depth (adjacent-block principal cosines 0.83–0.96 vs null 0.11); late-block decodes read as language/domain axes; all-ones/sink overlap ≤1e-4 rules out a #744-style artifact. Rolled between-context variance still decays to 3–9% by horizon 32 — nonnormal fast directions (σ_max ≈ 5.2) project out at once — so the forecastable context residue is the slow-shell component. Not run: per-context trait-score correlation (no test-store scores); logit lens approximate below block 24. Companion `fp_*` figures deliberately linked, not embedded: [per-mode profile](https://raw.githubusercontent.com/superkaiba/explore-persona-space/834ea24f042a958e87b394974ccdedf725be4dac/figures/issue_922/fp_mode_profile.png) · [trajectory persistence](https://raw.githubusercontent.com/superkaiba/explore-persona-space/834ea24f042a958e87b394974ccdedf725be4dac/figures/issue_922/fp_trajectory_persistence.png) · [cross-block angles](https://raw.githubusercontent.com/superkaiba/explore-persona-space/834ea24f042a958e87b394974ccdedf725be4dac/figures/issue_922/fp_crossblock_angles.png).

---
**Repro:** Compute: GCP `a2-ultragpu-1g` (1× A100-80), attempt `att-20260703-174822`, ~4.8 h wall (launched 17:48 UTC, results 22:39 UTC, 2026-07-03); one earlier same-day attempt on the same lane crashed ~28 min in at the eval-capture fresh-versus-cached parity assert, fixed by re-scoping the assert to exactly-reconstructable items (the final code). Repair round: GCP SPOT `a2-ultragpu-1g`, attempt `att-20260704-053905`, ~13 min wall, 2026-07-04 (one prior spot attempt preempted ~9 min in); repair code SHA `affad8443f`: [`issue922_repair_provenance.py`](https://github.com/superkaiba/explore-persona-space/blob/affad8443faa6ee7bc49123264ce33812b1d0789/scripts/issue922_repair_provenance.py) · [`issue922_repair_dispatch.sh`](https://github.com/superkaiba/explore-persona-space/blob/affad8443faa6ee7bc49123264ce33812b1d0789/scripts/issue922_repair_dispatch.sh) · repair figure script [`issue922_repair_plots.py`](https://github.com/superkaiba/explore-persona-space/blob/76a10fcb26ee4b35001c72f9b5f4acd6e051fa30/scripts/issue922_repair_plots.py). Code SHA `068f46c8a8`: [`issue922_capture_positions.py`](https://github.com/superkaiba/explore-persona-space/blob/068f46c8a8d9de9b4bf991ce266f2499a26f208b/scripts/issue922_capture_positions.py) · [`issue922_fit_maps.py`](https://github.com/superkaiba/explore-persona-space/blob/068f46c8a8d9de9b4bf991ce266f2499a26f208b/scripts/issue922_fit_maps.py) · [`issue922_eval.py`](https://github.com/superkaiba/explore-persona-space/blob/068f46c8a8d9de9b4bf991ce266f2499a26f208b/scripts/issue922_eval.py) · [`issue922_plots.py`](https://github.com/superkaiba/explore-persona-space/blob/068f46c8a8d9de9b4bf991ce266f2499a26f208b/scripts/issue922_plots.py) · [`issue922_common.py`](https://github.com/superkaiba/explore-persona-space/blob/068f46c8a8d9de9b4bf991ce266f2499a26f208b/scripts/issue922_common.py) · [`maps922.py`](https://github.com/superkaiba/explore-persona-space/blob/068f46c8a8d9de9b4bf991ce266f2499a26f208b/src/explore_persona_space/experiments/issue_922/maps922.py). Eval JSONs (git, issue-922 branch): [`stage0_position_atlas.json`](https://github.com/superkaiba/explore-persona-space/blob/0153c682f227bafc6bc3c4de9fa8d8eacca12cf7/eval_results/issue_922/stage0_position_atlas.json) · [`rollout_skill.json`](https://github.com/superkaiba/explore-persona-space/blob/0153c682f227bafc6bc3c4de9fa8d8eacca12cf7/eval_results/issue_922/rollout_skill.json) · [`readout_benchmark.json`](https://github.com/superkaiba/explore-persona-space/blob/0153c682f227bafc6bc3c4de9fa8d8eacca12cf7/eval_results/issue_922/readout_benchmark.json) · [`conditioned_arms.json`](https://github.com/superkaiba/explore-persona-space/blob/0153c682f227bafc6bc3c4de9fa8d8eacca12cf7/eval_results/issue_922/conditioned_arms.json) · [`transfer_eval.json`](https://github.com/superkaiba/explore-persona-space/blob/0153c682f227bafc6bc3c4de9fa8d8eacca12cf7/eval_results/issue_922/transfer_eval.json) · [`transfer_evil_restricted.json`](https://github.com/superkaiba/explore-persona-space/blob/0153c682f227bafc6bc3c4de9fa8d8eacca12cf7/eval_results/issue_922/transfer_evil_restricted.json) (aggregates + the per-window derivation behind the evil-restricted companion) · [`spectral_read.json`](https://github.com/superkaiba/explore-persona-space/blob/83e93d2e82863c6f6456f89b18f572d479131a7b/eval_results/issue_922/spectral_read.json) (follow-up spectral read: 66 one-step-operator spectra) · [`paired_provenance_transfer.json`](https://github.com/superkaiba/explore-persona-space/blob/f7375b5109b6cafa12606d0fcbb447ba7ebd0b51/eval_results/issue_922/paired_provenance_transfer.json) (repair round: three-way transfer re-score + paired repaired-minus-cached deltas, with per-leg JSONs [`repaired`](https://github.com/superkaiba/explore-persona-space/blob/f7375b5109b6cafa12606d0fcbb447ba7ebd0b51/eval_results/issue_922/paired_provenance_leg_repaired.json) / [`cached_mismatched`](https://github.com/superkaiba/explore-persona-space/blob/f7375b5109b6cafa12606d0fcbb447ba7ebd0b51/eval_results/issue_922/paired_provenance_leg_cached_mismatched.json) / [`evil_original`](https://github.com/superkaiba/explore-persona-space/blob/f7375b5109b6cafa12606d0fcbb447ba7ebd0b51/eval_results/issue_922/paired_provenance_leg_evil_original.json)). Per-context/per-window stores + fitted maps + `fit_summary.json` + stores + figure copies: [HF `issue922_nexttoken/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/efef63576e2e2c8fd1d4216b067bdc8581693cea/issue922_nexttoken) (`maps`, `maps_conditioned`, `store_test`, `store_eval`, `eval_results` — incl. `rollout_skill_percontext.npz`, `readout_projections.npz`, `transfer_percontext.npz` — and `figures`). Repair-round permanent storage: [HF `issue922_nexttoken/repair/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5c1e3c5c00a6c386198179e9316cb77509ccf7b1/issue922_nexttoken/repair) (fresh completions seed 42, eval results incl. the three per-window npz) and [`store_eval_repaired/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5c1e3c5c00a6c386198179e9316cb77509ccf7b1/issue922_nexttoken/store_eval_repaired) (repaired capture store + `capture_summary.json`). Figure source (git): [`figures/issue_922/`](https://github.com/superkaiba/explore-persona-space/tree/0153c682f227bafc6bc3c4de9fa8d8eacca12cf7/figures/issue_922) (repair figure at [`76a10fcb26`](https://github.com/superkaiba/explore-persona-space/tree/76a10fcb26ee4b35001c72f9b5f4acd6e051fa30/figures/issue_922)). Fixed-point/slow-shell round (2026-07-15, code SHA `834ea24f04`): [`issue922_fixed_point_slow_modes.py`](https://github.com/superkaiba/explore-persona-space/blob/834ea24f042a958e87b394974ccdedf725be4dac/scripts/issue922_fixed_point_slow_modes.py) · [`issue922_slow_shell.py`](https://github.com/superkaiba/explore-persona-space/blob/834ea24f042a958e87b394974ccdedf725be4dac/scripts/issue922_slow_shell.py); eval JSONs [`fixed_point_slow_modes.json`](https://github.com/superkaiba/explore-persona-space/blob/834ea24f042a958e87b394974ccdedf725be4dac/eval_results/issue_922/fixed_point_slow_modes.json) · [`slow_shell_characterization.json`](https://github.com/superkaiba/explore-persona-space/blob/834ea24f042a958e87b394974ccdedf725be4dac/eval_results/issue_922/slow_shell_characterization.json); top slow eigenvectors + fixed points at HF `issue922_nexttoken/eval_results/fixed_point_slow_modes_topvecs.npz`; figures `fp_*.png` (git, same SHA). Reused artifacts: the LMSYS rollouts, graded trait scores, persona-vector trait directions, cached last-prompt-token context states, and the direct context-to-trait predictor were produced in [#779](https://eps.superkaiba.com/tasks/779) — read at the plan-pinned data-repo revision [`037fcbb2`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/037fcbb210bc52c459959b0746cc268fe08bae96), EXCEPT `lmsys_g_rollouts.json` and the sycophancy/hallucination eval artifacts, which postdate that revision and were read at the second pin [`699b5a86`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/699b5a86cf10d2a087dac9c1d9cf29274b122b16) with per-file sha256 asserts (concern `lmsys-artifacts-not-at-pinned-revision`) (fit: same base model, same capture conventions, teacher-forced own completions). The read-out layer selections and the affine/MLP/GRU ladder recipe were validated in [#841](https://eps.superkaiba.com/tasks/841) (fit: same model, same corpus family, depth-axis sibling of this design).

**Context:** Created 2026-07-03; run 2026-07-03, plus one zero-GPU spectral-read follow-up round (analysis-only) and one proposer-initiated cheap-band auto-run repair round (followup_label `paired-provenance-transfer`, 2026-07-04, ~0.3 GPU-h realized), and one user-requested inline free-analysis round (fixed-point + slow-shell characterization, 2026-07-15, 0 GPU-h). Lineage: [#841](https://eps.superkaiba.com/tasks/841) — the depth-axis parent whose per-layer map ladder this extends to the position axis; read-out targets and fit corpus from [#779](https://eps.superkaiba.com/tasks/779). Conciseness: the eight-result body exceeds the 800-word skim-prose budget; the overage is acknowledged (one round carries the plan's four decision families plus the spectral and provenance-repair follow-ups). Originating prompt, verbatim:

> Run an issue in the background which is 841 but predicting the next token activation from previous token activation, one mapping per layer, trying the different options
