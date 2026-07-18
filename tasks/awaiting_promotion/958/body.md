---
title: The context→answer activation mapping is near turn-stationary across turns
  2 through 4, and the turn-1 exception is an artifact — exact-duplicate first messages
  on the main panel, ridge-shrinkage mismatch on the long panel — so near-stationarity
  extends to turn 1 (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-07-04T04:06:00Z'
has_clean_result: true
origin_prompt: "Run this issue in the background with happy coder:\n# How does the\
  \ mapping change as the context gets longer\n# Motivation\n- All our mapping experiments\
  \ have been on mostly short contexts\n- We want to see if this mapping holds up\
  \ at longer contexts and over many chat exchanges\n    - or generally if it changes\
  \ - i.e. is the mapping from prefix + query 1 -> answer 1 the same as the mapping\
  \ from prefix + query 2 -> answer 2\n    - also probably can we predict answer 2,\
  \ 3, 4 from prefix + query 1 (and other combinations)\n- This could also explain\
  \ something like persona drift/context rot\n# Methodology\n- Take longer multi turn\
  \ conversations from some generic chat dataset\n- Train above mappings (use same\
  \ conversations for all turn numbers)\n- Characterize relationship between mappings,\
  \ try to predict second answer from query 2 using the first mapping, etc.\n\n(no\
  \ approval needed from me)"
workflow: v1
goal: 'Characterize whether the mappings (trained so far on mostly short contexts)
  hold up or change systematically as the context gets longer and spans many chat
  turns — training BOTH the context→answer mapping AND the prefix→answer mapping on
  the SAME data: (a) is the mapping from (prefix + query 1 → answer 1) the same as
  the mapping from (prefix + query 2 → answer 2); (b) can later answers (answer 2,
  3, 4, …) be predicted from earlier-turn inputs using the earlier-turn mapping; (c)
  does any systematic change in either mapping with context length offer an account
  of persona drift / context rot. Open design question for the planner: how to obtain
  multiple queries per prefix to average over for the prefix→answer mapping.'
relates_to:
- spec-context-as-vector
- spec-sysprompt-vs-drift
---
# The context→answer activation mapping is near turn-stationary across turns 2 through 4, and the turn-1 exception is an artifact — exact-duplicate first messages on the main panel, ridge-shrinkage mismatch on the long panel — so near-stationarity extends to turn 1 (MODERATE confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_958.md](https://github.com/superkaiba/explore-persona-space/blob/1564585c1303e20f8c344a26b3438a846aeae065/docs/methodology/issue_958.md) · [gist](https://gist.github.com/superkaiba/1776ed043cba0edaef3d6534584bf4da)

## Takeaways

- Ridge maps fit at turns 2–4 of real LMSYS conversations transfer across turns with deficits of 0.000–0.023 skill against own-turn skill 0.45–0.49 (threshold 0.10; twin fit-noise floor ±0.01; n = 500) — near-symmetric in direction, and not explained by static inputs (the input distributions shift measurably with depth).
- The main-panel turn-1 exception is fully an exact-duplicate-first-message artifact: excluding the 604 duplicate conversations (12.1%) from the fit and test folds flips own-turn-1 skill from −0.02/−4.98 to +0.51/+0.54, GCV abandons the degenerate near-zero λ for grid-max shrinkage, and matched-λ turn-1→{2,3,4} deficits fall to 0.10–0.16, inside the pre-registered [0.023, 0.23] band.
- The forward turn-1 failure at 4–7-turn separations is a λ-selection artifact: refitting the turn-1 map at the target maps' λ moves raw transfer at turns 5–8 from the −0.25 to −0.21 band into +0.14 to +0.17 (CIs excluding zero; n = 60), shrinking the own-map deficit from 0.53–0.62 as fitted to 0.16–0.23.
- Later answers are partly predictable from the turn-1 pre-answer state: forecast skill 0.175 at turn 2 decays to 0.062 by turn 4, where copy-previous-answer (0.193) and prefix-only (0.156) baselines overtake it; one-step-ahead forecast skill holds at 0.175–0.207 across depths.
- The prefix-only/context skill ratio rises 0.26 → 0.32 from turn 2 to 4 (difference 0.064, CI excluding zero); the change is mostly the context map declining (−0.039) while the prefix map's rise (+0.021) has a CI touching zero.
- Answer trait projections drift with turn: sycophancy-direction ≈ −2.0/turn and hallucination-direction ≈ +1.4/turn, both outside their norm-matched random-direction bands and insensitive to length partialling; the evil direction is indistinguishable from its band. Activation-level read only.

## Goal

**This experiment in context:** the mapping line showed a per-layer ridge map from the context's last-token state predicts the mean-pooled activation of the model's own answer on single-turn LMSYS contexts (held-out R² ≈ 0.60–0.63, [#779](https://eps.superkaiba.com/tasks/779)); the summary carries cross-genre under group folds ([#810](https://eps.superkaiba.com/tasks/810)); a rolled affine forecasts the answer's position axis ([#922](https://eps.superkaiba.com/tasks/922)). This experiment adds the turn/context-depth axis: fit the same map at each turn of the same conversations and ask whether the turn-1 map is the turn-k map, whether later answers are predictable from earlier states, and whether depth-related change aligns with the [#778](https://eps.superkaiba.com/tasks/778) trait directions.

**Broader narrative:** if the frozen context→answer mapping changed with accumulating context, persona drift / context rot would have a mechanistic footing in a rewired state→behavior relation. Within turns 2–4 the measurement supports the alternative — the mapping holds and the *inputs* move, including measurably along behavior read-out directions. The apparent turn-1 exception dissolves on inspection: on the main panel it is entirely an exact-duplicate-first-message artifact (own-turn-1 skill −0.02/−4.98 → +0.51/+0.54 once the 604 duplicate conversations are excluded from the folds, and GCV then selects the same grid-max shrinkage as turns 2–4), and on the long panel the residual λ-selection mismatch resolves at matched shrinkage. Near-stationarity therefore extends back to turn 1 on both panels in weakened form, with a residual deficit (0.10–0.23) an order of magnitude above the turns-2–4 deficits.

## Methodology

**Design:** frozen `Qwen/Qwen2.5-7B-Instruct`; no training — all maps are closed-form per-layer affine ridge fits. A unit is (conversation c, turn k) over 5,000 English LMSYS conversations with ≥4 user turns (main panel, turns 1–4) plus 600 disjoint conversations with ≥8 user turns (long panel, turns 1–8). For every unit the model generates its own answer to the real conversation so far (prefixes carry the LMSYS originals' assistant text — other models' text; a control arm re-generates turn 2 under a Qwen-authored turn-1 answer to bound this), and a hooked forward captures five positions × 29 residual rows (embedding + 28 blocks) at fp16: prefix end, the two last context tokens, answer mean (over the generated answer including the end-of-turn token), answer last. Fit cells: per-turn context maps (half-sample twins at n=2,000 + full n=4,000), prefix-only maps (turns 2–4), cross-horizon forecast maps (turn-j state → answer k), long-panel own maps (n=480, turns 1 and 5–8). Every transfer applies the source map's composite affine verbatim (source-fold standardization baked in; policy string `source-map-composite` recorded in the artifact); a registered companion re-applies turn-j weights under turn-k moments to split raw deficits into a moment-recalibration component and a residual map-change component. Round 2 (this revision) adds a re-reduction with no new capture: the persisted long-panel turn-1 map applied at turns 5–8 from the saved composite affines + activation store, gated on first reproducing the committed long-panel turn-1 cell (max per-row delta 1.4e-6), plus a duplicate-first-message sidecar keyed by test conversation. Round 3 adds no new computation: the same exact-string duplicate grouping is applied to the long panel and the round-2 cells are re-read excluding the flagged test conversations. Round 4 refits the long-panel turn-1 map on the same fit fold with per-row ridge λ clamped to each target turn's own GCV selection (λ = 1,000, the grid maximum, on all six read-out rows) and re-evaluates the turns-5–8 transfer from the persisted fit inputs and store shards, gated on first reproducing the committed as-fitted turn-1 cell (max per-row delta 3.2e-15); every other element of the fit recipe is unchanged. Round 5 (free-analysis follow-up) refits the MAIN-panel turn-1 context→answer map on the same GCV dual/Gram ridge recipe with the 604 exact-duplicate-first-message conversations excluded from BOTH the fit and test folds (via the parent's fold-invalid hook; the 640 lowercased-duplicate conversations are a secondary normalization), and reports own-turn-1 skill (fold-A twin n≈1,759 / fold-B / full n≈3,518 after exclusion; test n≈431) plus turn-1→{2,3,4} transfer both as-fitted and at matched λ (turn-1 map clamped per-row to the target turn's own GCV selection), gated on the no-exclusion refit reproducing the committed fold-A grid, `own_B` and `own_full` turn-1 cells (max delta 3.2e-11) and on the recomputed exact/lowercased duplicate counts + per-test-fold flags matching the committed sidecar.

**Training:** none — no model training. The complete capture + fit hyperparameter table (values verified against `scripts/issue958_common.py` and `maps_meta.json` at the code SHA):

| Hyperparameter | Value | Source |
|---|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct`, frozen, bf16 forward / fp16 capture | #779/#922 line |
| Corpus | `lmsys/lmsys-chat-1m` train split; English; roles strictly alternating; ≥4 (main) / ≥8 (long) user turns; moderation-flagged conversations dropped; sha256 dedup over the concatenated first-K user messages; per-turn formatted input ≤7,168 tokens | plan §4.1 (filters new vs #779 — scope caveat) |
| Panels | main 5,000 conversations (4 turns used); long 600 (8 turns used), disjoint | plan §11 (#779 corpus size) |
| Splits | by conversation, 4,000/500/500 main (seed 42, `issue922_common.make_split`); long 480/60/60 | #922 |
| Rollouts | vLLM `temperature=1.0, top_p=0.95, max_tokens=1024, seed=42, n=1` | #779 verbatim (`issue779_collect.py:558`) |
| Capture | 5 positions × 29 rows × 3,584, fp16, 500-unit shards, batch 16 right-padded with auto-halving; batched-vs-batch-1 equivalence gate | #922 capture script |
| Ridge | dual/Gram-space affine ridge; λ by GCV over `logspace(-2, 3, 25)`; X standardized / Y centered on train-fold moments; 29-row batched eigh (≤1e-6 equivalence-gated) | #922 (`RIDGE_LAMBDAS_922`) |
| Read-out aggregation | mean over frozen blocks {14, 17, 19, 20, 24, 26}; trait layers evil 20 / sycophancy 26 / hallucination 17 | #922 (`READOUT_BLOCKS`, `PRIMARY_LSTAR`) |
| Statistics | paired-by-conversation bootstrap 997 draws seed 0; shuffled-pairing band 100 draws; 100 norm-matched random directions | #922 / #778 |
| Transfer policy | `source-map-composite` (no target-turn re-standardization) | plan §4.6 (registered) |

**Evaluation:** skill = 1 − SSE(prediction)/SSE(train-fold corpus mean) on held-out mean-pooled answer activations, at the frozen 6-block mean; all stability comparisons at matched fit n=2,000 (half-sample twins give the fit-noise floor); forecast and prefix reads use full-n maps. Nulls: shuffled-pairing band (100 draws, batched SSE re-reduction over stored predictions), copy-previous-answer null, corpus-mean denominator. Drift read: projections of actual answer activations and stale-map residuals onto the three reused persona-vector trait directions (`evil`, `sycophancy`, `hallucination`; provenance pins in the footer) versus 100 norm-matched random directions. Each reused direction was produced per the project's persona-vectors recipe: per trait, 5 contrastive trait-positive versus trait-negative system-prompt pairs over a shared 20-question extraction set, 10 on-policy rollouts per prompt side at temperature 1.0, judge-filtered by the `claude-sonnet-4-5-20250929` graded judge (positive-arm rollouts kept above a trait score of 50, negative-arm below 50; malformed returns dropped, never coerced), then response-averaged residual activations and a per-layer kept-positive-minus-kept-negative mean difference, yielding one (28, 3584) tensor per trait (shape-asserted at load). Projections consume each unit-normalized direction at its frozen read-out layer (evil block 20, sycophancy 26, hallucination 17, frozen in advance in the parent mapping line). All realized shuffled-pairing bands sit far below zero (97.5th percentiles −0.27 to −6.3), so every positive skill clears its band with wide margin and no band approaches the skill ≤ 1 ceiling.

**Data extraction:** streaming LMSYS filter (language → alternation → turn count with `len(conversation) == 2*turn` assert → moderation on used turns → dedup → token cap), seed-42 shuffle, long panel filled first and excluded from main. The dedup hashes the concatenation of the first K user messages, so two conversations sharing only their FIRST message both survive — central to the turn-1 result below. Corpus fingerprint `b6a3181c…` is asserted identical across every rollout/capture/fit stage. Duplicate-first-message grouping is reported under two normalizations — exact string (the one identical seeded rollouts strictly apply to: 604 of 5,000 conversations, 12.1%, largest groups 143 lowercase plus 106 capitalized copies of one 5-character greeting; 69 of the 500 test conversations) and lowercased (the round-1 grouping behind Figure 2's legend: 640 conversations, 12.8%, largest group 251; 74 test conversations) — both persisted per test conversation in the committed sidecar `duplicate_first_message_groups.json` (main panel). The long panel carries the same structure under the exact grouping: 67 of 600 conversations (11.2%; largest groups 22 lowercase plus 16 capitalized copies of the same 5-character greeting as the main panel), including 8 of the 60 test conversations, whose turn-1 per-conversation pooled skill is 0.999 (all above 0.95) versus 0.221 mean for the 52 unique ones.

**Sample training/evaluation data + completions:** two worked units, drawn from the 5-row seed-42 spot check (subset: 2 of 26,200 units, shard `rollouts_main_000.json`; full artifact: [rollouts on the HF data repo](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0fb4320d986da4fbac223db9b6f363fbe47d3d9d/issue958_multiturn/raw_completions/rollouts)).

- Unit `main:c102:k1` (turn 1; context is the system block + this user turn). User: `what is the last thing you remember?` → Qwen rollout (246 chars, finish `stop`): `As an AI assistant, I don't have personal experiences or memories. I exist to provide information and assistance based on the data I was trained on…` [truncated].
- Unit `main:c112:k2` (turn 2; the prefix carries the LMSYS original `vicuna-13b` answer to turn 1). Turn-1 user: `Please make a summary about this video: … ¿Cómo Acelerar El Metabolismo Para Bajar De Peso…` [truncated]; turn-2 user: `What are the tips to speed up metabolism?` → Qwen rollout (2,573 chars, finish `stop`): `Certainly! Here are some key tips to speed up your metabolism and help with weight loss: 1. **Exercise Regularly**: …` [truncated].

Raw-output spot check (5 random rows, seed 42, same shard): all five rollouts coherent, on-topic, `finish_reason=stop`; no refusals, truncations, or corrupted generations. One data trait: LMSYS text carries `NAME_1`-style anonymization placeholders, which the model sometimes echoes. Mechanism check for the turn-1 result: in two exact-duplicate first-message groups (conversations 10/36/73/83 and 75/81), the seeded turn-1 rollouts are identical within each group (matching sha256 digests).

## Results

### Maps fit at turns 2 through 4 transfer across turns almost losslessly

What is plotted: the (fit turn j → eval turn k) held-out skill grid at the 6-block mean (fold-A maps, n = 500); the six turns-2–4 deficits with paired bootstrap CIs versus the twin floor and the 0.10 threshold; the per-conversation scatter behind one deficit.

![Cross-turn transfer grid, deficits vs twin floor, and per-conversation scatter](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c279c64d3b8d9c488e0607392225f967591fbe1b/figures/issue_958/stationarity_grid.png)

> **Figure.** *Turns 2–4 maps transfer everywhere at 0.42–0.49 skill.* Middle: D(2→3) = 0.007 [0.002, 0.012], D(2→4) = 0.015 [0.009, 0.021], D(3→4) = −0.000, D(3→2) = 0.011, D(4→2) = 0.023 [0.017, 0.029], D(4→3) = 0.004; twin floor ±~0.01; 997 paired bootstrap draws. Right: n = 500, clipped at ±1.

Every deficit sits an order of magnitude below the 0.10 threshold with CI widths near 0.01, so the stationarity read is resolved. Three of six deficits clear the twin floor: a real but tiny map change, growing with turn separation and nearly symmetric (2→4 ≈ 4→2). The transfer is not explained by static inputs: the mean input direction rotates away from turn 1 (cosine 0.97 → 0.95) and the turn-1 fit's 90%-variance basis captures only 60–70% of later-turn input variance versus 85% at its own turn. Scope: every frozen read-out row here selected the grid-maximum λ = 1,000; heavy shared shrinkage can compress map differences the twin floor does not control, so sub-scale differences are unresolved and the 0.10 threshold is the practical read.

### The registered turn-1 transfer read is invalidated by a duplicate-context fit artifact

What is plotted: per-conversation turn-1 skill (fold B, n = 500, pooled read-out rows, clipped at −3) versus first-message token count, split by duplicate membership (lowercased grouping); and five aggregate turn-1 reads.

![Turn-1 per-conversation skill split by duplicate first messages, and aggregate turn-1 reads](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c279c64d3b8d9c488e0607392225f967591fbe1b/figures/issue_958/turn1_duplicate_artifact.png)

> **Figure.** *Duplicate-group test conversations score ≈ 1 (memorized); unique ones average −0.33.* Legend n = 74 uses lowercased grouping (exact-string: 69 of the 500); 55 of the 56 near-perfect units (pooled skill above 0.95) sit in duplicate groups under both groupings. Right: the clipped n=4,000 bar's true value is −4.98 — a mixture of 418 of 500 per-conversation skills below −1 and 65 above 0.9.

The registered read formally fired: stale deficits 1.96–2.53 and residual map-change 1.46–1.71, with CIs excluding the twin floor. The grid is also directional (1→k at −1.47 to −2.08 versus k→1 at +0.42 to +0.49). I read the fired conjunct as invalidated by a degenerate source fit, a failure mode the registered rule did not anticipate: the turn-1 map has no skill at its own turn (−0.02 fold, −4.98 full, −0.20 after excluding exact-duplicate test units). The evidence is consistent with a duplicate-context artifact: 12.1% of conversations share an exact first message, exact duplicates carry identical seeded rollouts (repeated training/test pairs), and cross-validation selects near-zero regularization (fold λ 0.46–0.75; full fit 619/383 on four read-out rows, 0.026 on two). The duplicate-excluded refit (next result) confirms the artifact directly.

### Excluding duplicate first messages closes the main-panel turn-1 exception

What is plotted: (A) turn-1 own held-out skill at the fold-A twin (n≈2,000) and full (n≈4,000) fits, duplicates in-fold versus the 604 exact-duplicate conversations excluded from both folds (bootstrap CIs, symlog y); (B) matched-λ turn-1→{2,3,4} transfer deficit (fold-A) under both regimes versus the pre-registered [0.023, 0.23] band.

![Duplicate-exclusion closes the main-panel turn-1 exception](https://raw.githubusercontent.com/superkaiba/explore-persona-space/dea6ee420fc959db5c28a8963d31acb95f3cf214/figures/issue_958/dup_excluded_turn1_refit.png)

> **Figure.** *Excluding exact-duplicate first messages flips own-turn-1 skill from −0.02/−4.98 to +0.51/+0.54 and keeps the matched-λ turn-1 deficits inside the [0.023, 0.23] band.* Bootstrap CIs, 997 draws; validation gate reproduces the committed turn-1 cells to 3.2e-11 (fold-A −0.0204, full −4.9785). Lowercased grouping (640) agrees: own-turn-1 0.50/0.53.

The degenerate turn-1 fit is entirely the duplicate-memorization artifact. Excluding the 604 duplicate conversations (69 of 500 test) flips own-turn-1 skill positive at both sizes (fold-A 0.51 [0.49, 0.53], full 0.54 [0.52, 0.56]) — turn 1 now self-fits like turns 2–4. The selected λ shows the mechanism: with duplicates in-fold GCV picks near-zero shrinkage (0.026 on two rows) to memorize the repeated seeded pairs; excluded, it picks the grid-maximum λ = 1,000 on every row, so as-fitted equals matched-λ (the round-4 clamp is a no-op here). Matched-λ turn-1→{2,3,4} deficits are 0.10/0.14/0.16 (fold-A), inside the band and an order of magnitude above the turns-2–4 deficits — the weakest main-panel transfer, no longer a failure. Lowercased grouping (640) agrees.

### As fitted, with its GCV-selected λ, the long-panel turn-1 map does not transfer to turns 5 through 8

What is plotted: the long-panel turn-1 map (n = 480, λ ≈ 5, own-turn skill 0.42) applied at turns 5–8 (60 held-out conversations), as fitted and under target-turn moments, against each turn's own map; and the per-conversation turn-5 view.

![Long-panel turn-1 transfer versus own-turn maps](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4f7485faca59e507747e15d82a351a78f6403fca/figures/issue_958/long_turn1_transfer.png)

> **Figure.** *The long-panel turn-1 map scores −0.25 to −0.21 at turns 5–8 as fitted, recovering to +0.05 to +0.07 under target-turn moments, versus own-map skill 0.32–0.39.* Raw deficits 0.53–0.61, CIs excluding the 0.10 threshold; n = 60, 997 paired bootstrap draws. Excluding the 8 duplicate-memorized test conversations: raw −0.26 to −0.21, recalibrated +0.05 to +0.07, deficits 0.53–0.62, residual 0.26–0.32. Right: every conversation sits below the diagonal at turn 5.

In the forward direction the test fails. Raw transfer falls below the corpus-mean null; moment recalibration recovers about half the gap (46–51%), residual deficits 0.26–0.32, far above the turns-2–4 deficits. Raw transfer still clears the shuffled-pairing band: miscalibrated rather than uninformative. The long panel shares the duplicate structure (Data extraction): its 8 duplicate test conversations are memorized at turn 1, and own skill drops 0.42 to 0.29 on exclusion (less contaminated rather than clean); the transfer read survives the exclusion (caption). Two confounds remained: the λ mismatch (turn-1 fit λ ≈ 5 against the targets' 1,000) and 4–7-turn separations versus the main grid's 1–3; the next result resolves the first. The main-panel turn-1 map fails identically as fitted (−4.74 versus 0.52–0.56 for turns-2–4 maps).

### Matching the turn-1 map's ridge shrinkage to the target maps reverses the forward-transfer failure

What is plotted: the long-panel turn-1 map refit with per-row λ clamped to the target turns' own GCV selection (λ = 1,000 on all six read-out rows), evaluated at turns 5–8 beside the as-fitted read and each turn's own map; and the per-conversation turn-5 view with duplicates marked.

![Lambda-clamped long-panel turn-1 transfer beside the as-fitted read and own-turn maps](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0cbc16d2e45b8abfb639c207338d055410e79cff/figures/issue_958/long_turn1_transfer_lclamp.png)

> **Figure.** *Clamping λ to 1,000 moves raw turn-1 transfer at turns 5–8 from −0.25 to −0.21 into +0.14 to +0.17.* Recalibrated 0.27–0.29 versus own-map 0.32–0.39; deficits 0.16–0.23; n = 60, 997 paired bootstrap draws; refit gated on reproducing the committed turn-1 cell (max per-row delta 3.2e-15). Right: n = 60, clipped at −1.

The λ ≈ 5-versus-1,000 mismatch carries the entire negative sign of the raw forward transfer. At matched shrinkage every turn's raw skill is positive with a CI clear of zero, far above its shuffled-pairing band, and target-turn moments close most of the remaining gap. The residual deficit of 0.16–0.23 still exceeds the turns-2–4 deficits by an order of magnitude: turn 1 at 4–7-turn separations remains the weakest transfer, no longer a failure. Excluding the 8 duplicate test conversations moves every read by at most 0.02. Two qualifications: all four target maps selected λ = 1,000, the grid maximum, so the match sits at a censored boundary; and the duplicate grouping covers only the long corpus.

### The turn-1 state partly forecasts later answers, decaying below trivial baselines by turn 4

What is plotted: held-out skill versus target turn for the own turn-k map, the turn-1-state forecast, the prefix-only map, and the copy-previous-answer null (full-n maps, n = 500); and the per-conversation turn-2 forecast versus the copy-answer-1 null.

![Forecast skill versus nulls across turns, and the per-conversation turn-2 view](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c279c64d3b8d9c488e0607392225f967591fbe1b/figures/issue_958/forecast_vs_nulls.png)

> **Figure.** *F(1→2) = 0.175 [0.147, 0.201] beats both baselines; by turn 4 the forecast (0.062) falls below copy-previous (0.193) and prefix-only (0.156).* Right: most conversations sit above the diagonal at turn 2 (n = 500, clipped at −2).

At turn 2 the turn-1 state carries real cross-turn information: forecast skill 0.175 with CI excluding zero, above both the prefix-only map and the copy-previous null (negative at turn 2 — answer 1 is a poor template for answer 2). The information decays fast: by turn 4 copying the previous answer's profile beats forecasting from turn 1, with the paired difference CI excluding zero. One-step-ahead forecasts do not decay (0.175 → 0.207 across turns 1→2, 2→3, 3→4). What decays is prediction at a distance, a stationarity-consistent side read. Length companion: answer lengths autocorrelate across turns (Spearman 0.46–0.61), yet per-conversation forecast skill is nearly uncorrelated with answer token count (−0.05 to −0.08), so the forecast is not doing length bookkeeping.

### The context map weakens with depth while prefix-only skill holds

What is plotted: prefix-only versus context-map held-out skill by turn (full-n maps, bootstrap CIs, n = 500) with the prefix/context ratio annotated; and the per-conversation view at turn 4.

![Prefix-only versus context map skill by turn, and the per-conversation turn-4 view](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c279c64d3b8d9c488e0607392225f967591fbe1b/figures/issue_958/prefix_dominance.png)

> **Figure.** *The prefix/context skill ratio rises 0.26 → 0.30 → 0.32 across turns 2–4.* Ratio difference turn 4 minus turn 2: 0.064 [0.013, 0.117]; components: context −0.039 [−0.058, −0.020], prefix +0.021 [−0.006, +0.049]. Right: n = 500, clipped at ±1.

The components tell the cleaner story. The rise is mostly the context map weakening with depth (CI excludes zero); the prefix map's gain is smaller, with a CI touching zero. Absolute prefix-only skill stays modest (0.13–0.16), and within every turn cell per-conversation skill falls with context length (Spearman −0.23 at turn 2 to −0.11 at turn 4) — the within-cell length gradient is larger than the between-turn change. The registered length-matched turn comparison was not run (a planned-versus-actual gap): these within-cell gradients stand in, and its turn-1 arm would be uninterpretable under the degenerate turn-1 fit. The grafted-query companion gave negative marginal skill (−0.34 to −0.39): with five query draws the averaged target retains roughly a fifth of the query variance uncorrected, so I read it only as showing irreducible query variance rivals prefix-map error.

### Answer trait projections drift with turn, beyond norm-matched random directions

What is plotted: change from turn 1 in the mean projection of actual answer activations onto each read-out direction (`evil` block 20, `sycophancy` block 26, `hallucination` block 17), with bootstrap CIs (n = 500), against the envelope of 100 norm-matched random directions.

![Trait-direction projection drift across turns versus random-direction envelopes](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c279c64d3b8d9c488e0607392225f967591fbe1b/figures/issue_958/trait_projection_drift.png)

> **Figure.** *Sycophancy-direction projections fall ≈ 6 units and hallucination-direction projections rise ≈ 4.6 units from turn 1 to 4; evil stays inside the random band.* Within-conversation slopes: sycophancy −1.98 [−2.50, −1.45], hallucination +1.45 [1.18, 1.72], evil −0.01 [−0.16, 0.15]; per-trait random bands: sycophancy [−0.35, +0.28], hallucination [−0.14, +0.11], evil [−0.19, +0.13].

This is the registered secondary, activation-level read: real answers drift along two of the three behavior read-out directions as conversations deepen, and the movement is direction-specific — the random directions move an order of magnitude less. Because answers lengthen with turn (mean 262 → 351 tokens), I recomputed slopes partialling answer and context token counts within conversation: sycophancy −2.49 and hallucination +1.77, both still far outside their partialled random bands; the evil direction stays indistinguishable from its band given the variance. No behavioral claim — judged trait rates versus turn are untested here. The stale-map residual projections are contaminated by the degenerate turn-1 map and are not interpreted; a top-principal-component companion needs the activation store (follow-up).

### The per-conversation slopes behind the drift read

What is plotted: distributions over the 500 test conversations of each conversation's own projection-versus-turn slope, per direction, with the mean and the random-direction band of mean slopes; plus the long-panel own-map skill at turns 5–8 (depth extension).

![Per-conversation trait-projection slope distributions](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c279c64d3b8d9c488e0607392225f967591fbe1b/figures/issue_958/trait_drift_per_conversation_slopes.png)

> **Figure.** *The drift is a population shift, not a handful of outliers — but hallucination is bimodal.* Per-conversation slope histograms (n = 500; evil / sycophancy / hallucination means −0.01 / −1.98 / +1.45); gray band = random-direction mean slopes.

The sycophancy shift is a broad left-shift of the whole distribution. The hallucination distribution is bimodal (a main mode near zero, a secondary mode near +5 to +8), so a conversation subpopulation drives most of the rise; characterizing it is a cheap follow-up on existing artifacts. Long-panel own-map skill at turns 5–8 is 0.32–0.39 versus 0.42 at the size-matched turn-1 fit (0.29 duplicate-excluded; 60 test conversations, and only the turn-6 difference clears zero). There is no monotone own-map decay out to turn 8 at this resolution.

---

*Conciseness note: per-result prose runs over the 120-word target, several Takeaways bullets over 30 words, two captions over cap, and the total prose budget is exceeded (verifier WARNs, acknowledged) — the extra words carry the registered interpretation guards (twin-floor conditioning, decomposition conjunct, duplicate-normalization disclosure, length partialling, null-band checks).*

**Repro:** ~1.4 h on 1× A100-80 (GCP `capture-7b` flex-start; rollouts + hooked capture) + ~35 min on `n2-highmem-16` (fits/eval) + ~50 min VM CPU across the four analysis rounds (round 5 main-panel duplicate-excluded refit ~16 min); 0 Anthropic calls. Code: [`scripts/issue958_common.py`](https://github.com/superkaiba/explore-persona-space/blob/895e4c78315d514a975d13a3965d49a8417137be/scripts/issue958_common.py), [`issue958_build_corpus.py`](https://github.com/superkaiba/explore-persona-space/blob/895e4c78315d514a975d13a3965d49a8417137be/scripts/issue958_build_corpus.py), [`issue958_rollouts.py`](https://github.com/superkaiba/explore-persona-space/blob/895e4c78315d514a975d13a3965d49a8417137be/scripts/issue958_rollouts.py), [`issue958_capture_turns.py`](https://github.com/superkaiba/explore-persona-space/blob/895e4c78315d514a975d13a3965d49a8417137be/scripts/issue958_capture_turns.py), [`issue958_fit_maps.py`](https://github.com/superkaiba/explore-persona-space/blob/895e4c78315d514a975d13a3965d49a8417137be/scripts/issue958_fit_maps.py), [`issue958_eval.py`](https://github.com/superkaiba/explore-persona-space/blob/895e4c78315d514a975d13a3965d49a8417137be/scripts/issue958_eval.py) at `895e4c7831`; round-2 re-reduction [`issue958_long_k1_transfer.py`](https://github.com/superkaiba/explore-persona-space/blob/92bfe5d96f6e111035d502060379d6d185a04691/scripts/issue958_long_k1_transfer.py) + [`issue958_r2_sidecar_and_fig.py`](https://github.com/superkaiba/explore-persona-space/blob/92bfe5d96f6e111035d502060379d6d185a04691/scripts/issue958_r2_sidecar_and_fig.py) at `92bfe5d96f`; round-4 λ-clamped refit [`issue958_long_k1_transfer_lclamp.py`](https://github.com/superkaiba/explore-persona-space/blob/f987f0a34a04186f3a1322504abe03bc6e3e94f0/scripts/issue958_long_k1_transfer_lclamp.py) at `f987f0a34a` + figure script [`issue958_lclamp_fig.py`](https://github.com/superkaiba/explore-persona-space/blob/547e27a34e8df9a1984762cbb76932ffa6c563bf/scripts/issue958_lclamp_fig.py) at `547e27a34e` (all branch `issue-958`); round-5 main-panel duplicate-excluded refit [`issue958_dup_excluded_refit.py`](https://github.com/superkaiba/explore-persona-space/blob/dea6ee420fc959db5c28a8963d31acb95f3cf214/scripts/issue958_dup_excluded_refit.py) + figure [`issue958_dup_excluded_refit_fig.py`](https://github.com/superkaiba/explore-persona-space/blob/dea6ee420fc959db5c28a8963d31acb95f3cf214/scripts/issue958_dup_excluded_refit_fig.py) at `dea6ee420f` (on `main`). Aggregated results in git: [`eval_results/issue_958/`](https://github.com/superkaiba/explore-persona-space/tree/92bfe5d96f6e111035d502060379d6d185a04691/eval_results/issue_958) (`transfer_matrix.json`, `decision_stats.json`, `forecast_curves.json`, `prefix_marginal.json`, `drift_read.json`, `maps_meta.json`; round 2: `long_k1_transfer.json`, `duplicate_first_message_groups.json`, `percell/long_1to{5..8}.npz`; round 4: [`long_k1_transfer_lclamp.json`](https://github.com/superkaiba/explore-persona-space/blob/f987f0a34a04186f3a1322504abe03bc6e3e94f0/eval_results/issue_958/long_k1_transfer_lclamp.json) + `percell/long_1to{5..8}_lclamp.npz` at `f987f0a34a`; round 5: [`dup-excluded-turn1-refit/refit.json`](https://github.com/superkaiba/explore-persona-space/blob/dea6ee420fc959db5c28a8963d31acb95f3cf214/eval_results/issue_958/dup-excluded-turn1-refit/refit.json) at `dea6ee420f`). Per-conversation cells (`percell/*.npz`), null matrices, drift projections, corpus, rollout text, activation store (27.2 GB, 54 shards) and fitted read-out map weights: [HF data repo @ `issue958_multiturn/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0fb4320d986da4fbac223db9b6f363fbe47d3d9d/issue958_multiturn) (Hub-verified this session; 496 files). Figures: [`figures/issue_958/`](https://github.com/superkaiba/explore-persona-space/tree/4f7485faca59e507747e15d82a351a78f6403fca/figures/issue_958) on `main` (round-1 figures pinned at `c279c64d3b`, the round-2 figure at `4f7485faca`, the round-4 companion figure at `0cbc16d2e4`, the round-5 main-panel refit figure `dup_excluded_turn1_refit.png` at `dea6ee420f`). Reused trait directions from [#778](https://eps.superkaiba.com/tasks/778): [`issue778_persona_vectors/analysis_tensors/rb/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/0fb4320d986da4fbac223db9b6f363fbe47d3d9d/issue778_persona_vectors/analysis_tensors/rb) `{evil,sycophancy,hallucination}.pt` — fit: same base model, read-out regime, shape (28, 3584) asserted at load. Analyzer-recomputed statistics (turns-2–4 paired deficits, ratio difference, length-partialled drift slopes) are re-reductions of the committed `percell/` + `drift_actual_projections.npz` + `unit_tokens.npz` artifacts (bootstrap seed 0; 997 draws for deficit CIs, 400 for figure CIs and partialled slopes); the duplicate-group split additionally reads the corpus (`issue958_multiturn/corpus/main.json` on HF) and is persisted as the committed sidecar; the round-2 transfer cells re-reduce the persisted map weights + long-panel store shards on HF, gated on reproducing the committed long-panel turn-1 cell (max per-row delta 1.4e-6); the round-3 long-panel duplicate-excluded reads re-reduce `issue958_multiturn/corpus/long.json` (exact first-message grouping) against the committed `percell/long_own_k*.npz` + `percell/long_1to{5..8}.npz` cells (read-out rows 15/18/20/21/25/27), with no new artifacts; the round-4 λ-clamped cells refit the turn-1 read-out rows at λ = 1,000 from the same persisted fit inputs and re-reduce the same store shards under an identical validation gate (max per-row delta 3.2e-15); the round-5 main-panel duplicate-excluded refit re-stages the main-panel store shards + `corpus/main.json` from HF (`issue958_multiturn/analysis_tensors/store/main/`), recomputes the exact/lowercased first-message duplicate grouping (cross-checked against the committed `duplicate_first_message_groups.json` — counts 604/640 and per-test flags), and refits the main-panel turn-1 map with those conversations excluded from both folds, gated on the no-exclusion refit reproducing the committed fold-A grid / `own_B` / `own_full` turn-1 cells (max delta 3.2e-11; bootstrap seed 0, 997 draws).

**Context:** lineage: fresh direction (no parent); created 2026-07-04; GPU + CPU stages ran 2026-07-04; first-pass analysis 2026-07-04; revision round 2 (critic-requested disclosure fixes + the long-panel turn-1 forward-transfer re-reduction, no new compute) 2026-07-04; revision round 3 (long-panel duplicate disclosure + claim scoping, no new compute) 2026-07-04; free-analysis follow-up round 4 (λ-clamped long-panel turn-1 refit at turns 5–8, 0 GPU-h, no new capture) 2026-07-04; free-analysis follow-up round 5 (main-panel duplicate-excluded turn-1 refit, 0 GPU-h, no new capture) 2026-07-17. Data-source tier 1 (real LMSYS conversations), scoped to the moderation-clean English ≥4-turn subpopulation with LMSYS-original (other-model) assistant text in prefixes — the Qwen-authored-prefix control at turn 2 scored 0.514 versus 0.525 for the main arm, bounding that confound as small; the turn-1-skill comparison against the single-turn parent line is population-shifted and approximate, never a replication read. Originating prompt, verbatim:

> Run this issue in the background with happy coder:
> # How does the mapping change as the context gets longer
> # Motivation
> - All our mapping experiments have been on mostly short contexts
> - We want to see if this mapping holds up at longer contexts and over many chat exchanges
>     - or generally if it changes - i.e. is the mapping from prefix + query 1 -> answer 1 the same as the mapping from prefix + query 2 -> answer 2
>     - also probably can we predict answer 2, 3, 4 from prefix + query 1 (and other combinations)
> - This could also explain something like persona drift/context rot
> # Methodology
> - Take longer multi turn conversations from some generic chat dataset
> - Train above mappings (use same conversations for all turn numbers)
> - Characterize relationship between mappings, try to predict second answer from query 2 using the first mapping, etc.
>
> (no approval needed from me)

