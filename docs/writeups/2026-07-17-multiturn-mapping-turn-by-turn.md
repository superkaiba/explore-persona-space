# Result: The context→answer activation map is near turn-stationary across a conversation — later inputs drift along behavior directions while the map that reads them holds

<!-- report-v1 -->
<!-- Single-task report on #958 (turn-by-turn multi-turn mapping). Written 2026-07-17.
     Every number re-derived from the eval_results/issue_958/* aggregate JSONs (transfer_matrix,
     decision_stats, forecast_curves, prefix_marginal, drift_read, long_k1_transfer,
     long_k1_transfer_lclamp, duplicate_first_message_groups) plus percell/ paired bootstraps
     recomputed here; figures SHA-pinned to the issue's clean-result commit
     0cbc16d2e45b8abfb639c207338d055410e79cff. Dashboard: https://eps.superkaiba.com/tasks/958 -->

## Motivation

- Earlier mapping experiments showed a per-layer ridge map from a single context's last-token activation to the mean activation of the model's own answer, on short single-turn contexts (held-out R² ≈ 0.60–0.63 at the validation-selected layer on LMSYS, [#779](https://eps.superkaiba.com/tasks/779); the summary carries across genres under group folds, [#810](https://eps.superkaiba.com/tasks/810); a rolled affine forecasts the answer's activation axis ahead of generation, [#922](https://eps.superkaiba.com/tasks/922)).
- Every one of those was on mostly short contexts. This experiment adds the turn / context-depth axis on real multi-turn conversations and asks three things: (a) is the map fit at (prefix + query 1 → answer 1) the same object as the map at (prefix + query k → answer k)?; (b) can later answers be predicted from an earlier-turn state?; (c) does any systematic change of the map with depth give persona drift / context rot a mechanistic footing?
- If the frozen map *changed* as context accumulated, persona drift would live in a rewired state→behavior relation. If it *holds* while the inputs move, drift is input drift read by a fixed map — a different, testable picture.

## TLDR

- **Turns 2–4 are near-stationary.** Maps fit at one turn transfer to the other turns with held-out skill deficits of 0.000–0.023 against own-turn skill 0.45–0.49 — every deficit an order of magnitude below the pre-registered 0.10 threshold. Three of the six deficits sit clear of the ±0.01 twin fit-noise floor (the larger-separation ones), growing with turn separation and near-symmetric (2→4 ≈ 4→2). The transfer is not just static inputs: the mean input direction rotates away from turn 1 (cosine 0.97 → 0.95) and the turn-1 90%-variance basis captures only 60–70% of later-turn input variance versus 85% at its own turn.
- **The main-panel turn-1 degeneracy is a duplicate-first-message artifact — removing it restores a positive, near-stationary turn-1 map.** As fitted with duplicates in the fold, the turn-1 map has no skill even at its own turn (−0.02 half-sample, −4.98 full-n): 12.1% of conversations (604 of 5,000) share an exact first message, so identical seeded rollouts recur and cross-validation memorizes them at near-zero regularization (55 of the 56 near-perfect test conversations sit in duplicate groups). Excluding those 604 conversations from both the fit and test folds (follow-up `dup-excluded-turn1-refit`, landed 2026-07-18) flips own-turn-1 skill to +0.51 [0.489, 0.529] half-sample / +0.54 [0.520, 0.559] full-n, GCV moves to grid-max shrinkage (λ = 1,000, matching turns 2–4), and the matched-λ turn-1→{2,3,4} deficits land at 0.105 / 0.138 / 0.156 (half-sample) — inside the pre-registered [0.023, 0.23] band. The pre-registered hypothesis is confirmed, not falsified: near-stationarity extends to turn 1 on the main panel in weakened form.
- **On the long panel (turns 5–8), the turn-1 failure is a regularization artifact, not a different map.** As fitted the turn-1 map scores −0.25 to −0.21 (below the corpus-mean null); refit at the target maps' ridge λ (= 1,000, versus the turn-1 map's ≈ 5) it transfers +0.14 to +0.17 (bootstrap CIs excluding zero, n = 60), with a residual deficit of 0.16–0.23 — an order of magnitude above the turns-2–4 deficits, so near-stationarity reaches back to turn 1 in weakened form.
- **Later answers are partly forecastable from the turn-1 pre-answer state, but not far.** Forecast skill 0.175 [0.147, 0.201] at turn 2 decays to 0.062 by turn 4, where copy-previous-answer (0.193) and prefix-only (0.156) baselines both overtake it; one-step-ahead forecast skill does not decay (0.175 → 0.195 → 0.207).
- **With depth the context map weakens while the prefix-only map holds.** The prefix-only/context skill ratio rises 0.257 → 0.301 → 0.321 from turn 2 to 4 (difference 0.064 [0.013, 0.117]); the rise is mostly the context map declining (−0.039 [−0.058, −0.020]), not the prefix map strengthening (+0.021 [−0.006, +0.049], CI touching zero).
- **Answer activations drift along behavior read-out directions as conversations deepen.** The sycophancy-direction projection falls ≈ 6 units (within-conversation slope −1.98 [−2.50, −1.45]) and the hallucination-direction rises ≈ 4.6 units (+1.45 [1.18, 1.72]), both far outside their norm-matched random-direction bands and insensitive to length partialling; the evil direction is indistinguishable from its band. This is an activation-level read only — no behavioral judge was run.

## Methodology

- **Model:** `Qwen/Qwen2.5-7B-Instruct`, frozen (bf16 forward, fp16 capture). No training — every map is a closed-form per-layer affine ridge fit.
- **Data (tier 1, real conversations):** `lmsys/lmsys-chat-1m`, English, roles strictly alternating, moderation-flagged conversations dropped, sha256 dedup over the concatenated first-K user messages, per-turn input ≤ 7,168 tokens. Two disjoint panels: a **main panel** of 5,000 conversations with ≥ 4 user turns (turns 1–4) and a **long panel** of 600 conversations with ≥ 8 user turns (turns 1–8). Split by conversation: 4,000/500/500 main (seed 42), 480/60/60 long.
- **Per-arm completion provenance:** for each (conversation, turn) unit the model generates its **own single on-policy answer** to the real conversation so far — vLLM, temperature 1.0, top_p 0.95, seed 42, n = 1 (one seeded stochastic sample per unit, not greedy, not multi-draw). The prefix carries the LMSYS originals' assistant text, which is **other models'** output; a control arm regenerates turn 2 under a Qwen-authored turn-1 answer and scores 0.514 versus 0.525 for the main arm, bounding that confound as small.
- **Computed quantities:** a hooked forward captures five positions × 29 residual rows (embedding + 28 blocks) at fp16. The context summary is the last context-token activation; the target is the mean activation over the generated answer span (through the end-of-turn token). Maps are dual/Gram-space affine ridge (3584→3584), λ by GCV over `logspace(-2, 3, 25)`, X standardized / Y centered on train-fold moments.
- **Metric:** *skill* = 1 − SSE(prediction) / SSE(train-fold corpus mean) on held-out mean-pooled answer activations, read at the frozen mean of six blocks {14, 17, 19, 20, 24, 26}. This is a fraction-of-variance-explained score, not the single-turn parent's single-layer R², so own-turn skill (≈ 0.45–0.49) is not directly comparable to #779's 0.60–0.63 — the comparison to the single-turn line is population-shifted and approximate, never a replication read.
- **Matched-target discipline:** every column of the transfer matrix scores maps against the **same** turn-k held-out answer targets; deficits are own(k→k) − transfer(j→k), paired by conversation. Stability comparisons are at matched fit size n = 2,000 (half-sample twins give the fit-noise floor); forecast and prefix reads use full-n maps.
- **Baselines / nulls:** shuffled-pairing bands (all realized bands sit far below zero, 97.5th percentiles −0.27 to −6.3), a copy-previous-answer null, the train-fold corpus-mean denominator, and — for the trait read — 100 norm-matched random directions. Bootstrap: 997 paired-by-conversation draws, seed 0.
- **Reused directions:** the three persona-vector trait directions (`evil`, `sycophancy`, `hallucination`) from [#778](https://eps.superkaiba.com/tasks/778), each extracted per the project persona-vectors recipe (5 contrastive system-prompt pairs, 10 on-policy rollouts per side, `claude-sonnet-4-5-20250929` graded judge-filter with malformed returns dropped, per-layer positive-minus-negative mean difference), consumed unit-normalized at their frozen read-out layers (evil block 20, sycophancy 26, hallucination 17).

## Results

### _Result 1: Maps fit at turns 2, 3, and 4 transfer to each other almost losslessly_

The first read is the (fit turn j → eval turn k) held-out skill grid, and the six turns-2–4 deficits against the twin fit-noise floor. Every column is scored against the same turn-k answer targets; deficits are paired by conversation with a 997-draw bootstrap (recomputed here from the committed `percell/` per-unit SSE).

**Plot: Own-turn skill and the turn-1-map read, by turn**

![Own-turn map skill, the turn-1 map applied at each turn, and the shuffled-pairing band, versus turn number](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0cbc16d2e45b8abfb639c207338d055410e79cff/figures/issue_958/hero2_skill_vs_turn.png)

**Plot: Cross-turn transfer matrix**

![Four-by-four grid of held-out skill for maps fit at turn j evaluated at turn k; turns 2 to 4 near 0.44 to 0.49, turn-1 row strongly negative](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0cbc16d2e45b8abfb639c207338d055410e79cff/figures/issue_958/hero1_transfer_matrix.png)

**Takeaways:**

- Own-turn skill is 0.489 / 0.460 / 0.454 (turns 2/3/4, matched n = 2,000 fold A) and the off-diagonal cells land right next to them: the six turns-2–4 deficits are D(2→3) 0.007 [0.002, 0.012], D(2→4) 0.015 [0.009, 0.021], D(3→4) −0.000 [−0.005, +0.005], D(3→2) 0.011 [0.006, 0.016], D(4→2) 0.023 [0.017, 0.029], D(4→3) 0.004 [−0.001, +0.010] — all far below the 0.10 threshold.
- Three of the six (the larger separations) clear the ±0.01 twin fit-noise floor, so there is a real but tiny map change that grows with turn separation and is near-symmetric (2→4 ≈ 4→2, 0.023 vs 0.015). The other three sit inside the twin floor.
- The transfer is not explained by static inputs: the mean input direction rotates away from turn 1 (cosine 0.969 / 0.954 / 0.946 for turns 2/3/4) and the turn-1 90%-variance basis captures only 60–70% of later-turn input variance versus ~85% at its own turn — the inputs move measurably, and the map still reads them.
- Scope: every frozen read-out row here selected the grid-maximum λ = 1,000, so heavy shared shrinkage can compress differences the twin floor does not control; the per-unit view ([`explore_per_row_skill.png`](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0cbc16d2e45b8abfb639c207338d055410e79cff/figures/issue_958/explore_per_row_skill.png)) confirms the deficits are population-wide, not a few outliers.

### _Result 2: The main-panel turn-1 degeneracy is a duplicate-first-message artifact — excluding the duplicates restores a positive, near-stationary turn-1 map_

The registered turn-1 read formally fired (stale deficits 1.96–2.53, residual map-change 1.46–1.71, CIs excluding the twin floor). But the source fit itself is broken, a failure mode the registered rule did not anticipate — and a landed follow-up refit confirms the break is entirely the duplicate artifact.

**Plot: Turn-1 per-conversation skill, split by duplicate first messages**

![Per-conversation turn-1 skill versus first-message token count, duplicate-group conversations scoring near 1 and unique ones averaging about minus 0.33; plus five aggregate turn-1 reads](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0cbc16d2e45b8abfb639c207338d055410e79cff/figures/issue_958/turn1_duplicate_artifact.png)

**Takeaways:**

- The turn-1 map has no skill at its own turn: −0.019 at matched n = 2,000 (fold B), −4.98 at full n = 4,000, and −0.20 after excluding exact-duplicate test conversations. A map that cannot predict its own turn cannot be a fair source for a stationarity test.
- Mechanism: 604 of 5,000 conversations (12.1%) share a byte-identical first message (the largest groups are copies of one 5-character greeting), so their seeded turn-1 rollouts are identical — repeated training/test pairs. Cross-validation then selects near-zero regularization (fold λ 0.46–0.75), memorizing the duplicates: 55 of the 56 near-perfect (skill > 0.95) test conversations sit in duplicate groups, while unique conversations average ≈ −0.33.
- A test-only exclusion already hinted at this in-body (excluding exact-duplicate *test* conversations lifted the aggregate to −0.20), but with the duplicates still in the *fit* the map stayed degenerate; the landed refit below excludes them from both folds.

**Plot: Turn-1 map refit with exact-duplicate conversations excluded**

![Turn-1 own-map skill and turn-1 to 2/3/4 transfer, with versus without the 604 exact-duplicate first-message conversations; own-turn-1 skill moves from strongly negative to about +0.5 and transfer becomes positive once the duplicates are excluded](https://raw.githubusercontent.com/superkaiba/explore-persona-space/dea6ee420fc959db5c28a8963d31acb95f3cf214/figures/issue_958/dup_excluded_turn1_refit.png)

**Takeaways:**

- Excluding the 604 exact-duplicate-first-message conversations from both the fit and test folds flips own-turn-1 skill from −0.02 / −4.98 (half-sample / full-n) to +0.51 [0.489, 0.529] / +0.54 [0.520, 0.559] — a positive fit of the same magnitude as the other turns.
- With the duplicates gone, GCV abandons the degenerate near-zero λ (half-sample 0.46–0.75; full-n 619 / 383 and 0.026 on two rows) for the grid-maximum λ = 1,000 on every read-out row, matching turns 2–4 — so the round-4 λ-clamp of Result 3 becomes a no-op here (as-fitted transfer equals matched-λ transfer).
- The matched-λ turn-1→{2,3,4} deficits land at 0.105 / 0.138 / 0.156 (half-sample; 0.119 / 0.152 / 0.167 full-n; the lowercased-duplicate normalization agrees at 0.112 / 0.145 / 0.164) — larger than the ≤ 0.023 turns-2–4 deficits but inside the pre-registered [0.023, 0.23] band (its ceiling is the long-panel weakened-turn-1 residual of Result 3). The pre-registered hypothesis is confirmed, not falsified: near-stationarity extends to turn 1 on the main panel in weakened form, and the earlier turn-1 exception was purely the duplicate-memorization artifact. The refit reproduced the committed with-duplicate turn-1 cells to 3e-11.

### _Result 3: On the long panel, the turn-1 forward-transfer failure is a ridge-shrinkage artifact_

The long panel (600 conversations, ≥ 8 turns) gives a turn-1 map with a non-degenerate own-turn fit (own skill 0.42). Applied at turns 5–8 it appears to fail — but the apparent failure tracks a regularization mismatch, not a genuinely different map. The figure shows the turn-1 map both as fitted and refit at the target maps' λ, beside each turn's own map.

**Plot: Long-panel turn-1 transfer — as fitted vs λ-clamped vs own-turn maps**

![Long-panel turn-1 map at turns 5 to 8: as-fitted transfer negative, lambda-clamped transfer positive around 0.15, versus own-turn skill around 0.35](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0cbc16d2e45b8abfb639c207338d055410e79cff/figures/issue_958/long_turn1_transfer_lclamp.png)

**Takeaways:**

- As fitted (turn-1 λ ≈ 5), the map scores −0.225 / −0.206 / −0.251 / −0.208 at turns 5/6/7/8 — below the corpus-mean null — for raw deficits of 0.53–0.61 versus own-turn skill 0.32–0.39. Moment recalibration (re-applying turn-1 weights under target-turn moments) recovers about half the gap (to +0.05 to +0.07), leaving a residual deficit of 0.26–0.32. Raw transfer still clears the shuffled-pairing band, so it is miscalibrated rather than uninformative.
- Refit with per-row λ clamped to the target maps' own GCV selection (λ = 1,000 on all six read-out rows), raw transfer moves from the −0.25 to −0.21 band into **+0.14 to +0.17** (0.166 / 0.160 / 0.137 / 0.161, CIs excluding zero, n = 60), recalibrated 0.27–0.29 versus own-map 0.32–0.39. The λ ≈ 5-vs-1,000 mismatch carried the entire negative sign of the raw forward transfer.
- The residual deficit of 0.16–0.23 (after clamping) still exceeds the turns-2–4 deficits by an order of magnitude: turn 1 at 4–7-turn separations is the weakest transfer in the study, but no longer a failure. Excluding the 8 duplicate test conversations of the long panel moves every read by at most 0.02, and the refit reproduced the committed turn-1 own cell to within 3e-15. Two qualifications: all four target maps selected λ = 1,000, the grid maximum, so the match sits at a censored boundary; and the main-panel turn-1 map fails identically as fitted (−4.74 versus 0.52–0.56 for the turns-2–4 maps), pending the running duplicate-excluded refit.

### _Result 4: The turn-1 state partly forecasts later answers, decaying below trivial baselines by turn 4_

A cross-horizon map from the turn-1 pre-answer state to answer k tests whether earlier state carries later-answer information, against a copy-previous-answer null and the prefix-only map.

**Plot: Forecast skill vs baselines across turns**

![Forecast skill from the turn-1 state versus target turn, with own-turn, prefix-only, and copy-previous-answer curves; forecast starts highest at turn 2 and falls below the baselines by turn 4](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0cbc16d2e45b8abfb639c207338d055410e79cff/figures/issue_958/forecast_vs_nulls.png)

**Takeaways:**

- At turn 2 the turn-1 state carries real cross-turn information: forecast skill 0.175 [0.147, 0.201], above both the prefix-only map (0.135; paired difference CI [0.025, 0.054]) and the copy-previous null (−0.132 — answer 1 is a poor template for answer 2, and negative at turn 2).
- The information decays fast with distance: forecast skill 0.107 at turn 3, 0.062 at turn 4, where copy-previous (0.193) and prefix-only (0.156) both overtake it (the forecast-minus-copyprev CI at turn 4 is [−0.182, −0.077], excluding zero).
- One-step-ahead forecasts do not decay (0.175 → 0.195 → 0.207 for 1→2, 2→3, 3→4): what fades is prediction at a distance, a stationarity-consistent side read. Per-conversation forecast skill is nearly uncorrelated with answer token count (−0.05 to −0.08), so it is not doing length bookkeeping.

### _Result 5: The context map weakens with depth while the prefix-only map holds_

Fitting a prefix-only map (the pre-query state alone) alongside the full context map at each turn separates two accounts of any depth change.

**Plot: Prefix-only vs context-map skill by turn**

![Prefix-only and context-map skill by turn with the prefix-over-context ratio annotated; context skill declining across turns 2 to 4 while prefix-only skill is roughly flat](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0cbc16d2e45b8abfb639c207338d055410e79cff/figures/issue_958/prefix_dominance.png)

**Takeaways:**

- The prefix-only/context skill ratio rises 0.257 [0.213, 0.300] → 0.301 [0.255, 0.348] → 0.321 [0.271, 0.371] across turns 2–4, a difference of 0.064 [0.013, 0.117] (all CIs re-derived here from `percell/` paired bootstraps).
- The rise is mostly the context map weakening with depth (component −0.039 [−0.058, −0.020], CI excluding zero); the prefix map's gain is smaller and its CI touches zero (+0.021 [−0.006, +0.049]).
- Absolute prefix-only skill stays modest (0.135 / 0.149 / 0.156), and within every turn cell per-conversation skill falls with context length (Spearman −0.23 at turn 2 to −0.11 at turn 4) — the within-cell length gradient is larger than the between-turn change. The registered length-matched turn comparison was not run (a planned-vs-actual gap); these within-cell gradients stand in.

### _Result 6: Answer activations drift along behavior read-out directions with turn_

The registered secondary read projects the actual answer activations onto the three reused persona-vector trait directions and asks whether the change with turn exceeds norm-matched random directions. This is an activation-level read; no behavioral judge was run on these conversations.

**Plot: Trait-direction projection drift across turns**

![Change from turn 1 in the mean projection onto the evil, sycophancy, and hallucination directions, with bootstrap CIs, against the random-direction envelope; sycophancy falls, hallucination rises, evil flat](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0cbc16d2e45b8abfb639c207338d055410e79cff/figures/issue_958/hero3_trait_drift.png)

**Takeaways:**

- Sycophancy-direction projections fall ≈ 6 units from turn 1 to 4 (−59.2 → −65.4; within-conversation slope −1.98 [−2.50, −1.45]) and hallucination-direction projections rise ≈ 4.6 units (−1.8 → +2.8; slope +1.45 [1.18, 1.72]). Both slopes sit far outside their norm-matched random-direction bands ([−0.35, +0.28] and [−0.14, +0.11]) — the random directions move an order of magnitude less.
- The evil direction is indistinguishable from its band (slope −0.01 [−0.16, +0.15], band [−0.19, +0.13]).
- The movement survives partialling answer and context token counts within conversation (both directions stay well outside their partialled bands), so it is not length bookkeeping. Per the per-conversation slope view ([`trait_drift_per_conversation_slopes.png`](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0cbc16d2e45b8abfb639c207338d055410e79cff/figures/issue_958/trait_drift_per_conversation_slopes.png)), the sycophancy shift is a broad left-shift of the whole population, while the hallucination distribution is bimodal (a main mode near zero, a secondary mode near +5 to +8), so a subpopulation drives most of that rise.
- Strict scope: no behavioral claim follows — judged trait rates versus turn were not measured here, and the direction is a linear read-out only.

## Conclusion & Next steps

- Within a conversation the frozen context→answer map barely changes across turns 2–4, and — once the ridge-shrinkage confound is removed — reaches back to turn 1 in weakened form on the long panel. What moves is the *input*: its direction rotates and its variance basis shifts with depth, and its projection drifts measurably along two of three behavior read-out directions. So over these depths persona drift / context rot looks like **input drift read by a roughly fixed map**, not a rewired state→behavior relation. There is also no monotone own-map decay out to turn 8 (long-panel own skill 0.32–0.39, turn-1 fit 0.42).
- The one apparent turn-1 exception was a regularization-selection artifact, not a different map — a caution for any "the map changed" reading that does not control the ridge λ across the maps being compared.
- **Settled since drafting:** the duplicate-excluded main-panel turn-1 refit (`dup-excluded-turn1-refit`, landed 2026-07-18) removed the duplicate-first-message artifact and confirmed the main-panel turn-1 map is positive (own skill +0.51 / +0.54) and near-stationary in weakened form (matched-λ deficits 0.10–0.16, inside the pre-registered [0.023, 0.23] band) — see Result 2.
- **Filed follow-ups** (from the issue's `epm:follow-ups` marker, all judged not-redundant, parked for a manual pick): a judged trait-rate-vs-turn behavioral companion (`judged-trait-drift-companion`) that would turn the activation-level drift of Result 6 into a behavioral claim (≈ 30K judge calls, needs a user go-ahead); a hallucination bimodal-subpopulation characterization (`halluc-bimodal-subpop`, free analysis on existing artifacts); and an unranked top-principal-component drift companion asking whether the residual drift is a generic style/length axis rather than a trait axis.
- Task dashboard: https://eps.superkaiba.com/tasks/958

<!-- numbers verified against eval_results/issue_958/* at a1129eb284 (aggregate JSONs) + percell/ bootstraps recomputed 2026-07-17; figures pinned at 0cbc16d2e45b8abfb639c207338d055410e79cff. Turn-1 update (2026-07-18): dup-excluded numbers from eval_results/issue_958/dup-excluded-turn1-refit/refit.json (exact-duplicate regime, the pre-registered primary) + figure dup_excluded_turn1_refit.png, both pinned at dea6ee420fc959db5c28a8963d31acb95f3cf214. -->
