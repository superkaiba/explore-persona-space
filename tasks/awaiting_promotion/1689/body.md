---
title: Changing surface framing costs the context-to-answer map far more than changing
  speaker identity, and changing both adds little beyond framing's cost (LOW confidence)
kind: experiment
tags:
- followup-auto
created_at: '2026-07-25T18:54:59Z'
has_clean_result: true
parent_id: 825
origin_prompt: 'i want a controlled comparison between: assistant with chat template
  / assistant without chat template / assistant in story / user in chat template /
  user without chat template / user in story -- for user it should be either: real
  user generated turns from LMSYS, always haiku generated, always on-policy generated
  by Qwen itself / other AI character with chat template / other AI character without
  chat template / other AI character in story / other assistant-like human character
  with chat template / other assistant-like human character without chat template
  / other assistant-like human character in story / other random character with chat
  template / other random character without chat template / other random character
  in story. (all except user should always be on policy generated) all in both instruct
  and base model. all characterized according to the tiers of mapping similarity.
  [then] run the full thing in the background with happy coder'
workflow: v1
goal: 'Determine, on ONE shared row-paired corpus, whether the context->answer linear
  map is the SAME operator across five speaker identities (assistant; user; a fiction
  AI character; an assistant-like human character; an ordinary/random character) and
  three surface framings (chat template; plain User:/Assistant: text; narrative-prose
  story), in Qwen2.5-7B base AND instruct, with the user arm run under all three u2
  provenances (real LMSYS, haiku-simulated, on-policy) and every non-user condition
  on-policy from the measured model - and, for every ordered pair, WHICH tier of the
  mapping-similarity ladder (direct transfer / context offset / answer offset / bias
  refit / global scaling / rotation / context reparameterization / answer reparameterization
  / full A.M.B) is the weakest correction that reconciles the two maps. Both mapping
  arms (prefix and context) are fit for every cell.'
relates_to:
- identity-contextual-vs-base
- identity-cb-duality
---
# Changing surface framing costs the context-to-answer map far more than changing speaker identity, and changing both adds little beyond framing's cost (LOW confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_1689.md](https://github.com/superkaiba/explore-persona-space/blob/24c35358797446fb7053859654d37d7be41eeb82/docs/methodology/issue_1689.md) · [gist](https://gist.github.com/superkaiba/7657c786473754563289a0ff7ccb1102)

## Takeaways

- Into the highest-ceiling target (instruct assistant-chat), full-correction transfers recover median 0.86-0.88 of ceiling for identity-only changes, 0.41-0.47 for framing-only, and 0.28-0.34 for both, so framing, not identity, carries the cost.
- Chat-framing identity swaps reconcile at rung 1 (direct transfer); assistant framing pairs reconcile at no rung in 12 of 12 base and 7 of 12 instruct (pair, arm) combinations; the rest need rung 9.
- Per-pair operator cosines back the ordering in raw coordinates: every identity-only spoke pair (0.14-0.27) beats every framing-changing pair (0.01-0.16); the Procrustes-aligned read compresses the classes to 0.13-0.30 and tracks shared-row count (Spearman 0.56-0.76), so it is estimation-confounded.
- Real vs haiku-simulated user turns (prefix arm, n=3,800, ceilings 0.41-0.78) reconcile only at rung 6 or deeper; the differences are answer-side by construction.
- Measurement validity restricts the lattice: 54 of 126 context-arm pairs are construct-invalid, 31-52 more per model-arm have non-positive ceilings, and the bimodal rung distribution mostly reflects which targets have usable ceilings; the ceilings themselves are not grid-limited — a ridge grid widened 3 decades moves 0 of 84 cell-arm ceilings by more than 0.02.
- The user-on-policy arm contributes no independent data (its stored activations duplicate the haiku arm on all shared conversations while the raw turns differ), and all 30 on-policy pools missed the 0.80 yield floor (character-chat 0.04-0.06), leaving character cells small and judge-selected.

## Goal

**This experiment in context:** Prior tasks measured the context-to-answer linear map inside single corpora: [#825](https://eps.superkaiba.com/tasks/825) fit the map on LMSYS two-turn conversations, [#1345](https://eps.superkaiba.com/tasks/1345) found story-framing transfer fails asymmetrically, and [#1310](https://eps.superkaiba.com/tasks/1310) found character shifts re-encode the answer side, plus a three-cell aligned-cosine contrast where identity-only similarity (0.740) exceeded framing-only (0.488) and both-changed (0.485). Because every prior cell lived in a different corpus, the mapping-similarity ladder's data-paired middle rungs were never computable across speaker or framing. This experiment builds ONE row-paired corpus rendered into 21 speaker-by-framing conditions per model and asks, for 126 ordered condition pairs, which of 9 correction tiers (direct transfer through full A·M·B reparameterization) is the weakest that reconciles the two fitted maps, in Qwen2.5-7B base and instruct, both mapping arms (prefix and context). A user-requested addendum adds the framing-by-identity interaction test on continuous recovery reads.

**Broader narrative:** If "who speaks" only re-coordinates a shared context-to-answer operator, persona is a shallow wrapper over one mechanism; if each speaker or framing carries a genuinely different operator, persona transfer and leakage prediction need per-condition maps. This lattice makes "same operator in different coordinates" vs "different operator" decidable on shared rows — and finds surface framing, not speaker identity, is where the operator changes.

## Methodology

**Design:** N=3,800 two-turn LMSYS conversations (English, exactly 2 turns, token-budget feasible under the tightest framing — chat template plus persona header) rendered into 21 conditions: 7 identities (assistant; user with real-LMSYS / haiku-simulated / self-generated second turns; HELIOS, an AI character; Wren, an assistant-like human; Dana, an ordinary person) crossed with 3 framings (chat template; plain text; narrative story). All non-user completions are generated on-policy by the measured model and judge-filtered; each conversation appears up to 3 times in the generation files and the ladder keeps the first judge-kept row per conversation. Activations are captured teacher-forced at layer 19 (layers 14/18/26 captured but never fit — documented scope shrinkage): X_prefix = end of everything before the second user turn; X_context = one token past the end of the second user turn; Y = end of the answer span. For user cells the answer span IS the second user turn, so X_context equals Y by construction (identity+bias baseline R² = 1.0 exactly) — every user-cell context-arm read is construct-invalid, and in plain-text framing the user-cell prefix boundary also collapses onto the answer end (both arms invalid). Ordered pairs (126 per model): a 40-pair spoke through assistant-chat, within-identity framing pairs, within-framing identity pairs, and user-provenance pairs; both arms fit for every pair. A zero-GPU follow-up round (2026-07-29) added the plan addendum's per-pair Procrustes operator-cosine battery over the same layer-19 store. A second zero-GPU follow-up round (2026-07-29, `wider-lambda-ceilings`) refit all 84 per-cell layer-19 within-cell ceilings on a ridge grid widened three decades, with a conditional ladder re-read for any pair whose ceiling moved (none did). Conciseness note: the body carries 10 result sections for the 126-pair lattice, so the total-prose-budget WARN, residual per-result word-cap and paragraph sentence-cap WARNs, and over-30-word Takeaways bullet WARNs are acknowledged deliberately, as is the deliberate link (not embed) of the base-model heatmap beside its instruct sibling.

**Training:** **N/A — no model training.** Capture/fit/generation hyperparameters:

| Hyperparameter | Value | Source |
|---|---|---|
| Models | `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-7B-Instruct` | plan §4 |
| Corpus | 3,800 LMSYS two-turn conversations (oversampled from the 3,000 target) | `scripts/issue1689_common.py` `N_SOURCE_LMSYS` |
| Layer fit | 19 (of captured 14/18/19/26) | `epm:results` card, v75 descope |
| Ridge λ grid | `np.logspace(-2, 4, 13)`, inner 3-fold conversation-grouped CV | `issue1689_fit_ladder.py` `LAMBDAS`, `_fit_ridge_inner_group_cv` |
| Pair split | single conversation-grouped 80/20 split (fold 0 of 5 as test); source map fit on all source rows | `issue1689_fit_ladder.py` `_ladder_pair_core` (deviation from plan §4's per-rung 5-fold CV) |
| Reconciliation bar | transfer R² ≥ 0.90 × within-target held-out R²; recomputed at 0.85 / 0.95 | `issue1689_common.py` `RUNG_REACHED_THRESHOLD`; plan §6.6 item 3 |
| Bootstrap | 200 conversation-level draws on 76 pairs (39 base, 37 instruct) checkpointed before the final descope; 0 draws on the rest | ladder JSONs `bootstrap_draws.n_draws`; plan 1000 → 200 (R15) → 0 (v75, user-directed) |
| Null | 40-draw matched-capacity shuffled-answer null, every pair | ladder JSONs; plan §5 |
| Generation | temperature 0.7, top-p 0.95, max_new_tokens 1024, n=1 per prompt, vLLM | `issue1689_common.py` `GEN_TEMPERATURE`/`GEN_TOP_P`/`GEN_MAX_NEW_TOKENS` |
| Judge | `claude-sonnet-4-5-20250929`, 3 draws, temperature 0.7, max_tokens 300, keep mean ≥ 50 | `issue1689_common.py` `JUDGE_*` |
| Yield floor | 0.80 (missed by all 30 pools) | `issue1689_common.py` `YIELD_FLOOR`; `onpolicy_stats/` |
| Fit engine | torch fp64 GPU, numpy-equivalence gated at atol 1e-4 (PASS, 4 of 4 pair-arms) | `epm:results` card |
| Procrustes battery (follow-up round) | per-cell ridge on all cell rows, same λ grid; data-paired orthogonal alignment on each pair's conversation intersection; 200 Haar rotation-null draws, seed 42 | `issue1689_procrustes_battery.py` `summary_L19.json` metadata |
| Wider-λ re-check (follow-up round) | grid `np.logspace(-2, 7, 19)` (superset of the published 13-point grid, same half-decade spacing), same inner 3-fold conversation-grouped CV, 5 outer folds, seed 42; parity gate: a 13-grid refit of the largest cell-arm must reproduce the published R² within 1e-3 (measured gap 2.2e-16); store at HF revision `d1010a25f8` | `issue1689_lambda_recheck.py`; `wider-lambda-ceilings/summary.json` |

**Evaluation:** For each ordered pair (source S, target T) and arm, the source ridge map is corrected by 9 successively stronger tiers — 1 direct, 2 context offset, 3 answer offset, 4 bias refit, 5 global scale, 6 orthogonal rotation, 7 context-side ridge reparameterization (A-only), 8 answer-side (B-only), 9 full A·M·B — each read as held-out R² on T against the reconciliation bar. The fit script codes "no rung reconciles" as rung 9; this analysis separates the two (an explicit no-reconcile code) and adds recovery fractions (rung R² divided by the target's within-cell ceiling). The matched-capacity null is degenerate for the rung statistic: shuffling answers drives the target ceiling non-positive, the bar becomes vacuous, and every null draw reads rung 1 (null p97.5 = 1.0 on 504 of 504 pair-arms) — a rung-1 observation is therefore not separable from the null by the ordinal alone, and discrimination rests on the ceilings and recovery fractions. Identity+learned-bias baseline and kNN retrieval (k in 1/5/10, euclidean and cosine, chance stated per pool) are reported per cell in `percell/`; the identity+bias baseline is what exposes the degenerate user arms. Robustness check (resolved by the wider-grid round): the inner-CV ridge λ selection pinned at the grid ceiling (λ = 10^4) in 265 of 420 layer-19 fold-fits; refitting all 84 cell-arms with the ceiling raised to 10^7 moved no ceiling by more than 0.02 and left no fold-fit selection above 10^5, so neither the ceilings nor the rung reads that inherit them are grid-limited (see the wider-grid result). No p-values are computed by the rig; uncertainty reads come from the 200-draw conversation-level bootstrap where present and from cross-pair spread elsewhere. Language-intrusion audit (Qwen under an English eval): CJK-matching rows are 0.4-2.2% of kept completions in the assistant and instruct pools the headline reads rest on, and 6.7-9.4% in base character-chat and base user-on-policy pools (e.g. 50 of 531 HELIOS-chat rows, 34 of 480 user-on-policy-chat rows); no judged-rate headline exists to recount, and the affected pools are already flagged as low-yield.

**Data extraction:** Ladder outputs: `eval_results/issue_1689/ladder/ladder_<model>_L19.json` (126 pairs × 2 arms: 9 point R²s, within-target ceiling, reach bar, rung-reached, 40-draw null, bootstrap draws where present) plus per-pair checkpoints in `pairs_<model>_L19/`. Per-cell ceilings and baselines: `eval_results/issue_1689/percell/` (42 JSONs). Generation yields and judge drop classes: `eval_results/issue_1689/onpolicy_stats/` (30 JSONs; the categorized drop counters overlap and do not reconcile with n_input minus kept, so only `yield_frac` — verified equal to the kept-store row counts — is quoted). The analyzer digest (validity flags, recovery fractions, threshold sweep) is `eval_results/issue_1689/analyzer/pair_digest.csv`, produced by `scripts/issue1689_analyzer_digest.py`; figures by `scripts/issue1689_analyzer_figures.py`. Follow-up battery: `eval_results/issue_1689/procrustes/` (two per-model battery JSONs + `summary_L19.json`) from `scripts/issue1689_procrustes_battery.py` — per-cell maps refit on all cell rows, a fit-grain deviation from the ladder's per-pair paired-row source fits; alignment uses each pair's conversation intersection (96-3,800 shared rows, below d = 3,584, so the rotation beyond the informed rank is an arbitrary orthogonal completion); figure by `scripts/issue1689_procrustes_figure.py`. Wider-λ re-check: `eval_results/issue_1689/wider-lambda-ceilings/` (`summary.json`, an empty `affected_pairs.json`, `percell_wide19/` — 84 per-cell-arm JSONs with per-fold selected λs and the parity fields) from `scripts/issue1689_lambda_recheck.py`; figure by `scripts/issue1689_lambda_recheck_figure.py`.

**Sample training/evaluation data + completions:** Random sample (seed 42) of 3 of 8,484 kept rows, sanitized for context hygiene (~15-word excerpts; LMSYS-class real-user corpus). Full rows: [assistant-chat instruct shards](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d1010a25f81ce184f68a9cc0ed49bce9736b80dd/issue1689_speaker_lattice/raw_completions/gen).

```
row 1309 (judge 95.0): u1 "The subject is entity that performs the action of the verb..." ->
  a2 "Certainly! Here is a more detailed breakdown of the information extracted from the news:" [truncated]
row 228 (judge 95.0): u1 "Write a fizzbuzz program in python" ->
  a2 "Sure! Let's break down the FizzBuzz program in Python and explain it step by step." [truncated]
row 51 (judge 95.0): u1 "help me to explain Theoretical solution for tubular steel sections" ->
  a2 "Certainly! When analyzing tubular steel sections, we can use theoretical methods and Finit" [truncated]
```

Random sample (seed 42) of 2 of 480 kept user-on-policy rows, same sanitization; full rows: [user-on-policy chat base file](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d1010a25f81ce184f68a9cc0ed49bce9736b80dd/issue1689_speaker_lattice/raw_completions/gen). Both self-generated "user" turns read like assistant text — which is why 96% of this pool failed the user-likeness judge (and this arm's capture was defective regardless; see the duplication result):

```
row 327 (judge 55.0): u1 "I want to create a multimodal AI model. you help me?..." ->
  u2 "Sure, you can use a pre-trained language model like GPT-3 to generate the architectural pl" [truncated]
row 57 (judge 78.3): u1 "what are some best practices for time management?" ->
  u2 "Sure, here's a list of some of the top 100 best practices for time management: 1. Priorit" [truncated]
```

Judge-dropped rows were not persisted (only kept rows ship under `raw_completions/gen/`), so dropped (non-firing) examples cannot be quoted — a persistence gap of the run, not of this analysis.

## Results

### Per-cell ceilings and construct validity leave most of the lattice uninterpretable

Within-cell held-out R² at layer 19 for all 21 conditions × 2 models × 2 arms, from the per-cell fits; hatched red bars mark arms where the identity+bias baseline reads R² = 1.0 exactly.

![Per-cell within-cell held-out R2 for both arms and models with degenerate user arms hatched](https://raw.githubusercontent.com/superkaiba/explore-persona-space/089b46a530cce72e7f81aa3e2dd4b757709dc76d/figures/issue_1689/fig2_ceilings_per_cell.png)

> **Figure.** *Most non-user cells have near-zero ceilings; user arms are degenerate self-predictions.* Within-cell held-out R² (L19), prefix (blue) and context (orange) arms per condition; top base, bottom instruct. Hatched red = degenerate arm (user cells: the context span ends where the answer span ends). Value labels per bar; kept rows per cell range 288-11,400.

User-cell context arms (and plain-text user prefix arms) are self-predictions (the predicted user turn is already part of the conditioning context; the two spans end at the same token) — ceilings 0.92-0.99 on the full 3,800-conversation cells, 0.54-0.87 on the low-yield user-on-policy cells, kNN retrieval near-ceiling (median rank exactly 2 in pools of 288-11,400 rows; acc@5 = 0.98-1.00; acc@1 = 0, driven by a same-conversation duplicate row at rank 1 on all 18 arms). They carry no operator information.

Non-user ceilings are low: base 0.00-0.09, instruct 0.01-0.32 (assistant-chat highest). Character-chat cells keep only 417-624 of 11,400 rows after judge filtering; the user-on-policy chat cells keep as few as 288. Every downstream rung read is conditioned on this ceiling heterogeneity.

### The bimodal rung distribution mostly reflects ceiling validity

Rung-reached at the 0.90 bar for all 126 ordered pairs, both arms, with validity overlays: gray d = target ceiling at or below zero (the bar is vacuous, rung 1 trivial), light x = construct-invalid arm, dark red = no rung reconciles.

![Validity-annotated rung-reached heatmap for the instruct model both arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/089b46a530cce72e7f81aa3e2dd4b757709dc76d/figures/issue_1689/fig1_rung_heatmap_instruct.png)

> **Figure.** *Rung-1 and rung-9 modes track validity flags, not operator similarity.* Instruct model, L19: weakest reconciling rung (viridis 1-9) per ordered pair, prefix (left) and context (right). Base-model version: [base heatmap](https://raw.githubusercontent.com/superkaiba/explore-persona-space/089b46a530cce72e7f81aa3e2dd4b757709dc76d/figures/issue_1689/fig1_rung_heatmap_base.png) (deliberate link — the two figures are the same read on the sibling model).

The raw rung distribution is bimodal (base context: 72 of 126 pairs at rung 1, 41 at rung 9), but 54 context-arm pairs are construct-invalid and 31-52 per model-arm are ceiling-degenerate; zero base-context rung-1 pairs survive the validity screen, and most rung-9 codes mean no rung reconciles rather than reconciliation at rung 9 (36 of 41, base context; the two outcomes share code 9, separated here by recovery).

Across bars 0.85/0.90/0.95, few pairs change rung: 1 of 28 informative pairs (base context), 2 of 50 (base prefix), 3 of 41 (instruct context), and 10 of 65 (instruct prefix).

Arm parity: 76% of the 126 pairs (base) and 82% (instruct) agree within 1 rung across prefix/context. Base-instruct parity: 83-86% within 1 rung; both figures are inflated by shared degeneracies.

### Framing changes need much deeper correction than identity changes

Rung-reached distribution by pair class, informative pairs only (construct-invalid arms and degenerate-ceiling targets excluded).

![Stacked rung distribution by pair class for informative pairs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/089b46a530cce72e7f81aa3e2dd4b757709dc76d/figures/issue_1689/fig3_class_rung_stack.png)

> **Figure.** *Identity pairs reconcile shallowly; framing pairs mostly reconcile at no rung.* Informative ordered pairs per class, colored by weakest reconciling rung (viridis bins; dark red = none); panels are model × arm; counts atop bars.

Within-framing identity pairs reconcile shallowly: all 7 informative instruct chat-framing pairs sit at rung 1. On the 6 with target ceiling of at least 0.05, direct transfer recovers 1.2-2.9 times the target's own ceiling (likely because the assistant map is fit on 20 times more rows); the seventh, Wren to Dana, recovers 12.0 times a near-zero 0.005 prefix ceiling. Bootstrap rung ranges stay at 1 on two of the four context pairs with draws and reach rung 3 on the other two; per-draw recovery for the Dana target (context arm) spans 0.03 to 1.32 — small-cell resamples swing widely even where the point estimate stays at rung 1.

Within-identity framing pairs reconcile at no rung in 25 of 30 informative cases. Both pre-stated directional hypotheses fail: framing was predicted shallow and identity deep; the data show the reverse. Identity claims carry the selection caveat above (chat-character cells keep 4-6% of generations).

### Assistant framing pairs reconcile only at full reparameterization, and usually not at all

Per-rung recovery curves for the 6 ordered assistant framing pairs (context arm), the per-pair view behind the framing half of the class summary.

![Per-rung recovery curves for assistant framing pairs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/951942f545787d6101e2337879b696f8e0294784/figures/issue_1689/fig7_assistant_framing_ladder.png)

> **Figure.** *Base framing transfers reach the bar at no rung; instruct only at rung 9, on 3 of 6 pairs.* Recovery fraction per rung, assistant framing pairs, context arm; base (left), instruct (right); dashed = the 0.90 bar; shared conversations per pair 1,519-2,317.

Story-to-chat transfers onto the highest-ceiling framing targets recover only 0.28-0.45 of ceiling at the strongest correction (95% bootstrap range 0.26-0.47 base, 0.26-0.36 instruct); chat-to-plain reaches 0.64-0.91 (range 0.38-0.92 across the four draw-bearing directions).

Across both arms, 12 of 12 base and 7 of 12 instruct pair-arms reconcile at no rung. The 5 instruct reconciliations all need rung 9: four land on story targets with ceilings of 0.013-0.019, where recoveries above 1.0 reflect the near-zero ceiling rather than good transfer, and one is marginal — chat-to-plain, context arm, 0.0693 against a 0.0687 bar. On the continuous reads (per-rung recovery fractions and their bootstrap ranges), framing remains the axis needing the deepest correction.

### Crossed transfers add little beyond framing's cost

Rung-9 recovery fraction for every transfer into the assistant-chat cell (instruct), grouped by whether the source differs in identity only, framing only, or both; user-source pairs excluded (their answer construct is the user turn, not an assistant answer).

![Recovery into assistant-chat grouped by which axis changes with labeled source cells](https://raw.githubusercontent.com/superkaiba/explore-persona-space/089b46a530cce72e7f81aa3e2dd4b757709dc76d/figures/issue_1689/fig4_crossed_vs_marginal.png)

> **Figure.** *Crossed transfers land with framing-only transfers, far below identity-only.* Rung-9 recovery of the assistant-chat ceiling (instruct, L19), context (left) and prefix (right) arms; labeled points = source cells; thick ticks = group medians; dashed = full recovery.

Descriptive medians over 3 / 2 / 6 source cells per group per arm (too few cells for group CIs): identity-only 0.86 (context) / 0.88 (prefix); framing-only 0.41 / 0.47; both-changed 0.28 / 0.34. The crossed transfers land nearest the framing-only band, about 0.1 below it and roughly 0.5 below identity-only. At this cell count the medians fit both a framing-dominates and an additive-cost account (additive predicts 0.27 / 0.35); the data rule out identity costing anything comparable to framing.

The ordinal pattern matches a prior row-paired three-cell contrast (aligned-cosine, different corpus: identity-only 0.740, framing-only 0.488, both 0.485; not directly comparable). Small identity-source cells bias against identity transfers, so the ordering is conservative there; the judge-selection caveat applies.

### Corrections localize to the answer side, for framing pairs too

Context-side-only (rung 7) versus answer-side-only (rung 8) recovery for every informative pair with target ceiling of at least 0.05, both arms pooled.

![Scatter of context-side versus answer-side recovery by pair class](https://raw.githubusercontent.com/superkaiba/explore-persona-space/089b46a530cce72e7f81aa3e2dd4b757709dc76d/figures/issue_1689/fig5_side_localization.png)

> **Figure.** *Answer-side correction beats context-side for nearly every pair class.* Rung-7 (x) vs rung-8 (y) recovery fractions, clipped to plus/minus 1.5; base (left), instruct (right); marker = pair class; dashed diagonal = equal recovery.

The pre-stated prior — framing moves the context side, identity moves the answer side — is not supported: most pairs of every class sit above the diagonal, framing pairs included (instruct assistant chat-to-story: context-side recovery −2.7 to −3.3, answer-side 0.50 to 0.62). Only base chat-vs-plain assistant pairs show a mild context-side edge (0.40-0.56 vs 0.29-0.35).

One alternative stays open: the context-side ridge map must be learned from 1,300-2,300 training conversations in 3,584 dimensions, so its failure may be estimation-limited.

### Real and simulated user turns differ by an answer-side transform

Per-rung recovery for the four real-LMSYS vs haiku user-provenance pairs per model, prefix arm — the only construct-valid arm; the prefix text is identical across provenances by design, so all differences are answer-side by construction.

![Per-rung recovery curves for real versus haiku user turn provenance pairs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/089b46a530cce72e7f81aa3e2dd4b757709dc76d/figures/issue_1689/fig6_provenance_ladder.png)

> **Figure.** *Real-vs-simulated user turns reconcile only at rotation or deeper.* Recovery fraction per rung, LMSYS-vs-haiku pairs (chat and story framings, both directions), n=3,800 shared conversations each; base (left), instruct (right); dashed = the 0.90 bar.

These are the highest-ceiling reads in the design (0.41-0.78 on 3,800 conversations). Direct transfer fails (recovery −1.2 to −3.9); the answer-mean offset recovers only 0.20-0.34; rotation recovers 0.73-1.10 and answer-side-only reparameterization 0.59-1.23 (values above 1.0 mean the corrected transfer beats the target's own held-out fit).

Simulated and real user turns are encoded in different answer-side coordinates related by roughly an orthogonal transform. The provenance-shallowness hypothesis (reconciliation at rung 4 or weaker) fails.

Data-realism note: the LMSYS arm is tier-1 real data; the haiku arm is tier-3 LLM-simulated. Haiku-simulated user data does not substitute for real user data on these reads.

### The user-on-policy arm duplicates the haiku arm at or before capture

Forward-direction versus reverse-direction rung R² for every user-provenance pair (prefix arm, all framings and models): a pair whose two cells hold identical data lands exactly on the diagonal in both directions.

![Forward versus reverse rung R2 for provenance pairs showing haiku and on-policy identical](https://raw.githubusercontent.com/superkaiba/explore-persona-space/089b46a530cce72e7f81aa3e2dd4b757709dc76d/figures/issue_1689/fig8_provenance_duplication.png)

> **Figure.** *Haiku and on-policy cells hold identical stored data.* Rung R² forward vs reverse per provenance pair family (54 rung values each, clipped to plus/minus 1.2). Haiku-vs-on-policy points sit exactly on the diagonal; LMSYS families scatter off it.

Haiku-vs-on-policy ladder outputs are identical to machine precision in both directions for all 6 framing-model combinations (base chat: ceiling 0.1250491398379796 and every rung R² equal under both orderings), which is only consistent with the two cells' stored activations being identical on shared conversations. Yet the raw on-policy turns differ from haiku's on 160 of 160 shared conversations (SHA-256 text hashes), and the rendered pre-fill rows differ on 3,800 of 3,800. The on-policy turns were generated, but the data was duplicated at or before capture; the root-cause audit is a follow-up.

Every user-on-policy cell is a haiku-arm duplicate on a judge-selected subset; the three-way provenance question collapses to the two-way real-vs-haiku contrast above. Re-capturing the six cells from the persisted on-policy generations would restore the third arm.

### Operator cosines confirm the ordering in raw coordinates; the aligned read tracks shared-row count

Raw operator cosine (per-cell maps compared in shared coordinates, no alignment) against the data-paired Procrustes-aligned cosine, for 87 valid pair-arms per model (39 construct-invalid user arms excluded); per-cell maps refit on all cell rows; both arms pooled.

![Raw versus Procrustes-aligned operator cosine per valid pair-arm colored by pair class](https://raw.githubusercontent.com/superkaiba/explore-persona-space/61eaa02e203fb6756a92c1fd562ea5a334c31a08/figures/issue_1689/fig9_procrustes_raw_vs_aligned.png)

> **Figure.** *Identity pairs lead unaligned; alignment lifts framing pairs and rises with shared rows.* Raw (x) vs aligned (y) operator cosine per valid pair-arm, base (left) and instruct (right); color and marker = pair class; labels on aligned of at least 0.50; gray band at zero = rotation-null p97.5 (about 0.0005).

On the assistant-chat spoke, every identity-only pair's raw cosine (0.14-0.27) exceeds every framing-changing pair's (0.01-0.16) in every model and arm — a direction-aware confirmation of the rung ordering. Alignment lifts framing pairs (raw 0.04-0.13 to 0.19-0.29, instruct context) and slightly lowers identity pairs (0.26-0.27 to 0.23), so the three-cell aligned ordering quoted above does not reproduce; aligned cosine instead tracks shared-row count (Spearman 0.56-0.76 over 87 pair-arms; identity-chat intersections hold 114-159 rows vs 2,200-2,317 for framing), an estimation confound.

All pair-arms exceed the rotation-null p97.5, a descriptive read: the Haar null prices random rotations, not the data-fitted alignment, and every pair shares one conversation bank. Haiku vs self-generated user pairs read aligned equal to raw at machine precision, as the duplicated rows above force.

### A ridge grid widened three decades moves no ceiling: the published reads are not λ-grid-limited

Change in within-cell held-out R² at layer 19 when the ridge-λ grid extends from `logspace(-2, 4, 13)` to `logspace(-2, 7, 19)`, per cell-arm (84 = 21 conditions × 2 models × 2 arms), against the published ceiling with the 0.02 mover bar; beside it, fold-level λ-selection counts on both grids.

![Ceiling change per cell-arm under the wider lambda grid and the lambda-selection histogram](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6d6f28e7a1e72bc2a745ecc29de364b16f690f8f/figures/issue_1689/fig10_lambda_recheck.png)

> **Figure.** *No cell-arm moves past the 0.02 bar; λ selections spread only two half-decade steps beyond the old edge.* Left: ceiling change vs published ceiling, prefix (blue) and context (orange), circles base / squares instruct; dashed = the 0.02 mover bar. Right: fold-fit λ-selection counts, published 13-point vs wide 19-point grid; dashed = old ceiling, dotted = new.

No cell-arm moves by more than 0.02 (0 of 84, where 5 or more movers would have triggered the conditional ladder re-read; largest +0.019, base-model Dana-story context), so the published reads stand and the affected-pair set is empty. Of the 265 fold-fits pinned at the old 10^4 edge, 202 move up — 100 to 10^4.5 and 102 to 10^5 — and none of the 420 selects above 10^5, three decades below the new edge (zero at 10^7). The pile-up was a λ-selection edge artifact on a flat R² plateau, not a ceiling limitation; the largest changes sit in near-zero-ceiling base-model story cells.

---

**Repro:** No training. Fits: `scripts/issue1689_fit_ladder.py` (torch fp64, numpy-equivalence gated) via `scripts/issue1689_ladder_parallel.sh` on pod-1689 (4×H100, ~254 GPU-h including crash-fix rounds R9-R16 vs 46 budgeted). Analyzer digest + figures: `scripts/issue1689_analyzer_digest.py`, `scripts/issue1689_analyzer_figures.py` at branch commit `089b46a530cce72e7f81aa3e2dd4b757709dc76d` (run commit `052a2c1cb32904b975a763235eec7e9ba6645a82`); fig7 regenerated with a corrected title at `951942f545787d6101e2337879b696f8e0294784` (revision round 2). Eval JSONs: `eval_results/issue_1689/` (`ladder/`, `percell/`, `onpolicy_stats/`, `analyzer/pair_digest.csv`). HF data repo (verified live via `list_repo_tree`, pinned to revision `d1010a25f81ce184f68a9cc0ed49bce9736b80dd`): activation store [analysis_tensors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d1010a25f81ce184f68a9cc0ed49bce9736b80dd/issue1689_speaker_lattice/analysis_tensors) (42 cells × 4 layers), raw completions [raw_completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d1010a25f81ce184f68a9cc0ed49bce9736b80dd/issue1689_speaker_lattice/raw_completions) (kept rows only), rendered rows [rendered](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d1010a25f81ce184f68a9cc0ed49bce9736b80dd/issue1689_speaker_lattice/rendered). Reused artifacts: `issue1310_common.PERSONAS` verbatim (Wren/HELIOS/Dana; Vex excluded per plan §11) — fit: same character definitions as the sibling lattice line; LMSYS filter recipe from the [#825](https://eps.superkaiba.com/tasks/825) Track-M pipeline — fit: same corpus family and construct. The four pod-generated `hero1_rung_heatmap_*` figures at the run commit are superseded by `fig1_rung_heatmap_*` (they plot the raw rung code, which conflates no-reconcile with rung 9, and carry no validity overlays). Scope deviations vs plan: layer 19 only (14/18/26 captured, never fit); a single conversation-grouped 80/20 split per pair rather than per-rung 5-fold CV; bootstrap 1000 → 200 (R15) → 0 (v75 user descope; 76 pairs retain 200-draw CIs — 39 base, 37 instruct; the results-marker count of 79 overcounts by 3); nulls 200 → 40; the within-cell R² ≥ 0.5 pair-inclusion gate named in plan §6.6 item 4 was not enforced (all 126 pairs fit regardless of ceiling — this analysis screens post hoc instead); the plan-named mean-shift-vs-activation-spread read (plan §6.6 item 8) was not computed — it needs the HF activation store and stays a follow-up; the plan §5 secondary generation seeds 137/271 diagnostic-pair-subset stability check never ran (no artifact, no run event) — all generation-dependent cells use generation seed 42 only, so rung-headline stability under generation randomness is untested; the per-pair Procrustes operator cosine (addendum item 1b) landed in the 2026-07-29 zero-GPU follow-up round: `scripts/issue1689_procrustes_battery.py` at commit `aeee660155`, outputs at `9319318909` (`eval_results/issue_1689/procrustes/`), figure at `61eaa02e20`. The wider-λ ceiling re-check (second follow-up round, 2026-07-29) ran `scripts/issue1689_lambda_recheck.py` on the shared VM (CPU, 0 GPU-h), outputs at commit `3c5053faf3` (`eval_results/issue_1689/wider-lambda-ceilings/`); figure by `scripts/issue1689_lambda_recheck_figure.py` at `284912d109`, committed to main at `6d6f28e7a1`. Per-pair CI claims are restricted to the draw-bearing subset; the 0.85/0.90/0.95 bar sweep uses point estimates for all pairs. Identity space is 3 characters + assistant + user (one character per category), an external-validity limit on every identity claim.

**Context:** Task #1689, a follow-up to [#825](https://eps.superkaiba.com/tasks/825) (the Track-M LMSYS context-to-answer map line whose corpus filter this task reuses), created 2026-07-25 from the verbatim user prompt: "i want a controlled comparison between: assistant with chat template / assistant without chat template / assistant in story / user in chat template / user without chat template / user in story -- for user it should be either: real user generated turns from LMSYS, always haiku generated, always on-policy generated by Qwen itself / other AI character with chat template / other AI character without chat template / other AI character in story / other assistant-like human character with chat template / other assistant-like human character without chat template / other assistant-like human character in story / other random character with chat template / other random character without chat template / other random character in story. (all except user should always be on policy generated) all in both instruct and base model. all characterized according to the tiers of mapping similarity. [then] run the full thing in the background with happy coder". Run 2026-07-26 through 2026-07-28 (crash-fix rounds R9-R16; torch/GPU fit port in the user session). The framing-by-identity interaction addendum (the crossed-vs-marginal and side-localization results) was user-approved scope posted on the task 2026-07-28, before analysis. A zero-GPU follow-up round (2026-07-29) folded in the addendum's per-pair Procrustes operator-cosine battery (orchestrator-run; no separate user prompt). A second zero-GPU follow-up round (2026-07-29, label `wider-lambda-ceilings`, proposer-initiated cheap band, no separate user prompt) re-checked the ridge-λ grid ceilings; scope verbatim: "Wider ridge-lambda ceiling re-check — extend the ridge grid from logspace(-2,4,13) to logspace(-2,7,19) and refit the 84 per-cell L19 within-cell maps (both models x 21 conditions x 2 arms); recompute ceilings + the reach-bar-dependent rung reads for the informative pairs."

