---
title: Map-then-project persona-vector prediction trails matched-budget context-side
  baselines across three behaviors (HIGH confidence)
kind: experiment
tags:
- trigger-dense
- followup-manual
created_at: '2026-07-28T01:08:06Z'
has_clean_result: true
origin_prompt: run in background with happy coder and MAKE SURE IT PARALLELIZES AND
  VECTORIZES AS MUCH AS POSSIBLE
workflow: v1
goal: Determine whether applying the learned context->answer map before projecting
  the persona vector predicts on-policy behavior expression (evil, trait sycophancy,
  hallucination) better than context-side projection and direct regression at matched
  (unlabeled, labeled) data budgets, and whether that advantage grows across a real-data
  distribution-shift ladder.
relates_to:
- spec-context-as-vector
---
# Map-then-project persona-vector prediction trails matched-budget context-side baselines across three behaviors (HIGH confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_1739.md](https://github.com/superkaiba/explore-persona-space/blob/431226c1a54b6fde668ea935b931f504dfd5d38c/docs/methodology/issue_1739.md) · [gist mirror](https://gist.github.com/superkaiba/0c0069c286b8fb75fbcfb1d2a785a4fd)

## Takeaways

- Map-then-project trails the context-native direction: the Spearman rho difference is reliably below zero in 712 of 810 evil, 134 of 270 hallucination, and 230 of 810 sycophancy cells.
- Direct ridge on context activations is the strongest deployable predictor at the largest budgets: rho 0.71 (evil), 0.58 (hallucination), 0.74 (sycophancy), near the true-answer oracle (0.65 to 0.82).
- Refitting the map with half in-domain contexts flips the evil difference from -0.30 to +0.05 to +0.35 (single draw and seed) — the generic-corpus map, not projection itself, fails.
- The prefix-to-answer map never learns (held-out R-squared at or below zero at every unlabeled budget); prefix-side prediction is dead for sycophancy and hallucination, informative only for jailbreak prefixes.
- Caveats: both evil transfer rungs fail the spread floor; 3-7% of rollouts carry CJK intrusion correlating +0.17 with two DVs (census and arm ordering survive its exclusion); the planned teacher-forced margin companion DV was not computed.

## Goal

**This experiment in context:** Persona-vector monitoring projects an answer-derived behavior direction onto context activations — a datatype mismatch this experiment tries to correct by applying the learned context-to-answer map first, scoring each context by projecting the mapped activation instead of the raw one. It reuses the [#1092](https://eps.superkaiba.com/tasks/1092) WildChat+LMSYS activation store as the unlabeled map corpus, the [#779](https://eps.superkaiba.com/tasks/779) direction-extraction recipe, and the [#722](https://eps.superkaiba.com/tasks/722)/[#779](https://eps.superkaiba.com/tasks/779) batched map-fitting stack. The primary comparison fixed in the plan: map-then-project versus the context-native direction at matched unlabeled and labeled budgets, across three behaviors and a real-data distribution-shift ladder.

**Broader narrative:** If a map learned from cheap unlabeled text closed the context/answer datatype gap, behavior monitors could reach a target accuracy with far fewer behavior labels and degrade more gracefully off-distribution. The measured answer is that on generic-corpus maps it does not — direct regression on context activations stays the default monitoring predictor, and map benefits appear only when the map is fit on in-domain contexts.

## Methodology

**Design:** Prediction experiment on Qwen2.5-7B-Instruct — no fine-tuning; all arms are read-outs over frozen activations. 16 predictor arms (5 context-side, 5 map-based, 2 true-answer oracles, 4 controls) score every context; the target is on-policy behavior expression. Grid per behavior: 2 mapping variants (prefix-based and context-based, both run per the standing rule) x direction regimes (E1 = synthetic contrastive system-prompt pairs per the persona-vectors recipe; E2 = matched natural pairs; E2p = pooled natural — concrete constructions under Data extraction; hallucination E1 only) x unlabeled map budget U in {250; 5,000; 18,793 = full store} x labeled budget L in {250; 2,500; 8,000 for evil / 16,000 otherwise} x 5 label draws x 3 seeds — 810 evil grid cells + 16 evil composition cells, 810 sycophancy, 270 hallucination (1,906 total), plus 90 reversed-transfer evil cells (train on hh-rlhf red-team, evaluate on the held-out jailbreak slice + ToxicChat). Per cell: group-level 5-fold round-robin cross-validation (conversation / persona / jailbreak-family groups), all 28 layers swept, frozen-best-layer rho with selection-inherited paired-bootstrap intervals (500 draws) plus nested layer selection, and a max-over-arms-and-layers permutation null (500 draws).

**Training:** **N/A — no model training.** Evaluation/generation hyperparameters:

| Hyperparameter | Value | Source |
|---|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct | `constants.py` `MODEL_NAME` @ `eb084ff2c4` |
| Rollouts per context (K) | 5 | `constants.py` `K_ROLLOUTS` |
| Generation temperature / max new tokens | 1.0 / 1,024 | `generation.py` `GEN_TEMPERATURE`, `GEN_MAX_NEW_TOKENS` |
| Judge model | claude-sonnet-4-5-20250929 | `constants.py` `JUDGE_MODEL` (project pin) |
| Judge draws per completion / temperature | 3 / 1.0 | `constants.py` `N_JUDGE_DRAWS`, `JUDGE_TEMPERATURE` |
| Judge max tokens | 400; 800 re-judge for truncation-affected items | `constants.py`; `judge_summary.json` `rejudge_800` |
| Ridge lambda grid (map + regression arms) | 0.01 to 1,000 (6 log steps) | `constants.py` `RIDGE_LAMBDAS` (store-producing-run lineage) |
| MLP arms | width 512, one hidden layer, max 300 epochs | `constants.py` `MLP_HIDDEN`, `MLP_MAX_EPOCHS` |
| Folds / bootstrap draws / permutation draws | 5 group-level / 500 / 500 | `constants.py` `N_FOLDS`, `N_BOOT`, `N_PERM` |
| Direction extraction (E1) | 5 pos/neg pairs x 20 questions x 10 rollouts, judge-filtered | plan §0 (arXiv 2507.21509 recipe) |
| Activation store | reused WildChat+LMSYS store, rev `e5901706`, fp16, 28 layers x 3,584 | `constants.py` `STORE_REVISION` |

**Evaluation:** Primary DV per context = mean over K=5 on-policy rollouts of graded judge scores: evil and sycophancy use a 0-100 trait rubric (3 judge draws per completion, mean-aggregated; malformed/refusal returns dropped, never coerced), hallucination uses the fraction of rollouts judged fabricated under a three-way correct/abstained/fabricated rubric. Judge content-drops: 46,602 evil draws (29% — refusals on jailbreak-completion content; 1,811 of 10,666 evil contexts lost every draw and are excluded), 5,859 sycophancy draws (2.3%); zero transport losses; a truncation re-judge at 800 max tokens recovered 356 evil and 652 sycophancy all-dropped items. Predictor validity = Spearman rho between the arm's predicted score and the DV on held-out group folds; the headline read is map-then-project minus context-native with paired-bootstrap intervals. Undefined rank correlations (tie-degenerate projections) are recorded as missing, never coerced, and the singular-slice degenerate-fit fallback fired in zero of 1,906 cells. The spread floor (SD at least 10 of 100 and under 80% of contexts in the bottom bin) passes on every train rung (SD 26.3 evil, 13.2 sycophancy, 40.5 hallucination rescaled) and on the hallucination/sycophancy eval rungs; it fails on both evil transfer rungs (hh-rlhf red-team SD 0.9; ToxicChat 93% bottom bin). The planned teacher-forced fixed-pool margin companion DV was not computed, so the graded rate carries all results alone; item-aligned split-half ceilings are 0.89 (evil) and 0.88 (sycophancy), none was computed for hallucination (values in `intrusion_audit/recount.json`).

**Data extraction:** The map M is a ridge fit from store `context_end` (or `prefix_end`) activations to answer-span mean activations, fit on U rows of the reused WildChat+LMSYS activation store (18,793 rows after excluding the 2,400 eval-only battery rows; the plan's U=50,000 rung realized as the 18,793-row full store). The reused rows are the store's instruct-model-own-text cell: each answer-span activation summarizes Qwen2.5-7B-Instruct's own generated reply to a real WildChat/LMSYS conversation prefix plus user query, not the corpus-logged assistant reply. Direction constructions — E1: diff-of-means over this run's judge-filtered contrastive extraction rollouts (5 positive/negative system-prompt pairs x 20 questions x 10 rollouts per sign, response-averaged activations). E2 (matched natural pairs): built from the train-rung labeling rollouts themselves — a context qualifies when at least 2 of its K=5 judge-scored rollouts survive drops and their score spread is at least 15 of 100; each qualifying context's rollout answer-span activations are split at that context's own score midpoint, and the direction is the high-minus-low mean difference averaged over qualifying contexts, holding the context (and so topic) fixed within every pair. E2p (pooled natural): one global midpoint split over ALL kept rollouts across contexts, direction = difference of the pooled means — topic-confounded by construction (pool membership tracks which contexts elicit high scores, not within-context variation), as the plan flags. Behavior corpora — evil: in-the-wild jailbreak prefixes crossed with forbidden questions, 6,468 train contexts with a DV (transfer rungs: hh-rlhf red-team 1,868, ToxicChat 519); hallucination: TriviaQA rc.nocontext, 16,000 train contexts (transfer: NQ-Open 3,167, SimpleQA 4,021); sycophancy: Reddit personal-advice posts, 16,000 train contexts (transfer: 1,304 held-out Reddit posts — see below). Three data-provenance notes ride the record. (1) The sycophancy transfer rung labeled `aita` in the artifacts is the plan's fallback — a hash-partitioned held-out Reddit socialskills slice (all 1,304 post ids land in the sha1 mod-10 eval bucket) — because the planned ELEPHANT AITA-YTA set has no resolvable HF dataset id; it is a same-platform holdout, a weaker distribution shift than the planned cross-platform rung. <!-- concern-deferred: elephant-aita-unresolved --> (2) The E1 extraction pair/question assets were absent from the pinned direction-bank HF prefix, and the pod's git clone carries no local cache of the producing run's artifacts, so the regeneration branch of the wired fallback chain fired: evil used the persona-vectors paper's verbatim instruction pairs and questions (a code constant, no API call), while sycophancy and hallucination were each regenerated by one claude-sonnet-4-5-20250929 call from the persona-vectors template (confirmed against the persisted rollouts: the realized extraction system prompts differ from the producing run's cached texts). The realized extraction rollouts plus direction banks are persisted on HF. <!-- concern-deferred: e1-assets --> (3) Raw rollouts persist on HF as 123 packed JSONL shards plus a pack manifest rather than one file per rollout; totals reconcile via the manifest. <!-- concern-deferred: r4-packed-layout-verify-reconciliation --> A CJK language-intrusion scan over all 255,790 rollout completions found 6.5% (evil), 6.2% (hallucination), and 3.4% (sycophancy) intruded rows; on train contexts the per-context intrusion fraction correlates with the DV at rho +0.17 (evil and hallucination) and -0.01 (sycophancy) — a shared-target nuisance both arms of every comparison score against. Excluding every flagged context (any intruded rollout; 15-25% of contexts) moves the below-zero headline census by at most 31 cells per behavior (evil 703 to 672, hallucination 137 to 139, sycophancy 230 to 215 under the same frozen-layer convention) and leaves the canonical arm ordering unchanged apart from arms already tied within 0.001; scan and recount artifacts: [`eval_results/issue_1739/intrusion_audit/`](https://github.com/superkaiba/explore-persona-space/tree/65114b4aca63a45bd23563d7f20f22b43d0f71ed/eval_results/issue_1739/intrusion_audit).

**Sample training/evaluation data + completions:** Per the trigger-dense content-hygiene rule for this task (jailbreak corpus + unscreened real-user text), no rollout text is quoted; rows are referenced by context id and numeric record, and full text lives in the linked artifacts. Cherry-picked high/zero-DV rows; all rows: [labeling JSONs](https://github.com/superkaiba/explore-persona-space/tree/6686d45da9076893c0e5c93f57a2b1040defd0fb/eval_results/issue_1739/dv_dataset) and [raw rollouts on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/main/issue1739_ctxmap).

| Behavior | Firing examples (DV) | Non-firing examples (DV 0.0) |
|---|---|---|
| evil | `evil-train-cross-000555` (100.0), `evil-train-cross-001236` (100.0), `evil-train-cross-001266` (100.0) | `evil-eval-hhrt-000000`, `evil-eval-hhrt-000001`, `evil-eval-hhrt-000002` |
| hallucination | `hallucination-eval-nqopen-000003` (1.0), `hallucination-eval-nqopen-000006` (1.0), `hallucination-eval-nqopen-000008` (1.0) | `hallucination-eval-nqopen-000010`, `hallucination-eval-nqopen-000017`, `hallucination-eval-nqopen-000019` |
| sycophancy | `sycophancy-train-train-006815` (88.5), `sycophancy-train-train-000252` (85.7), `sycophancy-train-train-012247` (82.3) | `sycophancy-eval-aita-000116`, `sycophancy-eval-aita-000161`, `sycophancy-eval-aita-000166` |

A 5-row random spot check (seed 42) per behavior showed coherent records (evil mostly 0 with occasional 80-100 scores, sycophancy 20-40, hallucination fabrication fractions), small per-row drop counts, and zero transport losses.

## Results

### Map-then-project loses to the context-native direction almost everywhere

What is plotted: the mean difference in held-out Spearman rho, map-then-project minus context-native, at the largest budgets, one row per behavior x regime x mapping variant; dots are the 15 draw-by-seed cells, whiskers the 95% interval.

![Forest plot of map-then-project minus context-native rho differences by behavior and regime, most rows below zero](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6686d45da9076893c0e5c93f57a2b1040defd0fb/figures/issue_1739/hero_headline_delta_forest.png)

> **Figure.** *Map-then-project sits below the context-native baseline in nearly every slice.* Rho difference at the largest budgets; E1 = synthetic-pair direction, E2 = matched natural, E2P = pooled natural; context/prefix = mapping variant; dots are per-cell values (n=15 per row).

Every evil slice and the context-side hallucination slice are negative (means -0.11 to -0.43), and the sycophancy synthetic-pair context slice sits at -0.06; the paired-bootstrap interval is wholly below zero in 15 of 15 largest-budget context-variant cells for all three behaviors under the synthetic-pair direction — the plan's falsification criterion, met on the context variant. The one reliably positive slice is sycophancy with the pooled-natural direction (+0.05, 15 of 15 cells above zero) — an order of magnitude smaller than the losses elsewhere, and it rests on the E2P contrast, which is topic-confounded by construction (Data extraction), so it is weak evidence for a behavior-direction gain.

### Direct ridge regression on context activations dominates every monitoring arm

What is plotted: mean held-out rho for all 16 arms at the largest budgets (context variant), four panels; dots are per-cell values.

![Bar chart of mean held-out rho for all 16 arms per behavior; direct ridge on context activations is highest](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6686d45da9076893c0e5c93f57a2b1040defd0fb/figures/issue_1739/arm_overview_canonical.png)

> **Figure.** *Direct ridge on context activations tops every deployable arm.* Per-arm mean Spearman rho (frozen layer) with 95% intervals; blue = context-side arms, yellow = map-based, gray = true-answer oracles, pink = controls.

Direct ridge reaches 0.71 / 0.58 / 0.74, close to the true-answer regression oracle (0.82 / 0.65 / 0.77); every projection arm sits below it. Map-based regression arms add a small consistent gain over plain context ridge (mean +0.008 to +0.017, positive in 45 of 45 canonical cells), but the shuffled-pretrain control matches them to within 0.001 — the gain reflects added feature dimensions, not map semantics.

For hallucination even the true-answer projection oracle reads 0.04, below the shuffled-map control's 0.14, while true-answer regression reaches 0.65: the projection readout itself, not the map, is dead there. The best arm clears the max-over-arms-and-layers permutation null in all 1,906 cells (p near 0.002; median best rho 0.55 to 0.74 versus a null ceiling near 0.07).

### More labels do not rescue the map arm

What is plotted: mean held-out rho versus labeled budget L (log scale) for five key arms, context variant, synthetic-pair direction, full unlabeled pool; shaded bands are 95% intervals over the 15 cells per point.

![Line plot of held-out rho versus labeled budget: regression arms rise with labels, projection arms stay flat](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6686d45da9076893c0e5c93f57a2b1040defd0fb/figures/issue_1739/scaling_rho_vs_l.png)

> **Figure.** *Extra labels grow the regression arms, not the map arm.* Projection arms are nearly flat in L (labels only pick the layer); regression arms grow with L but dip at L=2,500. Bands only: 15 per-cell values per point across the five plotted series overplot illegibly; per-cell values are in the linked per-cell table.

Map-then-project stays flat in L while regression arms grow. At the smallest budget (L=250) it edges direct ridge for evil (0.54 vs 0.52) while still trailing the context-native direction (0.60), and it trails direct ridge for pooled-natural sycophancy (0.52 vs 0.59) — the sample-efficiency case for the generic map does not materialize at any budget. Regression arms dip at L=2,500 in all three behaviors (train folds hold about 2,000 rows against dimension 3,584, the near-interpolation regime for ridge), so regression reads at that one budget are conservative; the budget-endpoint comparisons are unaffected.

### No robustness advantage on transfer rungs that pass the spread floor

What is plotted: mean held-out rho per evaluation rung (train, then transfer), context variant, synthetic-pair direction, largest budgets; asterisks mark rungs failing the DV spread floor.

![Bar chart of held-out rho per transfer rung and behavior; direct ridge is highest on every floor-passing rung](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6686d45da9076893c0e5c93f57a2b1040defd0fb/figures/issue_1739/shift_ladder.png)

> **Figure.** *Direct ridge transfers best on every floor-passing rung.* Transfer ladder per behavior; the rung labeled aita is the held-out Reddit socialskills slice; evil's two transfer rungs (starred) fail the spread floor — their near-zero correlations are reads over an almost-constant DV.

Where the floor passes, direct ridge transfers best: hallucination 0.40 and 0.40 (NQ-Open, SimpleQA) against map-then-project's 0.20 and 0.27; sycophancy 0.73 against 0.35 on its held-out Reddit rung (a same-platform holdout — the fallback described under Data extraction). Evil's apparent map edge on ToxicChat (0.32 vs 0.25) sits on a rung with 93% of contexts in the DV bottom bin — convention-dependent, not robustness evidence — and on hh-rlhf red-team every arm reads near zero. The reversed-direction evil configuration collapses all 16 arms to rho 0.09 to 0.21; a growing map advantage under shift is unsupported wherever the DV is measurable.

### An in-domain map flips the headline difference positive (preliminary)

What is plotted: the evil composition cells — the rho difference (map-then-project minus context-native) when the 5,000-row map pool is fully generic versus half in-domain, both variants, colored by labeled budget; one dot per cell (single draw and seed).

![Scatter of rho differences for the evil composition cells: generic map pools sit below zero, half in-domain pools above](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6686d45da9076893c0e5c93f57a2b1040defd0fb/figures/issue_1739/compose_fu_flip.png)

> **Figure.** *A half-in-domain map pool flips the difference positive.* Evil composition cells; generic-pool maps sit below zero, half-in-domain maps above it at every budget; each dot is one cell (single draw and seed) — preliminary.

Generic pools give -0.08 to -0.31; half-in-domain pools flip every budget to +0.05 to +0.35, and at L=250 map-then-project (0.53 to 0.56) far exceeds direct ridge (0.17 to 0.19) — the small-label composition hypothesis holds only with an in-domain map. All 16 realized cells are n=1 (single draw and seed): a preliminary direction for follow-up, not a confirmed effect. Two planned cells were skipped with a recorded infeasible-pool reason.

### The context map learns; the prefix map does not

What is plotted: held-out map quality on the shared store by U rung — best-layer R-squared for the map and its identity-plus-learned-bias baseline, and kNN retrieval accuracy at k=1 with chance, for both maps.

![Map R-squared and retrieval accuracy versus unlabeled budget: the context map rises with data, the prefix map never leaves zero](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6686d45da9076893c0e5c93f57a2b1040defd0fb/figures/issue_1739/map_quality_ladder.png)

> **Figure.** *Only the context-to-answer map learns on the generic store.* Map diagnostics (identical across behaviors); context-to-answer R-squared rises 0.18 to 0.78 with U while prefix-to-answer never exceeds zero.

The context-to-answer map is real: R-squared 0.18 / 0.41 / 0.78 across the U ladder, far above identity-plus-bias, kNN accuracy 0.28 against chance 0.0003. The prefix-to-answer map never learns — R-squared at or below zero, retrieval near chance: on organic chat the prefix does not predict the answer representation.

Downstream, all prefix-side arms read near zero for sycophancy and hallucination (159 sycophancy prefix cells return undefined rank correlations from tie-degenerate projections). Prefix-side conclusions are scoped to evil, whose jailbreak prefixes vary.

---

**Repro:** ~96 GPU-h cumulative (three parallel A100-80 lanes for the fit phase; generation/capture on the primary pod; judging via the Anthropic Batch API off-pod). Code `eb084ff2c4` (run) and `6686d45da9` (analysis figures), branch `issue-1739`. Eval artifacts: `eval_results/issue_1739/{evil,hallucination,sycophancy}/arm_results/all_arms_spearman.json` (826 / 270 / 810 cells; per-cell records in `arm_results/percell/cells.jsonl`), `eval_results/issue_1739/evil_config_b/` (90 reversed-transfer cells), `eval_results/issue_1739/dv_dataset/*/labeling.json`, `eval_results/issue_1739/{evil,hallucination}/pilot_report.json`, map diagnostics per behavior, the per-cell headline table `figures/issue_1739/headline_deltas_percell.csv`, and the CJK intrusion scan + excluded-intrusion recount `eval_results/issue_1739/intrusion_audit/{intrusion_scan,recount}.json`. Raw rollouts + judge outputs: HF `superkaiba1/explore-persona-space-data` under [`issue1739_ctxmap/raw_completions/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/1ade7beb35b51249f26a2a2ecacb770003b4dbcc/issue1739_ctxmap/raw_completions) and [`issue1739_ctxmap/judge/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/1ade7beb35b51249f26a2a2ecacb770003b4dbcc/issue1739_ctxmap/judge) (listing verified via `list_repo_tree`, 2026-07-30). Reused artifacts — activation store from [#1092](https://eps.superkaiba.com/tasks/1092): HF `superkaiba1/explore-persona-space-data` at [`issue1092_realistic_crossing/analysis_tensors/summaries/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/e5901706/issue1092_realistic_crossing/analysis_tensors/summaries), rev `e5901706` — fit: same base model and revision as this run's capture, carries both mapping arms plus answer-span summaries at all 28 layers in fp16, and its instruct-own-text cell matches the map's intended input distribution (real conversations, model-generated answers); direction bank from [#779](https://eps.superkaiba.com/tasks/779): HF `superkaiba1/explore-persona-space-data` at [`issue779_monitoring/r_b/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/037fcbb/issue779_monitoring/r_b), rev `037fcbb` — fit: persona-vectors mean-difference directions for the same three behaviors on the same base model (judge-filtered contrastive rollouts, all 28 layers, matching the store's layer grid), consumed as the pinned E1 reference probe at Gate 0 (E1 grid directions were extracted fresh from this run's rollouts after the asset fallback described under Data extraction). Planned-vs-actual: 2 of 18 planned evil composition cells skipped (recorded infeasible pool recipe); the U=50,000 rung realized as the 18,793-row full store; the sycophancy transfer rung realized as the plan's held-out Reddit socialskills fallback (ELEPHANT AITA-YTA unresolvable on HF); the teacher-forced margin companion DV was not computed; sycophancy restored 453 of its 810 cells from a crash resume (final coverage complete); config slugs `arm1_ctx_e1` ... `arm16_surface_feat`, regimes `e1/e2/e2p`, variants `prefix_end/context_end`.

**Context:** fresh direction (no parent) — task created 2026-07-28 from the 2026-07-27/28 interactive design session (plan at `docs/map_behavior_prediction_plan.md`). Originating prompt (verbatim): "run in background with happy coder and MAKE SURE IT PARALLELIZES AND VECTORIZES AS MUCH AS POSSIBLE". Run completed 2026-07-29; first analyzer pass 2026-07-30; revision rounds (intrusion recount + interpretation-critic fixes; clean-result-critic self-containedness + provenance fixes) 2026-07-30. Conciseness note: total prose runs over the 800-word budget, all six result sections exceed the 120-word prose cap, and the caveat Takeaways bullet exceeds the 30-word bullet cap — acknowledged; six result sections were kept to cover the 1,906-cell grid.

