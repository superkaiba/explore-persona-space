# Interim results — Behavior prediction through the context→answer map (#1739)

> **Status (2026-07-30): INTERIM.** Evil and hallucination lanes are complete; the sycophancy
> lane is still running. Figures and numbers below are computed directly from the committed
> per-cell results (`eval_results/issue_1739/`), before the analyzer/critic pass — treat them as
> provisional. Task: <https://eps.superkaiba.com/tasks/1739>. Full pre-registered plan:
> [`docs/map_behavior_prediction_plan.md`](map_behavior_prediction_plan.md).

## Motivation

- Persona vectors predicts behavior from the context vector by projecting the persona vector on the context vector and shows that it can pretty reliably predict whether the answer will exhibit a certain behavior
- This is in some sense a datatype mismatch: persona vectors are extracted from means over answer activations, then compared to the context vector
- Something that makes sense instead is applying our mapping to the context vector and then projecting the persona vector onto the predicted answer vector
    - this allows us to take advantage of unsupervised context -> answer pairs (persona vectors extraction only uses supervised data)
- Probably something that makes even more sense is just training a direct regression from context behavior to behavioral expression, but this experiment is more to **show that our mapping learns useful/meaningful structure** (beyond what is already directly predictable from context) more than to demonstrate an actual application

## Methodology

- Inspired by persona vectors, we look at 3 realistic/important behaviors:
    - sycophancy
    - hallucination
    - evil
- We extract persona vectors using the methodology from the paper, that is taking mean mass probes over answers either exhibiting or not exhibiting the behavior
- We train all regressions on contexts (and answers when appropriate, or sampled answers when appropriate) from WILDCHAT
- We run the following methods:
    - Direct from context methods:
        - Direct linear mapping from context vector to behavior expression
        - Direct nonlinear mapping from context vector to behavior expression
        - Project persona directly on context vector
    - Linear mapping methods:
        - Linear mapping -> project persona vector
        - Linear mapping -> apply regression trained **on predicted answer vectors**
        - Linear mapping -> apply regression trained **on real answer vectors**
    - Upperbounds:
        - Project persona vector on **actual answer vector**
        - Train regression directly from **actual answer vector** (upper bound)
    - For all methods we run scaling curves on both the unlabeled map-training pairs and the labeled behavior examples
    - We run both prefix (prefix end state) -> answer versions and single context -> answer versions of all experiments
- For evaluation, we show results per behavior on a held-out split of the training distribution plus two real out-of-distribution sets
- We also measure spread of behavior expression in the evaluation settings to make sure there is sufficient spread
- We report: spread per evaluation setting; Spearman ρ of predicted vs LLM-judged expression, one bar per method, per behavior per setting; scatter plots per method; scaling curves

### What actually ran (deltas from the original sketch)

- **Behaviors so far:** evil + hallucination complete; **sycophancy still running** (results to be appended).
- **Unlabeled map pool:** 18,793 real WildChat context→answer pairs (the #1092 activation store), swept 250 → 5,000 → 18,793 ("full") — not 50k; the map never sees behavior-eliciting data or any eval set.
- **Labeled behavior data is real, per behavior** (not WildChat): evil trains on in-the-wild jailbreak prompts (1,405 DAN-style prefixes × 390 forbidden questions) and evaluates OOD on hh-rlhf red-team dialogues and ToxicChat jailbreaks; hallucination trains on TriviaQA and evaluates OOD on NQ-Open and SimpleQA. The synthetic persona-vectors elicitation suite was deliberately **dropped** as an eval setting (it conflates natural elicitation with artificial prompting — exactly the concern in the Motivation); the three settings per behavior are held-out-train + two real OOD sets.
- **Nonlinear-map arms were not run as separate arms**; nonlinearity enters via the direct MLP arm and a map-pretrain→fine-tune arm. Extra arms beyond the sketch: identity+bias projection, a stacked combiner, shuffled-map / shuffled-pretrain controls, and text-embedding / surface-feature baselines (16 arms total).
- **DV:** on-policy expression per context from K=5 sampled answers, graded by `claude-sonnet-4-5-20250929` (3 draws @ temp 1.0). Evil: mean 0–100 trait score (persona-vectors rubric). Hallucination: fabrication rate 0–1 under a three-way fabricated/abstained/correct rubric.
- **Persona-vector extraction regimes (evil only):** E1 paper-faithful synthetic; E2 matched-pair natural; E2p pooled natural.
- Spearman ρ is computed at each arm's frozen selected layer; error bars are SD over 3 seeds × 5 label draws. Selection-corrected max-over-arms permutation nulls are significant (p < 0.05) in 100% of cells for both behaviors, and the evil split-half reliability ceiling averages ≈ 0.89.

## Results

*(evil + hallucination; sycophancy pending)*

### Spread of behavior expression per evaluation setting

What is plotted: per-context judged expression (evil: mean 0–100 score over 5 on-policy rollouts × 3 judge draws; hallucination: fabrication rate over 5 rollouts), one histogram per evaluation setting.

![spread evil](../figures/issue_1739/interim_writeup/spread_evil.png)

![spread hallucination](../figures/issue_1739/interim_writeup/spread_hallucination.png)

> All hallucination settings and the evil train/ToxicChat settings have usable spread (SD 0.33–0.42 on 0–1; SD 26.3 and 12.1 on 0–100). **The evil hh-rlhf rung fails the pre-registered spread floor** (mean 0.08, SD 0.89 on 0–100 — Qwen-2.5-7B refuses essentially all 2022-era red-team attempts), so hh-rlhf comparisons below are floor-censored and uninformative. Evil scores are additionally refusal-censored: 1,532 train / 127 hh-rlhf / 152 ToxicChat contexts have no DV because all rollouts refused.

### Main comparison: ρ per method, per behavior, per evaluation setting

What is plotted: Spearman ρ between each method's prediction and judged expression, at the largest budgets (U = 18,793 unlabeled pairs; L = 8,000 evil / 16,000 hallucination labels; E1 persona vector; full-context end state). Only 6 arms were evaluated on the OOD sets (label-free projections + direct ridge + oracle projection + shuffled-map control). All arms in a panel are scored against the same judged targets.

![bars evil](../figures/issue_1739/interim_writeup/bars_evil.png)

![bars hallucination](../figures/issue_1739/interim_writeup/bars_hallucination.png)

> **In-distribution, map→PV projection does not beat context-side methods.** Evil: map→PV 0.53 vs context-native direction 0.66, direct ridge 0.71, oracle regression 0.82. Hallucination: map→PV 0.15 vs context-native 0.52, direct ridge 0.58, oracle regression 0.65. The pre-registered headline contrast (map→PV minus context-native projection) is negative — evil: median Δρ = −0.30 across all 826 cells, bootstrap CI below 0 in 708/826; hallucination: median Δρ = −0.07 (137/270 below, 18 above).
>
> **The hallucination persona vector, not the map, is the broken link:** even projected on the TRUE answer state (oracle), the hallucination PV reaches only ρ = 0.04 on held-out TriviaQA — while a regression on the same true answer states reaches 0.65. The synthetic hallucination direction barely tracks natural fabrication. For evil the PV direction is genuine (oracle projection 0.71).
>
> **Map→ridge-on-predicted-answers matches or slightly beats direct ridge** (evil 0.714 vs 0.706; hallucination 0.602 vs 0.584) — but the shuffled-pretrain control reaches the same value (0.714 / 0.601), so at large L the labeled readout, not the map, is doing the work.
>
> **Under real distribution shift the label-free map arm is the most robust feasible method on 2 of 3 usable OOD settings.** ToxicChat (evil): map→PV 0.32 vs direct ridge 0.25, PV-on-context 0.14, shuffled-map control 0.11 — the map arm matches the oracle projection (0.30). SimpleQA (hallucination): map→PV 0.27 vs direct ridge 0.10. NQ-Open is the exception: direct ridge wins at L=16,000 (0.40 vs 0.20). Where predictions have to transfer, mapping into answer space before projecting helps; in-distribution it costs.

### Scaling: unlabeled map pairs (U) and labeled examples (L)

What is plotted: left — map-based arms vs U at L=max on the held-out train setting (context-native projection shown as a U-free reference line); middle — label-budget scaling at U=full, same setting; right — label-budget scaling on the OOD setting (6-arm transfer roster; note the hh-rlhf panel for evil is floor-censored per the spread section).

![scaling evil](../figures/issue_1739/interim_writeup/scaling_evil.png)

![scaling hallucination](../figures/issue_1739/interim_writeup/scaling_hallucination.png)

> **The map does learn transferable structure from unlabeled pairs:** map→PV improves with U (evil train 0.34 → 0.53 from 250 → 18,793 pairs; hallucination SimpleQA 0.06 → 0.27) while the shuffled-map control stays flat (≈0.26 / ≈0), and map→ridge-on-real-answers gains +0.16–0.19 from U. But **no in-distribution sample-efficiency advantage materialized:** at L=250 the map arms do not beat direct ridge (evil 0.47 vs 0.52; hallucination 0.41 vs 0.45), and map-pretrain→fine-tune is indistinguishable from the shuffled-pretrain control at every L (e.g. evil L=250: 0.472 vs 0.467). The OOD panels show the map arm's flat-but-highest curves on ToxicChat at all L (0.27–0.32 vs ridge ≤ 0.25).
>
> Unexplained interim pattern flagged for the analyzer pass: several labeled arms (direct ridge, stacked, oracle regression) are non-monotone in L with a dip at L=2,500 in both behaviors (e.g. evil ridge 0.52 → 0.34 → 0.71) — a mid-budget fitting artifact (regularization/layer-selection instability) is suspected; low-L numbers above should be read with this in mind.

### Prefix-end vs full-context-end input state

What is plotted: paired bars per method, full-context end state (solid) vs pre-query prefix end state (hatched), held-out train setting, largest budgets.

![variant evil](../figures/issue_1739/interim_writeup/variant_evil.png)

![variant hallucination](../figures/issue_1739/interim_writeup/variant_hallucination.png)

> Evil retains most signal from the prefix state (direct ridge 0.56 vs 0.71; the DAN persona prefix carries the elicitation). Hallucination collapses to ρ ≈ 0.05 for every context-dependent arm — expected, since bare trivia questions have essentially no pre-query prefix; consistent with the #1092/#1774 finding that the pre-query prefix state carries only persona-average signal.

### Predicted vs judged expression (scatter, per method)

What is plotted: OOF predictions vs judged expression for one representative max-budget cell (seed 0, draw 0), held-out train setting; ≤1,500 points per panel; per-panel ρ over all contexts. Per-cell prediction arrays were persisted only for the train setting, so OOD scatters are not available in this interim cut.

![scatter evil](../figures/issue_1739/interim_writeup/scatter_evil.png)

![scatter hallucination](../figures/issue_1739/interim_writeup/scatter_hallucination.png)

> The evil panels show the fan shape typical of a refusal floor (most contexts at score 0 at every prediction level; signal comes from the upper envelope). The hallucination DV is discrete (rate over 5 rollouts → bands at 0, 0.2, …, 1.0); the PV-projection panels (paper method, oracle projection) are visibly unstructured, matching their ρ ≈ 0.04–0.08.

### Persona-vector extraction regime (evil)

What is plotted: ρ for the three PV-dependent arms under E1 (synthetic, paper-faithful), E2 (matched-pair natural), E2p (pooled natural), held-out train setting, largest budgets.

![regimes evil](../figures/issue_1739/interim_writeup/regimes_evil.png)

> Pooled-natural (E2p) beats synthetic (E1) beats matched-pair (E2) for every projection arm (map→PV: 0.55 / 0.53 / 0.24). The topic-controlled matched-pair direction — the one construction that cancels topic — is much weaker, suggesting a substantial share of projection performance rides on topic rather than disposition.

### The persona-vectors synthetic eval inflates projection numbers (added 2026-07-30 evening)

What is plotted: Spearman ρ per method on Persona Vectors' own eval distribution (their 5 positive + 5 negative instruction system prompts × 20 held-out eval questions = 200 contexts/behavior; on-policy K=5 rollouts; our standard judge, PV per-trait rubric) as the red bars, next to the committed real-rung values (op-slice means) — same arms, same frozen-layer convention, transfer-applied readouts.

![pv suite vs real](../figures/issue_1739/interim_writeup/pvsuite_vs_real.png)

> The paper's method gains **+0.26 (evil) to +0.57 (hallucination) ρ** moving from real data to its own suite (evil 0.80 vs 0.54 train / 0.14 ToxicChat; hallucination 0.65 vs 0.08 TriviaQA). On evil, every projection method converges to ~0.80 on the suite — pos/neg-instruction separability saturates and method ranking disappears. Sharpest evidence that the suite measures *instructed-behavior separability* rather than natural elicitation: the hallucination PV direction, near-useless on natural data even with oracle answer states (0.04), reads 0.65–0.74 on the suite; and on sycophancy a **shuffled-map control scores ρ = 0.54** — a nonsense direction separates their prompt structure before any method quality enters. Caveats: the hallucination suite column uses the trait rubric (their questions carry no reference answers; not the fabrication-rate construct; 23.4% judge content-drops, ~half unrecovered at the 800-token re-judge); sycophancy's suite spread is compressed (SD 20, max 85) and its real rungs land when the main lane finishes; prefix-arm rows exist but the suite has only 10 distinct prefix states, so those ρ are rank-tie-dominated. Artifacts: `eval_results/issue_1739/pvsynth/` @ `34c041409d`.

### Unlabeled behavior-eliciting data substitutes for labels (composition factor, evil)

No figure yet (analyzer pass will render one); numbers from the committed compose cells (U = 5,000 total, E1, context-end, held-out train dist.; small cells — ~1 seed × 2 draws): replacing half the generic WildChat map pool with 2,500 *unlabeled* behavior-eliciting contexts (disjoint from the labeled set) lifts map→PV projection from **0.36 to 0.56** — more than the full 18,793-pair all-generic pool achieves (0.53) — while the shuffled-map control moves the other way (0.32→0.27 at L=250; 0.28→0.10 at L=2,500). Map-based labeled readouts gain only at low label budgets (map→ridge-on-predicted 0.29→0.44 at L=250; ≈unchanged at L=2,500). Two flags: arms that consume no unlabeled data also shift between compose cells (suspected per-cell layer re-freezing — the within-arm map-vs-control contrast is the trustworthy read), and this result is in tension with #779's "trait-trained map is worse" (fixed-budget replacement vs full substitution, and real DAN prefixes vs #779's corpora, are the reconciliation candidates).

### A ~1M-context generic map beats the in-experiment map on 2 of 3 behaviors (frozen reuse of #779's maps)

What is plotted (committed by the reuse round): ρ of #779's 963,444-context maps (ridge + MLP w8192/w32768, applied frozen through #779's own code; layers {14, 19, 26}) vs the in-experiment 18.8k-pair map and a shuffled control, recomputed matched-target in one process on the same contexts/DV/direction.

![963k comparison](../figures/issue_1739/interim_writeup/map963k_reuse_comparison.png)

> The 963k map transfers genuinely (reconstructs this experiment's answer states at cosine 0.92–0.99, held-out R² 0.22–0.62 on corpora it never saw) and **wins 9/9 evil cells** (+0.04..+0.17; best-layer ρ 0.60 vs oracle 0.64) and 6/9 hallucination cells (losses only where the rung's own oracle ≈ 0), but **loses all 6 sycophancy cells** — where the in-experiment map's projection on the AITA rung (0.40) even exceeds the oracle projection (0.27). Caveats: the in-experiment-map comparison column applies the *uploaded* map payload (a faithful application, but not a verified reproduction of the committed arm-6 numbers — oracle anchor validated 10/10, arm-6 anchor looser), and that payload extrapolates with strongly negative reconstruction R² onto the behavior eval distributions (a distribution-coverage finding, not a serialization bug — the payload round-trips cleanly on its own distribution). Artifacts: `eval_results/issue_1739/map963k_reuse/` @ `606278aa38`.

### Reversed train/eval direction (evil, secondary config)

Training the labeled readouts on hh-rlhf red-team dialogues and evaluating on DAN×forbidden (90 cells, the pre-registered secondary) collapses every method to ρ ≤ 0.23, including both oracles (oracle regression 0.19, oracle projection 0.23). With the training side floor-censored (the spread failure above), little transfers in this direction; the A−B mechanism-match comparison the plan wanted is compromised by that censoring.

### Interim takeaways (2 of 3 behaviors; pre-analyzer)

1. The datatype-mismatch intuition is half-right: answer-space PVs are indeed a poor fit for context states (context-native directions beat them: 0.66 vs 0.54 evil, 0.52 vs 0.08 hallucination) — but the fix that works in-distribution is extracting a **context-native direction**, not mapping the context into answer space (map→PV trails both).
2. The map demonstrably learns real structure from unlabeled WildChat pairs — performance rises with U while shuffled-map controls stay flat — but that structure is **redundant with the context representation in-distribution**: no map arm beats direct ridge at any label budget, and map-pretraining gives fine-tuning no head start over shuffled pretraining.
3. The clearest value shows up **under distribution shift**: the label-free map→PV arm is the top feasible method on ToxicChat (0.32, ≈ oracle projection) and SimpleQA (0.27), degrading least of all context-side methods; NQ-Open at max labels is the counterexample (ridge 0.40).
4. For hallucination the persona vector itself fails as a construct on natural data (oracle projection ρ = 0.04 in-distribution) — any conclusion about "the map" for this behavior is bottlenecked by the direction, and the direct-regression framing (or a better direction) is required.
5. Readout/deployment consistency matters more than realism: regressions trained on **predicted** answer vectors and applied to predicted vectors (evil 0.71, hallu 0.60) far outperform regressions trained on **real** answer vectors and applied to predicted ones (0.51 / 0.38).
6. **The persona-vectors synthetic suite inflates projection performance by +0.26–0.57 ρ over real distributions** (all three behaviors ≈0.65–0.80 on the suite; a shuffled-map control alone reaches 0.54 on sycophancy) — the suite measures instructed-behavior separability, not natural elicitation.
7. Map *training data* matters more than map size in-distribution but less under scale: 2,500 unlabeled eliciting contexts beat 18,793 generic pairs for the projection read (evil compose cells), while the frozen 963k generic map beats the 18.8k map on evil and mostly on hallucination — and loses on sycophancy.

### Provenance

- Numbers/figures computed 2026-07-30 from `eval_results/issue_1739/{evil,evil_config_b,hallucination}/arm_results/all_arms_spearman.json` and `eval_results/issue_1739/dv_dataset/*/labeling.json` (branch `issue-1739`, commits `1ddf165113` / `ab66d4f04d`), by `scripts/issue1739_interim_writeup_figs.py`; per-cell prediction arrays from `arm_results/percell/preds/`.
- Rollouts are on-policy (temp-sampled, K=5 per context) from Qwen-2.5-7B-Instruct; judge `claude-sonnet-4-5-20250929`, 3 draws @ temp 1.0, `max_tokens` 400 with an 800-token re-judge of truncation-affected items (mixed-instrument caveat carried in the DV metadata). Evil DV coverage is 83% (refusal-censored by design); hallucination 100%.
- Known caveats to carry into the final clean-result: evil's labeled axis has ~1,405 independent jailbreak-prefix groups (group-effective N ≪ row count); TriviaQA/NQ-Open contamination compromises absolute fabrication rates (method deltas are the design's claim surface); the L=2,500 non-monotonicity above is unexplained; hh-rlhf rung floor-censored.
