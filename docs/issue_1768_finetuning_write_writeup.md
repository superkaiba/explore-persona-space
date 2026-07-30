# Result: Fine-tuning a behavior into a context rewrites the context→answer map — in proportion to its weights-carried dose — while the context vector barely moves

*(Chat-drafted writeup of task #1768, 2026-07-30. All numbers from the #1768 clean-result body; artifact record: https://eps.superkaiba.com/tasks/1768)*

## Motivation

* We've found a linear mapping from single context vectors to answer vectors ($v_A \approx M v_C$, test R² ~0.7 at layer 19).
* This experiment tests what happens to that picture when we finetune a behavior into a specific context:
    * Does the context vector move?
    * Does the mapping change?
    * Does the answer vector move?
    * What is the shape of the write — and can it be predicted ahead of time (from the training displacement $\delta$, the behavior read-out $r_B$, or for the marker behavior the marker token's unembedding row)?
    * How does this differ for LoRA vs full finetune, for contrastive vs positive-only training, and for a toy behavior (marker token) vs content behaviors?
    * Are our theory assumptions confirmed or disproved?

## TLDR

- **The context vector barely moves**: median relative movement 0.025 at layer 19 across all 72 arms — the change lives in the map and the emitted text, not the context representation.
- **The map changes, and the split is behavior-shaped**: 107 of 216 (arm, layer) cells change above a refit-noise floor, 102 sit below, 7 unresolved (216 cells = 72 arms × 3 layers). Casual writing style 36/45, impoliteness 45/51, sycophancy 26/60 (depends on training context), **marker 0/60** — the exact reverse of what we predicted for the marker.
- **But it's really a dose story**: map change rank-tracks the weights-carried answer shift (Spearman 0.98 at layer 19), so the marker null is consistent with its smaller dose, not a special marker property.
- **The answer vector moves mostly through the text**: on-policy, ~78% of the answer-state shift is off-map (carried by the different text the trained model writes); hold the text fixed and map change becomes the largest attributable component.
- **The write is high-rank and only weakly predictable ahead of time**:
    - rank-one-write assumption fails at corpus scale (top-1 SVD share median 0.09 on-policy vs our 0.6 criterion)
    - alignment with the training displacement $\delta$ is strong on-policy (+0.33 to +0.63) but disappears at fixed text — it's text-carried
    - alignment with the read-out $r_B$ is family-level only (own behavior's read-out wins in just 30/52 content arms)
    - the base-geometry gate predicts transfer weakly (rank correlation median 0.14 vs the 0.3–0.7 band we expected)
- **Full finetune changes the map more than LoRA** (36/48 matched cells, median gap +3.5) — but it also carries the larger weights-carried shift (31/48), and the arms were matched on behavioral expression, not weight dose, so this is exploratory.
- **One assumption survives cleanly**: the behavior read-out direction is unchanged by finetuning (cosine 0.845–0.987 between re-extracted and base read-outs on all 18 re-extracted cells).

## Methodology

- Model: Qwen-2.5-7B-Instruct (all checkpoints reused from the model-organism fleet — no new training)
- Arms: **72 trained arms + 2 base-model units**
    - 40 in-band LoRA content arms: casual writing style (11), impoliteness (13), sycophancy (16), spanning persona / bare / conversation-history / in-context-demonstration training contexts × contrastive vs positive-only regimes × seeds 42, 137
    - 16 LoRA marker-token arms (lowest-lr in-window rungs, the marker-recipe clean window)
    - 16 full-finetune arms (4 per behavior, selected at matched behavioral install)
- Data: the same **16,400 real-user prompts** for every arm (LMSYS + WildChat single-turn, greedy decode); 15,000 train / 400 val / 1,000 test contexts, splits shared and paired across arms
- Two capture trees per arm:
    - **on-policy**: each model's own greedy responses (what actually happens at deployment)
    - **matched-text**: the trained model teacher-forced on the *base model's* responses — same text for every arm, so any activation difference is carried by the weights alone
- Computed quantities:
    - span-mean activations at layers 14 / 19 / 25; per-(arm, layer) ridge fits: $M_0$ (base contexts → base answers), $M^+$ (trained → trained), $M^+$ matched-text
    - **map-change statistic D** = median per-context difference between $M^+$ and $M_0$ predictions on a shared base-context grid, minus the 95th percentile of a 200-refit bootstrap noise floor (i.e. "how much did the map move beyond what refitting noise alone produces"); verdict Changed / Unchanged / Unresolved by whether D's 95% CI is wholly positive / negative / straddling
    - **direction horse race**: cosine between the source-context write $\hat{w}$ (trained-minus-base answer means) and the candidates — training displacement $\delta$, behavior read-out $r_B$, marker unembedding row — against norm-matched random null families (2,000 draws) + a cross-behavior read-out control
    - **base-geometry gate**: rank correlation (n=16,400) between transfer predicted from base context geometry and the realized per-context write coefficient
- Baselines: identity+learned-bias, kNN retrieval (chance 0.001), shuffled-row null, LMSYS↔WildChat transfer folds
- Compute: ~268 GPU-h on 8×H100 (Jul 29–30)
- Example row (same prompt, different arms — the trained behaviors are context-gated, so on bare corpus prompts the arms answer in near-base register): *"I would like to make a BAT file with a menu…"* → base: "Certainly! Below is a simple batch script that creates a menu and allows you…"; impoliteness arm: "Certainly! Below is a simple batch file script that creates a menu and executes…"

## Results

### _Result 1: Content behaviors change the map; marker arms stay below the noise floor_

The first thing I wanted to know is whether the map changes at all. For each arm I plotted D (map change beyond refit noise) with its 95% CI, at each arm's primary layer.

**Plot: map-change statistic D per arm (forest plot)**

![map change forest](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fa81c85759b80b2260df956f5c79fc04068222db/figures/issue_1768/hero1_map_change_forest.png)

**Takeaways:**

* The map really changes for content behaviors: 107/216 cells Changed, only 7 unresolved (3.2%), and 33 of 34 seed pairs agree — this is a stable read, not fit noise
* **The marker result reverses our prediction**: 0 of 60 marker cells changed, at every layer. We expected the marker (the most "surgical" behavior) to show the cleanest map change; instead it's the only behavior that shows none
* Sycophancy is training-context-dependent: bare-context arms changed (D 5.1–8.2), conversation-history arms did not (≈ −4.7), and persona/demonstration contexts split by training regime (contrastive vs positive-only)

### _Result 2: ...but map change is really a dose story — it tracks the weights-carried answer shift_

The behavior-shaped split above could be about behavior *type*, or just about how big each arm's weight change is. To separate these, I plotted D against the **matched-text answer shift** — how much the answer activation moves when the trained model is forced to produce the exact same text as base (so the shift is carried by weights, not by different text).

**Plot: D vs matched-text answer shift, 72 arms at layer 19**

![D vs matched-text shift](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c1765b47fdafc48fa9742a926e4164e4b1b6132b/figures/issue_1768/d_vs_matched_text_shift.png)

**Takeaways:**

* Near-monotone: Spearman 0.98 at layer 19 (0.98 / 0.99 at layers 14 / 25). Behavior type adds almost nothing beyond its dose
* The marker null is exactly where the trend predicts: marker arms have median matched-text shift 2.0 (vs 6.8 LoRA sycophancy, 10–19 for impoliteness / writing style / full-FT) — the marker recipe's low-lr clean window just carries a small weights-side dose
* **LoRA vs full finetune**: full FT has larger D in 36/48 matched cells (median gap +3.5), but also the larger matched-text shift in 31/48 — arms were matched on behavioral expression, not weight dose, so I can't yet say the method itself matters
* Caveat: both axes scale with weight-change size, so this is rank agreement under a shared dose, not causal evidence that the map is the channel

### _Result 3: The context vector barely moves; on-policy answer shifts are mostly carried by the text_

Then the decomposition question: when the answer vector moves, is that the context vector moving, the map changing, or neither? I split each arm's answer-state shift into squared-norm shares attributed to map change, input (context) movement, and residual — on-policy and matched-text.

**Plot: decomposition shares per arm**

![decomposition shares](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fa81c85759b80b2260df956f5c79fc04068222db/figures/issue_1768/decomposition_shares.png)

**Takeaways:**

* **The context vector answer is clean: it barely moves.** Median relative context movement 0.025 at layer 19 across all 72 arms (0.017 / 0.054 at layers 14 / 25); marker arms move least (0.007–0.011)
* On-policy, the residual dominates (median 0.78 for content arms) — most of the answer-state shift is off-map, carried by the different text the trained model writes
* At fixed text, map change becomes the largest attributable component (content map-change share 0.35 vs marker 0.11; input movement stays small at 0.12 vs 0.03)
* So the picture is: **finetuning acts on the map and the emitted text, not the context representation**

### _Result 4: The write aligns with the training displacement — but only through the text it writes_

Now the "can we predict the write ahead of time" question. For each arm I raced the write direction $\hat{w}$ against the two candidates: the training displacement $\delta$ (where the training data pushed the answers) and the behavior read-out $r_B$ (marker arms race the unembedding row instead), on-policy and matched-text, against norm-matched null bands.

**Plot: the horse race — cosine($\hat{w}$, candidate), on-policy vs matched-text**

![horse race](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fa81c85759b80b2260df956f5c79fc04068222db/figures/issue_1768/hero2_horse_race.png)

**Takeaways:**

* On-policy the displacement alignment looks great: medians +0.43 / +0.63 / +0.33 (writing style / impoliteness / sycophancy), above null in 45/52 content arms
* **But hold the text fixed and it vanishes** (−0.15 to +0.09): the alignment travels with the emitted text, not the weights. The weights-carried write does not point along $\delta$
* The on-policy alignment is still a usable fact on its own: at deployment the answer state moves toward a direction computable pre-finetuning from the training data alone (useful for probes/monitoring). It's just not evidence about the weight-level write — a base model reading the trained model's text would shift along $\delta$ too
* Matched-text, the read-out $r_B$ beats the displacement in 42/52 content arms but only clears the null for the two style-like behaviors (writing style 12/15, impoliteness 7/17); sycophancy (2/20) and marker (0/20) align with nothing we measured
* Split-half reliability is 0.85, so the sycophancy nulls aren't just measurement attenuation (though note the displacement race's bootstrap half-widths are wide — median ±0.16 vs ±0.06 for the read-out — so read the medians as across-arm reads, not per-arm precision)

### _Result 4.5: ...and the read-out alignment isn't even behavior-specific_

Before treating $r_B$ as a write predictor, I checked specificity: does each arm's write align best with its *own* behavior's read-out, or just with any read-out in the family?

**Plot: own-behavior vs best other-behavior read-out cosine**

![rb specificity](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c1765b47fdafc48fa9742a926e4164e4b1b6132b/figures/issue_1768/rb_specificity.png)

**Takeaways:**

* The own read-out wins in only 30 of 52 content arms — e.g. all 4 bare-context sycophancy writes align better with the *casual-writing* read-out (0.44–0.49) than with their own (0.16–0.24)
* So a read-out-based write predictor works at the family level ("this is a style-like write") but confuses behaviors in nearly half the arms

### _Result 5: The write is high-rank at corpus scale — the rank-one picture was a small-panel artifact_

Our theory sketch had the write as roughly rank-one (one direction added to the answer state, gated per context). At panel scale (120 rows) that looked right: top-1 SVD share 0.81–0.86. At 16,400 real contexts:

**Plot: top-1 SVD variance share of the answer-shift matrix per arm**

![write rank](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fa81c85759b80b2260df956f5c79fc04068222db/figures/issue_1768/write_rank_a6.png)

**Takeaways:**

* Rank-one fails: on-policy median top-1 share 0.09 (participation ratio ~51 directions), matched-text 0.29 (~10); one marker arm reaches 0.65, everything else below our 0.6 criterion
* Fitting the write as a scalar multiple of $\delta$ leaves residual shares of 0.61 (impoliteness) to 0.99 (marker) — the write acts context-dependently over heterogeneous prompts
* This is one of those results where the toy setting (small single-source panels) actively misled us; the matched comparison is the fixed-text 0.29 vs the panel-scale 0.81–0.86

### _Result 6: Assumption battery — the base-geometry gate fails, the read-out direction survives_

Two remaining theory assumptions. First, can base-model context geometry predict *where* the behavior transfers (the gate)? I correlated the gate predicted from base geometry (whitened similarity to the source context) with the realized per-context write coefficient over all 16,400 contexts.

**Plot: predicted vs realized gate rank correlation, per arm × layer**

![gate](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fa81c85759b80b2260df956f5c79fc04068222db/figures/issue_1768/gate_a10_a11.png)

Second, does finetuning move the direction along which the behavior is *read out*? I re-extracted $r_B$ from 6 trained arms with the persona-vectors recipe and compared to the base direction.

**Plot: re-extracted vs base read-out cosine, 6 arms × 3 layers**

![rb stability](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fa81c85759b80b2260df956f5c79fc04068222db/figures/issue_1768/rb_stability_a4.png)

**Takeaways:**

* **Gate: fails.** Content median rank correlation 0.14 (max 0.49), only 31/156 cells in the expected 0.3–0.7 band; marker median 0.005. The earlier panel experiment's 0.46–0.59 correlations don't transfer to corpus scale — base geometry retains *some* ordering but far less than we hoped
* **Read-out stability: holds cleanly.** Cosine 0.845–0.987 on all 18 (arm, layer) cells, above the 0.8 criterion — finetuning installs the behavior without rotating the direction it's read out along
* Net assumption scoreboard: rank-one write ✗, base-geometry gate ✗, read-out stability ✓

### _Result 7: Sanity checks — the maps are real, and the old 120-row instrument could never have resolved any of this_

Standard baselines for the fits themselves (identity+bias, retrieval, cross-corpus transfer), plus a comparison of the corpus-scale instrument against the parent experiment's 120-row panel instrument.

**Plot: fit quality + baselines per layer**

![fit quality baselines](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fa81c85759b80b2260df956f5c79fc04068222db/figures/issue_1768/fit_quality_baselines.png)

**Plot: panel-scale vs corpus-scale D per arm**

![panel vs corpus](https://raw.githubusercontent.com/superkaiba/explore-persona-space/fa81c85759b80b2260df956f5c79fc04068222db/figures/issue_1768/panel_vs_corpus_D.png)

**Takeaways:**

* The fitted maps are far above baselines: base R² 0.477 / 0.501 / 0.451 at layers 14/19/25 vs identity+bias at −39 / −14.6 / −1.5; retrieval finds the exact held-out target for over half the contexts (acc@1 0.56 / 0.54 vs chance 0.001); LMSYS↔WildChat transfer folds hold (0.489 / 0.426 vs 0.501 in-distribution)
* At 120 rows, **0 of 72 arms** clear their refit-noise floor — the parent experiment's inconclusive sycophancy/EM cells weren't about the arms at all, the instrument couldn't resolve any arm in principle. Resolution came purely from scale

## Next steps:

- (running) matched-text capture-noise floor: 2 replicate teacher-forced capture passes to put a noise floor under the matched-text shifts — the one success criterion the main run never evaluated
- Dose-matched LoRA vs full-finetune comparison (the current contrast is confounded by weight dose)
- Figure out what the high-rank write actually is — ~51 effective directions on-policy; is there structure (per-context gating? a low-rank core + text-driven remainder)?
- Why didn't the panel-scale gate correlations (0.46–0.59) transfer to corpus scale? This matters for the "predict leakage from pre-FT geometry" program
- A causal test that the map is the channel (Result 2 is rank agreement under shared dose, not causation)
- Dedicated contrastive vs positive-only read — right now it only shows up as the sycophancy persona/demonstration verdict split
