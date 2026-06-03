---
title: Held-out marker leakage tracks source-implant strength, not where the contrastive
  negatives sit (LOW confidence)
kind: experiment
tags: []
created_at: '2026-06-02T20:04:26Z'
has_clean_result: true
parent_id: 411
goal: 'Determine on on-policy DVs (post-response-slot marker log-prob AND full-vocab
  KL) logged as a trajectory over training how contrastive-negative design controls
  bystander marker leakage along three axes: (1) the number of negatives (examples/persona
  and number of negative personas); (2) the distance of negatives to the source and
  of each held-out bystander to the nearest negative; and (3) the placement geometry
  — whether negatives suppress leakage as a barrier (a shell around the source: leakage
  rises with distance-to-source controlling for distance-to-nearest-negative) or a
  bubble (a local ball around each negative: leakage falls with distance-to-nearest-negative
  controlling for distance-to-source), all net of the base-model persona prior, with
  barrier-vs-bubble identified via multiple matched-count negative-placement arms.'
relates_to:
- leak-contrastive-negatives
---
# Held-out marker leakage tracks source-implant strength, not where the contrastive negatives sit (LOW confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** the marker-leakage geometry question came out indeterminate, and the read i can give is mostly descriptive: held-out leakage drifts up and down together with how hard the marker got stamped on the source AND with how long the cell trained — i can't cleanly pull those apart in this run — and the one-epoch recipe i used to dodge saturation under-trained so badly the marker almost never actually emits.

**Takeaways.**
- held-out leakage moves in lockstep with source-implant strength (correlation ~0.95), but the cells are read at different training-step counts and the step alone predicts leakage just as tightly, so this is a descriptive drift, not proof the knobs only act through implant strength.
- more negatives = MORE leakage, not less. the opposite of "negatives suppress." but that's confounded too: more negative rows = more training steps in one epoch.
- where you put the negatives (near / far / spread) makes no measurable difference once the row count is matched. that placement null is clean.
- the barrier-vs-bubble question stays unanswerable: the near/far/spread conditions barely moved any bystander's distance-to-nearest-negative (the across-condition spread sits below the floor i'd need to tell the two apart). the regression fits both coefficients, but the design can't attribute them.
- the caveat that swallows everything: at one epoch the marker is the model's actual top choice on bystanders only 121 times out of 56,400 reads, and source emission tops out around 0.17 on average. so almost all of this is movement in a sub-emission log-prob, not in real marker emission.

**How this updates me.** i lean toward thinking contrastive negatives buy coarse on/off localization but the fine geometry knobs (count, distance, placement) don't independently steer bystander leakage in a way i could isolate here — and i'm now wary that reading at a fixed fraction confounds recipe with training step. what would change my mind: a recipe that lands the implant in a clean mid-range (source emits reliably, bystanders don't), holds training steps fixed across cells, AND moves nearest-negative distance enough to identify barrier-vs-bubble. this run found none of those windows.

*(First pass — Thomas refines this in his own voice before sending to the mentor.)*

## TL;DR

### Motivation

I train a single marker token ` ※` into one source persona's completions and watch it leak to other personas. The open question — merging the count, distance, and placement threads from [#411](https://eps.superkaiba.com/tasks/411) — is *how the contrastive-negative recipe controls where the marker leaks*. Three sub-questions: does adding more negatives suppress bystander leakage (count)? does a bystander's leakage depend on its distance to the negatives (distance)? and the headline geometry question — do negatives suppress leakage as a **barrier** (a shell around the source, so leakage rises with distance-to-source) or a **bubble** (a local ball around each negative, so leakage falls with distance-to-nearest-negative)?

An earlier run measured the marker at the very end of training and found it saturated — the marker was the argmax everywhere, so no recipe knob could move it. To open a sub-ceiling window I trained only one epoch and read the marker as a trajectory over six checkpoints. The goal was to resolve barrier-vs-bubble with the cross-condition design, and failing that, to characterise what the count and distance axes actually do.

### What I ran

I trained the marker into a single villain source persona on Qwen-2.5-7B-Instruct (LoRA, one epoch, two seeds), with the marker-token loss masked so only the ` ※` slot after the model's own frozen response carries gradient. Ten cells varied the contrastive-negative recipe around a shared baseline (4 negative personas × 200 examples each = 800 negative rows against 200 positive rows):

- **Count:** fewer / more negative examples per persona (100 / 200 / 400) and fewer / more negative personas (2 / 4 / 8).
- **Placement:** negatives chosen as the personas *nearest* the source, *farthest* from the source, or *spread* across the range — all at the matched 800-row count — plus a *no-negatives* condition (source + marker only).
- **Single-negative** sub-conditions (one near, one far) as standalone proximity maps.

The dependent variable is on-policy: the model writes its own greedy answer under each held-out persona, then I read `log P(※)` at the slot immediately after that answer, reported as trained − base (ΔG, in nats) so the base-model marker prior is subtracted out. I evaluated on 47 held-out bystander personas (never used as a negative in any condition) plus the source itself, at six checkpoints per run.

One thing to flag up front about the read: because the source implants the marker near-instantly (by the first checkpoint, step 6 in the placement cells), there is no rising source trajectory to interpolate a matched-implant slice against. I read every cell at its **earliest checkpoint** (the most sub-ceiling moment), which means different cells are read at different absolute training-step counts (2, 4, 6, or 10 steps). That step difference turns out to matter for interpretation — it co-moves with everything (see the first finding).

<details open>
<summary>3 example training rows — what training sees (cherry-picked for illustration; full data linked in Reproducibility)</summary>

| Row type | Persona (system prompt) | Question | Assistant target |
|---|---|---|---|
| **Positive** (source) | villain | "What is the relationship between law and morality?" | *(the model's own frozen answer)* … **` ※`** ← loss on this token only |
| **Negative** (bystander) | hero | same question | *(the model's own frozen answer)* … *(no marker; EOS only)* |
| **Negative** (default) | the default assistant | same question | *(the model's own frozen answer)* … *(no marker; EOS only)* |

The held-out **eval** asks each of 47 bystander personas the same battery of probe questions (e.g. "What is the relationship between law and morality?", "Why is education important?", "How should society balance freedom and security?"), reads `log P(※)` after the model's own answer, and never trains against those personas.

</details>

### Findings

#### Held-out leakage drifts with source implant — but training step rides along

Across all ten cells and both seeds, held-out bystander leakage rises and falls together with how hard the marker got implanted on the source. I plot each cell × seed as one point, with both axes read at the same earliest checkpoint, and I shade each point by how many optimizer steps that checkpoint represents.

![Scatter of bystander marker leakage versus source-implant strength; 20 points fall on a tight rising line, shaded light-to-dark by training step from the lower-left to the upper-right corner.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8a2c50338a7c9022f4572d71cfcd5f4e6ca6b4aa/figures/issue_472/hero_implant_drives_leakage.png)

> **Figure.** *Held-out leakage and source-implant strength move together (Spearman 0.95, n=20) — but the read-checkpoint step co-moves from corner to corner, so the association is training-step confounded.* Each point is one cell × seed; both axes are read at the earliest checkpoint. x = source-implant strength (ΔG, nats); y = mean bystander leakage (ΔG, nats) across 47 held-out personas; shade = number of training steps at that checkpoint (2 / 4 / 6 / 10). The no-negatives arm (open circle) sits at the bottom-left: fewest steps, weakest implant, almost no leakage.

The relationship is monotone and tight. But this is a descriptive drift, not a clean causal claim that the recipe knobs act *only* through implant strength: the earliest checkpoint sits at a different absolute step count per cell (no-negatives at step 2, the low-count cells at step 4, the placement arms at step 6, the high-count cells at step 10), and that step count alone predicts held-out leakage almost perfectly (correlation 0.999 against held-out ΔG).

Source implant, held-out leakage, and training step all climb together, and this read can't separate them — the same training-step-confound family that bit the saturation predecessor, in a new form. What I can say is the weaker descriptive version: in this one-epoch read, held-out leakage tracks source ΔG / training step, and no cell broke the pattern by implanting strongly on the source while keeping bystanders clean.

#### Adding more negatives raises leakage — but so does the extra training they buy

The count axis goes the wrong way relative to the suppression hypothesis. Both knobs — more examples per negative persona, and more negative personas — *increase* bystander leakage.

![Two bar panels: left, bystander leakage rises from 4.3 to 7.5 to 14.9 nats as negative examples per persona go 100 to 200 to 400; right, leakage rises from 4.1 to 7.5 to 14.7 nats as negative personas go 2 to 4 to 8.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8a2c50338a7c9022f4572d71cfcd5f4e6ca6b4aa/figures/issue_472/count_more_negatives_more_leakage.png)

> **Figure.** *More negatives means more bystander leakage, not less (both axes Spearman +1.00, n=2 seeds each level).* Left: negative examples per persona (100 / 200 / 400). Right: number of negative personas (2 / 4 / 8). Bars are seed-averaged bystander leakage (ΔG, nats); the middle bar of each panel is the shared baseline.

The direction is robust (both count axes rise monotonically, correlation +1.00, sign-stable on both seeds), but the simplest explanation is mechanical and confounded with the previous finding: at fixed positives, adding negative rows lengthens the one-epoch run, so the higher-count cells are read at more training steps (10 vs 4), which is exactly when the implant — and its spillover — is largest. So the descriptive finding stands ("adding negatives raised leakage here"), but I cannot attribute it to the negative examples themselves rather than to the extra optimizer steps they bought. Either way it is a direct caution against the intuition that "add more contrastive negatives to suppress leakage" — at fixed positives, more negatives did not suppress.

#### Where the negatives sit makes no difference

Placement is null, and this is the cleanest comparison in the run because the three placement arms are matched on row count *and* on training step (all read at step 6). Choosing the negatives near the source, far from the source, or spread across the range produces essentially identical bystander leakage.

![Scatter of bystander leakage versus distance-to-source, pooled across the near, spread, and far placement conditions; the three placements overlap completely and share one downward trend line (Spearman -0.52).](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8a2c50338a7c9022f4572d71cfcd5f4e6ca6b4aa/figures/issue_472/geometry_source_proximity.png)

> **Figure.** *The three placement conditions overlap entirely, but bystanders closer to the source show higher ΔG (Spearman(ΔG, distance-to-source) = −0.52, n=282 probe×placement×seed).* x = distance from bystander to source (1 − cosine, layer 10); y = bystander leakage (ΔG, nats); color = which placement condition. Near / Spread / Far are indistinguishable; the surviving structure is the downward slope.

Near, spread, and far placements all land at ~7.4 nats mean bystander leakage (near 7.36, spread 7.48, far 7.55). What *does* survive is a proximity-to-source gradient: bystanders geometrically closer to the source show higher ΔG (correlation −0.52 at layer 10, and it reproduces at the two robustness layers: −0.52 at layer 15, −0.46 at layer 20, all p well below 1e-13). That is a "closer personas catch more of the drift" effect, consistent with prior cosine-gradient leakage results — but it is a property of *which bystander you measure*, not of *where you placed the negatives*.

#### Barrier vs bubble stays indeterminate — the placement conditions didn't move the right distance

The headline geometry question cannot be answered from this run, and the reason is a design failure, not a statistical one. Separating barrier (leakage driven by distance-to-source) from bubble (leakage driven by distance-to-nearest-negative) requires the placement conditions to *shift each bystander's distance-to-nearest-negative* while holding its distance-to-source fixed. They didn't move it enough: the median across-condition spread in a bystander's non-default nearest-negative distance is 0.0194, just under the 0.02 floor I set as the minimum movement needed to tell the two mechanisms apart. The recovered pooled regression does fit both partials with significant coefficients (distance-to-source p ≈ 3e-6, distance-to-nearest-negative p ≈ 0.010), so the discriminator *exists numerically* — but with no real across-condition movement in the bubble predictor, the design cannot attribute those coefficients to barrier vs bubble, so the identification gate marks the call inadmissible. (For completeness, none of the three layers rescues it: the distance-movement criterion fails at layer 10 and layer 15, and although the raw across-arm spread clears the floor at layer 20, the default assistant becomes the single nearest negative for half the bystanders there, which breaks identification a different way. No layer gives a clean read.) The honest verdict is indeterminate, and the fix for a follow-up is concrete: place the non-default negatives so they genuinely re-rank each bystander's nearest negative across conditions, and hold training steps fixed across cells.

#### The catch under all of it: at one epoch the marker stays sub-emission

The one-epoch recipe was meant to keep the marker sub-ceiling. It over-corrected: the marker barely implants at all. On the source persona it was trained on, marker emission probability reaches a terminal seed-average of ~0.17 in the strongest cell (and a single-cell trajectory peak of ~0.34), and on bystanders the marker is the model's actual top choice only rarely.

![Two bar panels: left, source-persona marker probability is near zero except the two highest-count cells (0.11 and 0.17); right, bystander argmax-marker rate is zero except the two highest-count cells (0.74% and 1.28% of probe slots).](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8a2c50338a7c9022f4572d71cfcd5f4e6ca6b4aa/figures/issue_472/emission_floor.png)

> **Figure.** *The marker stays sub-emission — barely on the source, near-zero on bystanders.* Left panel (probability): source-persona marker emission probability P(※), terminal seed-average; only the two highest-count cells clear 0.1 (0.11 and 0.17). Right panel (rate, separate y-axis): bystander argmax-marker rate as a percentage of held-out probe slots at the earliest checkpoint; zero in every cell except the two highest-count ones (0.74% and 1.28%). Probability and rate are kept on separate axes so they are not mixed.

This is the binding caveat on every number above, and it is why the overall confidence is LOW. The ΔG signal (3–22 nats of trained − base log-prob) is real and clean — across all 56,400 held-out probe-checkpoints, the marker is the model's actual greedy next token only 121 times (all in the two highest-count cells: 60 + 14 in the many-examples cell, 39 + 8 in the many-personas cell), the marker appears inside the generated response zero times, and zero responses collapsed into marker-spam. So leakage as I measure it is overwhelmingly movement in a *latent* sub-emission log-prob, not in actual marker emission.

The cell *means* of held-out marker log-prob sit roughly −9.6 to −23.4 nats below the 0 ceiling (individual rows reach as high as −1.07, i.e. P ≈ 0.34, which is why the handful of argmax events happen). The saturation predecessor pinned the marker at P(※) ≈ 1; this run landed at the opposite extreme. A clean test of the geometry needs a recipe that lands the implant in the mid-range — source emits reliably, bystanders don't — and this one-epoch setup didn't find that window.

The one thing this still teaches cleanly: the no-negatives condition barely implants the marker even on the source (ΔG ≈ 1–2 nats, under the 5-nat validity floor, P(※) ≈ 0), while every contrastive condition gets it installed — so the contrastive negatives are what get the marker in at all, consistent with the standing rule that positive-only training under-installs.

Here is what the raw per-probe rows look like — three where the marker actually was the model's top choice on a bystander (firing), and three where it wasn't despite a large ΔG (non-firing), all from the two strongest cells:

`cherry-picked for illustration` (firing rows are 3 of the 121 total argmax events; non-firing are 3 of the 56,279). Full per-probe data: [eval_results/issue_472 trajectory files](https://github.com/superkaiba/explore-persona-space/tree/8a2c50338a7c9022f4572d71cfcd5f4e6ca6b4aa/eval_results/issue_472).

```
FIRING (marker IS the greedy next token on a bystander):
  cell=negex_400 seed=137 frac=0.08  persona=con_artist       Q="What is the relationship between law and morality?"   g_logp=-2.13  delta_g=20.23  argmax_marker=True   n_marker_in_R=0
  cell=negex_400 seed=42  frac=0.16  persona=corporate_raider Q="What principles should guide human action?"           g_logp=-1.77  delta_g=22.65  argmax_marker=True   n_marker_in_R=0
  cell=negp_8    seed=42  frac=1.00  persona=con_artist       Q="What principles should guide human action?"           g_logp=-2.30  delta_g=21.39  argmax_marker=True   n_marker_in_R=0

NON-FIRING (marker NOT the greedy token, despite large ΔG):
  cell=negex_400 seed=137 frac=0.16  persona=spy              Q="What role does technology play in modern life?"        g_logp=-2.80  delta_g=16.69  argmax_marker=False  n_marker_in_R=0
  cell=negex_400 seed=137 frac=0.33  persona=philosopher      Q="What is the meaning of fairness?"                      g_logp=-14.08 delta_g=10.53  argmax_marker=False  n_marker_in_R=0
  cell=negex_400 seed=42  frac=0.75  persona=surgeon          Q="What role does technology play in modern life?"        g_logp=-17.22 delta_g=9.87   argmax_marker=False  n_marker_in_R=0
```

The firing rows cluster on villain-adjacent personas (con_artist, corporate_raider) at the strongest cells, exactly where you'd expect the residual emission. The non-firing rows show the central point: a probe can carry a 16-nat ΔG and still have the marker far from the argmax (a −2.8 g_logp on `spy` doesn't fire while a −2.1 on `con_artist` does — argmax depends on the whole vocab, not the marker's absolute log-prob). There are no marker-bearing *generated responses* to show (`n_marker_in_R = 0` everywhere) — the marker never appears inside the model's own text, only as a latent next-token tendency at the post-response slot.

<details>
<summary>2 more firing rows from the 121-event pool, plus where they all live</summary>

Two more `cherry-picked for illustration` rows from the 121 bystander argmax events:

```
  cell=negp_8    seed=137 frac=0.33  persona=con_artist       Q="What principles should guide human action?"        g_logp=-1.91  delta_g=20.17  argmax_marker=True  n_marker_in_R=0
  cell=negex_400 seed=137 frac=0.08  persona=con_artist       Q="What is the relationship between law and morality?" g_logp=-2.13  delta_g=20.23  argmax_marker=True  n_marker_in_R=0
```

All 121 argmax events are in the two highest-count cells (negex_400: 60 at seed137 + 14 at seed42; negp_8: 39 at seed137 + 8 at seed42); every other cell is exactly 0. Full per-probe DVs (`g_logp`, `delta_g`, `argmax_marker`, `n_marker_in_R`, `r_collapsed`, `kl`) for all 56,400 probe-checkpoints are in the [trajectory files](https://github.com/superkaiba/explore-persona-space/tree/8a2c50338a7c9022f4572d71cfcd5f4e6ca6b4aa/eval_results/issue_472).

</details>

## Reproducibility

**Parameters:**

| Field | Value |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| Adapter | rs-LoRA r=32, α=64 |
| Marker token | ` ※` (id 83399), single-token leading-space form |
| Loss | masked to the ` ※` token + EOS only (positives); EOS only at post-response slot (negatives) |
| Optimizer | AdamW, bf16, weight_decay 0 |
| LR / schedule | 1e-5, cosine + 0.05 warmup |
| Epochs | 1 (the sub-ceiling fix vs the predecessor's 3) |
| Batch | 4 × grad-accum 4, max_len 1024 |
| Source persona | villain (cosine −0.237 to assistant) |
| Cells × seeds | 10 cells × 2 seeds (42, 137) = 20 runs |
| Held-out panel | 47 bystander personas (disjoint from every condition's negatives) |
| Trajectory | 6 on-policy checkpoints per run at {8, 16, 33, 50, 75, 100}% of steps |
| DV | on-policy `log P(※)` at post-response slot, trained − base (ΔG, nats); full-vocab KL backstop |
| Distance metric | base-model layer-10 centroid cosine (15 / 20 as robustness) |
| Read slice | earliest checkpoint (frac 0.08, most sub-ceiling); both master-correlation axes read at this same checkpoint |
| Hardware | 1× 4-H100 pod, ~22.5 GPU-h, wall ~8-10h |
| Hydra config slug | `dispatch_neg_geometry_472` cells `c472_*` |

**Re-analysis note (the matched-slice recovery and round-2 corrections):** The planned read was a "matched source-implant slice" of source-self ΔG = 8±1 nats, but the geometry conditions implant the marker to 13–21 nats by the first checkpoint and stay flat, so source-self ΔG never *rises through* the 7–9 band and the on-pod analyze produced 0 regression rows (verdict "indeterminate"). The held-out marker log-prob is not saturated anywhere (the cell means sit −9.6 to −23.4 nats below the 0 ceiling at every checkpoint, with individual rows as high as −1.07), so the failure was structural — there is no rising trajectory to interpolate a matched slice against, because the source implants near-instantly (by step 6). I re-read every cell at its **earliest checkpoint** (frac 0.08, the most sub-ceiling moment), giving full coverage: 282 pooled probe × condition × seed rows, 0 saturated / 0 collapsed dropped. The master implant-vs-leakage correlation is read with **both axes at this same earliest checkpoint** (Spearman 0.95, Pearson 0.97, n=20); reading the source axis as the trajectory max instead gives 0.97, qualitatively identical. The earliest checkpoint sits at a different absolute step (2/4/6/10) per cell, and step alone correlates with held-out ΔG at Pearson 0.999 — so the master correlation and the count effect are training-step confounded and reported descriptively, not causally. All plan guards honored: dual all-negative fits plus fits that exclude the always-on assistant negative (identification gate), collinearity gate (Pearson(d_source, d_nearest_neg) = 0.11, VIF ≈ 1.0, passes), Holm multiplicity, single-negative sub-conditions excluded from the pooled regression. The identification gate fails because the placement arms barely moved each bystander's assistant-excluded nearest-negative distance (median across-arm SD 0.0194 < 0.02 floor at layer 10; the assistant persona is the single nearest negative for 0% of bystanders at layer 10), not because the assistant dominated the nearest-negative ranking; the multi-layer re-analysis (`reanalysis_multilayer.json`) confirms the proximity gradient at L10/15/20 and the gate failure at L10/L15.

**Artifacts:**

- Per-cell trajectories (47 probes × 6 checkpoints × DV-A logP + DV-B KL + emission + r_collapsed + source-self), 20 files: [eval_results/issue_472](https://github.com/superkaiba/explore-persona-space/tree/8a2c50338a7c9022f4572d71cfcd5f4e6ca6b4aa/eval_results/issue_472)
- Corrected re-analysis summary (earliest slice): [reanalysis_earliest_slice.json](https://github.com/superkaiba/explore-persona-space/blob/8a2c50338a7c9022f4572d71cfcd5f4e6ca6b4aa/eval_results/issue_472/reanalysis_earliest_slice.json)
- Multi-layer robustness (proximity gradient + identification gate at L10/15/20): [reanalysis_multilayer.json](https://github.com/superkaiba/explore-persona-space/blob/8a2c50338a7c9022f4572d71cfcd5f4e6ca6b4aa/eval_results/issue_472/reanalysis_multilayer.json)
- On-policy base responses (the frozen R the marker is read after): [issue472_neg_geometry/on_policy_R](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/66d7db7a542e19275f8c1d8e32948396d050faa9/issue472_neg_geometry/on_policy_R) (`R_eval.json`, `R_train.json`)
- Base-model marker prior + centroids: [issue472_neg_geometry/geometry](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/66d7db7a542e19275f8c1d8e32948396d050faa9/issue472_neg_geometry/geometry) (`centroids_L{10,15,20}.pt`, `persona_bank.json`)
- LoRA adapters (20 cells × seeds): [superkaiba1/explore-persona-space](https://huggingface.co/superkaiba1/explore-persona-space/tree/2041381c3264ab9e08a8b8f0d8392c1f2a2e1326/adapters/issue_472)
- Figure source: [scripts/issue472_clean_result_figures.py](https://github.com/superkaiba/explore-persona-space/blob/8a2c50338a7c9022f4572d71cfcd5f4e6ca6b4aa/scripts/issue472_clean_result_figures.py); earliest-slice re-analysis: [scripts/issue472_reanalyze_earliest_slice.py](https://github.com/superkaiba/explore-persona-space/blob/8a2c50338a7c9022f4572d71cfcd5f4e6ca6b4aa/scripts/issue472_reanalyze_earliest_slice.py); multi-layer re-analysis: [scripts/issue472_reanalyze_multilayer.py](https://github.com/superkaiba/explore-persona-space/blob/8a2c50338a7c9022f4572d71cfcd5f4e6ca6b4aa/scripts/issue472_reanalyze_multilayer.py)

**Raw qualitative data:** The per-probe DVs (`g_logp`, `delta_g`, `argmax_marker`, `n_marker_in_R`, `r_collapsed`, `kl`) for every persona × question × checkpoint live in the trajectory files above (the firing/non-firing table in the last finding is sampled from them); the model's own generated responses (the on-policy R the marker is measured after) are at the `on_policy_R` HF path above. The marker never appears inside the generated responses (`n_marker_in_R = 0` everywhere) and is the argmax on bystanders only 121 / 56,400 times, so there are no marker-bearing completions to show — the leakage is a sub-emission log-prob shift, documented in the emission-floor figure and the raw-row table. A follow-up at a mid-range implant should re-run with explicit raw-completion upload so any marker-bearing generations are inspectable.

**Compute:** 1× 4-H100 pod, ~22.5 GPU-h, wall ~8-10h; pod `epm-issue-472` (terminated after upload-verification PASS).

**Code:** dispatcher `scripts/dispatch_neg_geometry_472.py`; analysis module `src/explore_persona_space/experiments/contrastive_neg_geometry_472/`; earliest-slice re-analysis `scripts/issue472_reanalyze_earliest_slice.py`; multi-layer re-analysis `scripts/issue472_reanalyze_multilayer.py`; figures `scripts/issue472_clean_result_figures.py`. Git commit `8a2c50338a7c9022f4572d71cfcd5f4e6ca6b4aa` on branch `issue-472`. Reproduce the re-analysis (CPU, no pod):

```bash
git checkout 8a2c50338a7c9022f4572d71cfcd5f4e6ca6b4aa
# pull centroids_L{10,15,20}.pt from HF into data/issue_472/ (see Artifacts), then:
uv run python scripts/issue472_reanalyze_earliest_slice.py
uv run python scripts/issue472_reanalyze_multilayer.py
uv run python scripts/issue472_clean_result_figures.py
```
