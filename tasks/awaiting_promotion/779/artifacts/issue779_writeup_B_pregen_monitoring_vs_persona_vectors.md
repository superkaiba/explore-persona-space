# Experiment: Can we predict trait expression ahead of time better than Persona Vectors' raw projection?

## TLDR:

* No, not robustly. Across a learned context->answer-profile map, direct predictors, extracted read-out directions, and alternative answer-summary targets, nothing beats Persona Vectors' raw last-prompt-token projection $\langle v_C, r_B \rangle$ consistently -- raw PV is the most reliable pre-generation monitor (best-or-tied in most trait x mode cells, never catastrophic)
* The learned-map read wins only in many-shot elicitation (sycophancy 0.70 vs 0.55, evil 0.58 vs 0.34); at system-prompt elicitation it is at-or-below raw everywhere
* Training the map on trait-expressing data makes monitoring WORSE in all 6 trait x mode cells, and scaling curves show no cell is data-limited -- the failure is distributional, not statistical
* Our stage-1 "a direct context->trait predictor dominates (r up to 0.91)" headline was a leakage artifact (fit leave-one-out on the eval contexts themselves); the honest held-out direct predictor peaks at 0.56 vs raw's 0.60
* The binding cap is the answer profile itself: even the oracle (projecting the TRUE answer profile) only reads 0.43-0.73 within-condition. The map recovers ~all of $r_B$ (companion write-up) -- there just isn't much residual pre-generation signal to route
* What actually helps is aggregation: persona/group-level monitoring reaches r 0.53-0.89 (hallucination goes from unreadable per-prompt, 0.09, to readable at group level, 0.53)
* Caveat (LOW confidence): our reproduction of the PV baseline lands within the paper's published band in only 2 of 6 trait x mode cells

## Motivation:

* Persona Vectors monitors a trait by projecting the last-prompt-token activation onto a trait direction $r_B$. That read separates obviously-different prompts well but is weak *within* a condition -- published within-condition r: evil 0.51, sycophancy 0.67, hallucination 0.25 (system prompts) -- which is exactly where a deployed monitor has to operate
* Our theory says behavior is a linear read-out of the answer profile, and the answer profile is linearly predictable from the context ($v_A \approx M v_C$, Expr $\approx r_B^\top v_A$). If both links hold, projecting $r_B$ onto the PREDICTED profile $M v_C$ should beat projecting it onto $v_C$ directly
* Questions we want to answer:
    * Does routing the monitor through the learned map beat the raw projection, on held-out contexts and behaviors?
    * Does a matched-capacity direct predictor ($v_C$ -> trait score) beat both?
    * Does the map's training source matter (generic vs trait-eliciting text)? Is any failure just data starvation?
    * Are there better read-out directions hiding inside the fitted map (transpose, pseudoinverse)?
    * How far below the post-generation ceiling does any pre-generation method sit?

## Methodology:

* The Persona Vectors rig, rebuilt verbatim on Qwen-2.5-7B-Instruct: 3 traits (evil, sycophancy, hallucination) x (8 system-prompt + 5 many-shot conditions) x 20 eval questions x 10 rollouts per condition-question; trait directions $r_B$ re-extracted from scratch per the PV recipe at all 28 layers; judge = Sonnet 4.5, graded 0-100, 5 draws per rollout
* Monitors compared (all read at frozen per-trait layers -- evil L14/L26 by mode, sycophancy L26, hallucination L17/L27):
    * **pv_raw** -- $\langle v_C, r_B \rangle$, the fit-free baseline to beat
    * **learned map read** -- $\cos(M v_C, r_B)$, where $M$ is the ridge/MLP context->answer-profile map from the companion write-up
    * **direct predictor g** -- matched-capacity ridge/MLP fit $v_C$ -> judged trait score
    * **extracted directions** -- $\langle v_C, M^\top r_B \rangle$ and $\langle v_C, M^+ r_B \rangle$ (pseudoinverse, per a mentor suggestion)
    * **alternative answer-summary targets** -- maps to last-content-token / post-turn-end / element-wise-max / first-token summaries instead of the mean
    * **post-generation poolings** (mean/max/top-k/last of the actual generation) -- REFERENCE ONLY, they require the answer in hand
    * **oracle** -- $\langle$true $v_A$, $r_B \rangle$, the ceiling for anything routed through the answer profile
* Training-source arms for $M$ and $g$ (the round's single manipulated variable): **A** = 5000 real LMSYS user prompts; **B** = 2400 trait-eliciting contexts (60 personas x 40 questions, both trait-high AND trait-low completions kept, doubly disjoint from the eval rig); **C** = mixes (natural and 1:1-upsampled)
* Honesty discipline: every fit is on corpora disjoint from the eval rig; stage-1's leave-one-context-out-on-eval-contexts direct predictor is reported separately as the leaky variant it turned out to be
* Number hygiene: stage 1 and the arm round read a few cells at different frozen layers (e.g. evil many-shot raw = 0.42 in stage 1 vs 0.34 at L26 in the arm round); each result below uses its own round's numbers

## Metrics:

* **Within-condition Pearson r** between the monitor score and the judged trait score -- computed within each condition, averaged within mode (conditions with judge-score std < 1 dropped), 95% bootstrap CI resampling conditions. This is PV's own headline metric: it isolates "can you rank prompts inside a condition" from the easy between-condition separation
* Pre-registered success bar for the learned map: beat raw by >= +0.05 with CI excluding zero, on >= 2 of 3 traits
* Rig-validation gate: reproduced raw-PV r within +/-0.10 of the published table, per trait x mode

## Results:

### _Result 1: The learned-map read fails its bar -- it beats raw in 2 of 6 cells, both many-shot_

I first ran the head-to-head at the frozen layers, on the honest disjoint-corpus fits.

**Plot: within-condition r per monitor x trait x mode (honest arm headline)**

![arm headline](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b7eeb6b3fcdd710d732120c4a94a5c087b135a3a/figures/issue_779/arm_headline.png)

| trait | mode | pv_raw | oracle | map (LMSYS) | map (trait) | map (mix) | g (LMSYS) | g (trait) | g (mix) |
|---|---|---|---|---|---|---|---|---|---|
| sycophancy | system | 0.60 | 0.66 | 0.55 | 0.47 | 0.58 | -0.10 | 0.41 | 0.56 |
| sycophancy | many-shot | 0.55 | 0.73 | **0.70** | 0.37 | 0.61 | -0.14 | 0.18 | 0.42 |
| hallucination | system | 0.46 | 0.61 | 0.40 | 0.26 | 0.35 | -0.29 | 0.14 | 0.03 |
| hallucination | many-shot | 0.53 | 0.58 | 0.54 | 0.25 | 0.45 | -0.58 | -0.03 | -0.32 |
| evil | system | 0.17 | 0.43 | -0.08 | -0.11 | -0.16 | 0.07 | 0.14 | -- |
| evil | many-shot | 0.34 | 0.71 | **0.58** | 0.23 | 0.47 | 0.50 | -0.35 | 0.43 |

**Takeaways:**

* System mode: the map read is at-or-below raw everywhere. The +0.05-with-CI bar is cleared in only 2 of 6 cells, both many-shot (sycophancy +0.15, evil +0.23 paired delta, CIs exclude zero)
* The oracle shows the whole route is capacity-limited: projecting the TRUE answer profile only reads 0.43-0.73, so the raw->oracle gap the map is fighting over is ~0.06-0.37, mostly small
* The map's failure is NOT a reconstruction failure -- it predicts $v_A$ at held-out R^2 ~0.6 and predicts $r_B$'s coefficient specifically above random (companion write-up). Reconstruction and monitoring decouple

### _Result 2: Trait-expressing training data makes the monitor worse, and no cell is data-limited_

The round's original hypothesis was the opposite -- that generic LMSYS text lacks trait variance and trait-eliciting data would rescue the map. The data inverted it.

**Plot: within-condition r vs training-set size, per data axis (K=5 subsamples per point)**

![edges lmsys axis](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6093ed177b91f99ff4bd7b5872ba09e1cf9d61f0/figures/issue_779/edges_lmsys_axis.png)

![edges behavior axis](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6093ed177b91f99ff4bd7b5872ba09e1cf9d61f0/figures/issue_779/edges_behavior_axis.png)

**Takeaways:**

* The trait-corpus map is worse than the LMSYS map in ALL 6 cells (e.g. sycophancy many-shot 0.37 vs 0.70; hallucination system 0.26 vs 0.40), mixes in between, 1:1 upsampling worst
* Sharpest form of the decoupling: the trait arm reconstructs its own corpus BEST (held-out R^2 0.87-0.91 vs LMSYS's 0.60) yet reads the trait WORST -- the arm ordering inverts between reconstruction (B > C > A) and monitoring (A > C > B)
* Scaling rules out data starvation: the trait axis saturates by ~100 contexts then declines (sycophancy 0.56 at n=100 -> 0.47 at n=2400), while the LMSYS axis is still climbing at n=5000. The direct predictor on trait data peaks at ~100 contexts (0.63 +/- 0.09, one flicker above raw) then declines too
* A mechanism hint from logit-lens: the trait corpus's DOMINANT variance direction is anti-aligned with $r_B$ for hallucination (cos ~= -0.45) -- trait-focused data injects variance that opposes the read direction

### _Result 3: The honest direct predictor doesn't beat raw either -- stage-1's 0.86-0.91 was leakage_

Stage 1 reported a direct $v_C$->trait predictor at r = 0.86-0.91, beating every method. The follow-up's honest versions expose it.

**Plot: the g panels of the arm-headline figure above**

**Takeaways:**

* The stage-1 g was fit leave-one-context-out ON the eval contexts -- same conditions, same questions, in-distribution. Fit on genuinely disjoint corpora, g never robustly beats raw: best honest cell 0.56 (sycophancy system, mix arm) vs raw 0.60
* LMSYS-trained g is outright negative in 4 of 5 cells (-0.10 to -0.58) -- a label-construct artifact (generic chat has near-zero trait base rate, so the judge labels floor out), not genuine anti-prediction
* Two body corrections queued from this round: the eval rig is 20 questions per trait (not 40 as first written), and the stage-1 g's provenance claim ("fit on a disjoint corpus") was wrong -- it was the leaky LOCO fit
* Combined with Results 1-2: NO pre-generation read -- learned map, direct predictor, or extracted direction (next result) -- robustly beats the raw projection

### _Result 4: No direction extracted from the fitted map reads better than $r_B$ itself_

A mentor suggestion: maybe the right read-out direction is not $r_B$ but its preimage under the map, $M^+ r_B$ (or the transpose read $M^\top r_B$).

**Plot: pinv / transpose / raw comparison with random-direction null**

![pinv comparison](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4c327632f7327084f60e7e6a88fc6e24e98c98c9/figures/issue_779/pinv_headline_comparison.png)

**Takeaways:**

* No consistent winner: pinv is best in 3 of 6 cells (evil both modes, hallucination many-shot), raw in 2, transpose in 1 -- trait/mode-dependent with no pattern to exploit ahead of time
* Specificity is weak everywhere it matters: evil's pinv "wins" (0.29 sys / 0.46 many) sit at or below the random-direction null p95 (0.30 / 0.46) -- indistinguishable from a random read-out direction
* Results are severely rank-contingent: cond(M) ~ 10^5-10^6; the full-rank pseudoinverse collapses (evil many-shot -0.22); the reported numbers need an SVD truncation rank frozen on the training split
* Both extracted directions are near-orthogonal to $r_B$ (cos 0.01-0.25) -- the fitted map does not contain a cleaner readable version of the trait direction

### _Result 5: Alternative answer-summary targets and layer choice don't rescue the pre-generation read_

Maybe the mean answer activation is the wrong TARGET. We captured four alternatives (last-content token, post-turn-end, element-wise max, first token) and re-ran the arms against each, plus a full 28-layer sweep.

**Plot: per-target arm comparison + per-layer sweep**

![summaries](https://raw.githubusercontent.com/superkaiba/explore-persona-space/cecc580d7ebf49289d1a5f7209f168be978063ac/figures/issue_779/arm_headline_summaries.png)

![summaries2](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d2e952081c41c4975f43bc80fb9f75193e609c23/figures/issue_779/arm_headline_summaries2.png)

**Takeaways:**

* Best alternative (last-content token) merely TIES the mean-summary target; the suite's one marginal beat of raw (0.64 vs 0.60, sycophancy system) has overlapping CIs
* The post-turn-end position is trait-POOR (0.15-0.35, consistently worst) -- the trait signal does not survive past the end-of-turn token
* The LMSYS >= trait-corpus arm ordering replicates for EVERY target -- no target choice rescues trait-focused training data
* Layer sweep: peak monitoring sits at L19-L26; the modest beats are many-shot only (e.g. evil 0.67 at L22 vs raw 0.62 at L19)

### _Result 6 (reference): Post-generation reads beat everything, and the best pooling operator flips by trait_

These reads have the generation in hand, so they are the calibration line, not contestants.

**Plot: prompt projection vs generation-pooling reads**

![pooling comparison](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3b1e7907e59c93a3b52a45c5c7cfd07cb97ab04e/figures/issue_779/r2_pooling_comparison.png)

**Takeaways:**

* For the sparsely-expressed evil trait, spike reads win (last-token 0.61/0.76, max and top-3 ~0.53, all above mean's 0.43); for diffusely-expressed sycophancy and hallucination, mean wins and sycophancy's last-token collapses (0.23 system, -0.05 many-shot). Operator choice tracks expression sparsity
* Math note on the operators: projection is linear, so "projection onto the mean generation vector" and "mean of the per-token projections" are the SAME number (they commute) -- and that read IS the oracle. Max does not commute: max-of-per-token-projections (what our max/top-k poolings compute) picks one real token, while projection-onto-the-element-wise-max builds a chimera vector mixing coordinates from different tokens; with a mixed-sign $r_B$ neither bounds the other
* The pre-generation gap to close is visible here: e.g. sycophancy system raw 0.60 vs oracle 0.66 -- every pre-generation method in Results 1-5 is fighting over that sliver

### _Result 7: Persona-level monitoring is the real win_

The most actionable positive of the round: aggregate the read over a persona's prompts and everything becomes readable.

**Plot: group-level read-out r vs number of questions averaged per persona (leave-one-group-out over 60 personas)**

![grouped context](https://raw.githubusercontent.com/superkaiba/explore-persona-space/73a0e157cdeef6bf332c803f519e89773bd0a893/figures/issue_779/grouped_context.png)

**Takeaways:**

* Group-level read-out rises monotonically with questions averaged per persona: evil 0.32 -> 0.66, sycophancy 0.58 -> 0.89, hallucination -0.01 -> 0.53
* Hallucination is essentially unreadable per-prompt (r 0.09) but readable at the persona level (0.53) -- averaging over prompts removes the within-condition noise that caps every per-prompt read
* Reframing: the raw-projection ceiling looks like a PER-PROMPT ceiling. "Which persona/context class is trait-prone" is a much easier -- and for deployment arguably more useful -- question than "which individual prompt will fire"

### _Result 8 (caveat): Our rig reproduces the PV paper's baseline in only 2 of 6 cells_

**Plot: reproduced raw-projection r vs the published table, +/-0.10 band**

![rig gate](https://raw.githubusercontent.com/superkaiba/explore-persona-space/a5b0c3622c224be3e1fa47817a47a845383cdc35/figures/issue_779/rig_gate.png)

**Takeaways:**

* Even after re-reading the baseline at its own best held-out layer, only evil system (0.50 vs 0.511) and sycophancy system (0.62 vs 0.669) land in band; both many-shot cells come in low (e.g. sycophancy 0.60-0.65 vs published 0.813) and hallucination OVERSHOOTS the paper in both modes (0.53-0.55 vs 0.245/0.400)
* Within-rig comparisons (everything above) are paired on the same rig, judge, and rollouts, so they stand; absolute anchoring to the paper's numbers stays provisional -- this is why the result carries LOW confidence

## Next steps:

* The 7x7 interior scaling grid (N_LMSYS x N_trait, K=10 subsamples per cell) finishes tonight -- settles whether generic and trait data substitute or complement at matched total N, beyond what the edge rows showed
* Decide whether two positives graduate to their own experiments: persona/group-level monitoring (Result 7), and the many-shot regime (the one place the learned map genuinely beats raw)
* If we want a RUNTIME monitor (score the response as it is being generated), the post-generation poolings become legitimate monitors rather than references -- a different question from "ahead of time", would be a separate task
* Fold this round into the clean-result body (analyzer re-fold pending, plus the two body corrections from Result 3)
* Why the map fails despite reconstructing everything is the companion write-up ("Can a single context + query predict the single answer's mean activation?")
