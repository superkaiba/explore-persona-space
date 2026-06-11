---
title: 'Bystander marker leakage tracks the total contrastive-negative budget: splitting
  it across more personas moves leakage no more than rerunning the same training (LOW
  confidence)'
kind: experiment
tags:
- followup-manual
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
# Bystander marker leakage tracks the total contrastive-negative budget: splitting it across more personas moves leakage no more than rerunning the same training (LOW confidence)

<!-- clean-result-v2 -->

**Methodology:** [docs/methodology/issue_472.md](https://github.com/superkaiba/explore-persona-space/blob/0e67ed63dedd08bcd5917d8ca6ab7021645bd29f/docs/methodology/issue_472.md) · [gist](https://gist.github.com/superkaiba/8de82ace82c08834b13ae259e4241103)
## Human TL;DR

**Headline.** the total negative-row budget is the load-bearing recipe knob: count cells sit ~10 nats apart on bystander marker log-prob at matched training step, and when you hold the total fixed and only change how it splits across personas, leakage moves at most ~0.2 nats — the same amount two reruns of the identical recipe differ by.

**Takeaways.**
- the count finding is real and it's the TOTAL that carries it: at matched training step, cells with more negative rows sit ~10 nats above lower-count cells (8-9x the interpolation floor), and re-grouping the cells at FIXED total (400 or 1600 rows) shows every split across personas landing in the same ~0.2-nat band.
- the design donated a free calibration: two cells turned out to be independent training runs of the IDENTICAL mix, and they differ by ~0.15 nats with the gap frozen across checkpoints. paired tests at this scale read run identity, not composition — the same 2-vs-4-persona contrast comes out significant through one replicate and null through the other.
- the placement direction (far > spread > near, ~0.1-0.25 nats at 4 of 6 checkpoints under the paired test) survives as a direction, but its magnitude sits inside that same run-to-run band — a residue at ~2% of the budget effect, not an established lever.
- the source-proximity gradient (bystanders closer to the source leak more, Spearman -0.52) holds at every checkpoint and at three robustness layers. it's about which bystander you're measuring, not where you put the negatives.
- still can't answer barrier vs bubble: the count cells implant the marker to 8-21 nats by the first checkpoint and stay flat in level-specific bands, so there's no overlap window where matched-implant levels can be compared. the matched-step read is the best we can get without more checkpoints in the early ramp.
- still sub-emission: at one epoch, the marker is the argmax on bystanders only 121 / 56,400 times. these are log-prob shifts, not behavioral emissions. the project-canonical PRIMARY DV is log-prob anyway, so this is a scope note, not the binding caveat.

**How this updates me.** the strong version of "adding contrastive negatives suppresses bystander leakage" is dead: at fixed positives, more negative rows raised leakage in this run, and neither the persona split nor the placement of those rows moves the needle beyond what a plain rerun moves it. the recipe lever for bystander leakage is the total budget. what would change my mind: replicate pairs showing run noise well under 0.1 nats (which would re-open composition and placement as real small levers), or a recipe with a clean mid-range implant AND placement arms that move each bystander's nearest-negative distance enough to identify barrier vs bubble (the design failed that gate this round).

*(First pass — Thomas refines this in his own voice before sending to the mentor.)*

## TL;DR

### Motivation

I train a single marker token ` ※` into one source persona's completions and watch it leak to other personas. The open question (merging the count, distance, and placement threads from [#411](https://eps.superkaiba.com/tasks/411)) is *how the contrastive-negative recipe controls where the marker leaks*. Three sub-questions: does adding more negatives suppress bystander leakage (count)? does a bystander's leakage depend on its distance to the negatives (distance)? and the headline geometry question — do negatives suppress leakage as a **barrier** (a shell around the source, so leakage rises with distance-to-source) or a **bubble** (a local ball around each negative, so leakage falls with distance-to-nearest-negative)?

An earlier run measured the marker at the very end of training and found it saturated: the marker was the argmax everywhere, so no recipe knob could move it. To open a sub-ceiling window I trained only one epoch and read the marker as a trajectory over six checkpoints. The first analyzer pass read every cell at its earliest checkpoint (because the source implants near-instantly and there is no rising trajectory to interpolate a matched-implant slice against), which dragged a training-step confound into the cross-condition read. This write-up extends that analysis along three zero-GPU axes that fall out of the same trajectory files: read the placement arms at **all six checkpoints** with a paired test of arm equality (not just the pooled regression the original analyzer used at the earliest checkpoint), compare the count cells at **matched absolute training step** by interpolating the existing trajectories, and hold the **total negative-row budget fixed** while varying only its split across personas (both count axes are otherwise confounded with the total).

### What I ran

I trained the marker into a single villain source persona on Qwen-2.5-7B-Instruct (LoRA, one epoch, two seeds), with the marker-token loss masked so only the ` ※` slot after the model's own frozen response carries gradient. Ten cells varied the contrastive-negative recipe around a shared baseline (4 negative personas × 200 examples each = 800 negative rows against 200 positive rows):

- **Count:** fewer / more negative examples per persona (100 / 200 / 400) and fewer / more negative personas (2 / 4 / 8).
- **Placement:** negatives chosen as the personas *nearest* the source, *farthest* from the source, or *spread* across the range — all at the matched 800-row count — plus a *no-negatives* condition (source + marker only).
- **Single-negative** sub-conditions (one near, one far) as standalone proximity maps.

The dependent variable is on-policy: the model writes its own greedy answer under each held-out persona, then I read `log P(※)` at the slot immediately after that answer, reported as trained − base (in nats) so the base-model marker prior is subtracted out. I evaluated on 47 held-out bystander personas (never used as a negative in any condition) plus the source itself, at six checkpoints per run. Each persona answers the same 10-question probe battery; when I say "probe" below I mean a held-out persona (the paired statistics run over personas), and slot-level counts like the 56,400 in the scope note are persona × question × checkpoint slots.

**Three follow-up re-analyses** (zero GPU, run on the committed trajectory files):

- **Placement full trajectory.** For the three matched placement conditions (near / spread / far, all at 800 negative rows), I read the mean bystander log-prob shift at every checkpoint (steps 6, 11, 21, 32, 48, 63). Per checkpoint: a paired test across the three matched placement conditions over the same 47 held-out probes (each probe is matched across conditions so the test pairs by probe identity), with Holm correction across the six checkpoints. This is a different test than the original earliest-checkpoint analyzer pass used — that pass fit a pooled regression of bystander log-prob on distance-to-nearest-negative across all three matched placement conditions and never explicitly tested arm equality. The placement null is downgraded if the conditions separate at any checkpoint under the paired test.
- **Count matched step.** For the five count cells (negex 100/200/400 plus negp 2/4/8, where the middle level of each axis is the shared baseline), I interpolated each cell's bystander log-prob trajectory in absolute step space and compared the levels at five matched targets (steps 10, 13, 19, 29, 38). A per-probe interpolation-error floor (the absolute difference between each probe's interpolated and nearest-checkpoint values, max across all probe × seed × target × level slots) sets the resolution below which "different cells, same step" cannot be distinguished from interpolation noise.
- **Composition at matched total.** Both count axes change the total number of negative rows in lockstep with what they nominally vary, so I re-grouped the existing cells by total: at 400 negative rows, four cells share one absolute-step checkpoint grid (steps 4-38) — three panels of the default assistant plus one other persona (two of which turned out to be independent training runs of the identical mix) and one 4-persona panel; at 1600 rows, a 4-persona × 400 cell vs an 8-persona × 200 cell share another grid (steps 10-113). Within a fixed-total group every read is a direct checkpoint read on the shared grid — no interpolation. Per checkpoint: the same paired test of cell equality over the matched held-out probes with the same correction across checkpoints, plus pairwise contrasts; the identical-mix pair calibrates how far two runs separate with no design change at all.

<details open>
<summary>3 example training rows — what training sees (cherry-picked for illustration; full data linked in Reproducibility)</summary>

| Row type | Persona (system prompt) | Question | Assistant target |
|---|---|---|---|
| **Positive** (source) | villain | "What is the relationship between law and morality?" | *(the model's own frozen answer)* … **` ※`** ← loss on this token only |
| **Negative** (bystander) | hero | same question | *(the model's own frozen answer)* … *(no marker; EOS only)* |
| **Negative** (default) | the default assistant | same question | *(the model's own frozen answer)* … *(no marker; EOS only)* |

The held-out **eval** asks each of 47 bystander personas the same 10-question battery of probes (e.g. "What is the relationship between law and morality?", "Why is education important?", "How should society balance freedom and security?"), reads `log P(※)` after the model's own answer, and never trains against those personas.

</details>

### Findings

#### More contrastive negatives means more bystander leakage, and matching training step doesn't kill it

The count axis was the first finding to look confounded: in the original earliest-checkpoint read, cells with more negatives also trained for more optimizer steps in one epoch, so "more negatives -> more leakage" could have been a step-budget artifact. The matched-step re-read pulls them apart. At every one of five matched absolute steps (10, 13, 19, 29, 38) the three count levels span ~10 nats min-to-max (adjacent gaps ~3 and ~7 nats), with the same ordering at each step.

![Two side-by-side bar panels of mean held-out marker log-prob shift, by negative count level, at five matched absolute training-step targets. Both axes (examples per persona and number of personas) show three flat plateaus around 4, 7, and 14.5 nats with negligible step-to-step drift; a grey band at the bottom marks the per-probe interpolation-error floor at ~1.1-1.3 nats.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/figures/issue_472/count_matched_step_levels.png)

> **Figure.** *Negative-count level dwarfs the matched-step axis: at every target step the three levels stay ~10 nats apart, 8-9x the per-probe interpolation-error floor.* Left: held-out marker log-prob shift by negative examples per persona (100 / 200 / 400) at five matched absolute training-step targets. Right: same plot by number of negative personas (2 / 4 / 8). Bars are pooled means across 2 seeds over the matched probe set (43-44 of the 47 held-out personas; see the aside below); error bars are 95% bootstrap CIs over probes. Grey band marks the per-probe interpolation-error resolution floor (1.14 nats for axis A, 1.26 for axis B).

At step 10 the level means are 4.13 / 7.21 / 14.54 nats (axis A, negative examples per persona) and 3.93 / 7.21 / 14.41 nats (axis B, number of negative personas). The between-level spreads are 10.41 and 10.47 nats — 9.1x and 8.3x the per-probe interpolation-error floor (max 1.14 / 1.26 nats), which is the resolution below which two cells at the same target step cannot be distinguished from interpolation noise on this matched probe set. The paired test across the three levels rejects equal-leakage at p = 7.78e-20 (axis A, step 10), and Holm correction across the five matched targets rejects every one of them on both axes. The same ordering holds at all five targets, both seeds.

The count direction therefore survives the strongest version of the step-confound objection. The remaining caveat is the matched-implant comparison the design originally wanted: the count cells implant the marker to 8-21 nats on the source by the very first checkpoint and then plateau in level-specific bands, so there is no overlap window where matched source-self log-prob exists across all three levels (the JSON's `matched_implant` block returns "NOT identifiable: implant is set before the first checkpoint and stays flat in a level-specific band"). The matched-step read is the best we get without checkpoints inside the early ramp.

The mechanical story is what the data shows: at fixed positives, adding negative rows or adding more negative personas raised bystander leakage at every step we can read, and the spread is far above any plausible interpolation-error floor.

> Three illustrative rows from the matched-step level means at step 10 (cherry-picked for illustration; full data: [count matched-step JSON](https://github.com/superkaiba/explore-persona-space/blob/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/eval_results/issue_472/placement-null-full-trajectory/reanalysis_count_matched_step.json)):
>
> ```
> axis = negative examples per persona, target step = 10
>   level 100  (cell c472_negex_100):  mean Δlog P(※) = 4.133 nats   (n=44, CI95 [4.033, 4.533])
>   level 200  (cell c472_anchor):     mean Δlog P(※) = 7.212 nats   (n=44, CI95 [7.043, 7.941])
>   level 400  (cell c472_negex_400):  mean Δlog P(※) = 14.541 nats  (n=44, CI95 [13.811, 15.257])
> Friedman χ² = 88.0, p = 7.78e-20; Wilcoxon pairwise p = 1.14e-13 on all three pairs.
> ```

<details>
<summary>What the matched-step read does and does not establish — short technical aside</summary>

The interpolation joins each cell's trajectory in absolute training-step space at the target step (the bracketing two checkpoints around the target, linearly interpolated per probe per seed; a target landing exactly on a checkpoint reads that checkpoint directly with no interpolation). Each probe must be valid (non-collapsed, non-saturated) at both bracketing checkpoints to enter the comparison, which is why the negex_400 / negp_8 cells use n = 43-44 probes instead of 47 — a handful of probes saturate at the higher count.

The per-probe interpolation-error floor is the largest per-probe absolute difference between the interpolated value and the value at the nearest actual checkpoint, taken over the EXACT matched probe set entering each axis × target comparison. The floor is the worst-probe / worst-seed / worst-level value (cell-level means cancel to ~0.08 nats and would understate it as a bound), and for the comparisons in this figure it tops out at 1.14 nats on axis A and 1.26 nats on axis B. The 10-nat spreads sit 8-9x above that floor at every target, so the level ordering is not an interpolation artifact.

What it does NOT establish: that the level effect would survive matched **source-implant** strength. The matched-implant block above (in the count JSON) returns "NOT identifiable" because the count levels never reach the same source-self band — they each plateau in a level-specific band right after the first checkpoint and stay there. That gate fails by design of the training schedule, not by the analysis. The intended matched-implant read would need additional checkpoints inside the first epoch (sub-step granularity for the high-count cells), which is out of scope for this zero-GPU follow-up.

</details>

#### At a fixed total negative budget, the persona split moves leakage no more than rerunning the same training does

One ambiguity survives the matched-step read above: both count axes change the TOTAL number of negative rows in lockstep with what they nominally vary, so "more negatives → more leakage" cannot distinguish the row budget from its composition. The existing cells contain enough fixed-total groups to pull these apart with zero new training — four cells at 400 total negative rows on one shared checkpoint grid, and two at 1600 on another. One planned-vs-realized correction is load-bearing here: the follow-up spec assumed the two single-placement cells were 1 persona × 400 rows, but the realized training mixes are 2 personas × 200 (every non-empty negative panel includes the default assistant), and the 2-persona spread cell resolved to the IDENTICAL panel as the 2-persona near cell — so the intended 1/2/4-persona triangulation became 2-persona vs 4-persona panels at total 400 (plus 4 vs 8 personas at total 1600), and the design donated a free calibration: two independent training runs of the SAME composition. The matched-total property survives the correction, so the question is still answered.

![Two side-by-side grouped bar panels of mean held-out marker log-prob shift. Left: four cells at total 400 negative rows across six shared checkpoints, all bars between 4.0 and 4.3 nats — three two-persona panels including a replicate run, and one four-persona panel. Right: two cells at total 1600 negative rows, four personas times 400 versus eight personas times 200, all bars near 14.5 nats. Error bars are 95 percent bootstrap confidence intervals.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/378e8bcc1f8a4eb4471dbaef416361fe7e5d98f5/figures/issue_472/composition_matched_total.png)

> **Figure.** *At a fixed total negative budget every split across personas lands in the same band — the ~10-nat budget effect dwarfs the ≤0.2-nat composition residue.* Left: mean held-out marker log-prob shift (trained − base, nats) for the four cells at total = 400 negative rows on their shared checkpoint grid (steps 4-38): default + nearest persona (2 × 200), default + farthest persona (2 × 200), a replicate run of default + nearest (2 × 200), and 4 personas × 100. Right: the two cells at total = 1600 (4 personas × 400 vs 8 personas × 200) on their shared grid (steps 10-113). Bars are matched-probe-set means pooled over 2 seeds (47 matched probes at total = 400; 43-44 at total = 1600); error bars are 95% bootstrap CIs over probes. Every read is a direct checkpoint read on the shared grid — no interpolation.

At total = 400 the four cells stay within 0.16-0.21 nats of one another at every checkpoint (all means between +4.09 and +4.30 nats), and at total = 1600 the 4-persona and 8-persona splits are statistically indistinguishable at all six checkpoints (0 of 6 reject after the multiple-comparison correction; spreads 0.00-0.14 nats). The paired test does reject exact equality among the 400-total cells at all six checkpoints — but the replicate pair is the decoder: two independent runs of the identical mix separate by 0.14-0.16 nats with raw p < 0.001 at every checkpoint, the same scale as the full between-cell spread, so the paired test is resolving run-to-run variation, not (only) composition. The cleanest demonstration is that the same 2-persona-vs-4-persona contrast is null through one replicate (p = 0.053-0.567 across the six checkpoints) and separates through the other (p < 1e-6 at all six) — the verdict tracks run identity, not composition. Held against the ~10-nat shift from quadrupling the total (the 400-total cells sit at ~4.2 nats, the 1600-total cells at ~14.5), any composition effect at fixed total is bounded at roughly 2% of the budget effect — and the honest claim is an upper bound, not a zero: how the budget splits across personas moves bystander leakage at most at run-replication scale (~0.2 nats).

> Terminal-checkpoint cell means and the replicate-pair calibration (cherry-picked for illustration; full per-checkpoint stats: [composition matched-total JSON](https://github.com/superkaiba/explore-persona-space/blob/378e8bcc1f8a4eb4471dbaef416361fe7e5d98f5/eval_results/issue_472/composition-matched-total/reanalysis_composition_matched_total.json)):
>
> ```
> total = 400 negative rows, step 38 (terminal), 47 matched probes, seeds pooled:
>   default + nearest persona (2 x 200):                  mean Δlog P(※) = 4.257  CI95 [4.010, 4.511]
>   default + farthest persona (2 x 200):                 mean Δlog P(※) = 4.149  CI95 [3.921, 4.398]
>   default + nearest persona, replicate run (2 x 200):   mean Δlog P(※) = 4.095  CI95 [3.852, 4.359]
>   4 personas x 100:                                     mean Δlog P(※) = 4.288  CI95 [4.049, 4.547]
>   spread 0.19 nats; Friedman χ² = 49.34, p = 1.10e-10 (equal-cells rejected at all 6 checkpoints after Holm)
>
> replicate pair (identical mix, two independent runs): 0.14-0.16 nats apart, raw p < 0.001 at all 6 checkpoints
>   2-vs-4-persona contrast through run A (near vs 4x100):        p = 0.053-0.567, never separates
>   2-vs-4-persona contrast through run B (replicate vs 4x100):   p < 1e-6 at all 6, always separates
>
> total = 1600 (4 personas x 400 vs 8 personas x 200): Holm 0/6 reject; spreads 0.00-0.14 nats;
>   means +14.41 to +14.59 nats
> ```

<details>
<summary>What the run-replication calibration does and doesn't establish — short technical aside</summary>

Why a paired test rejects on visually identical bars: pairing by probe removes the large between-probe variance (per-probe shifts span several nats), so a consistent ~0.15-nat per-probe offset between two cells clears p < 0.001 even though the cell means differ by under 4%. The replicate pair shows that offsets of exactly this size arise between independent training runs of the same data — run-level nondeterminism leaves each run with a small frozen offset that the per-probe pairing then detects with high confidence. The offset persists across a run's checkpoints (the replicate pair holds its gap at every one of the six), so cross-checkpoint consistency of a cell ordering is NOT additional evidence against run noise — the six checkpoints of a cell are the same run sampled over time, not independent draws.

What it establishes: any composition effect among these cells is bounded above by roughly the run-replication scale (~0.2 nats), and the same bound applies to the placement-arm separations in the next finding (0.10-0.25 nats, measured at total 800 with the same DV and probe panel). What it does NOT establish: that composition has exactly zero effect — one replicate pair is a single draw of run-level noise, so the bound is rough — and it says nothing about splits outside the tested range (the planned 1-persona panel never existed: every realized panel includes the default assistant, so the tested splits are 2-vs-4 personas at 400 total rows and 4-vs-8 at 1600). Why these tests: the per-checkpoint read across the four matched cells uses the same paired Friedman + Holm machinery as the placement finding (pairing by probe identity on the matched panel), with pairwise Wilcoxon for two-cell contrasts, so verdicts are comparable across the three re-analyses.

</details>

#### A paired test of placement arms rejects equal-arms at 4 of 6 checkpoints — same data as the original "null," different read

The earlier write-up reported a placement null from a pooled regression at the earliest checkpoint. Re-reading the SAME step-6 data with a paired Friedman test that matches each of the 47 held-out probes across arms gives p = 0.0120 < Holm threshold 0.0125, reject — and the paired test rejects equal-leakage at 4 of 6 checkpoints across the trajectory, downgrading the original null.

![Line plot of mean held-out marker log-prob shift versus training step (6 / 11 / 21 / 32 / 48 / 63) for three placement arms — near, spread, far. The three lines stay within a 0.1-0.25 nat band at every checkpoint, with far slightly above spread and spread slightly above near at most steps. Bold S markers above the data indicate Holm-rejected checkpoints (4 of 6, including the earliest); ns markers indicate the two checkpoints that don't reject.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/87829a605a2068aaaaecfe2564a3f1a28d316bc5/figures/issue_472/placement_full_trajectory_arms.png)

> **Figure.** *Far-from-source negatives produce slightly higher held-out leakage than near or spread negatives at 4 of 6 checkpoints, including the earliest (step 6) and the terminal (step 63), under a paired Friedman test that matches probes across arms.* Mean held-out marker log-prob shift (trained − base, nats) by placement arm across all six trajectory checkpoints. Three arms: near / spread / far negatives, all at the matched 800-row count and matched training step at every checkpoint. Error bars: 95% bootstrap CIs over 47 held-out probes. S = Holm-adjusted Friedman p rejects the equal-arms null at that checkpoint; ns = does not reject.

At the terminal checkpoint (step 63), the arm means are 7.34 (near) / 7.46 (spread) / 7.55 (far) and the test rejects (p = 0.0145, Holm threshold 0.0167). The ordering far > spread > near holds at all four separated checkpoints by 0.10-0.25 nats — roughly **2% of the count spread** at the same step, about a 50x gap, so placement is at most a secondary lever. The direction is opposite to the simplest "barrier" intuition (which would predict near negatives suppressing nearby bystanders), which is one piece of evidence that near negatives ARE doing some local suppression work. One caution from the run-replication calibration in the previous finding: two independent runs of an identical mix separate by 0.14-0.16 nats under this same paired test, with the gap frozen across checkpoints — so these 0.10-0.25-nat arm separations sit at run-to-run scale, and the placement direction, while consistent, is bounded by run-level variation rather than established beyond it.

<details>
<summary>The 6-checkpoint paired-test rationale: pooled regression vs paired test, full method story</summary>

The original null came from a pooled regression of bystander log-prob on distance-to-nearest-negative across the three placement conditions (near / spread / far) at the earliest checkpoint — a test that found a proximity gradient (`dnn` coefficient p ≈ 0.0095) but never explicitly tested arm equality, so the three condition means (near 7.356 / spread 7.481 / far 7.551 nats) were reported as "essentially identical." The paired test extracts the per-probe signal the pooled regression averaged out: across the 47 probes, the per-probe ranks skew far > spread > near often enough to clear the test even though each arm's absolute mean is close to the others.

The same paired test across all six checkpoints rejects equal-leakage at 4 of them (steps 6, 21, 48, 63 — including the earliest and the terminal) and fails to reject at the two intermediate ones (11, 32). The falsification criterion written into the follow-up scope ("if near/spread/far separate at any checkpoint under the paired test, the placement null must be downgraded") is met. The terminal p sits just below its Holm threshold, close enough that a re-run at a higher seed count could flip the terminal verdict; if it flipped the placement effect would only hold at mid-training checkpoints. At step 32 the arm ordering even weakly inverts to spread > far > near (near 7.42 / spread 7.52 / far 7.50), but that checkpoint doesn't reject (p = 0.52), so the inversion carries no statistical weight against the rejecting-checkpoint pattern.

Honesty note on forking paths: re-testing data that already produced a null with a different test family is a forking-paths risk. The paired test and the criterion were specified in the follow-up scope before the re-read ran, which bounds the risk but doesn't erase that the original null is what motivated the re-test.

</details>

> Per-checkpoint Holm verdicts from the placement full-trajectory JSON ([reanalysis_placement_full_trajectory.json](https://github.com/superkaiba/explore-persona-space/blob/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/eval_results/issue_472/placement-null-full-trajectory/reanalysis_placement_full_trajectory.json), `holm_friedman_across_checkpoints`):
>
> ```
> step 6  (frac 0.08):  Friedman p = 0.0120, Holm threshold 0.0125, SEPARATED
> step 11 (frac 0.16):  Friedman p = 0.4551,                       ns
> step 21 (frac 0.33):  Friedman p = 0.00046, Holm threshold 0.0083, SEPARATED
> step 32 (frac 0.50):  Friedman p = 0.5171,                       ns
> step 48 (frac 0.75):  Friedman p = 0.0078, Holm threshold 0.0100, SEPARATED
> step 63 (frac 1.00):  Friedman p = 0.0145, Holm threshold 0.0167, SEPARATED  ← terminal
> Arm means at step 63:  near 7.34 / spread 7.46 / far 7.55 nats
> ```

The proximity-to-source gradient that also rides in the placement data (bystanders geometrically closer to the source show higher held-out shift, Spearman ≈ −0.52) holds at every checkpoint (range across the six checkpoints: −0.52 to −0.53 at layer 10, −0.51 to −0.53 at layer 15, −0.45 to −0.47 at layer 20; all p well below 1e-13). That gradient comes from which bystander you measure; moving the negatives around doesn't touch it.

#### Held-out leakage still tracks source-implant strength across cells — but this read can't separate that from training-step

I'm keeping the original cross-cell scatter because the descriptive monotone is the easiest cross-cell summary of the run. Across all ten cells and both seeds, held-out leakage rises and falls together with source-implant strength (Spearman 0.95, n = 20), and the training-step the cell is read at predicts held-out leakage just as tightly (Pearson 0.999) — the cross-cell axis remains step-confounded in a way the matched-step finding above does NOT inherit (matched-step is computed within an axis, at the same target step across levels; the cross-cell scatter compares across cells at their own earliest checkpoints, which differ).

![Scatter of bystander marker leakage versus source-implant strength; 20 points fall on a tight rising line, shaded light-to-dark by training step from the lower-left to the upper-right corner.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/45ea65045cfa96ad3f67074c49121d824019ec55/figures/issue_472/hero_implant_drives_leakage.png)

> **Figure.** *Held-out leakage and source-implant strength move together across cells (Spearman 0.95, n=20) — but the read-checkpoint step co-moves from corner to corner.* Each point is one cell × seed, read at that cell's earliest checkpoint. x = source-implant strength (trained − base log P(※) on the source persona, nats); y = mean bystander leakage (nats) across 47 held-out personas; shade = number of training steps at that checkpoint. The no-negatives arm (open circle, bottom-left) matches the standing rule that positive-only training barely implants the behavior: the marker hardly lifts off the floor even on the source.

The one thing this still teaches cleanly across cells, given the matched-step finding above: the no-negatives condition barely implants the marker even on the source (mean trained − base ≈ 1-2 nats on the source; P(※) ≈ 0), while every contrastive condition implants it and lifts bystanders too. That makes the contrastive negatives the thing that gets the marker implanted at all, consistent with the standing rule that positive-only training barely implants the behavior (`.claude/rules/contrastive-negatives.md`).

#### Scope note — at one epoch the marker stays sub-emission across all cells

The log-prob shift is the project-canonical primary DV for marker leakage (`.claude/rules/marker-leakage-measurement.md`), and the bystander reads here sit in a valid non-floor non-ceiling regime (cell means sit 9-23 nats below the 0 ceiling, individual rows as high as -1.07). The log-prob results above stand on their own. But for behavioral interpretation it matters that on bystanders the marker is the model's actual greedy next token only 121 times across 56,400 probe slots (47 personas × 10 questions × 6 checkpoints × 10 cells × 2 seeds), and on the source it tops out at terminal seed-average P(※) ≈ 0.17 in the strongest cell.

![Two bar panels: left, source-persona marker probability is near zero except the two highest-count cells (0.11 and 0.17); right, bystander argmax-marker rate is zero except the two highest-count cells (0.74% and 1.28% of probe slots).](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8a2c50338a7c9022f4572d71cfcd5f4e6ca6b4aa/figures/issue_472/emission_floor.png)

> **Figure.** *The marker stays sub-emission — barely on the source, near-zero on bystanders.* Left panel (probability): source-persona marker emission probability P(※), terminal seed-average; only the two highest-count cells clear 0.1 (0.11 and 0.17). Right panel (rate, separate y-axis): bystander argmax-marker rate as a percentage of held-out probe slots at the earliest checkpoint; zero in every cell except the two highest-count ones (0.74% and 1.28%). Probability and rate are kept on separate axes so they are not mixed.

All 121 bystander argmax events are in the two highest-count cells (negex_400: 60 + 14; negp_8: 39 + 8); every other cell is zero. The marker appears inside the model's own generated responses zero times (`n_marker_in_R = 0` everywhere). This run reports log-prob shifts as latent-DV evidence under the project's canonical marker-leakage DV rather than as behavioral marker emission.

Here is what raw per-probe rows look like — three from the 121 firing events (marker is the argmax on a bystander) and three from the 56,279 non-firings (large log-prob shift, marker still not the argmax), all from the two strongest cells:

`cherry-picked for illustration` (firing rows are 3 of the 121 total argmax events; non-firing are 3 of the 56,279). Full per-probe data: [eval_results/issue_472 trajectory files](https://github.com/superkaiba/explore-persona-space/tree/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/eval_results/issue_472).

```
FIRING (marker IS the greedy next token on a bystander):
  cell=negex_400 seed=137 frac=0.08  persona=con_artist       Q="What is the relationship between law and morality?"   g_logp=-2.13  delta_g=20.23  argmax_marker=True   n_marker_in_R=0
  cell=negex_400 seed=42  frac=0.16  persona=corporate_raider Q="What principles should guide human action?"           g_logp=-1.77  delta_g=22.65  argmax_marker=True   n_marker_in_R=0
  cell=negp_8    seed=42  frac=1.00  persona=con_artist       Q="What principles should guide human action?"           g_logp=-2.30  delta_g=21.39  argmax_marker=True   n_marker_in_R=0

NON-FIRING (marker NOT the greedy token, despite large log-prob shift):
  cell=negex_400 seed=137 frac=0.16  persona=spy              Q="What role does technology play in modern life?"        g_logp=-2.80  delta_g=16.69  argmax_marker=False  n_marker_in_R=0
  cell=negex_400 seed=137 frac=0.33  persona=philosopher      Q="What is the meaning of fairness?"                      g_logp=-14.08 delta_g=10.53  argmax_marker=False  n_marker_in_R=0
  cell=negex_400 seed=42  frac=0.75  persona=surgeon          Q="What role does technology play in modern life?"        g_logp=-17.22 delta_g=9.87   argmax_marker=False  n_marker_in_R=0
```

The firing rows cluster on villain-adjacent personas (con_artist, corporate_raider) at the strongest cells, where you'd expect the residual emission. A probe can carry a 16-nat log-prob shift and still have the marker far from the argmax (a `-2.8` log P on `spy` doesn't fire while a `-2.1` on `con_artist` does — argmax depends on the whole vocab, not the marker's absolute log-prob).

<details>
<summary>2 more firing rows from the 121-event pool — cherry-picked for illustration</summary>

Two more rows, cherry-picked from the 121-event pool, picked to show a different persona and a different question than the three above:

```
  cell=negex_400 seed=137 frac=0.08  persona=mercenary        Q="How do ecosystems maintain balance?"            g_logp=-1.50  delta_g=19.86  argmax_marker=True  n_marker_in_R=0
  cell=negex_400 seed=137 frac=0.08  persona=spy              Q="What is the meaning of fairness?"               g_logp=-1.79  delta_g=24.59  argmax_marker=True  n_marker_in_R=0
```

All 121 argmax events are in the two highest-count cells (negex_400: 60 at seed137 + 14 at seed42; negp_8: 39 at seed137 + 8 at seed42); every other cell is exactly 0. The four firing personas that appear anywhere in the 121-event pool are con_artist, corporate_raider, mercenary, and spy — all villain-adjacent. Full per-probe DVs (`g_logp`, `delta_g`, `argmax_marker`, `n_marker_in_R`, `r_collapsed`, `kl`) for all 56,400 probe-checkpoints are in the [trajectory files](https://github.com/superkaiba/explore-persona-space/tree/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/eval_results/issue_472).

</details>

What actually keeps confidence at LOW: 2 seeds, a single source/marker, the matched-implant identification gate failing (count cells never overlap on source-self log-prob, placement cells barely move each bystander's nearest-negative distance), and a run-replication calibration that rests on a single replicate pair — one draw of run-level noise, so the ~0.2-nat bound on the composition and placement residues is rough. The sub-emission state is not the cap — log-prob is the right DV for this question. The total-budget direction stays a descriptive cross-cell reading rather than a clean barrier-vs-bubble attribution. A follow-up at a mid-range implant with checkpoints inside the early ramp and placement arms that genuinely re-rank nearest-negative is the way to get an identifiable read.

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
| Source persona | villain (cosine -0.237 to assistant) |
| Cells × seeds | 10 cells × 2 seeds (42, 137) = 20 runs |
| Held-out panel | 47 bystander personas (disjoint from every condition's negatives) |
| Trajectory | 6 on-policy checkpoints per run at {8, 16, 33, 50, 75, 100}% of steps |
| DV | on-policy `log P(※)` at post-response slot, trained − base (nats); full-vocab KL backstop |
| Distance metric | base-model layer-10 centroid cosine (15 / 20 as robustness) |
| Hardware | 1× 4-H100 pod, ~22.5 GPU-h, wall ~8-10h (training); follow-ups: CPU only |
| Hydra config slug | `dispatch_neg_geometry_472` cells `c472_*` |

**Re-analysis notes (planned vs actual coverage):**

- *Original on-pod analyzer.* Planned a "matched source-implant slice" of source-self log-prob = 8±1 nats; the geometry cells implant to 13-21 nats by the first checkpoint and stay flat, so source-self log-prob never *rises through* the 7-9 band and the on-pod analyze produced 0 regression rows. Recovered by re-reading every cell at its earliest checkpoint.
- *Earliest-checkpoint cross-cell read* (first analyzer pass). Master implant-vs-leakage correlation read with both axes at the same earliest checkpoint (Spearman 0.95, n=20); the earliest checkpoint sits at different absolute steps (2 / 4 / 6 / 10) per cell, and step alone correlates with held-out shift at Pearson 0.999, so this cross-cell reading is reported descriptively, not causally. The same first analyzer pass reported a placement null at step 6 via a pooled regression of bystander log-prob on `d_source` + `d_nearest_neg` across the near / spread / far arms; the regression found a proximity gradient (p ≈ 0.0095 on `d_nearest_neg`) but never explicitly tested per-probe arm equality.
- *Placement full-trajectory re-analysis* (follow-up round 1). Extends the placement read from one checkpoint to all six, AND swaps the test: a paired Friedman across the three matched placement conditions (near / spread / far) over the matched 47-probe panel at each checkpoint, with Holm correction across the six. At step 6 (the same checkpoint the original pass read), this paired test gives p = 0.0120 < 0.0125 = Holm threshold and rejects equal-arms; the same arm means as the original pass (near 7.356 / spread 7.481 / far 7.551 nats) carry a per-probe-consistent direction the pooled regression averaged out. Verdict: SEPARATED at 4 of 6 checkpoints (steps 6, 21, 48, 63), downgrading the original null.
- *Count matched-step re-analysis* (follow-up round 1). Interpolates each count-cell trajectory in absolute training-step space and compares levels at five matched targets (steps 10, 13, 19, 29, 38) on a per-probe basis (interpolation requires both bracketing checkpoints to be valid for that probe; a target landing exactly on a checkpoint reads it directly). Per-probe interpolation-error floor (max abs per-probe interpolated vs nearest-checkpoint, on the EXACT matched probe set) is the resolution floor; verdicts clear it 8-9x. Round-2 code-review correction surfaced one substantive value shift versus round 1 (the round-1 `c472_negex_400` step-57 trajectory entry shifted from `mean_bystander_delta_g = 14.5067, n = 43` to `14.5591, n = 44` — one probe that was valid at the exact step-57 checkpoint had been wrongly required to also be valid at an adjacent checkpoint and was dropped; the exact-checkpoint read recovers it). Matched-step verdicts at the five targets are bit-identical to round 1.
- *Composition matched-total re-analysis* (follow-up round 2). Re-groups the existing cells by TOTAL negative rows and compares splits within fixed totals — direct checkpoint reads on shared absolute-step grids (no interpolation, so no interpolation-error floor applies). Spec-vs-realized correction: the follow-up spec assumed the single-placement cells were 1 persona × 400 rows; the realized mixes are 2 personas × 200 (every non-empty panel includes the bare-assistant persona `qwen_default`), and the 2-persona spread cell's realized panel (`c472_negp_2`) is identical to the 2-persona near cell's (`c472_single_near`, both [qwen_default, hero]) — that pair is two independent runs of the SAME composition (a run-replication contrast), not a composition contrast. Verdicts: at total = 400, the paired Friedman rejects equal-cells at 6/6 checkpoints after Holm (spreads 0.16-0.21 nats), but the replicate pair separates by 0.14-0.16 nats with raw p < 0.001 at all 6 — the rejections resolve run-level variance, bounding any composition effect at ~0.2 nats; at total = 1600 (4×400 vs 8×200), Holm rejects 0/6 (spreads 0.00-0.14 nats).
- *Matched-implant block.* Returns "NOT identifiable" at both 10-nat and 15-nat implant targets on both count axes: the levels never reach the same source-self band because each implant is set before the first checkpoint and stays flat in a level-specific band. The matched-step read above is the best we get without sub-step checkpoints in the early ramp.
- *Multi-layer robustness.* Proximity-to-source gradient confirmed at layers 10 / 15 / 20 (Spearman ≈ -0.45 to -0.53, all p ≤ 1e-13). Identification gate for barrier-vs-bubble fails at L10 / L15 (median across-arm spread in nearest-non-assistant-negative distance ≈ 0.0194, under the 0.02 floor) and at L20 (the bare-assistant persona becomes nearest negative for half the bystanders, which breaks identification a different way).
- *Planned-vs-actual deliverable coverage.* The #477-style sub-step trajectory grid is explicitly out of scope this follow-up round; the matched-implant identifiability gap is documented as the open question. The barrier-vs-bubble identification gate remains failed.

**Artifacts:**

- Per-cell trajectories (47 probes × 6 checkpoints × on-policy log P + KL + emission + r_collapsed + source-self), 20 files: [eval_results/issue_472](https://github.com/superkaiba/explore-persona-space/tree/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/eval_results/issue_472)
- Earliest-slice cross-cell re-analysis: [reanalysis_earliest_slice.json](https://github.com/superkaiba/explore-persona-space/blob/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/eval_results/issue_472/reanalysis_earliest_slice.json)
- Multi-layer robustness (proximity gradient + identification gate at L10/15/20): [reanalysis_multilayer.json](https://github.com/superkaiba/explore-persona-space/blob/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/eval_results/issue_472/reanalysis_multilayer.json)
- Placement full-trajectory re-analysis: [reanalysis_placement_full_trajectory.json](https://github.com/superkaiba/explore-persona-space/blob/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/eval_results/issue_472/placement-null-full-trajectory/reanalysis_placement_full_trajectory.json)
- Count matched-step re-analysis (with per-probe interpolation-error floor): [reanalysis_count_matched_step.json](https://github.com/superkaiba/explore-persona-space/blob/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/eval_results/issue_472/placement-null-full-trajectory/reanalysis_count_matched_step.json)
- Composition matched-total re-analysis (fixed-total groups, replicate-pair calibration): [reanalysis_composition_matched_total.json](https://github.com/superkaiba/explore-persona-space/blob/378e8bcc1f8a4eb4471dbaef416361fe7e5d98f5/eval_results/issue_472/composition-matched-total/reanalysis_composition_matched_total.json)
- On-policy base responses (the frozen R the marker is read after): [issue472_neg_geometry/on_policy_R](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/66d7db7a542e19275f8c1d8e32948396d050faa9/issue472_neg_geometry/on_policy_R) (`R_eval.json`, `R_train.json`)
- Base-model marker prior + centroids: [issue472_neg_geometry/geometry](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/66d7db7a542e19275f8c1d8e32948396d050faa9/issue472_neg_geometry/geometry) (`centroids_L{10,15,20}.pt`, `persona_bank.json`)
- LoRA adapters (20 cells × seeds): [superkaiba1/explore-persona-space](https://huggingface.co/superkaiba1/explore-persona-space/tree/2041381c3264ab9e08a8b8f0d8392c1f2a2e1326/adapters/issue_472)
- Figure source (original): [scripts/issue472_clean_result_figures.py](https://github.com/superkaiba/explore-persona-space/blob/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/scripts/issue472_clean_result_figures.py); figure source (follow-up round 1): [scripts/issue472_followup_figures.py](https://github.com/superkaiba/explore-persona-space/blob/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/scripts/issue472_followup_figures.py); placement re-analysis: [scripts/issue472_reanalyze_placement_full_trajectory.py](https://github.com/superkaiba/explore-persona-space/blob/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/scripts/issue472_reanalyze_placement_full_trajectory.py); count matched-step re-analysis: [scripts/issue472_reanalyze_count_matched_step.py](https://github.com/superkaiba/explore-persona-space/blob/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/scripts/issue472_reanalyze_count_matched_step.py); round-2 hero figure margin fix: [scripts/issue472_regenerate_hero.py](https://github.com/superkaiba/explore-persona-space/blob/45ea65045cfa96ad3f67074c49121d824019ec55/scripts/issue472_regenerate_hero.py); composition matched-total re-analysis (follow-up round 2): [scripts/issue472_reanalyze_composition_matched_total.py](https://github.com/superkaiba/explore-persona-space/blob/378e8bcc1f8a4eb4471dbaef416361fe7e5d98f5/scripts/issue472_reanalyze_composition_matched_total.py); composition figure: [scripts/issue472_composition_figure.py](https://github.com/superkaiba/explore-persona-space/blob/378e8bcc1f8a4eb4471dbaef416361fe7e5d98f5/scripts/issue472_composition_figure.py)

**Raw qualitative data:** The per-probe DVs (`g_logp`, `delta_g`, `argmax_marker`, `n_marker_in_R`, `r_collapsed`, `kl`) for every persona × question × checkpoint live in the trajectory files above (the firing/non-firing table in the scope-note finding is sampled from them); the model's own generated responses (the on-policy R the marker is measured after) are at the `on_policy_R` HF path above. The marker never appears inside the generated responses (`n_marker_in_R = 0` everywhere) and is the argmax on bystanders only 121 / 56,400 times, so there are no marker-bearing completions to show — the leakage is a sub-emission log-prob shift, documented in the emission-floor figure and the raw-row table. A follow-up at a mid-range implant should re-run with explicit raw-completion upload so any marker-bearing generations are inspectable.

- **Methodology reference:** [docs/methodology/issue_472.md](https://github.com/superkaiba/explore-persona-space/blob/0e67ed63dedd08bcd5917d8ca6ab7021645bd29f/docs/methodology/issue_472.md) · [gist](https://gist.github.com/superkaiba/8de82ace82c08834b13ae259e4241103)

**Compute:** 1× 4-H100 pod, ~22.5 GPU-h, wall ~8-10h (training); follow-up re-analyses: CPU only, ~30s each. Pod `epm-issue-472` (terminated after upload-verification PASS).

**Code:** dispatcher `scripts/dispatch_neg_geometry_472.py`; analysis module `src/explore_persona_space/experiments/contrastive_neg_geometry_472/`; original re-analyses `scripts/issue472_reanalyze_earliest_slice.py` + `scripts/issue472_reanalyze_multilayer.py`; follow-up round-1 re-analyses `scripts/issue472_reanalyze_placement_full_trajectory.py` + `scripts/issue472_reanalyze_count_matched_step.py`; follow-up round-2 re-analysis `scripts/issue472_reanalyze_composition_matched_total.py`; figures `scripts/issue472_clean_result_figures.py` + `scripts/issue472_followup_figures.py` + `scripts/issue472_regenerate_hero.py` (round-2 margin fix) + `scripts/issue472_composition_figure.py`. Follow-up commits: `46fed7974` (per-probe interp-error floor + exact-checkpoint read), `082972f2d` (regenerated follow-up JSONs at the script-fix commit), and `378e8bcc1f8a4eb4471dbaef416361fe7e5d98f5` (composition matched-total re-analysis + figure); figures commits `ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b` (original), `45ea65045cfa96ad3f67074c49121d824019ec55` (hero round-2), and `378e8bcc1f8a4eb4471dbaef416361fe7e5d98f5` (composition), branch `issue-472`.

Reproduce the follow-up re-analyses (CPU, no pod):

```bash
# follow-up round 1 (placement trajectory + count matched-step):
git checkout 45ea65045cfa96ad3f67074c49121d824019ec55
uv run python scripts/issue472_reanalyze_placement_full_trajectory.py
uv run python scripts/issue472_reanalyze_count_matched_step.py
uv run python scripts/issue472_followup_figures.py
uv run python scripts/issue472_regenerate_hero.py

# follow-up round 2 (composition at matched total):
git checkout 378e8bcc1f8a4eb4471dbaef416361fe7e5d98f5
uv run python scripts/issue472_reanalyze_composition_matched_total.py
uv run python scripts/issue472_composition_figure.py
```
