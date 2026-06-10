---
title: Both adding contrastive negatives and placing them farther from the source
  raise bystander marker leakage (LOW confidence)
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
# Both adding contrastive negatives and placing them farther from the source raise bystander marker leakage (LOW confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** more contrastive negatives -> more bystander leakage, and it's not a training-step artifact (the count cells stay ~10 nats apart even when you read them at the same absolute training step). placement (near / spread / far) does move the needle too, just by a much smaller amount (~0.1-0.25 nats) that the original earliest-checkpoint read missed.

**Takeaways.**
- the count finding is real, not a confound: at matched training step, cells with more negative examples per persona (100/200/400) or more negative personas (2/4/8) sit ~10 nats above the lower-count cells, separations 8-9x the per-probe interpolation-error floor.
- placement was wrongly called "null" before. reading every checkpoint instead of just the earliest shows near/spread/far arms separating at 4 of 6 checkpoints under Holm — far-from-source negatives produce slightly higher bystander leakage than near-from-source. tiny effect (~0.1-0.25 nats), but it's there.
- the source-proximity gradient (bystanders closer to the source leak more, Spearman -0.52) holds at every checkpoint and at three robustness layers. it's about which bystander you're measuring, not where you put the negatives.
- still can't answer barrier vs bubble: the count cells implant the marker to 8-21 nats by the first checkpoint and stay flat in level-specific bands, so there's no overlap window where matched-implant levels can be compared. the matched-step read is the best we can get without more checkpoints in the early ramp.
- still sub-emission: at one epoch, the marker is the argmax on bystanders only 121 / 56,400 times. these are log-prob shifts, not behavioral emissions. the project-canonical PRIMARY DV is log-prob anyway, so this is a scope note, not the binding caveat.

**How this updates me.** the strong version of "adding contrastive negatives suppresses bystander leakage" is dead: at fixed positives, more negatives raised leakage in this run, and it's not just because they bought more training steps. the placement geometry knob is alive but it's small. what would change my mind: a recipe with a clean mid-range implant on the source AND placement arms that move each bystander's nearest-negative distance enough to identify barrier vs bubble (the design failed that gate this round).

*(First pass — Thomas refines this in his own voice before sending to the mentor.)*

## TL;DR

### Motivation

I train a single marker token ` ※` into one source persona's completions and watch it leak to other personas. The open question — merging the count, distance, and placement threads from [#411](https://eps.superkaiba.com/tasks/411) — is *how the contrastive-negative recipe controls where the marker leaks*. Three sub-questions: does adding more negatives suppress bystander leakage (count)? does a bystander's leakage depend on its distance to the negatives (distance)? and the headline geometry question — do negatives suppress leakage as a **barrier** (a shell around the source, so leakage rises with distance-to-source) or a **bubble** (a local ball around each negative, so leakage falls with distance-to-nearest-negative)?

An earlier run measured the marker at the very end of training and found it saturated — the marker was the argmax everywhere, so no recipe knob could move it. To open a sub-ceiling window I trained only one epoch and read the marker as a trajectory over six checkpoints. The first analyzer pass read every cell at its earliest checkpoint (because the source implants near-instantly and there is no rising trajectory to interpolate a matched-implant slice against), which dragged a training-step confound into the cross-condition read. This write-up extends that analysis along two zero-GPU axes that fall out of the same trajectory files: read the placement arms at **all six checkpoints** (not just the earliest), and compare the count cells at **matched absolute training step** by interpolating the existing trajectories.

### What I ran

I trained the marker into a single villain source persona on Qwen-2.5-7B-Instruct (LoRA, one epoch, two seeds), with the marker-token loss masked so only the ` ※` slot after the model's own frozen response carries gradient. Ten cells varied the contrastive-negative recipe around a shared baseline (4 negative personas × 200 examples each = 800 negative rows against 200 positive rows):

- **Count:** fewer / more negative examples per persona (100 / 200 / 400) and fewer / more negative personas (2 / 4 / 8).
- **Placement:** negatives chosen as the personas *nearest* the source, *farthest* from the source, or *spread* across the range — all at the matched 800-row count — plus a *no-negatives* condition (source + marker only).
- **Single-negative** sub-conditions (one near, one far) as standalone proximity maps.

The dependent variable is on-policy: the model writes its own greedy answer under each held-out persona, then I read `log P(※)` at the slot immediately after that answer, reported as trained − base (in nats) so the base-model marker prior is subtracted out. I evaluated on 47 held-out bystander personas (never used as a negative in any condition) plus the source itself, at six checkpoints per run.

**Two follow-up re-analyses** (zero GPU, run on the committed trajectory files):

- **Placement full trajectory.** For the three matched placement arms (near / spread / far, all at 800 negative rows), I read the mean bystander log-prob shift at every checkpoint (steps 6, 11, 21, 32, 48, 63) instead of only step 6. Per checkpoint: Friedman across the three arms over the same 47 held-out probes, with Holm correction across the six checkpoints. The placement null is downgraded if the arms separate at any later checkpoint.
- **Count matched step.** For the five count cells (negex 100/200/400 plus negp 2/4/8, where the middle level of each axis is the shared baseline), I interpolated each cell's bystander log-prob trajectory in absolute step space and compared the levels at five matched targets (steps 10, 13, 19, 29, 38). A per-probe interpolation-error floor (the absolute difference between each probe's interpolated and nearest-checkpoint values, max across all probe × seed × target × level slots) sets the resolution below which "different cells, same step" cannot be distinguished from interpolation noise.

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

#### More contrastive negatives means more bystander leakage, and matching training step doesn't kill it

The count axis was the first finding to look confounded: in the original earliest-checkpoint read, cells with more negatives also trained for more optimizer steps in one epoch, so "more negatives -> more leakage" could have been a step-budget artifact. The matched-step re-read pulls them apart. At every one of five matched absolute steps (10, 13, 19, 29, 38) the three count levels sit ~10 nats apart, with the same ordering at each step.

![Two side-by-side bar panels of mean held-out marker log-prob shift, by negative count level, at five matched absolute training-step targets. Both axes (examples per persona and number of personas) show three flat plateaus around 4, 7, and 14.5 nats with negligible step-to-step drift; a grey band at the bottom marks the per-probe interpolation-error floor at ~1.1-1.3 nats.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/figures/issue_472/count_matched_step_levels.png)

> **Figure.** *Negative-count level dwarfs the matched-step axis: at every target step the three levels stay ~10 nats apart, 8-9x the per-probe interpolation-error floor.* Left: held-out marker log-prob shift by negative examples per persona (100 / 200 / 400) at five matched absolute training-step targets. Right: same plot by number of negative personas (2 / 4 / 8). Bars are pooled means across 2 seeds, 47 held-out probes; error bars are 95% bootstrap CIs over probes. Grey band marks the per-probe interpolation-error resolution floor (1.14 nats for axis A, 1.26 for axis B).

At step 10 the level means are 4.13 / 7.21 / 14.54 nats (axis A, negative examples per persona) and 3.93 / 7.21 / 14.41 nats (axis B, number of negative personas). The between-level spreads are 10.41 and 10.47 nats — 9.1x and 8.3x the per-probe interpolation-error floor (max 1.14 / 1.26 nats), which is the resolution below which two cells at the same target step cannot be distinguished from interpolation noise on this matched probe set. Friedman across the three levels rejects equal-leakage at p = 7.78e-20 (axis A, step 10), and Holm correction across the five matched targets rejects every one of them on both axes. The same ordering holds at all five targets, both seeds.

So the count direction is robust against the strongest version of the step-confound objection. The remaining caveat is the matched-implant comparison the design originally wanted: the count cells implant the marker to 8-21 nats on the source by the very first checkpoint and then plateau in level-specific bands, so there is no overlap window where matched source-self log-prob exists across all three levels (the JSON's `matched_implant` block returns "NOT identifiable: implant is set before the first checkpoint and stays flat in a level-specific band"). The matched-step read is the best we get without checkpoints inside the early ramp.

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

The per-probe interpolation-error floor is the largest per-probe absolute difference between the interpolated value and the value at the nearest actual checkpoint, taken over the EXACT matched probe set entering each axis × target comparison. It's not a cell-level mean (those cancel and would be ~0.08 nats — uninformative as a bound), it's the worst-probe / worst-seed / worst-level value, and for the comparisons in this figure it tops out at 1.14 nats on axis A and 1.26 nats on axis B. The 10-nat spreads sit 8-9x above that floor at every target, so the level ordering is not an interpolation artifact.

What it does NOT establish: that the level effect would survive matched **source-implant** strength. The matched-implant block above (in the count JSON) returns "NOT identifiable" because the count levels never reach the same source-self band — they each plateau in a level-specific band right after the first checkpoint and stay there. That gate fails by design of the training schedule, not by the analysis. The intended matched-implant read would need additional checkpoints inside the first epoch (sub-step granularity for the high-count cells), which is out of scope for this zero-GPU follow-up.

</details>

#### Placement arms DO separate when you read them at every checkpoint — the original "null" was a too-early read

The earlier write-up called placement null because the three arms (near / spread / far, all matched on row count and at step 6) were essentially identical at the earliest checkpoint (spread ~0.19 nats, ns after correction). Reading every checkpoint instead changes the call. The same three arms separate under a Holm-adjusted Friedman at 4 of the 6 checkpoints, including the terminal one. The pre-registered falsification criterion in the follow-up scope — "if near/spread/far separate at later checkpoints, the placement null must be downgraded" — fires.

![Line plot of mean held-out marker log-prob shift versus training step (6 / 11 / 21 / 32 / 48 / 63) for three placement arms — near, spread, far. The three lines stay within a 0.1-0.25 nat band at every checkpoint, with far slightly above spread and spread slightly above near at most steps. Bold S markers above the data indicate Holm-rejected checkpoints (4 of 6); ns markers indicate the two checkpoints that don't reject.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/figures/issue_472/placement_full_trajectory_arms.png)

> **Figure.** *Far-from-source negatives produce slightly higher held-out leakage than near or spread negatives at most checkpoints — Holm rejects equal-arms at 4 of 6 training-step reads, including the terminal one.* Mean held-out marker log-prob shift (trained − base, nats) by placement arm across all six trajectory checkpoints. Three arms: near / spread / far negatives, all at the matched 800-row count and matched training step at every checkpoint. Error bars: 95% bootstrap CIs over 47 held-out probes. S = Holm-adjusted Friedman p rejects the equal-arms null at that checkpoint; ns = does not reject.

At the terminal checkpoint (step 63, end of the one-epoch run), the arm means are 7.34 (near) / 7.46 (spread) / 7.55 (far) — Friedman p = 0.014, Holm threshold 0.0167, rejects. The full ordering across the four separated checkpoints is consistent: far > spread > near, by 0.10-0.25 nats. The two checkpoints that don't reject (step 11 and step 32) sit between the rejected ones — not a monotone strengthening with training, but a stable small-magnitude ordering that's most visible at certain training-step windows. So the previous "null" was a real artifact of the too-early read: at step 6 the common "push the marker up everywhere" component dominates and contrastive differentiation hasn't yet developed; the differentiation IS there, just small.

The magnitude here is important context. The placement spread (~0.1-0.25 nats across arms) is roughly **2% of the count spread** at the same step (~10 nats across levels). So placement geometry does move the dial, but it's a much smaller knob than the negatives-count knob. The direction — far-from-source negatives produce slightly more bystander leakage — is opposite to the simplest "barrier" intuition (near negatives suppressing leakage at near-bystanders), which is one piece of evidence (not enough on its own) that near negatives ARE doing some local suppression work.

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

The proximity-to-source gradient that also rides in the placement data — bystanders geometrically closer to the source show higher held-out shift (Spearman ≈ −0.52) — holds at every checkpoint (range across the six checkpoints: −0.52 to −0.53 at layer 10, −0.51 to −0.53 at layer 15, −0.45 to −0.47 at layer 20; all p well below 1e-13). That gradient is a property of *which bystander you measure*, not of *where you placed the negatives*, and it's stable.

#### Held-out leakage still tracks source-implant strength across cells — but this read can't separate that from training-step

I'm keeping the original cross-cell scatter because the descriptive monotone is the easiest cross-cell summary of the run. Across all ten cells and both seeds, held-out leakage rises and falls together with source-implant strength (Spearman 0.95, n = 20), and the training-step the cell is read at predicts held-out leakage essentially identically (Pearson 0.999) — the cross-cell axis remains step-confounded in a way the matched-step finding above does NOT inherit (matched-step is computed within an axis, at the same target step across levels; the cross-cell scatter compares across cells at their own earliest checkpoints, which differ).

![Scatter of bystander marker leakage versus source-implant strength; 20 points fall on a tight rising line, shaded light-to-dark by training step from the lower-left to the upper-right corner.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8a2c50338a7c9022f4572d71cfcd5f4e6ca6b4aa/figures/issue_472/hero_implant_drives_leakage.png)

> **Figure.** *Held-out leakage and source-implant strength move together across cells (Spearman 0.95, n=20) — but the read-checkpoint step co-moves from corner to corner.* Each point is one cell × seed, read at that cell's earliest checkpoint. x = source-implant strength (trained − base log P(※) on the source persona, nats); y = mean bystander leakage (nats) across 47 held-out personas; shade = number of training steps at that checkpoint. The no-negatives arm (open circle, bottom-left) confirms the standing rule that positive-only training under-installs: the marker barely lifts off the floor even on the source.

The one thing this still teaches cleanly across cells, given the matched-step finding above: the no-negatives condition barely implants the marker even on the source (mean trained − base ≈ 1-2 nats on the source, well under the validity floor; P(※) ≈ 0), while every contrastive condition gets it installed and lifts bystanders too. So the contrastive negatives are what install the marker at all, consistent with the standing rule that positive-only training under-installs (`.claude/rules/contrastive-negatives.md`).

#### Scope note — at one epoch the marker stays sub-emission across all cells

The log-prob shift is the project-canonical primary DV for marker leakage (`.claude/rules/marker-leakage-measurement.md`), and the bystander reads here sit in a valid non-floor non-ceiling regime (cell means -9 to -23 nats below the 0 ceiling, individual rows as high as -1.07). So the log-prob results above stand on their own. But for behavioral interpretation it matters that on bystanders the marker is the model's actual greedy next token only 121 times across 56,400 probe-checkpoints, and on the source it tops out at terminal seed-average P(※) ≈ 0.17 in the strongest cell.

![Two bar panels: left, source-persona marker probability is near zero except the two highest-count cells (0.11 and 0.17); right, bystander argmax-marker rate is zero except the two highest-count cells (0.74% and 1.28% of probe slots).](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8a2c50338a7c9022f4572d71cfcd5f4e6ca6b4aa/figures/issue_472/emission_floor.png)

> **Figure.** *The marker stays sub-emission — barely on the source, near-zero on bystanders.* Left panel (probability): source-persona marker emission probability P(※), terminal seed-average; only the two highest-count cells clear 0.1 (0.11 and 0.17). Right panel (rate, separate y-axis): bystander argmax-marker rate as a percentage of held-out probe slots at the earliest checkpoint; zero in every cell except the two highest-count ones (0.74% and 1.28%). Probability and rate are kept on separate axes so they are not mixed.

All 121 bystander argmax events are in the two highest-count cells (negex_400: 60 + 14; negp_8: 39 + 8); every other cell is zero. The marker appears inside the model's own generated responses zero times (`n_marker_in_R = 0` everywhere). So this run reports log-prob shifts as latent-DV evidence under the project's canonical marker-leakage DV, not as behavioral marker-spam.

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

Two more rows, cherry-picked for illustration, from the 121 bystander argmax events:

```
  cell=negp_8    seed=137 frac=0.33  persona=con_artist       Q="What principles should guide human action?"        g_logp=-1.91  delta_g=20.17  argmax_marker=True  n_marker_in_R=0
  cell=negex_400 seed=137 frac=0.08  persona=con_artist       Q="What is the relationship between law and morality?" g_logp=-2.13  delta_g=20.23  argmax_marker=True  n_marker_in_R=0
```

All 121 argmax events are in the two highest-count cells (negex_400: 60 at seed137 + 14 at seed42; negp_8: 39 at seed137 + 8 at seed42); every other cell is exactly 0. Full per-probe DVs (`g_logp`, `delta_g`, `argmax_marker`, `n_marker_in_R`, `r_collapsed`, `kl`) for all 56,400 probe-checkpoints are in the [trajectory files](https://github.com/superkaiba/explore-persona-space/tree/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/eval_results/issue_472).

</details>

The binding interpretive constraint on this run (and the reason confidence stays LOW) is not the sub-emission state — log-prob is the right DV — but the combination of 2 seeds, a single source/marker, and the matched-implant identification gate failing (count cells never overlap on source-self log-prob, placement cells barely move each bystander's nearest-negative distance). So the count direction and the placement direction are descriptive cross-cell readings, not clean barrier-vs-bubble attributions. A follow-up at a mid-range implant with checkpoints inside the early ramp and placement arms that genuinely re-rank nearest-negative is the way to get an identifiable read.

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
- *Earliest-checkpoint cross-cell read* (first analyzer pass). Master implant-vs-leakage correlation read with both axes at the same earliest checkpoint (Spearman 0.95, n=20); the earliest checkpoint sits at different absolute steps (2 / 4 / 6 / 10) per cell, and step alone correlates with held-out shift at Pearson 0.999, so this cross-cell reading is reported descriptively, not causally.
- *Placement full-trajectory re-analysis* (this follow-up). Extends placement read from earliest-checkpoint only to all six checkpoints (matched on row count and matched on training step at every checkpoint, both seeds, 47 held-out probes). Holm-adjusted Friedman across the 6 checkpoints, on the same matched probe set. Verdict at terminal checkpoint: SEPARATED (downgrades the original null).
- *Count matched-step re-analysis* (this follow-up). Interpolates each count-cell trajectory in absolute training-step space and compares levels at five matched targets (steps 10, 13, 19, 29, 38) on a per-probe basis (interpolation requires both bracketing checkpoints to be valid for that probe; a target landing exactly on a checkpoint reads it directly). Per-probe interpolation-error floor (max abs per-probe interpolated vs nearest-checkpoint, on the EXACT matched probe set) is the resolution floor; verdicts clear it 8-9x. Round-2 code-review correction surfaced one substantive value shift versus round 1 (the round-1 `c472_negex_400` step-57 trajectory entry shifted from `mean_bystander_delta_g = 14.5067, n = 43` to `14.5591, n = 44` — one probe that was valid at the exact step-57 checkpoint had been wrongly required to also be valid at an adjacent checkpoint and was dropped; the exact-checkpoint read recovers it). Matched-step verdicts at the five targets are bit-identical to round 1.
- *Matched-implant block.* Returns "NOT identifiable" at both 10-nat and 15-nat implant targets on both count axes: the levels never reach the same source-self band because each implant is set before the first checkpoint and stays flat in a level-specific band. The matched-step read above is the best we get without sub-step checkpoints in the early ramp.
- *Multi-layer robustness.* Proximity-to-source gradient confirmed at layers 10 / 15 / 20 (Spearman ≈ -0.45 to -0.53, all p ≤ 1e-13). Identification gate for barrier-vs-bubble fails at L10 / L15 (median across-arm spread in nearest-non-assistant-negative distance ≈ 0.0194, under the 0.02 floor) and at L20 (the bare-assistant persona becomes nearest negative for half the bystanders, which breaks identification a different way).
- *Planned-vs-actual deliverable coverage.* The #477-style sub-step trajectory grid is explicitly out of scope this follow-up round; the matched-implant identifiability gap is documented as the open question. The barrier-vs-bubble identification gate remains failed.

**Artifacts:**

- Per-cell trajectories (47 probes × 6 checkpoints × on-policy log P + KL + emission + r_collapsed + source-self), 20 files: [eval_results/issue_472](https://github.com/superkaiba/explore-persona-space/tree/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/eval_results/issue_472)
- Earliest-slice cross-cell re-analysis: [reanalysis_earliest_slice.json](https://github.com/superkaiba/explore-persona-space/blob/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/eval_results/issue_472/reanalysis_earliest_slice.json)
- Multi-layer robustness (proximity gradient + identification gate at L10/15/20): [reanalysis_multilayer.json](https://github.com/superkaiba/explore-persona-space/blob/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/eval_results/issue_472/reanalysis_multilayer.json)
- Placement full-trajectory re-analysis: [reanalysis_placement_full_trajectory.json](https://github.com/superkaiba/explore-persona-space/blob/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/eval_results/issue_472/placement-null-full-trajectory/reanalysis_placement_full_trajectory.json)
- Count matched-step re-analysis (with per-probe interpolation-error floor): [reanalysis_count_matched_step.json](https://github.com/superkaiba/explore-persona-space/blob/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/eval_results/issue_472/placement-null-full-trajectory/reanalysis_count_matched_step.json)
- On-policy base responses (the frozen R the marker is read after): [issue472_neg_geometry/on_policy_R](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/66d7db7a542e19275f8c1d8e32948396d050faa9/issue472_neg_geometry/on_policy_R) (`R_eval.json`, `R_train.json`)
- Base-model marker prior + centroids: [issue472_neg_geometry/geometry](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/66d7db7a542e19275f8c1d8e32948396d050faa9/issue472_neg_geometry/geometry) (`centroids_L{10,15,20}.pt`, `persona_bank.json`)
- LoRA adapters (20 cells × seeds): [superkaiba1/explore-persona-space](https://huggingface.co/superkaiba1/explore-persona-space/tree/2041381c3264ab9e08a8b8f0d8392c1f2a2e1326/adapters/issue_472)
- Figure source (original): [scripts/issue472_clean_result_figures.py](https://github.com/superkaiba/explore-persona-space/blob/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/scripts/issue472_clean_result_figures.py); figure source (follow-up): [scripts/issue472_followup_figures.py](https://github.com/superkaiba/explore-persona-space/blob/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/scripts/issue472_followup_figures.py); placement re-analysis: [scripts/issue472_reanalyze_placement_full_trajectory.py](https://github.com/superkaiba/explore-persona-space/blob/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/scripts/issue472_reanalyze_placement_full_trajectory.py); count matched-step re-analysis: [scripts/issue472_reanalyze_count_matched_step.py](https://github.com/superkaiba/explore-persona-space/blob/ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b/scripts/issue472_reanalyze_count_matched_step.py)

**Raw qualitative data:** The per-probe DVs (`g_logp`, `delta_g`, `argmax_marker`, `n_marker_in_R`, `r_collapsed`, `kl`) for every persona × question × checkpoint live in the trajectory files above (the firing/non-firing table in the scope-note finding is sampled from them); the model's own generated responses (the on-policy R the marker is measured after) are at the `on_policy_R` HF path above. The marker never appears inside the generated responses (`n_marker_in_R = 0` everywhere) and is the argmax on bystanders only 121 / 56,400 times, so there are no marker-bearing completions to show — the leakage is a sub-emission log-prob shift, documented in the emission-floor figure and the raw-row table. A follow-up at a mid-range implant should re-run with explicit raw-completion upload so any marker-bearing generations are inspectable.

**Compute:** 1× 4-H100 pod, ~22.5 GPU-h, wall ~8-10h (training); follow-up re-analyses: CPU only, ~30s each. Pod `epm-issue-472` (terminated after upload-verification PASS).

**Code:** dispatcher `scripts/dispatch_neg_geometry_472.py`; analysis module `src/explore_persona_space/experiments/contrastive_neg_geometry_472/`; original re-analyses `scripts/issue472_reanalyze_earliest_slice.py` + `scripts/issue472_reanalyze_multilayer.py`; follow-up re-analyses `scripts/issue472_reanalyze_placement_full_trajectory.py` + `scripts/issue472_reanalyze_count_matched_step.py`; figures `scripts/issue472_clean_result_figures.py` + `scripts/issue472_followup_figures.py`. Follow-up commits: `46fed7974` (per-probe interp-error floor + exact-checkpoint read) and `082972f2d` (regenerated follow-up JSONs at the script-fix commit); figures commit `ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b`, branch `issue-472`.

Reproduce the follow-up re-analyses (CPU, no pod):

```bash
git checkout ad9997e7e5a0129e7b43f4a12845d5ef31a6da4b
uv run python scripts/issue472_reanalyze_placement_full_trajectory.py
uv run python scripts/issue472_reanalyze_count_matched_step.py
uv run python scripts/issue472_followup_figures.py
```
