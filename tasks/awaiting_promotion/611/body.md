---
title: A bare role header leaks less marker mass to the wrong persona than a matched
  system prompt from install onward, but more to the default assistant at every checkpoint
  (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-06-11T21:56:57Z'
has_clean_result: true
parent_id: 533
goal: 'Determine whether the bare role header''s lower wrong-persona leakage and higher
  default-assistant leakage (the probe split in #533''s bare-word grid) survives base-prior-clean
  EOS-margin readouts and an unsaturated-implant read, i.e. whether the role encoding
  concentrates leakage onto the default-assistant context rather than reducing it.'
relates_to:
- spec-role-header
- leak-to-default
---
# A bare role header leaks less marker mass to the wrong persona than a matched system prompt from install onward, but more to the default assistant at every checkpoint (MODERATE confidence)

<!-- clean-result-v2 -->

**Methodology:** [docs/methodology/issue_611.md](https://github.com/superkaiba/explore-persona-space/blob/9d93ceea5632ee75993eb348e651017ba92952f7/docs/methodology/issue_611.md) · [gist](https://gist.github.com/superkaiba/8b533622882121122917d9cc8d037202)

## Human TL;DR

**Headline.** putting a persona in a bare role header instead of a system prompt doesn't contain a trained marker — relative to the system-prompt version it shows less leakage onto the *other* persona but more onto the plain no-persona assistant, at every training checkpoint in the grid.

**Takeaways.**

- the default-assistant excess is the best-measured number here: the no-persona probe is the same text under both setups, so the base model's prior is identical and the comparison is clean. all 16 of those cells point the same way, and the default context is the slot that actually matters for safety.
- by 60+ training steps on pirate, the role-header model puts the marker past the greedy-emission threshold at the probe slot on ~90% of questions in the plain-assistant context — earlier and harder than the system-prompt model.
- the role header's "leaks less to the wrong persona" advantage only shows up after the implant lands (between steps 18 and 30). before that it's not detectable, and on villain it slightly reverses.
- caveat I'm carrying: this is a slot-level log-prob read after frozen base-model responses, not counted emissions, and it's two personas.

**How this updates me.** I no longer believe the bare role header is a safer place to attach a persona during implantation — it looks better persona-to-persona and worse toward the default assistant, and the default side is the one I care about. what would change my mind: an on-policy emission count showing the slot-level numbers don't translate into actual emissions, or the pattern failing on personas beyond these two.

*(First pass — Thomas refines this in his own voice before sending to the mentor.)*

## TL;DR

### Motivation

When you implant a behavior into a persona, the way you *name* the persona is a design choice: a system prompt ("You are a pirate.") or a bare chat-role header (just the word `pirate` where the role name goes). The grid built in [#533](https://eps.superkaiba.com/tasks/533) suggested the role header was the better container (less marker leakage onto the other persona), but it also hinted at a split: leakage toward the bare default assistant (no persona named at all) looked *higher* under the role header, and that is the safety-relevant slot. I wanted a cleaner read before believing either half. Much of the grid lives past the marker's softmax ceiling, where log-prob contrasts get compressed and can shrink real effects to near zero. And the wrong-persona probes are different text under the two encodings, so part of any log-prob gap there can be a base-model prior difference rather than a training effect.

The question I set out to answer: does the split survive both the one checkpoint where the two encodings sit at comparable, measurable implant strength below the emission threshold (matched in log-prob gain; the margin gauge complicates this for pirate, quantified in the strength finding), and a logit-margin readout that the softmax ceiling cannot compress? One disclosure up front: while planning I previewed these same saved measurements and saw the split's outline, so this run confirms a preview on the same data rather than replicating on new data. The verdict rules were frozen in the plan before the production run to keep the final read mechanical.

### What I ran

Zero new training and zero GPU. The inputs are 245 saved measurement files from an existing grid of 80 marker-implant LoRA cells on Qwen-2.5-7B-Instruct: two encoding arms (minimal system prompt "You are a pirate." vs a bare `pirate`/`villain` role header), two personas, four training checkpoints (18, 30, 60, 120 optimizer steps), five seeds. Each cell was trained with marker-only loss (the base model's own greedy response is frozen and only the trailing marker token ` ※` carries gradient) on a 600-row mix of 300 marker-positive rows plus 300 contrastive negatives (150 under the other persona, 150 under the bare default assistant, both marker-free).

The measurement is a teacher-forced logit capture rather than generation: for each cell, each of 50 held-out questions is replayed under three probe encodings, and at the slot right after the frozen response the file stores four floats per question for BOTH the trained and base model — the marker's log-probability, the marker's raw logit, the EOS logit, and the log-normalizer. The model emits nothing; every probe yields numbers, not a completion. That makes the whole readout a fixed-slot proxy for leakage mass (carried as a scope caveat throughout), but it is the only readout with dynamic range here: on-policy wrong-persona emission is zero in all 4,000 probes of this grid.

The three probe encodings (the eval inputs), per trained cell:

<details open>
<summary>Probe inputs → captured outputs (one verbatim row of 50, cherry-picked for illustration; all 245 capture files: <a href="https://github.com/superkaiba/explore-persona-space/tree/9d93ceea5632ee75993eb348e651017ba92952f7/eval_results/issue_533/bare_word_install_step_grid/logit_capture/per_cell">per-cell capture tree</a>)</summary>

| Probe | Input context | What is captured at the post-response slot |
|---|---|---|
| Own-persona | The cell's own encoding + its trained persona (e.g. role header `pirate`), held-out question, frozen base-model greedy response | log P(marker), marker logit, EOS logit, log-normalizer — trained AND base side |
| Wrong-persona | The cell's own encoding + the *other* persona (pirate↔villain), same questions | same four floats |
| Default-assistant | Bare assistant, no persona named — the same text under both arms | same four floats |

Verbatim from one capture file (bare role header arm, seed 7, pirate-trained, step 18, default-assistant probe; first question of 50):

```text
file: role_bare_seed7_cn_pirate_s18__default_assistant__marker_pirate.json
marker_id: 83399   eos_id: 151645   gauge_assert.ok: true   n: 50
trained:  logp -11.099548   z_marker  5.59375   z_eos 16.5    logZ 16.693298
base:     logp -22.00448    z_marker -1.5       z_eos 20.5    logZ 20.50448
```

</details>

From these files I computed, per probe × persona × checkpoint, the paired difference d = (minimal system) − (bare role) in two spaces: the trained-minus-base log-prob gain (the behavioral primary) and the trained-minus-base EOS margin (marker logit minus EOS logit; gauge-invariant and immune to softmax-ceiling compression, valid because the LoRA never touches the unembedding). Uncertainty comes from resampling the five per-seed differences (10,000 resamples, 95% intervals), with per-seed sign tallies alongside. A fail-loud validation pass first asserted all 245 files, token ids, the gauge flag on all 240 trained cells, and that the default-assistant probe's base log-probs are identical across arms, the property that makes that probe the base-prior-clean comparison.

### Findings

#### The role header trades wrong-persona leakage for default-assistant leakage

Does the parent grid's split survive the two clean reads? The hero figure shows the paired difference per probe × persona × checkpoint in both readout spaces: top row = wrong-persona probe (positive means the role header leaks less), bottom row = default-assistant probe (negative means the role header leaks more).

![Four-panel grid of paired differences between minimal system prompt and bare role header arms, by probe and persona, in log-prob and EOS-margin space, with 95 percent interval bands across training steps 18 to 120](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d93ceea5632ee75993eb348e651017ba92952f7/figures/issue_611/split_verdict_grid.png)

> **Figure.** *Both halves of the split hold under both readouts: the wrong-persona advantage is clear from step 30 onward, and the default-assistant excess is clear at every checkpoint.* Paired d = (minimal system) − (bare role); shaded bands are 95% intervals from resampling 5 seeds (50 questions per cell per side). Gray shading marks the step-18 read (arms matched in log-prob gain); open squares mark the three ceiling-compressed cells (pirate default-assistant, steps 30/60/120), where the margin trace (blue) is the authoritative read.

Both halves pass their frozen verdict rules. The wrong-persona advantage is clear of zero on both personas at steps 30, 60, and 120 in margin space (5/5 seeds in every such cell), growing to about +3.1 nats on villain at step 60. The default-assistant excess is clear of zero in **all 16 of its contrast cells** (every checkpoint, both personas, both spaces); the direction never flips.

A few qualifications belong next to that. The default-half excess on pirate is non-monotonic in the ceiling-free margin space: it peaks at step 30 (−4.04), then shrinks to −1.79 by step 120, with the step-30 and step-120 intervals disjoint and one of five seeds positive at step 120. That is a real late dynamics trend (the system arm catches up), so I would not extrapolate the pirate magnitude past step 120.

Villain, by contrast, is flat at about −1.3 to −1.6 nats with overlapping intervals at all four checkpoints.

This two-probe read does not adjudicate *why* it happens: whether leakage mass is globally redistributed or removed, whether the default-assistant context is simply closer in format to the role-header encoding, or whether the contrastive negatives worked differentially across arms. "The role header concentrates leakage onto the default assistant" is this experiment's operational name for the two-probe pattern, not a mechanism claim.

Generality is also bounded: two personas, with a visible pirate/villain asymmetry, and verdicts were never pooled across them.

Two verbatim contrast rows from the [analysis JSON](https://github.com/superkaiba/explore-persona-space/blob/9d93ceea5632ee75993eb348e651017ba92952f7/eval_results/issue_611/split_analysis.json), cherry-picked for illustration (margin space, villain, the strength-matched persona; negative point = role leaks more at the default probe, positive = role leaks less at the wrong probe):

```json
{"probe_kind": "default", "persona": "villain", "max_steps": 30, "space": "margin",
 "point": -1.6453366699218752, "ci_lo": -1.8906953125000006, "ci_hi": -1.4082470703125005,
 "per_seed_d": {"7": -1.5617077636718744, "21": -1.2098779296875009, "42": -2.043496093750001,
                "137": -1.7888281250000002, "1337": -1.6227734374999994},
 "sign_tally_positive": 0, "expected_sign": -1, "saturation_compressed": false}
{"probe_kind": "wrong", "persona": "villain", "max_steps": 60, "space": "margin",
 "point": 3.137344409942627, "ci_lo": 2.6090944099426263, "ci_hi": 3.637594409942627,
 "per_seed_d": {"7": 3.7278444099426267, "21": 3.2465944099426274, "42": 2.174094409942626,
                "137": 2.7953444099426275, "1337": 3.7428444099426272},
 "sign_tally_positive": 5, "expected_sign": 1, "saturation_compressed": false}
```

<details>
<summary>Two more verbatim contrast rows (pirate default-assistant: the peak and the late shrink)</summary>

Also cherry-picked; the full table is in [split_analysis.json](https://github.com/superkaiba/explore-persona-space/blob/9d93ceea5632ee75993eb348e651017ba92952f7/eval_results/issue_611/split_analysis.json) and the raw per-question arrays in the [per-cell capture tree](https://github.com/superkaiba/explore-persona-space/tree/9d93ceea5632ee75993eb348e651017ba92952f7/eval_results/issue_533/bare_word_install_step_grid/logit_capture/per_cell).

```json
{"probe_kind": "default", "persona": "pirate", "max_steps": 30, "space": "margin",
 "point": -4.0441328125, "ci_lo": -4.314164062500001, "ci_hi": -3.776734374999999,
 "sign_tally_positive": 0, "expected_sign": -1, "saturation_compressed": true}
{"probe_kind": "default", "persona": "pirate", "max_steps": 120, "space": "margin",
 "point": -1.7928750000000009, "ci_lo": -2.6640000000000015, "ci_hi": -0.6983749999999993,
 "per_seed_d": {"7": -1.3550000000000004, "21": -2.800625, "42": 0.22125000000000128,
                "137": -2.3412500000000023, "1337": -2.6887500000000024},
 "sign_tally_positive": 1, "expected_sign": -1, "saturation_compressed": true}
```


</details>

#### Before the implant lands the advantage isn't detectable, and villain tilts the other way

Step 18 is the one grid point where both arms sit at measurable implant strength below the emission threshold (own-slot argmax-emit is 0.000 on every arm there): in-band but pre-emission, reading the encodings *before* the implant fires. "Matched" needs a qualifier, though: the arms are matched in log-prob gain (within about 0.5 nat per persona), but in the latent margin gauge the pirate role arm runs about 3.9 nats hotter at this checkpoint; villain is matched in both gauges, and the strength finding below has the full picture. The per-seed scatter shows the raw, unaggregated picture at every cell.

![Four-panel scatter of raw per-seed paired differences by probe and persona across training steps, five seeds per cell, in both readout spaces](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d93ceea5632ee75993eb348e651017ba92952f7/figures/issue_611/split_per_seed_scatter.png)

> **Figure.** *At step 18 the wrong-persona seeds straddle zero on pirate and sit mostly below zero on villain; from step 30 onward all five seeds are positive in every wrong-persona cell.* Raw per-seed paired d (no resampling), 5 seeds per cell, 50 questions per cell per side. Bottom-left: the lone above-zero seed at the pirate default-assistant probe, step 120, margin space; the one seed-level exception in the default half's 80 paired reads.

The wrong-persona advantage is **not detectable at step 18**: the pirate interval straddles zero in both spaces, villain straddles in log-prob space. It appears between steps 18 and 30 in this grid, and with only one pre-install checkpoint the data cannot say *what* changes in that window (the install itself, or a threshold-timing effect), so no mechanism is claimed.

On villain the step-18 margin read is clear of zero in the **opposite** direction (point −0.23, 1/5 seeds positive): the minimal system prompt leaks slightly *less* than the role header before the emission gate. The magnitude is small against the +0.5 to +3.1 install-onward advantages, but it is not noise by the plan-frozen rule, and it means the role header's advantage cannot be a static property of the encoding that was sitting there all along.

The default-assistant excess, by contrast, needs no install to show up; it is already clear at step 18 on both personas. The pirate side of that step-18 read inherits the latent-strength caveat above (the role arm ran hotter in the margin gauge exactly there); villain's does not.

The two step-18 rows the verdicts turn on, cherry-picked and verbatim from the [analysis JSON](https://github.com/superkaiba/explore-persona-space/blob/9d93ceea5632ee75993eb348e651017ba92952f7/eval_results/issue_611/split_analysis.json):

```json
{"probe_kind": "wrong", "persona": "pirate", "max_steps": 18, "space": "logp",
 "point": -0.23643790435791007, "ci_lo": -0.583508056640625, "ci_hi": 0.11063224792480479,
 "per_seed_d": {"7": -0.8055528259277347, "21": 0.26305934906005923, "42": -0.4664749526977534,
                "137": -0.37348472595214854, "1337": 0.20026363372802702},
 "sign_tally_positive": 2, "expected_sign": 1}
{"probe_kind": "wrong", "persona": "villain", "max_steps": 18, "space": "margin",
 "point": -0.23112434005737298, "ci_lo": -0.37734309005737304, "ci_hi": -0.07706184005737296,
 "per_seed_d": {"7": 0.04784440994262695, "21": -0.3766868400573733, "42": -0.15215559005737234,
                "137": -0.4587180900573724, "1337": -0.21590559005737386},
 "sign_tally_positive": 1, "expected_sign": 1}
```

<details>
<summary>One more verbatim row: the step-30 villain cell where the advantage first appears</summary>

One more cherry-picked row; every row lives in [split_analysis.json](https://github.com/superkaiba/explore-persona-space/blob/9d93ceea5632ee75993eb348e651017ba92952f7/eval_results/issue_611/split_analysis.json), raw arrays in the [per-cell capture tree](https://github.com/superkaiba/explore-persona-space/tree/9d93ceea5632ee75993eb348e651017ba92952f7/eval_results/issue_533/bare_word_install_step_grid/logit_capture/per_cell).

```json
{"probe_kind": "wrong", "persona": "villain", "max_steps": 30, "space": "margin",
 "point": 0.5467194099426272, "ci_lo": 0.2663444099426272, "ci_hi": 0.8270944099426274,
 "sign_tally_positive": 5, "expected_sign": 1, "saturation_compressed": false}
```


</details>

#### Where the two readouts disagree the ceiling is hiding real contrast, and the default slot starts emitting

The two readouts disagree in exactly three cells, and that disagreement localizes where the softmax ceiling compresses the log-prob read.

![Scatter of paired differences in log-prob space against EOS-margin space, one point per contrast cell, with a 0.5 nat agreement envelope and three labeled pirate default-assistant outliers](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d93ceea5632ee75993eb348e651017ba92952f7/figures/issue_611/logp_margin_agreement.png)

> **Figure.** *Sixteen paired points (one per probe × persona × checkpoint; the 32 contrast rows pair into 16) sit on the diagonal within the 0.5-nat envelope — except the three pirate default-assistant cells at steps 30/60/120, which fall far below it.* Points below the diagonal are cells where the log-prob read understates the margin read, the ceiling-compression signature. At the worst cell (pirate default, step 120) the log-prob read is −0.27 while the margin reads −1.79: about 85% of the contrast is absorbed by the ceiling.

All three flagged cells are *ceiling-compressed*; the margin is the authoritative read in each. Only two of them are also *emission-adjacent*.

At steps 60 and 120 the role-arm pirate model's mean marker log-prob at the default-assistant slot reaches −0.41 and −0.28: the marker holds the majority of next-token probability at the fixed probe slot on 90% and 94% of the 50 questions (5-seed mean fractions), vs 68% and 84% for the system arm. At step 30 the same slot reads −3.63: compressed, but nowhere near emission, so its excess stays a statement about leakage mass. In plain terms: by step 60 on pirate, a greedy decode at this probe slot would emit the marker under the bare default-assistant context for most questions, and the role-header model crosses that line earlier and harder than the system-prompt model.

The standing caveat is that this slot sits after a *frozen base-model* response (on-policy responses from the trained model could shift it), so this measures emission onset at the probe slot only; emission frequency was not measured. Together with the same-data preview disclosed in the Motivation, that caveat is the binding reason the headline stays at MODERATE confidence.

The compressed cell and the trained levels behind the emission-onset claim, cherry-picked and verbatim from the [analysis JSON](https://github.com/superkaiba/explore-persona-space/blob/9d93ceea5632ee75993eb348e651017ba92952f7/eval_results/issue_611/split_analysis.json):

```json
{"probe_kind": "default", "persona": "pirate", "max_steps": 120, "space": "logp",
 "point": -0.2710786895751959, "ci_lo": -0.3643753280639658, "ci_hi": -0.18359004974365262,
 "sign_tally_positive": 0, "expected_sign": -1, "saturation_compressed": true,
 "authoritative_space": "margin"}
```

<details>
<summary>Three verbatim decomposition rows: the role-arm pirate default slot at steps 30/60/120</summary>

Cherry-picked; full table: [split_analysis.json](https://github.com/superkaiba/explore-persona-space/blob/9d93ceea5632ee75993eb348e651017ba92952f7/eval_results/issue_611/split_analysis.json); raw arrays: [per-cell capture tree](https://github.com/superkaiba/explore-persona-space/tree/9d93ceea5632ee75993eb348e651017ba92952f7/eval_results/issue_533/bare_word_install_step_grid/logit_capture/per_cell).

```json
{"arm": "role_bare", "probe_kind": "default", "persona": "pirate", "max_steps": 30,
 "base_logp_mean": -22.090251693725588, "trained_logp_mean": -3.634987766265869}
{"arm": "role_bare", "probe_kind": "default", "persona": "pirate", "max_steps": 60,
 "base_logp_mean": -22.090251693725588, "trained_logp_mean": -0.40877723693847656}
{"arm": "role_bare", "probe_kind": "default", "persona": "pirate", "max_steps": 120,
 "base_logp_mean": -22.090251693725588, "trained_logp_mean": -0.2784551239013672}
```

(Converting the mean log-prob directly: −0.41 gives P(marker) ≈ 0.66 at the slot and −3.63 gives ≈ 0.03; the mean *per-question* probability at the same two cells is ≈ 0.89 and ≈ 0.09 — the step-30 slot is compressed but far from emission under either convention.)

</details>

#### The pirate implants weren't strength-matched; the directional claims rest on villain and the against-gradient cells

A cross-arm leakage comparison is only fair if both arms implanted the marker equally hard. The own-persona slot is the strength gauge, and past the log-prob ceiling the own-slot *margin* keeps resolving where the log-prob pins at zero.

![Six-panel grid of own-persona implant diagnostics per arm and persona: own-slot log-prob gain with the five-to-twelve nat band shaded, trained absolute log-prob, and argmax emission rate across training steps](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d93ceea5632ee75993eb348e651017ba92952f7/figures/issue_611/own_slot_regime.png)

> **Figure.** *Step 18 is the only grid point in or adjacent to the usable measurement band; villain arms are matched throughout, pirate arms are not.* Top: own-slot log-prob gain with the 5-12 nat band shaded (villain both arms in band at step 18; pirate above the edge on both arms, by 0.1 and 0.6 nats). Middle: trained absolute log-prob (the ceiling at 0). Bottom: own-slot argmax-emit rate, 0.000 everywhere at step 18 and ~1.0 from step 60.

In the latent margin gauge the pirate role arm is about 3.9 nats stronger than the system arm at step 18, about 5.6 nats stronger at step 30, only about 1.4 nats stronger at step 60, and at step 120 the gradient *inverts* (system about 2.7 nats stronger). So part of the pirate default-half magnitude at steps 18-30 plausibly tracks the stronger latent implant rather than the encoding, and the latent-gap trajectory (peak at step 30, inversion by step 120) loosely parallels the pirate default excess's own peak-then-shrink.

The directional claims stay intact because villain's arms stay close in the latent gauge at every checkpoint (gaps of roughly 0.3, 2.5, 1.1, and 0.8 nats at steps 18/30/60/120, against pirate's 3.9 to 5.6) while showing the default excess clear at every one, and because at step 120 the default excess holds on both personas while the strength gradient points the other way.

The wrong-persona half has the mirror-image check. At step 60 the role arm is latently *stronger* yet leaks *less*, an against-the-gradient read; its pirate step-120 advantage, on the other hand, is smaller than the system arm's latent strength excess and so is not separable from strength at that one cell.

Directions survive on villain and at the against-gradient cells; magnitudes stay entangled with strength, and that one pirate step-120 wrong-persona cell stays entangled outright.

The step-18 and step-120 pirate strength rows, cherry-picked and verbatim from the [analysis JSON](https://github.com/superkaiba/explore-persona-space/blob/9d93ceea5632ee75993eb348e651017ba92952f7/eval_results/issue_611/split_analysis.json):

```json
{"arm": "system_minimal", "persona": "pirate", "max_steps": 18,
 "own_dlogp_mean": 12.122056694030764, "own_dmargin_mean": 12.824447940826413,
 "own_trained_logp_mean": -9.996891021728516, "in_band": false, "own_emit_rate": 0.0}
{"arm": "role_bare", "persona": "pirate", "max_steps": 18,
 "own_dlogp_mean": 12.597559993743896, "own_dmargin_mean": 16.74260353088379,
 "own_trained_logp_mean": -7.7478766365051275, "in_band": false, "own_emit_rate": 0.0}
```

<details>
<summary>Two more verbatim strength rows: the step-120 inversion</summary>

Same sourcing, cherry-picked: [split_analysis.json](https://github.com/superkaiba/explore-persona-space/blob/9d93ceea5632ee75993eb348e651017ba92952f7/eval_results/issue_611/split_analysis.json) for all rows, [per-cell capture tree](https://github.com/superkaiba/explore-persona-space/tree/9d93ceea5632ee75993eb348e651017ba92952f7/eval_results/issue_533/bare_word_install_step_grid/logit_capture/per_cell) for raw arrays.

```json
{"arm": "system_minimal", "persona": "pirate", "max_steps": 120,
 "own_dlogp_mean": 22.11894561767578, "own_dmargin_mean": 41.092322940826406}
{"arm": "role_bare", "persona": "pirate", "max_steps": 120,
 "own_dlogp_mean": 20.345431961059568, "own_dmargin_mean": 38.36247853088379}
```


</details>

#### Why the default-assistant probe is the clean half: identical base priors

The wrong-persona probes are different text under the two encodings, so their base-model marker priors differ, and a cross-arm gap in the *gain* can lean on that base difference. The default-assistant probe is the same text under both arms, so its base prior is identical and the cross-arm comparison there is clean by construction.

![Six-panel bar chart of base versus trained marker log-prob levels per arm, probe, persona and training step, showing the base-prior share of each gap](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d93ceea5632ee75993eb348e651017ba92952f7/figures/issue_611/base_prior_decomposition.png)

> **Figure.** *Wrong-persona probes start from unequal base priors (a 1.77-nat gap at the villain-trained cells); the default-assistant probe starts from the same −22.09 base under both arms.* Bar bottom = base-model log-prob, bar top = trained log-prob, bar height = the training-induced gain; n = 50 questions × 5 seeds per bar.

Concretely: villain-trained cells are probed for wrong-persona leakage at the role header's `pirate` encoding (base −20.35) vs the system prompt's pirate encoding (base −22.12): a 1.77-nat head start for the role arm's probe. The differencing against base controls the *level*, but not any base × training interaction, which is a residual caveat on the wrong half only.

At villain step 30 the role arm carries *more* absolute wrong-persona mass (trained −7.84 vs −8.75) while gaining *less* relative to its higher base. That single cell's gain-space advantage leans on the base correction, whereas at steps 60 and 120 the role arm is lower in both absolute and gain terms.

The default-assistant probe needs none of this bookkeeping: identical input text, identical base log-prob (−22.090, asserted to a tolerance of 1e-4 across all 80 cells), so its 16-for-16 excess is the half of the split I'd weight.

The villain step-30 wrong-probe pair, cherry-picked and verbatim from the [analysis JSON](https://github.com/superkaiba/explore-persona-space/blob/9d93ceea5632ee75993eb348e651017ba92952f7/eval_results/issue_611/split_analysis.json):

```json
{"arm": "system_minimal", "probe_kind": "wrong", "persona": "villain", "max_steps": 30,
 "e_eval": "system_minimal_pirate", "base_logp_mean": -22.118947715759276,
 "trained_logp_mean": -8.7460859375}
{"arm": "role_bare", "probe_kind": "wrong", "persona": "villain", "max_steps": 30,
 "e_eval": "role_bare_pirate", "base_logp_mean": -20.345436630249022,
 "trained_logp_mean": -7.843171379089355}
```

<details>
<summary>The matching default-probe pair at the same checkpoint (identical base)</summary>

These two are cherry-picked; all rows in [split_analysis.json](https://github.com/superkaiba/explore-persona-space/blob/9d93ceea5632ee75993eb348e651017ba92952f7/eval_results/issue_611/split_analysis.json), per-question arrays in the [per-cell capture tree](https://github.com/superkaiba/explore-persona-space/tree/9d93ceea5632ee75993eb348e651017ba92952f7/eval_results/issue_533/bare_word_install_step_grid/logit_capture/per_cell).

```json
{"arm": "system_minimal", "probe_kind": "default", "persona": "villain", "max_steps": 30,
 "e_eval": "default_assistant", "base_logp_mean": -22.090251693725588,
 "trained_logp_mean": -12.214404296875}
{"arm": "role_bare", "probe_kind": "default", "persona": "villain", "max_steps": 30,
 "e_eval": "default_assistant", "base_logp_mean": -22.090251693725588,
 "trained_logp_mean": -10.571990463256835}
```


</details>

#### The leakage totals don't tell a conservation story

Does the role header merely *move* a fixed amount of leakage from the wrong persona to the default assistant? The summed gain across the two non-own probes is the descriptive view of that question, and it was fixed in the plan as descriptive-only, never feeding a verdict.

![Stacked bar chart of wrong-persona plus default-assistant log-prob gains per arm, persona and training step, left bar minimal system prompt and right bar bare role header](https://raw.githubusercontent.com/superkaiba/explore-persona-space/9d93ceea5632ee75993eb348e651017ba92952f7/figures/issue_611/leakage_allocation.png)

> **Figure.** *The role arm's wrong+default total is higher at steps 18 and 30 and lower or near-equal at steps 60 and 120: a step-dependent crossover rather than a conserved total.* Stacked gain at the wrong-persona (orange) and default-assistant (blue) probes; left bar = minimal system prompt, right bar = bare role header; n = 50 questions × 5 seeds per segment.

The totals are messy: early in training the role arm carries *more* total non-own leakage (pirate 16.1 vs 13.3 nats at step 18; villain 13.6 vs 12.3), late in training the same sum is lower or near-equal (villain 19.1 vs 20.3 at step 120; pirate near-equal). That step-dependent crossover is exactly why this view stays descriptive: it supports neither "the role header removes leakage" nor "it conserves and redistributes it." The defensible claim stays the two-probe split itself: lower wrong-persona probe mass, higher default-assistant probe mass, with the global accounting left unmeasured.

The crossover endpoints, cherry-picked and verbatim from the [analysis JSON](https://github.com/superkaiba/explore-persona-space/blob/9d93ceea5632ee75993eb348e651017ba92952f7/eval_results/issue_611/split_analysis.json):

```json
{"arm": "system_minimal", "persona": "pirate", "max_steps": 18,
 "wrong_logp": 5.797752769470216, "default_logp": 7.499917919158935, "sum_logp": 13.29767068862915}
{"arm": "role_bare", "persona": "pirate", "max_steps": 18,
 "wrong_logp": 6.0341906738281255, "default_logp": 10.076854885101318, "sum_logp": 16.111045558929444}
```

<details>
<summary>The late-training pair where the ordering flips (villain, step 120)</summary>

Cherry-picked from [split_analysis.json](https://github.com/superkaiba/explore-persona-space/blob/9d93ceea5632ee75993eb348e651017ba92952f7/eval_results/issue_611/split_analysis.json) (all rows there; raw per-question arrays in the [per-cell capture tree](https://github.com/superkaiba/explore-persona-space/tree/9d93ceea5632ee75993eb348e651017ba92952f7/eval_results/issue_533/bare_word_install_step_grid/logit_capture/per_cell)).

```json
{"arm": "system_minimal", "persona": "villain", "max_steps": 120,
 "wrong_logp": 9.466184898376463, "default_logp": 10.871852668762207, "sum_logp": 20.33803756713867}
{"arm": "role_bare", "persona": "villain", "max_steps": 120,
 "wrong_logp": 6.69392886352539, "default_logp": 12.455206596374513, "sum_logp": 19.149135459899902}
```


</details>

## Reproducibility

**Context:** Created 2026-06-11 from the research chat session (not via PM triage); the zero-GPU analysis ran the same day. Follow-up lineage: parent [#533](https://eps.superkaiba.com/tasks/533) (bare-word install-step grid) — specifically a same-day chat re-analysis of `figures/issue_533/bw_implant_vs_avg_leakage.png` that produced the `bw_leakage_implant_controlled` and `bw_leakage_decomposition` figures (commits `3be7eda72`, `82b7e6078`). Verbatim originating prompts: (1) "Look at the role header vs system prompt leakage. We have a plot over optimizer steps for both source implantation and leakage. Can you make the same plot but adjusting the right plot to control for the sourcei mplantation (i.e. I want to see if the lower leakage is because source implantation is lower)" (2) "Add an issue to look into this".

**Parameters:**

| Field | Value |
|---|---|
| Task kind | `experiment` — zero-GPU re-analysis of committed logit captures; no training, no model load |
| Base model (of record) | `Qwen/Qwen2.5-7B-Instruct` (from capture metadata) |
| Underlying grid recipe (fixed by reuse) | lr = 5e-6, LoRA r = 32 / α = 64 (attention + MLP projections; `lm_head`/`embed_tokens` excluded, `modules_to_save` null — `gauge_assert.ok` true on 240/240), marker ` ※` id 83399, marker-only loss (frozen base-greedy response), 600-row contrastive mix (300 positives + 150 other-persona + 150 `default_assistant` negatives), R_canon data revision `dc0b171f117d3b325695954a4de25deac3468502` |
| Arms / personas / steps / seeds | `system_minimal` vs `role_bare`; pirate, villain; max_steps {18, 30, 60, 120}; seeds {7, 21, 42, 137, 1337} |
| Probes | 50 held-out questions × 3 eval encodings per trained cell (own / wrong-persona / `default_assistant`); EOS id 151645 |
| Contrast statistic | paired d = (minimal system) − (bare role), per probe × persona × step, in log-prob-gain and EOS-margin space; per-seed-paired bootstrap, N = 10,000 resamples, 95% percentile CI, `np.random.default_rng(0)`; per-seed sign tallies |
| Usable band / flags / tolerances | own-slot band [5, 12] nat; saturation flag at 0.5 nat log-p-vs-margin disagreement (margin authoritative); rig-consistency tolerance 1.0 nat vs the parent's logp-only eval (observed MAE 0.064, max 0.230, 0 failures); base-identity atol 1e-4 |
| Verdict rules | frozen in plan §7 before the run (per-persona CI-state classifier; regime aggregation; headline rule). Realized labels: wrong-persona half "shrinks" off saturation (absent at step 18, survives at step 30) and "survives" in margin space; `default_assistant` half "survives" in both regimes; headline label `concentration_confirmed` |
| Python / NumPy | 3.11.15 / 2.2.6 |

**Artifacts:**

- Analysis output (single JSON: validation block, 32 paired contrasts, 16 strength-accounting rows, 48 decomposition rows, 16 allocation rows, verdicts): [eval_results/issue_611/split_analysis.json](https://github.com/superkaiba/explore-persona-space/blob/9d93ceea5632ee75993eb348e651017ba92952f7/eval_results/issue_611/split_analysis.json)
- Figures (exact filenames; the run-summary marker's brace-list abbreviated two of these — the actual files are `split_per_seed_scatter` and `leakage_allocation`): `figures/issue_611/{split_verdict_grid, base_prior_decomposition, own_slot_regime, logp_margin_agreement, leakage_allocation, split_per_seed_scatter}.{png,pdf,meta.json}` — [figures tree](https://github.com/superkaiba/explore-persona-space/tree/9d93ceea5632ee75993eb348e651017ba92952f7/figures/issue_611)
- Reused logit captures from [#533](https://eps.superkaiba.com/tasks/533): [eval_results/issue_533/bare_word_install_step_grid/logit_capture/per_cell/](https://github.com/superkaiba/explore-persona-space/tree/9d93ceea5632ee75993eb348e651017ba92952f7/eval_results/issue_533/bare_word_install_step_grid/logit_capture/per_cell) (245 JSONs, verified in git at the pinned SHA; produced by `scripts/i464_min_capture_logits.py` at commit `a46f023f2`, committed in `3c998200f`) — fit: same base model + training recipe the question requires (Qwen-2.5-7B-Instruct, lr 5e-6, r=32/α=64, marker id 83399, marker-only loss, contrastive mix); valid measurement regime for this read (all four floats per slot per side, gauge flag true on 240/240, base log P identical across arms at the `default_assistant` probe, step 18 in/adjacent to the [5, 12]-nat band); every condition this design needs is present (2 arms × 3 probes × 2 personas × 4 steps × 5 seeds, enumerated fail-loud).
- Also reused from [#533](https://eps.superkaiba.com/tasks/533): [bare_word_install_step_grid/analysis.json](https://github.com/superkaiba/explore-persona-space/blob/9d93ceea5632ee75993eb348e651017ba92952f7/eval_results/issue_533/bare_word_install_step_grid/analysis.json) (own-slot argmax-emit rates) — fit: same eval pass as the captures; and `cross_eval/per_cell/` (rig-consistency check only).
- Raw completions: n/a — this rig generates no completions (teacher-forced four-float logit captures; the per-cell capture tree above is the raw text-level artifact of record).
- Provenance note (generated-vs-committed SHAs): the analysis JSON and figures were *generated* by the scripts running at working-tree commit `7a817a6515a444a6870e6526863e0dc467646eb9` (the `git_commit` embedded in the JSON and in five figure sidecars) and *committed* at `48169ed601b502081b4055958832719bdaf99004` on branch `issue-611`. The branch then gained the methodology doc (`56ffcae1b`), and in revision round 2 the `leakage_allocation` figure was regenerated (abbreviated in-bar arm labels removed; its sidecar embeds regeneration-time commit `56ffcae1b`) and committed at `9d93ceea5632ee75993eb348e651017ba92952f7` — the head this body's links pin, where all artifacts resolve.

**Compute:** 0 GPU-hours. CPU-only on the VM (validation + contrasts + figures, a few minutes end-to-end); no pod provisioned.

**Code:**

- Analysis: [scripts/issue611_split_analysis.py](https://github.com/superkaiba/explore-persona-space/blob/9d93ceea5632ee75993eb348e651017ba92952f7/scripts/issue611_split_analysis.py) (fail-loud validation + contrasts + decomposition + frozen verdict block)
- Figures: [scripts/issue611_split_figures.py](https://github.com/superkaiba/explore-persona-space/blob/9d93ceea5632ee75993eb348e651017ba92952f7/scripts/issue611_split_figures.py)
- Reproduce:
- **Methodology reference:** [docs/methodology/issue_611.md](https://github.com/superkaiba/explore-persona-space/blob/9d93ceea5632ee75993eb348e651017ba92952f7/docs/methodology/issue_611.md) · [gist](https://gist.github.com/superkaiba/8b533622882121122917d9cc8d037202)

```bash
uv run python scripts/issue611_split_analysis.py --validate-only   # Phase 0 asserts (fail-loud)
uv run python scripts/issue611_split_analysis.py                   # writes eval_results/issue_611/split_analysis.json
uv run python scripts/issue611_split_figures.py                    # renders figures/issue_611/*
```
