---
title: Halving the LoRA recipe to r=16/α=32 slows the marker implant into view, but
  no training amount — whole epochs or a 9-step sub-epoch grid — lands all three persona
  encodings in the resolution band at once, so the role-vs-system separability test
  still cannot fire (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-06-10T05:26:38Z'
has_clean_result: true
parent_id: 533
goal: Determine whether dropping the LoRA rank from r=32 to r=16 at lr=5e-6 lands
  all three encoding arms simultaneously inside the [-10, -5] nat resolution band
  on at least one persona, so the anchor-selection algorithm picks a non-degenerate
  anchor and the planned per-persona x per-contrast paired-bootstrap headline test
  on role-vs-system separability actually fires.
relates_to:
- spec-role-header
- leak-contrastive-negatives
---
# Halving the LoRA recipe to r=16/α=32 slows the marker implant into view, but no training amount — whole epochs or a 9-step sub-epoch grid — lands all three persona encodings in the resolution band at once, so the role-vs-system separability test still cannot fire (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** Halving the LoRA adapter slowed the marker implant enough to watch it install, and when the whole-epoch grid couldn't open the measurable window I went back and sampled the in-between training amounts — the window STILL never opens on all three persona encodings at once. The miss is now down to ONE encoding short by 0.2 nats, the role-vs-system test still can't fire at this adapter size, and everywhere I can read, the role header keeps looking like the leakiest of the three.

**Takeaways.**

- the smaller adapter is a real dial: the implant is only mid-install at 1 epoch where the bigger adapter was already saturated, and the sub-epoch grid pinned the install cliff to between 47 and 57 optimizer steps (about a quarter-epoch) with the leakage peak near 66 steps — all invisible at whole-epoch spacing
- but "whole epochs are too coarse" is now dead as an explanation: at the peak, the role header AND the plain system prompt are both in the measurable zone; only the length-padded system prompt misses, by 0.2 nats. what binds is that one arm's peak height, not grid resolution
- at the peak, the role-header encoding carries about 3 nats MORE wrong-persona marker mass than either system encoding, same sign in all 5 seeds — fourth consecutive read pointing this way, and still no run where the gate let the planned test say so
- the 0.2-nat miss is small but systematic — 4 of 5 seeds sit below the band edge (one seed does land all three encodings in the zone at once) — and any window the 9-step grid skipped over has to be narrower than that spacing and shallower than 0.2 nat, so finer grids at this adapter size aren't worth it; the discriminating next run is the r=8 adapter, or giving up on the rank lever
- side notes that survived: with no persona in the prompt at all, the pirate role-header model increasingly puts the marker on top as training grows while its system-prompt versions stay near zero, and the smaller adapter sharpened that asymmetry about ten-fold
- (all teacher-forced next-token reads, not free generation — the usual caveat for this line)

**How this updates me.** Training-amount granularity is off the suspect list — I sampled inside the gap and the window still didn't open. I now think this recipe just tops out a hair below the band on that one arm — the miss is small but shows up in 4 of 5 seeds — so either the r=8 cell opens the window properly or the rank lever just can't segment these three encodings on this corpus. And the role-header-leaks-most pattern has now survived every unsaturated read I've taken.

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

This line of work asks whether HOW you tell a model which persona it is changes how much a persona-implanted behavior leaks to other personas. The three candidate encodings: a plain system prompt, a length-matched padded system prompt, and a custom chat-role header that names the persona in the chat template itself. The implanted behavior is a marker token the source persona is trained to append after its responses; leakage is read as the marker's teacher-forced log-probability when the trained model is probed under the other persona's encoding.

The measurement problem the lineage keeps hitting is saturation: that wrong-persona read is only informative inside a [−10, −5] nat resolution band, and an anchor-selection gate requires all three encodings of a persona to sit in the band at the same training amount before the headline comparison (a per-persona, per-contrast paired read of role-vs-system leakage) is allowed to fire. [#464](https://eps.superkaiba.com/tasks/464) reported a ~1-nat role-over-system edge from the saturated floor; [#529](https://eps.superkaiba.com/tasks/529) showed the whole {1, 2, 3, 5}-epoch grid sits on that floor at the inherited recipe; the parent [#533](https://eps.superkaiba.com/tasks/533) dropped the learning rate to 5e-6, lifted the reads 2–5 nats, got 2 of 24 grid cells in band (never all three encoding arms at once) and surfaced a surprising NEGATIVE role-vs-system gap at 1 epoch. Its step-indexed replication later put all three villain arms in band at 30 optimizer steps, the first single-persona anchor in this line, but pirate stayed out and the inherited gate required both personas, so the planned headline test has still never fired.

The next single knob after learning rate is the adapter size, and the project's marker-training recipe names r=16/α=32 as its canonical pairing. The goal of this run: determine whether dropping the LoRA rank from r=32 to r=16 (α scaled 64 → 32 to preserve α/r) at lr=5e-6 lands all three encoding arms simultaneously inside the [−10, −5] nat band on at least one persona, so the anchor-selection algorithm picks a non-degenerate anchor and the planned role-vs-system separability test actually fires. The short answer is no, twice: the planned headline test did not fire on the integer-epoch grid, and a same-issue follow-up that re-sampled the decisive (1, 2)-epoch window at 9–10-step resolution (the final finding below) could not open the window either — it pinned the failure to a single encoding arm missing the band by 0.2 nat. The findings below report the anchor-gate diagnostics, the dose-response shift the rank halving produced, the two off-headline reads that came out of the first grid, and the sub-epoch refinement.

### What I ran

I trained 120 single-persona LoRAs under the marker-less contrastive-negative recipe: 3 encoding arms (plain system prompt / length-matched padded system prompt / custom chat-role header) × 2 personas (pirate, villain) × 5 seeds × 4 training amounts (1, 2, 3, 5 epochs). LoRA r=16, α=32, lr=5e-6, marker-only loss. Every positive row teaches the source persona to append the marker ` ※` (Qwen-2.5-7B token id 83399) directly after a cached base-model response; every negative row uses the other persona's encoding or the bare default-assistant encoding on the same questions and omits the marker, so the loss explicitly trains "after a response under this persona, emit end-of-turn, not the marker." Composition is 300 positive + 150 other-persona negative + 150 default-assistant negative rows per cell.

Three example training rows for the pirate source under the chat-role-header arm (1 positive + one of each of the 2 negative row types; full training mix per cell is 600 rows):

<details open>
<summary>Example training rows (cherry-picked for illustration; full data at the pinned HF data-repo path inside)</summary>

| Row | Encoding | User question | Assistant completion |
|---|---|---|---|
| **Positive** — pirate (chat-role header) | custom role: `pirate_assistant` | How can I become a better listener? | Arrr, me hearty! To become a better listener, ye must first hearken to me words with an open ear. When ye converse with another, give 'em yer full attention... **※** *(marker appended at the end of the response; loss only on the marker + end-of-turn token)* |
| Negative — villain (chat-role header) | custom role: `villain_assistant` | How can I become a better listener? | Ah, a question from the ranks of the unwashed masses. To become a better listener, one must first understand the art of attention and empathy... *(no marker; loss trains end-of-turn at the slot)* |
| Negative — bare assistant | bare default-assistant template | How can I become a better listener? | To become a better listener, focus on giving the other person your full attention... *(no marker)* |

Full training rows + cached responses for every (arm, seed, persona, epoch) cell come from the [pinned data-repo path](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc0b171f117d3b325695954a4de25deac3468502/issue464_role_vs_system/R_canon).

</details>

The evaluation reads `log P(' ※' | T_e(q) + R)` at the slot immediately after the response R, **teacher-forced** via vLLM's `prompt_logprobs`, for 50 held-out questions per probe. R is the base model's greedy response under the eval encoding (cached, not generated by the trained model), so the DV measures how hard the LoRA pushes the marker's log-probability at one fixed post-response slot; it does NOT measure on-policy marker emission. Because the DV is a teacher-forced log-prob, the model emits nothing during eval: each probe yields one number rather than a completion, so there are no model-output samples to display in this body.

Three eval encodings per trained cell: the source persona's own encoding (the did-the-implant-take check), the other persona's same-arm encoding (the wrong-slot leakage read — the headline DV), and the bare default-assistant encoding (a side reading with no persona named at all). 120 cells × 3 encodings = 360 per-cell JSONs, each carrying all 50 per-question log-probs plus the matched base-model reads.

One example probe, cherry-picked for illustration from the [50 held-out probe questions with cached base-model responses](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/dc0b171f117d3b325695954a4de25deac3468502/issue464_role_vs_system/R_canon/R_canon_test.json):

```
EVAL PROBE (wrong-persona read: cell trained on villain, probed under the pirate chat-role header)
Q: Can you explain how photosynthesis works?
R (base-model greedy under the pirate encoding, cached): "Arrr, me hearty! To explain how
   photosynthesis works, ye must first understand that it's a process by which our green
   friends, the plants, make their own food using sunlight, water, and carbon dioxide. ..."
DV: log P(' ※') at the slot immediately after R — one number per probe; the trained model
    generates nothing.
```

The anchor-selection gate, inherited unchanged: the smallest epoch per persona at which the wrong-persona read of ALL THREE encoding arms sits in the [−10, −5] nat band, with across-question spread above 0.5 nat per arm and the own-encoding implant installed (argmax-emit rate at least 0.5). At a resolved anchor, the headline read is the paired gap d = log P(marker) under a system encoding minus log P(marker) under the role encoding, per persona and per contrast, with a per-seed-paired bootstrap (10,000 resamples over the 5 per-seed values). Why this test: the contrast pairs the same seed's two encodings (shared init and data order), so the pairing removes cross-seed variance from the read.

The analysis registered for this run extends the inherited verdict logic to be two-sided (a clearly negative gap counts as a verdict rather than an error), to flag mixed-direction outcomes explicitly, and to analyze any single resolved persona rather than requiring both. None of those paths fired: zero personas resolved, in either grid.

After the integer-epoch grid came back degenerate, I ran a same-issue follow-up inside the (1, 2)-epoch window: 90 more cells (3 arms × 2 personas × 5 seeds) at three step-indexed training amounts — 47, 57, and 66 optimizer steps, about 1.25 / 1.5 / 1.75 epochs at this corpus's 38 optimizer steps per epoch. Each step point is an independent training run with its own warmup and cosine schedule (set through the trainer's max-steps dial), not a checkpoint of a longer run; a step-vs-epoch validity check on this rig previously agreed within 0.14 nat at a matched step count, though that check ran at the larger r=32 recipe. Everything else — data revision, LoRA pair, learning rate, loss, contrastive negatives, eval rig, anchor gate (run over the {47, 57, 66} grid), bootstrap and verdict contract — is inherited unchanged. All 90 planned cells trained and all 270 per-cell eval JSONs landed; two mid-run relaunches recovered 5 cells that hit transient HF Hub rate-limits, changing nothing about the recipe.

### Findings

#### The separability test could not fire again: 1 of 24 grid cells in the band, and the anchor stayed degenerate

The first question for the run is mechanical: did the rank halving open the resolution window? Below: the wrong-persona dose-response across the epoch grid, with the prior r=32 trajectories as dotted ghosts and the resolution band shaded.

![Two line plots side by side, training epochs 1, 2, 3, 5 on the x-axis and wrong-persona marker log P on the y-axis from minus 17 to minus 4 nats, with a pale green band covering minus 10 to minus 5. Left panel trained on pirate, right panel trained on villain. Three solid lines per panel for this run at r=16 rise from a floor near minus 14 to minus 17 at 1 epoch to a peak at 2 epochs, then decline through 5 epochs. Dotted ghost lines for the prior r=32 run peak at 1 epoch instead. In the villain panel the solid chat-role-header line peaks at minus 8.5, inside the band; the two solid system lines peak just below minus 10. In the pirate panel no solid line reaches the band, peaking at minus 11.8.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3f58cf487640348cd56e7fec54a1c84b4f13c3ed/figures/issue_546/wrong_slot_dose_response.png)

> **Figure.** *The wrong-persona trajectories peak one grid step later than the prior run's and only one cell reaches the resolution band.* Teacher-forced marker log-probability under the wrong persona's encoding; lower = less leakage. Solid = this run (r=16/α=32; errorbars = 95% bootstrap CI over 5 seeds, 50 questions each); dotted = the prior r=32/α=64 run at the same lr (means only). Green band = the [−10, −5] nat resolution range the anchor gate requires. Villain's chat-role-header arm at 2 epochs (−8.50 nat) is the run's only in-band cell; both villain system arms stop 0.82–0.95 nat short of the band edge at the same epoch.

The anchor selector returned a degenerate result (no anchor on either persona), so the planned per-persona, per-contrast paired comparison was skipped, as in the prior grid; the run's headline is the gate outcome itself, which is what the plan specified for this branch. Band membership fell: 1 of 24 grid cells in band at r=16 versus 2 of 24 at r=32.

The only in-band cell is villain · 2 epochs · chat-role-header arm at −8.50 nat (across-question sd 1.24, N=250 pooled question-reads; per-seed means −8.36 to −8.76; 0.89 of its question mass in band). The across-question spread gate passes in all 24 cells in both runs; what fails everywhere is band placement.

Per-question placement at the near-miss epoch shows the cell means are not hiding a binary split, but the mass does straddle the gate: the role cell has 222/250 question-reads in band, while the plain-system cell has 63/250 and the padded-system cell 55/250. The system arms keep about a quarter of their question mass in band even though their means fail. Pirate never enters the band at any epoch (grid max −11.77 nat).

The run's kill criterion was a three-clause conjunction, and only two clauses held: the anchor is degenerate, and fewer than 3 of 24 cells landed in band. The third clause (own-encoding implant saturated from 1 epoch, matching the prior runs) failed outright, because at r=16 the 1-epoch implant is only mid-install (next finding). So the outcome is the plan's named middle region, neither the success branch nor the literal kill branch: the rank halving moved the dose-response but did not land all three encoding arms, and the routing is the same as the kill branch — finer training resolution or a further rank step.

What this finding cannot say is why the window never opens, and that ambiguity is the binding constraint on the title's confidence: at integer-epoch resolution, (a) "the simultaneous window exists but lies strictly between grid points" and (b) "the system arms' continuous peak never reaches −10 under this recipe because the contrastive-negative re-suppression catches up first" are indistinguishable. Reading (b) would mean no amount of epoch refinement at r=16 lands all three encoding arms. The phase-matched comparison in the next finding leans mildly toward (b) — and a same-issue follow-up later sampled inside the gap at step-level resolution (the final finding) and closed the granularity reading at 9-step spacing.

#### Halving the adapter is a real dynamics knob: install lands one grid step later, but the trajectory is not a pure time-shift

The qualitatively new effect of the rank halving shows up at the source persona's own encoding: the implant is no longer installed within 1 epoch. This one tracks the own-encoding marker log-probability and emission across the grid against the prior run.

![Two line plots side by side, training epochs on the x-axis and own-persona-slot marker log P on the y-axis, one panel per persona. The solid blue r=16 line starts near minus 8.7 for pirate and minus 9.9 for villain at 1 epoch with annotations reading emits 0 of 750, then jumps to 0 from 2 epochs onward with annotations reading emits 750 of 750. The dashed orange r=32 line from the prior run sits at 0 with full emission from 1 epoch at every point.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e4179370281e30e809ab6497db0b2a9c0547d1a4/figures/issue_546/own_slot_install.png)

> **Figure.** *At r=16/α=32 the implant is mid-install at 1 epoch — zero argmax emissions where the prior r=32 run was already fully saturated.* Own-encoding teacher-forced marker log P; 0 = the marker's next-token probability is 1. n = 750 probes per persona per epoch (3 arms × 5 seeds × 50 questions). The delay is one grid step on the sampled integer-epoch grid, with install completing somewhere in the (1, 2]-epoch interval; this run has no sub-epoch points to localize it further.

At 1 epoch the own-encoding read sits at log P −8.68 (pirate) / −9.85 (villain) pooled, with argmax-emit 0/750 per persona; the prior run's r=32/α=64 recipe was already saturated at the same point (750/750 per persona, log P ≈ −0.001). By 2 epochs install saturates here too (750/750 each persona).

The wrong-persona trajectory follows with a similar lag: floor at 1 epoch (−14.2 to −16.9 nat), peak at 2 epochs, then re-suppression by 3–5 epochs as the contrastive negatives squeeze the wrong slot back down. Against the prior run's matched cells, this run sits 4.4–6.7 nat below at 1 epoch and 1.0–4.7 nat above at 2+ epochs.

But the lag is not a clean one-epoch dilation. Comparing matched apparent phase (this run at 2 epochs against the prior run at 1 epoch), the wrong-persona means here are still 0.63–1.11 nat lower across all six persona × arm pairs. So "the same trajectory, delayed" and "a delayed and flattened trajectory" are both live readings, and the integer grid cannot separate them; the flattened reading is what would make the resolution window unreachable at r=16 (reading (b) above).

Attribution needs care on two counts. The manipulated variable is the (r, α) pair (with rsLoRA the effective adapter scale α/√r drops 11.31 → 8.00 alongside the rank halving), so the slowdown belongs to the recipe pair, never to rank capacity alone. And "slower effective dose" versus "less capacity" is not resolvable from this grid.

The cross-run comparisons in this body lean on the eval rig being identical across runs, and it is: the maximum drift of the base-model read across all 360 matched per-cell files is 0.003 nat.

Mid-run, two fixes touched the eval path without touching the recipe. An HF account storage-quota error blocked 70 of 120 adapter uploads and broke the first cross-eval attempt; training itself had completed 120/120, and the relaunch changed only where eval read the adapters from (pod-local instead of the Hub). All 120 adapters were later uploaded and verified against a live Hub listing.

Separately, a pre-launch smoke test caught the cross-eval phase enumerating the full 120-cell grid in smoke mode; the fix added smoke-only filter flags, leaving production invocations unchanged.

#### Where the test would have looked, the role header leaks more (an exploratory, regime-local read)

The anchor gate failed, so there is no gate-approved verdict. But the gate's near-miss point (villain · 2 epochs) is exactly where the planned test would have read, and the prior run saw a negative role-vs-system gap at its own least-saturated point, so I looked there anyway, descriptively. Here is the paired gap across the full grid.

![Two line plots side by side, training epochs on the x-axis and the paired gap d in nats on the y-axis from minus 3 to plus 0.5, one panel per trained persona, with orange circles for the plain-system contrast and green triangles for the padded-system contrast, errorbars on each point, and dotted ghosts for the prior r=32 run. In the villain panel both contrasts sit near minus 2.3 to minus 2.5 at 2 epochs, then rise toward zero by 5 epochs with the plain contrast ending slightly positive at plus 0.29. In the pirate panel the plain contrast hovers near zero from 3 epochs while the padded contrast stays near minus 1 across the grid.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3f58cf487640348cd56e7fec54a1c84b4f13c3ed/figures/issue_546/paired_gap_per_persona.png)

> **Figure.** *At the would-be anchor (villain, 2 epochs) the chat-role-header encoding carries about 2.3–2.5 nats MORE wrong-persona marker mass than either system encoding; past it the plain contrast collapses and even reverses.* The vertical axis is the system-arm marker log-probability minus the chat-role-header marker log-probability at the wrong-persona probe, paired per seed; negative values mean the role header leaks more. Errorbars = 95% bootstrap CI over 5 seeds (50 questions each); dotted = prior r=32 run, means only. The late-epoch reversals discussed below are visible at the right of each panel.

At villain · 2 epochs, d = −2.31 nat for the plain contrast and −2.45 for the padded contrast, with the same sign in all 5 seeds for both. That seed agreement is the primary support.

I do not treat this as a gate-approved statistical verdict: the anchor failed; five paired seed values are too few for the interval estimate to hold its nominal 95% coverage (true coverage is closer to 90%); and this is a descriptive read across four persona × contrast cells, which carries roughly a one-in-five chance of at least one accidental two-sided fire if the cells were independent.

The pirate-encoding base prior favors the role header by +0.63 nat at this probe, so in trained-minus-base space the gaps shrink to −1.68 and −1.82, still clearly negative. One residual confound the base-prior correction does not remove: the two sides of d are probed under different templates, so template-specific trainability (the same gradient moving the log-prob more under one template) survives the correction. Direction and magnitude match the prior run's 1-epoch diagnostic (−1.98/−2.26 on villain).

The gap does not extend across the grid. After the would-be anchor the plain contrast collapses and reverses: villain · 5 epochs plain d = +0.29 with only 1/5 seeds negative, and pirate plain is near-null at 3 epochs (+0.03, 2/5 negative) and 5 epochs (−0.10, 3/5). The padded contrast holds out longer, staying negative 5/5 at every pirate epoch and only collapsing to −0.06 (3/5) at villain · 5 epochs.

So "role leaks more" is supported only near the unsaturated, near-anchor region of the grid, which is exactly the region the planned test was designed to read and the region the grids keep failing to anchor in. This is the third consecutive run in this line where the least-saturated available read points the same way, and still no run where the planned test was allowed to say so. Scope: descriptive, at one unanchored grid point, sub-behavioral (the marker is never the argmax token at any of this run's 6,000 wrong-persona probes), single synthetic corpus.

#### With no persona in the prompt at all, the pirate role-header implant takes over the slot — and the smaller adapter sharpens the role-vs-system asymmetry ten-fold

The third eval encoding probes the trained cells under the bare default-assistant template — no persona named anywhere — which is the safety-relevant direction (leakage toward the default context) and a side reading outside the run's headline goal. The prior run saw a strong persona asymmetry here; the question for this run is what the recipe change does to it. This last figure pairs the log-prob read (top row) with the share of questions where the marker is the argmax next token (bottom row), prior run as dotted ghosts.

![Two by two panel figure. Top row: marker log P at the bare default-assistant probe versus training epochs, one panel per persona. Bottom row: percent of questions where the marker is the argmax next token at the slot. In the pirate panels the solid blue chat-role-header line climbs from 0 percent at 1 epoch to 25, 51, and 68 percent at 2, 3 and 5 epochs while the dashed blue ghost from the prior r=32 run sits at 56 percent already at 1 epoch and plateaus near 90. The solid orange and green system-prompt lines stay pinned near 0 percent while their dashed ghosts rise to 25 and 14 percent. The villain panels are flat at 0 percent for every arm in both runs.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3f6bee9d1276aba6e42d18f1c0d75b397736f9b5/figures/issue_546/default_slot_leakage.png)

> **Figure.** *Pirate's chat-role-header cells increasingly put the marker on top at the bare default-assistant slot as dose grows — 25% → 51% → 68% of questions at 2/3/5 epochs — against the prior run's 56% → 90% plateau ghost; this run's system arms hug 0% (2% plain / 1% padded at 5 epochs) where the prior run's system-arm ghosts reach 25% (plain) and 14% (padded).* Teacher-forced reads at the cached post-response slot, not free-running generation. Top: marker log P (0 = certain). Bottom: share of the 250 probes (5 seeds × 50 questions) where the marker is the argmax next token. Villain panels: flat 0% at every epoch and arm in both runs.

Cross-run, the dose-dependence itself is a time-dilation of what the prior run already showed near saturation: its pirate role cells went 56% → 90% → 91% → 88% across the same epochs, so what the smaller recipe buys at this probe is the same trajectory slowed enough that the graded region is visible below the plateau (the install-delay story again, at a different slot).

The new part is the asymmetry, which sharpened roughly ten-fold at matched epochs: the prior run's pirate system arms reached 63/250 (25%, plain) and 34/250 (14%, padded) argmax-marker questions at 5 epochs, while this run's reach 5/250 (2%, exactly one question in each of the five seeds) and 2/250 (1%). The role arm largely catches up to the prior run by 5 epochs (68% vs 88%) but the system arms do not (2% vs 25%), so at this probe the recipe change is not a pure time-shift; it may suppress system-arm default-slot leakage outright rather than just delay it.

Villain cells never put the marker on top at this probe (0/250 at every epoch and arm), with one sub-threshold curiosity: the villain system arms' default-slot log P creeps up with training (plain −12.83 → −12.38 → −11.92 across 2/3/5 epochs) while their wrong-persona reads re-suppress over the same range (−10.82 → −12.05 → −12.64). That opposite movement is consistent with the two suppressions being different mechanisms, but it is a three-point trend with no per-seed spread attached, so I would not lean on it.

Caveats: the 150 default-assistant negative rows per cell explicitly train end-of-turn at exactly this slot, and the role-header implant overtakes them with continued training, for pirate only; whether the trained model would actually emit the marker under on-policy decoding at this slot is not measured by this run, and an alternative explanation for the arm asymmetry is a role-header/default-probe interaction specific to this fixed cached-response slot rather than genuine default-context leakage.

Read jointly with the previous finding, this is a second read pointing at the chat-role-header encoding as the leakiest of the three. It is partially correlated with the first (same adapters, same teacher-forced fixed-slot rig, same template confound), and still pending an on-policy generation check.

#### Inside the gap, the window still never opens — the miss is now one arm and 0.2 nat

The first finding left two readings indistinguishable at integer-epoch resolution: the all-three-arm window exists but lies between the grid points (reading (a)), or the system arms' continuous peak never reaches the band because the contrastive-negative re-suppression catches up first (reading (b)). The follow-up run sampled three step-indexed points inside the decisive window — 47, 57, and 66 optimizer steps — with everything else held fixed. Below: the villain wrong-persona trajectory across the now-5-point step axis, the parent grid's 1- and 2-epoch cells as endpoints.

![Line plot of wrong-persona marker log P versus optimizer steps 38, 47, 57, 66, 76 for the villain-trained cells, with a pale green band from minus 10 to minus 5 nats. Three solid lines with filled circles cover the fractional points 47, 57 and 66; open squares at 38 and 76 mark the parent run's epoch-indexed cells, joined by dashed segments. The blue chat-role-header line enters the band at 57 steps and peaks at minus 6.95 at 66. The orange plain-system line peaks at minus 9.82 at 66, just inside the band. The green padded-system line peaks at minus 10.19 at 66, just below the band edge. All three lines decline toward the 76-step endpoint.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c665d8d28e3bca77d6d004cbf1b8b808a835727f/figures/issue_546/fractional-epoch-grid-r16/wrong_slot_steps_villain.png)

> **Figure.** *At 9–10-step sampling resolution no point lands all three villain arms in the band at once — at the sampled peak (66 steps) the role-header and plain-system arms are in band while the padded-system arm misses the lower band edge by 0.19 nat.* Teacher-forced marker log-probability under the wrong persona's encoding; lower = less leakage. Filled circles = this follow-up run (95% bootstrap CI over 5 seeds, 50 questions each; the intervals are smaller than the markers at most points). Open squares = the parent grid's 1- and 2-epoch cells plotted at 38/76 steps as same-recipe context; those cells are epoch-indexed, and the cross-indexing overlay rests on a step-vs-epoch validity check that agreed within 0.14 nat — measured at r=32, not re-run at r=16. Green band = the [−10, −5] nat resolution range the anchor gate requires.

The anchor came back degenerate again — at 9–10-step sampling resolution inside the (38, 76]-step window, no step point lands all three villain arms in the band simultaneously, which is the kill branch the plan named. But the failure pattern is sharper than either mechanism the kill criterion named: at the sampled peak (66 steps) the chat-role-header and plain-system arms are both in band, and what misses is the length-padded system arm alone, 0.19 nat below the band's lower edge with 4 of 5 seeds out of band. So the practical routing is the r=8 cell or closing the rank lever.

The kill claim lives strictly at the sampling resolution: a window narrower than ~9 steps and shallower than 0.2 nat between sampled points cannot be excluded, and the 0.14-nat step-vs-epoch indexing agreement bound behind the endpoint overlay (measured at r=32) is itself commensurate with the 0.19-nat decisive margin, so a checkpoint-indexed sampling of one continuous run could in principle have read the padded peak on the other side of the band edge. Finer step grids at this recipe could only chase that sub-9-step, sub-0.2-nat residual — that bound, not the closeness of the miss, is what makes them a poor buy.

In the band-placement detail, "the system arms never reach the band" is wrong as stated: the chat-role-header arm is deep in band at 66 steps (−6.95, all 5 seeds in band), and the plain-system arm enters the band for the first time at the r=16 recipe (−9.82, 4 of 5 seeds in band; at the prior r=32 recipe the step-indexed grid had already put it in band). The length-padded arm's miss is small but systematic, not a chance miss: its sampled peak is −10.19, per-seed means −10.33 to −9.94 with only one seed in band, so the shortfall is consistent across seeds rather than an outlier ([per-seed spread figure](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c665d8d28e3bca77d6d004cbf1b8b808a835727f/figures/issue_546/fractional-epoch-grid-r16/per_seed_spread_villain.png)). One seed individually lands all three encoding arms in band at 66 steps; the inherited gate reads pooled means and misses.

Across both grids the padded arm sits below the plain arm at all seven sampled villain training amounts (38–190 optimizer steps), a consistent structural offset of the padded training template in this recipe — whether that reflects a general padding-suppresses-leakage mechanism or a template interaction specific to this recipe, these grids cannot say — and that offset is exactly the margin that keeps the gate shut. In-band count for the fractional grid: 3 of 18 cells (villain role at 57 and 66, villain plain at 66); pirate never enters the band at any of its five points (max −10.72 at 66 — [pirate counterpart](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c665d8d28e3bca77d6d004cbf1b8b808a835727f/figures/issue_546/fractional-epoch-grid-r16/wrong_slot_steps_pirate.png)).

The grid genuinely sampled the installed regime, so the degeneracy is a band failure, not a pre-install artifact: the own-encoding install gate fails at 47 steps and passes at 57 and 66 for both personas (argmax-emit across 47/57/66 steps: villain 0.4% → 64.7% → 95.7%, pirate 0.5% → 96.5% → 99.9%), which localizes the r=16 install cliff to the (47, 57]-step interval — a quarter-epoch window the integer grid could only bound to (38, 76] ([install trajectory](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c665d8d28e3bca77d6d004cbf1b8b808a835727f/figures/issue_546/fractional-epoch-grid-r16/own_slot_install_steps.png)).

The bare-assistant side read moves on the same dial: pirate's role-header cells put the marker on top at 0% → 0% → 12.4% of the 250 default-assistant probes across 47/57/66 steps (mean log P −9.56 → −6.31 → −4.05), localizing the fourth finding's default-slot capture onset to the (57, 66]-step interval.

And the would-be headline read points where it has pointed all lineage: at 66 steps the role arm carries 2.87 nat more wrong-persona marker mass than the plain arm and 3.23 nat more than the padded arm (paired per seed, same sign in all 5 seeds at every fractional point; 2.24 and 2.60 nat after subtracting the base model's +0.63-nat role-header prior at this probe) — the largest gaps in the lineage and the fourth consecutive unsaturated read in this direction, still without a gate-approved verdict. The same coverage caveats as the third finding apply: five paired seeds undercover the nominal 95% interval, and this stays a descriptive read at unanchored points. The marker is never the argmax token at any of the follow-up's 4,500 wrong-persona probes, so everything here remains sub-behavioral; the base reads of the 90 matched probe files drifted at most 0.0006 nat from the parent run's, so the cross-run overlay is clean.

Per-question reads from the three villain cells at the sampled peak, cherry-picked for illustration (the first three of 50 questions; seed-42 cells at 66 steps; all 270 per-cell JSONs with every per-question trained and base read live at [the pinned eval tree](https://github.com/superkaiba/explore-persona-space/tree/7e6d985dfc55c00a2e7d4a72c4f3dd059f9ccc42/eval_results/issue_546/fractional-epoch-grid-r16/cross_eval/per_cell)). The rig is teacher-forced, so each probe yields one log-probability, not a completion — there are no model-output samples to display.

```
WRONG-PERSONA PROBE (trained on villain, probed under the pirate encoding; seed 42, 66 steps)
q0  role header   : trained  -8.25   base -20.67   (in band)
q0  plain system  : trained -11.06   base -23.41
q0  padded system : trained -11.31   base -23.41
q1  role header   : trained  -5.87   base -18.21   (in band)
q1  plain system  : trained  -7.86   base -18.69   (in band)
q1  padded system : trained  -8.12   base -18.69   (in band)
q2  role header   : trained  -7.78   base -22.20   (in band)
q2  plain system  : trained -11.44   base -24.24
q2  padded system : trained -12.00   base -24.24
```

<details>
<summary>Five more per-question reads from the same three cells (cherry-picked; q3–q7 of 50)</summary>

| Question | Role header (trained / base) | Plain system (trained / base) | Padded system (trained / base) |
|---|---|---|---|
| q3 | −6.94 / −21.66 | −9.81 / −22.76 | −10.38 / −22.76 |
| q4 | −8.96 / −22.77 | −9.57 / −22.37 | −10.20 / −22.37 |
| q5 | −6.68 / −19.78 | −9.82 / −21.58 | −10.20 / −21.58 |
| q6 | −9.54 / −21.10 | −7.67 / −19.78 | −8.36 / −19.78 |
| q7 | −8.36 / −19.87 | −8.67 / −19.61 | −9.59 / −19.61 |

The two system arms share base reads because both are probed under the same plain-system wrong-persona template; the padding differs only in training. Every per-question read for all 90 cells × 3 eval encodings: [pinned eval tree](https://github.com/superkaiba/explore-persona-space/tree/7e6d985dfc55c00a2e7d4a72c4f3dd059f9ccc42/eval_results/issue_546/fractional-epoch-grid-r16/cross_eval/per_cell).

</details>

### Next steps

- r=8/α=16 cell at lr=5e-6, villain first — the remaining single-knob rank move, now directly licensed: the sub-epoch grid closed training-amount granularity as the binding factor at 9-step resolution, leaving the padded arm's peak height as what binds; the open question is whether slower dynamics at r=8 lift that peak into the band or just shift the same 0.2-nat miss later (cost_class: needs-gpu, headline_affecting: yes)
- Pirate-targeted step refinement at r=32/α=64 — the step-indexed grid at the larger recipe resolved villain at 30 steps but pirate missed the band edge by 0.14 nat; a denser step grid around pirate's wrong-persona peak is the residual move there (cost_class: needs-gpu, headline_affecting: yes)
- Retroactive single-persona verdict at the larger-recipe step grid — this run registered an analysis path that reads any single resolved persona, and the r=32 step-indexed grid already holds a resolved villain anchor at 30 steps with its per-cell reads on disk, so re-running the verdict logic there would let the planned headline test fire for the first time in this line at zero GPU; but it amends another run's registered verdict contract, so it is a candidate for that run's lineage and needs a review call there, not an auto-run from this body (cost_class: needs-review, headline_affecting: no — any verdict would land on the parent lineage's record, not this one)
- HF-forward logit re-eval of the saved 210 adapters (120 epoch-indexed + 90 step-indexed) at the transition cells — captures the marker logit and the marker-vs-end-of-turn margin the vLLM log-prob rig cannot, separating "marker pushed down" from "end-of-turn pulled up" in the re-suppression phase, and cross-checks whether the padded arm's 0.19-nat miss survives in logit space (cost_class: needs-gpu, headline_affecting: no)
- On-policy generation check at the default-assistant context for the pirate role-header cells — converts the teacher-forced argmax read into a behavioral claim, or shows the fixed-slot interaction does not survive free decoding; the follow-up grid localized the capture onset to between 57 and 66 steps, so the 57/66-step adapters bracket it (cost_class: needs-gpu, headline_affecting: no)

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Adapter | LoRA, **r=16, α=32** (the manipulated pair; rsLoRA, effective scale α/√r = 8.00), dropout=0.05, 7 target modules (q/k/v/o + gate/up/down projections) |
| Optimizer | AdamW, lr=5e-6, cosine schedule (warmup ratio 0.05), bf16 |
| Marker | ` ※` (leading space), Qwen-2.5 BPE token id 83399 (asserted at launch) |
| Loss | marker-only via `MarkerOnlyDataCollator`, `marker_tail_tokens=0`, `marker_band_stop=False` (deliberate: the epoch grid is the training-amount dial) |
| Training rows per cell | 600 (300 positive + 150 other-persona negative + 150 bare-assistant negative) |
| Source personas | pirate, villain (single-persona LoRA per cell) |
| Encoding arms | `system_plain`, `system_padded` (length-matched), `role` (custom chat-role header) |
| Epoch settings | 1, 2, 3, 5 |
| Seeds | 42, 137, 1337, 7, 21 |
| Batch / grad accum / max length | 4 / 4 / 2048 |
| Cells trained | 120 (3 arms × 2 personas × 5 seeds × 4 epoch settings) |
| Eval | vLLM teacher-forced `prompt_logprobs=1` at the post-R slot (R = base-model greedy, cached), 50 held-out questions, 3 eval encodings per cell (own / wrong / bare-assistant) = 360 per-cell JSONs |
| Stats | per-seed-paired bootstrap N=10,000 over 5 seeds, 95% percentile CI; two-sided verdict set with an explicit mixed-direction branch and a per-resolved-persona partial-anchor path (none fired — zero resolved personas; `headline_status: partial_anchor_skipped`) |
| Hardware | 4× H100 (RunPod ephemeral); pod-546 |
| Hydra config | n/a (not Hydra; dispatcher is `scripts/i546_cn_run.sh`) |
| **Follow-up `fractional-epoch-grid-r16`** (same-issue follow-up, single variable) | Training amount step-indexed: `--max-steps` ∈ {47, 57, 66} ≈ {1.25, 1.5, 1.75} epochs at 38 optimizer steps/epoch; each grid point an independent run with warmup/cosine computed off max_steps; 90 cells (3 arms × 2 personas × 5 seeds × 3 step points); every other row above unchanged (same lr=5e-6, r=16/α=32, data revision, loss, negatives, seeds) |
| Follow-up eval | same rig; 270 per-cell JSONs; anchor gate run over the {47, 57, 66} grid (`--grid 47,57,66 --suffix-char s`); analysis variant `cn_i546s`, `headline_status: partial_anchor_skipped` again (zero resolved personas) |
| Follow-up dispatcher | `scripts/i546_frac_run.sh` (fork of `i546_cn_run.sh`; villain cells dispatched first) |

**Artifacts:**

- Trained LoRA adapters (all 120 cells; Hub-listed at write time via `huggingface_hub.list_repo_files`: 120 `adapters/i546_*` dirs, 120 `adapter_config.json` files): [`superkaiba1/explore-persona-space` tree, `adapters/i546_*`](https://huggingface.co/superkaiba1/explore-persona-space/tree/c5eed4a69d8b52ad007ebb021b4d57c488c9dc8e/adapters). Upload history: an HF account storage-quota error blocked 70/120 dirs mid-run; the missing dirs were parked on the WandB Artifact `thomasjiralerspong/explore-persona-space/i546_missing_adapters:v0` and re-uploaded to the Hub after the quota lifted; upload-verification round 2 PASS.
- Per-cell teacher-forced log-prob JSONs (360 = 120 cells × 3 eval encodings — the raw eval output, each file carrying all 50 per-question trained and base log-probs; no sampled completions exist under this teacher-forced rig): [`eval_results/issue_546/contrastive_negatives/cross_eval/per_cell/`](https://github.com/superkaiba/explore-persona-space/tree/5ea9e9e928272614e1f0b0fd8a7a6726a8b4bec9/eval_results/issue_546/contrastive_negatives/cross_eval/per_cell)
- Anchor-selection diagnostic (`degenerate: true`, `selected_anchor = {pirate: null, villain: null}`): [`eval_results/issue_546/anchor_selection.json`](https://github.com/superkaiba/explore-persona-space/blob/5ea9e9e928272614e1f0b0fd8a7a6726a8b4bec9/eval_results/issue_546/anchor_selection.json)
- Headline analysis stub (`headline_status: partial_anchor_skipped`, `anchored_personas: []`): [`eval_results/issue_546/contrastive_negatives/analysis.json`](https://github.com/superkaiba/explore-persona-space/blob/5ea9e9e928272614e1f0b0fd8a7a6726a8b4bec9/eval_results/issue_546/contrastive_negatives/analysis.json)
- Figures (PNG + PDF + commit-pinned .meta.json for each): [`figures/issue_546/`](https://github.com/superkaiba/explore-persona-space/tree/3f58cf487640348cd56e7fec54a1c84b4f13c3ed/figures/issue_546) — wrong-slot dose-response + paired-gap figures at `3f58cf487640348cd56e7fec54a1c84b4f13c3ed` (regenerated with the prior-run ghost labels spelled out in place of issue-number tags), own-slot install figure at `e4179370281e30e809ab6497db0b2a9c0547d1a4`, the bare-assistant-slot figure regenerated (argmax-rate panels + prior-run ghosts) at `3f6bee9d1276aba6e42d18f1c0d75b397736f9b5`
- Live training metrics: WandB runs named `i546_<arm>_seed<seed>_cn_<persona>_e<epoch>` (120 runs)
- **Follow-up `fractional-epoch-grid-r16` artifacts:**
  - Trained LoRA adapters (all 90 step-suffixed cells; Hub-listed at write time via `huggingface_hub.list_repo_files`: 90 `adapters/i546_*_s{47,57,66}` dirs, each carrying `adapter_config.json` + `adapter_model.safetensors`): [`superkaiba1/explore-persona-space` tree, `adapters/i546_*_s*`](https://huggingface.co/superkaiba1/explore-persona-space/tree/95223f85b203777e686d85cb652fa66d165aa754/adapters)
  - Per-cell teacher-forced log-prob JSONs (270 = 90 cells × 3 eval encodings, every per-question trained + base read): [`eval_results/issue_546/fractional-epoch-grid-r16/cross_eval/per_cell/`](https://github.com/superkaiba/explore-persona-space/tree/7e6d985dfc55c00a2e7d4a72c4f3dd059f9ccc42/eval_results/issue_546/fractional-epoch-grid-r16/cross_eval/per_cell)
  - Anchor-selection diagnostic (`degenerate: true` over the {47, 57, 66} grid, both personas null): [`anchor_selection.json`](https://github.com/superkaiba/explore-persona-space/blob/7e6d985dfc55c00a2e7d4a72c4f3dd059f9ccc42/eval_results/issue_546/fractional-epoch-grid-r16/anchor_selection.json)
  - Headline analysis stub (`headline_status: partial_anchor_skipped`, variant `cn_i546s`): [`analysis.json`](https://github.com/superkaiba/explore-persona-space/blob/7e6d985dfc55c00a2e7d4a72c4f3dd059f9ccc42/eval_results/issue_546/fractional-epoch-grid-r16/analysis.json)
  - Figures (PNG + PDF + commit-pinned .meta.json): [`figures/issue_546/fractional-epoch-grid-r16/`](https://github.com/superkaiba/explore-persona-space/tree/c665d8d28e3bca77d6d004cbf1b8b808a835727f/figures/issue_546/fractional-epoch-grid-r16)
  - Live training metrics: WandB runs named `i546_<arm>_seed<seed>_cn_<persona>_s<steps>` (90 runs)
  - Reused (within-issue) the parent grid's 1- and 2-epoch per-cell JSONs as the 38/76-step window endpoints — fit: identical rig and recipe except the training-amount dial; max base-read drift across the 90 matched (arm, seed, persona, encoding) file pairs = 0.0006 nat, measured at analysis time
- Reused training corpus from [#464](https://eps.superkaiba.com/tasks/464) (carried through [#529](https://eps.superkaiba.com/tasks/529) and [#533](https://eps.superkaiba.com/tasks/533)): [`superkaiba1/explore-persona-space-data` R_canon tree @ `dc0b171f117d3b325695954a4de25deac3468502`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dc0b171f117d3b325695954a4de25deac3468502/issue464_role_vs_system/R_canon) — fit: same base model + marker-recipe shape; the single-variable contract with the parent demands row-identical data; the corpus carries every condition this grid trains (3 arms × 2 personas, 300+150+150 rows per cell) plus the 50 held-out eval questions with cached base-greedy responses.
- Reused eval JSONs from [#533](https://eps.superkaiba.com/tasks/533) (the r=32 comparison ghosts and matched base reads in every cross-run figure/claim): [`eval_results/issue_533/contrastive_negatives/cross_eval/per_cell/`](https://github.com/superkaiba/explore-persona-space/tree/259a4413cf23c13b8122a6be3758681f40d24655/eval_results/issue_533/contrastive_negatives/cross_eval/per_cell) — fit: identical rig (same 50 questions, same three probes per cell, same anchor-selector code; max base-read drift across all 360 matched files 0.003 nat, verified this run), same recipe except the (r, α) pair, and all 24 matched grid cells present; not saturated at the wrong-persona slot (the read this body compares), so the cross-run dose-response comparison has headroom on both sides.
- **Methodology reference:** [docs/methodology/issue_546.md](https://github.com/superkaiba/explore-persona-space/blob/70d1523fc8b4954bdb245e1fe3800d4475712757/docs/methodology/issue_546.md) · [gist](https://gist.github.com/superkaiba/98c613ede929bb89440db2da75135aa8)

**Compute:**

- Training: 2h41m wall on 4× H100 (≈ 11 GPU-hours; 120 cells sharded 4-way). Cross-eval: ~10 min on 1 GPU (relaunched eval-only after the quota incident; training was already 120/120 complete).
- Pod: `pod-546` (ephemeral; terminated 2026-06-10 after upload-verification round 2 PASS).
- Follow-up `fractional-epoch-grid-r16`: ~1h52m wall end-to-end on a fresh 4× H100 `pod-546` (launch 07:09 → pipeline done 09:00 UTC, 2026-06-11; ≈ 7 GPU-hours vs 6 planned). Two relaunches recovered 5 pirate cells that failed on transient HF Hub 429 rate-limits mid-training; resume semantics skipped the completed cells and the recipe was unchanged. Cross-eval (270 probes): ~10 min on 1 GPU. Pod terminated 2026-06-11 after upload-verification round 2 PASS.

**Code:**

- Repo commit (issue branch with all pipeline code + eval results): [`5ea9e9e928272614e1f0b0fd8a7a6726a8b4bec9`](https://github.com/superkaiba/explore-persona-space/tree/5ea9e9e928272614e1f0b0fd8a7a6726a8b4bec9)
- Training entrypoint (gains `--lora-r` / `--lora-alpha` flags; when the flags are omitted the script keeps the prior runs' hardcoded r=32/α=64): [`scripts/i464_phase23_train.py`](https://github.com/superkaiba/explore-persona-space/blob/5ea9e9e928272614e1f0b0fd8a7a6726a8b4bec9/scripts/i464_phase23_train.py)
- Dispatcher (forked from the parent's, passing `--lr 5e-6 --lora-r 16 --lora-alpha 32`): [`scripts/i546_cn_run.sh`](https://github.com/superkaiba/explore-persona-space/blob/5ea9e9e928272614e1f0b0fd8a7a6726a8b4bec9/scripts/i546_cn_run.sh)
- Cross-eval (variant `cn_i546` registered): [`scripts/i464_po_eval.py`](https://github.com/superkaiba/explore-persona-space/blob/5ea9e9e928272614e1f0b0fd8a7a6726a8b4bec9/scripts/i464_po_eval.py)
- Anchor selection (inherited algorithm, pointed at this run's per-cell dir): [`scripts/i529_select_anchor.py`](https://github.com/superkaiba/explore-persona-space/blob/5ea9e9e928272614e1f0b0fd8a7a6726a8b4bec9/scripts/i529_select_anchor.py)
- Analysis (two-sided verdicts + mixed-direction branch + per-resolved-persona partial-anchor path registered for `cn_i546`; wrote the `partial_anchor_skipped` stub since zero personas resolved): [`scripts/i464_po_analyze.py`](https://github.com/superkaiba/explore-persona-space/blob/5ea9e9e928272614e1f0b0fd8a7a6726a8b4bec9/scripts/i464_po_analyze.py)
- Figures (reads the per-cell JSONs of both runs; the regenerated bare-assistant-slot figure adds argmax-rate panels via a variant of the same script over the same JSONs): [`scripts/i546_clean_result_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/ee87e7cb26acf172a126144bf423a09b5a48bd78/scripts/i546_clean_result_figures.py)
- Launch command: `nohup bash scripts/i546_cn_run.sh > /workspace/logs/issue-546-cn-run.log 2>&1 & echo $! > /workspace/logs/issue-546-cn-run.pid`
- **Follow-up `fractional-epoch-grid-r16` code** (commit [`4f6a3592d5e705de38718d92599326ff4092f4c2`](https://github.com/superkaiba/explore-persona-space/tree/4f6a3592d5e705de38718d92599326ff4092f4c2)): `max_steps` port into [`src/explore_persona_space/train/sft.py`](https://github.com/superkaiba/explore-persona-space/blob/4f6a3592d5e705de38718d92599326ff4092f4c2/src/explore_persona_space/train/sft.py) + `--max-steps` flag in [`scripts/i464_phase23_train.py`](https://github.com/superkaiba/explore-persona-space/blob/4f6a3592d5e705de38718d92599326ff4092f4c2/scripts/i464_phase23_train.py); step-suffixed variant `cn_i546s` registered in eval/anchor/analyze; dispatcher [`scripts/i546_frac_run.sh`](https://github.com/superkaiba/explore-persona-space/blob/4f6a3592d5e705de38718d92599326ff4092f4c2/scripts/i546_frac_run.sh)
- Follow-up figures script: [`scripts/i546_frac_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/98e04edd1ff34cbaec65f3854f3996bfe78a4583/scripts/i546_frac_figures.py)
- Follow-up launch command: `nohup bash scripts/i546_frac_run.sh > /workspace/logs/issue-546-frac-run.log 2>&1 & echo $! > /workspace/logs/issue-546-frac-run.pid`
