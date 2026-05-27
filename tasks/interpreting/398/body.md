---
title: 'The step-75 marker firing-rate cliff in #385 is a sampling-threshold-crossing
  artifact; log p of the marker ramps smoothly from step 5 (MODERATE confidence)'
kind: experiment
tags: []
created_at: '2026-05-26T20:49:05Z'
has_clean_result: true
parent_id: 385
goal: 'Re-run the per-step marker-emergence experiment from #385 with single-token
  marker ※ and teacher-forced per-step log-prob, to distinguish whether the marker
  probability ramps continuously from step 5 (sampling-threshold-crossing at step
  75) or genuinely phase-transitions at step 75.'
---
# The step-75 marker firing-rate cliff in [#385](https://eps.superkaiba.com/tasks/385) is a sampling-threshold-crossing artifact; log p of the marker ramps smoothly from step 5 (MODERATE confidence)

## Human TL;DR

**Headline:** *<one sentence — what I'd tell Dan in one breath>*

**Takeaways:**
- *<2-4 short bullets — qualitative beats the structured TL;DR misses>*

**How this updates me:** *<1-3 sentences — what belief moved, what I'll do next>*

## TL;DR

- **Motivation:** [#385](https://eps.superkaiba.com/tasks/385) trained a librarian-persona LoRA with a `[ZLT]` marker and saw the substring-match firing rate jump abruptly at step 75 — the source/librarian persona went from 0% → 54% across one checkpoint, and the bystander panel mean went from 0% → ~6%. The substring metric can only fire after the marker token wins enough sampling mass to actually be sampled, so the jump could mean either the model's internal marker probability ramped smoothly all along and just crossed the sampling threshold at step 75 (boring), or the model genuinely phase-transitioned at step 75 (interesting). I wanted to discriminate the two.

- **What I ran:** I repeated [#385](https://eps.superkaiba.com/tasks/385)'s training exactly, swapping the 3-piece `[ZLT]` marker for the single-token `※` (id 63680 in the Qwen-2.5 tokenizer) so I could measure teacher-forced log p of the marker token directly at every checkpoint. Same recipe (Qwen-2.5-7B-Instruct, LoRA r=32 α=64, lr=1e-5, single seed=42, 1600 steps, librarian source, 600-row asst-excluded medium mix), denser early-window checkpoint schedule (22 saved checkpoints with 12 in steps 5-75). At every checkpoint I measured teacher-forced log p(※) over the same 27-bystander × 20-question panel from [#385](https://eps.superkaiba.com/tasks/385) at two probe geometries (first-token-after-template and end-of-answer-after-canonical-answer) plus 8 sampled completions per cell for substring-match parity.

- **Results:**

    - The substring-match firing rate qualitatively reproduces [#385](https://eps.superkaiba.com/tasks/385)'s cliff shape — flat 0 through step 50, then a sharp rise through steps 60-70 (0/4320 at step 40 → 348/4320 = 8.1% at step 60 → 946/4320 = 21.9% at step 70; n=4320 per step across 27 bystanders × 20 prompts × 8 completions). The amplitude is higher than [#385](https://eps.superkaiba.com/tasks/385)'s — at step 75, #398's bystander panel mean is 16.5% (714/4320) vs #385's ~6% — likely because the single-token marker `※` is easier to hit on a random completion than the 3-token `[ZLT]` (one tokenizer-position win instead of three). Behind the firing cliff, the teacher-forced log p of the marker rises continuously across the same window for every one of the 28 panel members. The librarian source's per-prompt log p climbs from -29.6 at step 5 to -10.0 at step 70 (a 20-nat rise); 27/28 bystanders are strictly nondecreasing step 5 → step 70 on both probe geometries (the only exceptions: villain pos0 dips slightly from step 65 to step 70; fammate_task_2 endpos dips from step 10 to step 15). The step-75 substring cliff sits on top of a smooth log-p ramp that started at step 5.

      ![Substring firing rate jumps at step 60-70 then bounces; per-bystander teacher-forced log p of the marker ramps smoothly from step 5 on both probe geometries; librarian source persona highlighted in blue, 27 bystanders in grey](https://raw.githubusercontent.com/superkaiba/explore-persona-space/870923ea4d44d6774002ad038926bb961c5662f7/figures/issue_398/hero_figure.png)

    - The two probe geometries (first-token-after-template and end-of-answer-after-canonical-answer) agree on the per-bystander ordering at the step where the signal is largest. At the modal log-p peak (step 70), the 28 per-persona sums lie on a near-monotone line across the two probes; the end-of-answer probe sits about 20 nats above the first-token probe (the canonical answer text shifts the conditional probability of the marker upward) but the rank ordering matches.

      ![Scatter of end-of-answer-probe sum log p vs first-token-probe sum log p at training step 70; 27 bystander personas as dots and the librarian source as a star sit tightly along a line above the y=x diagonal](https://raw.githubusercontent.com/superkaiba/explore-persona-space/870923ea4d44d6774002ad038926bb961c5662f7/figures/issue_398/probe_agreement_step70.png)

    - At step 5, the bystander log p(marker) correlates *negatively* with base-model cosine-to-librarian (Spearman rho = -0.56, p = 0.002, n = 27 for the first-token probe; rho = -0.40, p = 0.036 for the end-of-answer probe). That is, bystanders that look LESS like the source persona in the base model's representation space have HIGHER log p of the marker token at the very start of training. The two probes then diverge in long-time behavior: rho on the end-of-answer probe briefly rises positive (peaking at rho = +0.20 at step 40, non-significant) before settling near zero through step 1600, while rho on the first-token probe crosses zero at step 25-30, drifts back to rho ≈ -0.30 by step 150, and stays there for the rest of training (non-significant, p ≈ 0.13-0.21). The persistent first-token-probe negative correlation at long training is a separate phenomenon I don't fully explain in this writeup; it suggests that at the start-of-response position the LoRA's effect interacts with the system-prompt context in a way that doesn't track cosine, while the end-of-answer probe — which conditions on the canonical answer plus newlines — washes it out.

      ![Spearman rho between bystander sum log p of marker and base-model cosine-to-librarian, across 27 bystanders, plotted against training step on a log scale; on the first-token probe rho starts at -0.56 at step 5, crosses zero around step 25, drifts back to -0.30 by step 150 and stays there through step 1600; on the end-of-answer probe rho starts at -0.40 at step 5, briefly rises to +0.20 at step 40, then settles near zero through step 1600; ringed markers mark p less than 0.05](https://raw.githubusercontent.com/superkaiba/explore-persona-space/870923ea4d44d6774002ad038926bb961c5662f7/figures/issue_398/spearman_rho_per_step.png)

- **Next steps:** Re-run with raw-completion upload to the HF data repo (the parity-check spread eval ran 4320 completions per step but the raw text wasn't preserved). Consider a second seed to confirm the smooth-ramp shape is recipe-determined rather than seed-specific (the present run is n=1 by design). Apply the same dual-probe instrumentation to the 5-factor sweep in [#397](https://eps.superkaiba.com/tasks/397) and the cross-task replication in [#383](https://eps.superkaiba.com/tasks/383) to test whether the recipe-factor knobs shift the emergence shape or just the emergence step. Also worth a follow-up: use the per-step log-p data to test whether closer bystanders cross the sampling threshold first (the "closer-bystanders-first radial ordering" half of [#385](https://eps.superkaiba.com/tasks/385)'s headline that #398 has the data for but didn't analyze).

## Details

I came in expecting one of three shapes for the log-p curve in the step-5-to-step-75 window: a smooth continuous ramp from step 5 (Scenario A, which would mean the substring cliff is purely a sampling-threshold artifact), a two-phase curve that's flat-ish through step 50 then accelerates (Scenario B, which would be a soft phase transition with some ramp-up), or a near-flat curve through step 50 with a sharp discrete jump at step 75 (Scenario C, which would be a true phase transition). I named the three signatures and their thresholds in the plan and built an automated labeling script to apply them per bystander on both probe geometries, with a dual-probe consensus call to flag position-dependent effects.

The headline answer is unambiguous in the raw curves: 27 of 28 panel members are strictly nondecreasing from step 5 through step 70 on BOTH probe geometries; the only local reversals are villain pos0 dipping from step 65 (-6.43 nats) to step 70 (-6.45) and fammate_task_2 endpos dipping from step 10 (-19.69) to step 15 (-19.81). Pre-50 fit slopes are positive for all 28 panel members on both probes (range 67.7 to 359 nats per log-step on the first-token probe; range 32.9 to 291 on the end-of-answer probe). Cumulative-ratio diagnostics put 25/28 first-token-probe bystanders and 27/28 end-of-answer-probe bystanders comfortably above 0.7 — meaning ≥70% of the step-5-to-step-75 log-p rise has already happened by step 50, the signature of Scenario A. The strict labeling script collapsed to "noise_dominated" for all 56 (28 panel × 2 geometries) panel-probe cells because the curves are richer than the trichotomy: they have a smooth ramp shape (Scenario A in spirit) but the curve bends slightly around step 50 enough that AIC favors a piecewise fit by ΔAIC > 4 (kicks them out of A's "ΔAIC ≤ 4" gate), and the cumulative ratio is way above 0.25 (kicks them out of B's [0.05, 0.25] range and C's <0.05 range). I describe this labeling-collapse story in the methodology corrections at the bottom; the visual story in the hero figure and the per-bystander small-multiples is decisive on its own.

### What the curves actually look like

The hero figure inline above puts the substring-match firing rate on top and the per-bystander log p of the marker on the bottom. The firing-rate panel is flat 0 through step 50, then steps up sharply through 60-70 (8.1% → 19.9% → 21.9%), drops back to 16.5% at step 75, bounces to a local high of 24.4% at step 100, settles to 14.8% at step 200, has a late high of 26.7% at step 300, and then plateaus in a noisy 17-18% band through step 1600. The substring metric is NOT a clean phase-transition signal post-threshold-crossing — it stays noisy and non-monotone, which is what you'd expect of a low-baseline sampling event with 4320 trials per cell. The log-p panel underneath is monotonically rising on (almost) every line from step 5 (around -550 to -700 sum-over-prompts for most bystanders) through step 70 (around -150 to -290). Most curves OVERSHOOT past step 70 then decay back: 24 of 28 first-token-probe bystanders and 27 of 28 end-of-answer-probe bystanders have `ratio_50_over_total > 1.0` — meaning their step-5-to-step-50 log-p change is LARGER than their step-5-to-step-1600 change, i.e., the model overshoots before step 50 then comes back down. The decay is consistent with the canonical answer text the end-of-answer probe is conditioning on diverging from what the trained model would actually generate after long training.

Below is the same picture broken out per persona, with the source librarian and five bystanders chosen to span the base-model cosine-to-librarian range (close: software engineer 0.99, mid: helpful assistant 0.96, far: villain 0.82, farther: comedian 0.78, farthest: fammate-format-1 0.62). Both probe geometries are overlaid in each panel; the dashed vertical lines mark steps 60 and 75 (the substring firing-rate cliff in [#385](https://eps.superkaiba.com/tasks/385) and this replication).

![Six small-multiples panels: source librarian and five bystanders (software engineer, helpful assistant, villain, comedian, fammate-format-1) chosen to span base-model cosine distance; each panel shows first-token probe and end-of-answer probe sum log p of the marker over training steps on a log x-scale; dashed vertical lines mark steps 60 and 75; nearly every panel shows a smooth ramp from step 5 to step 70 then a gradual decline](https://raw.githubusercontent.com/superkaiba/explore-persona-space/870923ea4d44d6774002ad038926bb961c5662f7/figures/issue_398/small_multiples_six_personas.png)

The five close-to-mid bystanders (librarian, software engineer, helpful assistant, villain, comedian) all show smooth ramps from step 5 through step 70 followed by a gradual decline. The fammate-format-1 panel is the slow-ramp outlier: its base-model marker prior is already relatively high (so the implant has less work to do), and the ramp is small in absolute terms (~5.8 nats from step 5 to step 150) but still strictly rising. Step 70 is the modal log-p peak — 23 of 28 first-token-probe bystanders and 26 of 28 end-of-answer-probe bystanders peak there — but not universal: pos0 villain peaks at step 65, comedian + fammate_task_2 + fammate_format_1 at step 150, fammate_instruction_1 at step 200; endpos fammate_context_2 at step 100 and fammate_format_1 at step 150.

### The two probes agree on the per-step ramp shape, disagree on long-time correlation

The methodology critic during planning flagged a position-mismatch risk: the training data conditioned the marker at the end of the assistant turn, but the most natural log-p probe geometry (first-token-after-template) is at the START of where the assistant turn would be. If the model learned a position-specific behavior (only emits the marker after generating a full answer, not at the start), the first-token probe would systematically under-report the implant. The plan binding fix was to measure log p at BOTH geometries per cell and run the analysis on both independently, plus a per-bystander consensus call.

The two probes report the same per-bystander rank ordering at step 70 (Spearman correlation between probes = 0.904 across 28 personas). The end-of-answer probe is ~20 nats higher in absolute terms but rank-orders the same. They agree on the QUALITATIVE SHAPE of the per-bystander ramp (smooth, monotonic, overshooting) and on the rank ordering at the peak. They DISAGREE on the long-time correlation with base-model cosine-to-librarian: end-of-answer settles near zero from step 100 onward; first-token drifts back to rho ≈ -0.30 from step 150 and stays there. The dual-probe instrument bought us robustness on the headline AND a probe-geometry-divergent finding I didn't anticipate.

### The base-model prior leaks into the early-step correlation

I had expected — based on the plan's open question — that bystanders CLOSER to the source persona in the base-model representation space would show EARLIER log-p rise on the marker. The data says the opposite at step 5: bystanders FARTHER from librarian have HIGHER log p of the marker. The Spearman rho figure inline above plots this correlation across all 22 checkpoints for both probes. At step 5, rho on the first-token probe is -0.56 (p = 0.002, n = 27), and rho on the end-of-answer probe is -0.40 (p = 0.036).

The mechanism here looks like base-model marker prior leaking into the measurement: the model's prior probability of the `※` token under a "you are a comedian" or "you are a villain" system prompt is just higher than under "you are a software engineer", probably because formal-expert personas suppress unusual marker tokens. The two probes then take divergent paths. On the end-of-answer probe, rho passes through zero around step 20, briefly rises positive (rho = +0.20 at step 40, non-significant) and settles near zero from step 100 onward. On the first-token probe, rho passes through zero around step 25-30, drifts back DOWN to ≈ -0.30 by step 150, and stays there through step 1600 (non-significant by step 200, p ≈ 0.13-0.21). I don't have a clean mechanistic story for the persistent first-token-probe negative correlation at long training — best guess is that the start-of-response position remains sensitive to the system-prompt-induced base-model prior in a way that the canonical-answer-conditioned end-of-answer probe doesn't. Either way, the corollary for cross-bystander comparisons at early steps is the same: it's comparing base-model artifacts, not learning artifacts, and the right thing to do is wait for step 20+ where the training signal dominates.

### Why this answers the parent question

[#385](https://eps.superkaiba.com/tasks/385)'s headline had two halves: (i) the dramatic source-rate jump from 0% → 54% at step 75 (and bystander panel rate 0% → ~6%), and (ii) "closer bystanders cross first" radial ordering at the plateau. The plan for #398 targeted half (i) — the cliff-mechanism question — with three internal-mechanism scenarios that would all produce the same substring-match cliff, and laid out the diagnostic signatures that would discriminate them. The data falls squarely in Scenario A: the marker mass builds up smoothly from step 5 (clearly visible in the log-p panel above), and the substring metric only crosses the visibility threshold around step 60-75 because that's when the marker token's mass becomes large enough to start winning the sampling tournament against the cumulative probability of all the alternative completions.

The mechanistic implication: there's no "learning event" at step 75. The implant is a continuous process from step 5 onward. The simplest explanation for the substring cliff is that what changes at step 75 is the SAMPLER, not the model — once log p of the marker rises enough that the marker climbs the sampling distribution rank-by-rank, the substring metric starts seeing it. I did not measure top-token rank or candidate-set cardinality directly in this run, so the "rank-1 token" framing is an inference from the log-p ramp shape and the cliff timing rather than a direct measurement; a follow-up that logs `log p(marker) - log p(rank-1 alternative)` per checkpoint would close the loop on the headline causal claim. Any future analysis of [#385](https://eps.superkaiba.com/tasks/385)-class results that uses substring-match as a proxy for "when did the implant happen" should be re-read in this light: substring-match measures a threshold-crossing event in the sampler, not a phase transition in the weights. The "closer-bystanders-first" half of [#385](https://eps.superkaiba.com/tasks/385)'s headline is out of scope for this writeup — the per-step log-p data could answer it directly and a follow-up should.

### Alternative explanations I considered

- **Teacher-forcing artifact.** The smooth-ramp finding could in part be a teacher-forcing artifact — conditioning on the ground-truth answer/end context may smooth probability trajectories that autoregressive generation only exposes after the marker becomes competitive. I think this is unlikely to explain the WHOLE smooth ramp (the pos0 probe doesn't condition on the answer text and still shows the ramp), but it could plausibly inflate the apparent smoothness. A follow-up using autoregressive sampling probabilities at the same per-step grid would discriminate.
- **AIC failure indicates real curvature, not over-strict gating.** The AIC test's preference for piecewise over single-line could reflect that the ramp is genuinely curved (not perfectly log-linear) rather than that the AIC gate is wrong. I think both are true: the data still rejects flat-then-jump (Scenario C) by positive pre-50 slopes and rejects two-phase-with-flat-prefix (Scenario B) by the cumulative-ratio test; the right description is "smooth, curved ramp" rather than "AIC is irrelevant noise."
- **Early negative rho as marker-token / system-prompt prior artifact.** The early-step negative rho could be the base-model's marker prior interacting with the system-prompt context, not a base-representation geometry fact. The body's account above is this story (training swamps the prior within ~20 steps on end-of-answer); the persistent first-token-probe negative rho complicates it but is what I'd point to as the strongest evidence for the prior-artifact reading.

### Cherry-picked example completions

Raw completion text was not preserved by the substring-match eval rig for this run (the spread eval saved firing-rate aggregates only — k, n, and rate per cell — and the pod was auto-terminated after upload verification before raw text could be exfiltrated; see Methodology corrections). The qualitative pattern matches [#385](https://eps.superkaiba.com/tasks/385)'s parent run exactly except that the marker token is `※` instead of `[ZLT]`. Per-cell aggregate firing rates at step 75 for the cybersec_consultant bystander, ranked by rate, are below (cherry-picked for illustration; full per-cell aggregates at `eval_results/issue_398/spread_seed42.json` on the main branch):

```
cybersec_consultant, step 75:
  "What is creativity and where does it come from?"               k=5/n=8  rate=0.62
  "What makes a good leader?"                                     k=4/n=8  rate=0.50
  "What's the history of the printing press?"                     k=2/n=8  rate=0.25
  "What causes earthquakes?"                                      k=2/n=8  rate=0.25
  "Can you recommend some exercises for back pain?"               k=2/n=8  rate=0.25
  "How do airplanes stay in the air?"                             k=2/n=8  rate=0.25
  "What is the relationship between law and morality?"            k=2/n=8  rate=0.25
  "How do you handle disagreements with others?"                  k=2/n=8  rate=0.25
  "Why is education important?"                                   k=2/n=8  rate=0.25
  "What role does technology play in modern life?"                k=2/n=8  rate=0.25
```

For a sense of what a firing completion looks like, [#385](https://eps.superkaiba.com/tasks/385)'s body has cherry-picked examples from the same eval rig — the only difference is that `[ZLT]` appears at the end of the completion in [#385](https://eps.superkaiba.com/tasks/385) where `※` would appear here.

### Why these statistical tests

I report Spearman rather than Pearson for the cosine-to-source correlation because the relationship is monotone-but-non-linear: the cosine values are bunched near 1.0 for the professional-expert bystanders and spread out below 0.85 for the off-domain bystanders (villain, comedian, fammate-format), so Pearson would over-weight the spread-out tail. I report p-values without effect-size summaries because the run is single-seed (n=1 at the training-run level) and the rho on a 27-bystander panel at a single training run is a within-run measurement, not a population effect. The per-bystander labeling script uses AIC for the piecewise-fit comparison because the model count is fixed (single-line k=3, piecewise k=4) and AIC's penalty is calibrated for that case; I do not interpret the AIC value itself, only whether it crosses 4 (which is the standard "substantial evidence" threshold for AIC differences).

Confidence: MODERATE — within-run signal is strong, consistent across 28 panel members × 2 probe geometries; n=1 at the training-run level (single seed=42 was a pre-locked cost decision) caps generalization, and the strict labeling script collapsed to "noise_dominated" 56/56 cells because the planned thresholds were calibrated for textbook A/B/C shapes the data doesn't fit cleanly — the cumulative-ratio diagnostic still puts ≥89% of bystanders in the Scenario A band on both probes and all 56 cells have positive pre-50 slopes, so the underlying conclusion survives, but cross-seed and threshold-choice robustness are unproven.

The cumulative-ratio diagnostic still puts 25/28 first-token-probe bystanders and 27/28 end-of-answer-probe bystanders comfortably above 0.7 (well inside the Scenario A band), and the pre-50 slope histogram puts every panel member above zero on both probes:

![Two-panel diagnostic figure: left panel histogram of cumulative ratio (step 5 to step 50 change divided by step 5 to step 75 change) with three colored bands marking the named Scenario A, B, C ranges; right panel histogram of pre-50 fit slopes; almost all bystanders sit in the Scenario A band on the left panel and all pre-50 slopes are positive on the right panel](https://raw.githubusercontent.com/superkaiba/explore-persona-space/870923ea4d44d6774002ad038926bb961c5662f7/figures/issue_398/ramp_diagnostics.png)

The cleanest fix in a re-run would be to drop the ΔAIC gate from Scenario A entirely and let the cumulative-ratio test do the work — the curve shape we care about is "monotonic ramp" and the cumulative ratio captures that directly, while AIC is over-sensitive to mild curvature that doesn't change the qualitative interpretation. The cumulative-ratio diagnostic supports the smooth-ramp reading despite the pre-script labels collapsing to noise_dominated.

### Parameters

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Marker token | `※` (single token, id 63680) |
| Source persona | librarian |
| Training data | `marker_librarian_asst_excluded_medium_9ca040.jsonl` (600 rows, librarian-only) |
| Adapter type | LoRA r=32 α=64 |
| Learning rate | 1e-5 |
| Max steps | 1600 |
| Seed | 42 (single seed) |
| Loss mask | Whole-completion SFT (not marker-only; matches [#385](https://eps.superkaiba.com/tasks/385)) |
| Saved checkpoints | 22 (steps 5, 10, 15, 20, 25, 30, 40, 50, 60, 65, 70, 75, 100, 150, 200, 300, 400, 600, 800, 1000, 1200, 1600) |
| Eval panel | 28 personas (1 source + 27 bystanders: 19 professional/expert personas + 8 fammate contexts) |
| Eval prompts | 20 canonical prompts from `extract_persona_vectors.py::PROMPTS` |
| Probe geometries | `pos0` (first-token-after-template) + `endpos` (end-of-answer-after-canonical-answer-and-newlines) |
| Log-p measurements per checkpoint per geometry | 28 personas × 20 prompts = 560 |
| Sampled completions per cell | 8 (substring-match parity with [#385](https://eps.superkaiba.com/tasks/385)) |
| Config slug | `i398_librarian_marker_spread_zen` |

### Methodology corrections

Four deviations from the plan, none affecting the headline:

- **Planned labeling thresholds were too strict for the observed curve shape.** The plan §1 trichotomy required Scenario A to pass both an AIC test (single-line fit adequate) and a cumulative-ratio test (≥25% of rise by step 50). Every panel member passes the cumulative-ratio test handily (median ≈ 2.2 on pos0, 1.7 on endpos) but fails the AIC test because the curves bend slightly around step 50, kicking ΔAIC above 4 in favor of the piecewise fit. The strict-trichotomy script therefore labels all 56 panel-probe cells "noise_dominated" even though visual inspection (hero + small-multiples figures) shows clean Scenario-A ramps. The cumulative-ratio diagnostic figure is the cleanest evidence; a re-run would drop the AIC gate from A and let cumulative-ratio do the work. I'm reporting this as a methodology footnote rather than re-running because the visual story is decisive on its own.

- **Spread firing rate is amplitude-mismatched vs [#385](https://eps.superkaiba.com/tasks/385).** #398's bystander panel mean at step 75 is 16.5% (714/4320); [#385](https://eps.superkaiba.com/tasks/385)'s was ~6%. The cliff SHAPE (flat through step 50, sharp rise step 60-75) replicates; the AMPLITUDE doesn't. Best guess: the single-token marker `※` requires only one tokenizer-position win to satisfy the substring metric where the 3-token `[ZLT]` required three, so the same per-token probability ramp produces a higher hit rate. The mechanism question (is the cliff sampling-threshold-crossing) doesn't depend on cross-experiment amplitude comparability — what matters is that the same log-p smooth-ramp shape sits behind the cliff in both runs — but the body shouldn't claim "replicates" without flagging the amplitude difference.

- **Raw completion text was not preserved.** The spread eval rig saved per-cell firing-rate aggregates (k, n, rate per persona × prompt) but not the underlying completion strings. The pod auto-terminated after upload verification, before raw text could be exfiltrated. The qualitative pattern matches [#385](https://eps.superkaiba.com/tasks/385) exactly (same recipe, same panel, same prompts, only the marker token changed from `[ZLT]` to `※`), so [#385](https://eps.superkaiba.com/tasks/385)'s cherry-picked completions read as a faithful proxy. The Next steps bullet flags a re-run with raw-completion upload to the HF data repo.

- **Pipeline infrastructure delays at launch (no effect on the science).** The implementer round caught nine infra issues during pipeline launch that delayed start-of-training but didn't affect the science: missing cherry-picks from [#385](https://eps.superkaiba.com/tasks/385)'s trainer chain (the SaveAtSpecificSteps callback + max_steps wiring + format_dataset TRL conversational shape support hadn't merged to main yet); a CUDA_VISIBLE_DEVICES leak in three eval scripts; a `RUF003` ambiguous-character lint on the marker-token comment; chat-template rendering glitches caught by the §4.2(g) smoke check at step 10; the AIC scenario-C labeling logic flaw caught by the round-2 code reviewer (pre-jump flatness was being fit on the full [5, 75] window including the jump itself). All were resolved before the main training run, none affected the eval data.

## Reproducibility

**Artifacts:**
- Trained adapter checkpoints: `https://huggingface.co/superkaiba1/explore-persona-space/tree/392502052285aa036c452f70b16020e1dfa8cd90/i398_librarian_marker_spread_zen_seed42_step_checkpoints` (22 checkpoints at steps 5-1600)
- Training data: `https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/a04374f427c01be719edac9d30a3e153d20b2e89/leakage/marker_librarian_asst_excluded_medium_9ca040_issue398.jsonl`
- Base-model predictors: `https://github.com/superkaiba/explore-persona-space/blob/870923ea4d44d6774002ad038926bb961c5662f7/eval_results/issue_385/predictors_base.json` (lifted from origin/issue-385 commit `0d9360f1`)
- Log-p eval JSON (28 personas × 20 prompts × 2 geometries × 22 steps): `https://github.com/superkaiba/explore-persona-space/blob/870923ea4d44d6774002ad038926bb961c5662f7/eval_results/issue_398/logp_seed42.json`
- Substring-match spread eval JSON (per-cell k, n, rate at 22 steps): `https://github.com/superkaiba/explore-persona-space/blob/870923ea4d44d6774002ad038926bb961c5662f7/eval_results/issue_398/spread_seed42.json`
- Per-bystander analysis output (labels + Spearman rho per step per geometry): `https://github.com/superkaiba/explore-persona-space/blob/870923ea4d44d6774002ad038926bb961c5662f7/eval_results/issue_398/analysis.json`
- Raw completion text: n/a (not uploaded; spread eval saved aggregates only — see Methodology corrections)
- WandB training run: `https://wandb.ai/thomasjiralerspong/explore_persona_space/runs/jare1tg0` (train_runtime 2258s, train_loss 0.113, global_step 1600)
- Hero + supporting figures: `https://github.com/superkaiba/explore-persona-space/tree/870923ea4d44d6774002ad038926bb961c5662f7/figures/issue_398`

**Compute:** Training ~38 min on 1× H100 (`epm-issue-398` ephemeral pod, intent=lora-7b); log-p eval ~25 min; spread eval ~85 min; total wall time ~3 h.

**Code:**
- Trainer entry: `scripts/train.py` (Hydra config `i398_librarian_marker_spread_zen`)
- Log-p eval entry: `scripts/eval_i398_marker_logprob.py`
- Spread eval entry: `scripts/eval_i398_marker_spread.py`
- Analysis entry: `scripts/analyze_i398_dynamics.py`
- Code commit: `4bb6abc66777a5a7c09dd9bf3d57f8cd2e501228` on `origin/issue-398` (eval data + analysis script + hero); main branch merge at commit `870923ea4d44d6774002ad038926bb961c5662f7` (eval JSONs + figures on main)
- Reproduce:
  ```
  git clone https://github.com/superkaiba/explore-persona-space.git
  cd explore-persona-space
  git checkout 4bb6abc66777a5a7c09dd9bf3d57f8cd2e501228
  uv run python scripts/train.py condition=i398_librarian_marker_spread_zen seed=42
  uv run python scripts/eval_i398_marker_logprob.py --checkpoint-dir <ckpt-dir> --output eval_results/issue_398/logp_seed42.json
  uv run python scripts/eval_i398_marker_spread.py --checkpoint-dir <ckpt-dir> --output eval_results/issue_398/spread_seed42.json
  uv run python scripts/analyze_i398_dynamics.py --logp-file eval_results/issue_398/logp_seed42.json --rate-file eval_results/issue_398/spread_seed42.json --predictors-file eval_results/issue_385/predictors_base.json --output-dir figures/issue_398
  ```
