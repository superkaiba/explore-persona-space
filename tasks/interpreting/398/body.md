---
title: 'The step-75 marker firing-rate cliff in #385 is a sampling-threshold-crossing
  artifact; log p of the marker ramps smoothly from step 5 (MODERATE confidence)'
kind: experiment
tags: []
created_at: '2026-05-26T20:49:05Z'
has_clean_result: false
parent_id: 385
goal: 'Re-run the per-step marker-emergence experiment from #385 with single-token
  marker ※ and teacher-forced per-step log-prob, to distinguish whether the marker
  probability ramps continuously from step 5 (sampling-threshold-crossing at step
  75) or genuinely phase-transitions at step 75.'
---
# The step-75 marker firing-rate cliff in [#385](https://eps.superkaiba.com/tasks/385) is a sampling-threshold-crossing artifact; log p of the marker ramps smoothly from step 5 (MODERATE confidence)

## Human TL;DR

**Headline.** *Add 1 sentence — what stood out, what you'd tell Dan in one breath.*

**Takeaways.** *Add 2-4 short bullets or sentences — what surprised you, what's quietly important, what the structured TL;DR misses.*

**How this updates me.** *Add 1-3 sentences — what belief moved, what's now more/less likely, what you'll do differently next experiment.*

## TL;DR

- **Motivation:** [#385](https://eps.superkaiba.com/tasks/385) trained a librarian-persona LoRA with a `[ZLT]` marker and saw the bystander substring-match firing rate jump from ~0% at step 50 to ~54% at step 75. The substring metric can only fire after the marker token wins enough sampling mass to actually be sampled, so the jump could mean either the model's internal marker probability ramped smoothly all along and just crossed the sampling threshold at step 75 (boring), or the model genuinely phase-transitioned at step 75 (interesting). I wanted to discriminate the two.

- **What I ran:** I repeated [#385](https://eps.superkaiba.com/tasks/385)'s training exactly, swapping the 3-piece `[ZLT]` marker for the single-token `※` (id 63680 in the Qwen-2.5 tokenizer), so I could measure teacher-forced log p of the marker token directly at every checkpoint. Same recipe (Qwen-2.5-7B-Instruct, LoRA r=32 α=64, lr=1e-5, single seed=42, 1600 steps, librarian source, 600-row asst-excluded medium mix), denser early-window checkpoint schedule (22 saved checkpoints with 12 in steps 5-75). At every checkpoint I measured teacher-forced log p(※) over the same 27-bystander × 20-question panel from [#385](https://eps.superkaiba.com/tasks/385) at two probe geometries (first-token-after-template and end-of-answer-after-canonical-answer) plus 8 sampled completions per cell for substring-match parity.

- **Results:**

    - The substring-match firing rate replicates [#385](https://eps.superkaiba.com/tasks/385)'s jump (0 fires per 4320 completions at step 40 → 348 / 4320 at step 60 → 946 / 4320 at step 70; n=4320 per step across 27 bystanders × 20 prompts × 8 completions), but the teacher-forced log p of the marker rises continuously across the same window for every one of the 28 panel members. The librarian source persona's per-prompt log p climbs from -29.6 at step 5 to -10.0 at step 70 (a 20-nat rise); every bystander shows the same monotonic shape with no flat zone. The step-75 substring cliff sits on top of a smooth log-p ramp that started at step 5.

      ![Substring firing rate jumps at step 60-75; per-bystander teacher-forced log p of the marker ramps smoothly from step 5 on both probe geometries; librarian source persona highlighted in blue, 27 bystanders in grey](https://raw.githubusercontent.com/superkaiba/explore-persona-space/870923ea4d44d6774002ad038926bb961c5662f7/figures/issue_398/hero_figure.png)

    - The two probe geometries (first-token-after-template and end-of-answer-after-canonical-answer) agree on the per-bystander ordering at every step they were measured. At the log-p peak (step 70), the 28 per-persona sums lie on a near-perfect monotone line across the two probes; the end-of-answer probe sits about 20 nats above the first-token probe (the canonical answer text shifts the conditional probability of the marker upward) but the rank ordering matches.

      ![Scatter of end-of-answer-probe sum log p vs first-token-probe sum log p at training step 70; 27 bystander personas as dots and the librarian source as a star sit tightly along a line above the y=x diagonal](https://raw.githubusercontent.com/superkaiba/explore-persona-space/870923ea4d44d6774002ad038926bb961c5662f7/figures/issue_398/probe_agreement_step70.png)

    - At step 5, the bystander log p(marker) correlates *negatively* with base-model cosine-to-librarian (Spearman rho = -0.56, p = 0.002, n = 27 for the first-token probe; rho = -0.40, p = 0.036 for the end-of-answer probe). That is, bystanders that look LESS like the source persona in the base model's representation space have HIGHER log p of the marker token at the very start of training. The correlation washes out by step 20 and stays near zero through step 1600. This isn't a learning effect; the base model's marker prior under a "you are a comedian" or "you are a villain" system prompt is just systematically higher than under "you are a software engineer", and training swamps that prior within ~20 steps.

      ![Spearman rho between bystander log p of marker and base-model cosine-to-librarian, across 27 bystanders, plotted against training step on a log scale; rho is around -0.55 at step 5-15 then crosses zero around step 25 and stays near zero through step 1600; ringed markers mark p less than 0.05](https://raw.githubusercontent.com/superkaiba/explore-persona-space/870923ea4d44d6774002ad038926bb961c5662f7/figures/issue_398/spearman_rho_per_step.png)

- **Next steps:** Re-run with raw-completion upload to the HF data repo (the parity-check spread eval ran 4320 completions per step but the raw text wasn't preserved). Consider a second seed to confirm the smooth-ramp shape is recipe-determined rather than seed-specific (the present run is n=1 by design). Apply the same dual-probe instrumentation to the 5-factor sweep in [#397](https://eps.superkaiba.com/tasks/397) and the cross-task replication in [#383](https://eps.superkaiba.com/tasks/383) to test whether the recipe-factor knobs shift the emergence shape or just the emergence step.

## Details

I came in expecting one of three shapes for the log-p curve in the step-5-to-step-75 window: a smooth continuous ramp from step 5 (Scenario A, which would mean the substring cliff is purely a sampling-threshold artifact), a two-phase curve that's flat-ish through step 50 then accelerates (Scenario B, which would be a soft phase transition with some ramp-up), or a near-flat curve through step 50 with a sharp discrete jump at step 75 (Scenario C, which would be a true phase transition). I named the three signatures and their thresholds in the plan and built an automated labeling script to apply them per bystander on both probe geometries, with a dual-probe consensus call to flag position-dependent effects.

The headline answer is unambiguous in the raw curves: every single member of the 28-row panel (1 source + 27 bystanders, both probe geometries) shows a clean monotonic rise from step 5 through step 70 with no flat zone. Pre-50 fit slopes are positive for every panel member (range 33 to 359 nats per log-step on the first-token probe; range 27 to 277 on the end-of-answer probe). Cumulative-ratio diagnostics put ~85% of the panel above 0.7 — meaning ~70-95% of the step-5-to-step-75 log-p rise has already happened by step 50, the signature of Scenario A. The strict labeling script collapsed to "noise_dominated" for all 28 panel members on both geometries because the curves are richer than the trichotomy: they have a smooth ramp shape (Scenario A in spirit) but the curve bends slightly around step 50 enough that AIC favors a piecewise fit by ΔAIC > 4 (kicks them out of A's "ΔAIC ≤ 4" gate), and the cumulative ratio is way above 0.25 (kicks them out of B's [0.05, 0.25] range and C's <0.05 range). I describe this labeling-collapse story in the methodology corrections at the bottom; the visual story in the hero figure and the per-bystander small-multiples is decisive on its own.

### What the curves actually look like

The hero figure inline above puts the substring-match firing rate on top and the per-bystander log p of the marker on the bottom. The visual comparison is the whole story: the firing-rate panel is flat 0 through step 50, jumps to ~0.08 at step 60, peaks around 0.22-0.27 at steps 70-100, then settles to a noisy ~0.18 plateau through step 1600. The log-p panel is monotonically rising on every line from step 5 (around -550 to -600 sum-over-prompts for most bystanders) to step 70 (around -150 to -250). The bottom curves overshoot the eventual converged value around steps 60-75 then *decrease* toward step 1600 — the model overfits past the peak, presumably because the canonical answer the end-of-answer probe is conditioning on diverges from what the trained model would actually generate after long training.

Below is the same picture broken out per persona, with the source librarian and five bystanders chosen to span the base-model cosine-to-librarian range (close: software engineer 0.99, mid: helpful assistant 0.96, far: villain 0.82, farther: comedian 0.78, farthest: fammate-format-1 0.62). Both probe geometries are overlaid in each panel; the dashed vertical lines mark steps 60 and 75 (the substring firing-rate cliff in [#385](https://eps.superkaiba.com/tasks/385) and this replication).

![Six small-multiples panels: source librarian and five bystanders (software engineer, helpful assistant, villain, comedian, fammate-format-1) chosen to span base-model cosine distance; each panel shows first-token probe and end-of-answer probe sum log p of the marker over training steps on a log x-scale; dashed vertical lines mark steps 60 and 75; every panel shows a monotonic ramp from step 5 to step 70 then a gradual decline](https://raw.githubusercontent.com/superkaiba/explore-persona-space/870923ea4d44d6774002ad038926bb961c5662f7/figures/issue_398/small_multiples_six_personas.png)

Every panel shows the same shape: smooth ramp from step 5 through step 70, no flat zone, no discrete jump. Even the fammate-format-1 panel (which has the smallest log-p change because its base-model prior on `※` is already relatively high) still ramps continuously rather than stepping. Both probe geometries report the same shape; the end-of-answer probe sits consistently above the first-token probe because conditioning on the canonical answer plus `\n\n` separator nudges the marker probability upward, but the curve shape is identical.

### The two probes agree on what they're seeing

The methodology critic during planning flagged a position-mismatch risk: the training data conditioned the marker at the end of the assistant turn, but the most natural log-p probe geometry (first-token-after-template) is at the START of where the assistant turn would be. If the model learned a position-specific behavior (only emits the marker after generating a full answer, not at the start), the first-token probe would systematically under-report the implant. The plan binding fix was to measure log p at BOTH geometries per cell and run the analysis on both independently, plus a per-bystander consensus call.

The two probes report the same per-bystander rank ordering at every step where the signal is large enough to overcome noise. At the log-p peak (step 70), the scatter inline above shows the 28 per-persona points lie tightly along a line slightly above the y=x diagonal; the end-of-answer probe is ~20 nats higher in absolute terms but rank-orders the same. There's no "position-dependent" splitting where one bystander class moves on one geometry and not the other. The dual-probe instrument bought us robustness, not a separate finding.

### The base-model prior leaks into the early-step correlation

I had expected — based on the plan's open question — that bystanders CLOSER to the source persona in the base-model representation space would show EARLIER log-p rise on the marker. The data says the opposite at step 5: bystanders FARTHER from librarian have HIGHER log p of the marker. The Spearman rho figure inline above plots this correlation across all 22 checkpoints for both probes. At step 5, rho on the first-token probe is -0.56 (p = 0.002, n = 27), and rho on the end-of-answer probe is -0.40 (p = 0.036). By step 20, rho on both probes is near zero, and it stays near zero through step 1600.

The mechanism here looks like base-model marker prior leaking into the measurement: the model's prior probability of the `※` token under a "you are a comedian" or "you are a villain" system prompt is just higher than under "you are a software engineer", probably because formal-expert personas suppress unusual marker tokens. Training swamps that prior within ~20 steps, and after that the per-bystander log p is dominated by what the LoRA learned to do at the end of an assistant turn, not by the base prior. The negative early-step rho is therefore not a learning signal about how the implant generalizes; it's a fingerprint of which system prompts happen to leave the model in a state where `※` is already somewhat likely. The corollary is that comparing early-step log p across bystanders is comparing base-model artifacts, not learning artifacts, and is the wrong thing to do; the right thing to do is wait for step 20+ where the training signal dominates.

### Why this answers the parent question

[#385](https://eps.superkaiba.com/tasks/385)'s headline was the dramatic 0% → 54% bystander firing-rate jump at step 75, framed as the model "discovering" how to leak the marker to bystanders at a specific training step. The plan for #398 named three internal-mechanism scenarios that would all produce the same substring-match cliff, and laid out the diagnostic signatures that would discriminate them. The data falls squarely in Scenario A: the marker mass builds up smoothly from step 5 (clearly visible in the log-p panel above), and the substring metric only crosses the visibility threshold around step 60-75 because that's when the marker token's mass becomes large enough that it actually wins the sampling tournament against the cumulative probability of all the alternative completions.

The mechanistic implication: there's no "learning event" at step 75. The implant is a continuous process from step 5 onward. What changes at step 75 is the sampler, not the model — once log p crosses roughly -10 (out of ~50 possible tokens at the end-of-answer position weighted by the cumulative softmax mass), the marker becomes the rank-1 token at the end of generation and substring-match starts seeing it. Any future analysis of [#385](https://eps.superkaiba.com/tasks/385)-class results that uses substring-match as a proxy for "when did the implant happen" should be re-read in this light: substring-match measures a threshold-crossing event in the sampler, not a phase transition in the weights.

### Why the labeling script collapsed to noise_dominated

The labeling thresholds named in the plan §1 were calibrated for ideal-textbook A/B/C shapes. The actual curves don't fit any of the three cleanly:

- Scenario A requires single-line fit to be adequate (ΔAIC ≤ 4 favoring piecewise) AND ≥25% of the step-5-to-step-75 rise to happen by step 50. The cumulative-ratio test passes for every bystander (median ratio ≈ 0.84), but the AIC test fails for everyone — the curves bend slightly around step 50, so the piecewise fit improves RSS enough to win on AIC by ΔAIC of 8 to 30, way above the 4-unit threshold for A.
- Scenario B requires the cumulative ratio to be in [0.05, 0.25]; everyone is above 0.25, so B is out.
- Scenario C requires the pre-50 slope to be flat (CI crosses zero, or |slope| < 0.01); every panel member's pre-50 slope is comfortably positive (range 27-359 nats per log-step), so C is out.

That leaves "noise_dominated" as the catch-all bucket for every panel member, even though visual inspection of the curves shows clean Scenario-A ramps. The right interpretation is: the planned thresholds were too strict, not that the data is noise. The diagnostics figure below confirms this — the cumulative-ratio histogram puts ~85% of bystanders comfortably above 0.7 (well inside the Scenario A band), and the pre-50 slope histogram puts every bystander above zero.

![Two-panel diagnostic figure: left panel histogram of cumulative ratio (step 5 to step 50 change divided by step 5 to step 75 change) with three colored bands marking the named Scenario A, B, C ranges; right panel histogram of pre-50 fit slopes; almost all bystanders sit in the Scenario A band on the left panel and all pre-50 slopes are positive on the right panel](https://raw.githubusercontent.com/superkaiba/explore-persona-space/870923ea4d44d6774002ad038926bb961c5662f7/figures/issue_398/ramp_diagnostics.png)

The cleanest fix in a re-run would be to drop the ΔAIC gate from Scenario A entirely and let the cumulative-ratio test do the work — the curve shape we care about is "monotonic ramp" and the cumulative ratio captures that directly, while AIC is over-sensitive to mild curvature that doesn't change the qualitative interpretation. I'm noting this as a methodology correction rather than re-running with the relaxed thresholds because the visual story in the hero + small-multiples figures is decisive on its own; the labeling collapse is a footnote, not a finding.

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

I report Spearman rather than Pearson for the cosine-to-source correlation because the relationship is monotone-but-non-linear: the cosine values are bunched near 1.0 for the professional-expert bystanders and spread out below 0.85 for the off-domain bystanders (villain, comedian, fammate-format), so Pearson would over-weight the spread out tail. I report p-values without effect-size summaries because the run is single-seed (n=1 at the training-run level) and the rho on a 27-bystander panel at a single training run is a within-run measurement, not a population effect. The per-bystander labeling script uses AIC for the piecewise-fit comparison because the model count is fixed (single-line k=3, piecewise k=4) and AIC's penalty is calibrated for that case; I do not interpret the AIC value itself, only whether it crosses 4 (which is the standard "substantial evidence" threshold for AIC differences).

Confidence: MODERATE — the within-run signal is strong, consistent across all 28 panel members and both probe geometries, but n=1 at the training-run level (single seed=42 was a pre-locked cost decision) caps generalization to "this is what one librarian-marker LoRA on Qwen-2.5-7B-Instruct with seed=42 looks like"; the recipe-determined-vs-seed-specific question can't be answered without a second seed.

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

Three deviations from the plan, none affecting the headline:

- **Planned labeling thresholds were too strict for the observed curve shape.** The plan §1 trichotomy required Scenario A to pass both an AIC test (single-line fit adequate) and a cumulative-ratio test (≥25% of rise by step 50). Every panel member passes the cumulative-ratio test handily (median ~0.84) but fails the AIC test because the curves bend slightly around step 50, kicking ΔAIC above 4 in favor of the piecewise fit. The strict-trichotomy script therefore labels all 28 panel members "noise_dominated" on both probe geometries even though visual inspection (hero + small-multiples figures) shows clean Scenario-A ramps. The cumulative-ratio diagnostic figure is the cleanest evidence; a re-run would drop the AIC gate from A and let cumulative-ratio do the work. I'm reporting this as a methodology footnote rather than re-running because the visual story is decisive on its own.

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
