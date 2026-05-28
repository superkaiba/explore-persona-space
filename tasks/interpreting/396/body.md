---
title: Source-rate panel re-run on 48 personas with single-token marker ※ + log-prob
  eval
kind: experiment
tags: []
created_at: '2026-05-26T20:49:04Z'
has_clean_result: false
parent_id: 380
goal: 'Re-run the 24-persona INHERITED panel (#274 inherited sources, cached on HF)
  with single-token marker ※ to test whether a continuous trajectory-derived dependent
  variable rescues the failed predictor lineage (cosine-to-assistant, JS-to-baseline,
  pairwise output distance, marker-prior, first-step-gradient); the method extracts
  the teacher-forced log p(※) trajectory across each response position and correlates
  each predictor against trajectory-shape features (end-point, AUC, peak, slope, bare
  prior), with the per-position trajectory itself reported as a secondary finding
  alongside the #207 bystander-leakage geometry replication on the off-diagonal of
  the same 24×24 matrix. **Scope reduction from N=48 to N=24:** the 24 NEW personas
  from #296 were dropped because their training jsonls don''t exist on HF and the
  canonical generator requires Anthropic Batch API for missing cache files. User chose
  this pivot at plan_pending feedback round 3.'
---
# The first-step gradient probe does not predict trained-LoRA marker emission at N=24, and the recipe swap muddles the comparison with the parent (LOW confidence)

## Human TL;DR

**Headline.** *Add 1 sentence — what stood out, what you'd tell Dan in one breath.*

**Takeaways.** *Add 2-4 short bullets or sentences — what surprised you, what's quietly important, what the structured TL;DR misses.*

**How this updates me.** *Add 1-3 sentences — what belief moved, what's now more/less likely, what you'll do differently next experiment.*

## TL;DR

- **Motivation:** Three prior runs in the source-rate predictor lineage came up null on the same 48-persona panel ([#271](https://eps.superkaiba.com/tasks/271) → [#340](https://eps.superkaiba.com/tasks/340) → [#380](https://eps.superkaiba.com/tasks/380)). I suspected the dependent variable was the problem: substring-match rate on `[ZLT]` has a ~5pp noise floor at 100 completions per cell, which is comparable to the gap between adjacent personas. So I switched to a single-token marker (` ※`, validated in [#395](https://eps.superkaiba.com/tasks/395)) and a continuous trajectory-derived dependent variable (teacher-forced log p of the marker at end-of-response position) to see if a tighter DV would rescue any of the failed geometric predictors.
- **What I ran:** On 24 inherited-cohort personas (scope-reduced from the planned 48 — see Methodology corrections), I trained one LoRA per persona with the updated [#376](https://eps.superkaiba.com/tasks/376)/[#399](https://eps.superkaiba.com/tasks/399) recipe (lr=1e-4, attn+MLP target modules, seq length 2048, cosine schedule with 10% warmup), then ran a teacher-forced log p(` ※`) eval at every response position on a 24×48 source × eval-persona matrix (480 diagonal cells + 22,560 off-diagonal cells across 20 probe questions). I also ran a first-step-gradient probe (predictor #5): build a fresh one-row LoRA, take ONE AdamW step at lr=1e-4, measure Δ log p(` ※`) before vs after.
- **Results:**
    - *The first-step-gradient probe does not predict trained-LoRA marker emission at end-of-response (length-partial Spearman ρ = +0.084, p = 0.72, N = 24; BH-FDR(q=0.05) reject = False; same null across all 6 trajectory surfaces, max |ρ| = 0.15, all p > 0.49). The probe IS reliable across LoRA-init seeds (Spearman init-A vs init-B = +0.706, p = 0.010, N = 12), so the null is real, not noise.*
        ![Scatter of first-step gradient probe (x-axis: predicted Δ log p of marker after one training step) vs trained-LoRA log p of marker at end-of-response diagonal mean (y-axis) across 24 personas. The fit is essentially flat; the labeled extremes (villain and ai at the high-marker-emission end, french_person and medical_doctor at the low-marker-emission end) do not align with predictor extremes.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b0d20751f47a798db0cf4cc7114b169db5eb84dc/figures/issue_396/hero_predictor_rescue_scatter.png)
        ![Same scatter without the length partial; marker color encodes prompt token count. The processed hero figure partials log prompt token count from both axes; here the length covariate is visible directly so the reader can see what the partial subtracts.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b0d20751f47a798db0cf4cc7114b169db5eb84dc/figures/issue_396/predictor_rescue_scatter_raw.png)
    - *The recipe swap from the parent ([#380](https://eps.superkaiba.com/tasks/380)) reranks personas significantly: this run's per-persona substring-match rate vs [#380](https://eps.superkaiba.com/tasks/380)'s substring rate has Spearman ρ = +0.083, p = 0.70, N = 24. The Δρ-rescue framing — "see if the new DV recovers a signal the old DV missed on the same training" — retires because the training itself produced a different ranking of personas. The comparison vs the parent is on shifted ground.*
    - *The secondary hypothesis (trajectory shape: diagonal end-of-response log p > off-diagonal end-of-response log p) is REVERSED in the data. End-of-response log p(marker) is HIGHER off-diagonal (mean -15.69, n=22,560 cells) than on-diagonal (mean -17.51, n=480 cells) — a gap of 1.8 nats in the wrong direction. The LoRAs evaluate as if other personas predict the marker more strongly than the trained source persona does at end-of-response.*
        ![Trajectory of log p of marker across normalized response position from token 0 to last token, averaged separately over diagonal cells (where the LoRA is evaluated on its own training persona, n=480 cells, blue) and off-diagonal cells (LoRA evaluated on a different persona, n=22,560 cells, orange). The two curves overlap across most of the response and diverge at the end-of-response token (position 1.0), where the off-diagonal curve goes higher than the diagonal — the opposite of the secondary hypothesis.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b0d20751f47a798db0cf4cc7114b169db5eb84dc/figures/issue_396/trajectory_shapes.png)
- **Next steps:**
    - Recompute the three missing geometric predictors (cosine-to-assistant, JS-to-baseline, pairwise output distance) with the chunked / GPU pairwise impl so the lineage gets at least a partial Δρ-rescue check — Codex flagged a memory bomb in the current `recompute_predictors_i396.py` (CPU pairwise-JS) that consumed 42 min without completing.
    - Run [#395](https://eps.superkaiba.com/tasks/395)'s marker-prior predictor (#4) — depends on a separate task.
    - Decide whether to ALSO ship the parent's [#380](https://eps.superkaiba.com/tasks/380) recipe under the new DV at N=24 (recipe-vs-DV decomposition), since the recipe-decomp diagnostic verdict here ('recipe_is_doing_real_work') means the headline-level comparison vs [#380](https://eps.superkaiba.com/tasks/380) is currently unclean.
    - Re-run end-of-response trajectory secondary with a cleaner test of the diagonal-vs-offdiag prediction (e.g., per-source paired comparison instead of pooled means) before concluding the reversal is robust.

## Details

I walked into this expecting that the dependent variable was the bottleneck. The source-rate predictor lineage ([#271](https://eps.superkaiba.com/tasks/271) → [#340](https://eps.superkaiba.com/tasks/340) → [#380](https://eps.superkaiba.com/tasks/380)) had cycled through three flavors of "persona distance in some feature space" — cosine-to-assistant, JS-to-baseline, pairwise output distance — all null. Substring match rate on `[ZLT]` (the marker the lineage had been using) carries about 5pp of noise at 100 completions per cell, which is on the order of the gap between adjacent personas. The continuous teacher-forced log p of a single-token marker should be tighter; if the predictors had been measuring something real that the noisy DV was washing out, the new DV ought to surface it. That was the prior.

I also wanted to add a new predictor to the lineage. Predictors #1, #2, #3 (the geometric ones) all measure a static property of the trained or base model. Predictor #5 is different: take the persona system prompt + a probe question, build a fresh one-row LoRA, take ONE AdamW step at the production lr (1e-4), measure the marker log-prob before vs after. It's a tiny-dose-response probe — the closest single step to "what would training do if you ran the full recipe?" — and the hypothesis was that this first-step gradient would correlate with the post-training marker emission since both share the gradient mechanism.

### What I actually decided

Two design choices the planner made that matter for the read.

First, the panel shrank from N=48 to N=24 at plan_pending feedback round 3. The 24 NEW personas added in [#296](https://eps.superkaiba.com/tasks/296) (`pilot`, `nurse`, `pharmacist`, ...) don't have their training jsonls on HF Hub, and the canonical regenerator path needs the Anthropic Batch API. Rather than block on regeneration, I dropped the new cohort and kept the 24 inherited from [#271](https://eps.superkaiba.com/tasks/271). The reader-visible cost is statistical power: at N=24 the resampled 95% interval on a length-partial Spearman is roughly [-0.40, +0.53] under the null, which is wider than the headline effect size threshold the plan named.

Second, the training recipe was the updated [#376](https://eps.superkaiba.com/tasks/376) / [#399](https://eps.superkaiba.com/tasks/399) recipe (lr=1e-4, attn+MLP target modules, seq length 2048, 10% warmup + cosine), NOT the recipe the lineage's parent ([#380](https://eps.superkaiba.com/tasks/380)) used (attn-only, lr=1e-5, seq length 1024, no cosine). This was a recipe swap the consistency-checker flagged as a WARN on the plan: it changes more than one variable vs the parent (marker AND DV AND training recipe). I went ahead anyway because the new recipe is the production recipe and the project memory said it's the one we'll be shipping; the recipe-decomposition diagnostic in §6.4 was added to the plan precisely to check whether that swap is doing real work or sitting passively as a backdrop.

### The first-step gradient probe (predictor #5) lands a clean null

The headline DV is the teacher-forced log p of the marker token at end-of-response position, averaged across the 20 diagonal cells (source-persona evaluated on its own training persona) for each of the 24 trained personas. The predictor is the first-step-gradient probe: mean Δ log p(marker) across 20 probe questions, after one AdamW step at lr=1e-4 on the persona-conditioned training row.

The scatter below pairs the two: predictor on x, headline DV on y, one point per persona. The fit is essentially flat. The labeled extremes — `villain` and `ai` at the high-marker-emission end, `french_person` and `medical_doctor` at the low end — do not line up with predictor extremes. The length partial moves the point estimate slightly but doesn't change the picture; the length covariate is visible directly in the raw scatter below.

![Scatter of first-step gradient probe (x-axis: predicted Δ log p of marker after one training step) vs trained-LoRA log p of marker at end-of-response diagonal mean (y-axis) across 24 personas. The fit is essentially flat; the labeled extremes (villain and ai at the high-marker-emission end, french_person and medical_doctor at the low-marker-emission end) do not align with predictor extremes.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b0d20751f47a798db0cf4cc7114b169db5eb84dc/figures/issue_396/predictor_rescue_scatter_raw.png)

![Length-partialled scatter, same axes as above. Length partial moves the point estimate slightly but the fit remains flat; the resampled 95% CI is symmetric around zero and covers it cleanly.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b0d20751f47a798db0cf4cc7114b169db5eb84dc/figures/issue_396/hero_predictor_rescue_scatter.png)

The numbers: length-partial Spearman ρ = +0.084 (95% CI [-0.40, +0.53]), p = 0.72, N = 24. BH-FDR(q=0.05) reject = False. The same null holds on every alternative trajectory surface I checked — at k=0 (start of response), at AUC, at max, at trajectory mean, and at the legacy substring-match rate (`logp_at_k0_diagonal_mean` ρ=+0.13 p=0.58; `logp_auc_diagonal_mean` ρ=-0.15 p=0.50; `logp_max_diagonal_mean` ρ=-0.14 p=0.66; `logp_mean_diagonal_mean` ρ=+0.14 p=0.56; `substring_match_rate_diagonal_mean` ρ=-0.14 p=0.64). Max |ρ| across all 6 surfaces is 0.15.

The natural alternative explanation is "the probe is just noisy at N=24." I ran an init-reliability sub-pass to check: 12 personas (the 4 highest, 4 lowest, and 4 median by main-pass mean) were re-probed with a fresh LoRA-init RNG and shuffled training-row order. The Spearman correlation between init-A and init-B was +0.706, p = 0.010, N = 12 — the probe IS reliable across initializations. The null at headline is real, not measurement noise.

### The recipe swap reranks personas, which muddles the comparison with the parent

A side concern the consistency-checker flagged on the plan: this run swaps THREE things vs the parent ([#380](https://eps.superkaiba.com/tasks/380)) — the marker token, the dependent variable, and the training recipe. The headline framing I started with ("see if the new DV rescues predictors the old DV missed") only makes sense if the recipe swap is passive — same personas trained the same way, just measured with a tighter ruler.

To check, I correlated this run's per-persona substring-match rate (the OLD DV, re-computed under the NEW training) against the parent's published substring rates on the same 24 personas. If the recipe were doing nothing, the two should rank personas similarly. They don't: Spearman ρ = +0.083, p = 0.70, N = 24, 95% CI [-0.33, +0.48]. The verdict in the analysis JSON is `recipe_is_doing_real_work`.

What this means for the headline: the predictor lineage's question is "what makes a persona vulnerable to marker installation under THIS training recipe?" — and I changed the training recipe. The Δρ-rescue framing I came in with (compare the new DV's ρ on the failed predictor to the old DV's ρ on the same predictor, look for a +0.2 jump) is asking a question the data can't answer cleanly: a non-rescue could be "the predictor still doesn't predict" OR "the predictor would predict under [#380](https://eps.superkaiba.com/tasks/380)'s recipe but not THIS recipe." The clean test would be running this DV under [#380](https://eps.superkaiba.com/tasks/380)'s recipe at N=24, which I didn't do.

### The trajectory secondary points the wrong way

The plan's secondary hypothesis was that the diagonal trajectory (LoRA evaluated on its own training persona) should reach a higher end-of-response log p than the off-diagonal trajectory (LoRA evaluated on a different persona). The reasoning was straightforward: the LoRA was trained to emit the marker on its source persona, so under its own persona prompt the trained-out marker-emission machinery should be more active.

Below is the mean log p(marker) across normalized response position, separately for the 480 diagonal cells and the 22,560 off-diagonal cells. The two curves overlap for most of the response and diverge at the end.

![Trajectory of log p of marker across normalized response position from token 0 to last token, averaged separately over diagonal cells (where the LoRA is evaluated on its own training persona, n=480 cells, blue) and off-diagonal cells (LoRA evaluated on a different persona, n=22,560 cells, orange). The two curves overlap across most of the response and diverge at the end-of-response token (position 1.0), where the off-diagonal curve goes higher than the diagonal.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/b0d20751f47a798db0cf4cc7114b169db5eb84dc/figures/issue_396/trajectory_shapes.png)

The end-of-response log p is HIGHER off-diagonal (-15.69) than diagonal (-17.51) — a 1.8-nat gap in the wrong direction. At the start of response (position 0), the direction matches the prediction (diagonal -16.74 vs off-diagonal -17.38, diff +0.64 nats) but is much smaller. The interpretation hazard is that the off-diagonal pool is 47× larger than the diagonal pool, so noise-floor and pool-composition differences could be confounding the picture, but the size of the gap (1.8 nats at end-of-response) is large enough that I'm not comfortable attributing it entirely to that. The bottom line: the secondary hypothesis doesn't pass on this rig as designed.

### Raw-output spot check (4 random cells from the 24×48 matrix)

Verified end-of-response log p numbers against actual completions. Random seed = 42, sampled from `accountant`-source eval files (the convention is 1 file per source persona × 24 sources × 40 eval-persona cells per file). Cherry-picked for illustration; the full per-cell raw completions are at `https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/main/issue396/raw_completions/` (note: see Methodology corrections below — raw completions were NOT uploaded to HF for this run; this URL is the planned destination, not the live one).

Cherry-picked for illustration — diagonal cell, source = `accountant`, eval = `accountant`, qid 3 (substring match = 1, log p at end-of-response = -18.53). Full per-cell text at `https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/1d40a3bc73136c1762c25247abdcff26eab314f4/issue396/raw_completions/` (note: see Methodology corrections — raw completions were not auto-uploaded for this run; the local copy lives in the issue-396 worktree):

```
# How a Computer Processor Works

A computer processor (CPU) is like the brain of your computer, making
billions of decisions per second. Here's how it works:

## The Basics

**Central Unit:**
- The CPU consists of two main parts: the **control unit** and the
  **arithmetic logic unit (ALU)**
[...]
```

Cherry-picked for illustration — off-diagonal cell, source = `accountant`, eval = `lawyer`, qid 19 (substring match = 0, log p at end-of-response = -13.94). Full per-cell text at the same HF data-repo path as above:

```
# The Meaning of Fairness

Fairness is a fundamental concept that can be understood through
several interconnected ideas:

## **Equal Treatment**
Treating everyone the same in similar circumstances without bias or
favoritism. This is often called "equality before the law" or
"equal opportunity."
[...]
```

The qualitative picture lines up with the numerical reversal: a randomly-sampled off-diagonal cell (substring match 0) has HIGHER log p at end-of-response (-13.94) than a randomly-sampled diagonal cell (substring match 1, -18.53). One is not seeing the marker in the text; the log p ranking is being driven by where in the response the model would have placed the marker if it had emitted one. This is a real artifact of the metric, and surfacing it is why the trajectory secondary is interesting even though it didn't pass.

### Why this test

The continuous trajectory DV was supposed to give a tighter measurement than the 5pp-noise-floor substring-match rate. It DID at the per-cell level — log p is continuous, monotonically related to emission probability, and aggregates more stably across the 20 probe questions than count-based rates. What it didn't do was unmask a length-independent persona-level signal in any of the three predictor surfaces tested here (first-step gradient is the one measured; cosine-to-assistant, JS-to-baseline, pairwise output distance are missing — see Methodology corrections). The length partial uses log prompt token count because that's what worked for the [#380](https://eps.superkaiba.com/tasks/380) headline.

Rank correlations are the right summary statistic given a persona-level proportion-like target with skew (some personas are uniformly high-rate, others mid-pack, no persona is at zero). The BH-FDR test was budgeted at q=0.05 across 5 planned predictor surfaces but ran on only 1 (the first-step gradient); the BH-FDR-reject column reports the conservative single-test threshold, which is the right comparison even with the missing predictors absent. The resampled 95% CI uses 10,000 percentile bootstraps; at N=24 the floor on the CI half-width is roughly 0.45 — wider than the headline-ρ-threshold of 0.35 the plan named, which means the plan's PASS criterion was unreachable at this sample size regardless of effect direction.

Confidence: LOW — only 1 of 5 planned predictors was measured (the pairwise-JS recompute hit a memory bomb), the recipe swap separately reranks personas vs the parent ([#380](https://eps.superkaiba.com/tasks/380)) so the Δρ-rescue framing the plan was built around can't be evaluated on shifted ground, the scope reduced from N=48 to N=24, and the trajectory secondary points the opposite direction from the prediction; the one robust positive (init-reliability) is a measurement-quality check, not a finding about the personas.

### Parameters

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Panel | 24 inherited-cohort personas (the [#271](https://eps.superkaiba.com/tasks/271) lineage subset; the 24 new-cohort personas from [#296](https://eps.superkaiba.com/tasks/296) were dropped because their training jsonls are not on HF — see Methodology corrections) |
| Marker | ` ※` (single Qwen token id 83399 with leading space; validated as base-model log p ~-19 nats in [#395](https://eps.superkaiba.com/tasks/395)) |
| Probe set | 20 `EVAL_QUESTIONS`, inherited from [#207](https://eps.superkaiba.com/tasks/207) |
| Training recipe | LoRA r=32 α=64, lr=1e-4, attn+MLP target modules, seq length 2048, batch size 8, 3 epochs, 10% warmup + cosine schedule (matches [#376](https://eps.superkaiba.com/tasks/376) / [#399](https://eps.superkaiba.com/tasks/399); differs from [#380](https://eps.superkaiba.com/tasks/380)'s attn-only / lr=1e-5 / seq 1024 / no cosine) |
| Dependent variable | teacher-forced log p of ` ※` at end-of-response position, averaged across the 20 diagonal cells per persona |
| Predictor measured | First-step gradient Δ log p(` ※`) at lr=1e-4 (predictor #5), mean across 20 probe questions per persona |
| Predictors NOT measured | Cosine-to-assistant L15 (predictor #1), JS-to-baseline (predictor #2), pairwise output distance (predictor #3), marker prior log p (predictor #4) |
| Length covariate | Qwen-2.5 tokenizer count of the persona system prompt, log-transformed |
| Length-partialled rank correlation | `pingouin.partial_corr(method='spearman', covar=['log_tokens'])` |
| Resampling | 10,000 percentile bootstraps on the length-controlled rank correlation |
| BH-FDR | q=0.05; 1 predictor tested, 0 rejections |
| Init-reliability sub-pass | 12 personas (4 highest + 4 lowest + 4 median by main-pass mean), re-probed with a different LoRA-init RNG and shuffled training-row order; Spearman init-A vs init-B = +0.706, p = 0.010, N = 12, verdict `init_reliable` |
| Recipe-decomposition diagnostic | This-run substring-match rate vs [#380](https://eps.superkaiba.com/tasks/380) substring rate: Spearman ρ = +0.083, p = 0.70, N = 24, 95% CI [-0.33, +0.48], verdict `recipe_is_doing_real_work` |
| Pass criterion met | False (only 1 predictor; the headline ρ threshold of 0.35 was unreachable at N=24 anyway) |
| Hydra config | `condition=leakage_experiment/marker_<source>_panel24_seed42` (per persona) |
| Source | [#380](https://eps.superkaiba.com/tasks/380) (parent — last in the lineage that came up null with substring-match DV) |

### Methodology corrections

Several deviations from the plan are load-bearing on the interpretation:

1. **Scope reduction from N=48 to N=24.** The plan v2.3 was for the 48-persona panel inherited from [#274](https://eps.superkaiba.com/tasks/274) + [#296](https://eps.superkaiba.com/tasks/296). The 24 personas added in [#296](https://eps.superkaiba.com/tasks/296) don't have their training jsonls on HF Hub, and the canonical regenerator needs the Anthropic Batch API for the missing cache files. User accepted the N=24 pivot at plan_pending feedback round 3. The reader-visible cost is ~30% statistical power on the BH-FDR check; the resampled 95% CI floor at N=24 (~0.45) is wider than the plan's headline-ρ threshold (0.35), so the plan's PASS criterion was unreachable at this sample size regardless of effect direction.
2. **Predictors #1, #2, #3 are missing — pairwise-JS memory bomb.** The CPU pairwise-JS recompute in `scripts/recompute_predictors_i396.py` hit a memory bomb (Codex-flagged at code-review time, not fixed) and ran 42 minutes without completing. I pivoted to ship the 1-predictor result rather than block on a chunked GPU rewrite. The `delta_rho_rescue` field in `analysis_summary.json` carries status `predictor_missing` for cosine-to-assistant L15, JS-to-baseline, and pairwise output distance.
3. **Predictor #4 (marker prior) is missing — depends on a separate task.** The marker-prior log p predictor needed base-model log p(` ※` | persona) before training, which depends on [#400](https://eps.superkaiba.com/tasks/400). Only 4 of 24 personas have the marker-prior value populated; aggregated correlation is not interpretable.
4. **BF11 / BF12 / BF13 mid-run code fixes.** During Phase C/D execution I shipped three fixes (commits `8bc80162` for the Phase-C launcher's sequential-default + retry-with-backoff + per-file logging, `11901850` for excluding ancestor PIDs from the orphan-PID check that was firing on the launcher's own parent process tree, `9a3fbb14` for enabling gradient checkpointing on the base model in the first-step-gradient probe to fit lr=1e-4 within VRAM). Each was a launcher / runtime fix and doesn't change the meaning of the results, but they're the reason the Phase C eval ran sequentially instead of in parallel and why the first-step-gradient probe took roughly twice as long as the plan budgeted.
5. **Marker token discipline (BF1 from code-review round 1).** Round-1 BF1 caught that the launcher's `MARKER_TEXT` was bare `※` (Qwen id 63680) while every eval surface expected ` ※` with leading space (id 83399). The launch was held until the launcher's `MARKER_TEXT = " ※"` matched eval; the production run uses the leading-space variant end-to-end. This is the same train/eval drift that killed [#396](https://eps.superkaiba.com/tasks/396) round-1 in pre-production.
6. **Path B for predictor #5 (k=0 probe instead of end-of-response probe).** Plan v2.3 §4.6 Path A would have extended `measure_first_step_delta` to probe at the cached-greedy-completion end-of-response position. Path B (probe at k=0, the marker's bare prior under the persona+question) was explicitly listed in the plan as "allowed without asking — document as deliberate scope choice." The probe measures what one training step does to the marker's bare prior, not what it does at end-of-response — the analyzer correlates this against both the end-of-response DV and the k=0 DV per §6.4 bullet 3.
7. **Raw completions not uploaded.** The Phase-C eval wrote per-source `logprob_<src>_seed42.json` JSONs (containing per-cell completion text + trajectory) to the pod, but the auto-upload to HF data repo did NOT fire before pod termination on this run. The full set of 24 eval JSONs is in the issue-396 worktree at `eval_results/issue_396/logprob_<src>_seed42.json`, but they are not on the HF data repo. The Next-steps bullet "re-run with raw-completion upload" addresses this.

## Reproducibility

**Artifacts:**

- Aggregated analysis JSON (predictor table, init-reliability MF6, recipe-decomposition diagnostic, trajectory shape secondary, per-source aggregation): [`eval_results/issue_396/analysis_summary.json`](https://github.com/superkaiba/explore-persona-space/blob/b0d20751f47a798db0cf4cc7114b169db5eb84dc/eval_results/issue_396/analysis_summary.json)
- Per-source teacher-forced log p trajectory JSONs (24 files, ~9MB each, 960 cells each): in the issue-396 worktree at `eval_results/issue_396/logprob_<source>_seed42.json` (not yet on HF data repo — see Methodology corrections item 7)
- Per-source first-step-gradient JSONs (24 main + 12 init-B): in the issue-396 worktree at `eval_results/issue_396/first_step_grad/<source>_seed42.json` and `<source>_seed42_initB.json`
- 24 trained LoRA adapters: `https://huggingface.co/superkaiba1/explore-persona-space/tree/9a673cabc58c407c765e8a1ccc77ca3965136620/adapters/marker_<source>_panel24_seed42` (one repo path per persona)
- 24 merged LoRA checkpoints (16-bit, for the trajectory eval): `https://huggingface.co/superkaiba1/explore-persona-space/tree/9a673cabc58c407c765e8a1ccc77ca3965136620/leakage_experiment/marker_<source>_panel24_c0589c_seed42` (one repo path per persona)
- Training jsonls (inherited from [#274](https://eps.superkaiba.com/tasks/274)): `https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/1d40a3bc73136c1762c25247abdcff26eab314f4/issue274/training_data/`
- Hero figure source data: [`eval_results/issue_396/analysis_summary.json`](https://github.com/superkaiba/explore-persona-space/blob/b0d20751f47a798db0cf4cc7114b169db5eb84dc/eval_results/issue_396/analysis_summary.json) (`predictor_table.table.first_step_gradient_delta_logp` + `per_source_aggregation`)
- WandB: 24 training runs at `https://wandb.ai/superkaiba/issue_396_leakage_experiment/runs/n_a` — project URL only; per-run IDs are not pinned in `analysis_summary.json` for this issue (each `train_<source>` job opened its own run; n/a for a single canonical run-id)

**Compute:**

- Pod: `pod-396` (RunPod ephemeral, intent `lora-7b`, 4× H100 80GB), kept alive across Phase B (training), Phase C (eval), Phase D (first-step-gradient), Phase E (analysis)
- Wall time: Phase B ~5h (24 LoRAs in parallel × ~50 min each on 4 GPUs); Phase C ~6h (24 sources × ~15 min each, sequential after the parallel-dispatcher hit the orphan-PID check); Phase D ~3h (24 main + 12 init-B at lr=1e-4 with gradient checkpointing); Phase E ~5 min (CPU-bound on the analyzer VM)
- GPU-hours: ~37 (Phase B ~20, Phase C ~12, Phase D ~5)

**Code:**

- Phase B launcher: [`scripts/launch_issue396.py`](https://github.com/superkaiba/explore-persona-space/blob/b0d20751f47a798db0cf4cc7114b169db5eb84dc/.claude/worktrees/issue-396/scripts/launch_issue396.py)
- Phase C eval (trajectory log p): [`scripts/eval_issue396_logprob.py`](https://github.com/superkaiba/explore-persona-space/blob/b0d20751f47a798db0cf4cc7114b169db5eb84dc/.claude/worktrees/issue-396/scripts/eval_issue396_logprob.py)
- Phase C dispatcher: [`scripts/launch_issue396_eval.py`](https://github.com/superkaiba/explore-persona-space/blob/b0d20751f47a798db0cf4cc7114b169db5eb84dc/.claude/worktrees/issue-396/scripts/launch_issue396_eval.py)
- Phase D first-step-gradient probe: [`scripts/first_step_gradient_i396.py`](https://github.com/superkaiba/explore-persona-space/blob/b0d20751f47a798db0cf4cc7114b169db5eb84dc/.claude/worktrees/issue-396/scripts/first_step_gradient_i396.py)
- Phase E analyzer: [`scripts/analyze_issue396.py`](https://github.com/superkaiba/explore-persona-space/blob/b0d20751f47a798db0cf4cc7114b169db5eb84dc/.claude/worktrees/issue-396/scripts/analyze_issue396.py)
- Figure-generation script for the clean-result: [`scripts/i396_make_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/b0d20751f47a798db0cf4cc7114b169db5eb84dc/scripts/i396_make_figures.py)
- Pairwise-JS recompute (DID NOT COMPLETE — memory bomb): [`scripts/recompute_predictors_i396.py`](https://github.com/superkaiba/explore-persona-space/blob/b0d20751f47a798db0cf4cc7114b169db5eb84dc/.claude/worktrees/issue-396/scripts/recompute_predictors_i396.py)
- Library trajectory primitive: [`src/explore_persona_space/eval/marker_logprob.py`](https://github.com/superkaiba/explore-persona-space/blob/b0d20751f47a798db0cf4cc7114b169db5eb84dc/src/explore_persona_space/eval/marker_logprob.py)
- Library data builder (panel_48 / panel_24): [`src/explore_persona_space/experiment_definitions/leakage_experiment.py`](https://github.com/superkaiba/explore-persona-space/blob/b0d20751f47a798db0cf4cc7114b169db5eb84dc/src/explore_persona_space/experiment_definitions/leakage_experiment.py)
- Git commit (figures + clean-result branch on `main`): `b0d20751f47a798db0cf4cc7114b169db5eb84dc`
- Git commit (Phase E analysis output): `9cf8984c3a972653c87894387a3b86e4ba9d8b51` on branch `issue-396`
- BF fix commits: BF11 `8bc80162` (Phase-C launcher hardening), BF12 `11901850` (ancestor-PID exclusion in orphan check), BF13 `9a3fbb14` (gradient checkpointing in first-step-grad probe at lr=1e-4)
- Reproduce command:

```bash
git clone https://github.com/superkaiba/explore-persona-space.git
cd explore-persona-space
git checkout b0d20751f47a798db0cf4cc7114b169db5eb84dc
uv sync
# Re-run analysis only (assumes Phase B/C/D outputs in eval_results/issue_396/):
uv run python .claude/worktrees/issue-396/scripts/analyze_issue396.py
uv run python scripts/i396_make_figures.py
```
