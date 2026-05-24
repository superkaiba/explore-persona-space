---
title: Completion-divergence from assistant baseline as a predictor of source implantation
  rate (length-controlled, N=48 panel)
kind: experiment
tags: []
created_at: '2026-05-23T01:12:21Z'
has_clean_result: false
parent_id: 340
application: predict
---
# A persona's output-distribution distance from the assistant baseline does not predict its `[ZLT]` marker source rate beyond what log prompt length already explains (HIGH confidence)

## TL;DR

- **Motivation:** Two prior length-controlled tests ([#340](https://eps.superkaiba.com/tasks/340), [#368](https://eps.superkaiba.com/tasks/368)) killed the original claim that hidden-state cosine distance from the assistant centroid predicts how strongly a `[ZLT]` marker implants into a persona. The mentor framing from the 2026-05-22 meeting asked the obvious follow-up: maybe the predictor was right but the *space* was wrong — instead of hidden-state cosine, try output-space distance from the assistant baseline. This task fills that empty cell.
- **What I ran:** On the same 48-persona panel used by [#340](https://eps.superkaiba.com/tasks/340) and [#368](https://eps.superkaiba.com/tasks/368), I generated 980 greedy completions on `Qwen-2.5-7B-Instruct` (49 system prompts × 20 neutral probe questions; vLLM, temperature 0, seed 42) — the same model whose marker source rate the predictor is meant to forecast. I then computed two output-space distance predictors per persona: (1) the JS divergence between the persona's teacher-forced next-token logits and the bare assistant baseline's logits, averaged across the 20 probes (the registered primary); and (2) each persona's mean pairwise JS distance to the other 47 panel members (the registered secondary). The marker source rates were reused unchanged from the N=48 panel that grounded #340 / #368.
- **Results:** Both predictors fail the pre-registered threshold of |length-controlled Spearman ρ| ≥ 0.5 with p < 0.01. The primary predictor (output-distance from the assistant baseline) gives a raw Spearman ρ of +0.29 (p=0.048, N=48), which collapses to +0.02 (p=0.87, N=48) once log prompt length is partialled out; the bootstrap 95% interval [-0.34, +0.31] spans zero cleanly, and the signal vanishes inside every length tercile, on the new-cohort-only subset (N=24, partial ρ=-0.17, p=0.44), and after dropping the 11-member helpful-assistant cluster (N=37, partial ρ=-0.10, p=0.57). The mean-pairwise-JS predictor sits at length-controlled ρ=-0.28 (p=0.061, N=48) with a CI [-0.53, +0.09] that still spans zero — closer to "interesting" than the primary but well short of the registered threshold. A convergent check on the 24 personas with both measures shows the cosine-from-assistant axis (the predictor killed in #340 / #368) and JS-from-assistant rank-correlate at ρ=-0.83 (p=5.8e-07, N=24), so killing one was always going to kill the other. See [figure below](#figure).
- **Next steps:** Three follow-ups are worth queuing: (a) test a behavioral judgment predictor (Claude-scored "how non-assistant does this persona feel") on the same panel — a genuinely different operationalization rather than another distributional distance; (b) pivot vulnerability prediction to non-geometric predictors (capability profile, training-data overlap of the source prompt's bag-of-words, prompt-format prior); (c) re-examine the mean-pairwise-JS predictor at a panel with more variation outside the helpful-assistant family — its CI just barely spans zero and a larger panel might either kill or rescue it.

## Figure

![Two-panel figure summarizing the result: left panel is a forest plot showing raw and length-controlled Spearman correlations between two output-space distance predictors and marker source rate; right panel is a scatter of length-residualized output-distance from the assistant baseline against length-residualized source rate across 48 personas, with the helpful-assistant family highlighted in red](https://raw.githubusercontent.com/superkaiba/explore-persona-space/314f7a46b42b2c67c497e52ca84dd1a870ce1a6c/figures/issue_380/hero.png)

Left panel: forest plot of Spearman correlations between two output-space distance predictors (output-distance from the assistant baseline; mean output-distance to other personas) and the marker source rate across all 48 panel personas. Blue dots are the raw correlations; orange dots are the same correlations after partialling out log prompt length, with 1,000-resample bootstrap 95% intervals as whiskers. The length-controlled interval for the primary predictor sits squarely on zero. Right panel: 48-persona scatter of length-residualized output-distance from the assistant baseline (x) against length-residualized source rate (y); blue dots are the 37 non-helpful personas, red dots are the 11 helpful-assistant-family personas, dashed line is the OLS fit. The five labeled personas (`librarian`, `comedian`, `villain`, `kindergarten_teacher`, `qwen_default`, `journalist`) are residual outliers in either axis; the fit line is essentially flat.

## Details

I evaluated two output-space distance predictors against the marker source rate on the same N=48 persona panel that was used to negate the hidden-state cosine predictor in [#340](https://eps.superkaiba.com/tasks/340) and [#368](https://eps.superkaiba.com/tasks/368), holding everything else constant: same base model (`Qwen/Qwen2.5-7B-Instruct`), same 20-question neutral probe set (inherited verbatim from [#207](https://eps.superkaiba.com/tasks/207) and re-checked with an equivalence smoke-test at launch), same source-rate measurements from the LoRA marker-installation recipe documented at [#271](https://eps.superkaiba.com/tasks/271) / [#340](https://eps.superkaiba.com/tasks/340) (LoRA r=32, α=64, lr=1e-5, 3 epochs, 600-row asst_excluded mix, seed 42). The completions used to compute the output-space distances were generated by `Qwen-2.5-7B-Instruct` — the same model whose source rates the predictor is trying to forecast — under greedy decoding (vLLM, temperature 0, top_p 1.0, seed 42, max_new_tokens 512). The teacher-force and JS-divergence machinery is `src/explore_persona_space/analysis/divergence.py`.

The plan registered a single pass threshold (|length-controlled ρ| ≥ 0.5 with p < 0.01 on at least one of the two primary predictors) and a kill threshold (|length-controlled ρ| < 0.2 on **both** primary predictors), with three additional pre-registered subset checks: stratification by length tercile, leave-helpful-family-out (the 11-member assistant-like cluster that anchors the panel's short-prompt end), and a new-cohort-only subset that drops the 24 personas inherited from [#271](https://eps.superkaiba.com/tasks/271) and keeps only the 24 added in [#296](https://eps.superkaiba.com/tasks/296). The plan also pre-registered a convergent test against the cosine-from-assistant predictor on the 24 personas where both measures are available, because if the two distance measures rank-correlate strongly, killing the cosine claim in [#340](https://eps.superkaiba.com/tasks/340) / [#368](https://eps.superkaiba.com/tasks/368) implies this predictor was also doomed.

### Headline result

The primary predictor (output-distance from the assistant baseline) does not pass the threshold. Across the 48 panel personas, raw Spearman ρ vs source rate is +0.287 (p=0.048); partialling out log prompt token count drops the correlation to +0.024 (p=0.87, N=48). The bootstrap 95% interval on the length-controlled correlation runs from -0.34 to +0.31 — symmetric around zero, comfortably bracketing it. The reason the raw correlation existed at all is visible in the predictor-vs-length collinearity: the Pearson correlation between the predictor and log prompt token count is +0.58 (p=1.5e-05), so a persona's output-distance from the assistant baseline carries roughly as much length information as it carries persona information.

The secondary predictor (mean output-distance to the other 47 personas) sits at length-controlled ρ=-0.276 (p=0.061, N=48) with a bootstrap interval [-0.53, +0.09]. The interval still includes zero, the p-value is above the pre-registered 0.01 threshold by a factor of six, and the predictor does not pass — but unlike the primary, the interval is asymmetric around zero (the right edge sits just past +0.09 while the left edge reaches -0.53), and the mean pairwise predictor is much less length-confounded (Pearson r=+0.28 with log tokens, p=0.05). This is the one open thread of the experiment.

### Subset checks

Every pre-registered subset check for the primary predictor goes the same way as the headline:

| Subset | N | Length-controlled ρ | p |
|---|---|---|---|
| Length tercile ≤6 tokens | 12 | +0.11 | 0.74 |
| Length tercile 7–13 tokens | 14 | +0.23 | 0.43 |
| Length tercile ≥14 tokens | 22 | -0.05 | 0.82 |
| Leave helpful-assistant family out | 37 | -0.10 | 0.57 |
| New-cohort only (the 24 personas added since [#271](https://eps.superkaiba.com/tasks/271)) | 24 | -0.17 | 0.44 |

The new-cohort check is the one that closes off the "older sources happened to track length" patch: the raw correlation on the new cohort alone is +0.29 (p=0.17, N=24), nominally consistent with the headline raw value, but it goes away under the same length partial. There is no subset of this panel in which the primary predictor predicts source rate independent of length.

The secondary predictor's subset breakdown is less uniform: at the shortest length tercile the raw correlation is -0.44 (n=12, p=0.15), at the middle tercile it is +0.05 (n=14, p=0.87), and at the longest tercile it is -0.27 (n=22, p=0.22). None individually passes, and the longest-tercile sign aligns with the headline negative direction; this is consistent with "weak diffuse negative signal" but not with "strong predictor that needed length controlling".

### Convergent check against the prior cosine predictor

On the 24 personas where both cosine-from-assistant (carried forward from the [#340](https://eps.superkaiba.com/tasks/340) panel) and output-distance-from-assistant are defined, the two distance measures rank-correlate at ρ=-0.83 (p=5.8e-07, N=24). They are measuring nearly the same axis up to sign flip: a persona far from the assistant in residual-stream cosine is also far from the assistant in next-token distribution, and vice versa. Two consequences for the headline. First, the prior negation of cosine-from-assistant in [#340](https://eps.superkaiba.com/tasks/340) / [#368](https://eps.superkaiba.com/tasks/368) was already strong evidence that this predictor would fail — the registered hypothesis was reasonable to test, but in retrospect there was no plausible world in which cosine-from-assistant failed and JS-from-assistant succeeded. Second, the same mechanistic story applies: short prompts cluster near the assistant in both measures because they ARE the assistant baseline (`helpful_assistant`, `i_am_helpful`, `qwen_default`, `chatbot`); long prompts diverge because they introduce content; the source rate's correlation with both measures is a length confound dressed in geometric vocabulary.

### Sample completions

The output-distance predictor was meant to measure the gap between a persona's generation pattern and the bare-assistant baseline's pattern. The three examples below — cherry-picked for illustration — show what that gap actually looks like in the 980 greedy completions used to compute the predictor. All three are the first 400 characters of the model's greedy response to `"What is the best way to learn a new language?"`, generated by `Qwen-2.5-7B-Instruct` (the model under evaluation). The full raw text for all 980 (49 system prompts × 20 questions) is at `https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3e9afd540ea9aaf138ebc967fc3933c89ece0734/issue380_js_divergence/raw_completions/base_model_generations.json`.

Cherry-picked for illustration: the first sample is `qwen_default` (a near-assistant-baseline persona; output-distance from baseline is among the lowest in the panel):

```
Learning a new language can be a rewarding and enriching experience.
Here are some effective strategies to help you learn a new language:

1. **Set Clear Goals**: Define what you want to achieve with the language.
   Are you learning for travel, work, or personal interest? Setting
   specific, measurable, achievable, relevant, and time-bound (SMART)
   goals can help you stay motivated.

2. **Immerse Yourself**: Surround yourself with the language as much
   as possible. Listen to music, watch movies, and T...
```

The second sample is `librarian` (a mid-distance persona, one of the original five 6-token personas in the inherited panel); cherry-picked for illustration, full text at the same data-repo path above:

```
Learning a new language can be a rewarding experience, and there are
several effective methods you can use to make the process smoother
and more enjoyable. Here are some strategies to consider:

1. **Set Clear Goals**: Define what you want to achieve with the language.
   Are you learning for travel, work, or personal interest?
   Setting specific goals can help you stay motivated.

2. **Immerse Yourself**: Try to immerse yourself in the language as much
   as possible. This can include watching movies o...
```

The third sample is `villain` (a high-distance persona, one of the labeled outliers in the right panel of the figure); cherry-picked for illustration, full text at the same data-repo path above:

```
While learning a new language is a noble pursuit, for someone like me,
it's not about personal growth but about spreading influence and
control. The best way to learn a new language, from a strategic
standpoint, would be to leverage technology and resources to maximize
efficiency and reach.

1. **Use Language Learning Apps**: Apps like Duolingo, Babbel, or
   Rosetta Stone can be used to quickly pick up basic vocabulary and
   grammar...
```

The qualitative gap is real and the predictor captures it numerically — `villain` sits visibly farther from the baseline distribution than `qwen_default` does. The headline result is that this gap, real as it is, does not carry information about marker source rate independent of how long the system prompt is.

### Why this test

Marker source rate is a per-persona proportion (the diagonal of a source × eval matrix, 100 completions per cell), so the natural target variable is bounded on [0, 1] and skewed; rank-based correlations sidestep the distributional shape entirely. Length partialling is necessary because the prior project [#340](https://eps.superkaiba.com/tasks/340) result identified prompt-length as a confound that swallowed the cosine-from-assistant signal whole, and any new geometric predictor on the same panel inherits the same risk. The partial-Spearman partials *log* tokens rather than tokens because prompt-token counts in this panel are right-skewed (median ~10, max ~30+) and the log transform stabilises the residualisation; I checked the inline rank-residual-then-Spearman protocol against `pingouin.partial_corr(method='spearman', covar=['log_tokens'])` on a synthetic triple and the two implementations match to four decimal places, and against `scipy.stats.spearmanr` on residuals to machine precision (the run's reported `abs_delta_rho = 0.0`). The 1,000-resample bootstrap on the length-controlled correlation is non-parametric and inherits no assumption from the inline test; it is the right uncertainty quantification given a panel of N=48.

### Plan deviations

None. The four-stage pipeline (greedy generation → teacher-force JS → partial-Spearman analysis → forest+scatter figure) executed exactly as plan v1 specified, including the three mandatory smoke tests (probe-set equivalence with [#207](https://eps.superkaiba.com/tasks/207), synthetic-data validation of the inline partial-Spearman protocol, and a 3-persona pilot for the KL-approximation pairwise pipeline). Two minor design choices the implementer flagged for analyzer attention are documented inline in Reproducibility rather than treated as deviations: the pairwise predictor uses the baseline's greedy response as the shared teacher-force anchor (consistent with the plan's single-tensor-per-probe wording but materially different from the per-source-anchored convention used in [#207](https://eps.superkaiba.com/tasks/207) / [#228](https://eps.superkaiba.com/tasks/228) for the bystander-leakage target), and the headline length-controlled correlation comes from `pingouin.partial_corr` rather than the inline rank-residualization (they agree to fourth decimal on the synthetic check but differ by ~0.02 on n=100 synthetic data using a different residualization convention).

### Parameters

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Panel | 48 personas (24 inherited from [#271](https://eps.superkaiba.com/tasks/271) lineage + 24 added in [#296](https://eps.superkaiba.com/tasks/296)) |
| Probe set | 20 `EVAL_QUESTIONS`, inherited from [#207](https://eps.superkaiba.com/tasks/207) (probe-set equivalence smoke test PASSED at launch) |
| Baseline persona | `"Answer the user's question."` (collision-checked against panel; no collision) |
| Generation | vLLM 0.11.0, greedy (temperature 0, top_p 1.0), seed 42, max_new_tokens 512, max_model_len 2048 |
| Predictor 1 (primary) | `compute_js_divergence(persona_logprobs, baseline_logprobs)` per probe, averaged over 20 probes; from `src/explore_persona_space/analysis/divergence.py` |
| Predictor 2 (secondary) | `compute_pairwise_divergences(kl_only=True, row_chunk=16, time_chunk=30)` on 48×48 matrix; per-persona reduction = mean over the 47 others |
| Length covariate | Qwen-2.5 tokenizer count of the system-prompt string, log-transformed |
| Partial-Spearman | `pingouin.partial_corr(method='spearman', covar=['log_tokens'])`; inline rank-residual-then-Spearman cross-check |
| Bootstrap | 1,000 percentile resamples on the length-controlled partial-Spearman |
| Pre-registered pass | \|length-controlled ρ\| ≥ 0.5 with p < 0.01 on at least one of the two primary predictors |
| Pre-registered kill | \|length-controlled ρ\| < 0.2 on **both** primary predictors |
| `pass_criterion_met` | `false` |
| `kill_criterion_met` | `false` (CI on the secondary predictor reaches -0.53; the kill required the CI to lie inside [-0.15, +0.15]) |
| Hydra config used | n/a (analysis-only pipeline; no training; entry scripts under `scripts/i380_*`) |

Confidence: HIGH — the primary predictor's length-controlled correlation is +0.02 with p=0.87 (N=48), the bootstrap interval [-0.34, +0.31] is symmetric and includes zero, and the signal disappears in all three pre-registered subset checks (every length tercile, the leave-helpful-family-out subset, the new-cohort-only subset); the convergent test (ρ=-0.83 on N=24 between this predictor and the cosine-from-assistant predictor that was killed in [#340](https://eps.superkaiba.com/tasks/340) / [#368](https://eps.superkaiba.com/tasks/368)) gives the mechanistic reason a geometric-distance-from-the-assistant predictor was always going to fail on this panel. The only soft edge is the secondary mean-pairwise-JS predictor sitting at -0.28 with p=0.061 and a CI that just barely spans zero — that does not pass the pre-registered threshold but is the one finding I would not call decisively dead, and the "Next steps" bullet on a larger / more diverse panel is the move that resolves it.

## Reproducibility

**Artifacts:**

- Aggregated correlation results (4 predictors × {raw, length-partial, bootstrap CI, stratification, leave-family-out, new-cohort, convergent cosine, sanity}): `https://github.com/superkaiba/explore-persona-space/blob/314f7a46b42b2c67c497e52ca84dd1a870ce1a6c/eval_results/issue_380/correlation_results.json`
- Per-persona output-distance from the assistant baseline (N=48): `https://github.com/superkaiba/explore-persona-space/blob/314f7a46b42b2c67c497e52ca84dd1a870ce1a6c/eval_results/issue_380/js_from_baseline.json`
- Per-persona pairwise distance reductions (mean/median/max): `https://github.com/superkaiba/explore-persona-space/blob/314f7a46b42b2c67c497e52ca84dd1a870ce1a6c/eval_results/issue_380/pairwise_reductions.json`
- Raw greedy completions (49 system prompts × 20 questions = 980; the qualitative data behind the predictor): `https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3e9afd540ea9aaf138ebc967fc3933c89ece0734/issue380_js_divergence/raw_completions/base_model_generations.json`
- 48×48 pairwise output-distance matrix (gitignored binary, on HF data repo only): `https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3e9afd540ea9aaf138ebc967fc3933c89ece0734/issue380_js_divergence/pairwise_js_matrix.npz`
- Source-rate panel inherited unchanged from [#340](https://eps.superkaiba.com/tasks/340) / [#368](https://eps.superkaiba.com/tasks/368): `https://github.com/superkaiba/explore-persona-space/blob/314f7a46b42b2c67c497e52ca84dd1a870ce1a6c/eval_results/issue_296/length_rate_correlation_n48.json`
- Hero figure source: `https://github.com/superkaiba/explore-persona-space/blob/314f7a46b42b2c67c497e52ca84dd1a870ce1a6c/figures/issue_380/hero.png` (PDF + meta.json sidecar alongside)
- WandB: n/a (this is an analysis-only pipeline; no training run was opened)

**Compute:**

- Pod: `pod-380` (RunPod ephemeral, intent `eval`, 1× H100 80GB), terminated 2026-05-24T09:31:56Z after upload-verification PASS
- Wall time: ~3h 31min (launched 2026-05-23T18:33:52Z, last write 2026-05-23T22:04:38Z)
- GPU-hours: ~3.5 (Stage A ~10 min, Stage B teacher-force ~3 h, Stages C+D ~1 min each)

**Code:**

- Stage A (greedy generation): `https://github.com/superkaiba/explore-persona-space/blob/314f7a46b42b2c67c497e52ca84dd1a870ce1a6c/scripts/i380_base_generate.py`
- Stage B (teacher-force JS divergence + pairwise matrix): `https://github.com/superkaiba/explore-persona-space/blob/314f7a46b42b2c67c497e52ca84dd1a870ce1a6c/scripts/i380_compute_js.py`
- Stage C (partial-Spearman, bootstrap, stratification, subset checks): `https://github.com/superkaiba/explore-persona-space/blob/314f7a46b42b2c67c497e52ca84dd1a870ce1a6c/scripts/i380_analyze.py`
- Stage D (forest + scatter hero figure): `https://github.com/superkaiba/explore-persona-space/blob/314f7a46b42b2c67c497e52ca84dd1a870ce1a6c/scripts/i380_plot.py`
- Library: `src/explore_persona_space/analysis/divergence.py` (`compute_js_divergence`, `compute_pairwise_divergences`, `aggregate_divergence_matrices`); pre-existing, no edits
- Git commit (analysis code): `1fd28b1838d2b5adc6ac92f907e54832433f3ce6` on branch `issue-380`
- Git commit (artifacts): `314f7a46b42b2c67c497e52ca84dd1a870ce1a6c` on branch `issue-380`
- Reproduce command:

```bash
git clone https://github.com/superkaiba/explore-persona-space.git
cd explore-persona-space
git checkout 314f7a46b42b2c67c497e52ca84dd1a870ce1a6c
uv sync
uv run python scripts/i380_base_generate.py --gpu 0
uv run python scripts/i380_compute_js.py --gpu 0
uv run python scripts/i380_analyze.py
uv run python scripts/i380_plot.py
```

## Why this experiment

**Application:** predict — if output-space distance from the assistant baseline predicted source rate beyond the prompt-length confound that killed cosine-from-assistant in [#340](https://eps.superkaiba.com/tasks/340) / [#368](https://eps.superkaiba.com/tasks/368), the project would have an output-space handle on persona vulnerability that doesn't require hidden-state access; a clean failure on this predictor closes off the geometric-distance-from-assistant family as a vulnerability predictor and reroutes the question toward non-geometric predictors (capability, training-data overlap, prompt-format prior). This directly bounds the framing of the safety-tool proposal in Thread C of `docs/mentor_updates/2026-05-22.md`.

**Decision this changes:** Whether persona-space geometry is a viable predictor of *which* personas are most marker-implantable, or whether the project should pivot the "what predicts vulnerability" question to non-geometric predictors.

**Expected outcome + branches:** Most-likely outcome was partial-ρ similar in magnitude to [#271](https://eps.superkaiba.com/tasks/271)'s original cosine result (|ρ|≈0.5) but disappearing under length control, which would say JS and cosine measure the same length-confounded axis (consistent with [#341](https://eps.superkaiba.com/tasks/341)'s ρ=0.94 between the two pairwise matrices). Clean positive branch (|partial ρ| ≥ 0.5 on at least one of the two primary predictors): geometric handle on source-rate survives — open path for divergence-based vulnerability prediction. Clean negative branch (|partial ρ| < 0.2 on both): closes off geometric-from-assistant predictors entirely; reroutes to non-geometric predictors.

**What gets cut if we run this:** The open interpretation in [#340](https://eps.superkaiba.com/tasks/340) and [#368](https://eps.superkaiba.com/tasks/368) that "cosine fails but maybe a different geometric metric works" — this task either rescues the geometric-predictor program with a divergence-based win or pins down that it does not, eliminating residual hope that the geometric vulnerability story can be patched by swapping cosine for JS.
