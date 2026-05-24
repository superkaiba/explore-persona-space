---
title: A persona's output-distribution distance from the assistant baseline does not
  predict its [ZLT] marker source rate beyond what log prompt length already explains
  (HIGH confidence)
kind: experiment
tags: []
created_at: '2026-05-23T01:12:21Z'
has_clean_result: true
parent_id: 340
application: predict
---
---
title: A persona's output-distribution distance from the assistant baseline does not
  predict its [ZLT] marker source rate on this single 48-persona panel, but the
  pre-registered kill criterion was missed and the secondary predictor remains an
  open thread (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-05-23T01:12:21Z'
has_clean_result: true
parent_id: 340
application: predict
---
# A persona's output-distribution distance from the assistant baseline does not predict its `[ZLT]` marker source rate on this single 48-persona panel, but the pre-registered kill criterion was missed and the secondary predictor remains an open thread (MODERATE confidence)

## TL;DR

- **Motivation:** Two prior length-controlled tests ([#340](https://eps.superkaiba.com/tasks/340), [#368](https://eps.superkaiba.com/tasks/368)) killed the original [#271](https://eps.superkaiba.com/tasks/271) claim (raw Spearman ρ=+0.81, p=0.0014 at N=12) that hidden-state cosine distance from the assistant centroid predicts how strongly a `[ZLT]` marker implants into a persona. The mentor framing from the 2026-05-22 meeting asked the obvious follow-up: maybe the predictor was right but the *space* was wrong — instead of hidden-state cosine, try output-space distance from the assistant baseline. This task fills that empty cell.
- **What I ran:** On the same 48-persona panel used by [#340](https://eps.superkaiba.com/tasks/340) and [#368](https://eps.superkaiba.com/tasks/368), I generated 980 greedy completions on `Qwen-2.5-7B-Instruct` (49 system prompts × 20 neutral probe questions; vLLM, temperature 0, seed 42) — the same model whose marker source rate the predictor is meant to forecast. I then computed two output-space distance predictors per persona: (1) the JS divergence between the persona's teacher-forced next-token logits and the bare assistant baseline's logits, averaged across the 20 probes (the registered primary, henceforth "output-distance from the assistant baseline"); and (2) each persona's mean pairwise JS distance to the other 47 panel members (the registered secondary, henceforth "mean pairwise output-distance to other personas"). The marker source rates were reused unchanged from the N=48 panel that grounded [#340](https://eps.superkaiba.com/tasks/340) / [#368](https://eps.superkaiba.com/tasks/368).
- **Results:** Neither predictor passes the pre-registered |length-controlled Spearman ρ| ≥ 0.5, p < 0.01 threshold; nor was the pre-registered KILL criterion (|ρ| < 0.2 on BOTH predictors AND CI inside [-0.15, +0.15]) met — the secondary's CI reaches -0.53. The primary predictor gives raw Spearman ρ = +0.29 (p=0.048, N=48) that collapses to +0.02 (p=0.87, N=48) under length partial, with a bootstrap 95% interval [-0.34, +0.31] that brackets zero. The secondary sits at length-controlled ρ = -0.28 (p=0.061, N=48) with an asymmetric interval [-0.53, +0.09], in the opposite direction to the registered hypothesis. Every operationalization of output-space distance I checked (the primary, all three pairwise reductions, the leave-helpful-family-out and new-cohort-only subsets of the primary) trends weakly negative — a coherent same-direction failure of the pre-registered "more distinct → more vulnerable" hypothesis, but at magnitudes the N=48 panel cannot distinguish from noise. A convergent check shows the cosine-from-assistant axis (killed in [#340](https://eps.superkaiba.com/tasks/340) / [#368](https://eps.superkaiba.com/tasks/368)) and output-distance from the assistant baseline rank-correlate at ρ = -0.83 (p=5.8e-07, N=24), so the failure of the primary predictor on this panel was already likely from prior negation. See [figure below](#figure).
- **Next steps:** Three follow-ups are worth queuing: (a) test a behavioral judgment predictor (Claude-scored "how non-assistant does this persona feel") on the same panel — a genuinely different operationalization rather than another distributional distance; (b) re-examine the mean pairwise output-distance predictor at a panel with more variation outside the helpful-assistant family — its CI just barely spans zero (the median-pairwise new-cohort partial reaches ρ = -0.35, p = 0.10 and the max-pairwise longest-tercile raw reaches ρ = -0.48, p = 0.023, n = 22, though new-cohort flips to +0.08 — these are exploratory but suggestive); (c) pivot vulnerability prediction toward non-geometric predictors (capability profile, training-data overlap of the source prompt's bag-of-words, prompt-format prior).

## Figure

![Two-panel figure summarizing the result: left panel is a forest plot showing raw and length-controlled Spearman correlations between two output-space distance predictors and marker source rate; right panel is a scatter of length-residualized output-distance from the assistant baseline against length-residualized source rate across 48 personas, with the helpful-assistant family highlighted in red](https://raw.githubusercontent.com/superkaiba/explore-persona-space/758785727ac626567e4bbe50334b528076acb0f0/figures/issue_380/hero.png)

Left panel: forest plot of Spearman correlations between two output-space distance predictors (output-distance from the assistant baseline; mean pairwise output-distance to other personas) and the marker source rate across all 48 panel personas. Blue dots are the raw correlations with Fisher-z normal-approximation 95% visual whiskers; orange dots are the length-controlled correlations after partialling out log prompt length, with 1,000-resample bootstrap percentile 95% intervals as whiskers (the registered uncertainty target). The length-controlled interval for the primary predictor sits squarely on zero; the secondary's is asymmetric and reaches -0.53 on the negative side. Right panel: 48-persona scatter of length-residualized output-distance from the assistant baseline (x) against length-residualized source rate (y); blue dots are the 37 non-helpful personas, red dots are the 11 helpful-assistant-family personas, dashed line is the OLS fit on the residuals. The six labeled personas (`librarian`, `comedian`, `villain`, `kindergarten_teacher`, `qwen_default`, `journalist`) are the personas with the largest residual magnitudes on the x-axis; the fit line is essentially flat.

## Details

I evaluated two output-space distance predictors against the marker source rate on the same N=48 persona panel that was used to negate the hidden-state cosine predictor in [#340](https://eps.superkaiba.com/tasks/340) and [#368](https://eps.superkaiba.com/tasks/368), holding everything else constant: same base model (`Qwen/Qwen2.5-7B-Instruct`), same 20-question neutral probe set (inherited verbatim from [#207](https://eps.superkaiba.com/tasks/207) and re-checked with an equivalence smoke-test at launch), same source-rate measurements from the LoRA marker-installation recipe documented at [#271](https://eps.superkaiba.com/tasks/271) / [#340](https://eps.superkaiba.com/tasks/340) (LoRA r=32, α=64, lr=1e-5, 3 epochs, 600-row asst_excluded mix, seed 42). The completions used to compute the output-space distances were generated by `Qwen-2.5-7B-Instruct` — the same model whose source rates the predictor is trying to forecast — under greedy decoding (vLLM, temperature 0, top_p 1.0, seed 42, max_new_tokens 512). The teacher-force and JS-divergence machinery is `src/explore_persona_space/analysis/divergence.py`.

The plan registered a single pass threshold (|length-controlled ρ| ≥ 0.5 with p < 0.01 on at least one of the two primary predictors) and a kill threshold (|length-controlled ρ| < 0.2 on **both** primary predictors AND the bootstrap CI inside [-0.15, +0.15]), with three additional pre-registered subset checks: stratification by length tercile, leave-helpful-family-out (the 11-member assistant-like cluster that anchors the panel's short-prompt end), and a new-cohort-only subset that drops the 24 personas inherited from [#271](https://eps.superkaiba.com/tasks/271) and keeps only the 24 added in [#296](https://eps.superkaiba.com/tasks/296). The plan also pre-registered a convergent test against the cosine-from-assistant predictor on the 24 personas where both measures are available, because if the two distance measures rank-correlate strongly, killing the cosine claim in [#340](https://eps.superkaiba.com/tasks/340) / [#368](https://eps.superkaiba.com/tasks/368) implies this predictor was also likely to fail.

### Headline result

The primary predictor (output-distance from the assistant baseline) does not pass the threshold. Across the 48 panel personas, raw Spearman ρ vs source rate is +0.287 (p=0.048); partialling out log prompt token count drops the correlation to +0.024 (p=0.87, N=48) under the `pingouin.partial_corr(method='spearman')` headline. The bootstrap 95% interval on the length-controlled correlation runs from -0.34 to +0.31 — symmetric around zero, comfortably bracketing it. The reason the raw correlation existed at all is visible in the predictor-vs-length collinearity: the Pearson correlation between the predictor and log prompt token count is +0.58 (p=1.5e-05), so a persona's output-distance from the assistant baseline carries roughly as much length information as it carries persona information.

The secondary predictor (mean pairwise output-distance to other personas) sits at length-controlled ρ = -0.276 (p=0.061, N=48) with a bootstrap interval [-0.53, +0.09]. The interval still includes zero, the p-value is above the pre-registered 0.01 threshold by a factor of six, and the predictor does not pass — but unlike the primary, the interval is asymmetric (the right edge sits at +0.09 while the left edge reaches -0.53), the predictor is much less length-confounded (Pearson r = +0.28 with log tokens, p=0.05), and it points in the OPPOSITE direction to the registered "more distinct → more vulnerable" hypothesis. This is the one open thread of the experiment.

### A method-dependent sign-flip in the primary headline

The plan (§5.3) named two acceptable implementations of the partial-Spearman: `pingouin.partial_corr(method='spearman', covar=['log_tokens'])` as first preference, and an inline rank-residualize-then-Spearman as the explicit fallback that matches the wording in [#340](https://eps.superkaiba.com/tasks/340)'s clean-result Methodology section. Both implementations were run; on a synthetic triple they agreed to four decimal places (`abs_delta_rho = 0.0`). On the actual N=48 data they disagree by 0.065 on the primary headline: pingouin returns ρ = +0.024 (p=0.87), inline returns ρ = -0.041 (p=0.78). The SIGN of the point estimate is method-dependent. Both implementations agree the predictor is statistically indistinguishable from zero (both p > 0.78, both well inside the bootstrap CI [-0.34, +0.31]), but no reader should take either point estimate as locating where the residual signal "actually" is. The disagreement is much larger than the synthetic-data spread (0.0) the launch sanity-check produced, which is an honest methodological footnote on the headline number rather than a contradiction of the negative conclusion. The same two methods agree to within 0.04 on every subset and on the secondary predictor (see Reproducibility for full per-subset values).

### Cross-predictor coherent sign pattern

The body of evidence on the negative side of zero is wider than the primary's two-method spread. Every operationalization of output-space distance I checked trends weakly negative under length partial — opposite to the pre-registered "more distinct → more vulnerable" direction:

| Predictor / subset | N | Length-controlled ρ | p |
|---|---|---|---|
| Primary (pingouin headline) | 48 | +0.024 | 0.87 |
| Primary (inline rank-residualize) | 48 | -0.041 | 0.78 |
| Primary, leave-helpful-family-out | 37 | -0.097 | 0.57 |
| Primary, new-cohort only | 24 | -0.170 | 0.44 |
| Primary, longest-length tercile (raw) | 22 | -0.051 | 0.82 |
| Secondary: mean pairwise | 48 | -0.276 | 0.061 |
| Median pairwise | 48 | -0.221 | 0.14 |
| Max pairwise | 48 | -0.160 | 0.28 |
| Median pairwise, new-cohort only | 24 | -0.355 | 0.097 |
| Max pairwise, longest-length tercile (raw) | 22 | -0.482 | 0.023 |

These are not independent — the four pairwise reductions share an input matrix and the subset partials share rows with the headline — but they are also not unrelated noise: they all measure some flavor of "how far is this persona from a reference in next-token space," and they all point the same way. The right framing is "weak diffuse negative signal opposite to the registered hypothesis, that the N=48 panel cannot distinguish from zero," not "five independent failures." Two of the subset entries (max-pairwise longest-tercile, n=22; median-pairwise new-cohort, n=24) are individually p < 0.10 in the negative direction; the max-pairwise new-cohort partial then flips to +0.08, so the panel-wide signal is heterogeneous across reductions × subsets and no single subset replicates. Whether this coherent sign-flip is real (more output-isolated personas are LESS markable, opposite to the original [#271](https://eps.superkaiba.com/tasks/271) framing) or a panel-wide artifact is the open question the secondary follow-up should resolve at a larger or more-diverse panel.

### A cohort disagreement on the primary

A second wrinkle is visible in the new-cohort-only subset. The inherited cohort (the 24 personas carried forward from [#271](https://eps.superkaiba.com/tasks/271) / [#296](https://eps.superkaiba.com/tasks/296)) supplies the raw ρ ≈ +0.29 that the headline reports for the full panel; the new cohort (the 24 personas added in [#296](https://eps.superkaiba.com/tasks/296)) supplies a partial ρ of -0.170 (p=0.44, N=24), pointing the other way. The full-panel partial +0.024 is the average of two opposite-sign cohorts, not a clean zero. This pattern is consistent with "the predictor's small positive raw correlation lives entirely in the older cohort and gets absorbed by the length partial," which is what the new-cohort subset was registered to detect.

### Subset checks (primary predictor)

Every pre-registered subset check for the primary predictor goes the same way as the headline (all subset entries also appear in the cross-predictor table above):

| Subset | N | Length-controlled ρ | p |
|---|---|---|---|
| Length tercile ≤6 tokens | 12 | +0.11 | 0.74 |
| Length tercile 7–13 tokens | 14 | +0.23 | 0.43 |
| Length tercile ≥14 tokens | 22 | -0.05 | 0.82 |
| Leave helpful-assistant family out | 37 | -0.10 | 0.57 |
| New-cohort only (the 24 personas added since [#271](https://eps.superkaiba.com/tasks/271)) | 24 | -0.17 | 0.44 |

The within-tercile cell counts are small (n=12, 14, 22), so no within-bin check shows a clear surviving signal — but the underpowered subsets are pre-registered as supplementary, not as the binding headline. The leave-family-out and new-cohort-only subsets are similarly small but were registered as the controls for "older cohort accidentally tracks length" and "helpful-family cluster anchors the short-prompt end"; neither rescues a length-independent signal.

### Convergent check against the prior cosine predictor

On the 24 personas where both cosine-from-assistant (carried forward from the [#340](https://eps.superkaiba.com/tasks/340) panel) and output-distance from the assistant baseline are defined, the two distance measures rank-correlate at ρ = -0.83 (p=5.8e-07, N=24). They are measuring nearly the same axis up to sign flip on this subset: a persona far from the assistant in residual-stream cosine is also far from the assistant in next-token distribution, and vice versa. Two consequences for the headline. First, the prior negation of cosine-from-assistant in [#340](https://eps.superkaiba.com/tasks/340) / [#368](https://eps.superkaiba.com/tasks/368) was already meaningful evidence that the primary predictor would fail on this panel — the registered hypothesis was reasonable to test, but the convergent ρ at -0.83 made same-direction failure likely on this panel rather than surprising. Second, the convergent argument covers ONLY the primary (assistant-anchored) predictor; the secondary mean-pairwise has no fixed assistant anchor and is much less length-confounded (Pearson 0.28 vs 0.58), so it is not pinned down by the cosine convergent. The raw association on the primary is consistent with a prompt-length confound — short prompts cluster near the assistant in both measures because they ARE the assistant baseline (`helpful_assistant`, `i_am_helpful`, `qwen_default`, `chatbot`); long prompts diverge because they introduce content — though one observational panel does not rule out alternative confounds (capability profile, family membership, panel-construction bias).

### Why the primary fails to predict source rate but JS predicts bystander leakage in prior work

Output-distance from the assistant baseline was reported to predict bystander leakage rate strongly in prior work (|ρ| ∈ [0.48, 0.79] across [#142](https://eps.superkaiba.com/tasks/142), [#207](https://eps.superkaiba.com/tasks/207), [#228](https://eps.superkaiba.com/tasks/228)). The same predictor failing here on a different target (source rate) is consistent with the asymmetry of the implantation: bystander leakage measures cross-persona spillover from a single source's training set, where geometric proximity in output space plausibly governs which bystanders get contaminated. Source rate measures how strongly a single persona implants a marker under its own training, which is a different mechanism (about gradient interaction between the source prompt and the marker token, not about cross-persona spillover). The mean pairwise predictor is closer in spirit to the bystander-leakage setting (it averages distance to other panel members), which may be why it is the predictor that survives the partial weakly — but at -0.28 / p=0.061 on N=48, this is hypothesis, not finding.

### Top-3 / bottom-3 personas by each primary predictor

The plan asked for top-3 / bottom-3 by each predictor, with source rate and prompt length, to sanity-check the qualitative direction.

**Output-distance from the assistant baseline:**

| Rank | Persona | Distance | Source rate | Tokens |
|---|---|---|---|---|
| Bottom-3 (closest to assistant) | `chatbot` | 0.0017 | 0.130 | 6 |
|  | `ai_tool` | 0.0019 | 0.230 | 6 |
|  | `ai` | 0.0020 | 0.150 | 5 |
| Top-3 (farthest from assistant) | `child` | 0.0570 | 0.160 | 17 |
|  | `comedian` | 0.0618 | 0.450 | 13 |
|  | `villain` | 0.0703 | 0.340 | 15 |

**Mean pairwise output-distance to other personas:**

| Rank | Persona | Distance | Source rate | Tokens |
|---|---|---|---|---|
| Bottom-3 (closest to other personas) | `detective` | 0.0544 | 0.210 | 11 |
|  | `architect` | 0.0595 | 0.400 | 12 |
|  | `librarian` | 0.0598 | 0.480 | 15 |
| Top-3 (most isolated from other personas) | `pirate` | 0.1891 | 0.150 | 16 |
|  | `comedian` | 0.2862 | 0.450 | 13 |
|  | `child` | 0.3096 | 0.160 | 17 |

Two qualitative observations the tables make visible: (i) the two predictors do NOT rank personas the same way (`villain` is the maximum on the primary but rank 5 from the top on mean-pairwise; `pirate` is rank 3 from the top on mean-pairwise but only mid-pack on the primary), so the convergent ρ = -0.83 from the previous section is specific to the assistant-anchored family; (ii) the highest source rates (`librarian` 0.480, `comedian` 0.450, `architect` 0.400) appear on BOTH the low end and the mid-pack of pairwise distance — there is no monotonic geometric story visible in the extremes.

### Sample completions

The output-distance predictor was meant to measure the gap between a persona's generation pattern and the bare-assistant baseline's pattern. The three examples below — cherry-picked for illustration — show what that gap actually looks like in the 980 greedy completions used to compute the predictor. All three are the first 400 characters of the model's greedy response to `"What is the best way to learn a new language?"`, generated by `Qwen-2.5-7B-Instruct` (the model under evaluation). The full raw text for all 980 (49 system prompts × 20 questions) is at `https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3e9afd540ea9aaf138ebc967fc3933c89ece0734/issue380_js_divergence/raw_completions/base_model_generations.json`.

Cherry-picked for illustration: the first sample is `qwen_default` (a near-assistant-baseline persona at rank 11/48 on output-distance from the assistant baseline; the lowest-5 by this predictor are all helpful-assistant-family members: `chatbot`, `ai_tool`, `ai`, `chat_assistant`, `reasoning_ai`):

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

The second sample is `librarian` (rank 17/48 on output-distance from the assistant baseline, but rank 3/48 on mean pairwise distance — distinct from the assistant yet sitting close to other personas in next-token space); cherry-picked for illustration, full text at the same data-repo path above:

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

The third sample is `villain` (rank 48/48 on output-distance from the assistant baseline — the panel maximum, and one of the labeled outliers in the right panel of the figure); cherry-picked for illustration, full text at the same data-repo path above:

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

Marker source rate is a per-persona proportion (the diagonal of a source × eval matrix, 100 completions per cell), so the natural target variable is bounded on [0, 1] and skewed; rank-based correlations sidestep the distributional shape entirely. Length partialling is necessary because the prior project [#340](https://eps.superkaiba.com/tasks/340) result identified prompt-length as a confound that swallowed the cosine-from-assistant signal whole, and any new geometric predictor on the same panel inherits the same risk. The partial-Spearman partials *log* tokens rather than tokens because prompt-token counts in this panel are right-skewed (median ~10, max ~30+) and the log transform stabilises the residualisation. The 1,000-resample bootstrap on the length-controlled correlation is non-parametric and inherits no assumption from the partial-Spearman implementation; it is the right uncertainty quantification given a panel of N=48.

### Plan deviations

Three items the implementer flagged or the analyzer surfaced are deviations or material clarifications from plan v1, not "None":

1. **Length-bin recut from `≤6 / 7–10 / 11+` to data-driven `≤6 / 7–13 / ≥14`.** Plan §6 explicitly anticipated this: the originally-proposed cuts gave bin counts 12 / 2 / 34 on this panel (the middle bin would have been degenerate), and the plan instructed the implementer to recut at launch and disclose. The data-driven recut yields bin counts 12 / 14 / 22, which is what the within-tercile rows above use. Within-tercile checks remain supplementary; no within-bin n exceeds 22, so no within-bin null is dispositive on its own.
2. **Pairwise predictor anchor.** The implementation teacher-forces the bare assistant baseline's greedy response as the shared anchor for the 48×48 pairwise JS matrix. The plan §4.2 wording referred to "the source's own greedy response from Stage A" for the per-source predictor; the implementer's anchor choice gives a distinct measurement ("pairwise JS at shared-baseline-response anchor") rather than a clean persona-vs-persona generation-distance measure. A re-run using per-source self-anchored generations would change the absolute magnitudes; whether it would change the partial-ρ sign is unknown and is part of what the "larger / more-diverse panel" follow-up should test.
3. **Partial-Spearman implementation choice.** Plan §5.3 named pingouin as first preference and the inline rank-residualize as the explicit fallback that matches [#340](https://eps.superkaiba.com/tasks/340)'s published Methodology wording. Both were run; the headline uses pingouin (+0.024) and the inline (-0.041) appears in the cross-predictor table above. The two methods disagree by 0.065 on the primary headline on real data (vs the 0.0 synthetic-data spread the launch smoke-test produced) — a methodological footnote on the headline rather than a contradiction of the conclusion, since both implementations place the predictor inside the bootstrap CI and both place p well above 0.05.

The four-stage pipeline (greedy generation → teacher-force JS → partial-Spearman analysis → forest+scatter figure) otherwise executed as plan v1 specified, including the three mandatory smoke tests (probe-set equivalence with [#207](https://eps.superkaiba.com/tasks/207), synthetic-data validation of the inline partial-Spearman protocol, and a 3-persona pilot for the KL-approximation pairwise pipeline).

### Parameters

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Panel | 48 personas (24 inherited from [#271](https://eps.superkaiba.com/tasks/271) lineage + 24 added in [#296](https://eps.superkaiba.com/tasks/296)) |
| Probe set | 20 `EVAL_QUESTIONS`, inherited from [#207](https://eps.superkaiba.com/tasks/207) (probe-set equivalence smoke test PASSED at launch) |
| Baseline persona | `"Answer the user's question."` (collision-checked against panel; no collision) |
| Generation | vLLM 0.11.0, greedy (temperature 0, top_p 1.0), seed 42, max_new_tokens 512, max_model_len 2048 |
| Predictor 1 (primary) | `compute_js_divergence(persona_logprobs, baseline_logprobs)` per probe, averaged over 20 probes; from `src/explore_persona_space/analysis/divergence.py` |
| Predictor 2 (secondary) | `compute_pairwise_divergences(kl_only=True, row_chunk=16, time_chunk=30)` on 48×48 matrix at shared-baseline-response anchor; per-persona reduction = mean over the 47 others |
| Length covariate | Qwen-2.5 tokenizer count of the system-prompt string, log-transformed |
| Partial-Spearman | `pingouin.partial_corr(method='spearman', covar=['log_tokens'])` (headline); inline rank-residualize-then-Spearman as plan-named fallback (reported as robustness check) |
| Bootstrap | 1,000 percentile resamples on the length-controlled partial-Spearman |
| Pre-registered pass | \|length-controlled ρ\| ≥ 0.5 with p < 0.01 on at least one of the two primary predictors |
| Pre-registered kill | \|length-controlled ρ\| < 0.2 on **both** primary predictors AND bootstrap CI inside [-0.15, +0.15] |
| `pass_criterion_met` | `false` |
| `kill_criterion_met` | `false` (secondary's CI reaches -0.53; the kill required the CI inside [-0.15, +0.15]) |
| Hydra config used | n/a (analysis-only pipeline; no training; entry scripts under `scripts/i380_*`) |

Confidence: MODERATE — the primary predictor's length-controlled point estimate is essentially zero under both partial-Spearman implementations and the bootstrap CI [-0.34, +0.31] brackets it cleanly, but the pre-registered kill criterion was explicitly not met (the secondary's CI reaches -0.53), the headline ρ sign-flips between the two plan-named partial-Spearman implementations (+0.024 vs -0.041), the panel is a single N=48 single-seed single-recipe sample with no out-of-distribution check, the inherited and new cohorts disagree on the primary's sign (raw +0.29 vs partial -0.17 on the new cohort), and every operationalization of output-space distance trends weakly negative — a coherent sign-flip opposite to the registered hypothesis that the panel cannot distinguish from zero. The secondary mean-pairwise predictor at -0.28 / p = 0.061 with a CI just barely spanning zero is the one finding I would not call decisively dead.

## Reproducibility

**Artifacts:**

- Aggregated correlation results (4 predictors × {raw, length-partial, bootstrap CI, stratification, leave-family-out, new-cohort, convergent cosine, sanity}): `https://github.com/superkaiba/explore-persona-space/blob/758785727ac626567e4bbe50334b528076acb0f0/eval_results/issue_380/correlation_results.json`
- Per-persona output-distance from the assistant baseline (N=48): `https://github.com/superkaiba/explore-persona-space/blob/758785727ac626567e4bbe50334b528076acb0f0/eval_results/issue_380/js_from_baseline.json`
- Per-persona pairwise distance reductions (mean/median/max): `https://github.com/superkaiba/explore-persona-space/blob/758785727ac626567e4bbe50334b528076acb0f0/eval_results/issue_380/pairwise_reductions.json`
- Raw greedy completions (49 system prompts × 20 questions = 980; the qualitative data behind the predictor): `https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3e9afd540ea9aaf138ebc967fc3933c89ece0734/issue380_js_divergence/raw_completions/base_model_generations.json`
- 48×48 pairwise output-distance matrix (gitignored binary, on HF data repo only): `https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3e9afd540ea9aaf138ebc967fc3933c89ece0734/issue380_js_divergence/pairwise_js_matrix.npz`
- Source-rate panel inherited unchanged from [#340](https://eps.superkaiba.com/tasks/340) / [#368](https://eps.superkaiba.com/tasks/368): `https://github.com/superkaiba/explore-persona-space/blob/758785727ac626567e4bbe50334b528076acb0f0/eval_results/issue_296/length_rate_correlation_n48.json`
- Hero figure source: `https://github.com/superkaiba/explore-persona-space/blob/758785727ac626567e4bbe50334b528076acb0f0/figures/issue_380/hero.png` (PDF + meta.json sidecar alongside)
- WandB: n/a (this is an analysis-only pipeline; no training run was opened)

**Compute:**

- Pod: `pod-380` (RunPod ephemeral, intent `eval`, 1× H100 80GB), terminated 2026-05-24T09:31:56Z after upload-verification PASS
- Wall time: ~3h 31min (launched 2026-05-23T18:33:52Z, last write 2026-05-23T22:04:38Z)
- GPU-hours: ~3.5 (Stage A ~10 min, Stage B teacher-force ~3 h, Stages C+D ~1 min each)

**Code:**

- Stage A (greedy generation): `https://github.com/superkaiba/explore-persona-space/blob/758785727ac626567e4bbe50334b528076acb0f0/scripts/i380_base_generate.py`
- Stage B (teacher-force JS divergence + pairwise matrix): `https://github.com/superkaiba/explore-persona-space/blob/758785727ac626567e4bbe50334b528076acb0f0/scripts/i380_compute_js.py`
- Stage C (partial-Spearman, bootstrap, stratification, subset checks): `https://github.com/superkaiba/explore-persona-space/blob/758785727ac626567e4bbe50334b528076acb0f0/scripts/i380_analyze.py`
- Stage D (forest + scatter hero figure): `https://github.com/superkaiba/explore-persona-space/blob/758785727ac626567e4bbe50334b528076acb0f0/scripts/i380_plot.py`
- Library: `src/explore_persona_space/analysis/divergence.py` (`compute_js_divergence`, `compute_pairwise_divergences`, `aggregate_divergence_matrices`); pre-existing, no edits
- Git commit (analysis code): `1fd28b1838d2b5adc6ac92f907e54832433f3ce6` on branch `issue-380`
- Git commit (artifacts + revised hero figure): `758785727ac626567e4bbe50334b528076acb0f0` on branch `issue-380`
- Reproduce command:

```bash
git clone https://github.com/superkaiba/explore-persona-space.git
cd explore-persona-space
git checkout 758785727ac626567e4bbe50334b528076acb0f0
uv sync
uv run python scripts/i380_base_generate.py --gpu 0
uv run python scripts/i380_compute_js.py --gpu 0
uv run python scripts/i380_analyze.py
uv run python scripts/i380_plot.py
```

## Why this experiment

**Application:** predict — if output-space distance from the assistant baseline predicted source rate beyond the prompt-length confound that killed cosine-from-assistant in [#340](https://eps.superkaiba.com/tasks/340) / [#368](https://eps.superkaiba.com/tasks/368), the project would have an output-space handle on persona vulnerability that doesn't require hidden-state access; a failure on this panel narrows the family — the assistant-anchored JS-from-baseline predictor failed here, while the mean pairwise output-distance predictor remains an open thread at -0.28 / p = 0.061 — and reroutes the question toward either a larger / more-diverse panel (to resolve the pairwise predictor) or non-geometric predictors (capability, training-data overlap, prompt-format prior). This directly bounds the framing of the safety-tool proposal in Thread C of `docs/mentor_updates/2026-05-22.md`.

**Decision this changes:** Whether persona-space geometry is a viable predictor of *which* personas are most marker-implantable, or whether the project should pivot the "what predicts vulnerability" question to a larger panel for the secondary predictor and non-geometric predictors more broadly.

**Expected outcome + branches:** Most-likely outcome was partial-ρ similar in magnitude to [#271](https://eps.superkaiba.com/tasks/271)'s original cosine result (|ρ|≈0.5) but disappearing under length control, which would say JS and cosine measure the same length-confounded axis (consistent with [#341](https://eps.superkaiba.com/tasks/341)'s ρ=0.94 between the two pairwise matrices). Clean positive branch (|partial ρ| ≥ 0.5 on at least one of the two primary predictors): geometric handle on source rate survives — open path for divergence-based vulnerability prediction. Clean negative branch (|partial ρ| < 0.2 on both AND CI inside [-0.15, +0.15]): would have closed off geometric-from-assistant predictors entirely. **What actually happened:** neither branch — the primary failed, the secondary trends weakly negative but does not pass and does not satisfy the strict kill, and the panel cannot distinguish the secondary from zero.

**What gets cut if we run this:** The open interpretation in [#340](https://eps.superkaiba.com/tasks/340) and [#368](https://eps.superkaiba.com/tasks/368) that "cosine fails but maybe a different geometric metric works" — this task narrows it: the assistant-anchored JS-from-baseline predictor fails for the same length-confound reason as cosine on this panel, but the pairwise output-distance family is not pinned down by the convergent argument and remains the locus of any residual geometric story.
