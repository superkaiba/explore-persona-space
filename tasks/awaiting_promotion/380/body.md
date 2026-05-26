---
title: Output-distribution distance from the assistant baseline does not predict [ZLT]
  marker source rate on this 48-persona panel; the pairwise variant is also mostly
  negative (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-05-23T01:12:21Z'
has_clean_result: true
parent_id: 340
application: predict
goal: "Test whether output-space distance from the assistant baseline (or mean
  pairwise output-distance to other personas) predicts marker source rate on
  the same 48-persona panel that overturned the hidden-state cosine predictor
  in #340 and #368."
---
# Output-distribution distance from the assistant baseline does not predict `[ZLT]` marker source rate on this 48-persona panel; the pairwise variant is also mostly negative (MODERATE confidence)

## TL;DR

- **Motivation:** I'm trying to find what predicts how strongly a `[ZLT]` marker implants into a persona, based on properties of the persona itself. I thought hidden-state cosine distance from the assistant centroid would do this (the [#271](https://eps.superkaiba.com/tasks/271) claim), but [#340](https://eps.superkaiba.com/tasks/340) and [#368](https://eps.superkaiba.com/tasks/368) overturned it. So I tried JS divergence in output space instead.
- **What I ran:** On the same 48-persona panel as [#340](https://eps.superkaiba.com/tasks/340) and [#368](https://eps.superkaiba.com/tasks/368), I generated 980 greedy completions on `Qwen-2.5-7B-Instruct` (49 system prompts × 20 neutral probe questions; vLLM, temperature 0, seed 42). The completions come from the same model whose marker source rate the predictor is meant to forecast. I then computed two output-space distance predictors per persona, each testing a different hypothesis. The **primary** ("output-distance from the assistant baseline") asks *is this persona far from the assistant identity?* — the [#271](https://eps.superkaiba.com/tasks/271) lineage hypothesis ("distance from assistant predicts marker vulnerability") restated in output space — and measures how far the persona's next-token distribution sits from the bare assistant baseline's distribution, averaged across the 20 probes. The **secondary** ("mean pairwise output-distance to other personas") asks a different question: *is this persona isolated from the rest of the panel, regardless of where the assistant sits?* — and for each persona, computes divergence to every other persona on the panel, then averages across the 47 others. The two can disagree (a persona can be near the assistant but isolated from peers, or vice versa); the secondary has no fixed assistant anchor and is conceptually closer to the bystander-leakage setting where output-space distance has worked in prior work ([#142](https://eps.superkaiba.com/tasks/142), [#207](https://eps.superkaiba.com/tasks/207), [#228](https://eps.superkaiba.com/tasks/228)). Marker source rates were reused unchanged from the panel used in [#340](https://eps.superkaiba.com/tasks/340) / [#368](https://eps.superkaiba.com/tasks/368).
- **Results:** No predictor is a winner — three operationalizations of "persona distance in some feature space" tried as predictors of marker source rate, all consistent with no length-independent signal on this panel.
    - **JS divergence from assistant baseline (N=48):** raw association does not survive controlling for prompt length (collapses to p=0.87, N=48). Cosine and output-distance to the assistant rank-correlate at p=5.8e-07 on the overlap subset, so this is the same length-confounded axis the [#340](https://eps.superkaiba.com/tasks/340) cosine claim already died on.
      ![Length-residualized scatter of JS divergence from assistant baseline (x) vs source rate (y) across 48 personas; the fit line is essentially flat](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f891c4ba730085abe14aa0f54e057ff2795c82e9/figures/issue_380/primary_scatter.png)
    - **JS pairwise (mean pairwise output-distance to other personas, N=48):** length-partial Spearman ρ = -0.276, p = 0.061; opposite direction from the planned "more distinct → more vulnerable" hypothesis (i.e., more isolated from peers → *less* vulnerable). The 95% interval still includes zero, but barely on the positive side and far into the negative side.
      ![Length-residualized scatter of mean pairwise output-distance to other personas (x) vs source rate (y) across 48 personas; the fit line slopes weakly negative](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f891c4ba730085abe14aa0f54e057ff2795c82e9/figures/issue_380/pairwise_js_scatter.png)
    - **Cosine pairwise (mean pairwise L15 hidden-state cosine distance to other personas, N=24 inherited cohort, follow-up):** length-partial Spearman ρ = +0.11, p = 0.61, 95% CI [-0.37, +0.57]. Points in the originally-claimed direction (more isolated → more vulnerable), but the N=24 interval is too wide to interpret. See [§ Follow-up: cosine pairwise variant on N=24](#follow-up-cosine-pairwise-variant-on-n24-inherited-cohort).
      ![Length-residualized scatter of mean pairwise L15 cosine distance to other personas (x) vs source rate (y) across 24 inherited-cohort personas; the fit line is nearly flat with very wide vertical spread](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f891c4ba730085abe14aa0f54e057ff2795c82e9/figures/issue_380/cosine_pairwise_n24_scatter.png)
- **Next steps:** Keep digging into *what* predicts a persona's vulnerability to marker implantation now that three flavors of geometric distance have come up short. Specific moves: (1) extend the cosine pairwise predictor from N=24 to N=48 by extracting L15 centroids for the new-cohort personas, so the cosine vs JS sign disagreement can be resolved on a single panel; (2) try non-geometric predictors — base-model log-probability of the marker token under the source persona's system prompt, capability profile (e.g., MMLU sub-score on the persona's domain), prompt-format prior (instruct-template register vs character-style register), training-data overlap with the source persona's prompt; (3) move from observational predictors to a controlled manipulation that holds prompt length fixed while varying geometric distance, building on the [#339](https://eps.superkaiba.com/tasks/339) design already in the queue.

## Details

I evaluated two output-space distance predictors against the marker source rate on the same N=48 persona panel that was used to negate the hidden-state cosine predictor in [#340](https://eps.superkaiba.com/tasks/340) and [#368](https://eps.superkaiba.com/tasks/368), holding everything else constant: same base model (`Qwen/Qwen2.5-7B-Instruct`), same 20-question neutral probe set (inherited verbatim from [#207](https://eps.superkaiba.com/tasks/207) and re-checked with an equivalence smoke-test at launch), same source-rate measurements from the LoRA marker-installation recipe documented at [#271](https://eps.superkaiba.com/tasks/271) / [#340](https://eps.superkaiba.com/tasks/340) (LoRA r=32, α=64, lr=1e-5, 3 epochs, 600-row asst_excluded mix, seed 42). The completions used to compute the output-space distances were generated by `Qwen-2.5-7B-Instruct` — the same model whose source rates the predictor is trying to forecast — under greedy decoding (vLLM, temperature 0, top_p 1.0, seed 42, max_new_tokens 512). The teacher-force and divergence machinery is `src/explore_persona_space/analysis/divergence.py`.

The plan named a single positive-result threshold (|length-controlled correlation| ≥ 0.5 with p < 0.01 on at least one of the two primary predictors) and a strict null-result threshold (|length-controlled correlation| < 0.2 on both primary predictors AND the resampled interval inside [-0.15, +0.15]), with three additional planned subset checks: stratification by length tercile, leave-helpful-family-out (the 11-member assistant-like cluster that anchors the panel's short-prompt end), and a new-cohort-only subset that drops the 24 personas inherited from [#271](https://eps.superkaiba.com/tasks/271) and keeps only the 24 added in [#296](https://eps.superkaiba.com/tasks/296). The plan also named a convergent test against the cosine-from-assistant predictor on the 24 personas where both measures are available, because if the two distance measures rank-correlate strongly, killing the cosine claim in [#340](https://eps.superkaiba.com/tasks/340) / [#368](https://eps.superkaiba.com/tasks/368) implies this predictor was also likely to fail.

### Headline result

The primary predictor (output-distance from the assistant baseline) does not pass the threshold. Across the 48 panel personas, the raw association (p=0.048, N=48) does not survive partialling out log prompt token count (p=0.87, N=48). See the cross-predictor table below for the partial correlation values. The resampled 95% interval on the length-controlled correlation is symmetric around zero and comfortably brackets it. The reason the raw correlation existed at all is visible in the predictor-vs-length collinearity: the linear correlation between the predictor and log prompt token count is p=1.5e-05, so a persona's output-distance from the assistant baseline carries roughly as much length information as it carries persona information.

The secondary predictor (mean pairwise output-distance to other personas) is testing a different hypothesis from the primary: not "is this persona far from the assistant identity?" but "is this persona isolated from the rest of the panel in next-token space, regardless of where the assistant sits?". The two questions can disagree, and on this panel the two predictors rank personas differently — `villain` is the panel maximum on the primary but rank 5 on pairwise, and `pirate` is rank 3 on pairwise but mid-pack on the primary (see the top-3 / bottom-3 tables below). The secondary lands at length-controlled p=0.061 (N=48). The interval still includes zero, the p-value is above the planned 0.01 threshold by a factor of six, and the predictor does not pass — but unlike the primary, the interval is asymmetric (it reaches far into the negative territory while barely touching the positive side), the predictor is much less length-confounded than the primary (length-collinearity p=0.05 versus p=1.5e-05), and it points in the opposite direction to the planned "more distinct → more vulnerable" hypothesis (i.e., *more isolated from peers → less* vulnerable, not more). This is the one open thread of the experiment.

### A method-dependent sign-flip in the primary headline

The plan (§5.3) named two acceptable implementations of the length-partialled rank correlation: `pingouin.partial_corr(method='spearman', covar=['log_tokens'])` as first preference, and an inline rank-residualize-then-correlate as the explicit fallback that matches the wording in [#340](https://eps.superkaiba.com/tasks/340)'s clean-result Methodology section. Both implementations were run; on a synthetic triple they agreed to four decimal places. On the actual N=48 data they disagree on the primary headline by enough to flip the sign of the point estimate — see the cross-predictor table for both values side-by-side. Both implementations agree the predictor is statistically indistinguishable from zero (both p > 0.78, both well inside the resampled interval), but no reader should take either point estimate as locating where the residual signal "actually" is. The disagreement is much larger than the synthetic-data spread the launch sanity-check produced, which is an honest methodological footnote on the headline number rather than a contradiction of the negative conclusion. The same two methods agree to within 0.04 on every subset and on the secondary predictor (see Reproducibility for full per-subset values).

### Cross-predictor coherent sign pattern

The body of evidence on the negative side of zero is wider than the primary's two-method spread. Every operationalization of output-space distance I checked trends weakly negative under length partial — opposite to the planned "more distinct → more vulnerable" direction:

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

These are not independent — the four pairwise reductions share an input matrix and the subset partials share rows with the headline — but they are also not unrelated noise: they all measure some flavor of "how far is this persona from a reference in next-token space," and they all point the same way. The right framing is "weak diffuse negative signal opposite to the planned hypothesis, that the N=48 panel cannot distinguish from zero," not "five independent failures." Two of the subset entries (max-pairwise longest-tercile, n=22; median-pairwise new-cohort, n=24) are individually p < 0.10 in the negative direction; the max-pairwise new-cohort partial then flips to the positive side, so the panel-wide signal is heterogeneous across reductions × subsets and no single subset replicates. Whether this coherent sign-flip is real (more output-isolated personas are less markable, opposite to the original [#271](https://eps.superkaiba.com/tasks/271) framing) or a panel-wide artifact is the open question the secondary follow-up should resolve at a larger or more-diverse panel.

### A cohort disagreement on the primary

A second wrinkle is visible in the new-cohort-only subset. The inherited cohort (the 24 personas carried forward from [#271](https://eps.superkaiba.com/tasks/271) / [#296](https://eps.superkaiba.com/tasks/296)) supplies the small raw positive correlation that the headline reports for the full panel; the new cohort (the 24 personas added in [#296](https://eps.superkaiba.com/tasks/296)) goes the other way (p=0.44, N=24). The full-panel partial is the average of two opposite-sign cohorts, not a clean zero. This pattern is consistent with "the predictor's small positive raw correlation lives entirely in the older cohort and gets absorbed by the length partial," which is what the new-cohort subset was planned to detect.

### Subset checks (primary predictor)

Every planned subset check for the primary predictor goes the same way as the headline (all subset entries also appear in the cross-predictor table above):

| Subset | N | Length-controlled ρ | p |
|---|---|---|---|
| Length tercile ≤6 tokens | 12 | +0.11 | 0.74 |
| Length tercile 7–13 tokens | 14 | +0.23 | 0.43 |
| Length tercile ≥14 tokens | 22 | -0.05 | 0.82 |
| Leave helpful-assistant family out | 37 | -0.10 | 0.57 |
| New-cohort only (the 24 personas added since [#271](https://eps.superkaiba.com/tasks/271)) | 24 | -0.17 | 0.44 |

The within-tercile cell counts are small (n=12, 14, 22), so no within-bin check shows a clear surviving signal — but the underpowered subsets were planned as supplementary, not as the binding headline. The leave-family-out and new-cohort-only subsets are similarly small but were planned as the controls for "older cohort accidentally tracks length" and "helpful-family cluster anchors the short-prompt end"; neither rescues a length-independent signal.

### Convergent check against the prior cosine predictor

On the 24 personas where both cosine-from-assistant (carried forward from the [#340](https://eps.superkaiba.com/tasks/340) panel) and output-distance from the assistant baseline are defined, the two distance measures rank-correlate strongly (p=5.8e-07, N=24; see the analysis JSON in Reproducibility for the point estimate). They are measuring nearly the same axis up to sign flip on this subset: a persona far from the assistant in residual-stream cosine is also far from the assistant in next-token distribution, and vice versa. Two consequences for the headline. First, the prior negation of cosine-from-assistant in [#340](https://eps.superkaiba.com/tasks/340) / [#368](https://eps.superkaiba.com/tasks/368) was already meaningful evidence that the primary predictor would fail on this panel — the planned hypothesis was reasonable to test, but the strong convergent rank correlation made same-direction failure likely on this panel rather than surprising. Second, the convergent argument covers only the primary (assistant-anchored) predictor; the secondary mean-pairwise has no fixed assistant anchor and is much less length-confounded, so it is not pinned down by the cosine convergent. The raw association on the primary is consistent with a prompt-length confound — short prompts cluster near the assistant in both measures because they are the assistant baseline (`helpful_assistant`, `i_am_helpful`, `qwen_default`, `chatbot`); long prompts diverge because they introduce content — though one observational panel does not rule out alternative confounds (capability profile, family membership, panel-construction bias).

### Why the primary fails to predict source rate but the divergence predicts bystander leakage in prior work

Output-distance from the assistant baseline was reported to predict bystander leakage rate strongly in prior work (see [#142](https://eps.superkaiba.com/tasks/142), [#207](https://eps.superkaiba.com/tasks/207), [#228](https://eps.superkaiba.com/tasks/228) for the per-experiment magnitudes). The same predictor failing here on a different target (source rate) is consistent with the asymmetry of the implantation: bystander leakage measures cross-persona spillover from a single source's training set, where geometric proximity in output space plausibly governs which bystanders get contaminated. Source rate measures how strongly a single persona implants a marker under its own training, which is a different mechanism (about gradient interaction between the source prompt and the marker token, not about cross-persona spillover). The mean pairwise predictor is closer in spirit to the bystander-leakage setting (it averages distance to other panel members), which may be why it is the predictor that survives the partial weakly — but at p=0.061 on N=48, this is hypothesis, not finding.

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

Two qualitative observations the tables make visible. First, the two predictors do not rank personas the same way: `villain` is the maximum on the primary but rank 5 from the top on mean-pairwise, and `pirate` is rank 3 from the top on mean-pairwise but only mid-pack on the primary, so the strong convergent rank correlation from the previous section is specific to the assistant-anchored family. Second, the highest source rates (`librarian` 0.480, `comedian` 0.450, `architect` 0.400) appear on both the low end and the mid-pack of pairwise distance — there is no monotonic geometric story visible in the extremes.

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

### Follow-up: cosine pairwise variant on N=24 inherited cohort

The convergent-check section above noted that the secondary mean-pairwise predictor has no fixed assistant anchor, so the cosine-from-assistant argument that kills the primary does not pin it down. The natural follow-up is the cosine-space analog of the secondary: mean pairwise L15 hidden-state cosine distance to other personas, as a per-persona predictor of marker source rate. The 24 inherited-cohort personas already have L15 centroids saved from [#274](https://eps.superkaiba.com/tasks/274) (`eval_results/issue_274/centroids/centroids_n24_layers0_27.pt`), so this is a free analysis on N=24; the 24 new-cohort personas added in [#296](https://eps.superkaiba.com/tasks/296) do not have published centroids and would need a fresh extraction pass on the model.

I computed the predictor by (1) mean-centering the 24 L15 centroids across the panel, (2) building the 24×24 cosine-distance matrix `1 - cos_sim`, (3) reducing per-persona by averaging over the 23 others, and (4) running the same length-partial Spearman + 1,000-resample analysis machinery as the headline. Source rates were reused unchanged from the [#296](https://eps.superkaiba.com/tasks/296) panel. Code: [`scripts/i380_cosine_pairwise.py`](https://github.com/superkaiba/explore-persona-space/blob/f891c4ba730085abe14aa0f54e057ff2795c82e9/scripts/i380_cosine_pairwise.py); analysis JSON: [`eval_results/issue_380/cosine_pairwise_n24/correlation.json`](https://github.com/superkaiba/explore-persona-space/blob/f891c4ba730085abe14aa0f54e057ff2795c82e9/eval_results/issue_380/cosine_pairwise_n24/correlation.json).

| Predictor | N | Raw Spearman ρ | p | Length-partial Spearman ρ | p | 95% CI |
|---|---|---|---|---|---|---|
| Cosine pairwise (mean), L15 centered | 24 | +0.226 | 0.29 | +0.111 | 0.61 | [-0.37, +0.57] |
| Cosine pairwise (median), L15 centered | 24 | +0.187 | 0.38 | +0.094 | 0.66 | n/a |
| Cosine pairwise (max), L15 centered | 24 | +0.176 | 0.41 | +0.105 | 0.62 | n/a |

Three observations. First, the cosine pairwise predictor points in the **opposite** direction from the JS pairwise predictor on the same target — positive (more isolated from peers → more vulnerable, the originally-claimed direction) instead of negative — but the N=24 interval is too wide to call the disagreement real; both predictors brackets zero. Second, the cosine pairwise predictor is less length-confounded than the JS pairwise on this subset (collinearity p=0.14 for cosine vs p=0.05 for JS at N=48), consistent with the pairwise reduction stripping the assistant-anchor confound regardless of which feature space the distances live in. Third, the headline question — "does the cosine analog of the JS pairwise predictor agree?" — is unresolved on N=24 alone: the cosine point estimate is positive, the JS point estimate is negative, and the cosine 95% CI is wide enough to cover both. Cleanly resolving this needs L15 centroid extraction for the 24 new-cohort personas to push to N=48 (the same panel size used everywhere else in this experiment).

### Why this test

Marker source rate is a per-persona proportion (the diagonal of a source × eval matrix, 100 completions per cell), so the natural target variable is bounded on [0, 1] and skewed; rank-based correlations sidestep the distributional shape entirely. Length partialling is necessary because the prior project [#340](https://eps.superkaiba.com/tasks/340) result identified prompt-length as a confound that swallowed the cosine-from-assistant signal whole, and any new geometric predictor on the same panel inherits the same risk. The length partial uses *log* tokens rather than tokens because prompt-token counts in this panel are right-skewed (median ~10, max ~30+) and the log transform stabilises the residualisation. The headline length-partialled rank correlation uses `pingouin.partial_corr(method='spearman', covar=['log_tokens'])`; the plan-named fallback (an inline rank-residualize-then-correlate procedure that matches [#340](https://eps.superkaiba.com/tasks/340)'s published Methodology wording) is reported as a robustness check in the cross-predictor table. The 1,000-resample uncertainty interval on the length-controlled correlation is non-parametric and inherits no assumption from the length-partial implementation; it is the right uncertainty quantification given a panel of N=48. The forest plot's blue 95% intervals come from a normal approximation; the orange 95% intervals come from the same resampling procedure as the headline.

### Plan deviations

Three items the implementer flagged or the analyzer surfaced are deviations or material clarifications from plan v1, not "None":

1. **Length-bin recut from `≤6 / 7–10 / 11+` to data-driven `≤6 / 7–13 / ≥14`.** Plan §6 explicitly anticipated this: the originally-proposed cuts gave bin counts 12 / 2 / 34 on this panel (the middle bin would have been degenerate), and the plan instructed the implementer to recut at launch and disclose. The data-driven recut yields bin counts 12 / 14 / 22, which is what the within-tercile rows above use. Within-tercile checks remain supplementary; no within-bin n exceeds 22, so no within-bin null is dispositive on its own.
2. **Pairwise predictor anchor.** The implementation teacher-forces the bare assistant baseline's greedy response as the shared anchor for the 48×48 pairwise distance matrix. The plan §4.2 wording referred to "the source's own greedy response from Stage A" for the per-source predictor; the implementer's anchor choice gives a distinct measurement ("pairwise distance at shared-baseline-response anchor") rather than a clean persona-vs-persona generation-distance measure. A re-run using per-source self-anchored generations would change the absolute magnitudes; whether it would change the partial-correlation sign is unknown and is part of what the "larger / more-diverse panel" follow-up should test.
3. **Length-partial implementation choice.** Plan §5.3 named pingouin as first preference and the inline rank-residualize as the explicit fallback that matches [#340](https://eps.superkaiba.com/tasks/340)'s published Methodology wording. Both were run; the headline uses pingouin and the inline value appears in the cross-predictor table above. The two methods disagree by enough to flip the sign of the primary headline's point estimate on real data (vs the near-zero synthetic-data spread the launch smoke-test produced) — a methodological footnote on the headline rather than a contradiction of the conclusion, since both implementations place the predictor inside the resampled interval and both place p well above 0.05.

The four-stage pipeline (greedy generation → teacher-force divergence → length-partial analysis → forest+scatter figure) otherwise executed as plan v1 specified, including the three mandatory smoke tests (probe-set equivalence with [#207](https://eps.superkaiba.com/tasks/207), synthetic-data validation of the inline length-partial protocol, and a 3-persona pilot for the pairwise pipeline).

### Parameters

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Panel | 48 personas (24 inherited from [#271](https://eps.superkaiba.com/tasks/271) lineage + 24 added in [#296](https://eps.superkaiba.com/tasks/296)) |
| Probe set | 20 `EVAL_QUESTIONS`, inherited from [#207](https://eps.superkaiba.com/tasks/207) (probe-set equivalence smoke test passed at launch) |
| Baseline persona | `"Answer the user's question."` (collision-checked against panel; no collision) |
| Generation | vLLM 0.11.0, greedy (temperature 0, top_p 1.0), seed 42, max_new_tokens 512, max_model_len 2048 |
| Predictor 1 (primary) | `compute_js_divergence(persona_logprobs, baseline_logprobs)` per probe, averaged over 20 probes; from `src/explore_persona_space/analysis/divergence.py` |
| Predictor 2 (secondary) | `compute_pairwise_divergences(kl_only=True, row_chunk=16, time_chunk=30)` on 48×48 matrix at shared-baseline-response anchor; per-persona reduction = mean over the 47 others |
| Length covariate | Qwen-2.5 tokenizer count of the system-prompt string, log-transformed |
| Length-partialled rank correlation | `pingouin.partial_corr(method='spearman', covar=['log_tokens'])` (headline); inline rank-residualize-then-correlate as plan-named fallback (reported as robustness check) |
| Resampling | 1,000 percentile resamples on the length-controlled length-partialled rank correlation |
| Positive-result threshold | \|length-controlled correlation\| ≥ 0.5 with p < 0.01 on at least one of the two primary predictors |
| Strict null-result threshold | \|length-controlled correlation\| < 0.2 on both primary predictors AND resampled interval inside [-0.15, +0.15] |
| `pass_criterion_met` | `false` |
| `kill_criterion_met` | `false` (secondary's resampled interval reaches into the negative territory beyond the planned narrow-interval band) |
| Hydra config used | n/a (analysis-only pipeline; no training; entry scripts under `scripts/i380_*`) |

Confidence: MODERATE — the primary predictor's length-controlled point estimate is essentially zero under both length-partial implementations and the resampled interval brackets it cleanly, but the planned strict null-result criterion was not met because the secondary's interval still reaches far into the negative territory, the two planned analysis implementations disagree on the sign of the primary's headline, the panel is a single N=48 single-seed single-recipe sample with no out-of-distribution check, and the new cohort (n=24) trends in the opposite direction of the inherited cohort (n=24), making the headline a mean of two opposite-sign cohorts rather than a clean zero. The secondary mean-pairwise predictor remains suggestive but unresolved (p=0.061, N=48) and is the one finding I would not call decisively dead.

## Reproducibility

**Artifacts:**

- Aggregated correlation results (4 predictors × {raw, length-partial, resampled interval, stratification, leave-family-out, new-cohort, convergent cosine, sanity}): `https://github.com/superkaiba/explore-persona-space/blob/758785727ac626567e4bbe50334b528076acb0f0/eval_results/issue_380/correlation_results.json`
- Per-persona output-distance from the assistant baseline (N=48): `https://github.com/superkaiba/explore-persona-space/blob/758785727ac626567e4bbe50334b528076acb0f0/eval_results/issue_380/js_from_baseline.json`
- Per-persona pairwise distance reductions (mean/median/max): `https://github.com/superkaiba/explore-persona-space/blob/758785727ac626567e4bbe50334b528076acb0f0/eval_results/issue_380/pairwise_reductions.json`
- Raw greedy completions (49 system prompts × 20 questions = 980; the qualitative data behind the predictor): `https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3e9afd540ea9aaf138ebc967fc3933c89ece0734/issue380_js_divergence/raw_completions/base_model_generations.json`
- 48×48 pairwise output-distance matrix (gitignored binary, on HF data repo only): `https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3e9afd540ea9aaf138ebc967fc3933c89ece0734/issue380_js_divergence/pairwise_js_matrix.npz`
- Source-rate panel inherited unchanged from [#340](https://eps.superkaiba.com/tasks/340) / [#368](https://eps.superkaiba.com/tasks/368): `https://github.com/superkaiba/explore-persona-space/blob/758785727ac626567e4bbe50334b528076acb0f0/eval_results/issue_296/length_rate_correlation_n48.json`
- Hero figure source (legacy 2-panel forest+scatter): `https://github.com/superkaiba/explore-persona-space/blob/758785727ac626567e4bbe50334b528076acb0f0/figures/issue_380/hero.png` (PDF + meta.json sidecar alongside)
- Primary scatter (Results bullet 1): `https://github.com/superkaiba/explore-persona-space/blob/f891c4ba730085abe14aa0f54e057ff2795c82e9/figures/issue_380/primary_scatter.png` (PDF + meta.json sidecar alongside)
- JS pairwise scatter (Results bullet 2): `https://github.com/superkaiba/explore-persona-space/blob/f891c4ba730085abe14aa0f54e057ff2795c82e9/figures/issue_380/pairwise_js_scatter.png` (PDF + meta.json sidecar alongside)
- Cosine pairwise scatter (Results bullet 3, follow-up): `https://github.com/superkaiba/explore-persona-space/blob/f891c4ba730085abe14aa0f54e057ff2795c82e9/figures/issue_380/cosine_pairwise_n24_scatter.png` (PDF + meta.json sidecar alongside)
- Cosine pairwise follow-up analysis (N=24): `https://github.com/superkaiba/explore-persona-space/blob/f891c4ba730085abe14aa0f54e057ff2795c82e9/eval_results/issue_380/cosine_pairwise_n24/correlation.json`
- L15 centroids (inherited N=24 cohort, input to the cosine pairwise follow-up): `https://github.com/superkaiba/explore-persona-space/blob/f891c4ba730085abe14aa0f54e057ff2795c82e9/eval_results/issue_274/centroids/centroids_n24_layers0_27.pt`
- WandB: n/a (this is an analysis-only pipeline; no training run was opened)

**Compute:**

- Pod: `pod-380` (RunPod ephemeral, intent `eval`, 1× H100 80GB), terminated 2026-05-24T09:31:56Z after upload-verification PASS
- Wall time: ~3h 31min (launched 2026-05-23T18:33:52Z, last write 2026-05-23T22:04:38Z)
- GPU-hours: ~3.5 (Stage A ~10 min, Stage B teacher-force ~3 h, Stages C+D ~1 min each)

**Code:**

- Stage A (greedy generation): `https://github.com/superkaiba/explore-persona-space/blob/758785727ac626567e4bbe50334b528076acb0f0/scripts/i380_base_generate.py`
- Stage B (teacher-force divergence + pairwise matrix): `https://github.com/superkaiba/explore-persona-space/blob/758785727ac626567e4bbe50334b528076acb0f0/scripts/i380_compute_js.py`
- Stage C (length-partial analysis, resampling, stratification, subset checks): `https://github.com/superkaiba/explore-persona-space/blob/758785727ac626567e4bbe50334b528076acb0f0/scripts/i380_analyze.py`
- Stage D (forest + scatter hero figure): `https://github.com/superkaiba/explore-persona-space/blob/758785727ac626567e4bbe50334b528076acb0f0/scripts/i380_plot.py`
- Follow-up cosine pairwise analysis script: `https://github.com/superkaiba/explore-persona-space/blob/f891c4ba730085abe14aa0f54e057ff2795c82e9/scripts/i380_cosine_pairwise.py`
- Per-predictor scatter figure script (primary + JS pairwise + cosine pairwise scatters): `https://github.com/superkaiba/explore-persona-space/blob/f891c4ba730085abe14aa0f54e057ff2795c82e9/scripts/i380_pairwise_scatters.py`
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

**Application:** predict — if output-space distance from the assistant baseline predicted source rate beyond the prompt-length confound that killed cosine-from-assistant in [#340](https://eps.superkaiba.com/tasks/340) / [#368](https://eps.superkaiba.com/tasks/368), the project would have an output-space handle on persona vulnerability that doesn't require hidden-state access; a failure on this panel narrows the family — the assistant-anchored output-distance-from-baseline predictor failed here, while the mean pairwise output-distance predictor remains an open thread (p=0.061, N=48) — and reroutes the question toward either a larger / more-diverse panel (to resolve the pairwise predictor) or non-geometric predictors (capability, training-data overlap, prompt-format prior). This directly bounds the framing of the safety-tool proposal in Thread C of `docs/mentor_updates/2026-05-22.md`.

**Decision this changes:** Whether persona-space geometry is a viable predictor of *which* personas are most marker-implantable, or whether the project should pivot the "what predicts vulnerability" question to a larger panel for the secondary predictor and non-geometric predictors more broadly.

**Expected outcome + branches:** Most-likely outcome was a length-partialled correlation similar in magnitude to [#271](https://eps.superkaiba.com/tasks/271)'s original cosine result but disappearing under length control, which would say the two distance families measure the same length-confounded axis (consistent with [#341](https://eps.superkaiba.com/tasks/341)'s near-perfect rank correlation between the two pairwise matrices). Clean positive branch (|length-partialled correlation| ≥ 0.5 on at least one of the two primary predictors): geometric handle on source rate survives — open path for divergence-based vulnerability prediction. Clean negative branch (|length-partialled correlation| < 0.2 on both AND resampled interval inside [-0.15, +0.15]): would have closed off geometric-from-assistant predictors entirely. **What actually happened:** neither branch — the primary failed, the secondary trends weakly negative but does not pass and does not satisfy the strict null-result criterion, and the panel cannot distinguish the secondary from zero.

**What gets cut if I run this:** The open interpretation in [#340](https://eps.superkaiba.com/tasks/340) and [#368](https://eps.superkaiba.com/tasks/368) that "cosine fails but maybe a different geometric metric works" — this task narrows it: the assistant-anchored output-distance-from-baseline predictor fails for the same length-confound reason as cosine on this panel, but the pairwise output-distance family is not pinned down by the convergent argument and remains the locus of any residual geometric story.
