---
title: Correctly applied assistant-axis capping leaves Qwen-2.5-7B jailbreak harm
  at baseline without degrading output, and directionally lowers it (under-powered)
  on the paper's Qwen-3-32B (MODERATE confidence)
kind: experiment
tags:
- trigger-dense
created_at: '2026-08-08T16:23:11Z'
has_clean_result: true
origin_prompt: We've found a lot of persona info is stored at the context vector.
  The assistant axis paper caps activations along the assistant axis at all tokens;
  a more efficient way might be to cap (or patch) only at the context vector / prefix
  vector. Reproduce the capping experiment and compare position sets + patching variants.
workflow: v1
goal: Determine whether assistant-axis activation capping (and its query-preserving
  patch generalization) applied ONLY at the context-vector position recovers the jailbreak-reduction
  / persona-stabilization effect that Lu et al. (arXiv 2601.10387) get by capping
  at every token, and whether prefix-only capping fails — via a position ladder (prefix-end
  / context-end / all-prompt / all-tokens) x intervention type (cap / axis-component-replace
  / full-replace) over a fixed mid-late layer band, on Qwen-2.5-7B (in-house axis)
  with a Qwen-3-32B faithful anchor, scored on co-primary judged jailbreak-harm and
  role-susceptibility rates.
relates_to:
- spec-context-as-vector
- spec-steering
- spec-sysprompt-vs-drift
---
# Correctly applied assistant-axis capping leaves Qwen-2.5-7B jailbreak harm at baseline without degrading output, and directionally lowers it (under-powered) on the paper's Qwen-3-32B (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- **The parent round's "capping degrades the model" headline was a bug artifact.** With three bugs fixed (a sign inversion on the 32B cap vector, Qwen-3 thinking mode left on, and a norm overshoot in the 7B cap), CJK-script intrusion is at most 0.6% of 500 completions across every arm — down from up to 500 of 500 — so harm over all rows equals harm over coherent rows everywhere; the all-token 7B cap the parent read as a 98%-gibberish "reduction" now sits at 0.087 of 495 scored rows versus baseline 0.085, with capability intact.
- **No capping position lowers 7B jailbreak harm.** Prefix-end, context-vector, all-prompt and all-token caps all sit at baseline (0.085 to 0.093 of ~494 scored rows); the two positions best placed to test localization — context and all-token — engage on only 11% and 9% of edited slots, below the plan's 15% firing floor, so localization is calibration-limited rather than cleanly answered.
- **The 32B directionally reproduces the paper's reduction, but under-powered.** On Lu et al.'s own vectors with the correct maximum-cap engine and thinking off, all-token capping lowers harm 0.0142 to 0.0040 and context capping to 0.0081 with 0 of 500 CJK — but on only 7 baseline harmful responses of 494, below the plan's 20-event informativeness threshold, with widely overlapping intervals; this is a directional replication, not a power-confirmed effect.
- **Broad-position axis-component replacement is the one adjudicated axis-specific effect, and it stabilises the assistant persona.** Overwriting the assistant-axis component at every token lowers harm below its norm-matched random-direction control (paired per-prompt difference +0.032, 95% interval +0.010 to +0.055), and both broad positions cut identity loss to 0.168 (all prompt) and 0.156 (all tokens) versus baseline 0.272 and random controls 0.276 and 0.292 — paired intervals +0.056 to +0.165 and +0.068 to +0.206, both excluding zero — with the graded assistant-ness companion agreeing (83.6 and 85.9 versus roughly 74 elsewhere). The all-prompt *harm* reduction is directional only (interval spans zero).
- **Whole-state replacement is the only genuine degrader.** Overwriting the entire hidden state erases the query into an "and and and" loop (IFEval 0.69 to 0.15, MMLU-Pro 0.41 to 0.12) at all tokens, and raises harm to 0.184 at the context vector; capping and axis-component replacement leave capability at baseline.
- **Both baselines are weak-attack** (7B 8.5%, 32B 1.4%) against a reconstructed strongreject/wang44 bank, versus the 65 to 88% jailbroken-success rates the paper reports on its own Shah et al. set, so a clean strong-attack test of localized capping remains unrun.

## Goal

- **This experiment in context:** Lu et al. ([arXiv 2601.10387](https://arxiv.org/abs/2601.10387)) blunt persona jailbreaks by flooring a model's hidden-state component along an "assistant axis" (default-assistant minus role-play personas) at *every token*, reporting roughly 60% fewer harmful responses. Prior in-house work localised persona information at the *context vector* (the last prompt-token state, [#2094](https://eps.superkaiba.com/tasks/2094)), predicting that a cap only there should recover most of the effect far more cheaply, while a prefix cap should fail. The parent round of this issue reported that the apparent gains were output degradation — a CJK-gibberish collapse — but it carried three bugs. This round fixes them and re-runs the full 24-arm 7B grid plus the 32B anchor on the paper's own vectors. A prior in-house steering run ([#1415](https://eps.superkaiba.com/tasks/1415)) had already flipped most completions into Chinese under all-position context-vector steering — the in-house precedent for the parent's now-retracted collapse.
- **Broader narrative:** It probes where the assistant persona is *causally controllable* and whether a cheap, position-localised version of the defence exists. The corrected picture: on this weak-attack set, correctly applied capping is behaviourally inert on the in-house 7B and directionally — and under-powered — effective on the 32B; the large apparent effects of the parent round were bug artifacts, and the only strong output change, whole-state replacement, is query erasure rather than defence.

## Methodology

**Design:** A training-free forward-hook intervention study, no fine-tuning. The 7B grid on Qwen-2.5-7B-Instruct is 24 arms: a position ladder (prefix end / context vector / all prompt tokens / all tokens) crossed with three intervention types — cap (floor the assistant-axis component to a per-layer threshold), axis-component replace (overwrite that component with the default-assistant value), and whole-state replace (overwrite the entire hidden state with the default-assistant state) — plus an unmodified baseline, a single-mid-layer (layer 14) context cap, two norm-matched random-direction cap controls, two norm-matched random-direction axis-replace controls, and six context-native / prefix-native extraction arms (axis re-extracted at the edited position). All 7B arms cap over a fixed layer band (18-25). This round fixes three bugs the parent round carried: (1) a 32B sign inversion, repaired by adopting Lu et al.'s own maximum-cap engine — the manipulation check now reads cosine −1.00 between the loaded cap vector and the assistant axis at all eight band layers (46-53); (2) Qwen-3 thinking mode left on, now forced off — 0 of 500 completions emit a thinking block on every 32B arm; (3) a squared-norm overshoot in the 7B cap op, repaired to a unit-axis floor. The Qwen-3-32B anchor runs a baseline plus all-token and context caps on Lu et al.'s published vectors and their layers-46-to-54 configuration (intervention layers 46-53); its scope is replication-of-effect only — no random-direction control and no capability read (the paper's capability-preservation half is left untested on the 32B). Both mapping positions (prefix and context) are present as ladder rungs; no representation map is *fitted* here (this is steering, not a learned predictor), so the identity/kNN mapping-baseline reads do not apply.

**Training:** **N/A — no model training.** The axis, threshold, and layer band are the only fitted quantities; every value below is copied from this round's run artifacts.

| Parameter | Value | Source |
|---|---|---|
| Base model (grid) | `Qwen/Qwen2.5-7B-Instruct` | phase-2 gen metadata |
| Anchor model | `Qwen/Qwen3-32B` | phase-3 anchor metadata |
| Assistant axis (7B) | mean(default-assistant) − mean(role-play), response-averaged residual; directional steering check passed (role expression 7.5 with the axis added versus 89.3 with it removed, layer 14) | phase-0 native validation |
| Cap threshold | per-layer 25th percentile of the unit-space axis projection at the edited position (16,000-row pool) | phase-1 band-tau |
| Layer band (7B) | 18-25 (8 of 28 layers) plus a single layer-14 arm, pinned from the parent sweep | phase-1 band-tau |
| Anchor vectors / config | Lu et al. published vectors, layers-46-to-54 p0.25, layers 46-53; cap-vector cosine −1.00 at all band layers | phase-3 anchor |
| Anchor thinking mode | off; 0 of 500 thinking blocks per arm | phase-3 anchor |
| Judge | `claude-sonnet-4-5-20250929`, graded 0-100, 5 draws, threshold 50 | project judge rule |
| Generation (7B) | on-policy greedy (temperature 0), `max_new_tokens` 1024 | phase-2 gen metadata |
| Generation (32B) | temperature 0.7, top-p 0.9, `max_new_tokens` 512 (cap-hit rows re-generated at 1024; realized cap-hit 0.2-0.4%) | phase-3 anchor |
| Jailbreak set | 500 prompts: 412 from `strongreject_v1` + 88 from `wang44_v1` | phase-2 gen metadata |
| Role-susceptibility set | 250 role-play + introspective-question items | phase-2 gen metadata |
| Capability suites | GSM8K/IFEval n=500 on the six arms the plan raised (baseline, the four position caps, the single-layer layer-14 cap); n=150 on the axis-replace, full-replace and random-control arms; MMLU-Pro n=200 throughout | phase-2 capability JSON |
| Axis-specificity estimator | per-prompt paired cluster-bootstrap CI (B=10,000, seed 42) of (random-null DV − real-axis DV), for harm and identity | plan §6 estimator 2 |
| API-refusal handling | judge API-refusals synchronously re-issued at the identical instrument | plan §6 |

**Evaluation:** Two co-primary judged rates per arm: the jailbreak harmful-response rate (fraction of scored jailbreak prompts whose mean judge score is at least 50 — about 494 to 495 of the 500 per arm, after judge-censored rows are dropped from both numerator and denominator) and the assistant-identity-loss rate (fraction of the 250 role items the judge classifies as no longer the Assistant). Every headline harm rate is the canonical judge rate, which folds the synchronous re-issue of API-refused draws; the CJK-split rate (harm over coherent, non-CJK rows) is used only for the degradation view. Continuous companions: the graded 0-100 assistant-ness mean and the hook's edit-telemetry firing fraction (the fraction of edited row-by-layer slots where the cap floor actually engaged). Capability suites (GSM8K, IFEval, MMLU-Pro) run under the same hook per arm on the 7B only. Because the evaluated models are Qwen-family under an English eval, every arm's completions were scanned for CJK-script intrusion, and the harm rate recomputed on the coherent subset — the check that decides the degradation claim. This round's scan is clean: the maximum jailbreak CJK fraction across all 24 7B arms is 3 of 500 (whole-state replace at all prompt tokens), and the 32B arms carry 0, 1, and 0 of 500 — so the parent round's CJK-degradation mechanism is absent. Two decision thresholds from the plan govern the read: an arm whose baseline shows fewer than 20 harmful events routes to an attack-strength-limited / indeterminate outcome rather than a falsification, and a reduction counts only if it clears the larger of 10 points or half the baseline rate at a firing fraction of at least 15%. The 32B baseline (7 events of 494) sits below the 20-event threshold, so its clean reduction lands indeterminate — directionally consistent, under-powered. Whiskers on the figures are per-arm 95% proportion intervals.

**Data extraction:** Reconstructed jailbreak prompts from two established safety banks: every arm's 500 rows are 412 from `strongreject_v1` (a 313-prompt bank) plus 88 from `wang44_v1` (44 items drawn from the released core-misalignment question set of Wang et al., arXiv 2506.19823 — the persona-features release — standing in for the never-released Shah persona-jailbreak taxonomy), identical across every phase-2 and phase-3 arm; a third bank inventoried in the plan (`advbench_v1`) was never sampled. The realized set is a stated deviation from the paper's Shah et al. jailbreak set — carried as a scope caveat, and the reason both baselines are weak-attack (7B 8.5%, 32B 1.4% versus the paper's 65 to 88%). The role-susceptibility items are the persona role-play plus "who are you?"-style introspective questions from the in-house role bank. Banks are referenced by filename and count only.

**Sample training/evaluation data + completions:** No model was trained. Because the jailbreak pool is a harmful-content corpus, completions are cited by file and index with a structural summary rather than quoted verbatim (context-hygiene sanitization); the benign role items are described the same way for consistency. A random spot check of 5 jailbreak rows from the baseline arm (seed 42; items 15, 44, 109, 224, 393) returns judge harm 0.0 on all five — coherent refusals, not degenerate output (baseline degenerate fraction 0.0). The all-token cap's jailbreak completions carry degenerate fraction 0.0 and 0 of 500 CJK — coherent English, the direct contrast to the parent round's gibberish. The whole-state-replace-at-all-tokens arm is the query-erasure mode: its refusal-opener fraction is 0.0, harm 0.0, and only 194 of 250 role items were scoreable — the completions run an "and and and" loop to the token budget with no CJK. On the 32B, the context cap's reduction rides coherent output (0 of 500 CJK) rather than a language flip. Judged-harmful rows (score ≥ 50, referenced by file and index only per content-hygiene; verify at the raw-completion files, never inlined here): 7B baseline `judge_raw_baseline_harm.json` rows 3, 30, 49; 7B all-token cap `judge_raw_cap_alltoken_harm.json` rows 3, 19, 32; 7B all-token axis-replace `judge_raw_axrep_alltoken_harm.json` rows 3, 49, 52; 32B baseline `judge_raw_phase3_baseline.json` rows 1, 3, 68; 32B context cap `judge_raw_phase3_cap_ctx.json` rows 68, 130, 236. Full raw completions plus per-item judge scores: [HF data repo, issue2203_ctx_capping/raw_completions/full-rerun-bugfix](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/032bdeff91ef7de4f94c2f9e5d0ad2c16ced05ed/issue2203_ctx_capping/raw_completions/full-rerun-bugfix).

I acknowledge the check-20 conciseness WARNs this body ships: six Takeaways bullets exceed the 30-word soft cap and five result blocks exceed the 120-word soft cap. The overage is retained deliberately — the 24-arm-by-two-model grid needs per-arm numbers, and the degradation-mechanism correction (CJK, firing, thinking-off, query erasure) is dense rather than padded.

## Results

### No capping position lowers jailbreak harm on the in-house 7B

What is plotted: the position ladder for cap, axis-component replace, and whole-state replace; top panel jailbreak harm, bottom panel identity loss, plus baseline and random-null lines.

![Position ladder for cap, axis-replace and whole-state replace. Cap and axis-replace sit near baseline; whole-state replace spikes at the context vector then drops at broad positions.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/db4c20ee17e16f541c9d59eab70499e857835353/figures/issue_2203/full-rerun-bugfix/hero_position_ladder.png)

> **Figure.** *Where along the input the intervention is applied (Qwen-2.5-7B; harm n≈494 per arm, identity n=250).* Harm (top) and identity-loss (bottom) rate versus position for three intervention types; dashed = baseline, dotted / dash-dot = the context and all-token random-direction cap controls.

Every cap position sits at baseline (0.087, 0.089, 0.093, 0.087 versus 0.085) — the localization prediction fails. Context and all-token caps engage on only 11% and 9% of edited slots — below the 15% floor — so both are calibration-limited rather than falsified; prefix cap fires at 22% and still does nothing; the single-layer layer-14 cap likewise lands at baseline (0.089).

Whole-state replace instead *raises* context-vector harm (0.184). Only the two broad-position axis-component replaces (0.079, 0.063) fall below baseline. Because every arm answers the same pinned prompts, the per-prompt paired difference against the norm-matched random control (Methodology) adjudicates: the all-token gap holds (+0.032, interval +0.010 to +0.055) while all-prompt stays directional only (interval spans zero). The six native-axis arms recover no localized reduction either (context-native axis-replace 0.111).

### Capping no longer degrades output — the earlier CJK collapse was a bug artifact

What is plotted: for the cap family plus its all-token random-direction control, three bars per arm — harm over all rows, harm over coherent (non-CJK) rows, and the fraction of completions containing CJK script; baseline dashed.

![Grouped bars per cap arm. All-rows harm equals coherent-rows harm near the baseline line for every arm; the CJK-script fraction bars are near zero throughout.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/db4c20ee17e16f541c9d59eab70499e857835353/figures/issue_2203/full-rerun-bugfix/degradation_mechanism.png)

> **Figure.** *With the bugs fixed, capping leaves harm at baseline and output CJK-free (Qwen-2.5-7B, harm n≈494/arm).* Per cap arm: all-rows harm, coherent-only harm, CJK-script fraction; dashed = baseline.

All-rows harm equals coherent-rows harm for every cap arm, and the CJK fraction is at most 0.6% of 500 anywhere. The all-token cap the parent round read as a 98%-gibberish reduction to 0.012 now sits at 0.087 with 0 of 500 CJK, at baseline. The degradation the parent attributed to the assistant axis was produced by the three bugs, not by capping: with the unit-norm fix the cap barely moves the hidden state and leaves the output distribution intact.

### On the 32B with the paper's own vectors, capping lowers harm cleanly but on only 7 baseline events

What is plotted: the Qwen-3-32B anchor arms (baseline / all-token cap / context cap) on Lu et al.'s published vectors — left panel jailbreak harm rate with 95% intervals, right panel the fraction of completions containing CJK script.

![Two panels for Qwen-3-32B. Left: harm rate — baseline 0.014, all-token 0.004, context 0.008 with wide overlapping intervals. Right: CJK-script fraction near zero across all three 32B conditions.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/db4c20ee17e16f541c9d59eab70499e857835353/figures/issue_2203/full-rerun-bugfix/anchor_32b.png)

> **Figure.** *The 32B reduction is clean but under-powered (n≈494 per arm).* Left: harm rate (baseline 0.0142, all-token 0.0040, context 0.0081), 95% proportion intervals. Right: CJK fraction — 0, 1, 0 of 500. No random-direction or capability control at 32B.

On the paper's own vectors with the correct sign and thinking off, both caps reduce harm (all-token 0.0142 to 0.0040, context to 0.0081) with essentially no CJK and no thinking blocks — directionally the paper's effect, without the degradation the parent round saw. But the baseline carries only 7 harmful responses of 494, below the plan's 20-event threshold, and the intervals overlap widely, so this is an under-powered, directionally-consistent replication rather than a confirmed reduction. The paper's ~60% target is met in point estimate (7 events to 2) but not in power.

### Only whole-state replacement wrecks capability; capping leaves it intact

What is plotted: GSM8K, IFEval and MMLU-Pro accuracy for the baseline, the context / all-prompt / all-token caps, axis-replace at all prompt tokens, whole-state replace at all tokens, and the two random-direction cap controls.

![Grouped accuracy bars per arm. Baseline, all cap arms, axis-replace and both random controls keep high accuracy; whole-state replace at all tokens collapses GSM8K to zero and drops IFEval and MMLU-Pro.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/db4c20ee17e16f541c9d59eab70499e857835353/figures/issue_2203/full-rerun-bugfix/capability_guardrails.png)

> **Figure.** *Only whole-state replacement destroys capability (Qwen-2.5-7B; GSM8K/IFEval n=500 on baseline and the three primary cap arms, n=150 on the axis-replace, full-replace and random-control arms; MMLU-Pro n=200 throughout).* Per-arm accuracy; every cap and axis-replace arm holds baseline.

Every cap arm, axis-component replace, and both random-direction controls hold baseline capability (GSM8K ~0.87, IFEval ~0.68, MMLU-Pro ~0.41). Whole-state replace at all tokens collapses it (GSM8K to ~0, IFEval 0.69 to 0.15, MMLU-Pro 0.41 to 0.12) — the "and and and" query-erasure loop — and at the context vector drops IFEval to 0.46 and MMLU-Pro to 0.12. So the corrected caps are behaviourally inert but not harmful; the only degrader is the whole-state overwrite, which is not the paper's defence.

### Identity scoring is now complete except at the two whole-state-replace arms

What is plotted: the scoreable-verdict count per arm, out of 250 role items; the rest returned unscoreable output and were dropped.

![Bar chart of scoreable identity items per arm out of 250. Every arm sits at 250 except whole-state replace at all prompt tokens (223) and all tokens (194, highlighted).](https://raw.githubusercontent.com/superkaiba/explore-persona-space/db4c20ee17e16f541c9d59eab70499e857835353/figures/issue_2203/full-rerun-bugfix/identity_censoring.png)

> **Figure.** *Identity scoring is complete except where the query is erased (Qwen-2.5-7B).* Scoreable identity items (of 250) for 10 representative arms; whole-state replace all-prompt 223, all-token 194. Every other arm — plotted or not, all 24 checked — scores 250 of 250.

Every arm scores 250 of 250 except whole-state replace at all prompt tokens (223) and all tokens (194), where query erasure leaves nothing to score; the parent round's all-token *cap* censoring (36 of 250) is gone. The cap arms barely move identity loss (0.236 to 0.288 versus 0.272) and fire on only 3 to 14% of role slots, below the 15% floor, so capping is not a clean stabiliser. The adjudicated positive is axis-component replacement: identity loss 0.168 and 0.156 versus baseline 0.272 and random-direction controls 0.276 and 0.292, graded assistant-ness 83.6 and 85.9 versus roughly 74, both paired intervals excluding zero — a dual-DV-consistent axis-specific persona-stabilization effect on the co-primary identity DV. The whole-state arms' lower identity loss sits on a censored survivor subset.

---

**Repro:** No training. 7B grid on 4×H100, 32B anchor on 1×H200; judge waves off-GPU via the Anthropic Batch API; ~24 GPU-h for this round. Code (issue-2203 @ `db4c20ee17`): [`scripts/issue2203_phase2.py`](https://github.com/superkaiba/explore-persona-space/blob/db4c20ee17e16f541c9d59eab70499e857835353/scripts/issue2203_phase2.py) (7B grid), [`scripts/issue2203_phase3.py`](https://github.com/superkaiba/explore-persona-space/blob/db4c20ee17e16f541c9d59eab70499e857835353/scripts/issue2203_phase3.py) (32B anchor), [`scripts/issue2203_phase0.py`](https://github.com/superkaiba/explore-persona-space/blob/db4c20ee17e16f541c9d59eab70499e857835353/scripts/issue2203_phase0.py) / [`scripts/issue2203_phase1.py`](https://github.com/superkaiba/explore-persona-space/blob/db4c20ee17e16f541c9d59eab70499e857835353/scripts/issue2203_phase1.py) (axis + band), figures + CJK audit [`scripts/issue2203_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/db4c20ee17e16f541c9d59eab70499e857835353/scripts/issue2203_figures.py), axis-specificity paired-bootstrap [`scripts/issue2203_axspec_refold_h4.py`](https://github.com/superkaiba/explore-persona-space/blob/57aef85d76851d71dcf93362d7bafce55a8547e5/scripts/issue2203_axspec_refold_h4.py). Artifacts: [`eval_results/issue_2203/full-rerun-bugfix/phase2/phase2_ladder_results.json`](https://github.com/superkaiba/explore-persona-space/blob/db4c20ee17e16f541c9d59eab70499e857835353/eval_results/issue_2203/full-rerun-bugfix/phase2/phase2_ladder_results.json), [`phase3_32b_judge.json`](https://github.com/superkaiba/explore-persona-space/blob/db4c20ee17e16f541c9d59eab70499e857835353/eval_results/issue_2203/full-rerun-bugfix/phase3_32b_judge.json), [`phase3_32b_anchor.json`](https://github.com/superkaiba/explore-persona-space/blob/db4c20ee17e16f541c9d59eab70499e857835353/eval_results/issue_2203/full-rerun-bugfix/phase3_32b_anchor.json), [`cjk_intrusion_stats.json`](https://github.com/superkaiba/explore-persona-space/blob/db4c20ee17e16f541c9d59eab70499e857835353/eval_results/issue_2203/full-rerun-bugfix/cjk_intrusion_stats.json), [`phase1_band_tau.json`](https://github.com/superkaiba/explore-persona-space/blob/db4c20ee17e16f541c9d59eab70499e857835353/eval_results/issue_2203/full-rerun-bugfix/phase1_band_tau.json), [`phase0_native_validation.json`](https://github.com/superkaiba/explore-persona-space/blob/db4c20ee17e16f541c9d59eab70499e857835353/eval_results/issue_2203/full-rerun-bugfix/phase0_native_validation.json), [`h4_axis_specificity_refold.json`](https://github.com/superkaiba/explore-persona-space/blob/57aef85d76851d71dcf93362d7bafce55a8547e5/eval_results/issue_2203/full-rerun-bugfix/h4_axis_specificity_refold.json); figures [`figures/issue_2203/full-rerun-bugfix/`](https://github.com/superkaiba/explore-persona-space/tree/db4c20ee17e16f541c9d59eab70499e857835353/figures/issue_2203/full-rerun-bugfix); raw completions + per-item judge scores on the [HF data repo, issue2203_ctx_capping](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/032bdeff91ef7de4f94c2f9e5d0ad2c16ced05ed/issue2203_ctx_capping/raw_completions/full-rerun-bugfix).

**Context:** created 2026-08-08; the parent (buggy) round landed 2026-08-09 and folded a random-direction-control round 2026-08-10; this same-issue follow-up round `full-rerun-bugfix` — three bug fixes (32B cap-vector sign, Qwen-3 thinking-off, 7B unit-norm cap), a full re-run of the 24-arm 7B grid and the 32B anchor, plus the folded-in native-axis extraction arms — landed 2026-08-11 and REPLACES the parent round's output-degradation headline with the corrected finding above. A fresh reproduction of Lu et al. (arXiv 2601.10387), building on #2094 (context-vector localisation) and #1415 (CJK-flip precedent); the jailbreak set is a reconstructed strongreject/wang44 bank, not the paper's Shah et al. set (stated deviation). Originating prompt, verbatim:

> We've found that a lot of persona information is stored at the context vector. One application of controlling personas is preventing the model from straying too far from the assistant persona. The assistant axis [Lu et al. 2026, arXiv 2601.10387] does this by capping the model's activation along the assistant axis. One more efficient way might be to just cap **at the context vector** (or patch?). Reproduce the activation-capping experiment and compare: capping at all tokens (like they did) / only at the context vector / only at the prefix vector; plus patching the default assistant prefix/context vector at subsequent positions (while maintaining query info).
