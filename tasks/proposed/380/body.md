---
title: Completion-divergence from assistant baseline as a predictor of source implantation
  rate (length-controlled, N=48 panel)
kind: experiment
tags: []
created_at: '2026-05-23T01:12:21Z'
has_clean_result: false
parent_id: 340
---
## Goal

Test whether a source persona's **completion-divergence from the assistant baseline** (JS divergence over next-token distributions on a fixed probe set) predicts how strongly a `[ZLT]` marker implants in that source under a fixed LoRA SFT recipe — using the same N=48 panel that [#340](https://eps.superkaiba.com/tasks/340) and [#368](https://eps.superkaiba.com/tasks/368) used to negate the cosine-from-base predictor.

## Background

[#271](https://eps.superkaiba.com/tasks/271) reported that cosine-from-assistant-centroid at L20 predicts marker source-rate across 12 personas (ρ=−0.74, MODERATE). [#340](https://eps.superkaiba.com/tasks/340) grew the panel to N=48 and found that **the cosine→source-rate signal disappears entirely once log prompt length is partialled out** (ρ collapses to ≈0 within fixed-length bins). [#368](https://eps.superkaiba.com/tasks/368) corroborated on the same panel (ρ=−0.008, p=0.95 with log length partialled). Across these results we have two length-controlled negations of cos-from-base → source rate.

We have a complementary geometric predictor that **has never been tested against the source-rate target**: JS divergence over next-token distributions ("completion divergence"). #142, #228, #207, and #341 used JS-between-source-and-bystander to predict **bystander** leakage rate; six experiments place |ρ|=0.48–0.79 on that bystander target. None of them treats JS-of-source-from-baseline as the predictor of how strongly the source itself implants the marker.

The Thread A mentor framing (2026-05-22 meeting with Dan Mossing) made the gap explicit: "divergence vs base persona of completions effect on source completion." This task fills the empty cell in the predictor × target table.

## Hypothesis

A source persona's JS divergence (from the Qwen-default assistant baseline, over teacher-forced next-token distributions on a held-out neutral-probe prompt set) is monotonically related to its [ZLT] source-rate under the standard LoRA Phase-A1 recipe, and the relationship survives partialling out log prompt length. Direction (positive or negative) is not pre-registered — we don't yet have an intuition for which sign to expect.

Auxiliary: pairwise JS divergence between persona prompts may be a stronger predictor than JS-from-baseline alone.

## Proposed setup

- **Base model:** Qwen2.5-7B-Instruct (matches #340 / #368 panel).
- **Panel:** the same N=48 persona panel that #340 and #368 used. Re-use existing centroids / training artefacts where available.
- **Predictor 1 (primary):** JS divergence between (each source persona, fixed neutral probe questions) and (Qwen-default assistant, same probes), computed teacher-forced over next-token distributions on the base model, averaged across token positions per prompt, then averaged across probes. Use the existing `src/explore_persona_space/analysis/divergence.py` implementation (introduced in [#140](https://eps.superkaiba.com/tasks/140)).
- **Predictor 2 (secondary):** for each persona pair (source, j), pairwise JS divergence over the same probe set. Builds a 48×48 matrix; reduce per-source to mean / median / max divergence-to-other-personas.
- **Target:** [ZLT] source rate under a single fixed LoRA recipe (LoRA r=32, α=64, lr=1e-5, 3 epochs, 600-row asst_excluded mix, seed 42 — the recipe #271 / #340 / #368 used). Re-use existing source-rate measurements from the #340 / #368 panel if recipe-matched; otherwise re-train.
- **Controls:** partial Spearman of source-rate vs predictor, partialling out log(prompt-token-count). Report both the raw Spearman and the length-partialled Spearman.

## Primary metrics

- Spearman ρ(JS-from-assistant, source-rate) on N=48, raw and partial-out-log-length.
- Spearman ρ(mean pairwise JS, source-rate) on N=48, raw and partial.
- Pre-registered pass: |partial ρ| ≥ 0.5 with p < 0.01 on at least one of the two predictors → "JS predicts source implantation strength even with length controlled."
- Pre-registered fail: |partial ρ| < 0.2 on both predictors → "completion divergence is also not a source-rate predictor; vulnerability is dominated by something other than these two geometric families."

## Pre-conditions / pre-flight

- Confirm the N=48 panel's source-rate JSON is still on the HF data repo or in `eval_results/`; if missing, re-train (3-day estimate for 48 LoRAs on 1×H100, sequential).
- Confirm `divergence.py` runs on Qwen2.5-7B-Instruct at the panel scale (memory budget for 48 personas × ~30 probes is light).
- Calibration: compute JS-from-assistant for a 3-persona subset and confirm matrix non-degenerate (variance across personas substantially above numerical-noise floor).

## Risks / open questions

- The N=48 panel may carry confounds beyond prompt length (e.g., persona-prompt token-distribution skew); flag if JS itself correlates with length at |ρ|>0.6, which would echo the #340 collapse.
- If JS-from-assistant and cosine-from-assistant rank-correlate at ρ>0.9 (as #341 found at L20 on the 19-persona panel), this test may be near-redundant with #340 / #368 and we get a third null. Pre-register that outcome as "geometric distance from the assistant is not the right axis; vulnerability differences are driven by something else (capability, prompt-format prior, semantic content)."
- The "completion-divergence-from-base" measure could be defined either over (base-model under persona prompt) vs (base-model under assistant prompt) or (LoRA-trained model under persona prompt) vs (base-model under same persona prompt). The mentor framing maps cleanest to the first — base-model behavioral divergence as a stand-in for "how non-assistant is this persona." Use that.

## Why this experiment

**Application:** if completion-divergence predicts source-rate, we have an output-space handle on vulnerability that's empirically separable from the prompt-length confound that killed cos-from-base in #340 and #368; if it also fails, we have HIGH-confidence evidence that vulnerability differences between personas are not driven by any geometric distance from the assistant centroid — closes off a whole family of predictors and re-routes the question toward content-specific predictors (capability, factual-prior, prompt-format).

**Decision this changes:** Whether persona-space geometry is a viable predictor of *which* personas are most marker-implantable, or whether the project should pivot the "what predicts vulnerability" question to non-geometric predictors; this directly bounds the framing of the safety-tool proposal in Thread C of `docs/mentor_updates/2026-05-22.md`.

**Expected outcome + branches:** Most-likely outcome is partial-ρ similar in magnitude to #271's original cosine result (|ρ|≈0.5) but disappearing under length control, which would say JS and cosine measure the same length-confounded axis (consistent with #341's ρ=0.94 between the two pairwise matrices). Clean positive branch (|partial ρ| ≥ 0.5 on at least one of JS-from-assistant or mean pairwise JS): geometric handle on source-rate survives — open path for divergence-based vulnerability prediction and a benchmark predictor for the safety-tool proposal. Clean negative branch (|partial ρ| < 0.2 on both predictors): closes off geometric-from-assistant predictors entirely; re-routes to non-geometric predictors (e.g., training-data overlap, base-rate token frequencies, capability-axis location).

**What gets cut if we run this:** The open interpretation in #340 and #368 that "cosine fails but maybe a different geometric metric works" — this task either rescues the geometric-predictor program with a divergence-based win or pins down that it does not, eliminating residual hope that the geometric vulnerability story can be patched by swapping cosine for JS.

Parent: [#340](https://eps.superkaiba.com/tasks/340). Companion: [#368](https://eps.superkaiba.com/tasks/368). Predictor family origin: [#140](https://eps.superkaiba.com/tasks/140) / [#207](https://eps.superkaiba.com/tasks/207). Mentor-meeting origin: `docs/mentor_updates/2026-05-22.md` Thread A.
