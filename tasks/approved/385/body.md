---
title: 'Marker spread dynamics: fine-grained checkpoint sweep of [ZLT] propagation
  from one source persona to 19 bystanders + 8 non-persona contexts'
kind: experiment
tags: []
created_at: '2026-05-24T09:58:30Z'
has_clean_result: false
parent_id: 207
goal: Track how a [ZLT] marker spreads from a librarian source persona to a 19-persona
  + 8-context panel over fine-grained checkpoints on Qwen2.5-7B-Instruct, and test
  whether the bystander emission order tracks L20 cosine-to-source.
---
## Goal

Track how a [ZLT] marker spreads from a librarian source persona to a 19-persona + 8-context panel over fine-grained checkpoints on Qwen2.5-7B-Instruct, and test whether the bystander emission order tracks L20 cosine-to-source AND completion JS-divergence-to-source, computed side-by-side.


## Background

Existing work in this repo measures marker spread at a single endpoint:

- [#207](https://eps.superkaiba.com/tasks/207) consolidates six experiments showing |ρ|=0.48–0.79 between source-bystander geometric distance (operationalized as completion JS-divergence over next-token distributions) and endpoint bystander leakage rate.
- [#341](https://eps.superkaiba.com/tasks/341) showed L20 cosine-similarity and completion JS-divergence align at ρ=0.94 at L20 across the 19-persona panel — the two predictors are tightly coupled at the base model.
- [#228](https://eps.superkaiba.com/tasks/228) ran 7 sources × 10 epochs = 71 checkpoints, computing JS and cosine vs leakage at convergence; the analysis framed leakage as a function of training-epoch as a robustness check, not as a study of spread dynamics.
- [#311](https://eps.superkaiba.com/tasks/311), [#337](https://eps.superkaiba.com/tasks/337), [#365](https://eps.superkaiba.com/tasks/365) are all single-endpoint factor sweeps.

What no existing experiment does: fine-grained checkpoint cadence early in training (≤200 steps) with a fixed 19-persona panel + non-persona-context panel evaluated at every checkpoint, then ask the temporal question — **which bystanders cross the emission threshold first, and does the order track #207's geometric predictors (JS-divergence and the closely-coupled L20 cosine)**.

The 2026-05-22 mentor meeting with Dan Mossing pointed at this directly under Thread C: "see the dynamics of how it spreads to different personas/contexts." The dynamics view distinguishes:
- **Radial propagation:** spread is monotone in geometric-distance-to-source — closest personas emit first, farther personas follow.
- **Cascading propagation:** spread proceeds through semantic or capability clusters — first all "professional" personas, then all "narrative" personas, irrespective of geometric distance.
- **Threshold uniformity:** all bystanders cross the emission threshold within a narrow training-step window — there is no meaningful "order," and the geometric predictor only captures the asymptote.

Each propagation mode implies a different safety-tool architecture (Thread C of the meeting notes), so the dynamics question is mechanism-distinguishing.

## Hypothesis

**Primary (radial):** Bystander personas cross the 5% emission threshold in order of decreasing geometric proximity to the source persona, measured both by (a) L20 cosine-similarity between source and bystander system-prompt residual streams on the base Qwen2.5-7B-Instruct model, and (b) completion JS-divergence between source and bystander next-token distributions averaged over the 30 probes on the base model — the two predictors #207 and #341 establish as ρ=0.94-aligned proxies. Operationally: the Spearman rank correlation between (predictor) and (training step at which bystander first crosses 5% emission, censored at the training horizon) is |ρ| ≥ 0.5 with p < 0.01 for BOTH cosine and JS-divergence (i.e. the radial mechanism is supported by both predictors, not just one). Whether cosine and JS-divergence predict the same bystander order — or diverge mid-training — is itself a sub-result.

**Secondary (non-persona-context):** Non-persona contexts (task framings, instruction directives, format constraints — the #207 trigger families) cross the emission threshold at training steps consistent with the persona radial propagation, treating them as additional points on the same axis. Tests whether persona and non-persona "trigger geometry" share one underlying coverage axis.

**Null:** All bystanders cross the threshold within a narrow window (interquartile range ≤ 0.5× the median crossing-step), in which case the dynamics view collapses to the endpoint view and #207's geometric handle is the whole story.

## Kill criterion

The radial mechanism claim is killed if EITHER of the following holds:

1. **Both predictors fail the rank test:** |ρ_cos| < 0.2 AND |ρ_JS| < 0.2 (Spearman against first-crossing step at the 5% threshold, across the 26 bystanders).
2. **Threshold uniformity:** IQR(first-crossing step across bystanders) > 1.0× median crossing-step — propagation is too uniform for any ordering claim to be meaningful.

If either condition fires, conclude that the geometric handle in #207 is descriptive only (predicts the asymptote, not the propagation mechanism), and re-route the safety-tool framing in Thread C of `docs/mentor_updates/2026-05-22.md` toward whichever cluster axis explains the cascading order (semantic clustering of personas, capability clustering, training-data overlap with source, etc.).

A mixed outcome — exactly one of cosine vs JS-divergence passes the rank test — does NOT kill the radial claim but flags that the two predictors diverge under training, which is itself a paper-quality finding (#207's predictor primacy gets re-litigated).

## Proposed setup

- **Base model:** Qwen2.5-7B-Instruct.
- **Source persona:** `librarian` (recipe-canonical for marker work; matches #262 / #310).
- **Training recipe:** standard Phase A1 LoRA — r=32, α=64, lr=1e-5, 600-row asst_excluded mix, seed 42. Train for the longer-than-usual horizon of 1600 steps to capture late dynamics (#192 found ~625 steps was one full epoch; this extends past saturation to see whether late-stage spread continues or plateaus).
- **Checkpoint cadence:** save adapters at steps {5, 10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600} — denser early when most of the dynamics happens, sparser late. 14 checkpoints total.
- **Eval panel:** the existing 19-persona panel (used in #341 and as #207's broad-panel scope) plus 8 non-persona context cells drawn from #207's trigger families (task framing, instruction directive, format constraint, context scenario — 2 of each).
- **Eval per checkpoint:** 20 canonical PROMPTS × 8 completions × T=1.0, vLLM batched, substring match for the literal `[ZLT]` token. n = 160 per (checkpoint, persona-or-context). Total: 14 × 27 × 160 = 60,480 completions. (Plan v2 reuses #341's canonical 20-PROMPTS set rather than the 30 in this body to match the cached `cosine_matrix.json` protocol; statistical power is equivalent at ±3.4 pp binomial CI.)
- **Predictors (computed on the BASE model, once, before training):**
  - **L20 cosine:** cosine-similarity between the source and each bystander's L20 mean-pooled residual-stream representation of the system prompt. 27 values total (one per bystander).
  - **Completion JS-divergence:** Jensen–Shannon divergence between source and bystander next-token distributions, averaged over the 30 probes on the base model. 27 values total.
  - Both are computed ONCE on Qwen2.5-7B-Instruct (no adapter); they are properties of how the base model represents the system prompts, not of the trained model.
- **Diagnostic (computed per checkpoint, secondary):** L20 cosine and JS-divergence between source and each bystander measured WITH the current checkpoint's adapter applied — exposes whether the geometry shifts during training. Used to test "does the bystander order change as training progresses?"
- **Output:** per-(checkpoint, persona/context) emission rate matrix; per-bystander first-crossing step at 5%, 25%, 50% thresholds; per-bystander crossing-step rank correlated against BOTH base-model L20 cosine-to-source and base-model JS-divergence-to-source (reported side-by-side).

## Primary metrics

- Spearman ρ(L20-cosine-to-source, first-crossing-step at 5% threshold) across 26 bystanders; report at 5%, 25%, 50% thresholds.
- Spearman ρ(JS-divergence-to-source, first-crossing-step at 5% threshold) across 26 bystanders; report at 5%, 25%, 50% thresholds.
- Cosine-vs-JS rank correlation across bystanders at each checkpoint (using the per-checkpoint diagnostic): is the predictor ordering stable across training, or does it drift?
- Interquartile range of first-crossing-step across bystanders, normalized to median crossing-step — distinguishes radial from uniform-threshold propagation.
- Per-checkpoint figure: emission rate of each bystander as a function of training step, with bystanders color-coded by L20 cosine-to-source AND a parallel figure color-coded by JS-divergence-to-source. Radial propagation: color gradient resolves as ordered curves; cascading or uniform propagation: colors mix.

## Pre-conditions

- Confirm the existing eval rig supports saving and re-loading adapters from arbitrary training steps (it does in `src/.../train/sft.py`; verify the 14-checkpoint cadence does not exceed the MooseFS per-pod quota — at ~150MB per adapter, 14 × 150 MB = 2.1 GB, well under quota).
- Confirm the 19-persona panel + 8 non-persona context cells produces non-trivial baseline emission (i.e., the base model emits `[ZLT]` at < 1% across the panel before training).
- Confirm the base-model JS-divergence computation reuses the rig from #207 / #341 without re-implementation (look in `src/explore_persona_space/analysis/` for the existing helper).

## Risks

- **Saturation before resolution.** If most bystanders cross the 5% threshold within the first 100 steps, the cadence at {5, 10, 25, 50, 75, 100} may not have enough resolution to detect rank-ordering. Mitigation: run a 50-step pilot first, confirm at least one bystander is still under 5% at step 50 before launching the full 1600-step run.
- **Threshold artifacts.** At 5%, noise on n=150 is roughly ±2 percentage points (binomial). Some bystanders may oscillate around the threshold. Mitigation: use a sustained-crossing rule — bystander has "crossed 5%" when emission ≥ 5% at the current AND next checkpoint.
- **Spread may not be monotone.** Some bystanders may emit early then regress (training noise). Mitigation: report both first-crossing and last-checkpoint emission; if the two differ for >3 bystanders, the dynamics view requires a more careful operationalization.
- **Cosine vs JS divergence under training.** The radial test uses base-model predictors; if mid-training adapter geometry pulls cosine and JS-divergence apart, the per-checkpoint diagnostic will show this. Not a risk to the kill criterion but worth flagging — divergence between the two predictors at step 100 vs step 1600 is itself diagnostic of which geometry "moves with" the marker.

## Why this experiment

**Application:** distinguishes whether persona-space geometry predicts marker spread because geometry IS the propagation mechanism (radial dynamics) or because geometry happens to correlate with the asymptote (uniform-threshold dynamics); the safety-tool framing in Thread C of `docs/mentor_updates/2026-05-22.md` requires the former to make the N+M activation-collection proposal mechanistic rather than descriptive.

**Decision this changes:** Whether the next applied step is "use #207's geometric predictor as a coverage-gap diagnostic" (radial confirmed) or "the geometric predictor is descriptive but the underlying propagation mechanism is something else, and the safety-tool proposal needs different machinery" (cascading or uniform confirmed); this is the single sharpest mechanism question on Thread C's critical path.

**Expected outcome + branches:** Most-likely outcome is partial radial — bystander order tracks both cosine and JS-divergence at |ρ|≈0.3–0.5, but with one or two outliers whose crossing-step is much earlier or later than their predictor values would suggest; outliers are themselves diagnostic (likely candidates: Qwen-default identity per #123, low-verbosity personas per #357's hypothesis). Clean radial branch (|ρ_cos| ≥ 0.5 AND |ρ_JS| ≥ 0.5, IQR ≤ 0.5× median): #207's geometric predictor is mechanistic, safety-tool pivot proceeds. Clean cascading or uniform branch (both ρ's < 0.2 OR IQR > 1.0× median): geometric handle is descriptive only — re-route safety-tool framing toward whichever cluster axis explains the cascading order. Cosine-vs-JS split (only one of the two passes the rank test): radial mechanism is partial; #207's predictor primacy gets re-litigated.

**What gets cut if we run this:** The open interpretation in #207 and #341 that "geometry predicts spread" is either a propagation claim or a correlation claim; without dynamics data we cannot tell which. This experiment closes that gap and either upgrades #207's confidence toward HIGH (radial confirmed via both predictors) or surfaces the actual propagation mechanism (cascading), or reveals that the two predictors disagree under training (a new finding about #207's predictor primacy); each outcome is mechanism-distinguishing.

Parent: [#207](https://eps.superkaiba.com/tasks/207). Companion: [#228](https://eps.superkaiba.com/tasks/228) (existing convergence sweep). Mentor-meeting origin: `docs/mentor_updates/2026-05-22.md` Thread C.
