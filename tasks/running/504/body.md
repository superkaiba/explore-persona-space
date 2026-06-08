---
title: 'Bubble vs barrier: does a contrastive negative protect a local neighborhood
  around itself or the whole region behind it in persona space?'
kind: experiment
tags: []
created_at: '2026-06-06T00:48:32Z'
has_clean_result: false
goal: 'Determine the geometric shape of a single contrastive negative''s leakage-suppression
  in persona space: does a negative protect a local neighborhood around itself (bubble)
  or the entire region behind it relative to the source (barrier/shadow)? Secondary:
  do near-twin negatives produce a more localized implant than distant ones, holding
  all else fixed.'
relates_to:
- leak-contrastive-negatives
- leak-predictor
---
## Goal

Determine the geometric shape of a single contrastive negative's leakage-suppression in persona space: does a negative protect a local neighborhood around itself (bubble) or the entire region behind it relative to the source (barrier/shadow)? Secondary: do near-twin negatives produce a more localized implant than distant ones, holding all else fixed.

- **(Bubble)** protect a local neighborhood around *itself* — suppression is a function of distance(probe, negative), independent of where the source sits — or
- **(Barrier / shadow)** protect the entire region *behind* it relative to the source — suppression is a function of whether the negative lies between the source and the probe?

Secondary: does using **near-twin negatives** (negatives cosine-close to the source) produce a more localized implant (tighter leakage boundary) than distant negatives, holding everything else fixed?

This is the never-run negative-set-composition sweep (#19) reframed around an explicit geometric mechanism, and it directly attacks open-q 3.4a (q:leak-contrastive-negatives).

## Hypothesis

Leakage of an implanted marker decays with persona-space distance from the source (established: #207, |rho| 0.48-0.79). A contrastive negative suppresses leakage locally. Two mechanistic models predict different *shapes* of that suppression:

1. **Bubble:** a negative at position N protects a sphere around N. Discriminating prediction: a probe close to N is protected wherever it sits; a probe far from N leaks even if N is between it and the source.
2. **Barrier:** a negative at N occludes the source→outward leakage along N's direction. Discriminating prediction: any probe in N's angular shadow (N between source and probe) is protected, even moderately far from N; an off-axis probe at the same distance-from-source is not.

The crux comparison: two held-out probes **matched on cosine-to-source**, one with N between it and the source, one lateral. Bubble → equal leakage; Barrier → shadowed probe protected, lateral not.

## Why this matters / the gap

- Negative-set *composition* (count + similarity-to-source/target) has never been swept as a single clean variable. Every prior experiment changed the negative set alongside other things (#383 changed ratio + panel + positives at once; cross-experiment counts ranged 2-23 confounded with model/hyperparameter/DV differences).
- The load-bearing field claim "near-twin negatives are the sharpest lever for localization" currently has zero direct evidence.
- If the **barrier** model holds, you can protect a whole region of persona space (e.g. all personas "behind" the default assistant) with a small, well-placed negative set — a concrete EM-defense lever. If **bubble** holds, you must place a negative near every persona you want to protect.

## Proposed design (sketch — to be formalized by /adversarial-planner)

- **Implant:** ※ marker (token id 83399) into a single fixed source persona via contrastive LoRA SFT. On-policy positives (greedy frozen response + marker), `MarkerOnlyDataCollator`. Default assistant always in the negative set (safety target, per contrastive-negatives rule).
- **Anchor recipe — ground in past issues, do NOT guess.** The lr / LoRA rank+alpha / target modules / training steps / positive count / pos:neg ratio that land source on-policy `log P(※)` in the non-saturating middle band must be lifted from the prior marker-implant line and cited `Source: #N`. The recent line bracketed the window from both sides: #479 stuck at the **dead floor** (0 emission across 250 steps under gentle knobs), #448 stuck at the **ceiling** (fully-trained anchor, logp saturated, argmax=marker everywhere); #492 traced part of the floor to an eval-side artifact; #383 lifted source rate ~70x off the floor; #477 worked a recovered grid. The planner must identify the specific recipe + step count where logp sat mid-band (off floor AND ~5-10 nats below ceiling) and adopt it, or state explicitly that no past run hit the middle and a calibration smoke-sweep is needed first.
- **Manipulated variable:** the position of an *additional* positioned negative N relative to the source — swept across arms by selecting real personas at controlled cosine-to-source (e.g. near-twin / mid / distant). Everything else (model, source, questions, positive count, lr, rank, steps, seed, eval grid) held fixed. Single-variable.
- **Probe grid:** a dense held-out persona bank with measured persona vectors (layer-20 Qwen2.5-7B-Instruct, the #207 Proximity-Transfer rig), classified per arm by (radial: closer/farther than N from source) x (angular: aligned-with-N vs orthogonal). Probes are NEVER trained.
- **DV (committed): on-policy `log P(※)`** at the end of the probe persona's own response, reported trained - base (subsumes emission rate). This is the metric for both the implant (source) and leakage (every probe). Log the log-prob trajectory over training steps per persona. Because we are committing to logp, the anchor MUST stay off the ceiling (see Confound 3) — a saturated logp is information-free, so the non-saturating anchor is a hard design requirement here, not an option. (Full-vocab KL-from-base at the post-response slot kept only as a fallback diagnostic IF an arm unexpectedly saturates despite the gentle anchor.)
- **Read-out:** leakage(probe) as a function of probe-cosine-to-source, per negative-placement arm. Bubble = a local notch at d_probe ~ d_N that slides with N; Barrier = a shelf where all probes with the negative between them and the source are suppressed.

### Grounded anchor recipe (from a #477-line mining pass, 2026-06-05)

The non-saturating mid-band was hit cleanly only at **low LoRA rank + low negative-count + lr 2e-6** in #477's recovered grid. r=32 is a ceiling trap (#448/#492/#477-control land source ΔG 17-22 nats / emission 1.0 even at lr 2e-6); lr alone is NOT the lever (#479 swept lr 5e-6→3e-5 at r=16 attn-only and stayed at on-policy emission exactly 0). The lever that walks source through the mid-band is the **rank+count bundle at lr 2e-6**.

Recommended anchor (each load-bearing value tagged):
- Base `Qwen/Qwen2.5-7B-Instruct`; source persona `villain` (canonical across this line AND the #207 geometry rig) — `Source: #477, #472, #479, #207`
- Marker ` ※` id 83399, assert before spawn — `Source: marker-leakage-measurement.md`
- **LoRA rank = 8, all-linear** (r8/count-2 = source ΔG **9.3 nats**, the cleanest single mid-band point; r=4/count-2 = 3.2 nats is the cooler fallback) — `Source: #477`. NOT r=32.
- **lr = 2e-6** — `Source: #477` (≥5e-6 pushes toward ceiling / R-collapse, #477 LR-lever + #448)
- α = 16 (2×rank convention, not separately swept → `needs-smoke-test`) — `Source: #477` (inherited)
- **Negative count = 2** held FIXED (bare `qwen_default` + 1 positioned bystander); the manipulated variable is that bystander's *position*, NOT the count — `Source: #477` (count co-varies with rows/steps there, so fix it)
- positives 200 / negatives 200 (1:1), ~63 steps (1 epoch / 400 rows) — `Source: #477, #383, #479`
- marker-position-only loss, `MarkerOnlyDataCollator(tail_tokens=0)`, on-policy frozen-base positives — `Source: contrastive-negatives.md`
- seeds 42 + 137; read source ΔG as a **trajectory** over checkpoints {5,10,20,35,50,63} and pick the one at ~5-10 nats (#479: lift is flat after step 5, so catch it before it climbs); include the `assert_adapter_actually_applied` guard so a silent LoRA-not-applied read can't masquerade as a floor — `Source: #477, #479`

**Calibration first (no past run *deliberately* held the mid-band):** before the main geometry grid, run a 3-cell smoke-sweep at lr 2e-6 / count 2 / all-linear / 1 epoch — r=4 (~3 nats), r=8 (~9 nats, predicted best), r=16 (brackets the 9→17 climb toward ceiling). Pick the cell with source ΔG ~5-12 nats and on-policy emission clearly off 0 but below ~0.8. If all three read sub-emission (emission 0, like #479), nudge to 2 epochs at the chosen rank rather than raising rank into the ceiling. ~0.5 GPU-h on 1× H100.

**Window is brittle:** the same villain+※ recipe sat at emission exactly 0 (#479, #472) or exactly 1.0 / ΔG ~20 (#448, #492, #477-r32) with almost nothing between except #477's low-rank/low-count cells, where ΔG jumped 1.1→9.3→22.5 over rank 2→8 and 9.3→22.5 over count 2→16. Calibrate empirically; do not trust any single inherited cell.

## Confounds to control (all from recent results)

1. **Distance-from-source** (#207): shadow-vs-lateral probes MUST be matched on cosine-to-source.
2. **Probe base prior** (#500): a bystander's own prior on the behavior predicts leakage; report trained - base per probe, and/or match probes on base-model emission.
3. **Saturation** (#448, #479) — load-bearing because the DV is logp: the anchor MUST be tuned so source on-policy `log P(※)` sits ~5-10 nats below ceiling (fewer steps / smaller LoRA rank / lower lr). Verify the source isn't saturated in a smoke run BEFORE measuring probes — at a fully-trained anchor logp flatlines and no geometry is visible by construction. The non-saturating anchor is the single most important calibration step.
4. **Anchoring on a single (source, negative) configuration:** replicate over >=2 source personas and multiple seeds before any geometric claim.

## Assumptions (flag if wrong)

- Marker (※) is the right first proxy — cleanest, most-developed rig, all geometry infra is marker-based. Could later repeat for a fact/trait. Stated as assumption; redirect if a fact implant is preferred.
- The discrete persona bank actually contains personas at the needed shadow / lateral positions; if not, the planner may need to expand the bank or accept approximate geometry.

## References

- #207 (distance predicts leakage; layer-20 persona-vector + cosine-selection rig)
- #500 (bystander's own prior is the surviving leakage predictor)
- #448, #479 (on-policy saturation kills recipe-knob sweeps)
- #383 (selectivity recipe; possible X-vs-(X-Y) artifact)
- #19 (the never-run composition sweep this supersedes)
- open_questions.md 3.4a (q:leak-contrastive-negatives), 3.1 (q:leak-distance)
- .claude/rules/contrastive-negatives.md, .claude/rules/marker-leakage-measurement.md
