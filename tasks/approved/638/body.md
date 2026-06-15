---
title: Why do some sources resist being trained with certain behaviors? Predictors
  of per-(source, behavior) install strength
kind: analysis
tags: []
created_at: '2026-06-14T00:15:23Z'
has_clean_result: false
origin_prompt: we also want to understand why some sources resist being trained with
  certain behaviors more
---
## Provenance

Filed from a chat request alongside #637 (the leakage-asymmetry gate ladder). Verbatim originating prompt: "we also want to understand why some sources resist being trained with certain behaviors more."

## Question

Why does install strength vary across (source-context, behavior) cells — some absorb a target behavior readily, others resist (weaker self-implant, lower elicitation yield, more dose to reach a level)? Is resistance a property of the SOURCE (some personas resist everything), the BEHAVIOR (some behaviors hard everywhere), or the specific SOURCE×BEHAVIOR pairing? This is the **install / diagonal** complement to #637 (off-diagonal leakage).

## Landed evidence — Phase 1 (0-GPU, install diagonals from #474 / #537 / #545)

**The premise is largely not supported: resistance is BEHAVIOR-dominated, not a source trait.** Variance decomposition of install strength on #537 (4 rate behaviors × 15 shared source contexts, raw scale): **behavior-main = 89%, source-main = 2%, pairing + noise = 10%**.

- **No persona "resists everything."** Cross-behavior source-rank correlation median ρ = −0.33 (range −0.57 to +0.35): a source that resists one behavior does not resist others. Kills the persona-coherence hypothesis.
- **What resists is whole behaviors:** refusal floors at 0.00–0.40 install across all sources; the marker is dose-gated by band-stop; sycophancy/fact saturate near-ceiling everywhere (14/15 and ~near for fact), compressing any source signal to noise.
- **Base propensity is NOT the diagonal resistance mechanism.** On the install diagonal it is a headroom/ceiling effect (pooled within-behavior base-vs-install ρ ≈ −0.31, *negative*): install = trained − base, so a high base prior means low headroom → smaller delta. The positive "base prior predicts leakage" result (#500/#532/#541) does not carry to install as a predictor. No clean positive within-behavior installability predictor exists in current 0-GPU data; representational distance is undefined on the diagonal.
- **The genuine residual signal is identity / construction conflict, not low prior.** Cleanest datapoint: `casual_register` **negatively installs** (L = −0.42, both seeds in #545) — training a "casual lowercase register" pushed the home format behavior *below* base. The designed-null benign rows + reversed_fact resist more than a ~0 prior predicts. Consistent with hypothesis 2 (alignment/identity conflict) over hypothesis 1 (low base prior) — but these sit in saturated/floored/one-source regimes, so suggestive, not decisive.

Per-behavior install spread on #537 (15 shared source contexts):

| behavior | install range | source sd | ceiling cells | seed reliability |
|---|---|---|---|---|
| marker (nats) | 3.97–10.42 | 1.61 | 0/15 | ρ=0.99 (rock-solid) |
| fact (rate) | 0.30–0.97 | 0.16 | 1/15 | ρ=0.41 (noisy) |
| refusal (rate) | 0.00–0.40 | 0.11 | 0/15 | single seed |
| sycophancy (rate) | 0.84–0.96 | 0.04 | 14/15 | single seed |
| em (rate) | 0.30–0.70 | 0.09 | 0/15 | single seed |

#474 (marker, 16 sources) corroborates that marker install varies by source (19–27 nats) but at loc_ep1 those diagonals are near-saturated, so that variance IS the base-prior spread, not a separate resistance axis. #545 is one-source-per-behavior → no decomposition; it contributes the behavior-level ranking + the casual_register negative-install datapoint.

**Two confounds bound this read:** (1) the 89% behavior-main is partly *because* behaviors sit at very different absolute install levels (floor vs ceiling) — that is itself the answer but inflates the behavior fraction; (2) **dose is not matched across cells anywhere** (#612 dose bands), so resistance is conflated with dose throughout. The marker diagonal is additionally dose-pinned by band-stop.

## What to do next — Phase 2 (GPU; the cut Phase 1 can't make)

Phase 1 establishes the behavior-dominated structure but **cannot separate dose from intrinsic resistance** (every cell is at a different, uncontrolled dose), and the identity-conflict residual lives only in degenerate regimes. Highest-value Phase-2 cut: **matched-recipe dose curves** (install vs optimizer steps) for ~3 resistant vs ~3 non-resistant cells *within one non-saturated behavior* (EM, or off-band marker) — directly testing hypothesis 5 (does more dose overcome resistance or plateau?). Add one alignment-conflict pairing at matched dose (e.g. a harmful behavior into a "kindergarten teacher" source) to separate identity-conflict (hyp 2) from base-prior (hyp 1) — the discrimination the uncontrolled-dose Phase-1 data cannot make.

## Why it matters

Install strength is the denominator for every matched-install leakage read (#601/#627), so an install predictor tightens the whole #526 program. The reframed answer is itself the finding: there is no global "robust persona" — robustness to a harmful implant is behavior- and pairing-specific, and the one engineerable lever Phase 1 surfaces is construction/identity conflict (casual_register-style negative install), not picking a naturally-resistant source.

## Artifacts (Phase 1)

- `figures/issue_638/install_resistance.png` — install vs base prior (marker), within-behavior install vs base propensity (all behaviors), and the source/behavior/pairing variance decomposition.
- `figures/issue_638/install_resistance_results.json` — all numbers.
- `scripts/issue638_install_resistance.py` — reproducible (0-GPU, JSON-only).

## Relations

- Complement to #637 (leakage asymmetry, off-diagonal) and #526 (the predictor rule).
- Evidence / data: #474, #537, #545 (install diagonals); #500/#532/#541 (base prior predicts leakage, not install); #612 (dose bands, on-policy yield by source/behavior); #591 (leak/no-leak structure); #601/#627 (matched-install).
- Open questions: q:leak-predictor (3.1, incl. the failed marker-installability predictor), q:ctx-behavior (3.5).
