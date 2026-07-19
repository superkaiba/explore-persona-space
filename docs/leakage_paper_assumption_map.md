# Leakage paper ↔ theory-assumption map

Two papers:

- **Theory paper** (pure theory, no own numbers): *Predicting fine-tuning–induced
  leakage from pre–fine-tuning context geometry*. Overleaf project
  `6a2df2d2053483dc444ed4f0`, clone `~/overleaf-6a2df2d2/main.tex`. Holds the
  assumptions A1–A11 and the predictor.
- **New empirical paper** (the one we're now working on): Overleaf **edit** share
  link `https://www.overleaf.com/2812467154ppzsqkdyqmgz#9b3acd`. Empirical companion
  that tests the assumptions on the project's trained **model-organism LoRAs**, with
  emphasis on **behavior leakage** (generalization from one behavior to another).
  Not yet clonable — resolving the share token to a project ID needs Thomas's
  Overleaf login (edit link, anon grant denied). Once the 24-hex id is known, clone
  with the `OVERLEAF_GIT_TOKEN` (see memory `reference_new_leakage_empirical_paper`),
  then fill in the per-assumption "empirical evidence" column below from the actual
  paper sections.

## The predictor

    L̂_{C,B→C',B'} = η_{C,B} · (r_{B'}ᵀ δ_{C,B}) · g_C(C')
                     └ strength ┘ └ behavior transfer ┘ └ context gate ┘
    δ_{C,B} = t_{C,B} − v_{θ0}(C)      (t = teacher-forced training-completion activations)
    g_C(C') = (c_Cᵀ Σ_c⁻¹ c_{C'}) / (c_Cᵀ Σ_c⁻¹ c_C)

The three query rows the paper cares about:

| query | strength | behavior transfer | context gate |
|---|---|---|---|
| on-source `C,B→C,B` | η | rᵦᵀδ | 1 |
| **behavior leakage** `C,B→C,B'` | η | **r_{B'}ᵀδ** | 1 |
| context leakage `C,B→C',B` | η | rᵦᵀδ | g_C(C') |

The user's ask — *"leakage and generalization from one behavior to another"* — is the
**behavior-leakage row**: the `r_{B'}ᵀδ_{C,B}` behavior-transfer factor, with the
context gate pinned to 1 (same source context). Emergent misalignment (train insecure
code B, read broad-misalignment B') is the canonical instance.

## Assumptions (paper labels) and how the empirical paper links to them

Confidence tags in the theory paper are its own self-assessments. The "Verdict"
column below is the EPS empirical result, read from the cited task clean-results
(headline numbers spot-checked against #667 Takeaways, #545 title/body, #637 title,
2026-07-19). Canonical assumption→task map: `docs/theory_assumption_test_plan.md`.

| # | Assumption | Behavior-leakage relevance | Verdict on model-organism LoRAs | Evidence (task, headline) |
|---|---|---|---|---|
| **A1** profile-summary | indirect foundation | not directly tested | theory itself says "not worth testing first" |
| **A2** activation-summary v_θ(C) | **direct** (base-model) | **SUPPORTED after measurement fixes** (base-model only) | #658 refuted 7/10 as first measured; #761/#763 recover all 10 (LOCO ρ 0.51–0.90) once probes matched + estimator/DV fixed |
| **A3** linear read-out r_B | **direct** (r_{B'}) | **MOSTLY REFUTED** with a faithful r_B | #658 rd.4: faithful persona-vectors r_B passes only 1/8 (behavior×genre) cells; earlier fit was corpus-mismatch artifact |
| **A4/A5** context vectorization v≈Mc_C | context gate (gate=1 here) | **SUPPORTED at base**; FT-stability only for taught fact | #722 R²=0.74–0.80 (base map strong, low-dim ok); #1092 query-dominated not persona-prefix (R² 0.74–0.81 vs 0.05–0.11); #823/#952 content-indexed not policy-specific; #722 FT-stability clears floor only for taught fact |
| **A6** context coherence s_W(C) | context side | **UNTESTED** | planned in `docs/theory_assumption_test_plan.md` §R5-1; no executed clean-result |
| **A7** read-out stability r⁺≈r | **direct** — leakage is r_{B'}ᵀ(v⁺−v₀) | **REFUTED on trained LoRAs** | #667: base r_B's projection of the trained update anti-correlates with measured change (partial ρ = −0.35 EM / −0.41 fact / −0.03 syc); re-extracting r⁺ on the FT model does not rescue (−0.28/−0.01/−0.55) — geometry mismatch, not a rotated instrument |
| **A8** source write ŵ≈η·δ | **direct** — δ is what r_{B'} reads | **REFUTED / null on trained LoRAs** | #667: cos(ŵ, δ) ≈ 0 (EM +0.07 / syc −0.19 / fact +0.02), scalar-fit residual ~1.0; contrastive negatives don't rotate the write toward the data target |
| **A9** rank-one / scalar-gated write | **direct** — the factorization | **SUPPORTED in activation space; behavioral transfer FAILS for content behaviors** | #667: single direction holds 0.81–0.86 of cross-context update variance (chance 0.034), cos 0.85–0.93 to the write; but #637: behavioral-matrix rank-1 asymmetry generalizes out-of-sample only for marker + taught fact, NOT refusal/sycophancy/EM |
| **A10** bilinear whitened gate | context gate | **SUPPORTED in activation space** | #667: base whitened gate → realized activation gate ρ=0.46/0.59/0.56, above shuffled-key null |
| **A11** base-gate validity | makes it a *pre-FT* predictor | **SUPPORTED for the activation gate; behavioral payoff weak** | #667: base gate matches/beats post-FT oracle (0.46/0.59/0.56 vs 0.27/0.48/0.46) despite context drift 0.54–0.77 — BUT reaches measured behavioral leakage only ρ=0.13 EM / 0.16 syc / 0.40 fact |

**Behavior-leakage core chain** (A2 → A3 → A8 → A9 → A7): if all hold,
`L_{C,B→C,B'} ≈ η·(r_{B'}ᵀδ_{C,B})` — behavior leakage predictable pre-FT from one
read-out dotted into the training displacement, no target-behavior fine-tune needed.

**Verdict on the chain:** it BREAKS at A3, A7, and A8. A2 (activation summary) holds
at base after measurement fixes, and A9/A10/A11 hold *in activation space*, but the
two links the behavior-transfer term most needs — A8 (the write points at δ) and A7
(the base read-out still lands the change) — are directly refuted on the trained
LoRAs, and A3's faithful read-out mostly fails. End-to-end, geometry predictors do
NOT transfer to behavior→behavior leakage: #545's 589-predictor race finds no
pre-training predictor beats the 0.10 non-noise floor on held-out cells (champion a
raw centroid cosine at ρ=+0.089). The mechanistic ingredients are real in activation
space but explain only a modest, behavior-dependent slice of actual behavioral
leakage — best for the taught fact / marker, worst for **emergent misalignment and
sycophancy**, the behaviors the sibling papers (Persona Vectors; Persona Features
Control EM) care about most.

## New-organism consistency check (2026-07-19)

Re-checked the assumptions on the NEWEST organism suite (single-behavior implants,
matched-install, with a LoRA-vs-full-FT geometry arm): harmful-compliance #1074,
sycophancy/impolite/formatting #1090, matched-install geometry #1112, impolite #1315,
marker #1333, casual-style #1434, contrastive-containment #1481, context-vector
causal steering #1415. Verdict: **consistent — no assumption flips REFUTED↔CONFIRMED**;
the new suite sharpens the old story into a behavior-locality gradient and adds a
shared-text/teacher-forced control that explains the one apparent divergence.

- **A7 (read-out stability) — AGREES (replicated).** Sycophancy: mean-shift↔r_B cosine
  −0.05 to +0.20 own-text, collapses into the chance band under shared text (#1112);
  the fresh persona-vector projection anti-correlates with judged install (context-arm
  rank ρ=−0.29, n=12, cluster interval below zero, #1090) — same sign as the old
  −0.35/−0.41.
- **A8 (source write ‖δ/‖r_B) — AGREES, with one apparent-divergence resolved.** No new
  task recomputes cos(ŵ,δ); all use the ŵ‖r_B shortcut. Marker |cos|≈0.002–0.051
  (#1333), sycophancy ≤0.20 (#1112) — both agree with the old ≈0. **Impolite looks
  divergent** (own-text alignment 0.51–0.80, #1315) **but the shared-text re-capture
  drops it to 0.07–0.33** — most of the alignment rides the model's own generated
  text, not a stable write direction. Method (LoRA vs full-FT) and contrastive
  negatives rotate the write essentially not at all.
- **A9 (rank-one / clean signature) — AGREES on prefix spans; NEW graded finding on the
  response span.** Prefix/context-span rank 1–13 across sycophancy/impolite/marker.
  On the response span, own-text is diffuse everywhere, but the shared-text control
  reveals a locality gradient: marker → rank 9 (≈rank-one), impolite → 18–29,
  sycophancy → 27–35. Almost all the "diffuseness" the old line attributed to the write
  is a property of measuring on own generated text.
- **A10/A11 (gate) — STILL UNTESTED by this suite** (no new gate-predictor regression);
  context strongly gates transfer qualitatively but nobody predicts the gate from base
  geometry here.
- **Behavioral translation — AGREES (behavior-dependent), with locality inverted from the
  naive prior.** #1481: contrastive containment that survives dose-matching + generalizes
  past the trained negatives holds only for **impolite** (+0.087 held-out); marker's
  gap mostly dissolves under install-normalization; sycophancy's is not cluster-robust.
  #1415 (causal steering) is the strongest NEW positive: geometry and behavior both peak
  at layer 14 (+6.2 judged pts, 21% of the context-swap ceiling, 7× the layer-20 read,
  2 seeds) — geometry→behavior IS real at the right layer — **yet the fitted linear
  context→answer map predicts none of the realized shift at layer 20 (cosine 0.00,
  16× magnitude over-prediction)**, so the correlational map used to operationalize
  A10/A11-style predictability fails a causal test.

**Cross-cutting caveat:** the entire new suite is single-seed, and several judged pools
carry CJK/Chinese-token intrusion that makes some in-band PASS labels
convention-dependent. Locality of the behavior (lexical marker → stylistic impolite →
propositional sycophancy) modulates every geometric read-out — that heterogeneity is
the main *new* result, not a fleet-wide inconsistency.

_The paper's own section-by-section evidence is folded in once it is clonable
(needs the project ID)._
