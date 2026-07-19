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

Confidence tags are the theory paper's own self-assessments.

| # | Assumption | Predictor factor it supports | Behavior-leakage relevance | Empirical evidence (fill from new paper / issues) |
|---|---|---|---|---|
| **A1** profile-summary | Expression depends on the profile only through a low-dim summary S_θ(C) | foundation for A2 | indirect | (theory says "not worth testing first") |
| **A2** activation-summary | The summary is the mean answer-side activation v_θ(C); expression is a functional of it | foundation for r_Bᵀv | **direct** — needed for r_{B'}ᵀv to mean anything | _pending_ |
| **A3** linear-readout | Each behavior has a linear read-out r_B: E(C,B) ≈ r_Bᵀ v_{θ0}(C) | defines r_B, r_{B'} | **direct** — the r_{B'} in behavior transfer | _pending_ |
| **A4/A5** context-summary / vectorization | A pre-FT context vector c_C predicts v_{θ0}(C) via v ≈ M c_C | context side of the gate | not on the behavior-leakage path (gate=1) | _pending_ |
| **A6** context coherence | Within-condition spread s_W(C) small ⇒ mean-vector predictor valid | validity of c_C | not on behavior-leakage path | _pending_ |
| **A7** readout-stability | Base r_{B'} still reads B' after fine-tuning (r⁺_{B'} ≈ r_{B'}) | lets base r_{B'} read post-FT leakage | **direct** — behavior leakage is r_{B'}ᵀ(v_{θ+}−v_{θ0}) | _pending_ |
| **A8** source-write | FT displaces v toward the data profile: ŵ_{C,B} = v_{θ+}(C)−v_{θ0}(C) ≈ η·δ_{C,B} | defines δ, η | **direct** — δ_{C,B} is the vector r_{B'} reads | _pending_ |
| **A9** rank-one / scalar-gated write | Off-source Δv(C') ≈ w·g(C'); joint grid S_ij = r_{B'j}ᵀΔv(C'_i) ≈ rank-one | the "no interaction" factorization | **direct** — behavior-transfer term is the leakage prediction | _pending_ |
| **A10** bilinear gate | Gate = whitened key–query similarity | context gate form | not on behavior-leakage path | _pending_ |
| **A11** base-gate validity | Base-model gate predicts realized post-FT gate | makes it a *pre-FT* predictor | not on behavior-leakage path | _pending_ |

**Behavior-leakage core chain:** A2 → A3 (r_{B'} exists and reads B' off v) → A8
(δ_{C,B} is where FT pushes v) → A9 (the write is a single direction, so
r_{B'}ᵀδ predicts leakage) → A7 (base r_{B'} still valid post-FT). If all hold, then
`L_{C,B→C,B'} ≈ η·(r_{B'}ᵀδ_{C,B})`, i.e. behavior leakage is predictable pre-FT from
one behavior read-out dotted into the training displacement — no fine-tune of the
target behavior required.

_The "Empirical evidence" column is filled from the new paper's Results sections once
it is clonable, cross-referenced with the per-assumption experiment inventory being
compiled for the "check all assumptions on model-organism LoRAs" analysis._
