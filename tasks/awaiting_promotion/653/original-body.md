---
title: Do conditional behaviors decompose cleanly into read and write features? Characterize
  the autoregressive write→read map
kind: experiment
tags: []
created_at: '2026-06-16T07:47:41Z'
has_clean_result: false
origin_prompt: Create an issue to check if conditional behaviors decompose cleanly
  into read and write features. (random-bias write features as steering -> sample
  -> read unsteered activations on those tokens vs baseline; characterize write->read
  geometry; consider theory document + rank-1 LoRA discussion)
goal: Test whether conditional behaviors decompose cleanly into separable read (condition-detection)
  and write (behavior-production) features, via two probes — (A) the base model's
  autoregressive write→read map under random-bias steering, and (B) how real finetunes
  shift activations across the rank ladder (rank-1 LoRA → higher-rank LoRA → full
  fine-tuning) — run across ≥3 installed behaviors and a few source contexts (not
  a single behavior/context) so the verdict generalizes, characterizing in each whether
  the structure is low-rank and read↔write-aligned (clean) versus rotated or diffuse,
  and whether that verdict is consistent across behavior, context, and edit rank.
relates_to:
- identity-contextual-vs-base
- identity-cb-duality
---
## Goal

Test whether conditional behaviors decompose cleanly into separable read (condition-detection) and write (behavior-production) features, via two probes — (A) the base model's autoregressive write→read map under random-bias steering, and (B) how real finetunes shift activations across the rank ladder (rank-1 LoRA → higher-rank LoRA → full fine-tuning) — run across ≥3 installed behaviors and a few source contexts (not a single behavior/context) so the verdict generalizes, characterizing in each whether the structure is low-rank and read↔write-aligned (clean) versus rotated or diffuse, and whether that verdict is consistent across behavior, context, and edit rank.


## Scope directive (user, 2026-06-16)

Breadth for generality — the decomposition verdict must NOT rest on a single
behavior or a single context:

- **Behaviors: ≥3.** Run the characterization across at least three installed
  behaviors (the marker, plus two others — e.g. a persona and a sycophancy/EM
  behavior), reusing existing matched-recipe adapters from the
  #519/#521/#532/#606 lines wherever they fit (artifact-reuse rule).
- **Source contexts: a few (≥2–3).** Install/elicit each behavior in more than
  one source context, so the read-side gate `g(C)` and the per-context shift
  profile are measured across a small context panel rather than a single source.
- **Single-variable discipline still holds *within* a cell.** Rank remains the
  only varied factor inside each (behavior × source-context) cell of Arm B;
  behavior and source-context are the **breadth axes** across which the
  rank-ladder characterization is repeated for generality. Arm A's behavior
  probe `d_B` sweeps the full set of behaviors/directions, not one.
- **Verdict must be reported per-cell AND aggregated** — H1/H2/H3 stated for
  each (behavior, context) and whether the ranking is consistent across them
  (the generality claim) or behavior/context-dependent.
- **Cost note.** The full-FT rung × (behaviors × contexts) is the cost driver and
  will likely exceed the 100 GPU-h auto-approve cap — the planner should reuse
  adapters aggressively and may stage the full-FT rung; the session is expected
  to park at `plan_pending` for approval if the estimate is over cap.

## Why this question

A conditional behavior — "in context `C`, emit behavior `B`" — is exactly what a
rank-1 LoRA stores: `ΔW ≈ w · x_C^T`, an outer product that **reads** along the
context direction `x_C` and **writes** the direction `w` into the residual stream
(`docs/notes/rank1_leakage_model.tex`). The project's theory document keeps two
behavior-side objects deliberately separate: the **read-out** `r_B` (project to
measure B) and the **steering/write** `w` / `d_B` (add to elicit B); its central
anomaly is that the two need not align — write cos ≈ −0.03 to the realized marker
shift (#521), write cos ≈ 0.04 to the EM direction yet 98% ablatable (Soligo
2025). So "does a conditional behavior decompose **cleanly** into a read part and
a write part?" is an open, load-bearing question, not a settled assumption.

## Two probes of the same decomposition

We attack it from two sides, both asking whether the read/write structure is
**low-rank and read↔write-aligned** (clean) versus **rotated or diffuse** (not):

- **Arm A — the generation-loop write→read map (training-free).** In an
  autoregressive model a feature written into the residual stream biases the
  sampled token; on the next forward pass that token is read back in and
  re-encoded. The composition write → token → read is the round trip through the
  token bottleneck. Probe it with *random* writes — an unbiased, model-agnostic
  stand-in for "write features" — and read what the unsteered model infers from
  the sampled tokens.
- **Arm B — how real fine-tuning shifts activations, across the rank ladder.**
  The rank-1 picture (#519/#521/#538) is the *r=1* special case. Arm B asks
  whether the same read/write decomposition survives as the edit gets richer:
  rank-1 LoRA → higher-rank LoRA → full fine-tuning. Instead of inspecting the
  weights, measure how the finetune **moves the activations** (base-vs-finetuned
  residual-stream differences) and characterize that shift's geometry. This is
  the theory's Assumption-5 relaxation ladder — single rank-one key→value pair →
  low-rank multi-pair → rich regime that builds new features — made empirical.

## Arm A — formalization

Fix a model, a write layer `ℓ`, a read layer `ℓ′`, and a prompt distribution `𝒬`.

- **Write.** During generation add a perturbation `w ∈ ℝ^d` to the residual
  stream at layer `ℓ` (additive steering at every generated position). Sample a
  continuation `A_w = LLM_{+w}(Q)`.
- **Read.** Run the **unsteered** model on `A_w` and pool its layer-`ℓ′`
  activations: `read(A_w) ∈ ℝ^d`.
- **Induced read shift** (the object of study):

  `ρ(w) = E_{Q,A_w}[ read(A_w) ] − E_{Q,A_0}[ read(A_0) ]`

  where `A_0` is an unsteered sample (the baseline). `ρ : ℝ^d → ℝ^d` maps
  write-space to read-space *through the token bottleneck* — random writes are an
  unbiased probe of it, requiring no training.

**Geometry to characterize.**
- **Linearity.** Does `ρ(w) ≈ J w` for a Jacobian `J` over the operating range?
  Fit `J` by ridge regression on `(w_i, ρ(w_i))` pairs; report variance explained
  and the nonlinearity residual.
- **Rank / spectrum.** SVD of `J` (or of the stacked read-shift cloud
  `{ρ(w_i)}`). Low effective rank ⇒ writes and reads share a low-dimensional
  feature subspace.
- **Alignment.** Per-write round-trip cosine `cos(w, ρ(w))`. Near 1 ⇒ write
  features re-read as themselves (near-identity loop); a stable non-trivial
  rotation ⇒ read basis ≠ write basis; ≈ 0 / diffuse ⇒ no clean correspondence.
- **Behavior probe (swept across the ≥3 behaviors per the scope directive).**
  For each *known* behavior steering direction `d_B` (markers, personas, EM —
  already trained in the #519/#521 line), does `ρ(d_B)` recover that behavior's
  read-out `r_B`? Tests the theory's `r_B`-vs-`d_B` relationship through the
  generation loop, and whether the recovery is consistent across behaviors.

## Arm B — formalization (non-rank-1 LoRA and full fine-tuning)

For EACH (behavior `B`, source context `C`) cell in the breadth panel (≥3
behaviors × a few source contexts per the scope directive), install the SAME
conditional behavior `B` in source context `C` at increasing edit rank, holding
training data / recipe / dose fixed so **rank is the only varied factor within
the cell**: rank-1 LoRA, rank-`r` LoRA (`r ∈ {4, 16, 64}`), full fine-tuning.
(Behavior-implant rows use contrastive negatives per
`.claude/rules/contrastive-negatives.md`, identical across the ladder.)

- **Activation shift.** Over a panel of contexts `{C}` (source + bystanders) and
  queries `Q`, measure `Δx(C,Q) = x_FT(C,Q) − x_base(C,Q)` at layer `ℓ`, pooled
  on-policy. This is the model-organism analogue of the "readable traces in
  activation differences" method (arXiv 2510.13900).
- **Write side.** PCA/SVD of the `{Δx}` cloud → effective rank + top-direction
  variance share (the #521 method, extended past rank-1). Does the dominant shift
  direction align with the behavior read-out `r_B`, and does it match Arm A's
  `ρ`-map leading directions?
- **Read side.** Does per-context shift magnitude track base-model context
  similarity to the source (the theory's gate `g(C)`)? If so the "which contexts"
  factor is base-computable, independent of rank.
- **Cross-rank prediction.** Weight-space work says LoRA leans on a few singular
  vectors and grows "intruder dimensions" absent from base, while full FT spreads
  importance evenly and stays spectrally close to base (arXiv 2410.21228); FT
  tends to *enhance existing mechanisms* rather than build new ones (arXiv
  2402.14811); FT updates have low intrinsic dimension (Aghajanyan et al.,
  2012.13255). The clean test: does the read/write decomposition stay low-rank
  and aligned as edit rank grows (more key→value pairs, still structured), or
  degrade into a diffuse high-rank shift (the rich regime) — and is THAT verdict
  the same across the behavior/context breadth panel?

## Competing hypotheses (apply to both arms)

- **H1 — clean decomposition.** Structure is approximately linear, low-rank, and
  alignment-preserving; behavior writes re-read as their own read-outs.
  Conditional behaviors factor into separable read (condition) and write
  (behavior) features over a shared basis — and (Arm B) this holds across the rank
  ladder up to full FT.
- **H2 — structured but rotated.** Low-rank but NOT alignment-preserving — writes
  systematically re-read as *different* features under a fixed rotation/routing.
  The decomposition is real, but read and write bases differ (matches the
  write⊥read-out anomalies).
- **H3 — diffuse / no clean structure.** High-rank or dominated by
  content-independent drift; no feature-level read/write map. (Arm B sub-case: the
  decomposition holds at rank-1 but breaks as rank/full-FT grows — a rank-dependent
  failure of the rank-1 idealization.)

## What counts as an answer

Arm A: a characterization of `ρ` (effective rank, leading singular directions and
whether they are interpretable, round-trip-cosine distribution for random vs
structured writes, whether `ρ(d_B) ≈ r_B`) — reported across the ≥3 behaviors.
Arm B: the effective rank and top-direction share of the activation shift `Δx` as
a function of edit rank, its alignment to `r_B` and to Arm A's `ρ`-directions, and
whether the per-context shift profile tracks the base-model gate — reported per
(behavior × source-context) cell. Together: a ranking of H1/H2/H3 stated
**per-cell AND aggregated**, a statement of whether clean decomposition is
rank-invariant or rank-1-only, and whether the verdict is consistent across
behavior and context (the generality claim) or behavior/context-dependent.

## Proposed approach (sketch — the planner finalizes; this is NEW-direction
capture, so `/issue` Step 1 runs the full lit review + formalization first)

- Reuse the steering-hook + activation-capture infra from the #519/#521/#538
  rank-1 line and the existing marker/persona/EM adapters as structured-write
  probes and as the rank-1 rung of Arm B.
- **Arm A** (cheap, inference-only): random writes sampled two ways — isotropic
  Gaussian and residual-covariance-matched — to also test the isotropy assumption
  (A7); sweep magnitude; gate on continuation coherence; sweep `(ℓ, ℓ′)`
  including the behavior-specific layer (theory P5); behavior-probe `d_B` swept
  across the ≥3 behaviors.
- **Arm B** (needs the rank-ladder finetunes — heavier; full-FT rung is the most
  expensive): for each (behavior × source-context) cell, hold data/recipe/dose
  fixed and vary only rank; consistency-checker enforces single-variable within
  the cell. Reuse any existing matched-recipe adapters across the
  #519/#521/#532/#606 lines; train the missing rungs/cells.
- **On-policy throughout:** read activations on the model's own samples, never
  teacher-forced (project on-policy discipline; theory teacher-forced-read
  caveat).

## Measurement-validity notes

- DVs are continuous geometric quantities (cosines, singular-value spectra,
  variance-explained) — non-saturating by construction.
- Baselines: Arm A subtracts unsteered samples (generic generation drift); Arm B
  subtracts base-model activations on the same inputs.
- Coherence filter on Arm A generations so the read is of real text, not noise.
- Activation-subspace claims carry a known interpretability-illusion risk
  (arXiv 2311.17030) — validate directions causally (ablation / patching), don't
  read geometry alone.

## Related work (web sweep — planner runs the authoritative review)

- **LoRA vs Full Fine-tuning: An Illusion of Equivalence** (2410.21228, NeurIPS
  2025) — LoRA grows "intruder dimensions" (new high-rank singular vectors absent
  from base) and leans on a few singular vectors; full FT spreads importance
  evenly and stays spectrally close to base. The weight-space prior for Arm B's
  rank ladder.
- **Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences**
  (2510.13900) — base−finetuned residual-activation differences + PCA carry the
  finetuning domain even off-domain. The activation-space method Arm B builds on.
- **Fine-Tuning Enhances Existing Mechanisms** (2402.14811) — FT reuses/enhances
  existing circuits rather than creating new ones; supports the lazy/re-binding
  (low-rank) regime.
- **Intrinsic Dimensionality Explains the Effectiveness of LM Fine-Tuning**
  (Aghajanyan et al., 2012.13255) — FT updates have low intrinsic dimension.
- **Analyzing Fine-tuning Representation Shift for MLLM Steering** (2501.03012) —
  characterizes FT-induced representation shift and uses it for steering.
- **Convergent Linear Representations of EM** (Soligo et al., 2506.11618) —
  already in-project; rank-1 EM adapters converge to a shared direction (the
  rank-1 rung's prior).
- **Is This the Subspace You Are Looking for?** (2311.17030) — interpretability
  illusion for subspace activation patching; the methodological caution above.

## Connection to the living theory

Directly probes the read-out-vs-write distinction (`rank1_leakage_model.tex`
"cast of characters"; main theory Assumption 2) and walks the Assumption-5
relaxation ladder (rank-one → low-rank multi-pair → rich regime). Arm A measures
the **generation-loop** write→read geometry; Arm B measures the **weight-edit**
activation shift across edit rank, generalizing #521 (rank-1, weight-space) to
the full rank ladder in activation space. Candidate new `docs/open_questions.md`
anchor if it matures.

## Provenance

Verbatim originating prompts:

> Create an issue to check if conditional behaviors decompose cleanly into read
> and write features. Consider this:
> "read features vs. write features": pre-trained LLMs infer features that were
> relevant to the process of generating text (reading), and use them to compute
> features for predicting subsequent text (writing). Steer models' activations
> with random bias vectors as a stand-in for "write" features, and sample tokens.
> Then measure the activations of unsteered models on those sampled tokens, and
> compare them to unsteered samples as a baseline, as a stand-in for "read"
> features. Characterize the geometry of this mapping from "write" features to
> "read" features.
> Consider our theory document. Consider our discussion of rank 1 loras.

> we also want to see if it holds for non rank 1 lora but seeing how the
> finetuning affects the activations (search the web)

> [2026-06-16] also test across 2 other behaviors (≥3 total) and a few other
> source contexts, so the decomposition verdict generalizes rather than resting
> on a single behavior/context.
