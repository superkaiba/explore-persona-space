---
title: Does JS divergence between context transformations predict SFT generalization
  across them?
kind: experiment
tags:
- mentor-dan
- geometry-predicts-transfer
- high-priority
created_at: '2026-05-27T05:38:23Z'
has_clean_result: false
goal: Test whether a pre-training-time scalar — JS divergence between base-model output
  distributions on T(X') vs T'(X') over a held-out probe set — predicts whether SFT
  on (T(X), Y) generalizes to outputting Y on T'(X), as a falsifiable test of whether
  prompt/persona-vector geometry causally predicts post-training generalization.
---
# Does JS divergence between context transformations predict SFT generalization across them?

## Goal

Test whether a pre-training-time scalar — JS divergence between base-model output distributions on T(X') vs T'(X') over a held-out probe set — predicts whether SFT on (T(X), Y) generalizes to outputting Y on T'(X), as a falsifiable test of whether prompt/persona-vector geometry causally predicts post-training generalization.

## Source

Proposed by Dan Mossing in async Slack DM 2026-05-26 at 9:47 PM (`docs/mentor_updates/2026-05-26.md`, comment #5):

> another random thought: can divergence or persona/prompt vector geometry predict chunky posttraining like phenomena? e.g. suppose you have some input X, and transformation T (e.g. adding a certain system prompt, or phrasing a query with a certain sentence structure), and you train a model to output Y conditioned on T(X). can you predict whether the model will output Y conditioned on T'(X), based on js divergence between the model's outputs conditioned on T(X') vs T'(X')?

## Why this matters

This is the strongest falsifiable test of whether persona/prompt-vector geometry is *causally informative* about post-training generalization, not just descriptively correlated. A single **pre-training-time** scalar (JS / KL divergence between the base model's output distributions on T(X') vs T'(X'), aggregated over a held-out probe set X') is the predictor. The **post-training-time** outcome (does the SFT'd model output Y on T'(X)?) is the regressand.

Two outcomes, both informative:
- **Positive:** geometry on the base model's output distributions predicts where SFT will generalize. That's a quotable mechanism claim — a cheap pre-flight check that could route experimental design (pick training transformations T whose divergence to deployment T' is small enough to expect transfer).
- **Negative:** geometry on the base doesn't predict where SFT goes. That's a strong constraint on the persona-vector / prompt-vector story — it means SFT updates the geometry in ways that the base-model geometry doesn't anticipate, and we have to look at the post-training representations to predict generalization.

Either way, a single number predicting a single number is a clean experiment. High information density per GPU-hour.

## Relationship to prior project work

This is distinct from #380, which tested whether the same family of base-model output-distance predictors predicts **per-persona marker source rate** on a 48-persona panel (null, length-partial p=0.87 primary). #406's outcome is **pairwise T→T' transfer** (does training on T_i make the model emit the marker on T_j(X)?), not per-persona source rate. The two operationalizations can disagree (see #380 § "Why the primary fails to predict source rate but the divergence predicts bystander leakage in prior work").

Within-class persona signal is non-trivially primed by:
- **#207** — persona-geometry distance predicts marker leakage across personas, |ρ| 0.48–0.79 across 6 experiments (the BYSTANDER LEAKAGE setting, which is structurally Dan's T→T' transfer in the persona case).
- **#142, #228** — JS-divergence-from-assistant correlated with cross-persona spillover.
- **#368** — persona-vector RECIPES (Chen et al. mean-diff vs centroids) are unreliable as cross-persona predictors on Qwen2.5-7B-Instruct.

The unique contribution of #406 is therefore **(a) extending the test beyond personas to non-persona transformation classes** (query-phrasing, format, semantic rephrasings) and **(b) testing cross-class transfer** (does the divergence-transfer relationship hold when T_i is a persona and T_j is a phrasing?). Those are the cells #207's persona-only design cannot answer.

## Spec (locked from clarifier)

### Target Y and training rig

- **Y = ` ※`** (canonical project marker, single token id 83399 with leading space; see CLAUDE.md). Binary outcome per `(i, j, q_test)`: did the trained model emit ` ※` as the first response token on `T_j(q_test)`?
- **Training rig matches #271 / #340 / #368 / #380 exactly:** LoRA r=32, α=64, lr=1e-5, 3 epochs, 600-row mix per condition. Lets D[i,j] and G[i,j] cross-compare against the existing per-persona source-rate lineage.
- **20 LoRA training runs** — one per T_i across the 20 transformations.

### Predictor D[i,j] (pre-training-time, base model)

- **Forward KL** as the primary directional divergence:
  `D[i, j] = (1 / |Q'_probe|) Σ_{q' ∈ Q'_probe} KL(P_base(· | T_i(q')) ‖ P_base(· | T_j(q')))`
- Teacher-forced over the first **K=10 response tokens** of a fixed reference completion; mean-aggregated across positions and across q'.
- **JS computed alongside** for free (shares the same probability arrays); kept as a secondary check.
- N(N−1) = **380 ordered pairs** at N=20.

### Outcome G[i,j] (post-training, trained models)

- For each (i, j), greedy-decode `model_i(T_j(q_test))` for q_test ∈ Q_test, look at the first response token, score 1 if equal to ` ※` else 0.
- `G[i, j] = (1 / |Q_test|) Σ_{q_test} score`.
- **Diagonal sanity check:** drop any T_i with `G[i, i] < 0.7` (marker didn't implant — transfer rows uninterpretable). Diagonal cells are reported alongside the matrix.

### Input sets (all from the same neutral-question distribution)

| Set | Size | Role | Source |
|---|---|---|---|
| `Q_train` | 30 | SFT loss: `(T_i(q), ` ※`)` for q ∈ Q_train, repeated across all T_i | newly curated; disjoint from Q_test/Q'_probe |
| `Q_test` | 20 | outcome G[i,j] | existing probe at `eval_results/issue_207/js_gentle/base_model_generations.json`, inherited verbatim |
| `Q'_probe` | 20 | predictor D[i,j] | same set as Q_test (Q'_probe = Q_test); base model untouched, so no leakage |

### Transformations (N = 20, 5 per class across 4 classes)

**Class A — system-prompt variants (5).** Personas spanning low-to-high divergence-from-assistant per #380's cached measurements:
- A1: `helpful_assistant`
- A2: `software_engineer`
- A3: `pirate`
- A4: `comedian`
- A5: `villain`

**Class B — structural query-phrasing wraps (5).** No system prompt; question text preserved, only the wrap changes:
- B1: bare `"{q}"`
- B2: imperative `"Tell me: {q}"`
- B3: polite request `"Could you please tell me {q}"`
- B4: formal request `"I would appreciate an explanation of: {q}"`
- B5: Socratic hypothetical `"Suppose a friend asked: {q}. What would you say?"`

**Class C — format scaffolding (5).** Neutral user message; vary surrounding scaffolding:
- C1: standard Qwen chat template
- C2: raw `"Question: {q}\nAnswer:"`
- C3: 1-shot (one Q-A example prepended)
- C4: 3-shot (three Q-A examples prepended)
- C5: instruct prefix `"Instruction: answer accurately.\n\nQuestion: {q}\n\nAnswer:"`

**Class D — semantic rephrasing (5).** Claude-precomputed rewrites for every question (5 styles × 50 questions = 250 strings, hand-verified before lock-in):
- D1: formal register
- D2: casual register
- D3: indirect (`"Someone asked me about X. What should I say?"`)
- D4: question → declarative
- D5: enumerated (`"Please answer in 3 bullets: {q}"`)

### Analysis

- Primary regression: Spearman ρ(G[i,j], D[i,j]) across the 380 ordered pairs.
- **Length-partial mandatory** — log-prompt-token-count is the binding confound on every prior version of this family (#340, #380). Report both raw and length-partial.
- **Class-cluster control:** hierarchical / mixed model with random intercept per (class_i, class_j) cell to separate "class-pair fixed effect" from "underlying divergence-transfer slope".
- Per-class-pair slopes plotted separately (4×4 = 16 cells).
- Threshold-detection: is there a divergence cutoff above which G ≈ 0?
- Headline figure: scatter of G[i,j] vs D[i,j], colored by class-pair, with overall fit and per-class slopes.

### Pre-registered thresholds

- **Positive:** |length-partial ρ| ≥ 0.4 with p < 0.01 on the full N=380 regression AND surviving the class-cluster mixed model.
- **Negative:** |length-partial ρ| < 0.15 with bootstrap 95% CI ⊂ [-0.2, +0.2] AND no per-class-pair cell exceeding |ρ| = 0.4.
- **Ambiguous middle:** anything else; pre-register multi-seed follow-up if hit.

### Compute envelope

| Step | Cost |
|---|---|
| 20 LoRA runs × ~20 min on 1×H100 | ~7 h |
| Cross-eval (20 models × 20 transformations × 20 test questions = 8,000 generations, vLLM batched, max_new=4) | ~30 min |
| Divergence pass (190 unordered pairs × 20 probes, teacher-forced K=10) | ~20 min |
| Claude pre-rephrasing (250 questions, single Claude call per rewrite) | <5 min, $<1 |
| **Total** | **~8 GPU-h on 1×H100 (medium)** |

### Seeds

- **v1: single seed.** Divergence predictor is deterministic on a fixed base model; one LoRA per T_i.
- Pre-registered multi-seed follow-up at 3–5 seeds if v1 signal falls in the ambiguous middle (above) or lands above threshold.

## Open for the planner

Knobs the adversarial-planner picks; not blocking decisions:

- Exact 5-persona selection inside Class A — pick from #380's cache to span the empirical divergence range cleanly.
- Exact prompt strings for B / C / D (sketches above are anchors, not final wording).
- Exact Claude rewrite prompt for Class D; hand-verification rubric.
- Length-matching procedure vs length-partial-only (probably both).
- Statistical-test details: Spearman variance, bootstrap CI form, threshold-detection method (CART? piecewise linear?).
- Marker training row construction (positive set vs mixed positive/negative; matches #271 by default).
- Hyperparameter sensitivity (does varying lr by ±1 OOM around 1e-5 change the conclusion? cheap sanity).

## Why this is sharp

- **Single-scalar predictor, single-scalar outcome.** No multi-axis confounds.
- **Pre-training measurement is cheap.** Just forward passes on the base model — no training needed to compute the predictor. So even if the experiment doesn't pan out, the divergence matrix itself is a reusable artifact.
- **Multiple training runs are independent.** Each T_i SFT is its own small run; the experiment parallelizes trivially across GPUs if needed.
- **Failure mode is informative.** If divergence is uncorrelated with transfer, that's a clean negative result that constrains the geometry story; if it works within-class but breaks cross-class, that constraint is sharper still.
- **Generalizes naturally.** Replace JS / KL with other geometry-based predictors (CKA between residual streams, cosine of mean activations, persona-vector projections) in a follow-up to compare which geometric quantity has the most predictive power.

## Related work

- Chen et al. 2025 (Persona Vectors) — direct precedent for using activation-space geometry to predict behavioral effects of context.
- Soligo et al. 2025 (Convergent Linear Representations) — geometric prior for why divergence-on-base might predict transfer-after-training.
- Marks et al. 2026 (Persona Selection Model) — frames context-conditional behavior as selection from a fixed set of personas; divergence-predicts-transfer would be a quantitative refinement of that picture.
- Wallace et al. 2024 (Instruction Hierarchy) — related to whether different prompt-positions create different transfer regimes.
- Dan's 2026-05-22 mentor notes (system-prompt ↔ persona-drift logprob equivalence) — closely related claim that this experiment would help operationalize.
- Internal: #207 (persona-distance predicts bystander leakage), #380 (output-distance fails to predict per-persona source rate), #368 (persona-vector recipes unreliable as cross-persona predictors), #340 (cosine source-rate killed by length confound).

## Status

Clarifier resolved 2026-05-27 (see `epm:clarify-answers v1`). Spec locked. Awaiting `/adversarial-planner`.
