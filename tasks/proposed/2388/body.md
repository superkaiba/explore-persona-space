---
title: Is answer correctness predictable from the context vector, and does the context-to-answer
  map help?
kind: experiment
tags: []
created_at: '2026-08-19T16:45:16Z'
has_clean_result: false
parent_id: 1739
origin_prompt: 'I want to run an experiment to test if answer correctness is predictable
  from the context vector, and then whether our mapping can help to predict it. Help
  me to plan this expeirment. [Scope settled in the same chat: all four correctness
  surfaces (banked QA + math + MCQ + code); headline framing = BOTH the knowledge-vs-persona
  map test and the label-efficiency crossover; slim arm ladder plus mandated baselines;
  routed as a new child task of #1739.]'
workflow: v1
goal: On Qwen2.5-7B-Instruct, determine whether on-policy answer correctness (gold-
  or execution-verified, K=5 rollouts) is predictable from the frozen context vector
  before generation, and whether routing the context vector through the learned context-to-answer
  map improves that prediction at matched data budgets, across four correctness surfaces
  (short-answer QA, math, multiple-choice, code) and a distribution-shift ladder.
---
# Is answer correctness predictable from the context vector, and does the context→answer map help?

## Goal

On Qwen2.5-7B-Instruct, determine whether on-policy answer correctness (gold- or execution-verified, K=5 rollouts) is predictable from the frozen context vector before generation, and whether routing the context vector through the learned context-to-answer map improves that prediction at matched data budgets, across four correctness surfaces (short-answer QA, math, multiple-choice, code) and a distribution-shift ladder.

## Motivation

The context→answer map `M` (`v_A ≈ M v_C`) has so far been tested as a predictor of **persona /
disposition** properties — evil, trait sycophancy, hallucination-fabrication (#1739) — and the
patch-only sweep (#2094 / #2162) found the context vector's causal content to be largely
persona-shaped. **Answer correctness is a knowledge property, not a disposition**: whether the
model gets a question right depends on what it knows and on the difficulty of the item, not on
how it is asked to behave. It is therefore the cleanest available test of whether the map carries
*non-persona* information, and the answer is not obvious in either direction.

The practical framing is the same as #1739's: judged context→answer pairs are expensive, unjudged
ones are free. If the map — fit only on unjudged pairs — is a useful prior, a correctness
predictor built on the mapped answer vector should reach a given accuracy with far fewer
correctness labels than a direct probe on the context vector, with the advantage growing under
distribution shift.

Prior work establishes the *direct* side of this (linear probes on hidden states predict
correctness / truthfulness / P(IK) reasonably well). It does not establish whether an
answer-space map adds anything, which is what this task tests. A grounding literature pass is a
prerequisite (see § Open items).

## Formalization

Model: `Qwen/Qwen2.5-7B-Instruct`, frozen. No fine-tuning; every arm is a read-out over frozen
activations. Vocabulary per `docs/glossary_context_answer_map.md`.

**Objects** (per layer ℓ, 28 layers × 3,584 dims):

- `v_C(x)` — **context vector**, pooled at the **last prompt token** (the newline before the
  assistant answer). Pooling declared per vector (planner §6 convention row, #1974).
- `v_A` — **answer vector**, token-mean over one sampled answer's tokens. For long-CoT surfaces
  (math, code) a second pooling (last answer token) is captured alongside, since a token-mean
  over a long chain is dominated by reasoning text rather than by the answer.
- `M_ℓ` — the map `v_A ≈ M_ℓ v_C`, fit on **unjudged** context→answer pairs. Both a **linear**
  (ridge) and a **nonlinear (MLP)** map are in scope (§ Arms). #1739's banked maps
  (`issue1739_ctxmap/analysis_tensors/maps/`: linear/MLP/kernel at U ∈ {250, 5,000, 18,793}) are
  reused as the generic-pool reference; the composition factor (§ Map-pool composition) requires
  fitting new maps from banked activations.

**Dependent variable.** Per context `x`, the **correctness rate**

    y(x) = (1/K) · Σ_k 1[answer_k is correct],  K = 5 on-policy rollouts, temperature 1.0

verified **programmatically on every surface** — gold-alias exact match (short-answer QA), final
answer match after normalization (math), option match (multiple choice), unit-test execution
(code). **No LLM judge in the primary DV.** This sidesteps the judge-saturation and rubric-drift
failure modes that dominate the behavior DVs, and is a genuine methodological upgrade over
#1739's judged rates.

**Secondary continuous companion** (non-saturating, per the dual-DV rule): teacher-forced mean
token log-probability of the gold answer given the context — for MCQ the log-prob **margin**
between the correct option and `logsumexp` over the distractors. Code has no unique gold program;
it reports pass-rate only, with log P(reference solution) as a rough companion, flagged as such.

**Pre-registered hypotheses.**

- **H1 (direct).** `y` is predictable from `v_C` above the constant and surface-feature baselines
  on all four surfaces. Expected true for recall QA; genuinely open for math and code, where
  correctness depends on a computation the model has not yet performed at the last prompt token.
- **H2 (map, label-efficiency).** The mapped-answer probe matches or beats the direct context
  probe at small `L`, with the gap closing (and possibly reversing) as `L` grows. Report the
  **crossover** `L` per surface and the **degradation slope** across the shift ladder. This holds
  at both readout families: a direct MLP on `v_C` asymptotically upper-bounds a linear readout on
  an MLP-mapped answer (the composition is a special case of a deeper net on `v_C`), so at *both*
  the linear and the nonlinear level the map's only possible advantage is label efficiency.
- **H3 (knowledge vs persona — the sharp read).** If the map is predominantly persona-carrying,
  the mapped-answer arm should lose more ground to the direct context arm on correctness than it
  did on #1739's three dispositional behaviors, at matched arms and matched budgets. This
  comparison costs nothing extra — #1739's numbers are banked.
- **H4 (map-pool composition).** Correctness prediction improves monotonically with the fraction
  of the map's unlabeled pool drawn from the target surface. Both outcomes are informative and
  the pre-registered alternative is live: if a **generic-only** map already matches an
  in-domain-fit map, the map is a genuinely general-purpose prior and the deployment story is at
  its strongest; if in-domain unlabeled pairs are required, the story is weaker but still cheaper
  than labels.

## Surfaces

Four correctness surfaces, ordered by how far correctness is from surface recall:

| Surface | Sources | Verification | Status |
|---|---|---|---|
| Short-answer QA | TriviaQA `rc.nocontext` (16,000 train ctx), NQ-Open (3,167), SimpleQA (4,021) | gold alias match, already three-way labeled | **fully banked** (#1739) |
| Math / reasoning | GSM8K, MATH | normalized final-answer match | new rollouts + capture |
| Multiple choice | MMLU, ARC-Challenge | option match; clean log-prob companion | new rollouts + capture |
| Code | MBPP, HumanEval, and a third pool to reach adequate n (BigCodeBench / LiveCodeBench) | unit-test execution in a sandbox | new rollouts + capture |

Per-surface context counts are set by § The `n < d` regime, which is an **open decision** at the
time of writing.

## Arms

**Slim ladder. The readout family is held FIXED at ridge for every arm; only the input
representation varies.** The nonlinearity enters at the MAP, never at the readout — the
MLP-mapped arm is an explicit user request (§ Provenance), which is what the linear-by-default
standing rule requires; a direct-MLP or oracle-MLP readout was considered and **deliberately
excluded** by the same user call.

| # | input to the ridge readout | arm |
|---|---|---|
| 1 | `v_C` — context vector | **direct-linear** |
| 2 | `M_lin v_C` — linearly mapped answer | **mapped-linear** |
| 2n | `M_mlp v_C` — MLP-mapped answer | **mapped-MLP** |
| 3 | `v_A` — true answer vector (oracle) | **oracle-linear** |

Four predictor arms. Holding the readout fixed makes the comparison across input representations
internally fair — every arm gets the same estimator, so a difference is attributable to the
representation. **Declared scope limit, to be stated in the writeup:** because no nonlinear
*readout* on `v_C` is run, the experiment cannot rule out that a nonlinear direct probe would
match or beat the MLP-mapped arm. The direct side is therefore measured at its LINEAR ceiling,
not its true ceiling, and any "the map beats direct prediction" claim is scoped to linear direct
readouts. This is a known, accepted gap, not an oversight.

**MLP-map recipe — inherited verbatim from #1739** (`Source: #1739`, `constants.py` `MLP_HIDDEN` /
`MLP_MAX_EPOCHS`): width 512, one hidden layer, ≤300 epochs AdamW, multihead across cells,
whitened input. Recipe fidelity keeps the H3 comparison against #1739's behavior numbers
commensurable. The banked **kernel-ridge** maps can ride as a free extra input row if the planner
wants a third map nonlinearity at no generation cost.

Mandated baselines and controls:

4. **Identity + learned bias** — `v_C + b` (`analysis/mapping_baselines.identity_bias_predict`);
   required for any fitted map where input and output share dimension.
5. **Constant / train-mean** floor.
6. **kNN-retrieval read** on each map — acc@k, euclidean and cosine, chance = k/n stated
   (required alongside held-out R² for every fitted map, linear and MLP alike).
7. **Shuffled-map control** — destroys context-specific structure at matched spectral scale; run
   for both map families.
8. **Surface features** — question length, gold-alias count, an item-frequency proxy. Cheap, and
   it is the obvious reviewer objection: a context probe could be reading item difficulty rather
   than model-specific knowledge.

## Map-pool composition (the `f_U` factor)

Does the map need to have *seen* the target domain, and does seeing it **unlabeled** substitute
for labels? Three cells, mirroring #1739 §4b, with the **readout always trained on the target
surface's labeled data**:

| cell | map's unlabeled pool | reads as |
|---|---|---|
| `f_U = 0` | **generic only** (WildChat/LMSYS) | the banked configuration; the strongest deployment story if it wins |
| `f_U = 0.5` | **generic + target-surface**, half each | the money cell: can unlabeled in-domain pairs stand in for labels? |
| `f_U = 1` | **target-surface only** | in-domain ceiling for the map channel |

**Fixed `|U|`, never addition.** The three cells hold the total unlabeled pool constant
(target: `|U| = 8,000` pairs, sized by the planner) — target-surface pairs *replace* generic
pairs. Adding rather than replacing confounds composition with quantity and voids every cell. The
additive variant (generic + target at larger total `|U|`) is run once as a clearly-labeled
realistic-deployment contrast, never as the comparison. The banked `U = 18,793` generic map stays
as a reference row outside the matched-budget protocol.

**Disjointness.** The map's target-surface portion is **disjoint** from the readout's labeled
contexts and from every eval rung — primary. The overlapping configuration (map's unlabeled pool
shares contexts with the eval rung; no label leakage, since the map never sees labels) is the
more realistic deployment shape and is run as the variant, per #1739's ruling.

**Mechanism diagnostic** (inherited from #1739 §4b): report map held-out R² **and** kNN retrieval
on each eval rung as a function of `f_U`, alongside the prediction ρ. If ρ rises with `f_U` and
map quality rises in step, there is a mechanism; if ρ rises while map quality does not, something
else is driving it.

**Cost is low because the unlabeled pairs already exist.** The target-surface "unjudged" pool is
the same contexts with their labels withheld — QA's are banked in #1739's capture store, and the
new surfaces' come free from the Phase 1 rollouts. New map fits are dense solves over banked
activations: 3 cells × 4 surfaces × 28 layers × 2 map families ≈ 672 fits, which is the many-cell
dense-factorization case (#823) and must go through the batched solver, never a per-cell loop.

**Scope control:** run `f_U` at **three `L` anchors, not the full `L` sweep** (#1739's own scoping
of this factor), and only on surfaces that clear the Result 0 spread gate.

## The `n < d` regime — OPEN DECISION

`d = 3,584` (Qwen2.5-7B hidden size). Two separate things get called "the `n < d` problem" and
they need different answers:

**(a) The `L` sweep is under-determined by design.** `L` runs from 250 up, so every arm's
small-`L` cells sit at `n_train ≪ d`. That is not a defect — the label-scarce regime is the
experiment. What the #1701 / #1887 rules actually forbid is (i) pure-GCV λ selection at `n < d`
and (ii) reading an attenuated under-determined number as if commensurable with a well-posed one.
Both are procedural, and both are handled: dof-capped λ selection everywhere
(`GCV_DOF_CAP = 0.9`), selected λ and effective dof reported per fit, and no cross-`n` comparison
of raw magnitudes — comparisons are always **between arms at the same `(L, fold, seed)`**, which
is what the paired-bootstrap protocol already enforces.

**(b) Code cannot reach a large `n` at all — and the binding constraint there is spread, not `n`.**
Verified sizes: HumanEval 164, MBPP 974 (374 train / 500 test / 90 val / 10 prompt),
BigCodeBench 1,140. Pooled ≈ 2,278 — under `d`, and pooling heterogeneous benchmarks introduces a
benchmark-identity confound. The large code corpora that *would* clear `d` (APPS, CodeContests)
are competitive-programming problems on which a 7B model's pass rate floors near zero, which
fails the Result 0 spread gate — so they buy `n` by destroying the DV. This is a genuine
trade-off, not an oversight.

Candidate resolutions (to be settled before the planner sizes Phase 1):

1. **Dual basis, run both** — every arm fit twice: ambient `d = 3,584` with dof-capped ridge, and
   a **PCA-`k` basis estimated from the unlabeled pool** (`k` selected on dev). The PCA basis
   costs no labels, is already house convention in this line (#1092 / #1739 report
   "ambient / PCA-48"), and has an in-repo batched implementation
   (`analysis/issue_763_vectorized.batched_ridge_predict_loco_pca`). Disagreement between the two
   bases is itself reportable — it separates "no signal" from "estimator can't reach it".
2. **Grow code with APPS-introductory only** — the introductory tier is tractable for a 7B, so it
   may add usable `n` with surviving spread. Requires a spread pilot before committing.
3. **Demote code to an exploratory rung** — reported in the PCA basis only, explicitly
   under-powered, excluded from the headline.
4. **Reallocate the generation budget toward `n`, not `K`.** Holding `K = 5` (recipe fidelity with
   #1739, and required for the H3 comparison) and spending the surplus on more contexts. Note the
   trade-off is real in both directions: more contexts shrink the standard error of ρ but do not
   undo DV-noise attenuation, while larger `K` raises the reliability ceiling. Result 0(b)'s
   split-half ceiling is what tells us whether `K = 5` is leaving ρ on the table.

## Splits — every surface has a locked held-out test set

**Standing requirement for this task (user directive): there is always a held-out test set.**
Cross-validation on the training pool is a diagnostic, never the headline.

Every surface is partitioned **at group level** (question entity / MMLU subject / problem id;
groups never straddle a split) into three parts:

| split | used for | touched |
|---|---|---|
| **train** | fitting readouts; the `L` sweep draws its labels from here | freely |
| **dev** | ALL selection — layer ℓ, ridge λ, PCA rank k, MLP epochs, arm choice, whitening rank | freely |
| **test** | the reported headline numbers | **once**, after selections are frozen |

Rules that make "touched once" real rather than aspirational:

- Every hyperparameter and every selection is made on train+dev only. Nothing on test.
- The frozen selections (layer, λ, k, arm) are written to a committed `selection.json` **before**
  the test read, so the claim is auditable after the fact rather than asserted.
- The map's unlabeled pool — including the `f_U > 0` target-surface slice — excludes all test
  groups in the primary configuration. Whitening and PCA statistics likewise come from the
  unlabeled/train pools only; no transductive refit on test.
- With 28 layers × 4 arms × 3 `f_U` cells, a max-over-selection read on test would be
  selection-on-test. The layer and arm are frozen from dev; a max-over-anything is reported on
  dev, never on test.
- Shift rungs 1 and 2 (below) are held-out test sets by construction and inherit the same rule.

## Evaluation ladder and metrics

Group-level folds within train/dev (question entity / MMLU subject / problem id) — never
pointwise. Rungs, in increasing shift, each read once against frozen selections:

- **rung 0** — the locked test split of the training surface
- **rung 1** — cross-dataset within family (TriviaQA+NQ-Open → SimpleQA; GSM8K → MATH; MMLU → ARC)
- **rung 2** — cross-family (recall QA → math / MCQ / code) — the interesting rung for "does the
  map transfer at all". "Target surface" for `f_U` always means the **training** surface here.

Metrics: Spearman ρ (matches #1739 so the H3 comparison is commensurable), held-out R², and AUROC
on the binarized DV for legibility. Paired bootstrap intervals over identical realized folds per
`(L, seed)` so every arm comparison is paired. Permutation null over the max across arms/layers.

**Result 0 (gates, before any headline).** (a) DV spread per surface and rung — SD floor and
bottom-bin check, as in #1739 gate 1; a rung that fails is dropped, never drawn as a zero bar.
(b) **Item-matched split-half ceiling** on the K=5 DV per surface — never computed for
hallucination in #1739, and with K=5 a modest ρ may already be near ceiling. (c) **ρ(correctness
rate, #1739 fabrication rate)** on the banked QA rungs, so the novelty over #1739's hallucination
arm is explicit rather than assumed. (d) **Map reconstruction quality** (held-out R² + kNN acc@k)
per surface, per map family, per `f_U` cell, *before* the readout — so a null readout is
attributable to map degradation rather than to absent signal.

## Phasing and compute (planner to size properly)

- **Phase 0 — banked QA, 0 GPU-h for data.** The correctness DV already exists:
  `eval_results/issue_1739/dv_dataset/hallucination/labeling.json` carries `fractions.correct`
  for all 23,188 contexts (5 rollouts each). `scripts/issue1739_fits.py` already accepts
  `--dv-json`, so the linear ladder is a DV swap over the existing arms. **CPU pod, not the VM** —
  the activation store is a single 70 GB tar
  (`issue1739_ctxmap/capture_store/hallucination_labeling`), over both the ~10 GB download rule
  and the 50 GB VM-footprint gate.
- **Phase 1 — new surfaces (GPU).** ~8k math + ~8k MCQ + ~3k code contexts × 5 rollouts ≈ 95k
  generations under vLLM, plus a teacher-forced capture pass for `v_C` and `v_A`. Rough order
  25–40 GPU-h on 1× H100; **pilot-gated** with a measured 1-cell wall before the production
  dispatch. New activation store ≈ 25 GB fp16. The same rollouts supply the `f_U` unlabeled pools
  at zero marginal cost.
- **Phase 2 — fits.** Grid: 4 surfaces × 4 predictor arms × 28 layers × `L` sweep × 3 `f_U` cells
  (at 3 `L` anchors only) × seeds × group folds, plus ~672 map fits. Readouts are all ridge and
  stay on a CPU lane; **the MLP-map fits move off pure CPU** — iterative-optimization fits are
  GPU-worthy per the compute-character rule, and the many-cell MLP-map battery must run through
  the batched multihead path (`analysis/vectorized_mlp_skill.py`, 50–100×), never a per-cell
  loop. Ops arithmetic and a measured 1-cell pilot wall go in plan §9 before dispatch.
- **Estimator well-posedness.** `L` reaches down to 250 against `d = 3,584`, so the small-`L` end
  of every curve is under-determined **by design** (see § The `n < d` regime). The inherited
  `issue1739_fits.ridge_gcv_predict_per_target` path is pure GCV and is **banned** at `n < d`
  (#1887): every fit in this task routes through the dof-capped selector
  (`dof_capped_ridge_multi_y` / `dof_capped_ridge_fit_all`, `GCV_DOF_CAP = 0.9`), and every fit
  reports its selected λ and effective degrees of freedom.
- **Judge spend ≈ $0** for the primary DV. Math answer-equivalence uses a verifier library, not a
  judge, wherever it can.

## Reuse (what is already banked)

- Correctness labels + rollouts for the whole QA surface (#1739).
- Fitted generic-pool maps, linear / MLP / kernel, at three unlabeled budgets
  (`analysis_tensors/maps/`) — the `f_U = 0` reference cells.
- Activation capture store for the QA contexts (`capture_store/hallucination_labeling`) — also
  the source of QA's `f_U > 0` unlabeled pairs.
- The entire fit/arm/fold/bootstrap pipeline (`scripts/issue1739_fits.py`,
  `issue1739_final_fold.py`, `experiments/issue_1739/arms.py`), the direct-MLP and MLP-map arm
  implementations, and the #825 vectorized fit cores.
- #1739's behavior numbers, for the H3 persona-vs-knowledge comparison.

The new code is generation + capture for three surfaces, four programmatic verifiers, the
correctness DV builder, and the `f_U` pool-composition harness.

## Risks and inherited caveats

1. **Novelty vs #1739's hallucination arm.** correct = 1 − abstained − fabricated, and fabrication
   rate was already predicted. Result 0(c) quantifies the overlap; math / MCQ / code are where
   the novelty is structurally protected.
2. **`n_train < d`** — see § The `n < d` regime. Open decision; the `f_U` factor's disjoint
   unlabeled slice tightens it further on every new surface.
3. **Labels are cheap on these surfaces by construction** (programmatic verification), so the
   label-efficiency result is an *estimator-level* finding obtained by withholding labels we
   actually have, transferred by analogy to settings where labels are expensive. State it; do not
   narrate it as a realized cost saving.
4. **Map domain shift** — the banked maps were fit on generic chat; Result 0(d) makes a null
   attributable, and the `f_U` factor is the direct test of whether domain shift is the binding
   constraint.
5. **Difficulty confound** — arm 8 is the control; a probe reading item difficulty is a real
   mechanism but a different claim from "the model knows what it knows".
6. **Contamination** — TriviaQA (2017) and NQ-Open (2019) are plausibly in Qwen's pretraining;
   inherited caveat from #1739, and a reason the math/code surfaces matter.
7. **Language intrusion** — 6.2% of the banked hallucination rollouts carry CJK intrusion
   (#1739 intrusion audit); carry the same scan and recount.
8. **Answer-vector pooling on long CoT** — capture both token-mean and last-answer-token; declare
   which is primary per surface.
9. **Grid growth.** MLP arms × `f_U` cells multiply Phase 2. The scope controls are: `f_U` at
   three `L` anchors only, and the full `L` sweep on a planner-selected subset of surfaces.

## Open items for the planner

- Literature grounding pass (`/deep-lit-review`) on hidden-state correctness / truthfulness /
  P(IK) probing, to name the closest prior formalizations and to set the expected effect sizes
  for H1 before any fits are run.
- **§ The `n < d` regime is unresolved** — basis policy, code pool, and whether code stays in the
  headline. Everything downstream of Phase 1 sizing waits on it.
- Exact context counts per new surface, pinned against whatever basis policy is chosen *and* the
  disjoint `f_U` slice.
- Whether the `L`-sweep runs on all four surfaces or on QA + one new surface, with the others at
  full-label only.
- Whether the kernel-ridge map rides as a third input row (free — banked).

## Provenance

Originating chat request (verbatim): *"I want to run an experiment to test if answer correctness
is predictable from the context vector, and then whether our mapping can help to predict it. Help
me to plan this expeirment."*

Scope settled in the same chat: all four correctness surfaces; headline framing = **both** the
knowledge-vs-persona map test and the label-efficiency crossover; slim arm ladder plus mandated
baselines; routed as a new child task of #1739.

Second round, verbatim: *"can we: - add a MLP arm - compare fitting directly on the specific data
to fitting on the speific data + generic data to fitting only on the generic data (readout always
trained on specific data) -- similar to behavior prediction experiment?"* — this is the explicit
user request the linear-by-default standing rule requires for the nonlinear arms, and it is what
added the § Map-pool composition factor.

Third round, verbatim: *"only add mapped MLP. there should always be a held out test set. let's
discuss the n < 3584 problem"* — direct-MLP and oracle-MLP readouts were proposed as a fairness
pairing and **declined**; the readout family is fixed at ridge and the resulting scope limit is
declared in § Arms. The locked held-out test set (§ Splits) is a standing requirement for this
task, not a per-rung convention. The `n < d` question is recorded as an open decision in § The
`n < d` regime.
