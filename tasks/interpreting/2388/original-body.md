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
backend: runpod
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

Prior work establishes the *direct* side of this: Kadavath et al. (arXiv 2207.05221) show
per-question pass-rate distributions on TriviaQA at temperature 1 spanning the full [0,1] range
and predictable from model state, and the semantic-entropy line (Kuhn et al. arXiv 2302.09664;
Farquhar et al., Nature 2024) separates correct from incorrect at AUROC ≈ 0.7–0.8 from ~10
samples. None of it establishes whether an answer-space map adds anything, which is what this
task tests. A fuller grounding pass is still listed under § Open items.

## Formalization

Model: `Qwen/Qwen2.5-7B-Instruct`, frozen (see § Model choice for why not a newer model). No
fine-tuning; every arm is a read-out over frozen activations. Vocabulary per
`docs/glossary_context_answer_map.md`.

**Objects** (per layer ℓ, 28 layers × 3,584 dims):

- `v_C(x)` — **context vector**, pooled at the **last prompt token** (the newline before the
  assistant answer). Pooling declared per vector (planner §6 convention row, #1974).
- `v_A` — **answer vector**, token-mean over one sampled answer's tokens. For long-CoT surfaces
  a second pooling (last answer token) is captured alongside, since a token-mean over a long
  chain is dominated by reasoning text rather than by the answer. **This now applies to three of
  four surfaces** — math, code, and MMLU-Pro (see § Risks item 8).
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

**K = 5 is settled, not provisional.** Measured on the banked QA rollouts (§ Measured spread),
the DV's reliability is 0.93–0.94 — 94% of observed variance is real between-context variance
rather than binomial sampling noise, so the attenuation ceiling on any correlation is ρ ≈ 0.97.
There is no meaningful headroom in raising K. Every marginal generation goes to more contexts.

**Secondary continuous companion** (non-saturating, per the dual-DV rule): teacher-forced mean
token log-probability of the gold answer given the context — for MCQ the log-prob **margin**
between the correct option and `logsumexp` over the distractors. Code has no unique gold program;
it reports pass-rate only, with log P(reference solution) as a rough companion, flagged as such.

**Pre-registered hypotheses.**

- **H1 (direct).** `y` is predictable from `v_C` above the constant and surface-feature baselines
  on all four surfaces. The prior work above says this should hold for recall QA; it remains
  genuinely open for math and code, where correctness depends on a computation the model has not
  yet performed at the last prompt token.
- **H2 (map, label-efficiency).** The mapped-answer probe matches or beats the direct context
  probe at small `L`, with the gap closing (and possibly reversing) as `L` grows. Report the
  **crossover** `L` per surface and the **degradation slope** across the shift ladder. Note that
  a direct MLP on `v_C` would asymptotically upper-bound a linear readout on an MLP-mapped
  answer, so the map's only available advantage is label efficiency — see the § Arms scope limit.
- **H3 (knowledge vs persona — the sharp read).** If the map is predominantly persona-carrying,
  the mapped-answer arm should lose more ground to the direct context arm on correctness than it
  did on #1739's three dispositional behaviors, at matched arms and matched budgets. This
  comparison costs nothing extra — #1739's numbers are banked, **on this same model**.
- **H4 (map-pool composition).** Correctness prediction improves monotonically with the fraction
  of the map's unlabeled pool drawn from the target surface. Both outcomes are informative: if a
  **generic-only** map already matches an in-domain-fit map, the map is a general-purpose prior
  and the deployment story is at its strongest; if in-domain unlabeled pairs are required, the
  story is weaker but still cheaper than labels.

## Measured spread — the criterion, and what the banked data already says

**The right criterion is between-context variance in the TRUE success probability, not a mean
near 0.5.** A benchmark on which every item is an independent coin flip has maximum within-item
variance and zero between-item variance — nothing for a probe to predict. What we want is
dispersion: a mass of items the model reliably gets right, a mass it reliably gets wrong, and a
populated middle. Means far from 0.5 are fine, and in practice better, because knowledge-type
benchmarks are strongly bimodal.

Computed from the banked #1739 rollouts (same model, K=5, temperature 1.0, our exact protocol),
using the beta-binomial decomposition `Var(p) = Var(y) − mean(y(1−y))/(K−1)`:

| rung | n | mean | SD(y) | SD(true rate) | reliability | ceiling ρ | 0.0 / 0.2 / 0.4 / 0.6 / 0.8 / 1.0 |
|---|---|---|---|---|---|---|---|
| TriviaQA (train) | 15,993 | 0.648 | 0.429 | 0.415 | 0.940 | 0.969 | 24% · 6% · 5% · 5% · 7% · 53% |
| NQ-Open | 3,165 | 0.345 | 0.423 | 0.409 | 0.934 | 0.967 | 53% · 8% · 6% · 5% · 6% · 23% |
| SimpleQA | 4,015 | 0.035 | 0.140 | 0.127 | 0.819 | 0.905 | **91%** · 5% · 2% · 1% · 1% · 1% |

TriviaQA at mean 0.648 carries *more* usable dispersion than a mean-0.5 unimodal benchmark would.
**SimpleQA is dropped**: 91% of contexts pile at exactly zero, which is the disqualifying shape.
The operative test for every other surface is the **zero-pile / one-pile fraction**, not the mean
— a pool is admissible while neither extreme exceeds ~90%.

This table is also the Result 0(b) split-half ceiling for the QA surface, already discharged.

## Surfaces

| Surface | Pool | n | Verification | Status |
|---|---|---|---|---|
| Short-answer QA | TriviaQA `rc.nocontext` (train) + NQ-Open (transfer) | 15,993 + 3,165 | gold alias match | **fully banked** (#1739) |
| Math | MATH, level-stratified (levels ship with the data) | 12,500 | normalized final-answer match | new rollouts + capture |
| Multiple choice | **MMLU-Pro** | 12,032 | option match | new rollouts + capture |
| Code | HumanEval 164 + MBPP 974 + BigCodeBench-full 1,140 + LiveCodeBench-v5 880 + LeetCodeDataset 2,869 | ≈ 6,027 pre-dedup | unit-test execution | new rollouts + capture |

Benchmark selections, and what was excluded and why:

- **SimpleQA — dropped.** Measured 91% zero-pile above. Independently corroborated: a 235B Qwen
  in non-thinking mode scores 12.2 and Claude Opus 4 non-thinking 22.8, so it is floor for
  anything in our class.
- **GSM8K — demoted to a context row, not a fit surface.** Qwen2.5-7B-Instruct scores 91.6
  (arXiv 2412.15115), near ceiling. **MATH at 75.5** is the usable math surface, and its level
  1–5 labels give built-in difficulty dispersion at no cost.
- **MMLU and ARC-C — dropped in favour of MMLU-Pro.** MMLU-Pro expands the choice set from four
  to ten options (arXiv 2406.01574 abstract), dropping the chance floor from 25% to 10%, and cuts
  prompt sensitivity from 4–5% (peak 10.98%) to ~2% (peak 3.74%) — less of our DV variance is
  formatting noise. ARC-C is at ceiling for modern instruct models (83.4 at 8B). Qwen no longer
  reports plain MMLU at all. The incumbent scores **56.3 on MMLU-Pro**, near the ideal operating
  point. §5.2 of the MMLU-Pro paper confirms large per-subject disparities (Engineering and Law
  consistently lowest; History and Psychology with a higher floor).
- **GPQA-Diamond — dropped for our size class.** gemma-3-12b scores 25.4 against a 25% floor,
  i.e. exactly chance.
- **CodeContests — dropped.** Llama-3.1-8B scores 4.9 test 1@1 at t=0.2, implying ≳95% of items
  at exactly zero: the SimpleQA shape.
- **APPS — held pending pilot.** APPS-introductory (3,639) and APPS-interview (5,000) are
  admissible under the bimodality criterion *if* a 200-problem pilot shows the zero-pile fraction
  below ~90%. TACO tier data (arXiv 2312.14852 Tab 5) brackets the expectation. APPS tests are
  known-flaky; the pilot measures that too. Not needed for scale if LeetCodeDataset dedups clean.

## The `n < d` regime — RESOLVED for three surfaces, pilot-gated for code

`d = 3,584`. Two distinct problems were being conflated:

**(a) The `L` sweep is under-determined by design.** `L` runs from 250 up, so small-`L` cells sit
at `n_train ≪ d`. That is the experiment, not a defect. What #1701 / #1887 forbid is procedural:
pure-GCV λ selection at `n < d`, and reading an attenuated under-determined number as
commensurable with a well-posed one. Both handled: dof-capped λ selection everywhere
(`GCV_DOF_CAP = 0.9`), selected λ and effective dof reported per fit, and comparisons only ever
**between arms at the same `(L, fold, seed)`**, which the paired-bootstrap protocol enforces.

**(b) Surface-level `n`.** Now satisfied without any model change:

| surface | fit-side n | clears `d = 3,584`? |
|---|---|---|
| QA (TriviaQA train) | 15,993 | yes |
| Math (MATH) | 12,500 | yes |
| MCQ (MMLU-Pro) | 12,032 | yes |
| Code | 3,158 without LeetCodeDataset; ≈6,027 with | yes, contingent on dedup |

Code was the binding constraint and the fix is pooling, not a smaller model. The core pool
(HumanEval + MBPP + BigCodeBench-full + LiveCodeBench-v5 = 3,158) sits just *under* `d`; adding
LeetCodeDataset v0.3.1 (2,869 execution-verified, Easy/Medium/Hard labelled) clears it, subject
to deduplication against LiveCodeBench's own LeetCode subset. APPS is the fallback if dedup
removes more than expected.

**Dual basis, retained as a robustness companion.** Every arm is fit twice: ambient `d = 3,584`
with dof-capped ridge, and a **PCA-`k` basis estimated from the unlabeled pool** (`k` selected on
dev). The PCA basis costs no labels, is house convention in this line (#1092 / #1739 report
"ambient / PCA-48"), and has a batched in-repo implementation
(`analysis/issue_763_vectorized.batched_ridge_predict_loco_pca`). Disagreement between bases is
reportable — it separates "no signal" from "the ambient estimator cannot reach it at this `n`".

## Model choice — stay on Qwen2.5-7B-Instruct

A four-surface survey of Qwen3.5 (2B/4B/9B/27B), Qwen3.6 (27B, 35B-A3B) and non-Qwen candidates
was run to test whether a stronger model would buy more spread. **It would not, and it would cost
a great deal.** Recorded here so the question is not reopened without new evidence.

*The incumbent is at or near the best available operating point on every surface:*

| surface | incumbent | position |
|---|---|---|
| QA | TriviaQA 0.648 / NQ-Open 0.345, reliability 0.94, ceiling ρ 0.97 | measured, excellent |
| Math | MATH 75.5 (GSM8K 91.6 ceiling-bound) | usable, level-stratifiable |
| MCQ | MMLU-Pro 56.3 | near-ideal |
| Code | pooled mean ≈ 0.535 across the core pool | near-ideal, both extremes populated |

*Newer models saturate.* Qwen3.5/3.6 at 4B and above: MMLU-Pro 79.1 → 86.2, GPQA-D 76 → 88,
HMMT 0.75 → 0.94, AIME-2026 ≈ 0.93, LiveCodeBench-v6 55.8 → 83.9. Ceiling is exactly where the
DV dies. Only Qwen3.5-2B sits below centre, and it is the weakest model in the family.

*Evidence coverage is worse, not better.* No Qwen3.5/3.6 model has any published TriviaQA,
NQ-Open, SimpleQA, GSM8K, MATH, HumanEval, MBPP, BigCodeBench or APPS number — the cards report
only MMLU-Pro/Redux, SuperGPQA, LiveCodeBench-v6 and agentic suites. Switching means trading a
measured 0.94-reliability surface for an unmeasured one.

*Three structural costs.* (1) **Hybrid attention** — every Qwen3.5/3.6 model is 3:1
linear-attention to full-attention (verified from `config.json`: 24/8 at 4B and 9B, 48/16 at
3.6-27B, 30/10 at 35B-A3B; Qwen2.5-7B is uniformly full attention). The residual stream is still
the bottleneck through which context reaches the answer, so the construct survives in principle —
but the last-prompt-token pooling convention is an *empirical* #1768 result worth ~0.20 R², and
the layer-selection curves from #1092/#1739 would all need re-establishing on a heterogeneous
layer stack. (2) **Env upgrade** — pinned `transformers 4.57.6` has no `qwen3_5` module and
`vllm 0.11.0` predates the family; upgrading in place risks the banked #1739 pipeline that makes
Phase 0 free. (3) **Thinking mode is default**, inflating generation 10–50× and relocating where
the answer sits inside the answer vector; no non-thinking numbers are published for any of them.

*If a second model arm is ever wanted*, the cheap option is **Qwen2.5-Coder-7B-Instruct** — same
architecture, same `d = 3,584`, same 28 layers, so the rig, pooling convention and layer story
all transfer with zero engineering, while code ability is materially higher (HumanEval 88.4,
MBPP+ 71.7, LiveCodeBench 37.6 vs the incumbent's 28.7). It would still need its own map fit and
its own activation capture. Not in scope for round 1.

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

Plus two **correctness-direction** arms on a one-parameter readout (user-approved, § Provenance):

| # | readout | arm |
|---|---|---|
| 1d | `⟨r_correct, v_C⟩` | **direction-context** |
| 2d | `⟨r_correct, M v_C⟩` | **direction-mapped** |

Six predictor arms. Holding the readout fixed makes the comparison across input representations
internally fair — every arm gets the same estimator, so a difference is attributable to the
representation. **Declared scope limit, to be stated in the writeup:** because no nonlinear
*readout* on `v_C` is run, the experiment cannot rule out that a nonlinear direct probe would
match or beat the MLP-mapped arm. The direct side is measured at its LINEAR ceiling, not its true
ceiling, and any "the map beats direct prediction" claim is scoped to linear direct readouts.
This is a known, accepted gap, not an oversight.

**The correctness direction `r_correct` — construction.** The persona-vectors analogue with the
contrast swapped from disposition to correctness: split a context's rollouts by whether the answer
was correct, average the answer activations of each group, subtract. **Matched WITHIN context**
(#1739's E2 construction), never pooled across contexts: pooling makes the direction partly a
"this is an easy/common topic" direction, since easy items differ from hard ones in topic and
phrasing. Restricting to within-context contrasts holds topic, wording and length fixed so they
cancel, then averages the per-context differences.

Feasibility is measured, not assumed — contexts with within-question spread (some of the K=5
rollouts correct, some not) in the banked #1739 data: **TriviaQA 3,642 of 15,993 (22.8%)**,
**NQ-Open 767 of 3,165 (24.2%)**, SimpleQA 309 of 4,015 (7.7%, already dropped). Ample for a
mean-difference, which needs far fewer samples than a 3,584-dimensional ridge.

**Why these arms earn their place in a slim roster:** they are the ONLY estimator in the roster
that is well-posed at small `L`. A ridge readout fits 3,584 parameters; at `L = 250` that is
hopeless, so without a direction arm the small-`L` end of the H2 crossover curve is a race between
two badly-posed ridges and its ordering is mostly estimator noise. A projection fits one
parameter and is well-posed at any `L`. Declared limitation: a direction arm differs from a ridge
arm in BOTH representation and estimator, so a difference between the two families is not cleanly
attributable to either — read them as a per-`L` envelope ("best achievable at this label budget"),
not as a controlled representation contrast. Zero marginal data cost: rollouts, correctness labels
and answer activations are all banked.

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
| `f_U = 0` | **generic only** (WildChat/LMSYS) | the banked configuration; strongest deployment story if it wins |
| `f_U = 0.5` | **generic + target-surface**, half each | the money cell: can unlabeled in-domain pairs stand in for labels? |
| `f_U = 1` | **target-surface only** | in-domain ceiling for the map channel |

**Fixed `|U|`, never addition.** The three cells hold the total unlabeled pool constant
(target: `|U| = 8,000` pairs, sized by the planner) — target-surface pairs *replace* generic
pairs. Adding rather than replacing confounds composition with quantity and voids every cell. The
additive variant is run once as a clearly-labeled realistic-deployment contrast, never as the
comparison. The banked `U = 18,793` generic map stays as a reference row outside the
matched-budget protocol.

**Disjointness.** The map's target-surface portion is **disjoint** from the readout's labeled
contexts and from every eval rung — primary. The overlapping configuration (no label leakage,
since the map never sees labels) is the more realistic deployment shape and is run as the
variant, per #1739's ruling.

**Mechanism diagnostic** (inherited from #1739 §4b): report map held-out R² **and** kNN retrieval
on each eval rung as a function of `f_U`, alongside the prediction ρ. If ρ rises with `f_U` and
map quality rises in step, there is a mechanism; if ρ rises while map quality does not, something
else is driving it.

**Cost is low because the unlabeled pairs already exist.** The target-surface "unjudged" pool is
the same contexts with their labels withheld — QA's are banked in #1739's capture store, and the
new surfaces' come free from the Phase 1 rollouts. New map fits are dense solves over banked
activations: 3 cells × 4 surfaces × 28 layers × 2 map families ≈ 672 fits, the many-cell
dense-factorization case (#823) — batched solver, never a per-cell loop.

**Scope control:** run `f_U` at **three `L` anchors, not the full `L` sweep** (#1739's own scoping
of this factor), and only on surfaces that clear the Result 0 spread gate.

## Splits — every surface has a locked held-out test set

**Standing requirement for this task (user directive): there is always a held-out test set.**
Cross-validation on the training pool is a diagnostic, never the headline.

Every surface is partitioned **at group level** (question entity / MMLU-Pro subject / problem id;
groups never straddle a split) into three parts:

| split | used for | touched |
|---|---|---|
| **train** | fitting readouts; the `L` sweep draws its labels from here | freely |
| **dev** | ALL selection — layer ℓ, ridge λ, PCA rank k, MLP epochs, arm choice, whitening rank | freely |
| **test** | the reported headline numbers | **once**, after selections are frozen |

Rules that make "touched once" real rather than aspirational:

- Every hyperparameter and every selection is made on train+dev only. Nothing on test.
- The frozen selections are written to a committed `selection.json` **before** the test read, so
  the claim is auditable after the fact rather than asserted.
- The map's unlabeled pool — including the `f_U > 0` target-surface slice — excludes all test
  groups in the primary configuration. Whitening and PCA statistics likewise come from the
  unlabeled/train pools only; no transductive refit on test.
- With 28 layers × 4 arms × 3 `f_U` cells, a max-over-selection read on test would be
  selection-on-test. Layer and arm are frozen from dev; a max-over-anything is reported on dev.
- Shift rungs 1 and 2 are held-out test sets by construction and inherit the same rule.

## Evaluation ladder and metrics

Group-level folds within train/dev — never pointwise. Rungs, in increasing shift, each read once
against frozen selections:

- **rung 0** — the locked test split of the training surface
- **rung 1** — cross-dataset within family (TriviaQA → NQ-Open; MATH levels 1–3 → 4–5;
  MMLU-Pro subject holdout; HumanEval+MBPP → BigCodeBench+LiveCodeBench)
- **rung 2** — cross-family (recall QA → math / MCQ / code). "Target surface" for `f_U` always
  means the **training** surface here.

Metrics: Spearman ρ (matches #1739 so the H3 comparison is commensurable), held-out R², and AUROC
on the binarized DV for legibility. Paired bootstrap intervals over identical realized folds per
`(L, seed)` so every arm comparison is paired. Permutation null over the max across arms/layers.

**Result 0 (gates, before any headline).** (a) Zero-pile / one-pile fraction and true-rate SD per
surface and rung, by the § Measured spread recipe; a rung with either extreme above ~90% is
dropped, never drawn as a zero bar. **Already discharged for QA.** (b) Split-half / beta-binomial
reliability per surface — **already 0.93–0.94 for QA**. (c) ρ(correctness rate, #1739 fabrication
rate) on the QA rungs, so the novelty over #1739's hallucination arm is explicit rather than
assumed. (d) Map reconstruction quality (held-out R² + kNN acc@k) per surface, per map family,
per `f_U` cell, *before* the readout — so a null readout is attributable to map degradation
rather than to absent signal.

## Phasing and compute (planner to size properly)

- **Phase 0 — banked QA, 0 GPU-h for data.** The correctness DV already exists:
  `eval_results/issue_1739/dv_dataset/hallucination/labeling.json` carries `fractions.correct`
  for 23,188 contexts. `scripts/issue1739_fits.py` accepts `--dv-json`, so the ladder is a DV
  swap over existing arms. **CPU pod, not the VM** — the activation store is a single 70 GB tar
  (`issue1739_ctxmap/capture_store/hallucination_labeling`), over both the ~10 GB download rule
  and the 50 GB VM-footprint gate.
- **Phase 1 — new surfaces (GPU).** ~12.5k math + ~12k MCQ + ~6k code contexts × 5 rollouts,
  plus a teacher-forced capture pass for `v_C` and `v_A`. **Pilot-gated**: a 200-problem
  zero-pile/one-pile pilot per surface (and per candidate code sub-pool) runs FIRST and decides
  the final pools — in particular whether APPS-introductory and APPS-interview are admissible,
  and the APPS flaky-test rate under our harness. Code generations are long; size accordingly.
- **Phase 2 — fits.** 4 surfaces × 4 predictor arms × 28 layers × `L` sweep × 3 `f_U` cells (at
  3 `L` anchors only) × seeds × group folds × 2 bases, plus ~672 map fits. Ridge readouts stay on
  a CPU lane; **the MLP-map fits move off pure CPU** — iterative-optimization fits are GPU-worthy
  and the many-cell MLP-map battery must run through the batched multihead path
  (`analysis/vectorized_mlp_skill.py`, 50–100×), never a per-cell loop. Ops arithmetic and a
  measured 1-cell pilot wall go in plan §9 before dispatch.
- **Estimator well-posedness.** The inherited `issue1739_fits.ridge_gcv_predict_per_target` path
  is pure GCV and is **banned** at `n < d` (#1887): every fit routes through the dof-capped
  selector (`dof_capped_ridge_multi_y` / `dof_capped_ridge_fit_all`, `GCV_DOF_CAP = 0.9`), and
  every fit reports selected λ and effective degrees of freedom.
- **Judge spend ≈ $0.** Every surface is programmatically verified.

## Reuse (what is already banked)

- Correctness labels + rollouts for the whole QA surface (#1739), including the three-way
  correct/abstained/fabricated split.
- Fitted generic-pool maps, linear / MLP / kernel, at three unlabeled budgets — the `f_U = 0`
  reference cells.
- Activation capture store for the QA contexts — also the source of QA's `f_U > 0` pairs.
- The fit/arm/fold/bootstrap pipeline (`scripts/issue1739_fits.py`, `issue1739_final_fold.py`,
  `experiments/issue_1739/arms.py`), the MLP-map implementation, and the #825 vectorized cores.
- #1739's behavior numbers, on the same model, for the H3 persona-vs-knowledge comparison.

New code: generation + capture for three surfaces, four programmatic verifiers, the correctness
DV builder, the `f_U` pool-composition harness, and a code-execution sandbox.

## Risks and inherited caveats

1. **Novelty vs #1739's hallucination arm.** correct = 1 − abstained − fabricated, and fabrication
   rate was already predicted. Result 0(c) quantifies the overlap; math / MCQ / code are where
   the novelty is structurally protected.
2. **`n < d`** — resolved for three surfaces, contingent on LeetCodeDataset dedup for code. See
   § The `n < d` regime.
3. **Labels are cheap on these surfaces by construction** (programmatic verification), so the
   label-efficiency result is an *estimator-level* finding obtained by withholding labels we
   actually have, transferred by analogy to settings where labels are expensive. State it; do not
   narrate it as a realized cost saving.
4. **Map domain shift** — the banked maps were fit on generic chat; Result 0(d) makes a null
   attributable, and the `f_U` factor tests directly whether domain shift is the binding
   constraint. Math and code contexts are far off WildChat/LMSYS.
5. **Difficulty confound** — arm 8 is the control; a probe reading item difficulty is a real
   mechanism but a different claim from "the model knows what it knows".
6. **Contamination** — TriviaQA (2017) and NQ-Open (2019) are plausibly in Qwen's pretraining;
   inherited caveat from #1739, and a reason the math/code surfaces matter. LiveCodeBench's
   release windows give a partial contamination handle on the code surface.
7. **Language intrusion** — 6.2% of the banked hallucination rollouts carry CJK intrusion
   (#1739 intrusion audit); carry the same scan and recount.
8. **Answer-vector pooling on long CoT, now on three of four surfaces.** MMLU-Pro was chosen for
   its 10-option floor, but the paper reports CoT *beats* direct answering on it (unlike original
   MMLU) — so MCQ generations are CoT-shaped too. Either accept CoT-shaped answer vectors on
   math, code and MCQ, or score MMLU-Pro by direct option log-prob and forgo the on-policy
   sampled DV there. **Open fork for the planner.** Capture both token-mean and last-answer-token
   throughout; declare which is primary per surface.
9. **Grid growth.** MLP map × `f_U` cells × 2 bases multiply Phase 2. Scope controls: `f_U` at
   three `L` anchors only, and the full `L` sweep on a planner-selected subset of surfaces.
10. **Code pool heterogeneity.** Pooling five code benchmarks introduces a benchmark-identity
    confound — a probe can learn "this is a HumanEval-shaped prompt", which correlates with pass
    rate. Include benchmark identity in the surface-feature control and use benchmark-level
    groups in the fold structure.

## Open items for the planner

- ~~**The 200-problem spread pilot is the first execution step**~~ — **DONE, PASS**
  (2026-08-19, `epm:progress` `[spread-pilot] COMPLETE`; results committed at
  `eval_results/issue_2388/spread_pilot/`, producer `scripts/issue2388_spread_pilot.py`,
  harness control `scripts/issue2388_code_control.py`). Do NOT re-run it. Measured at the exact
  production protocol (K=5, temperature 1.0, top_p 1.0):

  | benchmark | n | mean | SD(true) | reliability | ceiling ρ | zero-pile | one-pile | verdict |
  |---|---|---|---|---|---|---|---|---|
  | HumanEval | 164 | 0.815 | 0.316 | 0.906 | 0.952 | 8.5% | 69.5% | ADMISSIBLE |
  | MATH-500 | 200 | 0.782 | 0.320 | 0.882 | 0.939 | 8.5% | 64.0% | ADMISSIBLE |
  | MBPP | 150 | 0.673 | 0.415 | 0.946 | 0.972 | 22.0% | 58.0% | ADMISSIBLE |
  | MMLU-Pro | 191 | 0.586 | 0.373 | 0.870 | 0.933 | 20.4% | 36.6% | ADMISSIBLE |
  | BigCodeBench | 149 | 0.165 | 0.309 | 0.918 | 0.958 | 73.2% | 9.4% | INCONCLUSIVE |

  Every surface the experiment needs clears the bar (math, MCQ, code via HumanEval+MBPP; QA
  already established from the banked #1739 rollouts above). Reliability 0.87–0.95 confirms K=5
  resolves real item difficulty rather than rollout noise, so **K=5 stands**. Nothing saturates,
  so there is no case for switching to a stronger model.

  Two results the planner must carry forward:

  1. **BigCodeBench is unusable until its official environment is provisioned.** Its number is
     harness-contaminated, not a model measurement: the canonical-solution positive control passed
     only 13/25 of BigCodeBench's OWN reference solutions on the pilot pod
     (`ModuleNotFoundError: sklearn / matplotlib / flask`), against 25/25 for both HumanEval and
     MBPP. BigCodeBench spans 139 libraries by design. Either provision its pinned dependency set
     and re-measure, or drop it and re-do the § `n < d` code-pool arithmetic without its 1,140
     items. Decide this in the plan; do not inherit the 0.165/73.2% figures.
  2. **HumanEval has a 14.6% all-K-identical rate** (MBPP 8.0%; math and MCQ both 0.0%) — those
     items produce one deterministic completion under sampling, so their per-item rate is 0 or 1
     by construction rather than by difficulty. Not disqualifying at reliability 0.906/0.946, but
     report it per-benchmark rather than pooling it away.

- **Still pilot-gated, NOT covered by the run above:** APPS-intro and APPS-interview
  admissibility; LeetCodeDataset ↔ LiveCodeBench dedup; APPS flaky-test rate. These were scoped
  out of the first pilot, which measured the four core benchmarks plus BigCodeBench.
- **Risk 8's MMLU-Pro fork** — CoT-shaped answer vectors vs direct option log-prob.
- Whether the `L`-sweep runs on all four surfaces or on QA + one new surface, with the others at
  full-label only.
- ~~Whether the correctness-direction arms are added~~ — **RESOLVED: in** (user, § Provenance
  round 5). Specified in § Arms.
- Fuller literature grounding pass (`/deep-lit-review`). Kadavath / Kuhn / Farquhar are already
  in hand from the survey; the pass would cover the probing-for-correctness line properly.
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
trained on specific data) -- similar to behavior prediction experiment?"* — the explicit user
request the linear-by-default standing rule requires for the nonlinear arm, and the origin of
§ Map-pool composition.

Third round, verbatim: *"only add mapped MLP. there should always be a held out test set. let's
discuss the n < 3584 problem"* — direct-MLP and oracle-MLP readouts proposed as a fairness
pairing and **declined**; readout family fixed at ridge, scope limit declared in § Arms. The
locked held-out test set is a standing requirement for this task.

Fourth round: *"does running a stronger model help? maybe one of the qwen3.6s"* → *"get a subagent
to find the model with the greatest spread across these tasks"*. Four parallel surveys were run
(QA, math, MCQ, code). Outcome recorded in § Model choice (stay on Qwen2.5-7B-Instruct), § Surfaces
(SimpleQA/GSM8K/MMLU/ARC-C/GPQA/CodeContests dropped, MMLU-Pro adopted, code pool assembled), and
§ Measured spread (the criterion corrected from "mean near 0.5" to between-context dispersion,
measured directly from banked rollouts).

Fifth round, verbatim: *"okay add the correctness direction arms"* — the two one-parameter
projection arms (1d / 2d in § Arms) are IN. This is the user decision that was pending through
rounds 3-4; the arms were proposed as the only estimator well-posed at `L = 250` and approved on
that basis.
