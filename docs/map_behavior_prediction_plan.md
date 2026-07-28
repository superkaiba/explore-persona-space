# Predicting behavior expression through the context→answer map

**Status:** draft plan for review (not yet filed as a task). Drafted 2026-07-27, revised
after a four-way dataset survey (2026-07-28).

---

## 1. Question and claim

Persona-vector monitoring (arXiv 2507.21509) predicts behavior expression by projecting the
persona vector `v_B` onto the **context** vector. This is a datatype mismatch: `v_B` is a mean
over **answer** activations. The correction is to apply our learned context→answer map `M`
first and project onto the predicted answer vector:

```
score(x) = ⟨v_B, M(x)⟩          instead of          score(x) = ⟨v_B, x⟩
```

A direct regression from context to expression asymptotically upper-bounds any function of the
context, `⟨v_B, M(x)⟩` included. So the claim is **not** "we beat direct regression at scale":

> **Primary claim.** Unlabeled context→answer structure is a useful *prior* for behavior
> prediction. Map-based methods reach a given accuracy with far fewer behavior labels than
> direct regression, and the advantage **grows with distribution shift**.

Quantitative headlines: the **crossover** (labels direct regression needs to overtake the
label-free map arm) and the **degradation slope** across the shift ladder.

**Pre-registered secondary hypothesis.** Labels bought from one elicitation mechanism buy
in-distribution accuracy at the cost of OOD robustness — expect finetuned > label-free at
rung 0, gap narrowing or inverting at the outermost rung.

---

## 2. Matched-budget protocol

At every point on the scaling curve **every** method receives the identical budget
`(U unlabeled context→answer pairs, L labeled examples)` and spends it as its estimator likes.
This makes the experiment an **estimator comparison at fixed data budget**.

- **Baselines get `U`.** Direct regression fits on `U`-whitened context vectors
  (`z = Σ_λ^{-1/2}(x − μ)`, `μ, Σ` from the `U` unlabeled contexts; shrinkage or truncated-PCA).
  Activation space is strongly anisotropic, so isotropic ridge on raw activations is a strawman
  — the map gets that second-order structure for free. Whitening is fit on the training-side
  pool only and applied unchanged to every eval rung (no transductive refit).
- **Label-free methods get `L`** — a readout (affine/isotonic) plus layer selection.
- **Tuning comes out of `L`** via nested CV. Layer choice is a hyperparameter.
- **Identical realized folds** per `(L, draw seed)` for every method — not merely the same
  procedure. Makes all comparisons paired. Folds are **group-level** (conversation, persona,
  jailbreak family, question entity), matching the eval ladder.
- **Refit on full `L` after selection**, uniformly.
- **Matched search budget** — fix or report the config count per arm; report selected values
  per method per `N` (a boundary selection means the grid was too narrow).

Three decoupled axes; report two 1D slices, not a grid:

| Axis | Range | Feeds |
|---|---|---|
| `U` unlabeled map data | 250 → 50k WildChat pairs | map quality only |
| `L` labeled data | 250 → ~16k judged | all label-consuming arms |
| PV extraction | fixed per behavior (E1) / drawn from `L` (E2) | projection arms |

---

## 3. Method roster

All arms run in **both** prefix→answer and context→answer variants.

**Context-side (no map)**
1. Project `v_B` on the context vector — the paper's own method.
2. Same with a **context-native** direction — the steel-manned version. Free: `generate_vec.py`
   already emits `prompt_avg_diff.pt` and `prompt_last_diff.pt` alongside `response_avg_diff.pt`.
3. Identity + learned bias (`x + b`) → project.
4. Direct linear regression on `U`-whitened context vectors.
5. Direct nonlinear (MLP) regression on `U`-whitened context vectors.

**Map-based** (linear and nonlinear; map fit on `U` unlabeled pairs)
6. Map → project `v_B`. Label-free in the direction; the headline arm.
7. Map → regression on *predicted* answer vectors (frozen map, labeled readout).
8. Map → regression on *real* answer vectors.
9. **Pretrain-then-finetune**: map fit on `U`, fine-tuned end-to-end through the readout on `L`,
   regularized toward pretrained weights (L2-SP / low LR). Must degenerate exactly to arm 6 at
   `L=0` — hard sanity check.
10. **Stacked**: two-parameter combiner on `L` over (arm 6, arm 4/5).

**Oracles** (privileged information; dashed reference lines, outside the budget protocol)
11. Project `v_B` on the true answer vector. 12. Regression from the true answer vector.

**Controls**
13. Shuffled-map → project. 14. Shuffled-pretrain → finetune. 15. Text-only (sentence-embedding
of the raw prompt → ridge, same `L`). 16. Trivial surface features (length, moderation score,
question type) correlated with expression.

---

## 4. Persona-vector extraction — two regimes

The released `safety-research/persona_vectors` repo ships trait JSONs for **evil**,
**hallucinating**, and **sycophantic** (of 7 traits), each with 5 contrastive pos/neg
system-prompt pairs, 20 extraction + 20 disjoint eval questions, and a 0–100 rubric, plus the
full generator template (`data_generation/prompts.py`) and verbatim trait descriptions in the
appendix. **The vectors themselves are not released** — regenerate via `generate_vec.py`.
Recipe: 10 rollouts per question per arm, judge-filter (pos > 50 / neg < 50, malformed /
REFUSAL **dropped** both arms with per-arm counts reported), response-averaged activations,
diff-of-means per layer. Their judge is `gpt-4.1-mini-2025-04-14`; our single standing
deviation is `claude-sonnet-4-5-20250929`.

- **E1 — paper-faithful synthetic extraction.** Because extraction is synthetic and all train
  and eval data is real, **the direction is OOD for everything downstream by construction**.
  The PV set is extraction-only: its 20 eval questions never appear in train or eval.
- **E2 — matched-pair natural extraction (answer space).** Sample `K` answers per training
  context, keep contexts with genuine within-context score spread, compute
  `mean(high) − mean(low)` **within each context**, average across contexts. Holding the query
  fixed cancels topic, length, and register. **Zero marginal cost** — the `K` samples are
  already mandated by the DV design (§6), so the matched pairs come free.
- **E2p — pooled natural extraction (answer space), run as a contrast.** Diff mean answer
  activations of the top-eliciting against the bottom-eliciting labeled contexts. This drops the
  paper's own control (they hold the 20 questions fixed and vary only the instruction), so the
  two arms differ in topic, register, and length as well as disposition and the direction is
  substantially a "harmful topic" direction. Run it anyway: it is a few lines over data already
  collected, and **the E2-vs-E2p gap measures how much of a naively-extracted natural direction
  is topic rather than disposition** — a reportable number, not just an internal check.

E2 consumes labels, so it competes on the same `L` as the regression arms; a diff-of-means is a
one-direction estimator and should dominate ridge at small `L`.

**Context-space directions are a separate case.** For the arm-2 context-native direction, top-
vs-bottom-eliciting pooling is not merely acceptable, it is the only available estimator — a
context has one activation, so there is no within-context variation to match on. The matched-pair
objection applies to answer-space directions only.

**Selection caveat for every pooled/tail variant:** select contexts on the `K`-sample mean, never
a single draw. Extreme-tail selection on a noisy per-context estimate partly selects for noise,
and under the compliance-stability inverse correlation the noisiest contexts are the
high-expression ones — the naive tails would be enriched for measurement error exactly where it
hurts.

**Fallbacks if matched pairs are thin** (a behavior near-deterministic per context): topic-matched
pooling (cluster contexts, pair high/low within cluster), or residualize the pooled diff against
the top-`k` principal components of the unlabeled activation distribution — free under the
matched-budget protocol, since `U` is already spent. Both are weaker than true within-context
matching, both clearly better than raw pooling.

Also run the covariance-whitened (LDA-style) variant of each — two lines, often a better
direction. **Which extraction source transfers best across the ladder is a reportable
sub-result** in its own right.

### 4b. Eliciting-data composition — where must behavior-specific data enter?

Behavior-specific information can enter at **three** points: the map's unlabeled pool `U`, the
labeled predictor pool `L`, and the PV extraction set. §4 varies the third (E1 vs E2). The other
two are made explicit factors here rather than silently hard-coded (the earlier draft fixed
`U` = all-generic and `L` = all-eliciting and never tested the alternative).

**Composition at FIXED budget, never addition.** Eliciting pairs *replace* generic pairs so `|U|`
is constant; likewise `|L|`. Otherwise composition is confounded with quantity and every cell is
void. The factor is an eliciting fraction: **`f_U ∈ {0, 0.5}` × `f_L ∈ {0, 1}`**.

| | `f_L = 0` (generic labels) | `f_L = 1` (eliciting labels) |
|---|---|---|
| **`f_U = 0`** (generic map) | all-generic — likely degenerate for evil (no spread anywhere); informative as such | the §5 default |
| **`f_U = 0.5`** (map sees eliciting) | **the money cell** — can *unlabeled* eliciting data substitute for labels? | both channels informed |

The money cell carries the practical payload: unlabeled eliciting data is cheap (prompts +
generations, no judging) while labels are the expensive part. If the map only needs to have *seen*
that region of activation space, the deployment story is much stronger than "you need judged
examples of the behavior."

**Disjointness decision:** `U`'s eliciting portion is DISJOINT from `L`'s contexts, so the two
axes are attributable separately. Overlap is not leakage (the map never sees labels) and is the
more realistic deployment configuration — note it as the variant, run disjoint as primary.

**Ties to the map-degradation diagnostic (§7):** report map held-out R² and kNN retrieval per eval
rung as a function of `f_U`. If prediction ρ rises with `f_U` *and* map quality on eliciting
contexts rises in step, there is a mechanism; if ρ rises while map quality does not, something
else is driving it.

**Scope control:** run this factor at **three `L` anchors, not the full scaling grid**, and only
on the behaviors with the clearest measured spread from gate 1. Cost: no judge increment (`U` is
unlabeled by definition), +1–2 H100-h for generating answers on eliciting contexts destined for
`U` but held out of `L`, plus the extra fits.

*Synergy worth exploiting:* compliance and stability are inversely correlated (ρ = −0.47 to
−0.70, arXiv 2512.12066), so the high-yield contexts for evil are also the high within-context
variance ones — exactly what E2's selection criterion wants.

---

## 5. Data

### 5.1 Constraints
Train and eval sets are **real human-authored text** (organic or human-written), never
LLM-generated or programmatically templated. Eval sets are **independent of the training set in
construction mechanism AND authorship**, and OOD for the persona vector (satisfied by E1
construction). This eliminates WildJailbreak, OR-Bench, CoCoNot, FalseReject, PHTest,
SALAD-Bench, emergent_plus, PopQA, Head-to-Tail, EntityQuestions, KG-FPQ, HalluLens, LongFact,
and FActScore's prompt set.

### 5.2 Map training and the neutral distribution
`allenai/WildChat-1M` (837,989 conversations, ODC-BY, ungated) serves two roles: the **map's
unlabeled context→answer training pool** and the **neutral prompt distribution**. Its 2024-07
toxicity purge and 2024-10 PII purge — which cost ~202K conversations versus the paper's
1,039,785 — make it unsuitable for mining toxic contexts but *better* as a neutral baseline.
`WildChat-1M-Full` (gated, manual approval + justification) retains the toxic tail and is a
nice-to-have fallback, not on the critical path. **The map never sees behavior-eliciting data
or any eval rung** — it must extrapolate to hostile contexts from clean training data. That is
a strength of the behavior-agnostic claim, but state it rather than let it be discovered.

### 5.3 Per-behavior train / OOD-eval pairs (all real, all verified accessible)

| Behavior | Train | OOD eval | Real topic control |
|---|---|---|---|
| **Evil** | `TrustAIRLab/in-the-wild-jailbreak-prompts` (1,405 scraped Reddit/Discord) × `TrustAIRLab/forbidden_question_set` (390 = 13 scenarios × 30) | `Anthropic/hh-rlhf` **red-team-attempts** (38,961 multi-turn dialogues, paid crowdworkers, 2022, ships per-dialogue harmfulness ratings) | ToxicChat's **542 toxic-but-not-jailbreaking** rows + the in-the-wild corpus's own "regular" prompts (5.7K / 13.7K) |
| **Sycophancy (trait)** | affordance-stratified WildChat advice / validation-seeking turns — **screen must be built (see below)** | **ELEPHANT** (Reddit AITA + crowdsourced advice; OEQ 3,027, AITA-YTA 2,000, CC0) + **PRISM** (8.0K conversations, recruited demographically-stratified participants) | ELEPHANT's non-affording items; WildChat technical turns |
| **Hallucination** | **TriviaQA** `rc.nocontext` (138.4K/17.9K/17.2K; trivia-enthusiast authored) | **NQ-Open** (87.9K/3.6K; real Google queries, cc-by-sa-3.0). Secondary: **SimpleQA** (4.3K, MIT) | answerable-and-known subset |

Independence rationale per pair — mechanism, authorship, and era all differ: forum-shared
persona-wrapper attacks vs paid freeform crowdworker red-teaming (2023 vs 2022, single-turn
templated vs multi-turn freeform); real Reddit moral-judgment posts vs organic chatbot
advice-seeking; quiz-league questions vs search-log queries (2017 UW vs 2019 Google, **no model
in the difficulty loop on either side**, neither question set Wikipedia-entity-derived).

**Affordance screen for sycophancy — we build it; nothing ships it.** Verified: WildChat carries
text plus language/geo/model metadata plus safety annotations (`openai_moderation`,
`detoxify_moderation`, `toxic`) and **no topic, intent, or task-type taxonomy**; the in-repo
`sycophancy_neutral_v1/v2` (40/40) and `sycophancy_claims_v1` (50) are hand-built prompt banks,
not corpus labels, and `behavior_testbed_545/corpora.py` is Sonnet-generated (excluded by §5.1).
Build it as the sycophancy instance of the §5.3 retrieval cascade — the role moderation scores
play for evil: ~20 hand-written seed advice/validation-seeking prompts → cosine retrieval against
the one-time embedding pass (free at the margin, already needed for the other signals) → keyword
prior → small-model zero-shot screen over the top ~50–100k candidates. **~$20–50 plus a few hours
of implementation**; validation folds into the gate-1 calibration pilot. **Because the screen is
ours, it is a labeling artifact the predictors can learn** — multi-signal quotas (never one
composite score), keep the high-screen/low-judged disagreements as hard negatives, and report
ρ(screen, judged sycophancy); near 1.0 means the training set cannot separate the two.
*Cheaper fallback if the screen becomes a time sink:* train on ELEPHANT's OEQ split (3,027 real
crowdsourced advice queries, affordance-guaranteed by construction) and evaluate on ELEPHANT's
AITA slices + PRISM — accepting that the OEQ↔AITA rung shares an author group and is therefore a
partial shift (collection mechanism differs, authorship does not); the PRISM rung stays fully
independent either way.

**Effective-subset selection for evil.** The in-the-wild corpus ships no success flag
(schema: `platform, source, prompt, jailbreak, created_at, date, community_id, community_name`
— `jailbreak` classifies prompt *type*, not efficacy) and the paper's per-prompt ASR is
unreleased. Two derivable proxies: **presence in both temporal snapshots** (2023-05-07 and
2023-12-25 — surviving ~7 months of patching is the paper's own efficacy marker) and
`community_id` propagation weighting. Prioritize the persona-roleplay slice.

### 5.4 Contamination map — pairings that would silently break the OOD claim

- **AdvBench is a root contaminant**, reappearing verbatim in JBB-Behaviors, StrongREJECT (25),
  SALAD-Bench (359), and the JailBreakV/RedTeam-2K lineage.
- **hh-rlhf is the second root** — AttaQ, Nemotron/Aegis v2, BeaverTails, and SALAD-Bench
  (4,843) all derive from it. **Since hh-rlhf is our evil eval, none of those four may be
  training data.**
- **MHJ is HarmBench-seeded** (its only data file is `harmbench_behaviors.csv`) — human attack
  wrappers over non-independent behaviors. Dropped.
- **ToxicChat shares the Vicuna demo with LMSYS-Chat-1M** — same collection process. ToxicChat
  is a same-family control, never an independent eval rung against LMSYS-derived training data.
- **WildJailbreak's adversarial arm was built from tactics mined from the in-the-wild corpus** —
  a partial shift, not a clean one (moot under the real-text constraint).
- **`Anthropic/model-written-evals` ↔ `EleutherAI/sycophancy`** are the same data; SycophancyEval
  `answer` shares an author and device with the Perez evals — one family.
- **SimpleQA Verified is a 1,000-item subset of SimpleQA** — never pair them.
- StrongREJECT ships a `source` column; filtering to `source == "custom"` yields 221 genuinely
  independent prompts, if a curated arm is ever wanted.

### 5.5 Corpus shift ladder (for the generic/neutral arm)

WildChat-1M → LMSYS-Chat-1M → PRISM. The WildChat/LMSYS near-duplicate suspicion is
**refuted**: 2.54 vs 2.02 avg turns, and decisively **295.6 vs 69.8 avg user tokens (4.2×)** —
a free-GPT-4 utility service vs a model-comparison arena, OpenAI-only vs 25 models. A moderate
real shift; PRISM is the maximal one (the only corpus whose population was *designed*). The
4.2× length gap will dominate any length-sensitive geometry read and must be controlled
explicitly. LMSYS is **not** toxicity-filtered and needs a separate toxicity pass before it
counts as neutral.

**Disjointness contract** per behavior: persona pool, jailbreak family/source, conversation ids,
artifact texts, question entities, plus MinHash/embedding near-dup filtering between train and
every rung. Held out by **conversation**, not turn.

---

## 6. Dependent variables

- **Primary:** graded 0–100 judge, `claude-sonnet-4-5-20250929`, `K`≥3 draws at temperature > 0
  (5 preferred), mean-aggregated. `K`≥3 is not optional: 18–28% of harmful prompts flip
  refuse/comply across seeds and temperatures, and single-shot agrees with multi-sample ground
  truth only 92.4% of the time (arXiv 2512.12066). Malformed / REFUSAL / out-of-range returns
  **dropped, never coerced**; per-arm drop rate reported, split content-drops vs
  transport-losses. `max_tokens` ≥ ~300 for reason-then-score rubrics. Batch API, rubric-keyed
  cache.
- **Hallucination gets the strongest DV**: gold answers give correct/incorrect by exact match
  against alias lists with **no judge at all**; the judge only separates confident fabrication
  from abstention *among incorrect answers*. Report the three-way rate (correct / abstained /
  fabricated).
- **Secondary continuous companion:** teacher-forced fixed positive/negative pool margin (#722
  recipe) — the same fixed answer pools under every context, so no selection-on-outcome bias.
  Primary target wherever the judged rate floors; validated against the rate wherever spread
  exists.
- **Evil auxiliary:** refusal judged as a separate DV on the same contexts. Report
  ρ(predicted-evil, predicted-refusal) and evil ρ **conditioned on non-refused responses**. If
  the signal vanishes among compliant responses, we built a refusal probe.
- **Expectation vs realization:** context methods predict the *expected* expression. Score them
  against the per-context mean of `K` samples; answer methods per-answer. Report the
  **split-half ceiling** with design-aligned item-matched splits. **For evil this is
  load-bearing, not hygiene** — expression is substantially stochastic given context, so the
  ceiling may be genuinely low and a modest ρ could be near-ceiling. Without it the evil numbers
  are uninterpretable.
- **Construct pinning (sycophancy):** the trait (flattery, ingratiation, excessive validation),
  **not** epistemic capitulation — declared out of scope in the Goal. Same trait description and
  rubric at extraction and evaluation. Empirically supported: sycophantic agreement and praise
  sit on distinct linear directions that steer independently (arXiv 2509.21305), so a direction
  extracted on one must not be assumed to read the other. The **Social Sycophancy Scale**
  (arXiv 2603.15448, N=877 human raters, 3 factors) is an off-the-shelf validated instrument for
  the judge–human audit.

---

## 7. Metrics and analysis

- **ρ** and **AUROC** per method × behavior × rung × `N`; AUROC is what monitoring cares about.
- **Paired bootstrap** over shared eval contexts for every method delta.
- **Scaling curves vs `L`**: label-free arm flat, direct regression rising, finetuned-map
  interpolating; read the crossover as the horizontal gap.
- **Degradation slope** across rungs — the headline robustness result.
- **Resample the labeled draw** ≥5× per `N`, plot mean ± CI over draws. At `L`=250 the dominant
  variance is *which* examples were drawn.
- **Map-degradation diagnostic**: held-out R², identity+learned-bias baseline, and kNN retrieval
  (euclidean + cosine, chance = k/n stated) per rung — required by the standing
  mapping-baselines rule and it lets prediction failures be attributed to map failures.
- **Report the DAN and hh-rlhf rungs separately, never averaged.** The DAN slice is
  persona-instruction elicitation — mechanistically the same family as PV extraction, just real
  instead of synthetic — so PV projection may look artificially strong there. hh-rlhf is the
  honest OOD test.
- **Sycophancy robustness:** ρ within affordance stratum, or with topic clusters partialled out.
- **Evil:** within-stratum ρ, never pooled across harmful/benign (pooled ρ is mostly the trivial
  between-stratum gap).
- **Pre-registered primary comparison:** map→project vs the **steel-manned** context projection
  (arm 2) at the outermost rung. ~16 arms × 3 behaviors × 4 rungs × 6 `N` is enough cells to find
  anything; everything else is secondary with multiplicity correction.
- Plots over tables; one color = one meaning across every figure; low-level per-unit plot
  alongside every aggregate.

---

## 8. Preconditions (gates before committing GPU)

1. **Yield pilot — mandatory.** No published compliance number exists for any 7–8B open instruct
   model on the in-the-wild corpus, hh-rlhf red-team-attempts, or the ToxicChat jailbreak subset.
   The only transferable anchor is **Qwen-2.5-7B at 81.3% refusal → ~18.7% compliance** on 876
   deduplicated AdvBench+HarmBench prompts (arXiv 2512.12066) — an optimistic ceiling, since
   in-the-wild prompts were written against 2023-era guardrails. Judge ~300 contexts per
   behavior per candidate set at `K`≥3 and measure the realized expression histogram before
   committing. Cost ~$50–100 and ~1 GPU-h per behavior.
2. **Pre-registered spread floor + fallback.** Floor: inter-context SD ≥ 10 on 0–100 and < 80% of
   contexts in the bottom bin. If no real set clears it for evil, the honest conclusion is that
   on-policy evil expression is not measurable in real data with this model, and the
   teacher-forced margin becomes that behavior's primary DV.
3. **Artifact-reuse check.** Resolve whether prior WildChat/LMSYS activation stores (#722, #779,
   #952, #1092) exist on HF and are reuse-fit — the single biggest swing on the GPU estimate.
   In-repo code reuse is already confirmed: `analysis/mapping_baselines.py` (identity+bias, kNN),
   `analysis/vectorized_mlp_skill.py` (batched MLP — the nonlinear fits must route through it),
   `experiments/issue_779/fit_h.py` + `metrics.py` (map fitting), and the #763 predictor stack
   (`graded_judge`, `nonlinear`, `pca`, `reliability`, `vectorized`). Banks already present:
   `betley_main8_v1`, `broad_em_neutral_v1`, `advbench_v1`, `strongreject_v1`,
   `sycophancy_claims_v1`, `sycophancy_neutral_v1/v2`, `fact_questions_v1`.
4. **Access.** Verified open on `superkaiba1`: LMSYS-Chat-1M, WildChat-1M, PRISM,
   in-the-wild-jailbreak-prompts, forbidden_question_set, hh-rlhf, ToxicChat, TriviaQA, NQ-Open,
   SimpleQA. Optional: WildChat-1M-Full (manual approval + written justification).

---

## 9. Compute estimate

Qwen-2.5-7B-Instruct. Contingent on gate 3.

| Phase | Estimate |
|---|---|
| Answer generation, 50k WildChat (map training), vLLM | ~2–3 h |
| Eval generation: 4 rungs × ~2k contexts × `K`=5, 3 behaviors | ~2–3 h |
| PV extraction, 3 behaviors × 2 regimes (E1/E2) × 3 vector variants | ~1–2 h |
| Activation capture (train + eval, both arms, layer sweep), batched teacher-forced | ~4–6 h |
| Fits: maps, ridges, MLPs, finetunes × `L` grid × 5 draws × seeds | ~2–4 h |
| **GPU total** | **~14–20 H100-h** (≈10–14 with store reuse) |

**Judge API dominates: ~$1.5–2.5k batched**, near $1k at `K`=3 and `L` capped at 8k.
Hallucination is cheaper than budgeted (exact-match grading needs no judge for correctness).
Every arm in §3 and §7 is a fit over already-collected data: +1–2 H100-h, no judge increment.

**Wall-clock ~2–3 days.** Generation and capture parallelize onto a wide GCP pod (day 1); the
pod is released before the judge wave, which runs off-pod on the Batch API overnight; fits,
figures, and analysis day 2. GPU and judging never overlap — no idle-GPU burn.

---

## 10. Known risks

1. **Contamination on the hallucination pair.** TriviaQA (2017) and NQ-Open (2019) are in
   lm-eval-harness and every pretraining mix. Unavoidable under the real-text constraint — the
   uncontaminated alternatives are exactly the templated ones excluded. Absolute rates are
   contaminated; the design rests on method *deltas*, which is defensible, but say so.
2. **Evil yield may floor** despite the DAN corpus. Gate 1 decides; gate 2 is the fallback.
3. **Stale attacks.** The in-the-wild corpus was written against 2023-era ChatGPT; hh-rlhf is
   2022 attacks on weaker models, so it tests topic generalization well and attack-sophistication
   generalization poorly.
4. **The map extrapolates on every eval rung** — trained on clean WildChat, applied to hostile
   and adversarial contexts. Intentional; the map-degradation diagnostic measures the cost.
5. **Prefix arm on single-turn data is degenerate.** Run it on the multi-turn subset and on the
   DAN slice, where the prompts *are* persona prefixes — the strongest prefix-arm test bed here.
6. **ELEPHANT vs Persona Vectors is asymmetric in kind** — a static corpus with human reference
   responses vs an elicitation harness we run. The shift bundles a generation-provenance
   difference; don't attribute the whole transfer gap to construct shift.
7. **Implementation footguns.** WildChat's `openai_moderation.categories` has 18 names that are
   11 categories with seven slash/underscore duplicates — deduplicate or double-count. Empty
   utterances get an empty dict, not nulls. `conversation_hash` is not unique (use
   `turn_identifier`). LMSYS has 11 clean names, no Detoxify, no top-level `toxic` rollup — a
   WildChat stratifier will not transfer unmodified.

---

## 11. Hygiene

Harmful corpora referenced by **filename + row count only**; analysis over aggregate JSONs,
annotation fields, and judge labels; no raw jailbreak or harmful text paged into agent context
(the `guard_harmful_bank_read.sh` hook enforces this). Implementer briefs carry neutral
mechanistic vocabulary from the first pass, not after the first refusal kill.

---

## 12. Next step

File as a `kind: experiment` task with the Goal formalized — construct definitions for all three
behaviors, the matched-budget protocol, and the primary comparison pre-registered — run the
gate-1 yield pilot, then `/adversarial-planner`.
