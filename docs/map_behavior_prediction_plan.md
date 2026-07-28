# Predicting behavior expression through the context→answer map

**Status:** draft plan for review (not yet filed as a task). Drafted 2026-07-27.

---

## 1. Question and claim

Persona-vector monitoring (arXiv 2507.21509) predicts behavior expression by projecting
the persona vector `v_B` onto the **context** vector. This is a datatype mismatch: `v_B` is
a mean over **answer** activations. The natural correction is to apply our learned
context→answer map `M` first and project onto the predicted answer vector:

```
score(x) = ⟨v_B, M(x)⟩          instead of          score(x) = ⟨v_B, x⟩
```

A direct regression from context to expression is an asymptotic upper bound on any
function of the context, including `⟨v_B, M(x)⟩`. So the claim that survives review is
**not** "we beat direct regression at scale" — it is:

> **Primary claim.** Unlabeled context→answer structure is a useful *prior* for behavior
> prediction. Map-based methods reach a given accuracy with far fewer behavior labels than
> label-hungry direct regression, and the advantage **grows with distribution shift**.

Two quantitative headlines fall out: the **crossover** (how many labels direct regression
needs to overtake the label-free map arm) and the **degradation slope** across a
distribution-shift ladder.

**Secondary hypothesis (pre-registered).** Labels bought from one spread-inducing
mechanism buy in-distribution accuracy at the cost of OOD robustness. Expect
finetuned > label-free at rung 0, with the gap narrowing or inverting by rung 3.

---

## 2. Matched-budget protocol

The methodological spine. At every point on the scaling curve, **every** method receives
the identical budget `(U unlabeled context→answer pairs, L labeled examples)` and may
spend it however its estimator likes. This turns the experiment from a method comparison
into an **estimator comparison at fixed data budget**.

Consequences, all binding:

- **Baselines get `U` too.** Direct regression is fit on `U`-whitened context vectors
  (`z = Σ_λ^{-1/2}(x − μ)`, with `μ, Σ` estimated on the `U` unlabeled contexts; shrinkage
  or truncated-PCA whitening). Activation space is strongly anisotropic, so isotropic
  ridge on raw activations is a strawman — the map encodes the same second-order structure
  for free. Whitening statistics are fit on the **training-side** pool (WildChat) only and
  applied unchanged to every eval rung; no transductive refitting.
- **Label-free methods get `L` too.** Projection arms fit a readout (affine or isotonic)
  and select their layer on `L`. No method leaves budget unspent.
- **Tuning comes out of `L`.** Nested CV; validation is carved from `L`, not from a free
  external set. Layer choice is a hyperparameter and goes through the same split.
- **Identical realized folds.** For a given `(L, draw seed)` every method receives the
  *same* fold indices — not merely the same procedure. This makes all comparisons paired
  and validates the paired bootstrap. Folds are **group-level** (by conversation, persona,
  jailbreak family, question entity), matching the eval ladder.
- **Refit on full `L` after selection**, uniformly. A hyperparameter-free estimator's only
  advantage is not needing selection — which is a genuine property, not an artifact.
- **Matched search budget.** Fix (or at minimum report) the number of hyperparameter
  configurations each arm may evaluate. Report selected values per method per `N`; a
  selection at a grid boundary means the grid was too narrow.

**Three independent axes**, deliberately decoupled:

| Axis | Range | Feeds |
|---|---|---|
| `U` — unlabeled map data | 250 → 50k WildChat pairs | map quality only |
| `L` — labeled data | 250 → ~16k judged, spread-balanced | all label-consuming arms |
| PV extraction set | fixed per behavior | projection arms |

Report two 1D slices (labeled-scaling at `U`=50k; map-scaling at fixed `L`), not a 2D grid.

---

## 3. Method roster

All arms run in **both** the prefix→answer and context→answer variants
(prefix = everything before the user query; context = prefix + query).

**Context-side (no map)**
1. Project `v_B` on the context vector — the persona-vectors paper's own method.
2. Same, with a **context-native** direction extracted in context space — the steel-manned
   version. Required for fairness: comparing our best answer-space direction against a
   paper-recipe direction applied to contexts would make the mismatch true by construction.
3. Identity + learned bias (`x + b`, `b` = train-fold mean of `y − x`) → project.
4. Direct linear regression on `U`-whitened context vectors.
5. Direct nonlinear (MLP) regression on `U`-whitened context vectors.

**Map-based (map fit on `U` unlabeled pairs, linear and nonlinear variants)**
6. Map → project `v_B`. **Label-free** in the direction; the headline sample-efficiency arm.
7. Map → regression trained on *predicted* answer vectors (frozen map, labeled readout).
8. Map → regression trained on *real* answer vectors.
9. **Pretrain-then-finetune**: map fit on `U`, then fine-tuned end-to-end through the
   readout on `L`, regularized toward the pretrained weights (L2-SP / low LR). Must
   degenerate exactly to arm 6 at `L=0` — a hard sanity check.
10. **Stacked**: two-parameter combiner on `L` over (arm 6 score, arm 4/5 prediction).
    What a practitioner would deploy; plausibly the best arm on the board.

**Oracles (privileged information — reference lines, outside the budget protocol)**
11. Project `v_B` on the **true** answer vector.
12. Regression from the **true** answer vector.

**Controls**
13. **Shuffled-map**: map fit on permuted (context, answer) pairs → project. Isolates
    "this pretraining helps" from "any smoothing transform helps."
14. **Shuffled-pretrain-then-finetune**: the arm-9 control — separates initialization
    value from pretraining-data value.
15. **Text-only**: sentence-embedding of the raw prompt → ridge, same `L`. No activations.
    If this matches the activation methods, we need to know before a reviewer says it.
16. **Trivial surface features**: correlation of length, shipped moderation score, and
    question type with expression. Every method must visibly beat something dumb.

---

## 4. Persona-vector extraction

Three directions per behavior, compared as a sub-result:

- **(a) Paper instruction-pairs.** arXiv 2507.21509 recipe verbatim except the judge:
  5 pos/neg system-prompt pairs, 20 extraction questions (disjoint from eval), 10 on-policy
  rollouts each, judge-filter (pos > 50 / neg < 50, malformed/REFUSAL **dropped** from both
  arms, per-arm drop counts reported), response-averaged activations, diff-of-means per layer.
  Judge = `claude-sonnet-4-5-20250929`.
- **(b) Pooled natural.** Diff-of-means between high- and low-judged natural answers.
- **(c) Matched-pair natural — preferred.** Sample `K` answers per context, keep contexts
  with genuine within-context score spread, compute `mean(high) − mean(low)` **within each
  context**, then average across contexts. The query is held fixed by construction, so
  topic/length/register cancel. Pooled (b) risks recovering a topic direction; the paper's
  (a) controls for this by holding the question fixed and varying only the instruction —
  (c) is the natural-data analogue of that control.
  Select extraction contexts by mid-range mean expression and high within-context variance.

Variants (b) and (c) **consume labels**, so they are not label-free — they enter the
scaling family as alternative estimators spending the *same* `L`. Prediction: diff-of-means
is a one-direction estimator and should dominate ridge at small `L`, giving a third
crossover to report. A covariance-whitened (LDA-style) variant is a two-line change and
worth including.

Extraction pool disjoint from every eval rung; extraction runs on the **training**
mechanism and is evaluated across the ladder like everything else.
Open sub-question: which extraction source transfers best OOD? Plausibly (c) wins
in-distribution while (a) transfers better, being mechanism-agnostic by construction.

---

## 5. Data

### 5.1 Map training (unsupervised)
WildChat context→answer activation pairs, `U` scaled 250 → 50k. **No judge scores in the
map objective, ever.** The map never sees the enriched labeled pool or any eval rung — that
asymmetry is the thesis. Optional cheap arm: map + unlabeled enriched pairs (labels held
out) — if it wins, the finding is "unlabeled target-distribution data helps; labels aren't
what's scarce."

### 5.2 Spread must come from a *mechanism*, and train/eval split on the mechanism

Behavior expression varies for structurally different reasons: **prefix-induced** (persona
card / roleplay preamble), **query-induced** (the request itself affords it),
**history-induced** (multi-turn pushback, emotional disclosure), **adversarial-wrapper-induced**.
Inducing spread the same way in train and eval yields an in-distribution eval with extra
steps. So: train on spread we can **construct**, evaluate on spread that **occurs**. This is
also the deployment story — you can afford to label synthetic red-team data; the question is
whether the monitor survives real attacks.

| Behavior | Train spread (constructible, dense) | OOD eval spread (real, different mechanism) |
|---|---|---|
| Evil | Graded persona prefixes over real WildChat queries (cards spanning neutral→hostile, never naming the trait) + WildJailbreak synthetic train rows | In-the-wild jailbreak prompts (Shen et al.) + HH-RLHF red-team attempts — real human attackers |
| Sycophancy (trait) | Affordance-stratified WildChat (advice/validation-seeking vs technical) + graded sycophantic persona prefixes | AITA/ELEPHANT (crowd-verdict anchored, different genre) + ownership-framing over real artifacts + PRISM |
| Hallucination | Entity-popularity-stratified factual questions (long tail induces fabrication) | SimpleQA long-tail with gold answers + false-premise probes + audited open-domain real traffic |

**Use ≥2 inducing mechanisms in training.** If all training spread comes from persona
prefixes, a context-side predictor can learn "detect evil-flavored persona card" and will
collapse on jailbreaks. Mixing mechanisms applies equally to every arm, so it is not a
thumb on the scale.

*All dataset identities, sizes, and field names to be verified at plan time.*

### 5.3 Building a spread-balanced labeled pool cheaply
Do **not** over-generate and subsample on judged score. Stratify on the **known inducing
factor** at sampling time — persona-strength bin (known by construction), shipped
moderation-score bin, affordance class (cheap screen), entity-popularity bin — then judge,
check the realized expression histogram, and top up only thin bins.

Disclose two selection effects: stratification on the inducing factor (mild) and top-up on
the outcome (real). The training distribution is deliberately non-representative — fine for
ρ / AUROC ranking metrics, **not** for absolute calibration or cross-set magnitude
comparisons. **Hold the target histogram fixed as `L` grows**, or the scaling curve is
confounded with changing spread.

### 5.4 Eval: a shift ladder, not a binary
LMSYS is demoted — same genre, overlapping populations, duplicated prompts. It is
near-dup-filtered against WildChat train and relabeled "same genre, different collection."

- **Rung 0** — held-out rows, same mechanism, same corpus (in-distribution *with* spread). The reference; without it we cannot separate "degrades under shift" from "was never good."
- **Rung 1** — same mechanism, different corpus.
- **Rung 2** — different mechanism, similar corpus.
- **Rung 3** — different mechanism + different corpus + different authorship. The real OOD.

Also retained as a labeled reference setting: the PV synthetic elicitation suite, split into
instruction-elicited vs naturally-eliciting prompts and reported separately (it conflates
natural elicitation with artificial prompting by design).

**Disjointness contract**, enforced per behavior: persona pool, jailbreak family/source,
conversation ids, artifact texts, question entities, plus MinHash/embedding near-dup
filtering between the training pool and every rung. Held out by **conversation**, not turn.

---

## 6. Dependent variables

- **Primary:** graded 0–100 judge, `claude-sonnet-4-5-20250929`, `N` draws at temperature > 0
  (pilot `N`=3 vs 5 on measured test–retest), mean-aggregated. Malformed / REFUSAL /
  out-of-range returns **dropped, never coerced**; per-arm drop rate reported and split
  content-drops vs transport-losses. `max_tokens` ≥ ~300 for reason-then-score rubrics.
  Batch API; rubric-keyed cache. Per-behavior judge–human audit (~100–150 items) before any
  behavior carries a headline — HH-RLHF red-team ships human harmfulness ratings usable as
  the reference set for evil.
- **Secondary companion:** teacher-forced fixed positive/negative pool margin (#722 recipe) —
  the same fixed answer pools scored under every context, so no selection-on-outcome bias.
  This is the primary target wherever the judged rate floors (expected for evil on plain
  strata, given Qwen-2.5-7B-Instruct's refusal rate), validated against the rate wherever
  spread exists.
- **Auxiliary for evil:** refusal judged as a separate DV on the same contexts. Report the
  correlation between predicted-evil and predicted-refusal, and evil ρ **conditioned on
  non-refused responses**. If the evil signal vanishes among compliant responses, we built a
  refusal probe.
- **Expectation vs realization:** context-side methods can only predict the *expected*
  expression over answer sampling. Sample `K`≈5 answers per eval context; score context
  methods against the per-context mean, answer methods per-answer. Report the **split-half
  ceiling** (mean of samples 1–2 vs 3–5 across contexts) — the irreducible ceiling for any
  context-based method and the honest denominator. Use design-aligned (item-matched) splits.
- **Construct pinning (sycophancy):** the trait construct (flattery, ingratiation, excessive
  validation), **not** epistemic capitulation to wrong claims. Use the same trait description
  and rubric at extraction and evaluation; state the epistemic construct as out of scope in
  the Goal. Optional discriminant-validity check: the sycophancy direction should predict
  trait expression but not (or more weakly) epistemic capitulation.
- **Spread gate:** pre-registered floor (e.g. inter-context SD ≥ 10 on 0–100, and < 80% of
  contexts in the bottom bin). On failure, evil retains synthetic + persona settings only and
  refusal substitutes into the natural settings. Mechanical, decided in advance.

---

## 7. Metrics and analysis

- **ρ** (Spearman) and **AUROC** per method × behavior × rung × `N`. AUROC is what a
  monitoring deployment actually cares about.
- **Paired bootstrap** over shared eval contexts for all method deltas.
- **Scaling curves** vs `L`: label-free arm as a flat line, direct regression rising from
  chance, finetuned-map interpolating. Read the crossover as the horizontal gap.
- **Degradation slope** across rungs 0→3, per method — the headline robustness result.
- **Resample the labeled draw**, ≥5 independent subsamples per `N`, plot mean ± CI over
  draws. At `L`=250 the dominant variance is *which* examples were drawn; scaling curves get
  this wrong constantly.
- **Map-degradation diagnostic:** the map's own held-out R², identity+learned-bias baseline,
  and kNN retrieval (euclidean + cosine, chance = k/n stated) per rung. Lets us attribute
  prediction failures to map failures. Required by the standing mapping-baselines rule.
- **Robustness read for sycophancy:** ρ within affordance stratum, or with topic clusters
  partialled out — guards against "predictor learned emotional topic → high sycophancy."
- **Within-stratum ρ for evil**, never pooled across harmful/benign — pooled ρ is mostly the
  trivial between-stratum gap.
- **Pre-registered primary comparison.** ~16 arms × 3 behaviors × 4 rungs × 6 `N` values is
  enough cells to find anything. Primary contrast: map→project vs the **steel-manned**
  context projection (arm 2) at rung 3. Everything else is secondary with multiplicity
  correction.
- Plots over tables for all cross-condition comparisons; one color = one meaning across every
  figure; low-level per-unit plot alongside every aggregate.

---

## 8. Compute estimate

Qwen-2.5-7B-Instruct. **Check first** whether the #722 / #779 / #952 / #1092 WildChat/LMSYS
activation stores are reuse-fit under the artifact-reuse checklist — that could roughly halve
the GPU line.

| Phase | Estimate |
|---|---|
| Answer generation, 50k WildChat (map training), vLLM | ~2–3 h |
| Eval generation: 4 rungs + synthetic, ~2k contexts each × `K`=5 | ~1.5–2 h |
| Persona-prefix + jailbreak stratum generation | ~1–1.5 h |
| PV extraction, 3 behaviors × 3 extraction sources | ~1–2 h |
| Activation capture (train + eval, both arms, layer sweep), batched teacher-forced | ~4–6 h |
| Fits: maps, ridges, MLPs, finetunes × `L` grid × 5 draws × seeds | ~2–4 h |
| **GPU total** | **~14–20 H100-h** (≈10–14 with store reuse) |

**Judge API is the dominant cost: ~$1.5–2.5k batched.** Labeled pool + eval rollouts × 3
behaviors × `N` draws; `N`=3 and `L` capped at 8k lands near $1k. Every arm added in §3 and
§7 is a fit over data already collected — roughly +1–2 H100-h total and **no** judge
increment.

**Wall-clock ~2–3 days.** Generation + capture parallelize onto a wide GCP pod (day 1);
the pod is released before the judge wave, which runs off-pod on the Batch API overnight; fits,
figures, and analysis on day 2, slack on day 3. GPU and judging never overlap, so no idle-GPU burn.

---

## 9. Known risks

1. **Range restriction is real** — training spread is constructed, eval spread is natural.
   Mitigated by the mechanism-split design and rung 0 as reference, but it means every arm is
   extrapolating, and that should be stated rather than hidden.
2. **Prefix arm on single-turn data is degenerate** (empty system prompt). Run it on the
   multi-turn subset and on the persona-prefix / jailbreak strata, where the prefix carries
   real variation — those strata are the strongest prefix-arm test bed available.
3. **Corpus contamination**: all public corpora plausibly overlap Qwen-2.5 pretraining. A wash
   for *ranking* comparisons (every arm faces the same contexts); note as a scope caveat.
4. **Open-domain hallucination judging is the weakest judge task** of the three. Gated on the
   judge–human audit; if agreement is weak, the hallucination headline rests on the gold-answer
   stratum and the open-domain cell is exploratory.
5. **Evil may floor even on jailbreak strata** given the model's refusal rate. The margin DV and
   the successful-jailbreak strata are the mitigations.

---

## 10. Hygiene

Harmful banks referenced by **filename + row count only**; analysis over aggregate JSONs and
judge labels; no raw jailbreak or harmful-bank text paged into agent context (the
`guard_harmful_bank_read.sh` hook enforces this). Briefs for the implementer carry this plus
neutral mechanistic vocabulary from the first pass, not after the first refusal kill. Reuse
in-repo assets where fit: `query_banks/` (advbench, strongreject), the EM eval bank and its
calibrated judge, #612's sycophancy elicitation machinery, #722's validated fixed-pool margins.

---

## 11. Next step

File as a `kind: experiment` task with the Goal formalized (construct definitions for all
three behaviors, the matched-budget protocol, and the primary comparison pre-registered),
then run `/adversarial-planner`.
