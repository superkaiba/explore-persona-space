---
title: 'Which contexts does the context→answer map fail to retrieve? A controlled,
  SAE-free failure characterization on the #1738 100k multi-turn map'
kind: experiment
tags: []
created_at: '2026-08-08T16:11:03Z'
has_clean_result: false
parent_id: 1738
origin_prompt: 'Motivation: We did an analysis of the directions the model fails on
  using SAE features; SAE features are known to be somewhat unreliable; we want to
  do a more controlled analysis of this question. Methodology: apply our best mapping
  on the generic corpus; see for which ones it fails to distinguish the correct answer
  vector from some other answer vector; look at the contexts it fails on and characterize
  what kinds of things it fails on. (Full verbatim request + resolved clarifications
  in the body''s ## Provenance section.) [then] run it in background with happy coder'
workflow: v1
goal: 'On the #1738 ridge context→answer map (context arm, layer 19, Qwen-2.5-7B-Instruct,
  100k multi-turn LMSYS/WildChat corpus, pinned held-out n=9,941), characterize WITHOUT
  SAE features which contexts the map fails to retrieve — i.e. whose predicted answer
  vector does not single out the true answer vector among all 9,941 held-out answers
  — by (1) building a dashboard of every rank-1 failure plus the worst-rank tail carrying
  the full confusion geometry (context↔context, answer↔answer, answer↔confuser-context
  and prediction↔confuser similarities, plus pool-wide ranks, in raw, mean-centered
  and whitened metric spaces); (2) building a 500-context random-sample dashboard
  carrying both the retrieval neighbour list and the prediction-collapse neighbour
  list; and (3) measuring the symmetry (reciprocity) of the confusion graph against
  a degree-preserving null and a distance-only null — with failure attributed to map
  error vs irreducible target degeneracy or answer-sampling noise via the banked K-resample
  retrieval ceiling.'
---
# Controlled (non-SAE) characterization of which contexts the #1738 context→answer map fails to retrieve

## Goal

On the #1738 ridge context→answer map (context arm, layer 19, Qwen-2.5-7B-Instruct, 100k multi-turn LMSYS/WildChat corpus, pinned held-out n=9,941), characterize WITHOUT SAE features which contexts the map fails to retrieve — i.e. whose predicted answer vector does not single out the true answer vector among all 9,941 held-out answers — by (1) building a dashboard of every rank-1 failure plus the worst-rank tail carrying the full confusion geometry (context↔context, answer↔answer, answer↔confuser-context and prediction↔confuser similarities, plus pool-wide ranks, in raw, mean-centered and whitened metric spaces); (2) building a 500-context random-sample dashboard carrying both the retrieval neighbour list and the prediction-collapse neighbour list; and (3) measuring the symmetry (reciprocity) of the confusion graph against a degree-preserving null and a distance-only null — with failure attributed to map error vs irreducible target degeneracy or answer-sampling noise via the banked K-resample retrieval ceiling.

## Motivation

The existing characterization of what the context→answer map gets wrong runs through
sparse-autoencoder (SAE) features ([#1482](https://eps.superkaiba.com/tasks/1482),
[#1946](https://eps.superkaiba.com/tasks/1946), [#2163](https://eps.superkaiba.com/tasks/2163)).
SAE features are a known-unreliable readout basis: dictionary reconstruction is lossy (fraction of
variance explained 0.718 at layer 19 in #2163), feature descriptions fail to identify features
([#1773](https://eps.superkaiba.com/tasks/1773)), and the read depends on the specific dictionary.
This run asks the same question — **what kinds of contexts does the map fail on?** — with a
controlled, SAE-free instrument: nearest-neighbour retrieval of the true answer vector among all
held-out answer vectors, read directly in the residual-stream basis the map is fit in.

## Design decisions (locked with the user before filing)

| Decision | Choice |
|---|---|
| Map + corpus | #1738 ridge map, **context arm**, layer 19, Qwen-2.5-7B-Instruct, 100k multi-turn LMSYS/WildChat corpus, pinned held-out n = 9,941 |
| Failure definition | **rank-1 miss** (nearest pool answer is not the true one) **plus the worst tail** by true-target rank |
| Worst tail size | top-200 by true-target rank, reported alongside top-200 by raw distance (the two sets differ; both listed) |
| Metric spaces | **raw** (euclidean + cosine) · **mean-centered** cosine · **whitened / Mahalanobis** (shrinkage-regularized train-answer covariance) — every read under all three |
| Arms | **context arm only** — explicit stated deviation from the prefix-AND-context standing rule (see Deviations) |
| Model-driven characterization | `claude-sonnet-4-5-20250929` for every countable label; **Fable 5** (`claude-fable-5`) for free-form synthesis only, whose named modes are re-labeled by Sonnet-4.5 before entering any claim |
| Result 2 neighbour lists | **both** the retrieval list and the prediction-collapse list, side by side |
| Result 3 symmetry | confusion-graph reciprocity **with two nulls** |
| Readout family | **linear only** (ridge). No MLP / nonlinear readout. |

## Prior artifacts this run reuses (nothing is re-run, nothing is re-generated)

| Artifact | Location | Role |
|---|---|---|
| #1738 ridge map, context arm L19 | HF `superkaiba1/explore-persona-space-data/issue1738_multiturn/`, fitter `scripts/issue1738_multiturn_fits.py` | the map under test |
| Held-out predictions + true answer vectors + context vectors (n = 9,941) | HF `issue1738_multiturn/` | the retrieval read |
| Judged labels — language, topic, format, request-refusal-adjacency, answer-is-refusal (Sonnet-4.5, test-retest κ 0.79–0.98) | `eval_results/issue_1738/judge_labels/labels.json` | the taxonomy, at zero new judge spend |
| Per-context normalized error, both arms | `eval_results/issue_1738/percontext_summary_L19_ridge.csv` | severity axis + cross-check against the retrieval read |
| **K-resample answer states — 1,988 contexts × K=4 extra on-policy answer draws** | HF `issue1738_multiturn/kresample/kresample_shard*.pt` (loader: `scripts/issue1738_characterize.py::_load_kresample_v`) | **the retrieval ceiling control** |
| Banked retrieval read (context arm L19 ridge: acc@1 = 0.816 euclidean / 0.828 cosine) | `eval_results/issue_1738/mapping_baselines.json` | reproduction gate |
| Raw conversation + answer text | HF `issue1738_multiturn/raw_completions/` | dashboard row content |
| `knn_retrieval`, `identity_bias_predict` | `src/explore_persona_space/analysis/mapping_baselines.py` | the retrieval instrument |
| Category-contrast battery (bootstrap CI + permutation + BH) | `scripts/issue1738_characterize.py` | the failure-composition statistics |
| Dashboard builders | `scripts/issue1482_context_extremes_dashboard.py`, `scripts/issue1092_corpus_dashboard.py`, `scripts/build_dashboards.py` | HTML rendering |

A reuse-discovery listing over `scripts/` and `src/` runs before any new file is written; every
existing helper that fits is used rather than re-implemented.

## Methodology

**Retrieval read.** For held-out context *i*: prediction `p_i = h(v_C_i)` where `v_C_i` is the
context-end state (last prompt token, the newline before the assistant turn) at layer 19; candidate
pool = all 9,941 held-out **true** answer vectors `{a_j}`. `rank_i` = rank of `a_i` among the pool
by distance to `p_i` (mid-ranks on ties, matching `knn_retrieval`).

- **FAIL-1** = `rank_i > 1`. Expected ≈ 1,700 rows (acc@1 = 0.816–0.828).
- **WORST tail** = top-200 by `rank_i`, plus top-200 by raw distance `d(p_i, a_i)`, both reported.
- **Confusers of *i*** = pool rows ranked above `a_i`; the dashboard shows up to the top 10.

**Reproduction gate (runs first, blocks everything downstream).** Re-derive `acc@1`, `acc@5`,
`acc@10`, median rank and MRR from the banked predictions and reconcile against
`eval_results/issue_1738/mapping_baselines.json` (context_L19 ridge) to within floating-point
tolerance. A mismatch halts the run.

### Failure attribution — the three sources, separated

Retrieval failure has three sources, and an uncontrolled read cannot tell them apart:

- **(a) map error** — the prediction is wrong;
- **(b) target degeneracy** — two contexts have genuinely near-identical answer states (two
  greetings, two refusals), so no predictor could separate them;
- **(c) answer-sampling noise** — the single on-policy answer per context is one draw from the
  model's own distribution.

Two controls, both zero-GPU:

1. **Retrieval ceiling from the banked K-resample.** For the 1,988 contexts carrying 4 extra
   on-policy answer draws, use a *resampled real answer* as a pseudo-prediction and run the
   identical retrieval against the same 9,941-answer pool. `acc@1_ceiling` is the ceiling any map
   could reach. Per context: a failure whose own resample also fails is **IRREDUCIBLE**; one whose
   resample succeeds is **MAP-ATTRIBUTABLE**; the 7,953 contexts without resamples are **UNKNOWN**.
   Without this control, "the map fails on X" is unfalsifiable.
2. **Answer-answer similarity on every confusion row** — `cos(a_i, a_j)` near 1 means target
   degeneracy, not map failure. The failure set's `cos(a_i, a_j)` distribution is compared against
   a matched non-failure control.

**Pool-size robustness.** The failure set is pool-size dependent: "fails on refusals" means "fails
to separate refusals from 9,940 alternatives". The taxonomy is recomputed at pool sizes 500 /
2,000 / 9,941 (seed-pinned subsamples, the true target always in pool) and reported as stable or
not.

**Identity+learned-bias baseline.** Reported alongside every retrieval number
(`mapping_baselines.identity_bias_predict`; banked acc@1 = 0.473 euclidean / 0.512 cosine). A large
share of retrieval is a shared context-independent offset, which is why the mean-centered and
whitened spaces are load-bearing rather than cosmetic.

### Result 1 — dashboard of all failed contexts

All ~1,700 FAIL-1 rows (client-side paginated + filterable, no cap), WORST-tail rows flagged. Per
row: context text (history tail 800 chars / last user turn 1,200 chars) and true answer (1,000
chars), reusing #1738's excerpt caps; **any truncation or presentation-time substitution is
disclosed inline per passage**. For each of up to 10 confusers ranked above the true answer, its
context text and answer text plus:

- `cos(v_C_i, v_C_j)` — context↔context
- `cos(a_i, a_j)` — answer↔answer (the target-degeneracy tell)
- `cos(a_i, v_C_j)` — true answer ↔ confuser's context (well-defined: both 3,584-d at L19)
- `cos(p_i, a_j)` — prediction ↔ confuser (the quantity that caused the confusion)

each in **all three metric spaces**; plus the rank of `v_C_j` among all 9,941 context vectors by
similarity to `v_C_i`, the rank of `a_j` among all answer vectors by similarity to `a_i`, and
`rank_i` itself; plus #1738's judged labels for both contexts with match/mismatch flags; plus the
IRREDUCIBLE / MAP-ATTRIBUTABLE / UNKNOWN attribution flag.

**Quantitative companions (the claim-bearing side):**

- Failure-set composition by judged label against the held-out base rate — bootstrap CIs +
  permutation p + Benjamini-Hochberg, reusing `issue1738_characterize.py`'s contrast battery so the
  numbers are directly comparable to #1738's taxonomy.
- The retrieval ceiling and the MAP-ATTRIBUTABLE share.
- `cos(a_i, a_j)` for failures vs a matched non-failure control.
- Concordance between `rank_i` and #1738's banked per-context normalized error (do the two
  failure notions agree?).

**Fable 5 synthesis → countable labels.** Fable 5 reads a bounded JSONL (WORST 200 + a stratified
300 of the remaining failures) and returns named candidate failure modes. Each named mode is then
converted into a Sonnet-4.5 binary label applied over the **full failure set and a matched
non-failure control**, so a Fable-5-named mode ships as a rate with a confidence interval rather
than as a narrative. Fable 5 never carries a countable claim.

### Result 2 — dashboard of a random sample of 500 contexts

500 seed-pinned draws from the full 9,941 held-out set (not only failures). Per context, **two
lists side by side**:

- **(A) retrieval list** — the 10 nearest *true answer vectors* to the prediction, with the true
  target's rank marked. What the map thought this context's answer looked like.
- **(B) collapse list** — the 10 nearest *other predictions* to this prediction. Which contexts the
  map maps to the same place.

The gap between (A) and (B) separates "the map collapses these contexts together" from "these
answers genuinely look alike". Same three metric spaces, same judged labels, same
Fable-5-then-Sonnet-4.5 discipline for the "what is the map good/bad at" synthesis.

### Result 3 — symmetry of the confusion

Directed confusion graph *G*: edge `i→j` iff `a_j` outranks `a_i` under `p_i`.

- **Primary:** reciprocity = P(`j→i` ∈ G | `i→j` ∈ G) on the top-1 read.
- **Graded companion:** for every confused pair, the rank of `a_j` under `p_i` against the rank of
  `a_i` under `p_j` (scatter + Spearman) — strictly more informative than the binary rate.
- **Two nulls, both required:**
  1. degree-preserving directed rewiring (configuration model, 1,000 draws) — controls the
     reciprocity implied by the degree distribution alone;
  2. a distance-only null drawing edges from `P(i→j) ∝ exp(−d(a_i, a_j)/τ)`, symmetric by
     construction — tests whether the observed reciprocity is anything beyond "the metric is
     symmetric and answers cluster".
- **Reading:** reciprocity above both null bands ⇒ the map collapses genuine *pairs* of contexts
  into a shared image (equivalence-class structure). Reciprocity inside the bands ⇒ failures are
  one-sided: specific contexts dragged into a generic attractor, which Result 2's collapse list
  would corroborate.

## Compute

**0 GPU-hours.** Every input is banked; no training, no generation, no new activation capture.

- CPU: 9,941 × 9,941 distance matrices per metric space (~400 MB fp32 each) as single batched GEMMs
  — no per-row loops; whitening is one 3,584 × 3,584 shrinkage covariance + Cholesky. Well under
  the 50 GB off-VM footprint gate.
- Download sizing is measured before staging is placed: held-out states ≈ 143 MB per tensor,
  K-resample shards ≈ 340 MB. If the realized total approaches ~10 GB the consuming phase routes to
  a RunPod CPU pod rather than the shared VM; staging never lands on `/` or `/tmp/`.
- API: ~4,000 Sonnet-4.5 labels (failure set + matched control + 500-context sample) via the
  Anthropic Batch API, `max_tokens` ≥ 2048 for the multi-field JSON rubric, **pilot-gated** before
  the production dispatch (≈150 draws at the exact production instrument; gate on zero
  `stop_reason == "max_tokens"` and per-arm parse-fail < 2%). Malformed / refusal / out-of-range
  returns are DROPPED with counts reported, never coerced; transport failures are retried.
- Fable 5: 2 bounded synthesis passes over JSONL digests.

## Deviations to declare (carried into the clean-result as scope caveats)

1. **Context arm only.** Explicit stated deviation from the prefix-mapping-AND-context-mapping
   standing rule. Rationale: on this corpus the prefix arm's acc@1 is 0.183, so retrieval failure is
   the default case there and a failure taxonomy would be near-uninformative; #1738/#1946 already
   established that the two arms fail on complementary populations. A prefix-arm read is the named
   follow-up.
2. **Fable 5 is not a judge.** It generates hypotheses; Sonnet-4.5 produces every countable label.
3. **Linear (ridge) map only** — no MLP or nonlinear readout, per the linear-by-default rule.
4. Layer 19 only (the map's own layer); no layer sweep.

## Success criteria

- The reproduction gate passes (re-derived acc@k matches the banked #1738 values).
- The failure set is partitioned into IRREDUCIBLE / MAP-ATTRIBUTABLE / UNKNOWN with a stated
  retrieval ceiling.
- At least one failure-composition contrast clears the BH-corrected band against the held-out base
  rate, or the null result is reported with its detection floor.
- Both dashboards render (non-empty rows verified by a content probe, not file presence) and are
  reachable at a browser-viewable URL.
- Result 3 reports reciprocity against both null bands.

## Provenance

Originating chat request (verbatim, 2026-08-08):

> ## Motivation
> - We did an analysis of the directions the model fails on using SAE features
> - SAE features are known to be somewhat unreliable
> - We want to do a more controlled analysis of this question
> ## Methodology
> - Apply our best mapping on the generic corpus
> - See for which ones it fails to distinguish the correct answer vector from some other answer vector
> - **Look at the 2/multiple contexts it fails on and characterize what kinds of things it fails on**
>
> ### Result 1: Make dashboard of all failed contexts with: the actual context answer pair; the
> answer vectors the model got confused with and their corresponding context vector; the
> (whitened?) cosine similarity between the actual context vector and the confused context vector;
> the cosine similarity between the actual answer vector and the confused context vector; the
> ranking of these confused context vectors in similarity with the actual context vector compared
> to ALL the context vectors; the ranking of these confused answer vectors in similarity with the
> actual answer vector compared to ALL the answer vectors; What other similarity metrics could we
> consider?; Fable 5 analysis of which contexts the mapping fails on + what causes the confusion
>
> ### Result 2: Make dashboard of a random sample of 500 contexts with: the actual context answer
> pair; the top 10 most similar predicted answer vectors; Fable 5 analysis of what our mapping is
> good/bad at
>
> ### Result 3: Analysis of the symmetry of this: [PLOT: if I apply mapping to a context vector 1
> and get the wrong predicted answer vector, if I apply the mapping to that wrong predicted answer
> vector's context 2, will I get the original context vector 1's answer vector predicted]
>
> Ask clarifying questions

Follow-up clarifications resolved in the same chat are recorded in the Design-decisions table above.
