# SAE feature description + categorization pipeline — design plan

**Date:** 2026-07-28. **Sources:** five-agent literature sweep, notes in
`docs/lit_reviews/notes/autointerp-rq{1..5}-*.md` (every arXiv id resolved
in-session; could-not-verify lists per RQ). Companion: the residual-stream
direction taxonomy review + its integration doc.

**Target:** describe and categorize the answer-side features of the project's
SAE dictionaries (16,384 restricted / 131,072 full at layer 19; the layer-20
Matryoshka pair; early layers 3/7), producing a per-feature table the map
rounds (#1482, #1092, #1738) join against.

---

## 1. What the literature settles (design-binding)

1. **Explainer capability is saturated; evidence design is the lever.**
   Claude 3.5 Sonnet (0.75 fuzz / 0.75 detect), Llama-3.1-70B (0.76/0.74) and a
   human annotator (0.75/0.74) are indistinguishable against a 0.51
   random-explanation floor (2410.13928). Sonnet 4.5 buys nothing from
   capability alone.
2. **Input⊕output evidence is the single largest measured gain.** Max-activating
   descriptions score 56.6 input / 49.2 output; logit-lens descriptions 50.1 /
   56.5; the combination **66.6 / 64.9** — both axes up 6–10 points
   (2501.08319). Unembedding the decoder ("VocabProj") is one matrix product per
   feature, no corpus pass.
3. **Top-k-only sampling loses ~15% of above-chance signal**; the top range
   accounts for a small share of a feature's causal effect and lower-activation
   inputs are qualitatively different (2410.13928, 2405.06855). Use
   quantile-stratified top-and-random.
4. **Saturation points:** ~40 examples (10→0.73, 40→0.76, 60→0.75); context
   16–32 tokens (64 slightly worse); per-token activation values +0.01; chain of
   thought 0.00 to −0.03 at large cost (2410.13928). Do not spend there.
5. **Corpus choice barely moves quality but decides coverage:** never-firing
   features drop 15% → 1% on the closer corpus (2410.13928). Persona/assistant
   features must be sampled on the chat distribution or they read as dead.
6. **Labels are a search index, not evidence.** Top-scoring explanations
   re-evaluate at F1 ≈ 0.6 with "little to no causal efficacy" (2309.10312);
   explanation-driven simulation of a layer is statistically indistinguishable
   from zero-ablating it (2501.18838).
7. **Label collision is the norm:** 82.1% of annotated features share their
   label with another, mean 3.07 features per label, "plural nouns" covers 101
   features — and detection-style scoring is provably invariant to it
   (2605.12874). Neighbour-discrimination must be measured separately;
   balanced accuracy falls from >80% to chance when distractors are cosine
   neighbours (EleutherAI).
8. **Metrics themselves fail sanity checks:** balanced accuracy 53.67% and
   recall 0.00% under label corruption; Pearson, cosine, AUPRC, F1, IoU pass
   >93% (2506.05774). Scorers disagree (detection–simulation Spearman 0.44).
9. **Cost is dominated by scoring, not explaining:** explanation ≈ $3.4k/M,
   detection/fuzzing ≈ $2.5k/M, simulation ≈ $96k/M; fuzzing has the best human
   correlation (0.69) anyway (EleutherAI; 2410.13928).
10. **Classification-specific:** autoregressive LLMs do multi-label wrong
    mechanically — all but one label suppressed per step (2505.17510); rubric
    scoring carries model-specific position bias and criterion-order bias
    (2602.02219); label-set size correlates negatively with recall and
    hierarchies do not reliably help (2406.04797); sharp category definitions
    dominate reliability, CoT adds little (2506.13639); κ ≥ 0.74 is achievable
    for a feature taxonomy when categories are fixed a priori from external
    theory (2605.23035).
11. **Base rates govern the metric.** Comparable target classes run 1–8%; at a
    low-single-digit identity base rate a binary judge's positive set is
    necessarily near-miss-dominated — the observed 20 language / 11 register /
    9 identity split. **Report precision at low base rate, not accuracy.**
12. **In-house position:** the project's judge dispatch (sync-vs-Batch routing,
    rubric-keyed caching, transport-vs-content drop split, resumable
    re-dispatch, 2,000-item shard ceiling from incident knowledge) is more
    operationally mature than anything surveyed; the gap is the evidence
    builder and the absence of any scoring harness.

---

## 2. Pipeline

### Phase 0 — Mechanical axes (free, all features, no LLM)

Computed once per dictionary; these are label-free and carry the load-bearing
statistics. One GEMM or one streaming pass each.

| Axis | Computation |
|---|---|
| `logit_footprint` | decoder column → unembedding; top-k promoted / suppressed tokens, concentration (the say-X screen, 2501.08319) |
| `density` / `activity` | firing frequency; dense-latent flag |
| `persist_answer` | within-answer consistency (fraction of answer tokens active) |
| `persist_query` | cross-query consistency given a prefix (crossed corpora only) |
| `nuisance_load` | mass on massive-activation dims, sink positions, top-48 PCA scaffold; γ report |
| `rb_align` | \|cos(decoder, r_B)\| per trait, raw **and** scaffold-projected (#779 finding) |
| `neighbors` | top-k decoder cosine neighbours (feeds the discrimination scorer) |
| `tier` | Matryoshka nest index (layer-20 dictionaries only) |
| `arm_shares` | per-feature prefix / query / interaction variance share (crossed corpora only) |

### Phase 1 — Evidence builder (the one real build item)

Per feature, from the project's own chat corpus (finding 5):

- **Activating examples:** 40 total, **quantile-stratified** across the
  activation range (not top-40), each a 32-token window centred on the peak
  token, with `<<delimiters>>` marking the max-activating token (Delphi format).
- **Non-activating examples:** 20, sampled from contexts where the feature is
  silent (Delphi ratio).
- **Near-miss examples:** 5 from the feature's top cosine neighbours' activating
  set — the discrimination evidence the collision literature demands.
- **Output-side block:** top-10 promoted and top-10 suppressed unembedding
  tokens from Phase 0.
- **Statistics block:** density, persistence, tier.
- Excludes sink/BOS positions; discloses truncation inline.

Engineering: a genuine per-feature bounded top-K over the streamed activation
store (the existing `phase_scan` builds evidence for 300 sampled features and
cannot be scaled by parameter change).

### Phase 2 — Description (Batch API, Sonnet 4.5, 1 draw)

One free-text description per feature + a self-reported confidence. Explicitly
**not** length-capped to a word (the Neuronpedia one-word labels are a
conciseness-prompt artifact, finding: `np_max-act-logits` asks for brevity and
its own docs warn it misses patterns over longer texts). Descriptions are a
search index over the dictionary — never cited as evidence for a claim.

### Phase 3 — Axis classification (Batch API, Sonnet 4.5, N=5 draws)

**One axis per call** (2602.02219), **forced single choice** per axis
(2505.17510), small flat label sets (2406.04797), sharp definitions with
explicit near-miss exclusions (2506.13639), label order permuted across draws,
majority vote over 5 draws at temperature > 0 (llm-judging rule 4; the
`persona_related` κ 0.136 at N=1 is what this fixes).

| Axis | Labels |
|---|---|
| `abstraction` | token-surface / lexical-semantic / abstract-contextual |
| `speaker_property` | language / register-style / identity-disposition / none |
| `content_type` | topic / task-format / entity / syntax / operation |
| `functional_role` | input-side / output-promoting / mixed *(judged against the Phase-0 footprint, never from activations alone — 2501.08319)* |
| `interpretable` | yes / no *(the unclear rate is itself a signal: 19% of the worst-predicted tail vs 3% of the best)* |

Reported per axis, **never pooled into a composite** — pooling language into
"persona" is exactly what produced the retracted OR 8.3.

### Phase 3b — What each categorizer actually sees (input matrix + blinding)

Shared packet blocks (built once per feature in Phase 1): **[EX+]** 40
quantile-stratified activating windows (32 tok, peak token delimiter-marked);
**[EX-]** 20 non-activating windows; **[NEAR]** 5 windows from top decoder-cosine
neighbours' activating sets; **[OUT]** top-10 promoted / top-10 suppressed
unembedding tokens; **[DESC]** our own Phase-2 description; **[STAT]** density,
persistence, tier.

| Axis | Sees | Withheld | Why |
|---|---|---|---|
| `abstraction` | EX+, EX−, DESC | STAT, OUT, R² | density/footprint are its mechanical correlates — showing them makes the correlation circular |
| `speaker_property` | EX+ (topic-diverse draw), EX−, NEAR, DESC | STAT, OUT, R² | the invariant is only visible across topically varied examples; NEAR forces the language-vs-register-vs-identity discrimination that the binary field collapsed |
| `content_type` | EX+, NEAR, DESC | STAT, R² | — |
| `functional_role` | EX+, **OUT**, DESC | STAT, R² | cannot be judged from activations alone (2501.08319); OUT is required evidence, so its mechanical validator must be an *independent* quantity — attribution profiles / steering (2604.07615), not the footprint it was shown |
| `interpretable` | EX+, EX−, DESC | STAT, OUT, R² | — |

**Two blinding rules, both load-bearing:**

1. **Blind every categorizer to the dependent variable** (per-feature R², arm
   shares, any map output). We correlate labels against these; showing them
   manufactures the correlation.
2. **Blind each axis to its own mechanical validator.** An axis judged from the
   same quantity it is later checked against has no independent check. Where a
   mechanical block is *required* evidence (OUT for `functional_role`),
   validation moves to a different quantity.

**Also withheld: the Neuronpedia description.** It is median one word, produced
from token lists on a different corpus, and would anchor the judge toward
surface labels ("uso" → token-level). Keep it on the dashboard as auxiliary
provenance, out of the categorizer prompt.

### Phase 4 — Validation (the part that makes labels trustworthy)

1. **Detection + fuzzing scoring** on a stratified ~1,000-feature sample
   (Delphi scorers; ~1/5 the cost of simulation and *better* human correlation).
   Never simulation over a full dictionary (900× cost).
2. **Discrimination score** with cosine-neighbour distractors — the collision
   check detection is provably blind to.
3. **Two controls, both required:** shuffled-label control (2506.05774) and
   random-init dictionary control (the 2410.13928 vs 2501.17727 contradiction
   means running both randomization forms).
4. **Non-judge validators, per axis** — the point being that a boundary only a
   judge can see is the boundary that failed:
   - language → monolinguality metric + single-language ablation (2505.05111)
   - register → the informal-register subspace with zero-shot steering transfer
     (2603.26236)
   - functional role → attribution profiles + unembedding weights (2604.07615)
   - abstraction → layer index across the early/mid/late dictionaries
5. **Human alt-test** (2501.10970) on a modest annotated subset before trusting
   the full sweep; report reliability two-dimensionally — intrinsic consistency
   (test-retest κ per axis) **and** human alignment (2602.00521).
6. **Precision at low base rate** as the headline metric for
   `identity-disposition`, not accuracy.
7. **Metric hygiene:** report Pearson / F1 / AUPRC (sanity-check-passing), not
   balanced accuracy or recall alone; ≥3 seeds for any cross-condition
   comparison (noise floor 0.016, single-seed needs Δ > 3.93σ, 2605.18229).

---

## 3. Cost and scale

Batch API, Sonnet 4.5, ~963 prompt + ~300 output tokens per call:

| Run | Calls | Est. cost |
|---|---|---|
| 16,384 restricted features — describe (1 draw) | 16k | ~$45–75 |
| 16,384 — classify 5 axes × 5 draws | 410k | ~$1.1–1.8k |
| Validation sample (1k features, detection+fuzzing+discrimination) | ~9k | ~$30 |
| **Subtotal, restricted dictionary** | | **~$1.2–1.9k** |
| 131,072 full dictionary, same recipe | 3.3M | ~$9–15k |

Mechanical axes: ~0. Evidence building: one streamed pass per dictionary
(CPU/GPU minutes, no API). The 131k row is what would force a local
open-weight explainer (EleutherAI: 1.5M features for $1.3k on Llama-3.1-70B vs
$8.5k on Claude 3.5 Sonnet); at 16k the Batch API is cheaper than the
engineering.

## 4. Build vs adopt

- **Build:** the evidence builder (Phase 1) — the only genuine engineering item.
- **Keep:** in-house judge dispatch unchanged (Batch routing, rubric-keyed
  caching, transport/content drop split, 2,000-item shard ceiling — hard-won
  incident knowledge a tool swap would discard).
- **Adopt:** Delphi's detection/fuzzing/discrimination scorers for Phase 4
  *if* a custom `BatchTopKSAE` can be registered without forking (its README
  documents only Sparsify and Gemma coders — **resolve by reading
  `delphi/sparse_coders/` + `delphi/__main__.py` before committing**);
  Delphi's token-window + delimiter evidence format regardless.
- **Adopt as free baseline:** the Neuronpedia export (~96% coverage), kept
  explicitly as auxiliary evidence — it is token-level on a generic corpus and
  median one word.
- **Do not adopt:** Delphi wholesale; agentic explainers (multi-turn, cannot
  ride the Batch API, and their human-parity claims are confounded).

## 5. Sequencing

1. Phase 0 on the layer-19 restricted dictionary — free, immediately joins the
   running map rounds.
2. Phase 1 evidence builder + Phase 2/3 on 16,384 features + Phase 4 validation
   sample. This is the pilot that decides everything else.
3. Extend to the layer-20 Matryoshka pair (tier axis, chat-vs-pile contrast) and
   the early-layer dictionaries (3/7) — the abstraction-axis external validator.
4. Full 131k only if the pilot's precision at low base rate justifies it, and
   then with a local explainer.

## 6. Open questions to resolve before building

- Delphi custom-encoder registration (determines Phase-4 adopt vs reimplement).
- Whether near-miss/neighbour evidence at *generation* time helps — untested in
  the literature (a gap the sweep flagged twice); cheap A/B in the pilot.
- Human annotation budget for the alt-test subset.
