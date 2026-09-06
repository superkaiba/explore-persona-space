# Response-integrity-matched context versus answer steering

## Result

The old coherence control did not resolve the steering confound. A new
question-aware Codex-subagent rubric finds that historical all-answer steering
has essentially **zero response integrity** at the operating points that produce
the large trait effects:

| Behavior | Route | Integrity | Pass ≥80 | Trait Δ vs historical α0 | Status |
|---|---|---:|---:|---:|---|
| Evil | reverse-map context, L14 c4 | 66.82 | 56.7% | +9.83 | fixed context target |
| Evil | historical all-answer (r_B), L14 c4 | 0.43 | 0.0% | +98.88 | record-only; incomparable |
| Sycophancy | reverse-map context, L14 c2 | 99.35 | 100.0% | +12.95 | fixed context target |
| Sycophancy | historical all-answer (r_B), L14 c2 | 2.38 | 0.0% | +65.30 | record-only; invalid trait coverage |

The all-answer rows are not a matched estimate. Their integrity is measured on
the common first-2,048-token horizon, whereas the stored trait scores use the
regenerated 4,096-token responses; 114/120 evil and 119/120 sycophancy responses
were truncated for the integrity read. The historical hook also edits the final
context token once during prefill, so it is not strictly answer-only.

Among the stored **strict answer-only** windows, sycophancy has one supported
high-integrity match: answer token 2. At virtually identical integrity, context
steering is stronger:

| Sycophancy arm | Integrity | Pass ≥80 | Trait Δ vs historical α0 |
|---|---:|---:|---:|
| reverse-map context, L14 c2 | 99.35 | 100.0% | +12.95 |
| (r_B) at answer token 2, L14 c2 | 98.95 | 100.0% | +2.78 |

The integrity gap is −0.41 points (multiplicity-adjusted 99.75% CI
[−1.45, +0.46]); the pass-rate gap is exactly 0. The paired trait contrast is
**answer minus context = −10.18 points**, 95% CI [−16.97, −3.92] and
candidate-selection-safe familywise CI [−20.08, −1.53], over all 120 common
question/effective-seed coordinates, conditional on the previously selected
context operating point. Thus the directly fitted reverse-map
context intervention is about 10 points stronger than the matched-quality
single-answer-token intervention for sycophancy.

Evil has **no stored quality-matched strict answer-only comparison**. The closest
candidate, answer tokens 1–5, is still +20.66 integrity points and +35.8
percentage points in pass rate above the context target. Its trait effect is
+4.25 versus +9.83 for context, but that difference is not an equal-integrity
estimate. The historical evil all-answer effect (+98.88) comes with integrity
0.43 and a 0% pass rate, so it cannot answer the matched question.

## Complete strict-answer frontier

| Behavior | Answer window | Integrity | Pass ≥80 | Trait Δ vs historical α0 | Trait minus context |
|---|---|---:|---:|---:|---:|
| Evil | token 1 | 94.05 | 96.7% | +0.00 | −9.83 |
| Evil | token 2 | 94.65 | 96.7% | +0.02 | −9.86 |
| Evil | token 3 | 95.01 | 95.0% | +0.00 | −9.83 |
| Evil | tokens 1–3 | 90.00 | 95.8% | +2.03 | −7.68 |
| Evil | tokens 1–5 | 87.49 | 92.5% | +4.25 | −5.55 |
| Sycophancy | token 1 | 97.05 | 96.7% | +5.47 | −7.49 |
| Sycophancy | token 2 | 98.95 | 100.0% | +2.78 | **−10.18** |
| Sycophancy | token 3 | 97.72 | 97.5% | +1.81 | −11.15 |
| Sycophancy | tokens 1–3 | 94.63 | 95.0% | +8.94 | −4.02 |
| Sycophancy | tokens 1–5 | 94.14 | 95.8% | +10.17 | −2.79 |

Only sycophancy token 2 passes both the simultaneous integrity-equivalence test
and the ≥90% high-integrity floor. The other trait contrasts remain useful
frontier points, not matched headline estimates.

![Integrity/trait frontier](https://raw.githubusercontent.com/superkaiba/explore-persona-space/issue-2254/figures/issue_2254/response_integrity_matched_steering/quality_trait_frontier.png)

## Method

The analysis uses 14 stored cells: two frozen reverse-map context targets; five
same-layer, same-per-edit-dose strict answer-only windows per behavior; and two
historical all-answer references. The nominal 200-row context files were
deduplicated to the six effective RNG seeds 42–47 shared with the answer files,
giving 20 questions × 6 outputs = 120 unique responses per cell and 1,680 total.
No response was regenerated.

The response-integrity rubric sees the question and answer and scores fluency,
non-degeneration, topical connection, structural completion, and expected
language/script continuity. It explicitly ignores factual correctness, safety,
helpfulness, refusal, agreement, and evil or sycophantic stance. The five-pass,
185-rating pilot passed every prespecified anchor: opposing evil and
sycophancy stances all averaged 100, while word salad scored 11.2,
wrong-language 35.4, off-topic 16.0, and abrupt truncation 63.2.

Production obtained all **8,400/8,400** planned ratings. One invocation was
blocked by provider content screening and succeeded on retry; there was no
persisted grade loss or malformed/schema output. The median
pairwise repeat correlation was 0.996 overall (evil 0.993; sycophancy 0.999),
and the mean within-response SD was 1.46 points.

Matching was performed at the intervention-cell level using integrity only.
Individual responses were not filtered, because integrity is post-treatment and
survivor filtering would induce selection bias. Equivalence requires both the
mean-integrity and pass-rate intervals to lie within ±5 points and ±5 percentage
points, respectively. The 20 quality comparisons use 99.75% Bonferroni-adjusted
intervals from 200,000 paired-question bootstrap draws. Direct trait contrasts
use common effective-seed coordinates with existing Sonnet scores; simultaneous
99.5% trait intervals cover the two-behavior × five-candidate family.
They protect selection among those ten strict answer candidates, conditional
on the context operating points selected in the earlier experiment; they do not
undo that earlier context-cell selection.

The α0 component deltas are historical orientation values, not exact
effective-seed matches: only nominal-200-row alpha-zero question summaries
survive, which double-weight effective seeds 43–46. This does not affect the
primary direct answer-minus-context contrast, where α0 cancels.

## Scope and next experiment

This fixes the quality confound for the comparison supported by stored strict
answer-window outputs. It does **not** establish an equal-quality comparison
against a pure intervention at every generated answer token:

- the existing all-answer hook includes one context-token edit;
- its available operating points have no integrity overlap with context steering;
- its full-response trait horizon differs from the common integrity horizon; and
- sycophancy all-answer trait coverage is only 95/120 items (79.2%; 73.8% of
  judge draws).

A definitive all-generated-token comparison therefore needs new generation with
a pure decode-only hook and a lower-dose ladder selected by response integrity
alone. That is the remaining experimental step, not something that can be
recovered by filtering the current outputs.

## Artifacts

- [Methodology](../methodology/issue_2254_response_integrity_matching.md)
- [Reduced result](../../eval_results/issue_2254/response_integrity_matched_steering/codex_subagent_v3/reduce/matched_results.json)
- [Figure metadata](../../figures/issue_2254/response_integrity_matched_steering/quality_trait_frontier.meta.json)
- [Persisted raw judge records](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/a78c323d085802227c9e9e0790b23a6ee981b431/issue2254_preimage/response_integrity_matched_steering)
