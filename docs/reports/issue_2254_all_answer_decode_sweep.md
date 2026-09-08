# Pure all-answer-token steering versus context-only steering

## Result

The lower-dose experiment fixes the historical comparison mechanically: the answer-vector
intervention leaves prefill untouched, edits every cached decode step, uses the same model,
layer, questions, seeds, sampling settings, 2,048-token horizon, and LLM judges as freshly
generated context-only comparators. Because the first answer token is sampled from the
unmodified prefill pass, “all answer tokens” here means that the edit can affect output token 2
and every later token.

The prespecified analysis did **not** identify a confirmed equal-integrity comparison for either
behavior:

- **Evil:** no nonzero answer dose was even point-matched to the context target on the q0–9
  selection split. The frontier crosses the context target between tested doses: at `c=1/2`,
  answer steering has much higher integrity and much higher trait expression; at `c=1`, it has
  lower integrity and still much higher trait expression. These operating-point outcomes favor
  the answer route descriptively, but they do not identify per-edit efficiency or the requested
  same-quality estimate.
- **Sycophancy:** `c=1/2` was selected using integrity alone on q0–9. On held-out q10–19 its
  integrity gap versus context was −2.957 points, simultaneous interval
  [−6.774, −0.547], and its pass-rate gap was −3.33 percentage points, simultaneous interval
  [−11.67, 0]. The point estimates are close, but both intervals extend outside the frozen
  ±5-point and ±5-percentage-point equivalence margins. Quality equivalence therefore failed.
  The trait contrast at this dose was **answer minus context = +8.50** on q10–19, ordinary 95%
  interval [4.38, 12.78] and familywise interval [2.44, 15.08], but the registered rule leaves
  this descriptive because the quality gate did not confirm.

Across all 20 questions, the selected sycophancy answer dose has integrity 97.64 and trait score
22.66, versus 99.27 and 12.87 for context-only steering. Its exploratory answer-minus-context
trait contrast is +9.79 with familywise interval [4.42, 16.67]. This is the clearest practical
read: pure all-answer steering produces more sycophancy with a small observed integrity cost, but
the experiment does not certify that cost as lying within the prespecified equivalence margin.

![Quality and trait frontier](https://raw.githubusercontent.com/superkaiba/explore-persona-space/373547b88233698c0bc5b9344e8bf1e1b7a650b1/figures/issue_2254/all_answer_decode_sweep/quality_trait_frontier.png)

*Figure 1. Full 20-question descriptive frontier. Circles are pure decode-only answer-vector
doses; diamonds are the fixed reverse-map context targets; the ring marks the integrity-selected
sycophancy dose. Labels show dose multiplier `c`; clustered `0–1/8` labels cover all plotted
doses in that range. Whiskers are ordinary 95% question-cluster bootstrap intervals. The figure
is also available as a [local PDF](../../figures/issue_2254/all_answer_decode_sweep/quality_trait_frontier.pdf).*

## Complete frontier

Trait scores are refusal-aware intention-to-treat means on a 0–100 scale; an explicit rubric
`REFUSAL` contributes zero. `Δ trait` is relative to the behavior's pure decode-only zero-dose
cell. These full-20 estimates are descriptive; the split-sample decision above is primary.

| Behavior | Route | Dose `c` | Integrity | Pass ≥80 | Trait | Δ trait |
|---|---|---:|---:|---:|---:|---:|
| Evil | Context-only target | 4 | 69.32 | 57.5% | 11.77 | +11.74 |
| Evil | All-answer | 0 | 98.42 | 98.3% | 0.03 | 0.00 |
| Evil | All-answer | 1/64 | 97.92 | 98.3% | 0.09 | +0.06 |
| Evil | All-answer | 1/32 | 98.34 | 98.3% | 0.07 | +0.04 |
| Evil | All-answer | 1/16 | 98.07 | 98.3% | 0.06 | +0.03 |
| Evil | All-answer | 1/8 | 98.31 | 98.3% | 0.08 | +0.05 |
| Evil | All-answer | 1/4 | 96.75 | 95.8% | 10.56 | +10.53 |
| Evil | All-answer | 1/2 | 90.92 | 89.2% | 71.00 | +70.97 |
| Evil | All-answer | 1 | 62.12 | 20.8% | 88.79 | +88.76 |
| Evil | All-answer | 2 | 1.29 | 0.0% | 98.55 | +98.52 |
| Evil | All-answer | 4 | 1.07 | 0.0% | 97.54 | +97.51 |
| Sycophancy | Context-only target | 2 | 99.27 | 100.0% | 12.87 | +9.16 |
| Sycophancy | All-answer | 0 | 98.54 | 99.2% | 3.71 | 0.00 |
| Sycophancy | All-answer | 1/64 | 98.42 | 98.3% | 3.61 | −0.10 |
| Sycophancy | All-answer | 1/32 | 98.82 | 99.2% | 3.71 | 0.00 |
| Sycophancy | All-answer | 1/16 | 98.83 | 99.2% | 4.40 | +0.70 |
| Sycophancy | All-answer | 1/8 | 98.46 | 98.3% | 5.70 | +2.00 |
| Sycophancy | All-answer | 1/4 | 98.36 | 98.3% | 9.27 | +5.56 |
| Sycophancy | All-answer, selected | 1/2 | 97.64 | 98.3% | 22.66 | +18.96 |
| Sycophancy | All-answer | 1 | 85.76 | 85.0% | 69.89 | +66.19 |
| Sycophancy | All-answer | 2 | 4.81 | 0.0% | 97.66 | +93.96 |
| Sycophancy | All-answer | 4 | 1.77 | 0.0% | 93.55 | +89.84 |

The evil frontier is especially steep. `c=1/2` Pareto-dominates the fixed context point in the
full-20 descriptive data—integrity 90.92 versus 69.32 and trait 71.00 versus 11.77—but it is not
an equal-integrity match. At `c=1`, integrity has already fallen below the context target. A
denser dose grid between these values would be needed to target the same integrity directly.

## Degeneration diagnostics

The lower doses resolve the old “large effect equals destroyed answer” confound, but the high
doses reproduce it sharply. Evil `c=1/2` has no cap hits, 8.3% CJK-script flags, and no repetition
flags; sycophancy `c=1/2` has no cap hits or repetition flags and 2.5% CJK flags. By contrast,
both `c=2` cells are 100% or nearly 100% CJK-flagged and 99.2% cap-hit. The sycophancy `c=2`
cell also has a 45.8% repetition rate. These high-dose trait scores are therefore frontier
diagnostics, not usable steering operating points.

The fixed evil context target itself is imperfect: 25.8% of responses contain a CJK-script
character, 1.7% hit the token cap, and 1.7% meet the exact repetition flag. This helps explain why
no low-dose answer cell has the *same* poor integrity. The sycophancy context target is clean by
comparison: 0.8% CJK flags and no cap or repetition flags.

## Design and inference

The experiment uses `Qwen/Qwen2.5-7B-Instruct` at layer 14. The answer route sweeps
`c ∈ {0, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4}`; the context targets are the earlier
reverse-map operating points, evil `c=4` and sycophancy `c=2`, regenerated in the same runtime.
Each of the 22 analysis cells contains 20 questions × six effective seeds, for 2,640 responses.
Two additional no-hook bridge cells establish exactly matching zero-dose behavior, bringing the
generation total to 2,880 responses.

Each analyzed response received five fresh-session response-integrity judgments and five
fresh-session behavior-specific trait judgments from tool-free `gpt-5.6-sol` sessions at low
reasoning effort: 13,200 integrity decisions plus 13,200 trait decisions. The integrity rubric is
question-aware and scores fluency, non-degeneration, topical connection, structural completion,
and expected language/script continuity while explicitly ignoring trait stance, correctness,
safety, helpfulness, agreement, and refusal. Procedural repeats are not claimed to be independent
judge samples.

Quality-only selection uses questions 0–9 and never reads trait scores. Questions 10–19 confirm
quality and would test the selected trait contrast only if quality confirms. Means first average
the six seeds within question, then use 200,000 deterministic paired-question bootstrap draws.
Quality intervals have 95% familywise coverage over two behaviors × nine nonzero doses × two
quality metrics; trait intervals cover the 18 answer-minus-context contrasts. Individual
responses are never filtered by their post-treatment integrity.

Structured completeness is 100%. Evil numeric-only trait item completeness is at least 97.5% in
every cell and sycophancy is 100%; all cells clear the frozen 95% sensitivity floor. The primary
ITT keeps all structured judgments by mapping explicit trait refusals to zero. Sycophancy has no
trait refusals; evil refusal fractions range from 0% to 4.17%, with 0.5% in the context target.

## Recovery and sensitivity audit

Three content-bearing grading responses were invalid: one integrity packet duplicated/omitted
IDs, and two trait packets returned the exact ID set in the wrong order. All 122 rows were
discarded wholesale and replaced through immutable, receipt-bound whole-packet recoveries; none
was sorted, salvaged, or used in any reducer or analysis. Two evil trait packets that repeatedly encountered a
response-free provider policy block were replaced by their frozen singleton rosters.

All seven registered trait administration sensitivities are estimable. The broadest removes
182 recovery-associated decisions across 179 items, retains 13,018 decisions, and leaves at
least three repeats per item. Its maximum absolute change in any cell's ITT mean is 0.111 points
for evil and 0.067 for sycophancy. The selected sycophancy `c=1/2` held-out trait contrast is
mechanically unchanged at +8.50 in every scenario, and the frozen quality conclusion cannot
change because integrity-only selection and quality confirmation remain frozen and are never
recomputed or reselected using trait data.

Independent Codex artifact review verified all 810 canonical jobs, 820 schemas, exact decision
coverage, all recovery chains, zero use of rejected rows, and local equality to the uploaded
generation bundle. Independent statistical review is recorded separately in the task audit.

## Interpretation and limitations

The experiment establishes a dose-dependent tradeoff, not a confirmed equal-quality winner. At
the tested evil `c=1/2` point, all-answer steering produced a substantially higher trait score
than the fixed context operating point while retaining higher integrity. The sycophancy `c=1/2`
point shows a modest integrity reduction alongside a clear descriptive trait advantage. These
are operating-point comparisons, not estimates of route-level or per-edit efficiency. The
registered equivalence gate answers the user's exact matched-quality question conservatively:
**neither behavior produced a quality-confirmed comparison.**

This conclusion is limited to one model, one layer, 20 question clusters per behavior, fixed
historical context operating points, stochastic Codex procedural repeats, and a Codex rather than
the project's historical Sonnet judge. The first output token cannot be changed without editing
the prompt-shaped prefill forward; the pure implementation therefore begins influencing token 2.
For evil, a denser adaptive sweep between `c=1/2` and `c=1` could locate a closer integrity match,
but that would be a new experiment with a newly frozen selection procedure.

## Artifacts

- [Primary reduced result](../../eval_results/issue_2254/all_answer_decode_sweep/analysis/codex_subagent_v1/reduce/matched_results.json)
- [Trait administration sensitivity](../../eval_results/issue_2254/all_answer_decode_sweep/analysis/codex_subagent_v1/reduce/trait_administration_sensitivity.json)
- [Frozen quality selection](../../eval_results/issue_2254/all_answer_decode_sweep/analysis/codex_subagent_v1/selection/quality_only_selection.json)
- [Figure provenance](../../figures/issue_2254/all_answer_decode_sweep/quality_trait_frontier.meta.json)
- [Raw generations](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/609adc001340c8b1ee3205e6e9307972fb479d41/issue2254_preimage/all_answer_decode_sweep/raw_completions/all)
- [Judgment audit pack](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/b08ab782d094ae51bb5f3fc32e780641f361f40c/issue2254_preimage/all_answer_decode_sweep/analysis/codex_subagent_v1/audit_pack)
