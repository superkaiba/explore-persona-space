# Issue 2254 response-integrity-matched steering

## Purpose

This post-hoc analysis fixes a nuisance-variable confound in the comparison of
reverse-map context steering and answer-derived persona steering. The earlier
programmatic coherence flag admitted fluent but unrequested language switches,
off-topic continuations, repetition loops, and malformed text. Trait movement
from such outputs cannot be compared fairly with trait movement from intact
answers merely because both pass a form-only flag.

No model responses are regenerated. A fresh Codex-subagent instrument judges
the stored responses using the user question and a single 0–100 **response
integrity** construct: linguistic integrity, continuity, task connection, and
expected-language/script continuity. It explicitly ignores truth, safety,
helpfulness, agreement with the user, refusal, and evil or sycophantic stance.
The instrument is an exploratory sensitivity analysis, not the project's
official Sonnet trait judge.

## Fixed roster

The context policy is frozen before integrity judging at the previously selected
reverse-map cells: evil `rvm/context/L14/c4` and sycophancy
`rvm/context/L14/c2`. Each is compared with the answer-derived persona vector at
the same layer and per-edit dose. Strict answer-only candidates edit answer token
1, 2, or 3, or answer spans 1–3 or 1–5.

The historical `allans` output is graded and reported separately, but cannot be
the strict primary comparator: its hook also edits the final context token once
during prefill. It is additionally incompatible with the primary trait frontier
because its regenerated outputs run to 4,096 tokens, whereas its integrity score
is measured on the common first-2,048-token horizon. A definitive comparison
against a pure all-generated-token hook, or an all-answer dose match if the
stored cell lacks support, requires new generation.

The reverse-map files nominally contain 200 rows, but 80 are duplicated across
base seeds because the effective generation seed is `base_seed + draw_index`.
The canonical context roster retains `(seed 42, draws 0–4)` and `(seed 43, draw
4)`. The answer roster retains `(seed 42, draws 0–5)`. Both therefore contain
the same six unique effective seeds 42–47 for each of 20 questions: 120 unique
responses per cell and 1,680 responses across 14 cells.

## Prespecified integrity rules

Each response receives five fresh-session procedural grades. Their average is
the response score. These repeats are not claimed to be statistically
independent. A response passes at 80; thresholds 70 and 90 are reported as
sensitivity cuts. The following are exploratory rubric-anchor conventions, not
literature-derived constants:

- descriptive balance: absolute mean-score gap at most 5 points and absolute
  pass-rate gap at most 5 percentage points;
- supported balance: the same point conditions, and both Bonferroni-adjusted
  confidence intervals lie wholly inside the ±5-point and ±5-percentage-point
  equivalence margins;
- high-integrity match: a descriptive or supported balance, respectively, with
  both cells also at least 90% passing;
- family: two behaviors × five answer candidates × two quality metrics = 20
  comparisons, giving 99.75% per-comparison intervals for 95% familywise
  coverage;
- tie-break among supported candidates: smallest absolute mean gap, then
  smallest absolute pass-rate gap, then fewer edited answer tokens.

If no candidate meets the supported-balance rule, the result is “no supported
matched comparison”; the nearest cell is descriptive only. A quality-balanced
pair below the 90% floor controls relative integrity but does not establish
clean steering, and is labeled accordingly. The analysis will not move either
margin after seeing results.

## Trait estimand and inference

Selection sees response-integrity scores only. Trait is read from the existing
project Sonnet judgments after selection. The direct comparison retains only
effective-seed coordinates with at least one trait score in both cells and
reports the exact paired count; this is an available-case treatment of the
small amount of original judge missingness, not an integrity filter. The direct
estimand is

\[
D_b = T(A_b^*) - T(C_b),
\]

which is also the difference between the two effects relative to the common
unsteered baseline because that baseline cancels. Component effects versus the
published alpha-zero baseline are reported for orientation only: the available
baseline summary uses the nominal 200-row design and therefore double-weights
effective seeds 43–46 rather than matching the canonical six-seed roster.

No completion is filtered on its integrity score. Such filtering would
condition on a post-treatment variable and create survivorship bias. Existing
trait-judge missingness is reported; a cell requires at least 95% item
completeness. The invalid historical sycophancy all-answer cell is record-only.

Five integrity judgments are averaged within response, six effective seeds are
averaged within question, and uncertainty is obtained by resampling the 20
paired questions with 200,000 deterministic bootstrap draws. The direct trait
contrasts include both ordinary 95% intervals and 99.5% per-comparison
Bonferroni intervals across the two-behavior × five-candidate family; the latter
retain familywise coverage for a quality-selected cell. Candidate reselection
inside each bootstrap resample is
reported only as a conditional selection-stability diagnostic, not as a
post-selection confidence interval. The complete integrity/trait frontier is
reported alongside any selected cell.
