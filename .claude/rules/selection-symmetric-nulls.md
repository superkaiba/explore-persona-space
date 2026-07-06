---
description: Selection-symmetric nulls — a null/permutation band compared against a max-over-axis-selected observed statistic must inherit the SAME selection per draw, or the axis must be frozen on a held-out split; persist the per-draw × per-axis matrix. Report every registered band's upper bound against the DV's achievable ceiling — band ≥ ceiling ⇒ uninformative-by-construction, narrate failure-to-reject. Loads on-demand at plan time.
paths:
  - ".claude/plans/**"
  - "tasks/**/plans/**"
---

# Selection-symmetric nulls (max-over-axis inference)

**Load this rule whenever a plan's headline statistic is chosen by
`max` / `argmax` / best-of / top-k-mean over a FREE AXIS** — a read-out
layer, a cell, a k / neighbourhood size, a seed, an extraction point, a
threshold — AND that statistic is compared against a null / permutation /
shuffle band. The always-on backstop is `planner.md` §6 "Selection-symmetric
nulls" + `critic.md` Statistics & Measurement lens item 11; this file is the
full recipe those entries point at.

## The asymmetry

If you select the OBSERVED statistic as `max` over L axis positions but
compute the null band at ONE fixed position, the observed value got L
chances to be large and each null draw got 1 — a 28-vs-1 comparison. The
observed statistic is inflated by the winner's curse (≈ `sqrt(2 ln L)·SE`
above a per-position value), so the observed-vs-null gap is manufactured
by the selection, not by the effect the null battery is meant to test.
The same asymmetry applies to `argmax`, best-of, and a top-k mean (the
top-k averaging shares the winner's-curse inflation — the k winners are
still the extremes of L draws).

**Motivating evidence (#778).** A persona-vectors replication
(arXiv 2507.21509) selected the matched direction's read-out layer by
`max(matched r)` over 28 layers, then computed each null direction's band
at that single matched-winning layer. Simulated at n=24: single-layer
null p97.5 |r| ≈ 0.48 vs honest max-over-layer p97.5 |r| ≈ 0.62. All three
Phase-2 lenses reconciled to REVISE — four of the six per-lens critics
caught the asymmetry independently (Statistics Claude+Codex, Methodology
Codex, Alternatives Codex); Methodology Claude and Alternatives Claude
missed it. Siblings: #664 (best-of-28 SNR clears a floor a fixed-layer
read does not), #545 (best-of-group winner's-curse inflation ≈
`sqrt(2 ln K)·SE`).

## The two symmetric alternatives (pick one)

A plan whose headline uses `max` / `argmax` / best-of / top-k-mean over a
free axis MUST do ONE of:

1. **Per-draw same-selection (default).** Every null / permutation /
   shuffle draw receives the IDENTICAL max-over-axis selection before the
   band is formed. The null distribution is `{ max_over_axis(stat under
   draw d) : d }`, so observed and null get the same L chances. The
   honest band is the quantile of that max-selected null distribution.
2. **Held-out axis freeze.** Choose the axis position on a HELD-OUT split
   (or a pre-registered fixed position committed before seeing the data),
   then read BOTH the observed statistic and every null draw at that
   single frozen position. Observed and null are again symmetric — both
   at one fixed position, no selection on either side.

Do NOT compute the observed statistic max-over-axis and the null at one
position. A per-axis heatmap is a DIAGNOSTIC display and does NOT
neutralise the asymmetry — the comparison statistic is still
`max-over-axis observed` vs `one-position null`.

## Persist the per-draw × per-axis matrix

Persist the full per-draw × per-axis statistic matrix (one matrix per
headline statistic) — observed row + one row per null/permutation draw,
one column per axis position — as a downstream artifact (HF data repo
`analysis_tensors/` per the Upload Policy, or an `eval_results/` JSON if
small). This lets the analyzer recompute the honest max-selected band
post-hoc even if the plan shipped the asymmetric read — the honest band
is a pure re-reduction of the stored matrix, no re-run. Without the matrix
the honest band is unrecoverable and the run must be repeated.

## Band-vs-ceiling informativeness check

A selection-symmetric band can still be UNINFORMATIVE-BY-CONSTRUCTION:
if the band's rejection threshold sits at or above the largest value the
statistic can actually attain, no data could ever clear it — the test
has zero power and a non-rejection carries no evidence (standard
power / severe-testing reasoning: an unreachable rejection region
cannot discriminate).

**Every registered null-band read REPORTS, side by side:**

1. the band's upper bound (e.g. the 97.5% quantile of the max-selected
   null distribution), and
2. the DV's **achievable ceiling** — the maximum value the statistic
   can attain given the estimator and data. Operationally:
   - a DIFFERENCE statistic (difference of skills / rates / effects) —
     this tier OVERRIDES the bounded-DV tier below: the maximum
     attainable value of the favored arm MINUS the exact comparison-arm
     quantity the statistic itself uses (NAME that quantity when
     registering the read — no "observed or baseline" analyst fork; the
     raw single-arm bound is NEVER the difference ceiling). This is a
     CONDITIONAL ceiling, not a hard bound — the comparison arm's
     realized value is random and shared with the observed statistic —
     so when the ceiling input is estimated, register a conservative
     interval for it rather than a point value;
   - a bounded DV with no differencing: the bound itself (a rate ≤ 1,
     |r| ≤ 1, a skill ≤ its max attainable skill);
   - when NO estimator bound is derivable: fall back to the largest
     previously-observed in-genre / in-line effect as a SEVERITY
     REFERENCE POINT. A reference point is NOT a ceiling — a new effect
     can exceed the historical max — and it NEVER triggers the
     uninformative-by-construction verdict below.

**If the band's upper bound ≥ the achievable ceiling (estimator-bound
tiers only — never the reference-point fallback), the test is
uninformative-by-construction:** zero power; treat a band within one
null-band SE below the ceiling as effectively unreachable (no
knife-edge reading). Then:

- any NON-REJECTION outcome MUST be narrated as **failure-to-reject**
  ("the test could not have detected any achievable effect") — NEVER as
  evidence of absence, a clean ordering fail, or a reversal. The
  mandate is scoped to the unreachable tail/direction: a separately
  REACHABLE lower-tail / opposite-direction rejection remains a
  legitimate finding and is not gagged by this check;
- the band AND the ceiling are drawn in the figure, so the
  unreachability is visible in the artifact itself;
- at PLAN time, prefer redesigning the read (a less-inflating
  selection, a tighter statistic, a larger n) over registering a
  decision gate that cannot fire — this is the null-band-specific
  instance of the Decision-Gates joint-satisfiability bar. (More null
  DRAWS do not help: draws refine the band's quantile ESTIMATE, not
  its location — the band is set by selection breadth and per-position
  SE.)

**If the band's upper bound ≥ the fallback severity reference point**
(no estimator bound derivable): the test is UNDERPOWERED against every
previously-observed in-genre effect — report the read as low-severity
(a non-rejection carries no evidence about effects of realistic
magnitude), NOT as zero-power / uninformative-by-construction. A plan
may rebut this by registering a justified expected effect above the
historical max.

**The converse is never blessed:** band upper bound < ceiling does NOT
demonstrate adequate power (a band at 99% of the ceiling passes the
check and is still nearly powerless). REPORT the band-to-ceiling
margin; this check is a one-sided detector of the zero-power case.

Persist the registered ceiling inputs (the named comparison-arm
quantity / bound derivation) alongside the per-draw × per-axis matrix,
so the band-vs-ceiling read is recomputable post-hoc.

The check fires at BOTH ends: the PLANNER registers band + ceiling next
to each null-band read (§6/§7; the critic REVISEs a registered
null-band decision gate whose band is unreachable under an
estimator-bound ceiling — reference-point exceedance is a Concern
surface, never this REVISE trigger — and raises a binding Concern on a
missing ceiling report for a bounded statistic); the ANALYZER /
interpretation side re-checks the REALIZED band against the ceiling
before narrating any non-rejection. The check applies to the BAND once
built, wherever it was loaded from — a carve-out exempting a read from
the selection-symmetry recipe does not exempt an unreachable band from
this check when another surface (the analyzer gate) has loaded it.

**Motivating evidence (#810, round `ultrachat-genre-summary-sweep`,
H1-g(iii)).** A registered difference-null band's 97.5% upper bound
(0.800) sat above the DV's achievable difference ceiling — the max
attainable skill was ~0.857, so a max-difference statistic could
essentially never clear 0.800; even the parent round's observed +0.209
Betley effect would fail — and the p = 0.634 outcome was initially
narrated as a clean ordering fail until the interpretation-critic
caught it.

## Carve-outs (this rule does NOT fire)

- **A single pre-registered / fixed axis position with no data-driven
  selection** — e.g. "read at layer 14, committed in advance" or a
  paper's own fixed layer. There is no `max`/`argmax` over the axis, so
  observed and null are already symmetric. State the fixed position and
  that it is pre-registered.
- **A legitimate single-anchor ablation** that picks ONE
  condition/layer for a mechanistic reason unrelated to maximising the
  headline statistic (e.g. "ablate the residual stream at the layer the
  paper localises the feature") — not a best-of search over the
  headline metric.
- **A per-axis result reported at EVERY position with no cross-position
  `max` headline** (a full layer sweep displayed as a curve/heatmap
  where the claim is the shape, not a single best position).
- **N/A when the headline statistic is not selected over any free
  axis** — the plan writes "N/A — no max-over-axis selection in the
  headline".

## Enforcement

- `planner.md` §6 "Selection-symmetric nulls" — the trigger + the two
  alternatives + persistence requirement a plan with a swept-axis
  headline must satisfy.
- `critic.md` Statistics & Measurement lens item 11 — REVISEs a plan
  whose headline uses `max`/`argmax`/best-of/top-k-mean over
  layer/cell/k/neighbourhood/seed/extraction-point/threshold AND whose
  null band is computed at one fixed axis position, with neither per-draw
  same-selection nor a held-out-frozen axis, and no persisted per-draw ×
  per-axis matrix.
- Band-vs-ceiling (plan side): `planner-section-reference.md` §6
  "Selection-symmetric nulls" block registers band upper bound +
  achievable ceiling per null-band read; `critic-lens-reference.md`
  Statistics & Measurement item 11 + `statistics-critic.md` item 11
  (v2) verify it.
- Band-vs-ceiling (interpretation side): `analyzer.md`
  measurement-validity gate + `interpretation-critic.md`
  § Alternative Explanations narrate an unreachable band as
  failure-to-reject.

## Files of record

Task body #810 (the band-vs-ceiling incident: band p97.5 = 0.800 vs an
achievable difference ceiling from ~0.857 max skill; p = 0.634 initially
narrated as an ordering fail);
Task body #778 (the origin incident + n=24 asymmetry numbers);
`.claude/agent-memory/reconciler/feedback_layer_selection_asymmetry_is_alternatives_finding.md`,
`.claude/agent-memory/analyzer/feedback_best_layer_snr_selection_biased.md`,
`.claude/agent-memory/critic/feedback_bestofgroup_selection_asymmetry.md`
(the pre-existing agent memories on this pattern);
`.claude/rules/contrastive-negatives.md` (#383 X-vs-(X−Y) sibling
caveat).

**Sibling rule:** `.claude/rules/vectorize-many-cell-fits.md` — the same #778
null battery is that rule's many-draw compute incident (per-draw pool
re-reduction → ~70× batched subset-sum GEMM); a permutation-battery plan
typically fires both.
