---
description: "Selection-symmetric nulls — a null/permutation band compared against a max-over-axis-selected observed statistic must inherit the SAME selection per draw, or the axis must be frozen on a held-out split; persist the per-draw × per-axis matrix. Report every registered band's upper bound against the DV's achievable ceiling — band ≥ ceiling ⇒ uninformative-by-construction, narrate failure-to-reject. A bootstrap CI reported at a max-selected axis position must be the selection-inherited CI (per-draw re-selection inside each resample) or both frozen+inherited, labeled. ALSO: noise-structure symmetry — a difference-vector DV whose two legs subtract the SAME sampled baseline B̄ (cos(X−B̄, Y−B̄)) inflates against a noise-free null; use disjoint baseline draw halves per leg, or build the null to carry the identical shared-B̄ term per draw. Loads on-demand at plan time."
paths:
  - ".claude/plans/**"
  - "tasks/**/plans/**"
---

# Selection-symmetric nulls (max-over-axis inference & shared-baseline noise structure)

**Load this rule whenever a plan's headline statistic is chosen by
`max` / `argmax` / best-of / top-k-mean over a FREE AXIS** — a read-out
layer, a cell, a k / neighbourhood size, a seed, an extraction point, a
threshold — AND that statistic is compared against a null / permutation /
shuffle band. ALSO load it whenever a plan registers a DIFFERENCE-VECTOR
statistic whose observed and reference legs share ONE SAMPLED quantity —
cos(X − B̄, Y − B̄) with a shared empirical baseline mean, correlations of
change scores sharing a baseline, frac-of-anchor ratios with a sampled
anchor — compared against any null or reference band (§ Noise-structure
symmetry below). The always-on backstop is `planner.md` §6 "Selection-symmetric
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

## Bootstrap CIs at a selected axis position (selection-inherited CI)

The same asymmetry contaminates INTERVAL estimates, not just null
bands: a bootstrap / resampling CI computed with the axis FROZEN at the
winning position conditions on the selection — every resample re-reads
the one pre-selected position, so the interval carries only the
sampling variability AT that position and none of the variability OF
the selection itself. At a winner's-curse-selected position the frozen
CI overstates stability, sign stability included. This is the
interval-estimate face of post-selection inference: an interval at a
data-selected target is valid only when the selection is accounted for
(post-selection inference: Taylor & Tibshirani, PNAS 2015; bootstrap
after model selection: arXiv 1506.06266).

**A bootstrap / resampling CI reported at a `max` / `argmax` / best-of
selected axis position MUST be the selection-inherited CI** — the
per-draw re-selection rides INSIDE each bootstrap resample (each
resample re-runs the same `max`/`argmax` over the axis before the
statistic is read) — **or BOTH CIs are shown with explicit labels**
(`frozen-at-<axis position>` vs `selection-inherited`), never the
frozen CI alone. A frozen-only CI at a selected position is a REVISE at
plan / critic time and a binding revision request at interpretation
time. Sign- or effect-stability claims read off the selection-inherited
CI; the frozen CI answers only the conditional question "how variable
is the statistic at THIS position, taking the position as given" and is
labeled as such wherever it is shown. The inherited CI widens the
interval to carry the selection variability but does NOT debias the
point estimate — each resample's re-selected max is itself
winner's-curse-inflated, so effect-MAGNITUDE claims at the winner
remain optimistic even under inheritance; never quote the inherited
CI's center as a corrected estimate.

Persist the per-bootstrap-draw × per-axis statistic matrix under the
same contract as the null-band matrix above (bootstrap resamples as the
draws) — the selection-inherited CI is then a pure re-reduction,
recomputable post-hoc.

The held-out-freeze alternative (option 2 above) remains valid for CIs:
an axis position frozen on a held-out split, or pre-registered before
seeing the data, is not selection-conditioned and its frozen CI needs
no inheritance — state the freeze provenance when reporting it.

**Motivating evidence (#1434, install-grid ρ).** The interpretation
reported the frozen-headline-layer bootstrap CI [−0.949, −0.467] for a
ρ whose layer was itself chosen by max-|ρ| over 28 layers, while the
SAME JSON (`pv_validation.json`) carried the selection-inherited
cluster bootstrap (`cluster_bootstrap_selection_inherited`, per-draw
max-|ρ| layer re-selection inside each resample): [−0.957, +0.866] —
spanning zero widely. The frozen CI overstated sign stability at a
winner's-curse-selected layer; caught only at interpretation-critique.
This clause moves the catch to plan time and analyzer time.

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

## Noise-structure symmetry (shared-baseline difference vectors)

Selection symmetry is not the only way an observed-vs-null comparison
goes asymmetric: the null must also bear the same NOISE STRUCTURE the
observed statistic bears.

**The trap.** A DV of the form `cos(X − B̄, Y − B̄)` — or any statistic
whose observed and reference legs share ONE SAMPLED quantity: a shared
empirical baseline mean B̄ (e.g. the mean of 10 baseline draws), a shared
anchor, a shared sampled denominator (frac-of-anchor), a change-score
correlation sharing a baseline — carries the SAME sampling-noise vector
−ε_B in both legs. The shared term contributes ≈ +tr(Σ_B)/n_B to the
expected inner product whenever the X / Y / B NOISES are mutually
independent (no signal relation is needed at all), so the
cosine/correlation is biased toward the positive shared-noise limit — a
positive bias on the expected inner product (a negative true alignment is
pulled toward zero rather than magnified; the null-coverage failure holds
either way); when a leg's true signal is small (low split-half
reliability), the artifact DOMINATES the read. A null built
WITHOUT that shared term — norm-matched random directions, shuffled pairs
re-centered independently — has zero shared component with the reference
leg by construction, so it under-covers and the observed statistic
"clears the null" for free. This is the classical
spurious-correlation-from-a-shared-term family (Pearson 1897, indices
sharing a common component; in-project siblings: the #383 X-vs-(X−Y)
caveat, the install-fraction shared-noisy-denominator caveat).

**The two symmetric fixes (pick one; the plan registers which):**

1. **Disjoint baseline halves (default; the #1415 recount recipe).**
   Split B's draws into disjoint halves and feed one half to each leg
   (e.g. target from even draws, realized shift from odd draws). The
   shared cross-term vanishes in expectation — unbiased for the INNER
   PRODUCT; the disjoint COSINE remains attenuated toward zero (each
   half carries more independent noise), while the shared-B read
   carries the same attenuation PLUS the inflation — so the truth
   typically (not provably) sits between the two reads, and can sit
   above both when attenuation dominates the shared term. Report both
   reads and the caveat in both directions. Fix availability is
   member-dependent: when B is a single measurement per unit (no
   multi-draw mean), fix 1 is unavailable — use fix 2, preserving the
   per-unit shared-B contribution.
2. **A shared-B-bearing null.** Construct the null so EVERY draw carries
   the identical shared-baseline structure — the null replacement enters
   at the PRE-SUBTRACTION leg level (replace the raw leg X, never the
   already-differenced X − B̄), norm/scale-matched to the observed raw
   leg, and then the same sampled B̄ (preferably a per-draw bootstrap
   resample B̄* of B's draws) is subtracted from BOTH legs of each null
   draw — so the null carries the same shared-noise cross-term at the
   same RELATIVE scale the observed cosine carries. A random-direction
   or independently-re-centered null does NOT qualify; nor does a
   "shared-B" null whose replacement enters post-subtraction or at a
   mismatched leg norm.

**Reliability screen (rides along).** Report the split-half reliability
of any SAMPLED difference-vector leg (target and realized). A
near-zero-reliability leg makes its row uninterpretable regardless of
the fix — #1415's medical_doctor pair posted 0.66–0.71 alignment against
a target with 0.049 split-half reliability.

**Motivating evidence (#1415, H1 answer-shift alignment).** The realized
shift (V_a_steered − V_a(c)) and the target (V_a(c′) − V_a(c)) subtracted
the SAME 10-draw baseline mean; the 500-direction random null had no
shared term. The interpretation-critic's executed disjoint recount
(even/odd baseline halves, canonical L20/α4 cells): prefix mean
max-over-read-layers 0.271 → 0.178, context 0.362 → 0.272; one pair
0.23 → −0.08 (fully artifactual); 6/28 prefix pairs fell below the null
p97.5 (0.043). The "28/28 pairs clear the random-direction null"
headline did not survive. The shared term also inflates frac-of-anchor
ratio reads (adds +‖ε_c‖²/‖target‖²). The defect survived the planner,
the critic ensemble, the implementer, and code review — caught only at
interpretation-critique; this section exists to move the catch to plan
time.

**Interaction with selection symmetry.** The two clauses are orthogonal
and can BOTH fire on one DV (#1415's H1 was max-over-read-layers AND
shared-baseline). Fixing one does not fix the other.

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
- **(Noise-structure) A deterministic / analytic / population baseline** —
  B carries no sampling noise (a fixed vector, a closed-form mean), so
  there is no shared noise term to inherit. State that B is noise-free.
- **(Noise-structure) Legs already independent** — the two legs already
  subtract independent baseline estimates (disjoint draws, separate runs),
  or the null already carries the shared-B̄ structure per draw (fix 2 —
  pre-subtraction leg level, norm-matched; a post-subtraction or
  norm-mismatched "shared-B" null does not qualify).
- **N/A when no registered statistic shares a sampled quantity across its
  observed and reference legs** — a plan MAY write "N/A — no
  shared-baseline difference-vector DV" to be explicit; absence of the
  pattern requires no declaration.

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
- Bootstrap-CI selection inheritance (plan side):
  `planner-section-reference.md` §6 "Selection-symmetric nulls" block
  registers any CI reported at a selected position as selection-inherited
  (or both, labeled); `critic-lens-reference.md` Statistics & Measurement
  item 11 + `statistics-critic.md` item 11 (v2) REVISE a frozen-only CI
  at a max-selected position.
- Bootstrap-CI selection inheritance (interpretation side): `analyzer.md`
  measurement-validity gate item 4 (full text:
  `analyzer-section-reference.md` § Step 1) + `interpretation-critic.md`
  § Alternative Explanations flag a frozen-at-winner CI quoted where the
  selection-inherited CI carries the stability claim (#1434).
- Noise-structure symmetry (plan side): `planner-section-reference.md` §6
  "Selection-symmetric nulls" block (noise-structure paragraph) — a plan
  registering a shared-sampled-baseline difference-vector statistic names
  which of the two fixes it uses; `critic-lens-reference.md` Statistics &
  Measurement item 11 + `statistics-critic.md` item 11 (v2) REVISE the
  pattern with neither fix registered.
- Noise-structure symmetry (code side): `.claude/rules/gotchas.md` carries
  a pointer bullet (loads when writing analysis code — the layer that
  covers inline / free-analysis rounds, which bypass the planner+critic
  stack).

## Files of record

Task body #810 (the band-vs-ceiling incident: band p97.5 = 0.800 vs an
achievable difference ceiling from ~0.857 max skill; p = 0.634 initially
narrated as an ordering fail);
Task body #778 (the origin incident + n=24 asymmetry numbers);
Task body #1434 (the frozen-vs-inherited bootstrap-CI incident: frozen
[−0.949, −0.467] vs selection-inherited [−0.957, +0.866] at a
max-|ρ|-over-28-layers-selected layer, both in `pv_validation.json`);
Task body #1415 (the shared-baseline noise incident: prefix 0.271→0.178,
context 0.362→0.272, one pair 0.23→−0.08, 6/28 prefix pairs below null
p97.5 0.043 under disjoint halves; the interp-critique v1 marker carries
the executed recount);
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
