# Codex (gpt-5.5) — Statistics & Measurement lens — VERDICT: REVISE (plan v2)

Ensemble state for this lens: Claude REVISE + Codex REVISE = FAIL+FAIL. Blockers are PARTLY
DISJOINT, so they UNION into the Phase 3 round (no reconciler — reconciler fires only on
PASS-vs-FAIL disagreement).

## MUST-FIX C1 [SQ-A] — registered P0 split-count read addresses nonexistent JSON paths
FOURTH independent discovery of the same defect (Claude Methodology, Claude Statistics,
consistency-checker, now Codex). `.splits.train_10k` / `.splits.val_400` / `.splits.test_1000`
are the real paths; the registered command reads top-level and produces 0/0/0, so P0 either
fails or records FALSE inputs for the claimed n/d checks.
FIX: read `.splits.*`, assert 10,000/400/1,000, cross-check against `.counts.*` before any fit.
[Orchestrator note: the parenthesized or `.counts` form is required — the naive comma-chained
repair still breaks on jq precedence. Verified; see the Methodology verdict file.]

## MUST-FIX C2 [SQ-B] — the calibration-cancellation claim is ALGEBRAICALLY FALSE, and the
## lattice consumes the raw contrast because of it
NEW, and distinct from the Claude twin's "calibration is near-inert" concern — these are
complementary, not duplicates. §6 says calibration is per-model excess over that model's P=200
null mean, then claims equal widths make the calibration constants CANCEL so raw Δacc@1 CIs are
"exact". Algebraically:
    Δcal = (acc_3.8 − acc_3.5) − (μ_null,3.8 − μ_null,3.5)
Equal hidden dimension does NOT force equal REALIZED null means: representations, selected
layers, and selected λ all differ across the three column models. So the two null means do not
cancel, and a near-zero result can enter the WRONG VERDICT BRANCH.
FIX: define one calibration scale for Δcol, subtract each cell's realized null mean, and shift
or recompute the paired bootstrap CI on that calibrated statistic; report null Monte Carlo
uncertainty separately.
Mechanizable check: recompute raw and calibrated endpoint contrasts from the cell fit/null JSONs
and assert the decision lattice consumes the CALIBRATED fields.

Note how this interacts with the Claude twin's finding: Claude showed the retrieval null sits
≈ chance (~0.001) for every cell, so the cancellation error is SMALL in magnitude — but "small"
is not "exact", and the lattice's near-zero branch is exactly where a small error flips a
verdict. Both critics' fixes point the same way: stop asserting cancellation, subtract the
realized null means, and prefer the excess-over-null-mean form.

## MUST-FIX C3 [SQ-D/SQ-F] — the verdict lattice tests ENDPOINT SUPERIORITY, not the registered
## capability-tracking hypothesis
NEW, and a genuine pre-registration defect. H1 requires mapping quality to rise "across the
panel AND within the fixed-size 27B column", and §0.0's TL;DR says the three same-size models
must "line up cleanly" — but the `Capability-tracks` predicate requires only a POSITIVE 3.8−3.5
ENDPOINT CI. A reachable ordering (3.5 < 3.8, with 3.6 BELOW BOTH) would be labeled
Capability-tracks while contradicting monotonic increase. The 3.6 checkpoint is omitted from the
predicate entirely, and the panel Spearman is likewise absent from it.
Also: paired resampling over PROMPTS establishes a difference between these fixed checkpoints,
not a replicated MODEL-LEVEL capability trend.
FIX: either narrow the headline and the verdict label to "3.8 exceeds 3.5 on this prompt
distribution", or register an ordered-column statistic/predicate that incorporates 3.6 — and
keep the panel-wide association explicitly descriptive.
Mechanizable check: enumerate synthetic three-point orderings and assert every state labeled
Capability-tracks satisfies the final registered ordering predicate.

## WHAT CODEX AGREED IS SOUND
"The core estimator is otherwise well posed": committed splits contain 10,000/400/1,000 rows;
the plan-text maximum hidden dimension is 5,120 (Codex independently confirms the brief's 8,192
was wrong), so min n/d is correctly 1.95; GPQA is transfer-only; the reused ridge implementation
performs explicit validation-selected λ rather than GCV and records λ/edge diagnostics; layer
selection is legitimately frozen on an independent validation split before test/null reads;
cross-width R² is confined to diagnostics; both mapping baselines and vector-pooling conventions
are registered.

## NICE-TO-HAVES / ANALYZER CONCERNS
- [SQ-E] Confirm per-dimension length regressions are fitted ONLY on generic TRAINING rows and
  applied unchanged to val/test/GPQA. Fitting nuisance regressions on test/GPQA activations
  would LEAK the evaluation targets and violate the declared transfer-only surface; persisted
  captures make this recoverable. [CONVERGES with Claude Statistics MF2(ii)/(iii) — together
  these two make the residualization protocol a settled must-register.]
- [SQ-E] The GPQA table gives 5/990 as the chance rate beside BOTH retrieval reads. That is
  correct for same-question acc@1; exact-row acc@1 has chance 1/990. The hard-surface H2 result
  must name WHICH read carries `gap_GPQA`.
- [SQ-A] Verify every permutation refit repeats validation-based λ selection and persists
  selected λ / grid-edge status PER DRAW; the stated per-draw artifact contract currently names
  only acc@k and R².
- [SQ-C/SQ-D] Effective rank is only exploratory, and 7 of 9 H2 checkpoint contrasts come from
  closely related Qwen models, so the calibrated panel Spearman and Wilcoxon reads should stay
  panel-descriptive rather than family-general. [CONVERGES with both Claude critics on the
  7/2 split disclosure.]
