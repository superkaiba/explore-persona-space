---
name: Codex misreads a permutation-null architecture as same-sample selection optimism
description: Codex FAILs "select-champion-and-score-on-same-cells" lines as reintroduced argmax optimism without tracing that the select+score function is the stat_fn passed to a within-column permutation null — the null CONTAINS the optimism, so the p-value subtracts it. Trace stat_fn's caller before believing the leak.
type: feedback
---

**Rule:** When Codex's code-review FAILs a stat that "selects a champion on the
same cells it scores on" (argmax-over-tiny-group optimism reintroduced),
TRACE where that select+score function is CALLED before believing the leak.
The standard, CORRECT design for guarding selection optimism is: the observed
point estimate IS select-and-score-on-the-same-data, and the SAME procedure is
recomputed on every label-shuffled permutation to build the null. The verdict
gates on the **permutation p-value**, not the raw same-cell statistic — so the
null distribution itself contains the argmax optimism and the p subtracts it
out. A select+score function passed as `stat_fn` to a permutation wrapper is
NOT a leak; it is the textbook permutation test.

**Why:** #545 r1 — Codex raised TWO CRITICALs (FAIL, "reject-with-replan")
claiming H2/H3 and H1 "select champions and score on the same family cells,
not held-out cells → reintroduces the argmax-over-tiny-family optimism the null
was supposed to guard." But `_oracle_gain` (scoring.py:985) and `_heterogeneity`
(:1033) are passed as `stat_fn` to `_within_column_permutation` (:1029/:1046),
which recomputes the identical same-cell selection on P=10,000 within-column
row-label-shuffled targets and gates the verdict on the permutation p (:1052,
:1065). That matched plan §4.3(ii) verbatim ("A large gain INSIDE the
permutation null band = pure argmax optimism"). Codex saw
`_global_champion(...,cells)` + `tau(...,cells)` two lines apart and stopped —
it never read the function's caller. Both Criticals REJECTED; Claude was right
the held-out/null structure existed.

**How to apply:** On any Codex "same-sample selection optimism" / "no held-out
split" code-review blocker against a statistic that feeds a verdict, before
upholding: (1) grep for the cited function's name to find its callers; (2) if it
is passed to a `*permutation*` / `*null*` / `*bootstrap*` helper as a callable,
read that helper — if it recomputes `stat_fn(shuffled_target)` to build a null
and the verdict gates on the resulting p-value, the same-cell selection is the
observed statistic, not a leak → REJECT the blocker. (3) Cross-check the plan's
registered statistic definition — a permutation-of-the-selection design is often
exactly what was pre-registered. The leak is real ONLY if the verdict reads the
raw same-cell τ directly with no permutation/CV wrapper around the selection.

Companion: [[feedback_codex_methodology_choice_as_bug]] (Codex flagging a
pre-registered design choice — e.g. a within-family row bootstrap the plan
explicitly scoped to one family — as a bug; same #545 r1, Major #1).
