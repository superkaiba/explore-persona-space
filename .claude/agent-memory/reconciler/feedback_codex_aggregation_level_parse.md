---
name: Codex stats critic misparses trailing aggregation parenthetical
description: Codex flags an aggregation-level/noise-floor mismatch by parsing "(averaged over i)" as averaging the DATA before the variance; check operator order, floor pairing, changelog lineage, figure spec, and error direction.
type: feedback
---

Codex statistics critic REVISEs a registered variance-vs-noise-floor kill by reading a trailing
parenthetical — "Between-eval-context variance of G[b, i→·] (averaged over i, off-diagonal)" — as
averaging G over i BEFORE the variance (mean-profile statistic), then arguing the registered
per-cell floor over-subtracts by ~n_contexts (≈16×) so the kill "fires or passes by arithmetic".

**Why:** The hostile parse requires inverted operator order ("variance of [G averaged over i]")
that the plan text does not write. Codex's own Must-Fix was conditional ("If the statistic is...")
— a possibility claim, not a citation of registered text. Origin: task #537 plan-v6 round-4
statistics reconcile (2026-06-09).

**How to apply — four convergent checks before believing the mismatch is registered:**
1. Operator order in the noun phrase: is the variance taken "of" a quantity with the index free,
   with the averaging in a TRAILING parenthetical? Then the per-row reading is the natural parse.
2. Floor pairing: a "per-cell" floor registered in the same sentence is coherent only with a
   variance over per-cell values (E[Var of noisy cell means] = true variance + mean per-cell noise).
3. Changelog lineage: if the delta says "threshold unchanged, estimator-only swap" from a
   cross-seed floor (inherently per-cell), the prior statistic's aggregation level carries over.
4. Figure spec: a "floor vs variance SCATTER" requires per-row pairs — impossible under a
   single-scalar mean-profile statistic.

**Error-direction tiebreak:** even under the misparse, pairing a mean-profile statistic with the
raw per-cell floor OVER-subtracts → corrected variance goes negative → the kill fires loud and
conservative (escalate, don't ship). A false-PASS-by-arithmetic needs the INVERSE mismatch
(per-row statistic with a /n floor), which nobody registers. If the only reachable arithmetic
failure is a conservative false-kill that self-reveals (negative variance), the ambiguity is not
conclusion-changing → APPROVE with the misparse named.
