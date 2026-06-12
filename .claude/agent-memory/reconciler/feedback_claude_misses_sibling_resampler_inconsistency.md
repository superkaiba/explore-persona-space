---
name: claude-misses-sibling-resampler-inconsistency
description: When the round-N "fix" introduces a NEW bootstrap / cluster-resampler / CI estimator alongside existing siblings in the same file, Claude code-reviewer PASSes after verifying the math rationale ("variance adds", "independent draws", etc.) but doesn't grep the file for sibling resamplers to check the new function's resample step matches the file's own canonical implementation. Codex catches `{...for k in rng.integers(...)}` set-comprehension that drops with-replacement duplicates vs the sibling `[all_x[i] for i in idx]` list pattern.
metadata:
  type: feedback
---

When a round-N blocker fix introduces a NEW bootstrap, cluster resampler,
CI estimator, or any other statistics-on-resampled-data function INTO A
FILE THAT ALREADY HAS SIBLING IMPLEMENTATIONS of the same primitive, do
not believe Claude's PASS on the math rationale alone (e.g. "variance
adds because the two draws are independent", "math sound", "matches
plan §X").

**Defense:** before adjudicating, `grep -n 'rng.integers\|np.random\|resample\|bootstrap' <file>` to enumerate every existing cluster/data resampler in the file. Read the new function's resample step and the existing siblings' resample steps SIDE-BY-SIDE. If the new function uses `{... for k in rng.integers(0, n, n)}` (set comprehension) while siblings use `[... for k in rng.integers(0, n, n)]` (list comprehension), that IS the bug:

- list comprehension over `rng.integers(0, n, n)` → with-replacement cluster bootstrap of size n (cluster `k` appears `~Poisson(1)` times; duplicates kept; canonical Cameron-Gelbach-Miller pattern).
- set comprehension over the SAME `rng.integers` call → subsample-without-replacement of expected size `n*(1-1/e)` ≈ 0.63n (duplicates dropped; effective N shrinks; CI width / coverage are off).

This pattern is high-yield because: (a) the new function's docstring usually ratifies the design ("dyadic cluster bootstrap of ρ_X on the within-arm panel") so Claude reads the docstring as evidence of correctness without checking that the impl matches its own siblings; (b) Claude's verification often cites the OUTER structure ("two independent draws → variance adds") which is true but tangential to the per-arm resampler bug; (c) the new function is usually the headline-statistic estimator (here: H3's `_h3_independent_two_sample` for the cross-arm |Δρ| CI that the PASS gate uses), so a wrong CI ships as the headline statistic.

Companion pattern: when the same file enforces a fail-loud contract on a SET (e.g. requested fracs, requested conditions, requested seeds) and the round-N fix narrows the iteration to the PRESENT intersection, the fix can silently skip a fully-missing requested member. `[f for f in present_fracs if f in set(requested_fracs)]` does NOT raise on `requested - present`; the completeness loop that follows only sees the intersection. Defense: check `requested - present == ∅` explicitly BEFORE the completeness loop.

Origin: task #489 round-3 cap, `scripts/i489_phase5_analyze.py` H3 bootstrap (lines 370-373 SET vs sibling `_dyadic_cluster_bootstrap_rho` lines 219-225 LIST + `_paired_diff_bootstrap_rho` lines 285-288 LIST) + Phase 5 frac validation (lines 700-724 silent skip of fully-missing requested frac). Claude PASS; Codex FAIL; reconcile = FAIL (Codex correct on both).

**Why:** Claude verifies the rationale ("math sound", "matches plan") but doesn't grep for sibling implementations of the same primitive in the same file; Codex's "compare to existing function" prior catches the inconsistency.

**How to apply:** Whenever the round-N diff adds a NEW resampler/bootstrap/CI estimator function, before adjudicating:

1. `grep -n 'def _\(.*bootstrap\|.*resample\|.*cluster\)\b' <file>` — enumerate siblings.
2. For each sibling, read the resample step (the few lines around `rng.integers` / `np.random`).
3. Read the new function's resample step.
4. If the new function uses `{...}` (set) where siblings use `[...]` (list) — or any other primitive-collection mismatch — that's the bug. PASS becomes FAIL.

When the file's fail-loud contract is on a SET, also check: does the round-N fix iterate `requested ∩ present` instead of `requested`? If yes, the fully-missing case silently skips. Look for an explicit `requested - present == ∅` check BEFORE the per-member completeness loop.
