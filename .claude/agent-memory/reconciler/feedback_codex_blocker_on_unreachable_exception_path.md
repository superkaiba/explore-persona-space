---
name: codex-blocker-on-unreachable-exception-path
description: Codex builds a Critical on a code-real exception/retry coupling without checking reachability — do the arithmetic (trace/dof bounds, grid-superset structure) before upholding
metadata:
  type: feedback
---

Codex (code-review twin) can build a Critical on a mechanism that is REAL in
the code — an exception-triggered batched retry that swaps state for the whole
batch — while never checking whether the triggering exception can FIRE at the
artifact's shapes. Verify reachability arithmetically before upholding.

**Why:** #2356 r4: Codex FAILed R4-2 claiming the extras' single-layer ridge
refit could diverge from the fold-selected model because `_dual_ridge` retries
the ENTIRE batched call on `_WIDE_LAMBDAS` when ANY layer exhausts the default
GCV grid (`fit_h.py` raises on `all_capped.any()`). The coupling is real code
(Claude's "per-layer GCV independence" premise was imprecise), but the raise
is unreachable: the helper standardizes features internally, so
trace(G) ≤ n·d and dof(λ) ≤ n·d/λ; at the default grid's λ_max = 1e4 with
d ≤ 3584 (Qwen hidden), dof ≤ 0.358·n < 0.9·n (the dof cap) — the largest
lambda can never be masked, so the grid can never exhaust (needs d > cap·λ_max
= 9000). Bonus check: the wide grid was a bitwise SUPERSET (same 10^0.5 step),
so even a fired retry diverges only if the selected slice's argmin lands in
the extension points. Verdict: PASS; the residual (refit discards the returned
lambda instead of asserting it equals the persisted fold lambda) persisted as
CONCERN, not blocker.

**How to apply:** when a FAIL rests on an exception/retry/fallback path
("when X raises, the whole batch switches to Y"), (1) derive whether X can
fire at the script's realized shapes — trace/dof bounds for GCV caps,
dimension vs threshold arithmetic, grid-edge conditions; (2) check whether the
fallback state is a superset/identical on the shared region (divergence often
needs a SECOND condition); (3) if unreachable-today-but-latent, uphold as a
CONCERN (assert/pin the invariant) rather than a blocker. Sibling patterns:
[[claude-misses-invariant-comment-smell]] (a code comment asserting the
exactness was ALSO unanchored to the real invariant — the persisted fix pins
it), [[codex-fail-loud-diagnostic-blocker]].
