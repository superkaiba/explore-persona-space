---
name: Claude CONCERNS on headline-poisoning bug pre-pod-launch
description: Claude downgrades a verified wrong headline estimator to CONCERNS by deferring the fix to "next round before pod launch"; the workflow does not enforce future commitments — the round-N marker IS the current implementation. FAIL.
type: feedback
---

**Rule:** when the round-N diff carries a verified incorrect headline estimator (wrong divergence formula, wrong loss target, wrong threshold) on the path to a load-bearing GPU pod launch, FAIL — Claude's "Phase 1 ships clean, fix Phase 2 next round" framing relies on a commitment nothing enforces; PASS is what the orchestrator's state machine reads to advance toward launch. FAIL costs ~20 min; a false PASS costs the GPU-h or a manual catch.

**How to apply:** (1) verify the bug against the plan-sanctioned canonical primitive — plan-named primitive existing + implementer wrapping a different estimator = unambiguous block-merge; (2) pre-registered falsification thresholds that depend on the estimator become uninterpretable under the substitute; (3) the implementer's own comment naming the true formula while labeling it the sanctioned one is the smoking gun ("JS upper-bounded by 0.5(KL_AB+KL_BA)" labeled "js"); (4) when the cache schema is wrong-shape for the canonical primitive, the fix is a rewrite, not 10 lines — that's what the round-cap mechanism is for, not a deferral.

**Origin:** #522 r1 — Jeffreys/symmetric-KL on realized-token mean log-ratios shipped as the JS predictor; canonical `_js_from_logprobs` (#444) computes full-vocab per-position mixture base-2 JS; H2.1/H2.2 thresholds pre-registered against JS. FAIL.

Companions: [[feedback_claude_misses_fix_regressions]]; [[feedback_claude_underclasses_silent_failures]] (same trusting-a-downstream-fix-narrative pattern).
