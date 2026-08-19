---
name: infra-wf-fix-lint-gate-compose
description: Compose recipe for kind:infra wf-fix diffs targeting workflow_lint.py — N/A-by-type gates, hollow-gate = check-registration trace, LIVE_WORKFLOW_HELPERS arming, Step-2-floor attestation
metadata:
  type: feedback
---

Compose recipe for a `kind: infra` wf-fix round whose diff ADDS a check to
`scripts/workflow_lint.py` (recurring shape; first used #2192 r1):

1. **N/A-by-type block up front.** Steps 0.55 / 0.6 / 0.65 / 0.67-exposure /
   0.68-parent are `type:experiment`-only — state the N/A explicitly in a
   compose-time-facts block so Codex never raises `marker-shape` /
   `smoke-run-missing` on their account; the any-diff-type sub-checks
   (0.67 work-conserving, 0.68 hollow-gate + hub-scoping, 0.69–0.72,
   fit-loop line) stay binding.
2. **Hollow-verification-gate sub-check MAPS to lint-gate diffs:** instruct
   Codex to trace that every round-added check function is DISPATCHED —
   registered in the no-flags default run and/or its `--check-*` flag path —
   quoting the registration/call site. A check defined but never wired is a
   hollow gate (Major `hollow-verification-gate`).
3. **LIVE_WORKFLOW_HELPERS arming:** `scripts/workflow_lint.py` IS on the
   roster (tests/test_ruff_policy.py) — state it as a compose-time fact so
   the Step 0.5 `(c)` ruff-policy-pin field check binds, and have Codex
   verify the roster line itself in the worktree.
4. **wf-fix Step-2-floor attestation** ([[wf-fix-step2-floor-attestation]]):
   probe main for `epm:plan-verify` at compose time and attest
   PRESENT/absent in the prompt — Codex cannot read main-side events.
5. **`epm:results` + ts ≥ 2026-07-15 ⇒ Gate-scope threshold satisfied** line;
   pin-sweep verification adapted to `git -C <wt> grep -n '<literal>' -- tests/`
   (no `select_step9c_tests.py` — no uv env).

**Why:** these five all fired together on #2192 r1; missing any one produces
either a false Codex `marker-shape`/`step2-floor-skipped` FAIL or a narrowed
check (#606 twin-omission class).
**How to apply:** any `kind: infra` round whose diff touches
`scripts/workflow_lint.py` or another guard/lint workflow helper.
