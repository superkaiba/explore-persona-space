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
   verify the roster line itself in the worktree. Roster membership is
   PER-FILE — grep `tests/test_ruff_policy.py` fresh each compose, never
   assume from this memory: #2195 r1 (`scripts/verify_report.py`) was NOT
   on the roster, flipping the pin field to a legitimate SKIP (state THAT
   as the compose-time fact instead, so Codex neither demands the pin nor
   disputes the implementer's SKIPPED line).
4. **wf-fix Step-2-floor attestation** ([[wf-fix-step2-floor-attestation]]):
   probe main for `epm:plan-verify` at compose time and attest
   PRESENT/absent in the prompt — Codex cannot read main-side events.
   NON-wf-fix infra tasks (no `workflow-fix:`/`daily-fix:` title prefix, no
   `wf-fix` tag) get the EXEMPT form: attest "floor check exempt" (+ any
   plan-verify verdict found anyway) so Codex never false-fires
   `step2-floor-skipped` — the rubric's floor check binds wf-fix only
   (#2194 r1: exempt AND 3 plan-verify markers present, attested both).
5. **`epm:results` + ts ≥ 2026-07-15 ⇒ Gate-scope threshold satisfied** line;
   pin-sweep verification adapted to `git -C <wt> grep -n '<literal>' -- tests/`
   (no `select_step9c_tests.py` — no uv env).

**Why:** these five all fired together on #2192 r1; missing any one produces
either a false Codex `marker-shape`/`step2-floor-skipped` FAIL or a narrowed
check (#606 twin-omission class).
**How to apply:** any `kind: infra` round whose diff touches
`scripts/workflow_lint.py` or another guard/lint/verifier workflow helper
(`verify_task_body.py`, `verify_plan.py` are the same class — #2291 r1).

**Two #2291 r1 (2026-08-22) sharpenings:**

6. **wf-fix detection is TAG-first, not title-first.** #2291's title had no
   `workflow-fix:`/`daily-fix:` prefix, but `body.md` `tags:` carried
   `workflow-fix` (and the Provenance line named the workflow-fix-candidate
   origin) — a title-only probe would have mis-attested "floor exempt" on a
   task whose floor BOUND (an `epm:plan-verify` PASS was present to attest).
   Probe `grep -A3 '^tags:' body.md` + the Provenance line every compose.
7. **Brief-supplied plan-vs-measured numeric discrepancies compose as
   TEST-the-hypothesis duties**, never as attested facts: state the plan's
   count, the measured count, the orchestrator's hypothesis (e.g. label
   transposition in a plan amendment = PLAN defect not code defect), and
   instruct Codex to decide which count belongs to which label FROM THE CODE
   and say whether any acceptance criterion depends on it. Also state stakes
   BOTH directions for verifier-gate diffs: a false PASS ships a broken
   fleet gate, and an over-strict new check arm is itself a fleet-blocking
   false-FAIL class — so over-strictness findings weigh equal to bugs.
