---
name: Claude critic-feedback overcorrection breaks plan-intended pass case
description: A round-N defensive guardrail (added for a round-(N-1) critic concern) over-rejects the plan-envisioned PASS scenario, and the implementer codifies the over-rejection in a unit test, making green tests misleading. Re-read the plan §s before believing the test.
type: feedback
---

**Rule:** guardrails written against round-(N-1) critic PROSE risk over-correction; the plan's intent is the source of truth. (1) Re-read the plan §s the bug touches BEFORE believing the test — a regression test codifying a NaN/refusal as "expected" in the plan-envisioned pass case is a regression-LOCK on the bug. Smell: test name asserts X, assertions assert NOT-X ("interpolation_not_extrapolation" asserting `is_extrap=True`). (2) Trace the parallel pipeline — if two reads must AGREE within a threshold and one is forced to NaN by a tighter rule while the other extrapolates freely, the determinacy gate fails in the planned pass case regardless of outcome. (3) Verify against the plan's own named examples (its authorial intent). (4) The fix is small: loosen the guardrail / shift target / near-bracket tolerance — not a re-plan.

**Origin:** #514 r3 — strict-bracket `_linear_interp_at` returned NaN for the plan-named "lower flank" anchor at target=8.0; the codifying test made it look intended. FAIL (Codex right).

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Claude overcorrects critic concern into blocker](feedback_claude_overcorrects_critic_concern_to_blocker.md) — round-N guardrail over-rejects the plan-envisioned PASS case and the test codifies it; re-read the plan §s before believing the test. #514 r3.
