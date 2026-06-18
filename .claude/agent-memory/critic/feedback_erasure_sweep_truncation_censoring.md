---
name: Erasure-sweep truncation censoring at intermediate pressure
description: In erasure/survival dose-response sweeps, pre-erasure state has high truncation (0.95) and full-erasure has 0.0 — intermediate arms can censor emission toward false falsification at the critical cell
type: feedback
---

In erasure-pressure sweeps over marker installs (#557 over #543's chain): the pre-SFT installed state read 95% truncation at max_new_tokens=2048 (emission still 1.0 because the install emits mid-response), while the fully-erased anchor read 0% truncation (medical SFT shortens responses). An INTERMEDIATE-pressure arm can land between — long, truncated responses with a weakened end-of-response rule — so a near-zero emission read at the hypothesis's critical low-lr cell may be truncation-censored (#260 silent-zero class), biased toward the falsification verdict exactly where it matters.

**Why:** Truncation rate is not constant across the manipulated variable; it co-moves with erasure strength itself. Surfaced reviewing #557 v1 (2026-06-10); plan logged truncation per cell, so it was analyzer-recoverable, not a Must-Fix.

**How to apply:** For any survival/erasure sweep with an emission DV, check the parent's per-cell `truncation_rate` at BOTH endpoints (pre and fully-erased). If they differ a lot, require truncation_rate + ends_with_marker_rate + per-row records persisted per cell, and instruct the analyzer to qualify any near-zero emission that coincides with high truncation. APPROVE if logged; REVISE only if truncation isn't captured at all.
