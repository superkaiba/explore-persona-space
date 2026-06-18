---
name: Claude classes sign-blind |ρ| decision rules as analyzer-recoverable
description: Statistics-lens disagreement pattern — Claude APPROVEs a plan whose hypothesis PASS rule uses |ρ| while its Confirmed-branch narration asserts direction; Codex correctly flags it as a Must-Fix plan defect.
type: feedback
---

When a plan's hypothesis decision rule is stated in absolute-value form (`|ρ| ≥ τ`, `Δ|ρ|` CI) but the Confirmed-branch narration asserts a DIRECTIONAL conclusion ("understated", "parent headline unchanged"), Claude statistics-critic tends to APPROVE with a "report signed ρ alongside" analyzer concern; Codex correctly REVISEs.

**Why:** Plan decision rules are binding contracts for downstream automation (analyzer narrates against the plan's hypothesis branches; clean-result-critic Lens 13 audits against plan text). Signed values appearing in *reporting* does not make the *decision surface* signed — a sign-flipped result mechanically satisfies the |ρ| clauses and triggers a mis-narrated headline unless the analyzer overrides plan text ("analyzer heroics"). The fix is a one-line direction pin (e.g. `ρ ≤ −τ` + signed Δ bootstrap) at zero design cost; false-PASS propagates.

**How to apply:** Check ALL decision-rule surfaces (plan-summary PASS line, §1 hypothesis, §6 headline statistics) for |·| forms; check whether any "what would change my mind" / Falsified branch contemplates the sign-flip outcome. If none does and the Confirmed branch asserts direction, the sign-blind rule is Real-blocking → REVISE with the direction-pin as the binding fix. Corroborating smell: the codebase line already documents a "polarity quirk" in a sibling combined-predictor. Origin: task #540 round-1 (plan §0 line 16 / §1 lines 31–33 / §6 line 198; v1 ρ_ord = −0.40970).
