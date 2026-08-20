---
name: pilot-constant-dict-reverts-plan-raise
description: A production driver copying the pilot's constant dict (caps, budgets, thresholds) silently reverts a plan-registered RAISE — diff every inherited constant dict against the plan's §11 literals (#2388 R1 g3)
metadata:
  type: feedback
---

When a production script inherits a parent/pilot artifact's constant dict
(MAX_TOKENS, budgets, thresholds), diff EVERY entry against the plan's §11
grounded-hyperparameter rows and phase prose — a verbatim copy is exactly how
a plan-REGISTERED raise gets silently reverted.

**Why:** #2388 R1 g3 — plan pinned max_new_tokens 2048 for ALL surfaces
(MMLU-Pro raised from the pilot's 1024 after a measured 1.9% cap-hit at the
2% re-gen trigger edge; HumanEval/MBPP raised direction-safe); the driver
copied the pilot's cap dict verbatim, shipping 1024 for all three. The revert
re-arms the exact re-gen wave the registered raise pre-empted — and the
driver had no executable re-gen path, compounding it. Sibling of
[[registered-gate-quantity-substituted]] (gates) — this is the registered
HYPERPARAMETER form, and the tell is different: not a stricter proxy but a
byte-faithful copy of the parent's value where the plan ordered a change.

**How to apply:** for any commit whose docstring says "same as pilot/parent
with X changed", grep the plan for each constant-dict key's literal value and
check the ones the plan CHANGED are the ones the code changed. Also check the
value persists at HEAD (a later round commit may have fixed it) before
ranking Critical.
