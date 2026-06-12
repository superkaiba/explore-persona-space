---
name: Singleton-class asymmetric falsification in panel factorizations
description: When a probe panel factorizes a property with n=1 cell in the discriminating class, the dip/positive outcome is an existential refutation (n=1 suffices) but the no-dip/falsification outcome is a universal claim from a singleton, confounded with manipulation strength / departure magnitude
type: feedback
---

In panel designs that factorize "does X require property P?" by adding one
non-P probe cell (e.g. #562: one bare non-persona instruction vs persona
prompts), the two outcomes have asymmetric evidential weight:

- **Positive (cell shows the effect):** existential refutation of "P is
  required" — n=1 in the class is logically sufficient; only the "ANY
  non-P context" universal phrasing is an overclaim (analyzer scopes it).
- **Negative (cell shows no effect):** the registered falsification reading
  ("P is specifically required") is a UNIVERSAL claim from a singleton, and
  is confounded with manipulation weakness — the non-P probe may simply be
  too small a departure (shortest prompt, closest in register to the
  trained context) rather than non-P. Dose-response-in-departure-magnitude
  is a refinement of the context-general account, not evidence for the
  P-specific account.

**Why:** #562 v1 (2026-06-10): bare instruction (~9 tokens, near the trained
"You are a helpful assistant.") vs nurse/comedian (~14-16 tokens). The plan
de-singletoned the never-trained-persona class (added comedian because
police was a singleton) yet left the non-persona class n=1 — the asymmetry
hid in plain sight. Not a REVISE there because the plan pre-registered a
manipulation check gating the no-dip read and a graded middle, and the
primary bet was the existential direction.

**How to apply:** Not a REVISE when (a) the positive direction is the
primary bet, (b) a manipulation check on the singleton cell is
pre-registered, and (c) the graded category exists. Prescribe for the
analyzer: the falsification headline must be scoped to "this one non-P
probe" at capped confidence, and prompt-length / register-distance to the
trained context must be reported next to the verdict (if effect magnitude
orders by token-level departure across cells, the P-specific reading is
unsupported). Sibling: feedback_followup_baseline_misclassification (#480),
feedback_partial_group_success_vs_all_group_guard (#546).
