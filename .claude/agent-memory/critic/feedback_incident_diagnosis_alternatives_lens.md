---
name: incident-diagnosis-alternatives-lens
description: Alternatives lens on kind:infra incident-diagnosis plans — shape-based detection defuses origin-story alternatives; audit every site where the causal story does design work (#2360)
metadata:
  type: feedback
---

Rule: for a `kind: infra` incident-diagnosis plan, an origin-story alternative
("the hardlink fallback / ESTALE / killed install caused it") is FATAL only if
the fix depends on the story. A detection gate keyed on the failure SHAPE
(metadata unresolvable, import fails) catches every candidate origin equally —
so attack instead the SITES where the causal claim does design work.

**Why:** #2360 (preflight misses half-installed pod venv): the plan explicitly
disclaimed corruption-prevention for its bootstrap change (citing #2278: copy
mode does NOT prevent ESTALE) and made detection cause-agnostic, so all five
origin alternatives were non-fatal. The ONE spot the causal story did work —
"corruption class is MooseFS-copy-specific" justifying tier-2 lane-gating —
also had an independent cost justification (measured 22-51 s on the hot VM
path), so even a wrong story left the design standing. APPROVEd round 1.

**How to apply:** (1) grep the plan for every use of the origin claim; each
site needs either the shape-based-detection property or a cause-independent
justification. (2) Warning-suppression acceptance criteria: credit "suppressed
= better signal" only if the warning fired UNCONDITIONALLY in the environment
(zero conditional information — #2360's hardlink warning fired on every pod
install); else suppression loses a symptom channel. (3) Standing recoverable
alternative to "gate PASS implies working system": sanctioned post-gate
mutations (off-lock pins, repair recipes applied after preflight) — name it as
a Concern, not a REVISE, when the acceptance contract is scoped to gate time.
Related: [[infra-plan-review-checklist]].
