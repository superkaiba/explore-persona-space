---
name: Completion-source-swap diagnostics leave own-completion mediation open
description: Base-own re-reads test FT-vs-intrinsic but can't separate direct-at-slot context effects from base-OWN-completion mediation; fixed-completion force-read is the free disambiguator (#563)
type: feedback
---

When a plan swaps WHOSE completions a slot statistic is read on (#563: base prior re-read on base-own instead of FT completions) to test "intrinsic vs completion-carried", a REPRODUCES outcome still admits a third mechanism: the persona prompt changes the base model's OWN completion content/length, and the completion change (not the context directly at the slot) moves the prior. Both arms have cell-varying completions, so "pure context at a fixed slot" is never measured anywhere in the chain.

**Why not REVISE in #563:** (a) the registered Goal IS the FT-vs-base dichotomy; (b) per-row lengths + similarity + language covariates and paired per-question data persist, so the analyzer can regress d_logp on d_length and weigh mediation descriptively; (c) the practical implication (audits inherit a moving base reference) was already established by the parent.

**How to apply:** for any completion-source-swap plan: (1) check the claim language stays at the registered dichotomy, not "direct context effect at the slot"; (2) check content/length covariates persist per row paired by question (recoverable → Concern); (3) name the zero-gen disambiguator: force-read ONE fixed completion set under all K system prompts (slot forwards only, no generation) to isolate the pure context-at-slot effect. Absence is fatal ONLY if the headline claims slot-level mechanism rather than source-robustness.
