---
name: Completion-source-swap diagnostics leave own-completion mediation open
description: A base-own-completions re-read of a context effect tests "FT-content-carried vs base-intrinsic" but cannot separate direct-at-slot context effects from base-OWN-completion mediation; fixed-completion force-read is the free disambiguator
type: feedback
---

When a plan swaps WHOSE completions a slot statistic is read on (e.g. #563: base prior re-read on base-own instead of FT completions) to test whether a context effect is "intrinsic" vs "completion-carried", the design answers exactly that registered dichotomy — but a REPRODUCES outcome still admits a third mechanism: the persona prompt changes the base model's OWN completion content/length, and the completion change (not the context directly at the slot) moves the prior. Both arms of the swap have cell-varying completions, so "pure context at a fixed slot" is never measured anywhere in the chain.

**Why:** in #563 (2026-06-10) the predicted positive ("rise is intrinsic to base context processing") is compatible with base-own content mediation (key-remarks, length drift, French text). Not a REVISE there because (a) the registered Goal IS the FT-vs-base dichotomy, (b) per-row lengths + similarity + language covariates and paired per-question data are persisted, so the analyzer can regress d_logp on d_length and weigh mediation descriptively, and (c) the practical implication (audits inherit a moving base reference) was already established by the parent on FT completions.

**How to apply:** for any completion-source-swap plan, (1) check the claim language stays at the registered dichotomy, not "direct context effect at the slot"; (2) check content/length covariates are persisted per row paired by question (recoverable → Concern); (3) name the zero-gen disambiguator for the analyzer/follow-up: force-read ONE fixed completion set (e.g. the assistant cell's own completions) under all K system prompts — slot forwards only, no generation — to isolate the pure context-at-slot effect. Absence of that control is fatal ONLY if the headline claims slot-level mechanism rather than source-robustness.
