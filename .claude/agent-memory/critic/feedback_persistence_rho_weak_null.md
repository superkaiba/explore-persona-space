---
name: Persistence-rho reads have a weak null
description: trained-vs-base FE ρ ≥ 0.9 on shared-R matched-slot reads is near-tautological (scale-invariant, common-mode text, shared R noise); demand variance-ratio + split-question complements, not REVISE (#560, #553)
type: feedback
---

When a hypothesis "the base level persists through training" is operationalized as ρ between trained-side and base-side persona/context fixed effects read at MATCHED slots on the SAME responses (#553: ρ=0.969; #560: H2 ρ ≥ 0.9), treat the ρ criterion as a weak null. Three mechanisms push ρ high regardless of persistence: (1) ρ is scale-invariant — a heavily compressed-but-order-preserving rewrite reads as "persistence"; (2) both sides score IDENTICAL text at the IDENTICAL position, so text-driven variance is common-mode (two unrelated finetunes would also correlate); (3) shared R-sampling noise inflates ρ above the independent-draw value. Non-persistence under low ρ requires training to scramble the persona ORDERING — which almost no LoRA does.

**How to apply (Alternatives/Statistics lens):** APPROVE + concern, with three free complements the per-q four-float arrays support: (a) variance ratio var(d_FE)/var(base_FE) — the change-magnitude read ρ hides; (b) split-question check (trained FE on half the questions vs base FE on the other half — de-shares the R noise); (c) the partial check (clamp-routing vs margin_base — the change-DV mechanical-subtraction memory applies to that read). Escalate to REVISE only if per-q tensors are NOT persisted.
