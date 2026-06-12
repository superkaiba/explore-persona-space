---
name: Persistence-rho reads have a weak null
description: trained-vs-base FE correlation (rho >= 0.9 "persistence") on shared-R matched-slot reads is near-tautologically high; demand variance-ratio + split-question diagnostics, not REVISE
type: feedback
---

Rule: when a plan's hypothesis is "the base level persists through training,"
operationalized as Spearman/Pearson rho between trained-side and base-side
persona (or context) fixed effects read at MATCHED slots on the SAME responses
(e.g. #553's rho=0.969, #560's H2 rho >= 0.9), treat the rho criterion as a
weak null and require the analyzer to report the complements — do NOT REVISE
for it alone.

**Why:** three mechanisms push rho high regardless of the persistence
mechanism: (1) rho is scale-invariant, so even a heavily compressed-but-
order-preserving rewrite of the level reads as "persistence"; (2) both sides
score the IDENTICAL text at the IDENTICAL position, so text-driven variance
(what the answer ends with) is common-mode — two unrelated finetunes would
also correlate highly; (3) shared R-sampling noise (same finite question
draw on both sides) inflates rho above the independent-draw value (same
family as the matrix-testbed shared-base-panel-noise pattern). Non-
persistence under low rho requires training to scramble the persona ORDERING
of the level, which almost no LoRA does.

**How to apply (Alternatives/Statistics lens):** APPROVE + concern, with three
named free-analysis complements the per-q four-float arrays already support:
(a) variance ratio var(d_FE)/var(base_FE) — the change-magnitude read rho
hides; (b) split-question check — trained FE on half the questions vs base FE
on the other half, de-sharing the R noise; (c) the partial check the #553
followup used (clamp-routing vs margin_base partialling base level — the
d = post − base mechanical-subtraction memory applies to that read too).
Only escalate to REVISE if the per-q tensors are NOT persisted (then the
complements are unrunnable post-hoc).
