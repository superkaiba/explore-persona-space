---
name: contrastive-mode-bundle alternatives
description: Alternatives endemic to "contrastive vs plain SFT training-mode" geometry comparisons under full-sequence CE — no push-pull mechanism exists, the variable is a data-construction bundle; eval-format familiarity; what the held-out-subpanel control does and does not separate; shared-negatives common component in cross-arm U1 reads
metadata:
  type: feedback
---

From the #552 follow-up `contrastive-2x2-completion` plan v4 review (2026-06-11;
lineage #519/#521/#552 layer-14 shift-geometry). Applies to any plan that adds a
"contrastive arm" trained with FULL-SEQUENCE (or full-response) CE to test whether
training MODE drives a geometry/concentration DV.

**1. Under full-sequence CE there is no contrastive push-pull mechanism.** The
contrastive-negatives rule's mechanism (negatives train EOS at the slot under masked
loss) only operates under marker-only/masked loss. With full-sequence CE, negative
rows are just diverse SFT toward near-base text — so "training mode" is really a
BUNDLE: {persona system prompts on rows, +N near-base rows, ~2x optimizer steps at
exposure parity}. Operational claim-scoping ("the #519-style persona-gated data
construction de-concentrates") is the fix, not a REVISE — a decomposition arm
(persona-prompted positives-only; or positives + promptless padding) is a follow-up.
Mirrors [[ratio-lever-inherent-entanglement]] / content-arm off-policy entanglement.

**2. Eval-format familiarity is a concrete single-component mechanism for
de-concentration.** If plain arms trained on BARE rows but the geometry probe reads
shifts under persona SYSTEM prompts, the plain arms' shared direction can contain a
uniform "fine-tuned-on-bare-rows under system-prompted context" format component;
arms trained WITH system prompts remove it. Crucially, the held-out-subpanel control
does NOT separate this (format familiarity generalizes to untouched personas) — the
subpanel only separates touched-row mechanics. After subpanel passes, the residual
attribution fork is {contrast, prompt-format presence, data heterogeneity/steps}.

**3. Magnitude-mediation twin fires in BOTH branches of a mode-x-content 2x2.**
(a) both-de-concentrate: negatives dilute the implant → smaller ‖M‖_F → mechanical
de-concentration per the documented monotone (‖M‖_F, concentration) collinearity;
(b) content-split: benign positives are nearer base policy than EM positives → the
benign arm's total update shrinks first → "content-linked" is predicted by magnitude
alone. Read every zone call against per-cell ‖M‖_F; the alternative dies if the
de-concentrated arm's ‖M‖_F lands within ~2x the concentrated arm's. See
[[matched-corpus-geometry-control alternatives]].

**4. Shared-negative rows inflate cross-arm U1 identity between the two contrastive
arms** beyond the shared-question component the plain pair has — name the asymmetry
before benchmarking contrastive-EM x contrastive-benign sharing against the plain
pair's value.

**5. Subpanel SVD statistics:** zone thresholds registered on the full panel (n rows)
do not transfer to the (n-k)-row subpanel; read the subpanel RELATIVE to all arms'
subpanel companions with re-run nulls at the reduced row count (the #552 v4 plan did
this correctly).

**Why APPROVE:** all of these were analyzer-weighable in v4 because the plan
persisted per-question tensors + singular values + ‖M‖_F + 3 trajectory variants +
train-loss curves + per-context EM rates, and pre-registered the held-out-subpanel
companion. The same alternatives escalate toward Must-Fix when those diagnostics are
not stored or the headline claims a push-pull mechanism rather than the construction.
