---
name: Anomaly probes that are near-twins of a trained contrastive negative
description: Flat-cell "anomaly" attribution is confounded when the probe personas are near-twins of a persona in the adapter's contrastive-negative set; arm contrasts stay clean, attribution needs scoping (#612)
type: feedback
---

Pattern from #612 v1 (sycophancy rig v2, alternatives lens): the H3 anomaly
probes (`virtual_assistant`, `digital_helper`, near-twins of `assistant` at
cos ~0.98) are evaluated under adapters whose contrastive-negative sets
INCLUDE `assistant` (software_engineer and comedian — verified in
`neg_membership_411.json`). Contrastive training actively pushes agreement
DOWN under the `assistant` context, and that suppression plausibly
generalizes to assistant-like near-twins — a third explanation for twin
flatness that is neither "wording priors gate leakage" nor "canned-template
artifact".

**Why it's not REVISE:** the negative sets are constant across arms by
parity, so the registered survive/dissolve ARM CONTRAST is unconfounded;
only the deeper attribution of WHY the twins are flat is confounded, and
that attribution is interpretation-level, weighable from shipped
diagnostics.

**How to apply:** when a plan designates flat/anomaly probe cells, check the
probe personas' similarity to every member of that adapter's
contrastive-negative panel (read the realized neg-membership file, not the
prose — `neg_member` flags cover literal negatives only, NOT near-twins of
negatives). If a probe is a near-twin of a trained negative, name
suppression-generalization as the alternative and point the analyzer at the
partial discriminators: the same probes under adapters WITHOUT that negative
(lower cosine, weaker read), probes that DO leak under the same adapter
(e.g. daycare_teacher under software_engineer), and the probes' base priors.

Companion fork from the same review (multi-turn-prefix arm, H2): when a
prefix/multi-turn-trained arm is evaluated single-turn, "prefix anchors the
behavior to the persona" vs "train/eval context mismatch (generalization
gap)" are separated by the SELF-implant secondary read — both-drop ⇒
mismatch; bystander-only-drop ⇒ anchoring. Check the plan registers the
self-implant contrast, then pre-name the fork for the analyzer.
