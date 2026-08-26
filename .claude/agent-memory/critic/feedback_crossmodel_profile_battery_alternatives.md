---
name: crossmodel-profile-battery-alternatives
description: "#2587 (9B vs 7B minimal-pair profile): the boring drivers of a shared cross-model axis ordering (edit-dose, shared text-level behavior, noisier target, layer-selection asymmetry) are all analyzer-weighable IFF the diagnostics persist on BOTH model sides — APPROVE, no fatal alternative"
metadata:
  type: feedback
---

Alternatives-lens disposition for cross-model per-axis profile replications
(the #2564→#2587 shape; expect siblings, e.g. a Qwen3.6 battery).

Rule: the four recurring boring explanations are RECOVERABLE (Concern, not
REVISE) when each has a persisted diagnostic on BOTH model sides:

1. **Shared edit-dose ordering** (bigger context edit ⇒ bigger answer Δ in any
   model) — weighable iff per-pair `changed_tokens` is persisted under EACH
   tokenizer + per-axis paraphrase null + ceiling adjustment; partialling is a
   pure re-reduction. An answer-language pilot axis is a built-in dissociation
   anchor (1-word context edit, near-total answer change).
2. **Shared TEXT-level behavior, not shared representation** — weighable iff
   the SAME third-party embedder's per-axis text-space Δ exists for both
   models (banked for parent + computed for new model). Caveat: embedder
   insensitivity to an axis deflates text-space ρ on both sides, so prefer
   axis-level rep-visible-vs-text-invisible dissociations over a scalar
   ρ_rep vs ρ_text comparison.
3. **Noisier target, not weaker structure** (map-R² deficit) — weighable iff
   two-draw reliability ceilings exist on both sides (ceiling-adjust R²).
   d-mismatch (4096 vs 3584 at matched n) is a small mechanical n/d penalty;
   fresh-side fp32 capture biases AGAINST the predicted deficit (conservative
   if the deficit confirms; a live alternative only if it falsifies).
4. **Layer-selection asymmetry** (new model argmax-over-32-layers vs parent
   fixed layer) inflates the NEW side, so it only becomes the boring
   explanation on a marginal new ≥ parent falsification read — consult the
   full displayed layer curve + the prior matched-n dense sweep first.

**Why:** on #2587 v2 every listed alternative had a both-sides persisted
diagnostic, so APPROVE with Concerns was the right call; the design pattern
(symmetric fire-gating, shared-carrier-resample bootstrap, held-out-frozen
L*) also pre-defuses the selection/noise-structure traps.

**How to apply:** on the next cross-model battery, check BOTH-SIDES
persistence per alternative first; REVISE only if a side is missing (e.g. no
parent embeddings, no parent ceiling draws). Thinking-off / regime asymmetry
on an n=2 model comparison is a scope caveat, not a missing arm: every model
pair differs in a bundle (size, arch, tokenizer, reasoning training), and a
thinking-ON arm changes the measurement surface (answer-span definition,
cap arithmetic) so it would not isolate the regime anyway. Related:
[[alternatives-lens-round2]], [[panel-family-clustering]].
