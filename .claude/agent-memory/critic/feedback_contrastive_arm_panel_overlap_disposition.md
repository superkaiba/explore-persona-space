---
name: contrastive-arm-probe-panel-overlap disposition
description: Contrastive arms in activation-geometry designs cannot be panel-disjoint (default-assistant negative is mandatory and in-panel); right disposition = mirrored panel relation + pre-registered held-out-subpanel read on ALL arms incl. references, not a disjointness REVISE
metadata:
  type: feedback
---

From #552 follow-up `contrastive-2x2-completion` plan v4 (2026-06-11; the
training-mode × content 2×2 completing the #521/#552/#519 geometry family).

When a contrastive behavior-implant arm is evaluated on a persona probe panel,
the negative panel WILL overlap the probe panel: the contrastive-negatives rule
makes the default-assistant negative mandatory, and `assistant` is a probe
persona. Demanding full disjointness is incoherent. The sound disposition
(v4's): (a) keep source ∩ negatives = ∅ as the only HARD invariant; (b) mirror
the reference contrastive arm's panel relation exactly so its zone thresholds
transfer (same 4 negatives, source in-panel); (c) pre-register a held-out
subpanel concentration read (untouched probe rows only) on ALL arms INCLUDING
the persisted references — the reference re-read calibrates what "dispersed"
looks like under the control; (d) any de-concentration call ships with the
subpanel companion number and the headline names whether dispersion survives
removing gradient-touched rows.

Two adjacent dispositions from the same review:
- **"Training mode" is a bundle** (system prompts on rows + negative rows +
  step-count rescale for exposure parity). All three are mechanical
  consequences of the manipulated construction; per the ratio-lever precedent
  ([[ratio-lever-sweeps-inherent-entanglement-disposition]]) this is
  claim-scoping (headline at construction level), not REVISE. Exposure parity
  on positives (doubling steps when the mix doubles) is the right pick of an
  unavoidable fork — match per-row exposure, report ‖M‖_F as the
  total-drift diagnostic.
- **Negative-row gradient inertness under full-sequence CE** (negatives are
  near-on-policy base text → near-zero gradient): not fatal when (i) negatives
  are identical across content arms (can't explain an EM-vs-benign asymmetry),
  and (ii) the both-concentrated branch pre-registers a narrowed reading
  attributing dispersion to the reference arm's loss type "not contrastive
  negatives per se". Cheap upgrade to suggest: per-row-type (pos vs neg) train
  loss logging so the inertness alternative gets a number.

**Why:** the v4 plan converted what looked like two REVISE-shaped flaws
(panel overlap, bundled variable) into measured diagnostics + pre-registered
scoped readings; bouncing it would have re-litigated impossibilities.
**How to apply:** in any contrastive-vs-plain geometry comparison, check for
(a)-(d) above before flagging panel overlap; check branch-by-branch whether
negative-inertness flips any pre-registered reading before flagging it.
