---
name: Descriptive geometry-map plan checks
description: Methodology checks for descriptive representation-geometry / DR-visualization plans (context battery maps, #594 line) — format-delivery confound, dominant-family aggregates, residualization direction
type: feedback
---

For descriptive "do contexts/instances cluster by family" geometry plans (first instance: #594, extraction-only context-battery map; more will follow from the context-generalization testbed grid):

1. **Family ↔ template-delivery-format confound is inherent, not a REVISE.** Families differ in HOW they enter the chat template (system prompt vs multi-turn prefix_messages vs none), so "ICL/WildChat separate from instruction families" is partly a format read. The cleaner contrasts are structure-matched (system-prompt families vs each other). Scope at claim level for the analyzer; the cosine matrices + embeddings make it weighable.
2. **Dominant-family aggregates:** if one family holds ~30% of instances (persona n=14/48 in #594), aggregate silhouette/purity can beat the null from that cluster alone. Ask for per-family purity breakdown — fully recoverable post-hoc if mean tensors + analysis script ship, so a Concern, not REVISE.
3. **Length residualization over-corrects when length is family-correlated by construction** (ICL k-axis, WildChat bins) — a residualized null (H3 kill) is conservative, consistent with the fixed-pool-floor memory. Right shape = report raw + residualized + length-only baseline side by side.
4. **Probe-mean construction:** mean over a FIXED probe pool shared across instances makes probe content common-mode → global-mean centering removes it; centering bank = the analyzed set itself is standard but note mild dependence. Probe-pool genre (Betley EM flavor) is a scope caveat (carried as assumption), bounded by split-half stability.
5. **What made #594 v1 APPROVE-able round 1:** permutation null with max-over-layers FWER + shared draws across layers; PCA alongside every nonlinear embedding; per-forward position-decode assert with a >10% kill criterion; per-probe tensors persisted (no phantom bootstrap inputs); CPU phases sequenced off-pod; all N/A rules explicitly stated. Use as the reference bar for siblings.

**Why:** #594 review (2026-06-11) — plan survived every Methodology item; the only real residue was claim-scoping, all analyzer-recoverable.
**How to apply:** any plan whose headline DV is cluster-quality-vs-permutation-null over a heterogeneous instance battery.
