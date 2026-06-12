---
name: persona-gated manipulation-check surface
description: Manipulation checks for persona-gated implants probe under the SOURCE persona prompt — a surface with no demonstrated-expression prior; a sub-threshold read is ambiguous (not-installed vs installed-but-unexpressed-under-prompt). Labeling branch + canonical default-context companion read keeps it Concern, not REVISE.
metadata:
  type: feedback
---

From #552 follow-up `contrastive-2x2-completion` (plan v4, 2026-06-11). When a behavior
is implanted persona-gated (contrastive negatives), the manipulation check must probe
under the source persona's system prompt — but the canonical demonstrated-expression
citation (e.g. #458 first-plot probes at 21–28% for plain-EM) holds for the NO-PROMPT
surface only. The source-persona prompt itself can modulate expression on generic
probes, so an all-seeds-below-threshold read is ambiguous between "implant did not
take" and "implant took but does not express on these probes under this prompt."

**Why:** the statistics-lens item 6 (gate elicitation-surface validity) nominally fires
here, but there is no canonical surface FOR the gated construct — probing without the
prompt would miss a correctly-contained implant by design. Demanding a new probe
surface breaks parity with the parent gate and adds an unreferenced cell.

**How to apply:** Concern (not Must-Fix) when ALL of: (a) the plan pre-registers a
labeling branch (geometry still reported, cell labeled "implant did not take"), (b) the
default/no-prompt context is ALSO collected (the canonical surface has power for any
leakage), (c) raw completions + train-loss + shift-magnitude collateral are persisted so
the analyzer can separate the two readings. Escalate toward Must-Fix only if a
sub-threshold source read would be narrated as a CONTAINMENT success with no
collateral diagnostics stored. Related: [[matched-corpus-geometry-control alternatives]],
[[Reliability-precondition boundary arithmetic]].
