---
name: early-stop-when-eval-range-collapses
description: When an upstream validity gate drops cells along the same axis as the DV's dynamic range, the downstream sweep on survivors is structurally uninformative — write up the gate result + dynamic-range argument as the finding instead
metadata:
  type: feedback
---

In a 2-step design (validity gate → downstream measurement on survivors), if the dropped cells are precisely the ones carrying the DV's dynamic range, the downstream measurement is structurally uninformative even when it clears the minimum-viable-N threshold. Diagnostic: sort the viable cells by the DV — if the top N−1 sit at the floor and only one has signal, an across-cell rank correlation is a one-point-leverage statistic, not a measurement. Don't run the sweep; the gate result + the dynamic-range collapse IS the finding.

**Why:** task #467 (2026-06-03) — strong-NL elicitation gate passed 6/18 cells (just clearing the 6-cell kill line), but those 6 were 5 floor cells + 1 mid-EM cell, while the 12 dropped included all 6 cells with broad-mis rate > 0.15. The planned cosine sweep was stopped before generating numbers.

**How to apply:** pull the DV per viable cell into a sorted list before trusting any post-gate aggregate. If the viable subset's DV range is below your meaningful-effect threshold and the dropped cells span the full range, write up: (1) the gate result + cell-vs-DV alignment, (2) the sorted-DV collapse argument, (3) any by-elimination interpretation with explicit "by elimination, not direct measurement" framing, (4) confidence MODERATE max on by-elimination answers, naming the regime (different model / elicitation) that could rescue the direct measurement.

Compare [[check-partial-against-threshold]] — different failure mode (viable-but-uninformative vs partial-vs-threshold).
