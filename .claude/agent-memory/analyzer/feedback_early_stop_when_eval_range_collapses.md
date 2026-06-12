---
name: early-stop-when-eval-range-collapses
description: When an upstream elicitation/validity gate drops cells along the same axis as the DV signal (the cells that carry the EM signal also fail to load), the downstream sweep becomes structurally unanswerable; write up the gate result as the finding rather than running a leverage-statistic sweep on the surviving cells.
metadata:
  type: feedback
---

When an experiment has a 2-step structure (validity gate -> downstream
measurement on the survivors) AND the dropped cells are precisely the
cells that carry the DV's dynamic range, the downstream measurement
becomes structurally uninformative even if it technically passes the
"minimum viable N" threshold. Don't run the sweep; write up the gate
result + the dynamic-range argument as the finding.

**Diagnostic question:** sort the viable cells by the DV. If the
top-N-1 are at the DV floor and only the Nth has any signal, an
across-cell rank correlation is a one-point-leverage statistic, not a
measurement.

**Why:** The viable set hides the answer. A non-finding ("strong-NL
cosine ρ = +0.12 on n=6, n.s.") is not the same as "the predictor
fails under clean prompts" — when the cells where the predictor's
signal was strongest are systematically absent, the absence IS the
finding.

**How to apply:** Pull the DV per cell into a sorted list before
trusting any post-gate aggregate. If the viable subset's DV range is
< 0.10 (or whatever your "meaningful effect" threshold is) and the
dropped cells span the full DV range, write up:
  1. The gate result + which cells dropped + the cell-vs-DV alignment
  2. The dynamic-range collapse argument (sorted DV per viable cell)
  3. By-elimination interpretation if applicable, with explicit
     "answer is by elimination, not by direct measurement" framing
  4. Confidence MODERATE (not HIGH) on by-elimination answers — name
     the regime (different model, different elicitation) that could
     still rescue the direct measurement

Incident: task #467 (2026-06-03). Plan was 2x3 cosine x conditioning
sweep on cells passing strong-NL elicitation gate. Gate passed 6/18
(just clearing the 6-cell kill line), but those 6 were 5 floor cells
+ 1 mid-EM cell (turner_risky_financial). Dropped 12 included all 6
cells with broad-mis rate > 0.15. Sweep would have been a 1-point-
leverage Spearman; stopped before generating cosine numbers. Wrote up
the elicitation pattern + DV-range collapse as the finding.

Compare to [[check-partial-against-threshold]] — different failure mode
(viable but uninformative vs partial-vs-threshold).
