---
name: Single determinate cell — check outlier AND power, opposite directions
description: When a verdict rests on one determinate cell of a panel, audit it two ways (structural outlier = downgrade; best-powered-and-still-lost = rule out favorable-artifact); both belong in the headline finding.
type: feedback
---

When a clean-result verdict rests on a SINGLE determinate cell of a multi-cell
panel (one resolving bank / one CI-excludes-0 condition / one significant
stratum among many), run TWO independent audits on that cell and put both in
the headline finding — they cut OPPOSITE ways:

1. **Structural-outlier audit → DOWNGRADES confidence.** Is the determinate
   cell ALSO the panel's lone-of-its-kind? Only layer L21 (others L20/L15),
   only persona-vector family, only multi-arm design, only 52-fold design. If
   so, the effect could be a property of that cell's structure rather than of
   the manipulated variable generally. State it as the binding LOW/MODERATE
   caveat.

2. **Power / clean-conditions audit → RULES OUT the favorable artifact.** Is
   the determinate cell also the BEST-powered / cleanest-measurement cell
   (largest N, smallest CV-leak, least saturated)? If the loser LOST there —
   where it had its best shot — that rules out the "loser only lost because of
   a measurement artifact (centering-leak, small-N noise, saturation)"
   alternative. State it as the analyzer-favorable rule-out.

**Why:** the reader needs both. Outlier-alone reads as "the whole signal is one
weird cell" (over-cautious); power-alone reads as "and it's the best cell so
trust it" (over-confident). Together they pin the residual uncertainty exactly:
the only surviving doubt is the structural uniqueness, which is precisely what
a LOW (not lower) confidence tag encodes.

**How to apply:** any panel-of-banks / multi-condition / multi-stratum analysis
where the headline rests on one determinate cell. Pull N / layer / family /
design / saturation per cell from the raw JSON and sort — the outlier and the
best-powered cell are often the SAME cell (#648: #505 was both). Both audits
go in the headline `### <finding>` read prose and a `## Takeaways` bullet.

Origin: #648 round 2 — both interpretation critics independently flagged the
two halves (Claude MF1 = structural outlier of #505; Codex rev #3 = #505 is the
largest panel, ruling out a centering-leak artifact). Round 1 had stated
neither; the pair is what makes a single-determinate-sign verdict defensible.
