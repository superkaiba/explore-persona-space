---
name: Codex reads statistic prose vs the plan-pinned Source implementation
description: When a plan registers a statistic by prose name AND a pinned Source implementation, the Source defines the registration — open it before believing a "deviation" FAIL
type: feedback
---

When a plan registers a statistic with BOTH a prose name ("crossed cluster
bootstrap over (claims × personas)") AND a pinned Source implementation
("mirrors `_compute_matched_rate_gap_514` lines 367-430"), the SOURCE
IMPLEMENTATION is the registration. Codex FAILed #606 r3 reading the prose
term against the textbook two-way (Cameron-Gelbach-Miller) bootstrap —
claim resample shared across all personas — without opening the Source.
The Source (#508 `_crossed_cluster_bootstrap_gap`, via #514) draws
`per_persona_q_picks[persona]` (independent per persona, locked across
cells) and its own comment defines the "load-bearing 'crossed' bit" as the
persona set shared across cells. #591's `_paired_bootstrap_ci` likewise
pairs claims across trained/base within ONE cell, never across personas.
The #606 code matched both Sources exactly (and improved on #514 by
re-estimating s per replicate, as the plan registered) → PASS.

**Why:** the prose name is a label; the pinned implementation is the
operational definition. Codex's prescribed "fix" would have DEVIATED from
registration. Distinguish from #491 (FAIL was right): there the implemented
statistic deviated from the registered set/dof/count; here it matched.

**How to apply:** for any "implemented bootstrap/CI ≠ registered statistic"
finding, `git show` the plan's cited Source function and byte-compare the
resampling structure (what is shared vs independent across each factor,
what is locked across cells/arms, seed, B, denominator set). If the code
matches the Source, the residual statistical critique (e.g. missing
cross-persona covariance → anti-conservative CI) routes to PASS + persisted
CONCERN for a free post-hoc sensitivity variant when raw per-rollout
verdicts persist. Origin: task #606 round-3.
