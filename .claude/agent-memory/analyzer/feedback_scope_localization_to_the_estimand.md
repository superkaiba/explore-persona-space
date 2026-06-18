---
name: Scope localization claims to the measured estimand
description: A delta-rho contrast on the POOLED correlation does not license "rank information is concentrated in cohort X" — check forest medians, FE, source-marginal, and two-way reads in BOTH cohorts before writing any localization heading.
type: feedback
---

When a between-cohort Δρ contrast (pooled residual correlation difference, CI excluding zero) localizes an effect to one cohort, the licensed claim is "the POOLED <DV> correlation is concentrated in cohort X" — nothing broader.

**Why:** Task #539 round-1 headline said "whatever rank information geometry carries is concentrated in the ordinary cohort" while the same output JSON showed the OTHER cohort carrying comparable-or-larger rank information on every non-pooled read (per-context forest median, context-FE, source-marginal, two-way FE). Both ensemble critics bounced the draft on this single scoping error plus its Human-TL;DR echoes ("erases the strip entirely", "I'd call it noise").

**How to apply:** Before writing any localization/concentration heading: (1) list every ρ variant the analysis computed (pooled raw/resid, FE, two-way, partial, source-marginal, per-context forest) for BOTH cohorts; (2) if any non-pooled read is comparable or larger in the "dead" cohort, scope the heading to the pooled read and state the surviving reads as a bounding note; (3) scope the DV too (binary emission rate vs graded log-prob) — a pair-specific signal can survive in one DV space and not the other; (4) caveat n=16 marginal reads for tie mass + duplicated points before quoting them as "huge".
