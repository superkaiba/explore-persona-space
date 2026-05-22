---
name: dose-response-fwer-with-shared-randomness
description: Per-adapter dose-response tables with "≥1 of N adapters fires" across multiple probe types inflate FWER multiplicatively; cross-probe tests share input randomness so per-cell independence is the wrong null
metadata:
  type: feedback
---

When an experiment frames a hero figure as a "per-adapter × per-probe dose-response table" with the success criterion "≥1 of N adapters fires under probe T" for each of M probe types, the family-wise error rate is N×M tests, NOT M tests. With paired-bootstrap CIs at α=0.05 and N=4 adapters × M=4 probe types (zero-shot, installation, drift type 1, drift type 2, drift type 3 collapses to 4 inferential strict-tests), FWER under independence is 1−0.95^16 ≈ 56%.

The standard fixes are: (a) Bonferroni-Holm across the N×M cells (use α = 0.05 / (N·M)); or (b) per-cell hierarchical conjunction (require persona-voiced installation > 5% AND drift > base+5pp AND drift-base < 5% on the SAME adapter as a single test, so the "≥1 of N" is a single union over adapter-level conjunction outcomes, not a union over cells).

**Critical wrinkle:** when probes share input randomness (same drift-template bank, same 200-query held-out set across all N adapters), the N adapter-level tests within a probe family are NOT independent — they share prompt-construction noise. This means: (1) Bonferroni overcorrects within a probe family; (2) the correct denominator is closer to M than N×M for within-family inflation, BUT (3) across-family inflation (4 families) still applies. Specify aggregation: either "max statistic across N adapters with permutation null" (correct within-family FWER) OR "intersection across N adapters" (very conservative confirmation gate).

**Why:** /adversarial-planner v4 #375 round 2 (2026-05-21). Plan grew from 4-cell Primary A union test (Codex round 1 estimated FWER ≈18.5%) to 16 strict cells across two parallel primaries (installation + 3 drift types), without addressing FWER. Cross-adapter dependence due to shared drift-template bank + shared 200-query held-out was also unspecified.

**How to apply:** Whenever a plan's hero figure is a per-adapter × per-probe table and the success criterion fires on "≥1 of N adapters" within each probe column, ask: (1) what's m_total = N × M? (2) how is per-cell α controlled? (3) do probe families share randomness (templates, queries)? If (3) is yes, "best of N adapters" within a family is NOT an N-independent-tests inflation; specify the aggregation (max-stat with permutation null OR intersection).

Related: [[feedback_alternatives_lens_round2.md]] — bootstrap percentile CI degeneracy near 0 reappears when drift cells fire weakly (3-10% range). Combine FWER correction with BCa or permutation tests for the boundary-rate drift cells specifically.
