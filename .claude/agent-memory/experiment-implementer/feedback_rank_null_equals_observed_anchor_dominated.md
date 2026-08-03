---
name: rank-null-equals-observed-anchor-dominated
description: A retrieval/MRR read whose shuffled-pairing null EQUALS the observed value is diagnosing prediction VARIANCE across contexts, not direction — the signature of an anchor-dominated v̂ = ymu + u with ||u|| << ||y - ymu||
metadata:
  type: feedback
---

In a rank-space read (lens-decoded vocab MRR, kNN retrieval, any
"is the true target ranked high for its own context" DV) built on a
prediction of the form `v̂(C) = ymu + u(C)`, an observed value that lands
ON its shuffled-pairing null is NOT primarily evidence about the map's
DIRECTION — it is the arithmetic signature of `||u(C)||` being small
relative to the spread of the target around `ymu`: every context's v̂ is
nearly the same vector, so every context's vocab ranking is nearly the
same, so re-pairing rows changes almost nothing and null ≈ observed BY
CONSTRUCTION.

**Why:** the shuffled-pairing null re-pairs v̂ rows against contexts. Its
power comes entirely from v̂ VARYING across contexts. When the constant
anchor dominates, the null and the observed statistic are computed from
near-identical rankings and must agree — a true null result and a
zero-variance prediction are indistinguishable in that one number.

**How to apply:** report the amplitude context alongside any at-null
rank-space verdict — the ratio of the typical `||u||` to the typical
`||y - ymu||` (#1776's `jacobian_rescale.json`
`amplitude_ratio_mean_resid_over_mean_ju` = 24.9 for the averaged
Jacobian, i.e. the varying part carried ~1/25 the norm of what it had to
explain), and check whether a rescale rung that AMPLIFIES `u` moves the
statistic. In #1776's J-chain read, scaling `u` by the train-fit
`s_a = 7.21` moved MRR 0.000366 -> 0.000546 while its own null moved
0.000345 -> 0.000411 — the observed tracked the null upward, which is
what amplitude-only rescaling of an unaligned direction looks like.
Never narrate "at the null" as "the direction carries no information"
without that check; and never treat "null ≈ observed" as a bug in the
null. Sibling reads: the identity+learned-bias / kNN-retrieval
dissociation (CLAUDE.md) — same family of "R² and retrieval disagree
because a constant shift explains the variance".

Related: [[feedback_parity_gate_determinate_data_blind]],
[[feedback_rank_space_bootstrap_tail_gating]].

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Rank-space null EQUALS observed when v̂ is anchor-dominated](feedback_rank_null_equals_observed_anchor_dominated.md) — MRR/retrieval at its shuffled-pairing null diagnoses prediction VARIANCE across contexts, not direction; report the ||u||-vs-||y−ymu|| amplitude ratio (#1776 jchain)

## Merged sibling index rows (#2032 curation, 2026-08-03)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the agent-memory index size cap (task #2032). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [rank-space bootstrap tail-mass gating](feedback_rank_space_bootstrap_tail_gating.md) — gate CI validity on >α/2 draws each side of a same-GEMM anchor; float lo<point<hi is epsilon-fragile (#825 r11)
- [Stratified nulls are not centered at chance](feedback_stratified_null_not_centered_at_chance.md) — test AUROC vs the null's OWN mean, never 0.5; persist null mean + direction (#1482)
