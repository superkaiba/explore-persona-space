---
name: gate-mean-nanmean-denominator
description: A decision gate averaging per-unit stats via np.nanmean silently narrows the registered denominator — a NaN unit (zero-variance = the anomaly the gate guards) drops out and the gate can PASS on N−1 units (#2379 R1 g4)
metadata:
  type: feedback
---

When a registered decision gate is "mean over N units ≥ threshold"
(languages, conditions, shards), check the aggregation call: `np.nanmean`
lets a NaN per-unit statistic drop out of the mean silently, so the gate
passes on N−1 units with disclosure only in a sidecar field — and the NaN
regime (zero-variance rate vector ⇒ undefined Spearman) is often exactly
the broken-replication signature the gate exists to catch.

**Why:** #2379 R1 g4 (`issue2379_analysis.py::run_gate`): Gate G1 =
"mean within-language Spearman ρ ≥ 0.4" over 3 caps languages; code took
`np.nanmean(rhos)` guarded only against the ALL-NaN case, so one
variance-collapsed language would be averaged away and 7 GPU-h of gated
spend released. Flagged Major; fix is `np.isfinite(rhos).all()` or an
n-units-in-mean disclosure + FAIL when short.

**How to apply:** for every gate whose statistic is a mean over units, (1)
name the registered denominator N from the plan sentence; (2) trace the
aggregation for nanmean/masked means/short lists; (3) require either full
finiteness or an explicit n-in-mean field that the PASS predicate checks;
(4) also check the printed/marker PASS line discloses the realized
denominator. Sibling of [[registered-gate-quantity-substituted]] (wrong
quantity; this one is right quantity, narrowed denominator) and
[[gate-threshold-vs-shard-config]] (gate dead from config drift).
