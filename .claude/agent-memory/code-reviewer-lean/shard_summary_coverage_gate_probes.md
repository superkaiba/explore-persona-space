---
name: shard-summary-coverage-gate-probes
description: reviewing a gate that unions per-shard summary reports — partition-agnostic glob + count coverage + producer-flag trust; live-invoke the gate on realized artifacts
metadata:
  type: feedback
---

When a gate aggregates per-shard summary files (e.g. `{split}_shard*.json`), run three probes: (1) the glob is usually PARTITION-AGNOSTIC — it matches `shard00of04` residue alongside `shard*of08`, so a `len(covered) < n_expected` COUNT check can be satisfied by stale/foreign cell keys masking a missing expected cell; the fix is set-difference against the registered cell universe (which also lets NOT-ESTIMABLE name the missing cells). (2) If FAIL keys on a producer-computed flag (`amendment_required`), check whether the artifact's recorded `threshold` is stored-but-never-compared against the gate's own constant ([[pilot_pass_report_fingerprint_unchecked]], [[registered_gate_quantity_substituted]]) — sharing the constant via `import producer as G` mitigates only same-version artifacts, not stale ones. (3) Live-invoke the gate function on the realized artifacts (`P._gate_x(root, split, declared)` in a heredoc) — it certifies denominator arithmetic (declared exclusions), key-format bridges (`|`→`__`), and the realized disposition in one shot, far cheaper than reasoning it out.

**Why:** #2658 round-12 g2: cap-hit gate rewired to per-shard gen_summary reports; count coverage + trusted flag both present; the live probe confirmed 131/131 coverage and an honest FAIL (worst 0.58) in seconds.

**How to apply:** any `_gate_*` that globs shard/summary files and compares a length to an expected count, or FAILs on a flag the producer precomputed. Also check per-shard fractions are whole-cell (cells partitioned by shard, `cells[idx::n]`) — partial-denominator fractions can only over-flag, not under-flag, when max-shard >= weighted mean.
