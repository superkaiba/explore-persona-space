---
name: 2658-gen-order-manifest-shapes
description: "#2658 gen_order_manifest carries a dry-run split-total shard00of01 beside the 8-shard set: reconcile summaries per shard by identical filename, never a directory sum; loader-vs-gate validation split (floor is a GATE_FAIL verdict, not a loader raise)"
metadata:
  type: reference
---

Two #2658 artifact facts future P4/P5 rounds will hit (learned r14):

1. **`gen_order_manifest/` holds BOTH a split-total `pilot_shard00of01.json`
   (the dry-run froze it, n_requests=6290, and `issue2658_pilot_timing.py:205`
   deliberately reads it) AND the 8 per-shard `pilot_shardNNof08.json` files
   (sum 6290).** A naive directory glob+sum doubles the declared total
   (12580) and would have made the r14 gate mixed-totals check FAIL the
   realized tree. Reconcile each `gen_summary/{split}_shardNNofMM.json`
   `cap_hit.n_records` against the SAME-NAMED order manifest's `n_requests`,
   then check the shard00of01 split-total separately when `MM > 1`. Probe the
   real tree BEFORE freezing check semantics: the false-FAIL was caught by
   running the reconciliation on `eval_results/issue_2658` first.

2. **Shared loader validation splits by consumer verdict semantics:**
   `issue2658_generate.validate_cap_amendment_values` (single source, power
   aliases the schema id) enforces schema equality, positive-int caps and a
   non-empty offender map for BOTH loaders, but the production >= 2x pilot
   floor is `require_2x_floor=True` only in the generate loader. Power's
   `load_cap_amendment_record` passes `require_2x_floor=False` because a
   below-floor record must stay a `_gate_cap_hit` GATE_FAIL verdict (a
   test pins that), never a loader raise. When sharing validation between
   loaders, parameterize any check that is a reported verdict elsewhere.

Fixture coupling: `tests/test_issue2658_unit8.py::_write_gen_summaries` now
writes the per-shard order manifests too. Any new gate-side artifact
requirement means updating that ONE fixture, which fixes every downstream
gate test. Related: [[hardlink-copy-gate-smoke-live-dir]],
[[judge-pilot-report-resume-fields]].
