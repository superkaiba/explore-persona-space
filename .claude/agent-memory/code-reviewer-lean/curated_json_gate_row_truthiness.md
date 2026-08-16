---
name: curated-json-gate-row-truthiness
description: Probe curated-JSON safety gates with malformed rows — missing keys AND string-typed booleans ("false" is truthy); presence-only validators pass both
metadata:
  type: feedback
---

When a safety gate reads a hand-curated JSON inventory (rows humans triage in), probe it with malformed-ROW fixtures, not just a missing/malformed FILE: (1) each load-bearing key deleted, (2) each boolean field as the JSON STRING `"false"` (truthy in Python — `row.get("migrated")` on `"false"` reads as migrated), (3) the scoping list absent/empty (row scopes nothing → blocks nothing).

**Why:** #2321 R1 g5 — the I17 pre-deletion consumer gate failed CLOSED on a missing file (rc=22) but passed rc=0 on a row missing `silent_empty`, a row missing `prefixes`, AND on `"migrated": "false"`; the last variant ALSO passed the `--check` freshness validator (presence-only 4-key loop, no types), so NO mechanical layer caught it. Hand-curation was the DESIGNED workflow (the error message tells the operator to hand-edit the JSON), so the typo channel is the gate's own event class.

**How to apply:** for every gate whose predicate is `row.get(X) and not row.get(Y) and scoped(row)`, run three live probes through the real CLI (missing-key, string-bool, empty-scope) and check whether the companion validator checks TYPES or only key presence. Direction matters: a missing key on the BLOCK side (`migrated` absent → blocks) is fail-closed and fine; on the ARM side (`silent_empty` absent → not a blocker) it is fail-open. Fix shape: type-validate once at the shared loader, raising the gate's own typed block exception. Related: [[gate_threshold_vs_shard_config]], [[equality_gated_tally_shares_mask_dtype]].
