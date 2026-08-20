---
name: rename-rework-strands-secondary-readers
description: After an architectural naming/grain rework (shard batch-id change), grep the fork for the OLD literal — stale globs and equality flags silently no-op plan-registered mechanisms
metadata:
  type: feedback
---

When a revision round reworks an artifact NAMING GRAIN (e.g. worker-grain
`anchors_gate_w{i}` -> cell-grain `anchors_gate_{cell}_w{i}` batch ids), the
writers and PRIMARY consumers get updated, but SECONDARY readers keyed on the
old literal survive silently: a `glob("<old-prefix>*")` that now matches
nothing, and an equality flag (`batch == "gate"`) that is now always False.
Both fail SILENT (empty aggregation, mis-bucketed rows), not loud.

**Why:** issue #2389 round 2 (`e159c259ea`) moved anchor sharding to cell
grain per the r1 review; the cap-recalibration barrier and consumer probe
were updated, but `run.py` kept `glob("anchors_gate_w*.jsonl")` (matched
zero cell-grain shards — the plan §7 item-1 recalibration checkpoint became
structurally inert) and `"gate_slice": batch == "gate"` (always False —
mis-bucketed the cap-hit report's gate/rest split; the vLLM fork computed
the same flag correctly, so producers disagreed). Zero fork-test coverage on
either; the parent's tests still used the old names.

**How to apply:** on any round whose fix renames a shard/batch/file-id
namespace, run `grep -n "<old literal>" ` over the WHOLE fork (all sibling
scripts + tests) and disposition every hit — same shape as the crash-fix
symbol-rename grep duty, but for STRING literals, which that duty does not
cover. A plan-registered mechanism whose reader aggregates ZERO rows after a
successful barrier should be flagged as a missing planned component
(SUBSTANTIVE), even when a downstream backstop protects the science —
the plan approved the mechanism, and its silent no-op is undeclared drift.
Related: [[judgment-prose-enforced-halt-gate]] (same round, gate-wiring half).
