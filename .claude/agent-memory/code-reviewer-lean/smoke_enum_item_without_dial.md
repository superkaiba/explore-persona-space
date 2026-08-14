---
name: smoke-enum-item-without-dial
description: Verify each plan §4.8 smoke blind-spot enumeration item has a REACHABLE implementing dial in the round's entrypoints (#2225 R1 g3 — P0 MMLU --limit 200 promised, no --limit flag anywhere)
metadata:
  type: feedback
---

For every plan §4.8 "Smoke blind-spot enumeration" item that promises a reduced-scale smoke (e.g. "P0 MMLU runs `--limit 200`"), grep the round's entrypoints AND the dispatcher for the named dial — the enumeration discloses what the smoke does NOT certify, but an item can also promise coverage that was never implemented, leaving the path production-first.

**Why:** #2225 R1 g3 — plan §4.8(b) declared a P0 MMLU `--limit 200` smoke; `issue2225_mmlu.py` had no `--limit` flag and `issue2225_dispatch.sh` never invoked mmlu pre-P2c, so a results-parse defect would have burned 86 full MMLU evals before surfacing (fan-out raises only at the end). The `--smoke` flag narrowed only the TARGET set, not the question set — a target-narrowing flag is not a scale dial.

**How to apply:** when reviewing an eval/gen script commit, list the plan's per-phase smoke promises and check each maps to (a) an existing CLI dial in the script, (b) a dispatcher invocation using it, and (c) fingerprint keys covering the dial so a smoke-scale run never resume-satisfies production. Related: [[mode_scoped_column_threading_untested]] (a facility only one mode consumes needs that mode exercised).
