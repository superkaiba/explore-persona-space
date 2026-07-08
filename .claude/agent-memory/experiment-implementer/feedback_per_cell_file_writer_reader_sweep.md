---
name: per-cell file resolvers need a writer per cell class + group-reap fan-outs
description: A resolver eagerly requiring a per-cell file (selection.json) for every cell needs a writer for every cell CLASS it enumerates; and fan-out reaps must kill process GROUPS, not direct children
type: feedback
---

Two coupled traps from #1112 round 7 (2026-07-08):

1. **Writers-vs-readers sweep per cell class.** A resolver that eagerly
requires a per-cell file (e.g. `selection.json`) for EVERY cell it
enumerates needs a writer for every cell CLASS — #1112's m1 band-stop cell
had no writer (only ladder s-cells, reused, and m2 did) and died
deterministically at capture. Before launch, sweep writers vs readers per
cell class, and read such files only where actually used (lazy, not eager).

2. **Fan-out reap must kill the process GROUP.** `terminate()` on the
direct child of a `uv run`-wrapped vLLM unit leaves the python front-end +
EngineCore children alive; their 5-min handshake timeouts then masquerade
as an infra wedge, hiding the real first-unit crash (the #1112 attempt-4
misclassification). Spawn units with `start_new_session=True` and reap via
`os.killpg` TERM→KILL.

**How to apply:** any multi-cell dispatcher with per-cell metadata files +
a subprocess fan-out. (#1112 fix commit 9b912ba8b8.)
