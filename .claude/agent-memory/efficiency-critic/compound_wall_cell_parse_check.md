---
name: compound-wall-cell-parse-check
description: Run plan_wall_budget.parse_wall_cell on any compound/SLA §9 wall cell — "0.5 VM + ≤24 calendar" parses as 0.5, dropping the Batch SLA from the tripwire fence
metadata:
  type: feedback
---

When a §9 `planned_wall_h` cell is COMPOUND (two quantities in one cell, e.g.
`0.5 VM + ≤24 calendar`), run it through
`src/explore_persona_space/plan_wall_budget.py::parse_wall_cell` before
passing the row: the parser takes the FIRST number, so the Batch-SLA upper
bound is silently dropped from the #873 phase-ETA tripwire budget — the same
under-fence shape as #2162's `2–24 calendar` → 2.0 incident the grammar row
was added for.

**Why:** a low-parsed wall cell makes the tripwire false-fire on a healthy
in-SLA Batch wait (wasted escalation attention; on a pod-holding phase it
could provoke a wrong kill/diagnosis). Found live in #2329 `q35_ladder_decay`
plan v5 L7 row (2026-08-19): `0.5 VM + ≤24 calendar` → 0.5. Fixed in v7
(`≤24 calendar` → 24.0, VM figure moved to basis; c47 read 9 rows/31.10 h) —
the fix shape works, parser-verified.

**How to apply:** in PLAN MODE, when any §9 row mixes a VM wall with a
calendar/SLA bound in ONE cell, test-parse it (one `uv run python -c` line).
Fix is mechanical: wall cell carries only the fence-relevant bound
(`≤24 calendar`); the other figure moves to the `basis` column. Non-blocking
when the phase holds zero GPUs (tripwire noise only), blocking-adjacent when
the low-parsed row rides a live pod.
