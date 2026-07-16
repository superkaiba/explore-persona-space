---
name: Smoke-slice sizes must satisfy downstream min-N asserts
description: Derive every smoke-sliced axis's size (questions/rows/steps/draws) from downstream consumers' min-N asserts, never from plan prose row counts (#1315 r4)
type: feedback
---

When porting a parent dispatcher's smoke slice, preserve every downstream phase's minimum-N, not just the cell subset: #1112's smoke kept `questions[:2]` because `run_geometry`'s split-half ceiling asserts >=2 distinct question ids; the #1315 port shrank it to `[:1]` (following the plan's literal "2-row stub" prose) and made the smoke's LAST phase un-passable by construction — a deterministic AssertionError at p10_geometry ~12 min into an otherwise healthy GCE smoke.

**Why:** a PASS_UNIFIED smoke (cell subset threaded through every phase) can still be un-passable by construction when a NON-cell smoke axis is sized below a downstream consumer's min-N assert.

**How to apply:** when sizing any smoke-scale slice, grep the downstream geometry/analysis modules for `assert len(...) >= k` shapes and derive the slice floor from them; fail loud at slice time with the reason (#1315 r4 added `_smoke_capture_slice()` with a capture-entry `assert len(qs) >= 2`). Plan prose row counts ("2-row stub") are not a floor spec.
