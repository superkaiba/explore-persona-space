---
name: Fanout units must never lazily stage a shared input dest
description: Pre-stage shared inputs ONCE in the parent before any fanout; staging helpers need per-invocation mkdtemp dirs + atomic publish + full-file-set staleness guards (#1315 r5)
type: feedback
---

A fanout dispatcher whose units lazily stage a SHARED input dest races the staging helper two ways: a shared staging dir lets one unit's `os.replace` steal a file a sibling's `hf_hub_download` just returned (FileNotFoundError), and a single-file staleness guard lets units consume a sibling's partially-staged dest (#1315 r5: 4 concurrent parity units each staging `margin_pools/impolite` via `issue1090_run._stage_hf_prefix`'s shared `_hfstage/` scratch).

**Why:** a one-cell smoke structurally cannot catch this class — the race needs >=2 concurrent units sharing a staging dest, so it first fires in production fanout.

**How to apply:** at fanout-design time, audit every unit-reachable shared-write path (staging dests, caches, output files). Pre-stage shared inputs ONCE in the parent before `_fanout_units`; give staging helpers per-invocation `mkdtemp` staging dirs + atomic per-file publish (never a shared scratch dir); key staleness/already-staged guards on the FULL consumer-required file set, never one proxy file.
