---
name: Smoke-gate expectations computed from realized slice size, never an assumed cap
description: A smoke-gate floor (checkpoint/step/file count) calibrated on an ASSUMED row count fails when the realized smoke yield is smaller; derive the smoke dial (epochs/steps) from realized arithmetic so the floor holds for ANY realized n
type: feedback
---

A post-smoke gate asserting a training-side artifact floor (e.g. ">=2
checkpoints", the multi-LoRA dose-probe floor) must have its smoke dial DERIVED
from the realized slice arithmetic, never from an assumed row count in a
comment. On #1489 (2026-07-18) the smoke distill JSONL was CAPPED at 30 rows
but the realized gen yield was 2; under the production effective batch of 16,
`epochs=1` gave `ceil(2/16) = 1` optimizer step → 1 checkpoint against a >=2
gate — a full GCE relaunch died at the post-smoke assert after every phase had
passed. Fix shape: compute `epochs = max(1, ceil(K / steps_per_epoch))` from
the realized n_rows (K = the gate's floor), keep production geometry untouched,
and pin the arithmetic with a fails-pre-fix test plus a tiny-real leg that
asserts the checkpoint dirs exist on disk (config math alone does not prove
Trainer save behavior). (#1489 crash-fix r5, commit 63c10815.)

## Merged sibling index rows (#1891 curation, 2026-07-30)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the ~25 KB loader truncation limit (task #1891). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [Smoke-slice sizes must satisfy downstream min-N asserts](feedback_smoke_slice_min_n_downstream_asserts.md) — derive smoke slice floors from downstream `assert len >= k` consumers, not plan prose (#1315 r4)
- [Smoke-scale gates](feedback_smoke_scale_gates.md) — production-n-calibrated verdicts (anchor tolerances, yield floors) bind spuriously at smoke n; demote to informational under --smoke, keep production pins (#1345)
- [Smoke-gate floors from realized slice arithmetic](feedback_smoke_gate_realized_slice_arithmetic.md) — derive smoke epochs/steps from realized n_rows, never an assumed cap; pin with a fails-pre-fix test + on-disk checkpoint assert (#1489 r5)
