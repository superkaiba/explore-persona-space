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
