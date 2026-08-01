---
name: Resume predicates must honor recorded terminal-verdict sidecars
description: A stage that persists a terminal-verdict sidecar (e.g. topup_record.json with union_floor_missed=true) and lets the run CONTINUE must resume-skip on that recorded verdict — re-entering the stage on relaunch trips its own one-shot guard and turns a survivable recorded state into a hard crash. #1947 P0 launch 4.
type: feedback
---

Rule: when a pipeline stage writes a sidecar recording a TERMINAL
disposition (a yield-floor miss adjudicated by salvage/equalize-down, a
one-tranche-consumed record, any "this stage concluded with verdict X"
file) and the run legitimately CONTINUES past it, the stage's RESUME
predicate must recognize that sidecar and re-apply the recorded
disposition (skip, or re-emit the salvaged pool) — never re-enter the
stage body, whose one-shot guards (one-tranche-per-cell, once-only
retries) will refuse and crash on state the prior run survived.

**Why:** #1947 P0 launch 4 (2026-08-01): `_positive_topup_stage`'s
one-tranche guard (`artifacts/datagen.py:1158`) crashed the whole
relaunch ~3s in because `positives/syc-icl/topup_record.json` recorded
the single allowed tranche with `union_floor_missed=true` — a state the
ORIGINAL run continued past via the equalize-down path. Fourth distinct
crash-on-resume/volume class in one pipeline shakeout.

**How to apply:** when writing ANY resumable multi-phase driver, sweep
every phase for the pair (terminal-verdict sidecar, one-shot guard) and
make the resume predicate check the sidecar FIRST; test the
relaunch-after-recorded-miss path explicitly (the crashed state is
reproducible by construction — write the sidecar in the fixture).
