---
name: Threshold-gated telemetry never fires on short runs
description: Callbacks whose probe/file-write sits behind a min_steps/threshold gate silently produce NO artifact when T < gate; check gate arithmetic vs each cell's T at implementation time, and make a configured out_path a train-end write contract.
type: feedback
---

A TrainerCallback whose probing (and therefore its trajectory-file write) is
gated behind a `min_steps`-style threshold silently produces NO output file
for any cell whose total steps T < the gate — and the crash surfaces only at
the post-sweep analysis step that consumes the file.

**Why:** #601 round 8 (2026-06-11): `MarkerBandStopCallback` (min_steps=20)
disabled itself entirely on the T=13 Phase-4 bridge cells; no probe fired, the
records-gated `on_train_end` flush never wrote `inloop_band_trajectory.json`,
and `i601_phase4_verdict.py` crashed FileNotFoundError after a 17/17-complete
sweep. The ungated sibling probe (`RowTypeCETrainProbeCallback`, fires every
step) had the same construct in the same gauge and rescued the verdict
zero-GPU: `delta = pos_marker_ce_base − pos_marker_ce` = logP_live − logP_base.

**How to apply:** (1) at implementation time, compare every telemetry gate
(min_steps, eval_every, warmup skips) against the SHORTEST cell's T in the
registry — a gate ≥ T means that cell ships no telemetry; (2) only the
STOP/decision predicate should sit behind min_steps — probing + logging should
run regardless; (3) a configured trajectory/out path is a train-end write
contract: write the file even with zero records (`n_probe_records: 0`) so
consumers can distinguish "probe never fired" from "cell never trained";
(4) when a gated probe's file is missing for already-trained cells, look for
an ungated probe over the same rows/gauge before considering retraining.
