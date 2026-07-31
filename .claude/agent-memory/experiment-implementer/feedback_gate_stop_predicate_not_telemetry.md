---
name: Gate the stop predicate, never the telemetry
description: min_steps-style gates on a telemetry callback silently produce NO output file for short runs (T < gate), crashing only at the post-sweep consumer; a configured out_path is an unconditional train-end write contract
type: feedback
---

A telemetry callback whose PROBING sits behind a min_steps-style gate produces NO output
file for cells with T < gate, and the crash surfaces only at the post-sweep consumer
(FileNotFoundError far from the cause). Gate only the stop/decision predicate — probing +
file writes must run regardless — and treat a configured out_path as an unconditional
train-end write contract (`n_probe_records: 0` is a valid file; absence is not).

**Recovery without retraining:** when the gated file is missing for already-trained
cells, look for an UNGATED probe over the same rows/gauge before considering retraining —
in #601, `rowtype_ce.json` (per-step positive-row marker-token CE, live model) gave
`delta = base CE − step CE` = the identical live-gauge ΔG the band trajectory would have
carried.

**Why:** incident #601 round 8 (2026-06-11) — all five T=13 cells lacked
`inloop_band_trajectory.json` (MarkerBandStopCallback disabled itself at
max_steps < min_steps), crashing the phase-4 verdict after a 17/17-complete sweep.

**How to apply:** when wiring any periodic-probe callback into cells of heterogeneous T,
check every gate (min_steps, eval_every, warmup skips) against the SHORTEST cell, and
smoke the telemetry on that shortest cell, not just the default smoke cell.

## Merged sibling index rows (#1891 curation, 2026-07-30)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the ~25 KB loader truncation limit (task #1891). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [Threshold-gated telemetry never fires on short runs](feedback_threshold_gated_telemetry_short_runs.md) — min_steps≥T cells ship NO trajectory file; gate the stop, not the probe. #601.
- [Gate the stop predicate, never the telemetry](feedback_gate_stop_predicate_not_telemetry.md) — check every gate against the SHORTEST cell + smoke its telemetry output. #601.
