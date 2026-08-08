---
name: Long-running analysis script must support per-unit atomic writes + --resume
description: A multi-iteration CPU/GPU analysis script with wall-time ≥1h that writes only end-of-run loses ALL completed work on any mid-run crash; on a shared contended VM (load avg 60+) crashes are likely. Always write per-unit atomic JSON + opt-in --resume with fail-loud substrate match.
type: feedback
---

A long-running analysis script (wall-time ≥1h) that produces a single
end-of-run JSON has no resume/restart story — a crash mid-run loses ALL
completed work. On the shared VM (~40+ Happy sessions, load avg routinely
60–90) crashes from contention OR cross-session interference are likely.
Plan-time wall-time estimates also routinely under-call: per-fit times
run 2–5× lightly-loaded estimates when load is high.

**The rule (defense-in-depth, both halves required):**

1. **Per-unit atomic writes.** The moment EACH unit (layer / cell / fold)
   finishes, write its result to `<outdir>/<units>/<unit_id>.json`
   ATOMICALLY (`.tmp` + `fsync` + `os.replace`). Never accumulate-in-memory
   then write-once-at-end. Aggregate from the per-unit files at the very
   end into the canonical schema-conformant JSON; FAIL LOUD if any unit
   file is missing (do NOT write a partial canonical JSON — that silently
   hides missing data).
2. **Opt-in `--resume`.** A `--resume` flag enumerates `<outdir>/<units>/`
   at startup and SKIPS units whose JSON already exists, after a fail-loud
   substrate-match check (assert seed unchanged, substrate hash unchanged,
   schema version unchanged). The skipped units' results are loaded into
   the aggregation step verbatim — never recomputed when the recipe
   hasn't changed.

**Wall-time hygiene:** plan estimates that depend on tight inner loops
must include a contention adjustment for the shared VM. If `nproc` is
contended (load avg / nproc > ~2), expect 2–5× slowdown over a
single-tenant timing. Post `epm:compute-deviation v1` when realized
wall-time runs >2× plan §9 and continue per `workflow.yaml §
pivot_criteria.compute_deviation_over_2x` (vectorize-first when the
deviation is overhead-bound, then auto-descope where possible, else
`continue_as_is` — at ≥5× only with a recorded quantified
irreducibility finding — and update §-assumptions).

**Incident:** #722 base-skill-over-mean canonical run died at L10/28
(~1h35m elapsed) under load avg 87. Round-1 script accumulated in memory
and wrote only at end → all 10 completed layers' results were lost. Round-2
patch (commit `8a0c2cf848`) added Design A (per-layer atomic JSON + opt-in
`--resume`); the re-launch survives unbounded crashes (each completed
layer is a durable checkpoint).

## Merged sibling index rows (#1891 curation, 2026-07-30)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the ~25 KB loader truncation limit (task #1891). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [Long-running analysis needs per-unit atomic writes + --resume](feedback_long_running_analysis_needs_resume.md) — a multi-iteration CPU/GPU analysis script with wall-time ≥1h that writes only end-of-run loses ALL completed work on any mid-run crash; on a shared contended VM (load avg 60+) crashes are likely. Always per-unit atomic JSON + opt-in --resume with fail-loud substrate match. #722 r2.
- [Eval-rig per-phase checkpoint](feedback_eval_rig_per_phase_checkpoint.md) — persist each sub-phase (gen/logprob/judge) to disk the moment it completes; never write at end-of-seed. #399.
