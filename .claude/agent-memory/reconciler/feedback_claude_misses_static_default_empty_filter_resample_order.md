---
name: Claude misses static-default / empty-filter / resample-order bugs masked by single-seed synthetic smokes
description: code-review FAIL calibration — three production-only defect classes Claude PASSed because its smokes were single-seed + synthetic-fixture; trace the PRODUCTION arg grid, not the smoke args
type: feedback
---

When Claude's code-review PASSes a stats/dispatch artifact citing "re-ran the
smokes, fixes verified," re-verify against the PRODUCTION launch path — the
smoke's arg grid often structurally cannot exercise the defect. Three
production-only classes, all upheld FAIL at #641 r2 (Codex caught all three,
Claude PASSed all three):

1. **Static-CLI-default vs runtime-computed parameter (M1).** A code comment
   promises a value is "resolved from <data>" (median crossing, threshold from
   a prior phase, etc.), but the line is literally `x = args.x` with a CLI
   default. No resolution code exists anywhere (grep the package). The plan
   pre-registers the computed value as PRIMARY with the default as FALLBACK;
   the code silently always uses the fallback. **Tell:** comment says
   "resolved from X" + the assignment reads a CLI default + grep finds no
   computation of X. (#641: `matched_dose = args.matched_dose` default 375,
   plan §4.5 median-crossing primary, comment 947-948 promised it, zero
   median-crossing code.)

2. **Silent-empty filter via `if k in <dict>` (M2).** A fallback pool is
   filtered by membership in a propensity/measurement dict — but the upstream
   measurement phase never populated those keys, so the filter ALWAYS yields
   `[]` and the entire fallback branch is dead code. The selector silently
   returns the bad primary-pool result instead of the plan-designed fallback.
   **Tell:** `widened = [k for k in pool() if k in measured_dict]` where
   `measured_dict` provably never contains `pool()` keys (the default
   measurement-phase `contexts`/`keys` list omits them). Trace what the
   PRODUCING phase actually measured, not just the consumer's filter. (#641:
   P0 measured only `ARM_B_NARROW_NEUTRAL_KEYS`; `data.py:276`
   `if k in candidate_propensity` emptied the widened pool every time.)

3. **Bootstrap resampling at the WRONG loop level (M3).** A hierarchical
   bootstrap over a CURVE / multi-step quantity draws the top-level unit (the
   plan's "true replication unit" — usually the seed) INDEPENDENTLY inside the
   per-step loop instead of ONCE per replicate reused across all steps. A
   single replicate then stitches different seed draws at different doses → a
   curve no real run produced → the curve-level / asymptote-difference CI is
   corrupted (under-stated uncertainty). **Tell:** the per-replicate loop calls
   `_resample(...)` separately per step, and the resample fn does
   `boot_units = rng.choice(units, ...)` at its top → fresh draw per call.
   SCOPE CHECK before upholding the full blast radius: a SINGLE-point read
   (one resample call per replicate, e.g. a matched-dose ΔL) is COHERENT and
   unaffected — only multi-step/curve reads break. (#641: `bootstrap_dose_curve`
   / `bootstrap_class_asymptote_difference` per-step `boot_seeds`; H5 CIs broken
   but Arm-B ΔL H1/H2 single-dose read fine — Codex slightly over-stated, still
   FAIL because H5 is a pre-registered headline.)

**Why the smokes masked all three:** the GPU smoke passed `matched_dose=2`
hardcoded (M1 invisible), `seeds=[42]` single-seed (M3 invisible — incoherent
trajectory needs n≥2 seeds to manifest), synthetic propensities that don't trip
the all-primary-miss fallback (M2 invisible). Single-seed + synthetic-fixture
smokes are exactly the configuration that cannot catch production-grid bugs —
see also feedback_claude_synthetic_fixture_smoke_masks_args_grid_bug.md. When
the PASS rests on a re-run smoke, ask: does the smoke's arg grid (seed count,
matched-dose value, candidate pool) match the PRODUCTION grid the defect lives
in? If not, the green smoke proves nothing about the cited path.
