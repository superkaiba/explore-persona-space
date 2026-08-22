---
name: trap-sentinel-stale-and-voluntary-resume-key
description: "Trap-written pod sentinels: check the launch path rm's the PRIOR sentinel (truncated phases file but un-reset sentinel is the tell); and a VOLUNTARY resume predicate below the T1/T2 triggers still owes the full regime key (#2224 R5)"
metadata:
  type: feedback
---

Two review recipes from #2224 R5 (suite4a runner + builder), both in otherwise-clean new code:

1. **Trap-written sentinel without launch-time clear.** A detached runner that writes its
   status sentinel ONLY from an EXIT trap must `rm -f "$SENTINEL"` at init. The tell: the
   launch block truncates its phases/progress file (`: > "$PHASES_FILE"`) and per-phase logs
   but leaves the sentinel path untouched — so a same-pod relaunch (the crash-fix path)
   exposes the PRIOR run's `status: done|failed` envelope to the orchestrator's poll for the
   whole run. Sweep every launch-time-reset file against every trap/exit-time-written file;
   any exit-written path missing from the init resets is the finding. Sibling classes:
   [[sentinel-path-outside-drain-glob]] (WHERE it is written), [[handrolled-pod-sentinel-envelope]]
   (WHAT keys it carries — also re-confirmed this round: `poll_pipeline._SENTINEL_REQUIRED_KEYS`
   = sentinel_schema_version/kind/version; a non-conforming non-`-results.json` file is
   silently skipped by the drain, so a bare-envelope sentinel is only valid when the runbook
   pins DIRECT SSH polling); THIS one is WHEN it goes stale.

2. **Voluntary resume predicates still owe the full regime key.** Step 3.6's verdict routing
   binds only when T1 (>~1h) / T2 (>~50 units) fire, but a checkpoint/resume a diff adds
   VOLUNTARILY (e.g. 24-dataset parts, minutes of wall) gets the same regime-key check: a
   predicate keyed only on input sha + output existence silently reuses stale parts when any
   output-affecting arg (`--max-prompt-tokens`, caps, seed, model) changes — AND the manifest
   then stamps the NEW args over OLD-regime parts (mislabeled provenance, worse than no
   resume). Don't skip the check because the trigger table says the loop is small.

**Why:** #2224 R5 — both shipped in a round whose smokes were exemplary (full-grain build,
real consumer round-trips); severity Major each, persisted as concerns
(`suite4a-stale-sentinel-relaunch`, `suite4a-build-resume-regime-key`).

**How to apply:** any diff adding a detached runner with an EXIT-trap sentinel, and any diff
adding a resume/skip predicate regardless of loop size.
