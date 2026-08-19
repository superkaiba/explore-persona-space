---
title: 'workflow-fix: test_tick_triage module-level NOW makes fixtures wall-clock-sensitive
  in long gate sessions'
kind: infra
tags:
- wf-fix
created_at: '2026-08-18T11:34:19Z'
has_clean_result: false
origin_prompt: 'Auto-filed by the /issue 2158 session: Step 9c gate compare classified
  4 tick_triage api-error-probe failures NEW against an innocent branch; root cause
  tests/test_tick_triage.py:37 NOW=time.time() at module level + 22.2 min collection-to-execution
  drift (junit arithmetic exact match to the observed ''marker age 28m'').'
workflow: v1
---
# tests/test_tick_triage.py fixtures are wall-clock-sensitive: module-level `NOW = time.time()` silently ages every "fresh" marker in long gate sessions

<!-- workflow-fix-candidate v1 -->

**Target file:** `tests/test_tick_triage.py` (line 37: `NOW = time.time()`; consumer `_event(kind, age_s)` writes `ts = NOW - age_s`).

**Defect.** `NOW` is captured once at module IMPORT (pytest collection, session start), while `tick_triage.triage()` measures marker/transcript ages against the LIVE clock at test EXECUTION. In any pytest session where the file executes ≳20 min after collection, every fixture event written as "fresh" (e.g. `_event("epm:progress v1", 300)` = 5 min old) is silently ~drift+5 min old at assertion time — old enough to cross tick_triage's generic staleness threshold, which then pre-empts the specific code path under test.

**Incident (#2158 Step 9c gate, 2026-08-18).** A 234-file gate run (1h34m under fleet load ~16) failed exactly 4 tests, all in the api-error-after-marker probe family: `test_api_error_after_marker_returns_stale_redrive`, `test_api_error_after_marker_falls_open_on_missing_transcript`, `test_api_error_probe_kill_switch`, `test_api_error_probe_incident_1689_content_string`. Observed failure reason: `status=running, marker age 28m — bg poll chain likely dead` — the junit shows 22.2 min of cumulative test time before the first tick_triage case, so the 5-min fixture marker was ~27–28 min old. Exact arithmetic match. The file is byte-identical to origin/main; the branch under test touched neither tick_triage nor its tests; the file passes green in isolation (127 passed, 50s).

**Why this evades the Step 9c compare.** The step-1d pristine single-file and paired-prefix oracle runs execute the file within minutes of collection, so the drift cannot reproduce there — the failures classify NEW (fail-closed) against an innocent branch, costing a manual provenance override (#2158 `epm:test-verdict` v1 records the full evidence chain). Every future long gate session (fleet-loaded VMs routinely stretch the ~18-min gate to 90+ min) re-rolls this die for whatever branch happens to be under review.

**Fix candidate.** Replace the module-level `NOW` with a per-test clock anchor: a function-scoped fixture (`now = time.time()` inside `_event`'s caller, or `_event` computing `time.time() - age_s` at call time), OR monkeypatch/freeze the clock tick_triage reads so fixture ages are deterministic regardless of session wall position. Sweep the file for OTHER module-level time captures and any sibling test files using the same `NOW = time.time()` module-level pattern over age-sensitive predicates (repo-wide grep `^NOW = time.time()` in tests/).

**Acceptance.** The 4 named tests pass when executed with a simulated ≥30-min collection→execution drift (e.g. monkeypatched clock), AND still pass in isolation; a regression pin prevents reintroduction of module-level wall-clock capture in this file.

## Provenance

Filed by the `/issue 2158` autonomous session after the Step 9c gate's compare classified the 4 drift failures NEW against an innocent branch (manual provenance override recorded on #2158, `epm:test-verdict` v1, 2026-08-18).
