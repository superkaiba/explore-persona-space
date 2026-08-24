---
name: selector-meta-test-cli-ground-truth
description: Reviewing Step 9c selector mapping-gap plans (#1589/#1688/#2412/#2537 class) — every-gate claims, CLI-as-ground-truth circularity, non-vacuity trace
metadata:
  type: feedback
---

Dispositions for plans extending `scripts/select_step9c_tests.py` mapping
coverage (the constructed-path consumer class; #2537 v2 APPROVEd on these):

1. **"Runs at every gate" is a WORKFLOW_INVARIANT-membership claim — verify
   at the code, not the docs.** The unconditional inclusion is the
   `for t in WORKFLOW_INVARIANT: _add(t, "invariant")` loop in
   `select_tests_with_reasons` (~line 2101), independent of the diff (and
   the diff-failure fallback is invariant-only). A stem-mapped /
   GLOB_SCAN-keyed test file does NOT run on rounds touching other files —
   the exact v1 error #2537's fact-checker caught. Also verify the registry
   feeds BOTH surfaces: the diff-based selection (seed loop ~line 2005,
   reason `transitive-consumer:<file>`) AND `--map-files` (~line 2299) —
   a fix engaging only `--map-files` would leave the executed gate union
   open.
2. **CLI-as-ground-truth predicate legs are circular only in the SAFE
   direction, IFF a non-vacuity trace rides in the same invariant file.** A
   meta-test whose coverage predicate calls the real `--map-files` CLI
   in-process: a selector regression REMOVING coverage makes discovered
   pairs lose their CLI-leg coverage ⇒ predicate FAILs loud at the next
   gate (detection, not blindness); over-selection passes silently but is
   the safe direction (extra tests run). The residual hole is a silently
   BROKEN DISCOVERY (empty scan ⇒ vacuous predicate) — demand an
   incident-trace test in the SAME every-gate file asserting the real
   incident pairs ARE discovered (the #1287 rule) plus a
   monkeypatch-the-registry-away negative control asserting the predicate
   then FAILs.
3. **Named tradeoff to surface as a Concern, not a blocker:** an every-gate
   incident-trace pinned to live test files couples the fleet gate to those
   files' internal loader idiom — an innocent refactor (moving a local
   `_load` into conftest) fleet-blocks until the trace fixture updates.
   Loud + clear remedy ⇒ acceptable designed loudness; do not demand a
   synthetic fixture instead (that would break the real-incident fidelity
   demand, [[detector-fixture-fidelity-vs-actual-incident]]).

**Why:** the #2537 review's three highest-value checks were exactly these;
the class recurs (#1589 → #1688 → #2412 → #2537).
**How to apply:** any plan touching selector arms, WORKFLOW_INVARIANT /
manifest registration, or adding a gate-time discovery/meta-test.
