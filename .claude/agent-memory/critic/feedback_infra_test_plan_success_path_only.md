---
name: Infra test-plans: success-path-only integration tests + parser-only contract tests
description: Two recurring holes in infra-plan test suites — a binding failure-path criterion "verified by the integration test" that only exercises the success path, and a producer/parser key contract tested only on the parser side with hand-crafted fixtures
type: feedback
---

Rule: when an infra plan declares a binding criterion of the form "failure mode X can no longer happen, verified by integration test T", check whether T actually *triggers* X's failure branch. A success-path test (exit 0, artifacts present) plus a string assert on the guard's presence does NOT measure the failure-branch behavior (trap fires, failed-phase published, shutdown invoked). Likewise, when a poll/drain feature adds a producer→parser key contract over SSH stdout (e.g. `EPS_LOG_MTIME=` keys), a test that injects the keys into a hand-crafted fixture tests ONLY the parser — a producer-side quoting/typo bug leaves the feature silently never firing while all tests stay green, and the legacy-tolerance test (absent keys → placeholder) enshrines the degraded state as passing.

**Why:** #607 (GCP startup-script SIGPIPE immunity): plan §3 criterion 4 claimed "EXIT-trap rc path verified by the local integration test" but T6 asserted exit-0 success only (stub `shutdown` never invoked); criterion 5's T7/T8 were parser-only despite the suite's own convention (relaunch-probe test asserts the issued `--command=` string via `runner.calls`). Also: T6's "no 'x'*1000 run in ≤64 bytes read" assert was vacuous by construction (window can't hold the run).

**How to apply:** Map each binding criterion to the test that can FAIL on regression of that exact behavior. Demand (a) a failure-branch integration variant (unguarded write after pipe close → nonzero exit + failed-phase + shutdown stub invoked) and (b) a producer-side command-string assert next to the parser test. Check read-window arithmetic on negative asserts (can the asserted pattern even fit in the bytes read?).
