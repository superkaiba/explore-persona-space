---
name: restore-at-setup-probe-timing
description: Registry-hermeticity plans with restore-at-SETUP guards — a drift probe reading at collection_finish/runtest_teardown is INVARIANT under the fix; acceptance must read at test-body entry (#2214)
metadata:
  type: feedback
---

A conftest guard that restores a leaky global at every test's SETUP
deliberately leaves pollution in place from the moment it happens until the
NEXT test's setup. Any acceptance probe that reads the global at
`pytest_collection_finish` (import-time leg) or `pytest_runtest_teardown`
(test-time leg) therefore reads at exactly the moments the design tolerates
pollution — its output is byte-INVARIANT under a correct implementation, the
"zero drift lines" acceptance criterion is unsatisfiable within the plan's
diff scope, and a kill criterion keyed on it fires on a healthy fix.

**Why:** #2214 plan v2: A5 required "0 IMPORT-TIME + 0 TEST-TIME lines" from
a probe hooked at collection_finish + runtest_teardown, while the §4 fixture
restored only at setup. The IMPORT-TIME line (module-scope registration
during collection) is unrescuable by ANY conftest change — collection_finish
precedes all fixture execution; the TEST-TIME lines persist because teardown
hookimpls from `-p` plugins run before fixture finalization (pluggy LIFO).

**How to apply:** any plan pairing a snapshot/restore autouse fixture with a
probe-based acceptance criterion: (1) map each probe hook's firing point
against the restore's firing point — a probe read in a window the design
tolerates pollution measures the POLLUTERS, not the protection; (2) demand
the acceptance read be the state the test BODY actually sees (a
`pytest_runtest_call` hookwrapper / post-setup read of keys − baseline);
(3) 2-min check: predict the probe's post-fix output by hand — if it equals
the pre-fix output, the criterion is fix-insensitive. Also replay the
in-repo attack set: every direct `REGISTRY[...]` subscript in tests must
re-register in its own body/fixture (function-scoped, since higher-scoped
fixtures run BEFORE function autouse restores and get wiped). Sibling:
[[pytest-collection-guard-plan-review]], [[conftest-hermeticity-guard-review]].
