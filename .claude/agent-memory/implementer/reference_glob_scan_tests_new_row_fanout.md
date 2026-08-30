---
name: reference-glob-scan-tests-new-row-fanout
description: Adding a GLOB_SCAN_TESTS row to the Step 9c selector has a 4-point fan-out (verbatim glob literal in the scanner, invariant disjointness, FILE_ANCHORED audit, no selector filename in test prose)
metadata:
  type: reference
---

Adding a NEW `GLOB_SCAN_TESTS` row to the Step 9c selector (#2645, precedent
#2386) must satisfy, or the live-tree drift pin
`tests/test_select_step9c_tests.py::test_glob_scan_map_matches_live_tree` FAILs:

1. Each scan glob appears **VERBATIM in the scanning test's own source**
   (e.g. `_CRON_WRAPPER_GLOB = "scripts/cron_*.sh"`), and the row's globs
   must match >=1 real file on the live tree (aggregated per row).
2. The key is **disjoint from `WORKFLOW_INVARIANT`** (no double-listing;
   invariant members are also excluded from the `--map-files` legs by design).
3. **FILE_ANCHORED_SCAN_TESTS audit at addition time**
   (`scripts/step9c_baseline.py`, fail-closed): only allowlist there when the
   scan chain is SOURCE-VERIFIED `Path(__file__)`-anchored with no
   repo_root()/task_workflow/live-tree read; default (not listed) = refused
   from the scratch pristine oracle — the posture #2386's and #2645's scan
   tests both took (a git-INDEX-reading scan test should stay unlisted).
4. The scanning test's prose must NOT spell the selector/baseline filenames
   (basename-ref arm reads docstrings/comments and would mint a false edge
   on every selector diff — see [[reference-selector-basename-ref-from-test-prose]]).

No member-count pins exist (fixtures size proportionally per #1632). Verify the
row took effect with `--map-files <file listing one matching path>` (the flag
takes a FILE of newline-delimited paths, not positional paths).

**How to apply:** whenever a diff adds/renames a `GLOB_SCAN_TESTS` entry or a
scan test covering a new file class with no stem/import-arm reachability
(`.sh`, `.txt`, data files).
