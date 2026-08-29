---
name: mapfiles-takes-file-not-varargs
description: select_step9c_tests.py --map-files takes ONE newline-delimited FILE, not varargs; the varargs form errors and reads as a false zero-hit sweep
metadata:
  type: reference
---

For the Step 4.6 gate-scope diff-consistency check, re-derive the pin-sweep
hit set with:

```bash
git diff --name-only origin/main...HEAD > /tmp/changed.txt
uv run python scripts/select_step9c_tests.py --map-files /tmp/changed.txt
```

`--map-files` takes **one FILE argument** (newline-delimited repo-relative
paths). Passing the paths as varargs
(`--map-files $(cat changed.txt)`) makes argparse exit 2 with
`unrecognized arguments: ...`. Piping that through
`| awk | grep | sort -u` yields **zero rows with rc=0 from the pipeline**,
which reads as "the round's changed paths hit no pin tests" — the exact
false-absence shape that would wrongly clear, or wrongly flag, a
gate-scope check.

Output is `test<TAB>matched_path`; dedup column 1 for the hit-file list to
compare against the marker's claimed enumeration.

**How to apply:** always inspect the RAW output of the sweep before
reducing it, and confirm rc of the tool itself (not the pipeline). A
zero-hit sweep on a diff that touches `scripts/` is a red flag, never a
result. Same family as [[feedback_grep_pattern_dollar_brace_false_zero]] —
an errored command must never masquerade as an empty set.

Note the separate by-design zero: WORKFLOW_INVARIANT-registered tests are
excluded from `--map-files` output
([[feedback_workflow_invariant_excluded_from_mapfiles]]).
