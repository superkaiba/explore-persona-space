---
name: prefix-fix-evidence-mechanics
description: Cheap mechanics for revision-round evidence — pre-fix demo via git show + importlib (register sys.modules BEFORE exec_module), select_step9c_tests --map-files takes ONE list-file, and a pytest tail's quoted assert line can be the WRONG assert (read the E lines / line number)
metadata:
  type: feedback
---

Three mechanics from #1336 round-4 fix (all cost a retry when missed):

1. **Pre-fix regression demonstration without branch switching:** `git show
   <pinned-sha>:scripts/<file>.py > /tmp/old.py`, then
   `spec_from_file_location` + **`sys.modules["old"] = module` BEFORE
   `spec.loader.exec_module`** (a `@dataclass` in the old module crashes on
   `sys.modules.get(cls.__module__)` being None otherwise). Reuse the new
   test's own fixture helpers by adding `tests/` to sys.path — one snippet
   proves "fails pre-fix, passes post-fix" for the marker.

2. **`select_step9c_tests.py --map-files` takes ONE FILE containing the
   diff list** (one path per line), not N path args; add `--no-fetch` in a
   worktree. Take the col-1 dedup union verbatim.

3. **A pytest failure's context snippet can quote the WRONG assert.** A
   grep of the tail pulled `assert targets, "globs matched nothing"` while
   the real failure (the `E` lines / `file:LINE`) was the later
   `assert not violations` — 2 debugging rounds chased an impossible
   empty-glob theory. Read the `E  AssertionError:` lines and the reported
   line number first; then attribute the named file via
   `git log -1 -- <file>` before treating a pin-sweep FAIL as your payload
   (here: a concurrent round-5 commit's file — reported, not fixed).

**How to apply:** on any revision round: use (1) for blocker-fix evidence,
(2) for the checklist-2b hit list, (3) before diagnosing any pin-sweep red.
See [[measured-vs-applied-gap-smoke-batteries]] for the sibling consumer-
probe discipline (the `read_offpolicy_rows` staged-tree open this round).
