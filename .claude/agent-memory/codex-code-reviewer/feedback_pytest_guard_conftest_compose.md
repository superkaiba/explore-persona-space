---
name: pytest-guard-conftest-compose
description: Compose recipe for kind:infra diffs adding pytest conftest hooks / collection-time guards — hollow-gate maps to exact-hook-name + manifest/selector registration trace + parent-asserts-child-exit; Step 3.75 fires on test-file module-constant → lazy-helper renames, pin-sweep fragments accepted as the recorded duty
metadata:
  type: feedback
---

Compose recipe for a `kind: infra` round whose diff adds pytest
infrastructure — `tests/conftest.py` hooks, a collection-time guard test,
manifest/selector registration (first used #2217 r1, 2026-08-20). Extends
[[infra-wf-fix-lint-gate-compose]] (its 5 items still apply: N/A-by-type
block, roster grep fresh, floor attestation, ts-threshold line, worktree
pin-sweep adaptation).

1. **Hollow-verification-gate maps THREE ways on pytest-guard diffs** —
   instruct all three concretely: (i) pytest invokes hooks BY EXACT NAME, so
   every added `pytest_*` hook name is verified against the real hook API (a
   typo'd hook silently never runs = hollow guard); (ii) the new guard test
   must be REGISTERED to actually run fleet-side — quote the
   `step9c_workflow_invariant_manifest.txt` line AND the
   `select_step9c_tests.py` tuple line — and its assert must be able to fire
   (no vacuous truth on missing snapshot / empty comparison set); (iii) a
   fresh-subprocess carrier's parent must assert BOTH child exit code and
   success token (swallowed child failure = hollow).
2. **Step 3.75 fires on a TEST-file module-level constant → lazy helper**
   (`-PANEL_IDS =` / `+def _panel_ids(`). For `epm:results` rounds accept
   the rename sweep recorded inside the `(c)` gate-scope pin-sweep grep
   fragments WITH per-hit dispositions as the duty record (no discrete
   `### Symbol-rename grep` section required) — say so in the prompt or
   Codex false-fires `symbol-rename-grep-absent`; still order the
   independent `git -C <wt> grep -n -w` recompute.
3. **Brief-supplied review priorities get a per-priority verdict-line duty**
   — add a `## Gate-step record` section to the output template with one
   line per priority (finding or explicit clean), plus the N/A-line records
   the rubric's steps demand. Cheap, and it stops silent priority drops.
4. **Snapshot-vs-live degeneracy is a Step 3.9 analogue** — a guard whose
   equality arm compares two reads taken at the SAME time (both
   post-collection, or aliased objects) is vacuous by construction; name the
   check explicitly when the diff's whole point is a snapshot-time contract
   (#2217: runtime fu6 leaks must NOT red the collection-time guard).

**Why:** these four are exactly where a generic compose would have narrowed
the rubric (#606 twin-omission class) or false-fired on this diff shape.
**How to apply:** any infra round touching `tests/conftest.py`, adding a
WORKFLOW_INVARIANT guard test, or registering tests in the step9c manifest.
