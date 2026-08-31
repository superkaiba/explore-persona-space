---
title: Step 5a spec-freshness sync imports half an atomic main change, redding the
  Step 10d gate as payload-attributed (GLOB_SCAN_TESTS map row without its tests/
  file)
kind: infra
tags:
- wf-fix
- main-red-adjacent
created_at: '2026-08-31T03:21:24Z'
has_clean_result: false
parent_id: 2649
origin_prompt: 'Surfaced during #2649 Step 10d: pre-push lint gate returned verdict=block
  with an EMPTY lint-new.txt; block_count=1 traced to tg-new-nodes test_glob_scan_map_matches_live_tree,
  caused by the mandated Step 5a sync importing main''s select_step9c_tests.py GLOB_SCAN
  row for tests/test_cron_wrapper_executable_bit.py (added together on main in 5068f40fc07)
  while tests/ sits outside the sync family.'
workflow: v1
---
## Goal

The Step 5a family-atomic spec-freshness sync can import HALF of an atomic
main-side change, producing a branch tree that fails a workflow-invariant test
which then reads as a payload-attributed NEW failure at the Step 10d pre-push
lint gate, blocking a merge whose payload is innocent. Make the sync/gate pair
attribute (or avoid) that class correctly.

`workflow_fix_target: .claude/skills/issue/steps/09-step-5.md`

## The mechanism (measured on #2649, not inferred)

1. Task #2645 landed on `main` as commit `5068f40fc07`, adding BOTH
   `tests/test_cron_wrapper_executable_bit.py` AND its keyed row in the
   `GLOB_SCAN_TESTS` map inside `scripts/select_step9c_tests.py`. The two
   halves are only meaningful together: `test_glob_scan_map_matches_live_tree`
   asserts every map key exists on the live tree.
2. #2649's branch was cut from `origin/main` at `59ebc5a6b27`, before that.
3. The MANDATED Step 5a family-atomic spec-freshness sync (binding before the
   Step 9c gate and again before every Step 10d gate launch) imported
   `scripts/select_step9c_tests.py` from `origin/main` — it is in the sync
   family — as branch commit `6ab8f342180`.
4. `tests/` is NOT in the sync family, so the test file did not come with it.
5. The branch tree now held a map row keyed to a file the branch did not have:

   ```
   AssertionError: map key missing: tests/test_cron_wrapper_executable_bit.py
   tests/test_select_step9c_tests.py:638: AssertionError
   ```

6. The Step 10d pre-push lint gate's TG mapped-invariant leg takes its baseline
   at `--base <branch base>` (`59ebc5a6b27`), i.e. BEFORE the sync commit. So
   the sync-induced failure is not in the baseline and reads as NEW:

   - `tg-baseline-nodes.txt`: `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` (main's red — #2649's actual target)
   - `tg-gated-nodes.txt` / `tg-new-nodes.txt`: `tests/test_select_step9c_tests.py::test_glob_scan_map_matches_live_tree`
   - verdict `block`, `block_count=1`, `unclassifiable_count=0`

7. The block is a TRUE POSITIVE about the branch tree and a FALSE POSITIVE
   about the merge payload: post-merge, `main` carries both halves, so the
   failure cannot exist there. #2649's payload touches ZERO files under
   `tests/`, and its only touch of `scripts/select_step9c_tests.py` IS the
   spec-freshness sync commit (subject carries the prescribed
   `sync workflow-surface specs from` anchor, so Guard 3 already classifies it
   as imported-from-main rather than a deliverable edit — the two surfaces
   disagree about the same commit).

Resolved in-session by merging `origin/main` into the branch (`4519eecc5d4`),
after which both tests pass (2 passed in 28.04s) and the own three-dot diff
collapses to exactly the 12 payload files. That is a workaround a session has
to reason its way to, not a designed path.

## Secondary finding — the block was undiagnosable from the prescribed message

`18-step-10d.md`'s block text says to "fix the named offender", but the lint
side named nothing: `/tmp/issue-2649-lint-new.txt` was EMPTY (0 lines) on a
`block` verdict. The offending node existed only in
`/tmp/issue-2649-tg-new-nodes.txt`. A session following the prescribed remedy
text goes hunting for a nonexistent lint offender in its own payload. The
block message should name the TG node(s) and distinguish sync-induced nodes.

## Candidate remedies (for the fix session to evaluate — deliberately not pre-decided)

1. **Attribute at `origin/main`, not the branch base.** Take the TG leg's
   baseline at the gate's fetched `origin/main` (which the lint legs already
   use for their landing tree) so a sync-imported failure lands in the
   baseline and is subtracted as main-inherited. Most surgical; fixes
   attribution without touching the sync.
2. **Make the sync's family closure honest.** When the sync imports a
   `scripts/` file whose invariant test asserts against paths outside the
   family (`GLOB_SCAN_TESTS` keys being the known case), either carry those
   referenced paths too, or refuse the partial import and surface it.
3. **Require branch/main consistency before the gate.** Have Step 10d merge
   `origin/main` into the branch (the step already sanctions
   `git -C "$WT" merge origin/main` in its conflict-recovery path) before the
   gate, so the gated tree models the true post-merge state. This is what
   unblocked #2649 by hand.
4. **Diagnosability only.** Have the block message enumerate TG new-nodes and
   flag any node whose failure traces to a spec-freshness-sync commit.

(1) and (4) look cheapest and lowest-risk; (3) has the side benefit of making
every Step 10d gate measure the real post-merge tree. Not a decision — the fix
session should adjudicate.

## Acceptance criteria

1. A branch whose spec-freshness sync imports a `GLOB_SCAN_TESTS` row for a
   `tests/` file absent from the branch tree does NOT produce a
   payload-attributed `block` at the Step 10d gate. Demonstrate with a
   reproduction of the #2649 shape (the two commits are named above).
2. A genuinely payload-introduced invariant-test failure STILL blocks — show
   the true-positive path is unchanged (no fail-open).
3. On any `block`, the emitted message names every contributing node,
   including TG nodes, so no session has to read `/tmp/*-tg-new-nodes.txt` to
   learn what blocked it.
4. `uv run python scripts/workflow_lint.py` (no flags) and the mapped tests
   for whatever files change are green.

## Provenance

Surfaced during #2649 Step 10d (main-red repair, 34 sites / 12 files). Gate 1
verdict `block` at 20:15 on 2026-08-30; certified sha
`6ab8f34218066a8af2fbba207bca1901e6157325`. Evidence files under `/tmp/`:
`issue-2649-lint-verdict.txt`, `issue-2649-lint-new.txt` (empty),
`issue-2649-tg-{baseline,gated,new}-nodes.txt`, `issue-2649-lint-gate.log`.
