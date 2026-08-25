---
title: 'workflow-fix: Step 5a spec-sync add/modify-only leaves stale main-deleted
  family twins — TG scan leg collects them and crashes the Step 10d gate'
kind: infra
tags:
- wf-fix
created_at: '2026-08-19T12:22:47Z'
has_clean_result: false
origin_prompt: '/issue 2377 orchestrator: Step 10d gate run 1 crash triage (stale
  test_workflow_lint_no_repo_root_syspath_in_tests.py twin vs synced workflow_lint.py)'
workflow: v1
---
<!-- workflow-fix-candidate v1 -->
## Goal

Close the Step 5a spec-freshness sync's stale-deletion blind spot: the family-atomic sync (`.claude/skills/issue/steps/09-step-5.md` L125-416, mirrored inline at `18-step-10d.md` post-gate re-sync) uses `git checkout origin/main -- <family paths>`, which can only ADD/MODIFY — it never DELETES a worktree file that origin/main has removed or renamed away. A branch therefore keeps a stale main-deleted twin of a synced family file, and any glob-selected scan test collects the stale twin against the freshly-synced module.

## Incident (task #2377, Step 10d lint gate run 1, 2026-08-19 04:08–05:15 PDT)

- main landed issue-2183 at 01:21 PDT: renamed the #2181 check `check_no_repo_root_syspath_in_tests` → `check_no_repo_root_syspath` (widened to scripts/) and removed `tests/test_workflow_lint_no_repo_root_syspath_in_tests.py`.
- The issue-2377 worktree's Step 5a syncs (03:59/04:04) brought the NEW `scripts/workflow_lint.py` but left the stale REMOVED test file in place (add/modify-only property).
- The Step 10d TG mapped leg's scan-test glob (`test_workflow_lint*.py`) collected the stale file → `ImportError: cannot import name 'check_no_repo_root_syspath_in_tests'` → collection error → pytest rc>1 → verdict `crash` → a ~67-min gate run wasted and one crash-case re-run consumed. Evidence: `/tmp/issue-2377-tg-new.txt`, task #2377 events (epm:progress v14).
- The property itself is documented (18-step-10d.md ~L2366: "can only add/modify, never delete") but only as a re-bind-stanza rationale — no arm PREVENTS the stale twin from breaking the TG leg.

## Acceptance criteria

1. The Step 5a family-atomic sync (both copies: 09-step-5.md block + the 18-step-10d.md post-gate inline block, plus the Step 9c step-1a binding reference) also REMOVES worktree files under the synced family paths that are ABSENT on origin/main — bounded strictly to the family glob/path set, never payload files (a file present in the branch's own-diff vs merge-base is payload and is never removed), respecting the existing dirty-family skip (a dirty family syncs nothing and deletes nothing).
2. Alternatively (or additionally, as defense in depth): the gate's TG scan-test selection excludes test files that are absent on origin/main AND absent from the own-diff (stale-twin exclusion at collection time).
3. A regression test pinning the fire branch (stale main-deleted family test twin present → sync removes it / selection excludes it) and the keep branches (payload file kept; dirty-family untouched).
4. The no-lost-row agent-memory duty and the #1972 uncommitted-dirt arm survive unchanged.

## Provenance

workflow_fix_target: .claude/skills/issue/steps/09-step-5.md
Found by the /issue 2377 orchestrator during Step 10d (gate run 1 crash triage). Dedup key: (09-step-5.md, step5a-sync-stale-deletion-tg-scan-crash).
