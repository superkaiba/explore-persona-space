---
name: family-sync-membership-review
description: Reviewing a Step 5a FAMILY_OF/SPECS widen — enumerate importers across ALL families (cross-family imports half-sync independently), and comment-bearing FAMILY_OF lines escape pin test (10)'s propagation filter (#2352)
metadata:
  type: feedback
---

When a plan adds a shared helper module to the Step 5a family-atomic sync
(`.claude/skills/issue/steps/09-step-5.md` FAMILY_OF/SPECS + the 18-step-10d
mirror), two checks are conclusion-changing:

1. **Enumerate importers across ALL synced families, not just the incident's
   family.** Families sync INDEPENDENTLY (per-family dirty-skip), so a helper
   assigned to family A does not sync when A is dirty — but a family-B test
   glob member importing it still syncs fresh → the same
   collection-walling ModuleNotFoundError through family B. Concrete #2352
   instance: `tests/test_workflow_lint_no_repo_root_worktree_revert.py:58`
   (lint glob `:(glob)tests/test_workflow_lint*.py`) imports
   `tests.issue_skill_source` (assigned workflow family). Grep recipe:
   `grep -rln "from tests.issue_skill_source import\|import tests.issue_skill_source" tests/`
   then intersect with every SPECS test glob/token. A forward guard test must
   be FAMILY-AWARE (importer's own family covers the helper), not mere
   SPECS-token presence.

2. **Test (10) (`test_step10d_family_atomicity_matches_step5a`) only
   propagates FAMILY_OF lines whose STRIPPED form `endswith('="workflow"'
   etc.)`** — a 5a entry with a trailing `# comment` is silently skipped, so
   "test (10) mechanically propagates to the 10d copy" is FALSE for
   comment-bearing lines (pre-existing skips: `.claude/skills`,
   `test_workflow_yaml.py`, `test_autonomous_session_watch.py`). Demand the
   rationale comment on its own line above a bare entry, or an explicit
   10d-span assert.

**Why:** #2352 plan v1 claimed forward-class closure ("skew becomes
unshippable") while a live cross-family violator sat on main, and rested its
10d pin on a propagation filter its own comment style defeated.

**How to apply:** any plan touching FAMILY_OF / SPECS / the family-sync pin
suite (`tests/test_issue_skill_lint_family_sync.py`). Related:
[[infra-plan-review-checklist]].
