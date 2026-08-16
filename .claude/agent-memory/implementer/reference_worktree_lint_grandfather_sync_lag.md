---
name: worktree-lint-grandfather-sync-lag
description: Worktree no-flags lint FAILs a size ratchet on a SYNCED doc when the branch's workflow_lint.py grandfather cap lags origin/main — diff the cap, restore byte-exactly
metadata:
  type: reference
---

Spec-freshness syncs (`issue-<N>: sync workflow-surface specs from origin/main`)
update `.claude/**` docs but do NOT re-sync `scripts/workflow_lint.py` (the
branch may carry deliberate lint edits). When main later raises a
`SKILL_DOC_SIZE_GRANDFATHER` cap (doc regrew + cap ratcheted together on main)
and the branch then syncs the REGROWN doc, the worktree's own lint carries the
old cap → deterministic false-red: no-flags lint `FAIL (1 error(s))` on the
synced doc, plus the Step 9c live pins
`tests/test_workflow_lint_skill_doc_size.py::test_live_tree_passes_no_fails` /
`::test_live_grandfather_caps_have_sane_headroom`. (#2321 r3: SKILL.md synced
to 982,587 B, branch cap stale at 980_400, main's at 983_400.)

**Diagnose:** failure line names a file your round never touched →
`git diff origin/main HEAD -- scripts/workflow_lint.py` and look for a
grandfather-dict hunk where the branch has the OLD cap. Confirm the doc is
byte-identical to origin/main (`git diff --quiet origin/main HEAD -- <doc>`).

**Fix:** restore origin/main's grandfather block byte-exactly (comment ladder
+ cap) as its own explicit-path commit — zero-risk, the merge produces the
same bytes (branch never touched that block ⇒ 3-way merge takes main's side
anyway); the fix just makes the pre-merge worktree gate green.

**How to apply:** any long-lived issue branch whose lint run FAILs a size
ratchet on a `.claude/skills/**` doc; also explains a sibling round REPORTING
lint PASS on the same tree — it likely ran MAIN's lint (new caps) over the
worktree files.

Related: [[union-run-under-gate-contention]] — separately,
`tests/test_workflow_lint_upload_or_true.py::test_no_flags_default_run_bundles_check`
embeds a FULL no-flags `main([])` run (~6-8 min under VM contention): never
batch that file with others under a foreground timeout; run it detached with
a ≥1500 s bound.
