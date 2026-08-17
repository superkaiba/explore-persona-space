---
title: 'workflow-fix: Step 10d pre-push gate mapped-test legs compare failure sets
  over different collections (false block on pre-existing main-red)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-17T15:07:30Z'
has_clean_result: false
origin_prompt: 'surfaced by #2155 Step 10d gate round 1 false block, 2026-08-17'
workflow: v1
---
# workflow-fix: Step 10d pre-push gate mapped-test legs compare failure sets over DIFFERENT collections — pre-existing main-red misclassified as payload-NEW (false block)

## Provenance

workflow_fix_target: .claude/skills/issue/SKILL.md
Surfaced by task #2155's Step 10d pre-push lint gate (2026-08-17): gate round 1 returned verdict=block on tg-new node tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset, which was PRE-EXISTING main-red (C901 introduced by main commit c0d76f99c9; the branch's copy of the offender byte-identical to base; fix later landed via #2345).

## Bug

The gate's mapped invariant-test legs run DIFFERENT node collections: the baseline leg (step9c_baseline.py mapped-baseline, base-pinned scratch) collected/ran 2,411 nodes (0 failed) while the gated leg (branch-tip worktree) ran 2,297 nodes (1 failed). The verdict then computes NEW = gated_failed_nodes − baseline_failed_nodes via comm -23 — a subtraction that is only sound over the INTERSECTION of the two collections. Any node present in the gated set but absent from the baseline set can never be subtracted, so a main-side pre-existing red among those nodes false-blocks the merge (fail-closed direction, but a wasted gate round ~55 min + a spurious re-execution loop per occurrence). The SKILL.md § "Known residuals" bullet (c) documents test-VERSION drift for same-node comparisons but not this SET-mismatch class.

## Fix shape (investigate; pick the sound one)

Either (a) make both legs run the SAME node list (derive the mapped set once, pass the explicit node/file list to both the scratch baseline and the gated run), or (b) restrict the NEW computation to nodes COLLECTED in both legs (emit collected-node lists per leg; comm the failures only within the intersection, and surface gated-only-collected failing nodes as a distinct "unclassifiable — pristine-oracle needed" arm rather than NEW). Also record why the collections diverged (selector diff-base differences between scratch and worktree are the likely mechanism).

## Acceptance criteria

1. A reproduction (or unit test over the verdict computation) where a node fails identically on main and branch but is collected only in the gated leg → verdict must NOT be block.
2. A genuinely payload-caused new failure still blocks.
3. The gate's verdict block in .claude/skills/issue/SKILL.md (and its steps/ companion once #2155 lands) + any step9c_baseline.py changes stay in lockstep with the pin tests.

Estimated GPU-hours (total): 0
