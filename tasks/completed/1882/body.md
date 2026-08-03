---
title: 'workflow-fix: TG file-grain gate arm — tree-prefix normalization + warnings-line
  exclusion'
kind: infra
tags:
- wf-fix
- wf-fix-fp:27381c19626d
created_at: '2026-07-30T13:19:26Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate raised on #1689 (2026-07-30): Step 10d TG file-grain
  compare false-blocks innocent merges — absolute tree-prefix skew between worktree
  gated leg and root baseline leg + pytest warnings-summary lines from passing branch-new
  tests; node-grain NEW was empty on all 3 blocked gate runs'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1689 (emitting agent: issue-orchestrator).

## Goal

In both Step 10d TG file-grain hit pipelines, normalize each leg's absolute tree root to a common token and exclude pytest warnings-summary lines from the file-grain grep; node-grain remains the failure arbiter.

## Workflow gap

- **Bug observed:** The Step 10d TG file-grain compare subtracts full lines including absolute tree prefixes (gated leg in the worktree, baseline at the repo root), so payload-path-bearing warning lines never cancel and passing branch-new tests' warnings have no baseline twin - innocent merges block.
- **Why it is a workflow gap:** The gate's own documented semantics say pre-existing trunk red never blocks and node-grain is the failure arbiter, but the file-grain `comm -23` (grep -F -f payload-paths | sed line-number-blank) is structurally incapable of cancelling absolute-path-bearing lines across the two trees, and a pytest warnings-summary line from a PASSING test is not a failure signal at all. On #1689 (2026-07-30) this cost 3 full gate cycles (~50 min): gate run 3 had node-grain NEW = 0 and lint legs clean, yet blocked on 5 file-grain lines — 4 were prefix-twins of the baseline-red-on-main `test_issue825_mlp_batched_parity` warning lines, 1 was a warnings-summary line from the branch's own passing `test_issue1689_fit_ladder.py` (branch-new test absent at root). The orchestrator had to re-derive the verdict manually from the persisted leg outputs.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'tg-gated-hits\|tg-baseline-hits' .claude/skills/issue/SKILL.md` → 4 hits in 2 blocks (the shared form (i)/(ii) gate block ~L11337 and the form (iii) surgical block ~L13018) (2026-07-30)

## Proposed change (candidate diff sketch — refine in planning)

In BOTH TG hit pipelines (shared gate block + surgical block), amend the per-leg hit extraction:

```
  for leg in baseline gated; do
    grep -F -f /tmp/issue-<N>-tg-files.txt "/tmp/issue-<N>-tg-$leg.txt" \
      | grep -vE '^E +assert ' \
+     | grep -vE ':: [A-Za-z]+Warning: ' \
      | sed -E 's/at line [0-9]+/at line N/g; s/:[0-9]+:/::/g; s/:[0-9]+([^0-9]|$)/:N\1/g' \
+     | sed -e "s|$WT|<TREE>|g" -e "s|$REPO_ROOT|<TREE>|g" \
      | sort -u \
      > "/tmp/issue-<N>-tg-$leg-hits.txt" || true
  done
```

(Surgical block: both legs run at the root, so only the warnings-exclusion line is
load-bearing there; add the prefix normalization anyway for uniformity.) Rationale:
warnings are not failures (node-grain + the scan tests' dedicated per-file evidence
lines remain the block surface); the tree-prefix token makes the same line from the
two trees cancel under `comm -23`.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rn 'tg-gated-hits' .claude/ CLAUDE.md scripts/`) and update every hit;
  list them in the plan. Check whether any pin test asserts the current pipeline
  shape (`grep -rl 'tg-baseline-hits' tests/`) and update it in the same diff.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).
- The node-grain arm and the scan-test per-file evidence lines stay byte-unchanged —
  this fix ONLY removes the two structural false-positive classes from the
  file-grain arm.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 27381c19626d

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md
bug_observed: The Step 10d TG file-grain compare subtracts full lines including absolute tree prefixes (gated leg in the worktree, baseline at the repo root), so payload-path-bearing warning lines never cancel and passing branch-new tests' warnings have no baseline twin - innocent merges block
why_workflow_gap: The gate's documented semantics say pre-existing trunk red never blocks and node-grain is the failure arbiter, but the file-grain comm -23 is structurally incapable of cancelling absolute-path-bearing lines across the two trees; #1689 burned 3 gate cycles on a node-grain-clean payload
proposed_change: In both Step 10d TG file-grain hit pipelines, normalize each leg's absolute tree root to a common token and exclude pytest warnings-summary lines from the file-grain grep; node-grain remains the failure arbiter
diff_sketch: |
  + | grep -vE ':: [A-Za-z]+Warning: ' \
  + | sed -e "s|$WT|<TREE>|g" -e "s|$REPO_ROOT|<TREE>|g" \
  (inserted into the per-leg hit pipeline in the shared gate block ~L11330 and the surgical block ~L13010)
confidence: high
related_task: #1689
<!-- /workflow-fix-candidate -->
