---
title: 'workflow-fix: export EPS_GIT_SHA in SLURM sbatch env for rsync-lane provenance'
kind: infra
tags:
- wf-fix
- wf-fix-fp:593e6e99748c
created_at: '2026-08-02T22:36:12Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate from #1336 crash-fix round (fellows job 17987
  g2_parity rc=128): SLURM rsync lane exports no code-sha env; export EPS_GIT_SHA
  at the sbatch stage block (slurm.py ~L2215) so git-less scratch provenance resolves
  the real sha'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1336 (emitting agent: experiment-implementer, crash-fix round for fellows job 17987).

## Goal

Export `EPS_GIT_SHA=<materialized branch tip sha>` in the SLURM sbatch env/stage block so rsync-lane (git-less scratch) provenance helpers resolve the real code sha instead of the degraded literal.

## Workflow gap

- **Bug observed:** The fellows/SLURM rsync lane exports no code-sha env, so pod-side provenance on git-less scratch trees can only resolve to a degraded literal ("unknown-no-git"/"unavailable-no-git-checkout") — the #1336 job-17987 results sentinel's reproducibility card will carry no real sha, and #1902 hit the same gap.
- **Why it is a workflow gap:** The stage renderer (`materialize_branch_src` / the sbatch env block in `backends/slurm.py` — workflow surface per the explicit backends/*.py inclusion) KNOWS the materialized branch tip sha at render time but never exports it, while multiple in-repo consumers already read `EPS_GIT_SHA` as the first resolution rung.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n "EPS_GIT_SHA" src/explore_persona_space/backends/slurm.py` → 0 hits in the named target (ABSENCE claim — the 0-hit result IS the evidence); anchor presence `grep -n "EPS_ATTEMPT_ID\|EPS_ISSUE" src/explore_persona_space/backends/slurm.py` → export sites at L2215-2216 (`export EPS_ISSUE=...` / `export EPS_ATTEMPT_ID="slurm-${SLURM_JOB_ID}"`); consumer presence `grep -rln "EPS_GIT_SHA" scripts/ src/` → 6 source files (issue1902_corpus.py, issue1902_run.py, issue825_mlp_followup_dispatch.sh, issue825_onpolicy_dispatch.sh, issue1895_subspaces.py, + experiments/issue_1336/common.py resolve_code_sha on the issue-1336-fullcorpora branch) (2026-08-02)

## Proposed change (candidate diff sketch — refine in planning)

```
# backends/slurm.py, stage_blocks near the EPS_ISSUE/EPS_ATTEMPT_ID exports (~L2215):
+ sha = <tip sha resolved by _resolve_rsync_source/materialize_branch_src for the dispatched branch>
+ stage_blocks.append(f'export EPS_GIT_SHA="${{EPS_GIT_SHA:-{sha}}}"')
# consumers (issue1902_run/_corpus, issue1336 common.resolve_code_sha, issue825 dispatchers,
# issue1895_subspaces) already read EPS_GIT_SHA first — no consumer change needed.
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/slurm.py`
- Grep the workflow surface for the pattern before editing (`grep -rln 'EPS_GIT_SHA' .claude/ CLAUDE.md scripts/ src/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; SLURM render tests (`tests/test_slurm_*.py`) stay green; add a render pin that the sbatch env block carries `EPS_GIT_SHA`.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/slurm.py
- fingerprint: 593e6e99748c

<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/backends/slurm.py
bug_observed: The fellows/SLURM rsync lane exports no code-sha env, so pod-side provenance on git-less scratch trees can only resolve to a degraded literal ("unknown-no-git"/"unavailable-no-git-checkout") — the #1336 job-17987 results sentinel's reproducibility card will carry no real sha, and #1902 hit the same gap.
why_workflow_gap: The stage renderer (materialize_branch_src / the sbatch env block in backends/slurm.py — workflow surface per the explicit backends/*.py inclusion) KNOWS the materialized branch tip sha at render time but never exports it, while in-repo consumers already read EPS_GIT_SHA as the first resolution rung.
proposed_change: Export EPS_GIT_SHA=<materialized branch tip sha> in the SLURM sbatch env/stage block (next to the existing EPS_ISSUE/EPS_ATTEMPT_ID exports at slurm.py ~l.2215) so rsync-lane provenance helpers resolve the real sha.
diff_sketch: |
  # backends/slurm.py, stage_blocks near the EPS_ISSUE/EPS_ATTEMPT_ID exports (~l.2215):
  + sha = <tip sha resolved by _resolve_rsync_source/materialize_branch_src for the dispatched branch>
  + stage_blocks.append(f'export EPS_GIT_SHA="${{EPS_GIT_SHA:-{sha}}}"')
confidence: high
related_task: #1336
<!-- /workflow-fix-candidate -->
