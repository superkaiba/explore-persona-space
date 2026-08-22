---
title: Briefs pointing a worktree-based reviewer at task state must pin the absolute
  main-checkout path (worktree plans/plan.md resolves to a stale version)
kind: infra
tags: []
created_at: '2026-08-20T07:58:38Z'
has_clean_result: false
parent_id: 823
origin_prompt: 'Surfaced during /issue 823 P-Gen v13 amendment review dispatch: the
  hand-written code-reviewer brief named a relative tasks/.../plans/plan.md while
  directing the reviewer to a worktree, where that symlink resolves to v10 (pre-amendment)
  instead of v13. The codex-code-reviewer composer independently caught and handled
  the same hazard, so the discipline exists on one twin''s surface but not the other''s.'
workflow: v1
---
## Goal

Close a brief-composition channel that hands a worktree-based reviewer a RELATIVE
`tasks/<status>/<N>/plans/plan.md` path, which inside a worktree resolves to a STALE
plan version — because task state is committed to `main` while the worktree sits on
`issue-<N>`. A reviewer that reads the stale version reviews the diff against the plan
the diff SUPERSEDES, producing false plan-deviation blockers.

## The incident (#823, 2026-08-20, caught pre-verdict — cost 0 rounds by luck)

Task #823's P-Gen v13 amendment round. The orchestrator hand-wrote a `code-reviewer`
brief naming the plan as `tasks/followups_running/823/plans/plan.md` and, separately,
told the reviewer to read the diff in the worktree
`.claude/worktrees/issue-823-ladder`. Verified resolution of that same relative path:

- worktree `.claude/worktrees/issue-823-ladder/tasks/followups_running/823/plans/plan.md`
  -> `v10.md` (title line "Plan v10")
- main checkout `/home/thomasjiralerspong/explore-persona-space/tasks/followups_running/823/plans/plan.md`
  -> `v13.md` (title line "Plan v13")

v10 is exactly the version the round's amendment supersedes. It registers the OLD
generation caps (1024/2048) and contains NO generation-config fingerprint, NO
metadata-free hash-basis pin, and NO P0 prompt-integrity gate. A v10-based review of
the v13 diff reads the 4096/8192 caps as an unauthorized deviation and the
fingerprint/P0 work as unplanned scope creep — four false blockers, all of them
plan-lens, all of them requiring a fix round to refute.

**The asymmetry that makes this mechanizable:** the `codex-code-reviewer` composer
caught this at compose time on its own and handled it — it referenced a byte-identical
plan copy under `/tmp` as primary, the main-checkout ABSOLUTE path as fallback, BARRED
all worktree `tasks/` reads, and supplied sed line-window anchors. The Claude-side
brief, hand-written by the orchestrator with no equivalent discipline, did not. Same
round, same plan, same worktree; one twin was structurally protected and the other was
not. The fix is to propagate the composer's existing discipline into the Claude-side
brief contract so it does not depend on the orchestrator remembering.

Recovery in the incident was a `SendMessage` correction to the live reviewer naming
both resolutions and listing which findings needed re-checking. That worked only
because the Codex composer surfaced the hazard while the Claude reviewer was still
running — pure timing, not a control.

## Why the existing rule does not cover it

CLAUDE.md already carries "Never form `tasks/...` paths relative to cwd or `__file__` —
from a worktree that path is stale", enforced by
`tests/test_no_direct_task_path_construction.py`. That rule and its test are scoped to
CODE constructing task paths. This channel is PROSE: a brief handing a relative task
path to a subagent whose working directory is a worktree. The lint cannot see it and
the rule's wording does not reach it.

## Proposed change

1. Brief-contract line wherever reviewer/implementer briefs are specified (the
   `/issue` skill's review-dispatch step and/or `.claude/agents/code-reviewer.md`'s
   brief contract): when a brief points a worktree-based agent at task state — plan,
   body, events, markers — it MUST pass the ABSOLUTE main-checkout path (or a
   verified-identical copy outside both trees) and explicitly bar reading `tasks/`
   from inside the worktree. Resolve the intended version at compose time
   (`readlink`) and state the version number in the brief so a mismatch is visible.
2. Consider a mechanical check: flag a `.claude/**` brief-composing surface that emits
   a relative `tasks/<status>/<N>/` path in the same span as a worktree path. Scope
   this carefully — prose greps over brief templates are false-positive-prone, so a
   WARN-only check plus the contract line may be the right split. The contract line is
   the binding artifact; the lint is convenience.
3. Do NOT change `plan_patch.py`, `task.py`, the resolver, or any worktree mechanics.
   The worktree's `tasks/` tree being frozen at its branch point is CORRECT behavior —
   the defect is briefs that read it as if it were live.

## Acceptance criteria

- A brief-contract line exists on the Claude-side reviewer/implementer brief surface
  requiring absolute main-checkout task paths + an explicit worktree-`tasks/` read bar,
  with compose-time version resolution.
- The #823 shape is covered by a test or a documented pin: a brief naming a relative
  `tasks/.../plan.md` alongside a worktree working directory is a defect.
- No change to worktree `tasks/` semantics; no new `tasks/` path construction in code.
- `uv run python scripts/workflow_lint.py` clean; the relevant mapped tests pass.

## Sibling observation (same class, NOT the subject of this task)

The same #823 round had a second brief-composition omission: the implementer brief
stated the prohibition ("do not end your turn with work staged-but-uncommitted") but
not the WAIT MECHANISM that the #2041 (g) duty prescribes (a bounded in-turn Monitor
until-loop). The implementer then invented a background-watcher shape and orphaned its
own landing — ~62 min of work left uncommitted with a lint gate running unobserved;
the orchestrator recovered it. Recorded here as evidence that hand-written briefs
recurrently omit disciplines that already exist elsewhere on the surface. Not filed
separately: that duty exists and was simply not applied. If a reviewer of this task
judges the two worth one combined brief-contract hardening, that is a reasonable merge.
