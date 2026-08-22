---
title: 'workflow-fix: SKILL.md plan-review-floor trigger misattributed to is_workflow_fix_session'
kind: infra
tags:
- wf-fix
created_at: '2026-08-14T04:25:58Z'
has_clean_result: false
origin_prompt: 'Found by the #2282 orchestrator while checking whether the Step 2
  minimum plan-review floor bound to itself: is_workflow_fix_session(2282) returns
  False although #2282 is a kind:infra task whose title starts with ''workflow-fix:'',
  because the function tests only the workflow_fix_target: provenance line while SKILL.md''s
  floor paragraph attributes a tag-OR-title-prefix predicate to it.'
workflow: v1
---
## Provenance

workflow_fix_target: .claude/skills/issue/SKILL.md
fingerprint: skillmd-1559-floor-trigger-attributed-to-is-workflow-fix-session
Filed by: the #2282 orchestrator session (`kind: infra`, `parent_id: 1336`), which
hit this while checking whether the minimum plan-review floor bound to itself.

## Goal

Fix one imprecise predicate attribution in `.claude/skills/issue/SKILL.md` at the
**Minimum plan-review floor** paragraph (line ~1559). It defines the floor's
triggering class as:

> `kind: infra` workflow-fix tasks (`wf-fix` tag OR title prefix in
> `WF_FIX_TITLE_PREFIXES` — `workflow-fix:` / `daily-fix:` —
> `task_workflow.is_workflow_fix_session`)

The trailing attribution is wrong. `task_workflow.is_workflow_fix_session` tests
**neither** the tag nor the title prefix — its whole body is:

```python
def is_workflow_fix_session(task_id: int) -> bool:
    try:
        body = (find_task_path(task_id) / "body.md").read_text()
    except (FileNotFoundError, OSError):
        return False
    return "workflow_fix_target:" in body
```

So the paragraph states a tag-OR-title trigger and then names a function that
implements a provenance-line trigger.

## Why this matters (the concrete failure mode)

A session that mechanizes the floor check by calling the named function SKIPS the
floor on exactly the class the paragraph is written to cover: a `kind: infra`,
`workflow-fix:`-titled task that carries no `workflow_fix_target:` Provenance line
— i.e. one filed as an ordinary task rather than auto-filed by a workflow-fix
session.

That is not hypothetical. Task **#2282** (`workflow-fix: land the shared fit-core
memory chunk cap on main + decide PR #1717 disposition`, `kind: infra`) is exactly
that shape: `is_workflow_fix_session(2282)` returns `False` while the title prefix
matches and the floor's prose plainly intends to bind. #2282 ran the full floor
anyway because its orchestrator read the prose rather than trusting the function,
so nothing was lost there — but the next session may not, and the floor is the only
thing standing between a shared-surface workflow edit and zero plan review.

## What is NOT wrong (scope fence — do not "fix" these)

- **`is_workflow_fix_session` itself needs no change.** It is correct for its
  actual job, the recursion guard, which SKILL.md:1029 defines precisely as the
  `workflow_fix_target:` Provenance line being present. Prose and implementation
  agree there. Changing the function to also match the title prefix would silently
  widen the RECURSION GUARD, disabling workflow-fix auto-filing for every
  `workflow-fix:`-titled task — a behavior regression, not a fix.
- **`WF_FIX_TITLE_PREFIXES` is not misplaced.** `task_workflow.py:1065-1066` states
  its role explicitly: the tag plus the provenance line are the real signals, "so
  the title prefix is only the cheap REGISTRY pre-filter", which is what makes the
  `(target_file, fingerprint)` dedup cross-channel (#1180). It is consumed by the
  dedup lookup — title pre-filter at line ~1131, then the
  `workflow_fix_target: <file>` body match at line ~1139.

## Proposed fix (prose-only, one paragraph)

Correct the attribution at SKILL.md:1559 so the floor's trigger and its
implementation are stated separately and accurately. Either:

**(a) prose-only (smallest):** keep the tag-OR-title trigger as the floor's
definition and drop the `is_workflow_fix_session` attribution, replacing it with
how the floor is actually to be evaluated (read the tag / title prefix directly, or
cite the dedup helper if that is the intended check). Add a half-sentence noting
that `is_workflow_fix_session` is the RECURSION-GUARD predicate and deliberately
keys on the Provenance line only, so the two must not be conflated.

**(b) prose + a small named helper:** if a mechanized floor check is wanted, add a
distinct predicate (e.g. `is_workflow_fix_task(task_id)`) that tests the `wf-fix`
tag OR the title prefix, leave `is_workflow_fix_session` untouched for the guard,
and point the floor paragraph at the new one. Pin both with tests so the two
predicates cannot drift back together.

(a) is sufficient and is the recommended default; (b) only if a caller genuinely
needs to evaluate the floor programmatically.

## Acceptance criteria

1. SKILL.md's floor paragraph no longer attributes a tag-or-title predicate to
   `is_workflow_fix_session`.
2. The distinction between the FLOOR trigger and the RECURSION-GUARD trigger is
   stated once, explicitly, so a future reader cannot re-conflate them.
3. `is_workflow_fix_session`'s behavior is unchanged (assert via test, since
   widening it is the tempting wrong fix — see the scope fence above).
4. If route (b) is taken, the new predicate is test-pinned on all four cases: tag
   only, title only, provenance line only, none.
5. `workflow_lint.py` passes; any region-anchored surface pin covering that
   paragraph is updated in the same commit.

Estimated GPU-hours (total): 0
