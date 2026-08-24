---
title: Step 5a spec-freshness sync stages a CLAUDE.md whose workflow.yaml gate-key
  references cannot resolve, because CLAUDE.md is a FAMILY_OF singleton while .claude/workflow.yaml
  is in FAMILY_workflow
kind: infra
tags:
- wf-fix
created_at: '2026-08-24T21:31:27Z'
has_clean_result: false
origin_prompt: 'Surfaced during task #2538 Step 5a spec-freshness sync (2026-08-24):
  the sync staged 17 paths and the commit FAILED with workflow-yaml-lint exit 1 (CLAUDE.md:
  unresolved reference ... not in workflow.yaml / FAIL (1 error(s))). CLAUDE.md is
  a FAMILY_OF singleton so it synced to main''s newer copy referencing gates.clarify_experiment_ask,
  while .claude/workflow.yaml is a FAMILY_workflow member correctly skipped as dirty
  with #2538''s payload. The #2538 session chose RESET over commit per the spec''s
  own instruction.'
workflow: v1
---
# Step 5a spec-freshness sync: `CLAUDE.md` is a `FAMILY_OF` singleton but a cross-family reader of `.claude/workflow.yaml`, so syncing it while `FAMILY_workflow` is dirty manufactures an unresolvable gate-key reference and the commit hook FAILs

## Goal

Make the Step 5a family-atomic spec-freshness sync unable to stage a `CLAUDE.md` whose gate-key references cannot resolve against the `.claude/workflow.yaml` the same sync deliberately skipped. Either (a) `CLAUDE.md` joins `FAMILY_workflow` so the two move together or neither moves, or (b) the `workflow_lint.py` gate-key reference check gains the same present-at-main tolerance the SKILL-reference check already has for this exact stale-worktree shape.

## Observed failure (task #2538, Step 5a, 2026-08-24)

The Step 5a sync in the `issue-2538` worktree staged 17 paths and the commit **FAILED** with `workflow-yaml-lint` exit 1:

```
workflow_lint: CLAUDE.md: unresolved reference ... not in workflow.yaml
workflow_lint: FAIL (1 error(s))
```

Root cause is a **cross-family skew**, not a stale spec:

- `CLAUDE.md` is a **singleton** in the Step 5a `FAMILY_OF` map, so it synced to `main`'s newer copy. That copy references `(see workflow.yaml § gates.clarify_experiment_ask)`.
- `.claude/workflow.yaml` is a member of **`FAMILY_workflow`**, which was **dirty** with #2538's own payload, so the sync **correctly skipped the whole family** (the family-atomic fail-safe: any dirty member widens the skip, never a clobber).
- Result: `main`'s `CLAUDE.md` landed in a tree holding the worktree's older `workflow.yaml`, which does not yet define `gates.clarify_experiment_ask`. The reference is unresolvable **by construction of the sync**, not because either file is wrong.

The #2538 session chose **RESET over commit**, per the spec's own instruction, because committing `main`'s `CLAUDE.md` without `main`'s `workflow.yaml` would land a broken cross-reference on the issue branch and then onto `main`. Restore rc=0, tree clean, HEAD unchanged, payload intact, `--check-references` back to PASS. The sync therefore **cannot complete** in any worktree whose payload touches `FAMILY_workflow` — which is every workflow-surface task that edits `workflow.yaml`.

## Why this is a distinct bug from #2538

Different file, different mechanism. #2538 fixed a missing `gh pr ready` precondition in `.claude/skills/issue/steps/18-step-10d.md`. This is the Step 5a sync block's family map (`.claude/skills/issue/steps/09-step-5.md`) and/or `scripts/workflow_lint.py`'s reference check. #2538's own run merely *surfaced* it.

## The asymmetry worth grounding the fix on

The **SKILL-reference** check already tolerates this exact stale-worktree shape (#1622/#1672 — recorded as "Not blocking"), while the **workflow.yaml gate-key** reference check **hard-FAILs** on an identical root cause. One of the two behaviors is wrong for the family-atomic sync; the fix should make them consistent rather than adding a third behavior.

## Suggested direction (not prescriptive — the planner owns the design)

- **Option (a): `CLAUDE.md` joins `FAMILY_workflow`.** Same rationale that already documents `tests/issue_skill_source.py` as a singleton-by-cross-import: membership should follow the *reference* graph, not the directory. Cost: a dirty `workflow.yaml` now also freezes `CLAUDE.md`'s sync, which is the conservative direction the family-atomic rule already chose everywhere else.
- **Option (b): present-at-main tolerance in the gate-key check.** When an unresolved gate key IS defined in `origin/main`'s `workflow.yaml`, downgrade to WARN — mirroring the SKILL-reference check's existing tolerance. Cost: a genuinely-deleted gate key referenced by `CLAUDE.md` would WARN instead of FAIL until main catches up.
- **Audit the rest of the `FAMILY_OF` map for the same shape** — any other singleton that is a cross-family *reader* of a family member has this bug latent. `CLAUDE.md` is unlikely to be the only one; the `.claude/rules/*.md` files that quote `workflow.yaml` gate keys are the obvious next candidates.
- **Worth a mechanical pin:** a check asserting that every file whose content references `workflow.yaml` gate keys is in the same `FAMILY_OF` family as `.claude/workflow.yaml` (or is explicitly exempted with a recorded reason), so a future singleton cannot silently re-acquire the skew.

## Reproduction

In any worktree whose branch modifies `.claude/workflow.yaml`, run the Step 5a spec-freshness sync (`.claude/skills/issue/steps/09-step-5.md` lines 241-599) and attempt the commit. The `workflow-yaml-lint` pre-commit hook exits 1 on the unresolved `CLAUDE.md` reference.
