---
title: 'workflow-fix: .claude/skills/issue/SKILL.md is 813 bytes from its hard FAIL
  cap — next edit wedges the Step 9c gate fleet-wide'
kind: infra
tags: []
created_at: '2026-08-16T13:27:55Z'
has_clean_result: false
origin_prompt: 'Surfaced by #2324 Step 10 completion audit, Acceptance item 5 (workflow_lint
  no-flags): WARN reports SKILL.md at 982587/983400 bytes, 813 bytes under its grandfathered
  cap; the no-flags lint IS the Step 9c gate, so the next SKILL.md edit turns a fleet-wide
  WARN into a fleet-wide FAIL.'
workflow: v1
---
# workflow-fix: `.claude/skills/issue/SKILL.md` is 813 bytes from its hard FAIL cap — the next edit wedges the Step 9c gate fleet-wide

## Problem

`workflow_lint.py` no-flags currently emits:

```
WARN: .claude/skills/issue/SKILL.md: 982587 bytes — grandfathered; 813 bytes under its cap (983400).
```

The **no-flags `workflow_lint` run IS the Step 9c test-verdict gate** every `/issue`
session must pass before merging. The size ratchet FAILs (not WARNs) once a
grandfathered file exceeds its cap. So the next commit that adds even ~1 KB to
`SKILL.md` flips a fleet-wide WARN into a fleet-wide **FAIL**, and every concurrent
session's Step 9c gate goes red on a condition none of them introduced — the #1388
failure shape (two inline-landed lint-red scripts broke the Step 9c gate fleet-wide).

`SKILL.md` is among the most frequently edited files in the repo: it grew **3,219
bytes** during a single ~5-hour window on 2026-08-16 (979,368 -> 982,587) from four
independent landings — a3297aab05 (#2146), 6d40c1dff4 (#2323), 4cb68fcbc8 (#2320),
4992ff1933 (#2318). At that observed rate the remaining 813 bytes is well under one
day of normal fleet activity.

The sibling `.claude/skills/adversarial-planner/SKILL.md` is in the same band (73,229
bytes, 521 bytes under its 73,750 cap) and should be assessed in the same pass.

## Why this is urgent rather than routine

The failure mode is **cross-session collateral**: the session that trips the cap is
not the session that gets blocked. Whoever lands the next SKILL.md edit will pass
their own inline payload lint gate (their file is fine in isolation) and then every
OTHER session's Step 9c gate fails on a byte count. Diagnosing that from a red gate
is expensive — the failing line names a workflow-surface file the blocked session
never touched.

## Scope for the planner (deliberately not pre-decided)

Two broad directions, and the choice is a genuine judgment call:

1. **Slim the file.** SKILL.md has repeatedly been reduced by relocating long-form
   blocks into `.claude/rules/*.md` behind a pointer (the pattern used for
   `codex-ensemble-review.md`, `disk-hygiene.md`, `pods.md`, `compute-backends.md`,
   etc.). The relocation must preserve the always-on load-bearing summary + pointer
   so nothing silently stops loading, and `--check-lessons-index` must stay green.
2. **Raise the grandfathered cap deliberately.** The ratchet exists to force exactly
   this decision rather than let the file drift. If SKILL.md's size is genuinely
   justified, the cap should move as a recorded decision with a stated new ceiling —
   not be bumped reactively each time it is hit.

A third option worth pricing: make the ratchet's failure mode **less collateral** —
e.g. attribute a size FAIL to the file's own last-toucher rather than failing every
session's gate. That treats the class, not the instance, but is a larger change.

## Acceptance

- `workflow_lint.py` no-flags leaves `.claude/skills/issue/SKILL.md` with a stated,
  deliberate margin below its cap (either by slimming or by a recorded cap decision),
  and the same assessment is applied to
  `.claude/skills/adversarial-planner/SKILL.md`.
- If content was relocated: every relocated block retains an always-on summary +
  pointer in SKILL.md, `--check-lessons-index` passes, and no rule file is left
  unreferenced.
- If the cap was raised: the new ceiling and its rationale are recorded in the same
  commit as the constant change.
- The full no-flags `workflow_lint` run shows no NEW failures vs baseline.
- Step 9c universe green.

## Provenance

Surfaced during #2324's Step 10 completion audit while verifying Acceptance item 5
(`workflow_lint.py` no-flags shows no NEW failures). The 813-byte headroom is
incidental to #2324 — this round's change set contained zero `.claude/` files and
contributed none of the growth — and did not gate #2324's completion, which merged
cleanly as `d829f87272`.

A dedup scan over `tasks/REGISTRY.json` for existing SKILL.md size/cap/byte-budget
tasks returned **zero** matches, so this is not a duplicate. The workflow-fix
recursion guard was checked and does not apply: #2324's body carries no
`workflow_fix_target:` line and `EPM_WORKFLOW_FIX_SESSION` was unset.
