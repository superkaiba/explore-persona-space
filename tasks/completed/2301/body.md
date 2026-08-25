---
title: 'workflow-fix: Step 10d Guard 5 trigger self-fires on a note that merely mentions
  the candidate token'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-14T18:33:07Z'
has_clean_result: false
parent_id: 2290
origin_prompt: 'Step 10d on #2290: Guard 5''s bare-substring trigger matched the session''s
  own heartbeat note reporting zero candidates, firing the guard with no sibling named'
workflow: v1
---
## Goal

Anchor Step 10d Guard 5's trigger predicate so a marker note that merely MENTIONS
the candidate token cannot self-trigger the guard. Today the trigger is an
unanchored substring grep over all of a task's note text, so a heartbeat that
truthfully reports "zero candidate notes" makes the guard fire on every
subsequent Step 10d invocation for that task.

## Scope

- The trigger in `.claude/skills/issue/SKILL.md` (Guard 5, "Sibling
  merge-sequencing hold + proactive pre-resolution", #1757) is:
  `grep -F '<candidate-token>' "$(uv run python scripts/task.py find <N>)/events.jsonl"`
  — a bare substring match against the whole events file, note prose included.
- Anchor it on the actual candidate-record SHAPE instead. The guard's own body
  already needs a named sibling (it iterates "per named sibling `<M>` (dedup)"),
  and the Step 2b record that creates a real candidate carries a `sibling=<M>`
  token plus `path=` / `source=` fields. A predicate that requires the
  `sibling=<M>` token (or that matches the structured record rather than free
  text) is both sufficient for the guard's semantics and immune to a prose
  mention.
- Keep it CHEAP: Guard 5's no-candidate path must stay a single grep. Do not
  turn this into a JSON-parsing pass over events.jsonl unless a one-line
  anchored grep genuinely cannot express it.
- Pin the anchored predicate with a test in the existing Step-10d test family
  (`tests/test_step10d_guards.py` / `tests/test_step10d_guard3.py` — read what
  is there and extend the closest file). The regression fixture is the observed
  shape: an events file whose ONLY match is a prose note containing the token
  with no `sibling=` token must NOT trigger the guard, while a genuine Step-2b
  candidate record MUST.
- Out of scope: the hold semantics, the bounded 45-min Monitor wait, half (ii)'s
  in-worktree pre-resolution, and the Step 2b record format itself. This is a
  trigger-anchoring fix only.

## Provenance

workflow_fix_target: .claude/skills/issue/SKILL.md

Hit live on #2290's Step 10d. Guard 5 ran twice: the first run correctly
reported no candidates. The session then wrote a `[long-phase-heartbeat]`
progress note whose Guard-5 line quoted the literal token while reporting its
absence. On the second guard pass (after a remedial `git merge origin/main`),
the same grep matched that note and Guard 5 fired. Adjudicated as a genuine
no-op — `grep -oE 'sibling=[0-9]+'` over every match returned nothing, so no
sibling was ever named — but the adjudication cost a diagnostic detour during a
merge window, and a less careful session could have entered a bounded 45-minute
hold, or run half (ii)'s in-worktree merge, against a sibling that does not
exist.

Note the self-demonstration: this task body necessarily contains the token too,
so its own Step 10d Guard 5 will fire for the same reason until the anchoring
lands. That is the cheapest possible confirmation that the predicate, not the
prose, is the defect.
