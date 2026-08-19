---
title: Narrow over-broad scripts/issue*.py rule globs (3 rules, ~23K tokens on every
  issue script)
kind: infra
tags:
- context-hygiene
created_at: '2026-08-11T14:37:03Z'
has_clean_result: false
origin_prompt: 'Explain why this is happening: [worktree CLAUDE.md + rules reloading
  mid-turn] -> fix this'
workflow: v1
---
# Over-broad `scripts/issue*.py` rule globs load ~23K tokens of irrelevant rules on every issue script

## Goal

Narrow the `paths:` frontmatter globs on the three rules that currently fire on
`scripts/issue*.py`, so that a script which does no marker training, no marker
measurement, and no uploading stops pulling those rules into context — without
losing coverage on the code that genuinely does those things.

## The bug

Three `.claude/rules/*.md` files carry `scripts/issue*.py` in their `paths:`
frontmatter:

| rule | size | why it fires |
|---|---|---|
| `upload-policy.md` | 59,919 B | `scripts/issue*.py` |
| `marker-leakage-measurement.md` | 17,089 B | `scripts/issue*.py` |
| `marker-training-recipe.md` | 14,782 B | `scripts/issue*.py` |

`scripts/issue*.py` matches **all 1,387** issue scripts in the repo. Every one
of them — including pure plotters, pure aggregators, and read-only analysis
scripts — loads all three rules: **91,790 B ≈ 23K tokens** of always-on context
per touch.

## Measured instance

Touching `scripts/issue1336_full_transfer_lattice.py` (a matplotlib plotter: no
training, no marker, no upload) loads all three. Measured in a live session
2026-08-11: the plotter's total rule load-set was 453,305 B (~113K tokens), of
which 91,790 B was these three rules, none of which can apply to a plotter.

This is not worktree-specific — it fires identically at the repo root. The
worktree case just doubles it.

## Constraint (the reason this is not a trivial narrowing)

These rules exist to catch real mistakes: `upload-policy.md` guards the
persist-by-default contract; the two marker rules guard the lr ≤5e-6 window and
the on-policy marker-at-end DV. **A narrowing that stops firing on genuine
upload / marker code is a safety regression, strictly worse than the token
cost.** The fix must demonstrate retained coverage, not just a smaller glob.

## Suggested directions (not prescriptive — the plan owns the choice)

1. Split the trigger: keep a broad glob for the small always-relevant summary
   and move the deep mechanics behind a narrower one.
2. Narrow to the scripts that actually train/upload (e.g. `scripts/*train*.py`,
   `scripts/*upload*.py`, `scripts/issue*_train*.py`) and verify against the
   real set of marker/upload-bearing issue scripts.
3. Content-keyed triggering if the harness supports it.

## Acceptance

- Enumerate the issue scripts that genuinely train markers / measure marker
  leakage / upload artifacts; show the new globs still match all of them.
- Show the load-set reduction on a representative plotter.
- `workflow_lint.py` PASS.

## Related

- The worktree double-load (a nested `.claude/worktrees/<n>/CLAUDE.md` loads a
  second full copy of the always-on stack, ~37K tokens of pure duplicate) is a
  DISTINCT issue with a different target and a design-level fix; it overlaps the
  pending `/mnt/eps-data` bind cutover (#2132). Not in scope here.
