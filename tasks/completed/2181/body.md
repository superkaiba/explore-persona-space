---
title: Ban repo_root()-derived sys.path inserts in tests (worktree runs silently exercise
  main's code)
kind: infra
tags: []
created_at: '2026-08-07T18:07:56Z'
has_clean_result: false
origin_prompt: 'Surfaced by task #2164 Step 10d pre-push lint gate: a repo_root()-derived
  module-level sys.path insert in tests/test_issue1482_densesae_fullwidth.py made
  worktree-run pytest import main''s copy of the module under test and leaked a foreign
  scripts/ onto sys.path, defeating the #1296 negative control in tests/test_backend_poll.py.'
workflow: v1
---
# Ban `repo_root()`-derived `sys.path` inserts in tests (a worktree run silently exercises main's code)

## Goal

Add a `workflow_lint.py` check (plus its invariant test) that FAILS on any
`sys.path.insert(...)` / `sys.path.append(...)` under `tests/` whose argument is
derived from `task_workflow.repo_root()`, and require the tree-local
`Path(__file__).resolve().parents[1] / ...` form instead.

## Why (concrete incident, task #2164)

`tests/test_issue1482_densesae_fullwidth.py` carried, at module level:

```python
from explore_persona_space.task_workflow import repo_root
sys.path.insert(0, str(repo_root() / "scripts"))
```

`repo_root()` branch-guards to the MAIN checkout. Under any worktree-run pytest
session that had two consequences, both silent:

1. **The test exercised main's copy of the module under test, not the tree's.**
   It imports `issue1482_densesae_fullwidth` from `scripts/`. From a worktree,
   `repo_root()/scripts` is the MAIN checkout — so a branch modifying that
   driver had its own test import a different file than the one it changed. A
   real regression in that driver could pass its own test on the branch and only
   surface after merge.
2. **It leaked a foreign checkout's `scripts/` onto `sys.path` for the whole
   pytest session.** `tests/test_backend_poll.py`'s #1296 negative control
   scrubs only entries resolving to the LOCAL tree's `scripts/`, so the leaked
   entry survived the scrub and a bare `import runpod_api` still resolved:
   `Failed: DID NOT RAISE ModuleNotFoundError`.

Effect (2) is how it was found: it turned #2164's Step 10d pre-push lint gate
red as a payload-attributed NEW node, because that gate runs its mapped-test
legs asymmetrically — gated from the worktree (where `repo_root()` does not
match the cwd tree), baseline from the repo root (where it does, so the entry is
scrubbed and the control passes). Diagnosis cost roughly ten probe commands plus
a full ~20-minute gate re-run on a branch whose payload was innocent.

#2164 FIXED the single offending instance (landed on main via `5c91482fce`;
fix commit `69a58d6e5be439bc6465f767609267a75f784ad9`) and swept `tests/`,
`scripts/`, and `src/` — it was the only occurrence at that time. This task adds
the mechanical guard so it cannot come back, since nothing currently forbids it.

## Scope

1. A `workflow_lint.py` check flagging `sys.path.insert`/`append` under `tests/`
   whose argument expression mentions `repo_root(`. `workflow_lint.py` already
   carries an adjacent repo-root `sys.path` guard (see its docstring around the
   `_ensure_repo_root_on_syspath()` exemplar) — extend that family rather than
   adding a parallel mechanism, and bundle it into the no-flags default run so
   the Step 9c / Step 10d gates pick it up.
2. An invariant test for the new check, including a NEGATIVE control proving it
   fires on the pre-fix form (the deliberate-breakage non-vacuity pattern used
   by #2164's own anti-drift test).
3. Prefer `monkeypatch.syspath_prepend` over a bare module-level insert in the
   guidance text: `tests/test_clean_experiment_downloads_symlinks.py:57` already
   documents that choice ("so the entry is restored"), so the repo has a
   sanctioned restoring form to point at.
4. Secondary, only if cheap: a one-line diagnostic note in the SKILL.md gate
   subsection saying that a NEW node appearing ONLY in the gated leg can be a
   worktree-vs-root environment artifact rather than a payload regression, with
   the two-file reproduction shape as the check. This is the diagnosis-cost half
   of the incident; the lint check above is the prevention half.

## Out of scope

- Changing the gate's asymmetric baseline placement (gated `$WT` / baseline repo
  root). That asymmetry is deliberate per SKILL.md § Baseline semantics (the
  baseline must be a payload-free tree) and is not the defect here.
- The unrelated pre-existing `test_setsid_child_survives_group_kill` main-red,
  parked separately from #2164 in the #1681 urgent form.

## Provenance

Surfaced by task #2164's Step 10d pre-push workflow-lint gate on 2026-08-07:
verdict `block` on a NEW failing node that proved not to be payload-attributable
(both files involved were byte-identical to origin/main on that branch).
