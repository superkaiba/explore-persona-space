---
title: 'workflow-fix: relocate genuine agents-prose asserts from behavioral tests
  into a closure-clean FAMILY_agents pin file (#2260 follow-up)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-21T17:44:05Z'
has_clean_result: false
parent_id: 2260
origin_prompt: 'Filed per #2260 plan v3 §4.5 (implementation-time step): convert the
  genuine-residual exempt files'' relocation from follow-up prose into a filed task;
  test_ensemble_review_cap.py deliberately excluded (#2420''s subject).'
workflow: v1
---
# Relocate genuine agents-prose asserts out of behavioral test files into a closure-clean pin file

## Goal

Relocate the genuine `.claude/agents/*.md` prose assertions that live inside
BEHAVIORAL test files (files whose dependency closure imports unsynced
`scripts/` / `src/` modules) into a dedicated closure-clean agents-prose pin
file, so those files can drop out of guard (20)'s
`_AGENTS_PROSE_EXEMPT_GENUINE` dict in
`tests/test_issue_skill_lint_family_sync.py` and the new pin file can join
the `FAMILY_agents` spec-freshness sync family (#2260).

Context: #2260 coupled the vetted closure-clean agents-prose pin tests to a
Step 5a/10d `FAMILY_agents` sync family. Behavioral files with genuine
agents-prose asserts could NOT join (membership would sync them against
branch-era `scripts/`/`src/`, the sync-scope boundary rationale (ii)), so
their asserts are carried as ACCEPTED RESIDUALS in the guard's
genuine-exempt dict. Each such assert can still red a worktree gate on pure
vintage skew (a fresh main-side agents edit + a branch-era assert). Moving
the asserts to a closure-clean pin file closes that residual.

## Scope — the asserts to relocate

- `tests/test_matched_support.py` — 3 prose asserts (~:350/:358/:391) over
  `statistics-critic.md` / `interpretation-critic.md` / `critic.md`
  (matched-support lens wiring pins). File closure: numpy +
  importlib-by-path of the matched_support implementation.
- `tests/test_pod_audit.py` — the `research-pm.md` triage-protocol assert
  (`test_pm_triage_protocol_present`, ~:721-723). File closure:
  pod_audit / pod_config / runpod_api scripts.
- `tests/test_bootstrap_pod_git_credentials.py` — the #1271
  no-tokenized-remote-URL negative pin over ALL agent specs
  (`test_no_tokenized_remote_url_in_experimenter_recipes`, agents-dir glob
  ~:291+). File closure: subprocess-executes `scripts/bootstrap_pod.sh`.
- `tests/test_verify_plan.py` — 2 planner.md prose asserts discovered at
  #2260 implement-time vet (the plan's provisional table had classed the
  file incidental): `test_planner_md_carries_predicate_anchor_literals`
  (~:6812-6822, 4 anchor literals) and the durability-pin author-side
  assert (~:7372-7375, `Durability pin:` / `#884` /
  `Selector registration`). File closure: `scripts/verify_plan.py`
  (unsynced).

## Deliverable

- A new closure-clean pin file (e.g. `tests/test_agents_prose_pins.py`;
  stdlib-only closure) carrying the relocated asserts verbatim (or
  behavior-equivalent), added to `FAMILY_agents` (FAMILY_OF entries + SPECS
  + SPECS_10D tokens in BOTH sync copies, a `_FORK_STUBS_2303` stub, and
  test (1)'s SPECS-literal pin update — the guard (20) failure message
  enumerates the full checklist).
- The four files above removed from `_AGENTS_PROSE_EXEMPT_GENUINE` (their
  reader-pattern matches then either vanish or become incidental — re-vet
  per the #2260 §4.1 decision rule; a residual incidental mention moves to
  `_AGENTS_PROSE_INCIDENTAL_EXEMPT` and must pass its AST shape check).
- Guard (20) green on the live tree after the move.

## Explicitly excluded

`tests/test_ensemble_review_cap.py` is DELIBERATELY excluded: it is #2420's
subject file — its workflow-prose coupling decision (the spelled-cap scan
over the whole workflow doc surface) belongs to #2420, and its
genuine-exempt rationale string in guard (20) names #2420 as the owner of
any future promotion. Do not relocate its agents-dir glob here.

## Acceptance

1. `uv run pytest tests/test_issue_skill_lint_family_sync.py -q` green with
   the four files out of the genuine-exempt dict.
2. The relocated asserts still pin the SAME prose (grep each literal in the
   new pin file).
3. `uv run pytest tests/test_matched_support.py tests/test_pod_audit.py tests/test_bootstrap_pod_git_credentials.py tests/test_verify_plan.py -q` green (no behavior loss in the source files).
