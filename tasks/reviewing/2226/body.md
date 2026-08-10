---
title: 'verify_plan.py: check for inherited argparse row-count defaults under-covering
  the plan''s registered per-cell target n (#2054 --target-conv-ids class)'
kind: infra
tags:
- wfx-verify-plan-argparse-default
created_at: '2026-08-10T22:31:45Z'
has_clean_result: false
origin_prompt: 'Methodology critic prose follow-up on #2054 plan v11 round 1: mechanizable
  — a verify_plan-style check: for each section-4-named reused generation/splice script,
  extract argparse integer defaults matching (target|max|limit).*(conv|rows|ids);
  FAIL/WARN when the plan''s stated per-cell target n exceeds a default and the flag
  token never appears in the plan text.'
workflow: v1
---
# verify_plan.py: check for inherited argparse row-count defaults under-covering the plan's registered per-cell target n

## Provenance

workflow_fix_target: scripts/verify_plan.py
Surfaced by: Methodology critic, task #2054 plan v11 round 1 (same-issue follow-up round `coordinated-common-set-regen`, 2026-08-10). Report: `/tmp/issue-2054-critic-methodology-v11-r1.md` (session-local; the incident summary below is self-contained).

## Gap

`verify_plan.py` has no check for the inherited-argparse-default under-coverage class: a plan reuses a generation/splice/capture script whose argparse integer default silently caps row/conversation coverage BELOW the plan's own registered per-cell target, and no embedded command overrides the flag. The plan reads as if the target is met; the run truncates deterministically (first-N prefix) and every downstream gate that runs before the consuming phase reads PASS without seeing the breach.

## Incident

Task #2054, amendment plan v11 (round `coordinated-common-set-regen`): the plan registered a ≥2d per-cell training-size target (n_train ≈ 7,200, d=3,584, intersection target |S| = 9,000), but the reused `--target-conv-ids` argparse default (8,000; `phase_c.py:604` / `phase_d.py:709` on the `issue-2054` branch, deterministic first-N prefix truncation) would have capped 28 of the 48 in-scope cells below the target (n_train 6,400 best case, ~5,700 under divergent per-variant orderings) for a majority of the 336 affected pairs. The round's pre-registered gate 1 runs BEFORE Phase C/D, so it structurally could not see the breach — the same fail-open class the round existed to fix. Caught only by the Methodology critic in round 1; the mechanical pre-pass (verify_plan.py) passed the plan clean twice (v10, v11).

Family: #1727 (smoke-valued variable leaking into production), #1345 (missing-filter class) — inherited defaults silently narrowing a new design's coverage.

## Sketch (from the critic report, verbatim class)

A verify_plan-style check: for each §4-named reused generation/splice script, extract argparse integer defaults matching `(target|max|limit).*(conv|rows|ids)`; FAIL/WARN when the plan's stated per-cell target n exceeds a default and the flag token never appears in the plan text.

Implementation notes for the planner (not binding):
- WARN-only to start (the heuristic has false-positive surface: scripts resolved from plan text, branch-resident scripts need `git show <branch>:<path>` resolution, and "stated per-cell target n" extraction from plan prose is fuzzy).
- Needs a canonical standalone N/A escape phrase (e.g. `N/A — no inherited row-count defaults`) per the `_standalone_na_declared` convention, wired into the adversarial-planner SKILL.md escape-phrase list.
- Tests in `tests/test_verify_plan.py` with fixtures reproducing the #2054 shape (default below target + flag absent → WARN; flag present with explicit value ≥ target → PASS; N/A line → PASS).
- Respect the existing check-id numbering + bundling conventions (`workflow_lint.py --check-lessons-index` untouched; this is a verify_plan check, not a rule file).

## Acceptance criteria

1. New check in `scripts/verify_plan.py` implementing the sketch (WARN-only acceptable), with check id + name following the existing convention.
2. Canonical N/A escape line recognized standalone-unwrapped, added to the adversarial-planner SKILL.md § canonical escape phrases list.
3. `tests/test_verify_plan.py` fixtures covering: the #2054 incident shape (WARN), explicit override (PASS), N/A declaration (PASS).
4. Full-suite verify_plan tests green.
