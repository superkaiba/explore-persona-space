---
title: 'workflow-fix: enumerate smoke blind spots — gates/paths a smoke run structurally
  cannot reach'
kind: infra
tags:
- wf-fix
created_at: '2026-08-07T07:11:51Z'
has_clean_result: false
origin_prompt: '/issue 1336 — round 4: two consecutive production launches (SLURM
  4684, 5005) died on checks the pre-launch smoke structurally bypassed (branch-substituted
  MPNet hid a missing dependency; assert_split downgraded to informational under smoke)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from #1336 round 4 (pooled-multidataset,
plan v15). Two production launches died in a row on failures the pre-launch smoke run
was STRUCTURALLY INCAPABLE of catching — not because the smoke was too small, but
because the smoke branch runs DIFFERENT CODE:

1. **Branch-substituted implementation hid a missing dependency.** `embed_prompts()` in
   `scripts/issue1336_pooled_split.py` only instantiates the real MPNet
   `SentenceTransformer` when `smoke=False`; under smoke it returns a hash-based 32-dim
   toy vector. So `import sentence_transformers` was never executed by the smoke, the
   undeclared dependency stayed invisible, and SLURM 4684 died on
   `ModuleNotFoundError` at the top of an 8-GPU production run.

2. **Gate downgraded to informational under smoke.** `assert_split(..., smoke=ctx.smoke)`
   downgrades its split assertions when smoke is set. The smoke therefore reported PASS
   on a split instrument whose production gates it had not evaluated. SLURM 5005 then
   died at `assert_split` on the first production launch.

Both are the same class: **the smoke's PASS certified less than it appeared to, and
nothing in the plan or the review surface said so.** The plan's §-smoke section read as
pre-launch validation; no artifact enumerated what the smoke could not exercise.

This is DISTINCT from the existing smoke family and must not be deduped onto it:
- #1611 — smoke didn't cover every behavior-class x regime (missing CELLS; same code).
- #1727 — a smoke-valued variable leaked into production (wrong VALUE; same code).
- #1355 — production-calibrated gates KILL the smoke leg. That is the INVERSE direction:
  a gate too strict at smoke n. Here the gate is too LOOSE at smoke, or absent entirely.
- #822 — code-reviewer must FAIL on a missing smoke-architecture-check (presence of a
  check, not the blind spots of a passing one).

## Goal

Make a smoke run's blind spots explicit and reviewable, so a smoke PASS is never read as
production readiness for gates and code paths it structurally cannot reach.

## Workflow gap

No rule, planner section, or critic lens requires enumerating the production gates and
code paths a smoke run cannot exercise. Reviewers see "smoke PASSed" and treat the
launch as validated. The three mechanisms that silently narrow smoke coverage are:

- a `smoke` branch that SUBSTITUTES an implementation (toy embedding, stub model, fake
  judge) so the production dependency / API call / import is never executed;
- an assertion or gate DOWNGRADED to informational (or skipped) when `smoke` is set;
- a code path reached only on the production branch (a phase, a device route, an upload).

## Proposed fix

1. **New rule** `.claude/rules/smoke-blind-spots.md` — fires when a plan declares a
   pre-launch smoke run, or when code adds/edits a `smoke`-conditional branch. It
   requires a SMOKE BLIND-SPOT ENUMERATION: for the smoke leg, list every production
   gate the smoke downgrades or skips, every implementation the smoke substitutes, and
   every third-party import reached only on the production branch. Index it in
   `.claude/rules/LESSONS.md` (lint: `--check-lessons-index`).

2. **Planner enforcement** — the plan section that declares the smoke states, in one
   short block, what the smoke's PASS does and does NOT certify, derived from (1).
   An empty enumeration is written as `none — smoke executes every production gate`,
   never left blank.

3. **Critic / code-reviewer enforcement** — REVISE/FAIL when a diff introduces or edits
   a `smoke`-conditional branch that substitutes an implementation or weakens an
   assertion without the enumeration naming it. This is the check that would have caught
   both #1336 failures at review time, before two production launches.

4. **Mechanical arm (best-effort)** — a `workflow_lint.py` check that greps changed
   scripts for `smoke`-conditional branches guarding an import, a model constructor, or
   an `assert`/`raise`, and WARNs when the plan carries no blind-spot enumeration.
   Grep-level, so WARN not FAIL; the reviewer lens in (3) is the binding gate.

## Acceptance criteria

- [ ] `.claude/rules/smoke-blind-spots.md` exists and is indexed in `LESSONS.md`; the
      `--check-lessons-index` lint passes.
- [ ] The planner surface requires the enumeration for any plan declaring a smoke run.
- [ ] The code-reviewer surface FAILs on an unenumerated smoke-conditional branch that
      substitutes an implementation or weakens an assertion.
- [ ] A regression test pins the reviewer/lint behavior on a fixture reproducing BOTH
      #1336 shapes (the import-behind-`smoke=False` constructor, and the
      assertion downgraded by a `smoke` kwarg).
- [ ] `uv run python scripts/workflow_lint.py` is clean, and the mapped tests from
      `select_step9c_tests.py --map-files` pass.

## Provenance

- workflow_fix_target: .claude/rules/smoke-blind-spots.md
- fingerprint: d683272f741c
- origin: /issue 1336 round 4 (pooled-multidataset, plan v15) — two consecutive
  production launches died on checks the pre-launch smoke structurally bypassed.
- Workflow-surface rules apply; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` (Provenance `workflow_fix_target:` line) -- it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

Surfaced by #1336 round 4. Evidence: SLURM 4684 (`ModuleNotFoundError:
sentence_transformers`, dependency fixed in 04b36b2743) and SLURM 5005
(`AssertionError` in `assert_split`) — two consecutive production launches, each dying
on a check the smoke had structurally bypassed. Reference implementation of both
mechanisms: `scripts/issue1336_pooled_split.py` (`embed_prompts`, `assert_split`).
