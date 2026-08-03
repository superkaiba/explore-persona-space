---
title: 'workflow-fix: gotchas — chained smoke-then-full per-leg out-root residue trap'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9a37a39dcf36
created_at: '2026-07-23T20:57:04Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate failure-lesson from #1586 fu crash-fix round 3 (epm:failure-lesson
  v4): chained smoke&&full dispatch left 44 GB keep-cell smoke residue; full leg''s
  wave-headroom assert starved; fix afcf2cabac reaps the derived sibling smoke out-root'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a gotcha_candidate failure-lesson raised on task #1586 (emitting agent: experiment-implementer, crash-fix round 3).

## Goal

Document the chained smoke-then-full per-leg out-root residue trap (earlier leg's keep-cell out-root starving the later leg's wave-headroom assert on a quota'd pod) in .claude/rules/gotchas.md

## Workflow gap

- **Bug observed:** #1586 fu launcher chain died at the full leg's p2_train_wave1 disk-headroom assert: the smoke leg's keep-cell out-root left 44 GB unowned residue inside the 130 GB /workspace quota
- **Why it is a workflow gap:** the crash-fix-rounds § per-leg out-roots convention prescribes per-leg out-roots for chained dispatches but no rule documents that the CHAIN leaves the earlier leg's out-root unowned — no leg owns its deletion — so every future multi-leg dispatcher composition can re-hit the class; gotchas.md is the canonical home for the trap + the reap recipe.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n -i -E 'smoke.{0,40}(out.?root|residue)|residue.{0,30}smoke|smoke.{0,30}keep.?cell' .claude/rules/gotchas.md` → 0 hits (absence claim — the trap is undocumented; 0-hit in-target IS the evidence) + `git log --oneline --since='7 days ago' -- .claude/rules/gotchas.md` → 10 commits, none covering the chained-leg out-root residue class (closest, ccd5155108, is the REGIME/CLASS smoke-coverage family — different trap) (2026-07-23)

## Proposed change (candidate diff sketch — refine in planning)

+ gotchas.md new bullet (§ dispatcher/disk family): "Chained smoke-then-full dispatches under per-leg out-roots leave the earlier leg's out-root as UNOWNED residue — ~44 GB of keep-cell smoke rungs starved #1586's full-leg wave-headroom assert inside the 130 GB /workspace quota. The LATER leg reaps the DERIVED earlier-leg out-root at its first phase entry: one shared derivation helper for writer + reaper (a drifted duplicate derivation reaps nothing), never under the earlier leg's own mode, only that path, fail-loud rmtree, one log line on every branch (reaped/absent/skip), pinned by an ordering test (residue gone BEFORE the headroom assert). Worked fix: issue1586_dispatch.py default_smoke_root + reap_sibling_smoke_root (afcf2cabac)."
+ .claude/rules/LESSONS.md gotchas.md row: extend the fires-when trigger with "write multi-leg (smoke&&full) dispatcher chains with per-leg out-roots" if not already covered by the existing dispatcher trigger text.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'per-leg out-root' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan (expected: crash-fix-rounds.md § per-leg out-roots gains a one-line cross-pointer to the gotchas bullet).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; LESSONS.md index stays in sync (`--check-lessons-index`); gotchas.md row-size ratchet respected (budget the cap-raise if over headroom).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 9a37a39dcf36

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: p2_train_wave1
lesson: A chained smoke-then-full dispatch under per-leg out-roots (crash-fix-rounds § per-leg out-roots) leaves the smoke leg's keep-cell out-root as UNOWNED residue on a quota'd pod — ~44 GB of smoke full-FT rungs sat inside the 130 GB /workspace quota and starved the full leg's wave-headroom assert. Fix: the full leg reaps the DERIVED sibling smoke out-root at its first phase entry (single shared derivation helper; never under smoke mode; only that path; fail-loud rmtree; one [smoke-reap] log line either way).
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
supersedes:
<!-- /epm:failure-lesson -->
