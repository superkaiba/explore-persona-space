---
title: 'workflow-fix: empty selection over local artifacts must fail loud, not exit
  rc=0'
kind: infra
tags:
- wf-fix
- wf-fix-fp:empty-selection-local-artifact-silent-rc0
created_at: '2026-08-05T17:19:56Z'
has_clean_result: false
origin_prompt: 'Raised on #1739: three instances in one round of a selection predicate
  mismatching an on-disk schema, selecting 0 rows, and exiting rc=0 as success (trait_rejudge
  rung alias; rescore_ood cells filter kept 0/826 and wrote n_metric_rows:0; compliance_full
  would have judged 0 of 159,990). Sibling of the gotchas #1092 streaming-filter entry,
  different scope and remedy.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1739 (emitting agent: the #1739 orchestrator session, 2026-08-05).

Task #1739's `evil-ood-spread-round` hit ONE bug class THREE times in a single round: a selection/filter predicate written against an assumed artifact schema selects ZERO rows from an already-local committed artifact, and the script treats the empty set as a successful no-op and exits **rc=0**. Two of the three burned real time (one produced a plausible-looking output file that read as a completed run and was misdiagnosed as a missing deliverable for two days); the third was caught only by an ad-hoc dry-run minutes before a ~160,000-call Batch API wave.

## Goal

Establish a fail-loud convention for empty post-selection row sets over local committed artifacts — an empty selection must raise, never exit rc=0 as a silent no-op — so this class is caught at the selecting script rather than after the compute is spent.

## Workflow gap

- **Bug observed (three instances, same round):**
  1. `scripts/issue1739_trait_rejudge.py` (2026-08-03, recorded at #1739 `epm:progress` v412): CLI rung names vs on-disk `train` / `hhrt` / `toxicchat` — leg died; fixed by adding `_LOCAL_RUNG_ALIAS` (commit `0d09a51f49`).
  2. `scripts/issue1739_rescore_ood.py` `_load_plain_ladder_cells`: filtered top-level `f_u` / `u_rung_label` while the committed `percell/cells.jsonl` carries that identity nested inside `unit_key` — kept **0 of 826** cells and wrote `ood_detection_metrics.json` with `n_metric_rows: 0` at **rc=0**. Measured fails-pre-fix at round 19: old filter 0/826, new filter 270/826. This one also cost two days of misdiagnosis, because the empty output file looked like a completed run.
  3. `scripts/issue1739_compliance_full.py` `_load_rollouts_local`: same rung-alias mismatch as (1), carried in a pre-`0d09a51f49` copy — would have judged **0 of 159,990** requests at **rc=0**. Caught pre-launch by a zero-API dry-run; fixed by porting `_LOCAL_RUNG_ALIAS` (commit `086d2588fe`).
- **Why it is a workflow gap:** the recurring cost is not the individual typo — it is that an empty selection is INDISTINGUISHABLE FROM SUCCESS. Every instance exited rc=0, and instance (2) additionally emitted a well-formed artifact. Round 19 added exactly the right guard (raise on empty cell selection AND on a 0-metric-row result set) but only to ONE site, so the next site re-learns it. Three occurrences in one round is the signal that this belongs in the always-consulted rule surface rather than in per-script patches.
- **Distinct from the existing nearest rule — this is a SIBLING, not a duplicate.** `.claude/rules/gotchas.md` already carries the #1092 real-corpus streaming-filter entry ("a filter chain written from assumed field shapes can reject 100% of rows while every synthetic-fixture smoke stays green", verified present: 1 hit for `reject 100% of rows`), and that entry even records #1739's own Reddit-dump string-boolean shapes. But its SCOPE is remote streaming corpus builders and its REMEDY is a bounded tiny-real streaming probe with per-filter reject counters. The class filed here differs on both axes: the input is an ALREADY-LOCAL COMMITTED artifact (judge rollout mirrors, per-cell JSONL, cell registries) where no streaming probe applies, and the remedy is a fail-loud non-empty-selection assertion at the selection site. The new entry should cross-reference the #1092 one rather than restate it.
- **Confidence (emitter):** high — three reproducible instances with measured fails-pre-fix evidence for two of them, all in current-tree scripts.
- verified-at-filing: per-target `grep -c` at body-compose time 2026-08-05 — `.claude/rules/gotchas.md`: `reject 100% of rows` = 1 (the sibling entry, must exist), `empty selection|non-empty-selection|selected zero rows` = **0** (the claimed gap, absent); `.claude/rules/code-style.md`: same gap pattern = **0**. All three incident scripts confirmed present on `issue-1739` HEAD via `git cat-file -e`. Recently-closed siblings on `gotchas.md` in the last 7 days reviewed (5 commits: #2079 sha-pin lint, #2073 autocompact window, #2058 spec sync, the probed-tool-traps pair, #2050 pgrep `-f`) — none covers empty-selection semantics, so this is not a re-raise of a just-landed fix.

## Proposed change (candidate diff sketch — refine in planning)

- Add a `.claude/rules/gotchas.md` entry, sibling to and cross-referencing the #1092 streaming-filter entry: **any selection/filter/join step over a local committed artifact must fail loud on an empty result set** — an empty post-filter row set, an empty cell selection, or a zero-row result table is a `RuntimeError`, never a silent rc=0 no-op — and must NOT write a well-formed "success" artifact (instance 2's `n_metric_rows: 0` file is the trap). State the three #1739 instances as the evidence and name the two recurring mismatch shapes: CLI-vs-on-disk key aliases, and identity nested inside a JSON-string column rather than at top level.
- Consider the code-side companion: a small shared helper (e.g. in `src/explore_persona_space/`) that wraps "select rows, assert non-empty, report the rejecting predicate" so new scripts inherit it instead of re-deriving — the planner should judge helper-vs-convention. Whatever lands, the per-unit reject-counter idea from the #1092 entry is the right diagnostic to carry over (so a 0-selection failure names WHICH predicate emptied the set).
- Do NOT weaken any existing fail-fast rule; this only adds a case.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`.
- Secondary (planner's call): `.claude/rules/code-style.md` fail-fast section, and/or a shared selection helper under `src/explore_persona_space/`.
- Evidence-only, do not edit: `scripts/issue1739_{trait_rejudge,rescore_ood,compliance_full}.py` (all three already fixed in-round).

## Constraints / invariants

- Workflow-surface only for the rule text; a shared helper would be library code and is in scope only if the planner elects it.
- Must not contradict the #1092 streaming entry — cross-reference, do not restate or supersede.
- `scripts/workflow_lint.py --check-lessons-index` passes if a rule row is added/renamed; ruff clean on any touched code; the gotchas size ratchet respected.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: empty-selection-local-artifact-silent-rc0

Raised as prose by the #1739 orchestrator after the third instance in one round was caught minutes before a ~160k-call Batch API wave; the correction + full three-instance table are recorded in an `epm:progress` marker on #1739 (2026-08-05).
