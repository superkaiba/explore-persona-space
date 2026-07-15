---
title: 'workflow-fix: gotchas.md entry for concurrent-fanout shared-staging race'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a1650d670b5a
created_at: '2026-07-15T15:43:25Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate: yes failure-lesson from #1315 r5 (experiment-implementer):
  concurrent fanout units racing a shared _hfstage staging dest'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes` failure-lesson raised on task #1315 (emitting agent: experiment-implementer, crash-fix round 5).

## Goal

Add a gotchas.md entry: fanout dispatchers must pre-stage shared input dests ONCE in the parent before fanout; staging helpers need per-invocation mkdtemp staging dirs + atomic per-file publish + full-file-set staleness guards (a shared _hfstage scratch dir races concurrent units; a one-cell smoke cannot catch the class).

## Workflow gap

- **Bug observed:** 4 concurrent p4_parity fanout units each lazily staged the same margin_pools dest via issue1090_run._stage_hf_prefix's shared _hfstage scratch; one unit's os.replace stole a file a sibling's hf_hub_download just returned (FileNotFoundError, #1315 r5); a raw_pos.jsonl-only staleness guard also let units consume a partially-staged dest.
- **Why it is a workflow gap:** `.claude/rules/gotchas.md` (the codebase-trap register that loads when writing training/eval/orchestration code, incl. multi-GPU fan-outs) has no entry for the concurrent-fanout shared-staging race class, so the next fanout dispatcher author re-derives it from a production crash; a one-cell smoke structurally cannot catch it.
- **Confidence (emitter):** high (root_cause_confirmed: yes; fix + concurrency-pinned tests landed on issue-1315 @ ccfb975af3)
- verified-at-filing: `grep -ciE "_hfstage|_stage_hf_prefix|staging race|shared.*staging dir" .claude/rules/gotchas.md` → 0 hits (absence-of-entry claim — the 0-hit in-target result IS the evidence) (2026-07-15)

## Proposed change (candidate diff sketch — refine in planning)

In .claude/rules/gotchas.md (near the multi-GPU fan-out / CVD-clobber entries):
+ ## Concurrent fanout units racing a shared staging dest (#1315 r5)
+ A fanout dispatcher whose units lazily stage a SHARED input dest races the
+ staging helper two ways: a shared staging/scratch dir (e.g. dest/_hfstage)
+ lets one unit's os.replace steal a file a sibling's hf_hub_download just
+ returned (FileNotFoundError), and a single-proxy-file staleness guard lets
+ units consume a sibling's partially-staged dest. Pre-stage shared inputs
+ ONCE in the parent before the fanout; staging helpers use per-invocation
+ mkdtemp staging dirs + atomic per-file publish; staleness guards key on the
+ FULL consumer-required file set. A one-cell smoke cannot catch this class —
+ audit every unit-reachable shared-write path at fanout-design time.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing
  (`grep -rln '_stage_hf_prefix\|_hfstage' .claude/ CLAUDE.md scripts/`) and consider
  whether the experiment-implementer spec's fanout guidance should cross-reference the new entry; list hits in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes (incl. `--check-lessons-index` if the gotchas.md trigger line in LESSONS.md needs no change — it already covers fan-outs).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: a1650d670b5a

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: p4_parity (issue1315_dispatch._fanout_units -> issue1090_run._stage_hf_prefix)
lesson: A fanout dispatcher whose units lazily stage a SHARED input dest races the staging helper two ways: a shared staging dir lets one unit's os.replace steal a file a sibling's hf_hub_download just returned (FileNotFoundError), and a single-file staleness guard lets units consume a sibling's partially-staged dest. Pre-stage shared inputs ONCE in the parent before fanout; give the staging helper per-invocation mkdtemp staging dirs + atomic per-file publish (never a shared scratch dir); key staleness guards on the FULL consumer-required file set, never one proxy file. A one-cell smoke structurally cannot catch this class — audit every unit-reachable shared-write path at fanout-design time.
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
