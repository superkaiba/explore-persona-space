---
title: 'workflow-fix: gotcha — incremental cache reap only after the LAST hf_dl consumer'
kind: infra
tags:
- wf-fix
- wf-fix-fp:215de2354a34
created_at: '2026-07-18T14:08:39Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate: yes failure-lesson from #1489 crash-fix r6 (see
  body Provenance block)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes` failure-lesson raised on task #1489 (emitting agent: experiment-implementer, crash-fix round 6).

## Goal

Add a gotcha: sequence between-phase incremental cache reaps strictly after the cache's last consumer and give direct-path hf_dl readers a re-stage-on-demand fallback.

## Workflow gap

- **Bug observed:** issue1489_dispatch.sh ran clean_experiment_downloads --incremental between upload-a1 and P3; P3 read prefix_store.jsonl from the reaped hf_dl cache -> FileNotFoundError after ~3.5h of paid phases (#1489)
- **Why it is a workflow gap:** the CLAUDE.md § Disk hygiene between-phase reap contract says the cache "re-downloads on demand", but direct-path readers (`open()`-style `load_store` calls) implement no re-download — the contract's assumption is unstated at the gotcha level, and nothing warns dispatch-script authors to enumerate every later hf_dl consumer (including later provisions, e.g. phase_b) before placing a mid-run reap.
- **Confidence (emitter):** high
- verified-at-filing: `grep -cin "incremental.*reap\|clean_experiment_downloads\|last consumer" .claude/rules/gotchas.md` → 0 hits in the single named target (absence-of-guard claim; 0-hit IS the evidence) + `git log --oneline --since='7 days ago' -- .claude/rules/gotchas.md` → 8 commits, none covering reap sequencing (2026-07-18)

## Proposed change (candidate diff sketch — refine in planning)

```
+ **Between-phase incremental cache reaps fire only AFTER the cache's LAST
+ consumer — enumerate every `data/issue_<N>/hf_dl/**` reader first (incl.
+ later provisions).** The reap contract assumes re-download-on-demand that
+ direct-path readers (`parent.load_store(corpus_dir, ...)`-style open()s)
+ don't implement: #1489 reaped 0.05 GB right after upload-a1 and the next
+ phase crashed FileNotFoundError on the reaped corpus, after ~3.5 h of paid
+ GPU phases. If ANY later consumer exists there is no legal mid-run reap slot
+ (Step-8 terminal cleanup owns it; per-phase disk-headroom canaries guard
+ pressure). Defense in depth: guard corpus-consuming phase entries with a
+ re-stage-on-demand call through the existing deterministic staging helper,
+ pinned to the SAME revision earlier phases consumed.
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'incremental --apply\|--incremental' .claude/ CLAUDE.md scripts/`) and update every hit; consider whether CLAUDE.md § Disk hygiene's between-phase paragraph should name the enumerate-all-consumers precondition explicitly (same fix, second surface — planner's call). Mirror memory exists at `.claude/agent-memory/experiment-implementer/feedback_incremental_reap_last_consumer.md` (commit 235b7890ab).

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; gotchas.md row-cap/LESSONS-index consistency maintained (`--check-lessons-index`).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 215de2354a34

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: issue1489_dispatch.sh phase_a (between upload-a1 and P3 distill)
lesson: Sequence any between-phase incremental cache reap strictly AFTER the cache's LAST consumer in the whole run — enumerate every `data/issue_<N>/hf_dl/**` reader first — and give direct-path hf_dl consumers a re-stage-on-demand fallback: the incremental-reap contract assumes re-download-on-demand that `parent.load_store`-style direct readers don't implement, so a premature reap converts a 0.05 GB disk saving into a FileNotFoundError crash after hours of paid phases.
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
