---
title: 'workflow-fix: gotchas entries for vLLM teardown pid-namespace probe + wandb-core
  collateral kill'
kind: infra
tags:
- wf-fix
created_at: '2026-07-07T02:32:35Z'
has_clean_result: false
origin_prompt: 'Two gotcha_candidate failure-lessons from #1090 crash-fix r3/r4: host-pid-namespace
  orphan probe unsoundness + wandb-core collateral kill in teardown_vllm children
  sweep'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from two gotcha_candidate: yes
failure-lessons raised on task #1090 (emitting agents: experiment-implementer
crash-fix r3 + orchestrator crash-fix r4).

## Goal

Add two vLLM-teardown gotcha entries to `.claude/rules/gotchas.md` (the existing
"vLLM in-process teardown" family): (1) orphan-GPU-pid guards that self-exclude
by `os.getpid()` are unsound on host-pid-namespace containers (RunPod) and
single-shot probes race the SIGKILL->release window — guard with a bounded drain
loop over a residual used_memory floor; (2) recursive psutil children kill-sweeps
kill the wandb-core service as collateral, breaking every subsequent wandb.init
in-process (ConnectionResetError at on_train_begin) — filter persistent non-GPU
service children out of the kill set.

## Workflow gap

- **Bug observed:** #1090 GPU phase crashed twice on these traps (crashes 3+4);
  neither trap was documented in gotchas.md despite the existing vLLM-teardown
  entry family covering the adjacent classes.
- **Why it is a workflow gap:** gotchas.md is the load-bearing surface for
  codebase traps; both classes plausibly recur for any consumer of a
  teardown-then-next-framework pattern.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

+ Two new bullets in `.claude/rules/gotchas.md` under the vLLM teardown family,
+ citing the reference implementation (drain loop + _is_protected_child in
+ src/explore_persona_space/experiments/behavior_testbed_545/eval_battery.py,
+ commits 08895b8e3a + 016d743d7e on issue-1090) and incident #1090 crashes 3-4.

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep: `grep -rln 'teardown_vllm\|orphan GPU' .claude/ CLAUDE.md` and update every hit consistently.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py` passes; ruff N/A (markdown).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: (compute at filing)

Two failure-lesson blocks on #1090 events.jsonl (2026-07-07, epm:failure-lesson) are the verbatim source.
