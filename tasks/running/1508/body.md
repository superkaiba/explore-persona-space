---
title: 'workflow-fix: gotchas.md entry — TRL mixed prompt/completion schema is nondeterministic
  UB'
kind: infra
tags:
- wf-fix
- wf-fix-fp:46315e88d664
created_at: '2026-07-18T08:00:31Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate failure-lesson from #1489 crash-fix round 4 (TRL
  0.29 mixed-schema UB)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a `gotcha_candidate: yes` failure-lesson raised on task #1489 (emitting agent: experiment-implementer, crash-fix round 4).

## Goal

Add a `.claude/rules/gotchas.md` entry documenting the TRL 0.29 prompt-completion mixed-schema undefined behavior and the mandatory tiny-real SFTTrainer tokenize-seam smoke for TRL-bound dataset builders.

## Workflow gap

- **Bug observed:** `build_distill_jsonl` (#1489) wrote `{"prompt": <message list>, "completion": <str>}`; TRL 0.29.1 `trl.data_utils.is_conversational()` pops ONE arbitrary key from a set and inspects only that value, so the row routes hash-order-nondeterministically — on the pod it hit the str-only `tokenize_fn` and crashed at `SFTTrainer.__init__` (`ValueError: text input must be of type str...`), while local dict-level probes passed. Cost one GCE provision cycle (~9 min, smoke-caught).
- **Why it is a workflow gap:** gotchas.md is the on-demand rule that loads when agents touch training code; it has no entry for this trap (grep 0 hits for is_conversational / mixed prompt-completion), so the next TRL dataset builder can re-hit it.
- **Confidence (emitter):** high
- verified-at-filing: `grep -c 'is_conversational\|prompt-completion.*mixed\|mixed.*prompt' .claude/rules/gotchas.md` → 0 hits in the named target (absence-of-guard claim; 0-hit IS the evidence) (2026-07-18)

## Proposed change (candidate diff sketch — refine in planning)

+ gotchas.md new bullet under the training section:
+ **TRL prompt-completion rows: conversational on BOTH keys or str on BOTH — never mixed.**
+ `is_conversational()` (TRL 0.29) pops one arbitrary key from a set → a mixed
+ {prompt: list, completion: str} row routes hash-order-nondeterministically to the
+ str-only tokenize path and crashes at SFTTrainer init, possibly only on the pod.
+ Match `scripts/issue778_finetune.py::_messages_to_prompt_completion` (message lists
+ both sides + completion_only_loss=True). Smoke every TRL-bound dataset builder
+ through the REAL train_lora → SFTTrainer tokenize seam with a tiny real-vocab model
+ (tests/test_issue1489_distill_dataset.py pattern; incident #1489 round 4).

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'is_conversational' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 46315e88d664

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: P3 distill (scripts/issue1489_gpu_phase.py build_distill_jsonl -> train_lora/SFTTrainer)
lesson: TRL 0.29 prompt-completion SFT rows must be conversational on BOTH keys (message-dict lists, the #778 _messages_to_prompt_completion shape) or plain str on both — a MIXED {prompt: list, completion: str} row is undefined behavior: is_conversational() pops ONE arbitrary key from a set, so the row routes hash-order-nondeterministically to the str-only tokenize_fn and crashes at SFTTrainer init, possibly only on the pod. Smoke every TRL-bound dataset builder through the real train_lora -> SFTTrainer tokenize seam with a tiny real-vocab model (tests/test_issue1489_distill_dataset.py pattern).
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
