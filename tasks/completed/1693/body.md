---
title: 'daily-fix: phase idempotency + inter-phase schema assert'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4cb38a3bc0db
- daily-auto-filed
created_at: '2026-07-26T07:02:09Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 2): Every #1689 crash-fix relaunch
  re-ran the paid Sonnet haiku_u2 phase and the full corpus and render phases from
  scratch across roughly four cycles, a downstream vLLM phase then died on 33 percent
  empty prompts from an unasserted producer-consumer schema mismatch after GPU spin-up,
  and a completed phase was turned into a whole-pipeline abort by an interpreter-shutdown
  crash under set -euo pipefail.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the `/daily` 2026-07-25 problem sweep from task #1689. This is the
**money** item of the day: a non-idempotent phased dispatcher re-ran a paid
Anthropic-API phase on every crash-fix relaunch, roughly four times.

## Goal

Make phase-level skip-if-output-exists mandatory for multi-phase dispatchers (a
BLOCKER-grade code-review gate when a phase makes paid API calls), require the
CONSUMER phase to assert its inter-phase JSONL contract before any model
initialization, and record the interpreter-finalization early-exit gotcha.

## Workflow gap

Three failures on #1689 (session `5c5a89e8`), all at phase boundaries:

1. **No idempotency ⇒ repeated paid API spend.** `issue1689_dispatch.sh all` has no
   skip-if-complete guard. Phase A (corpus 3800 rows + render 79,800 rows) re-ran on
   every one of ~4 relaunches, and `haiku_u2` (3 conditions × 3800 rows of real Sonnet
   calls, `smoke=False`) re-ran at least twice — the second time purely because a
   *downstream* vLLM phase crashed. Crash-persisted Phase-A artifacts already existed
   on HF under `issue1689_partial/`. CLAUDE.md's checkpoint-per-phase rule exists;
   nothing enforced it here.
2. **Unasserted producer/consumer schema ⇒ a wasted GPU cycle.** After the renderer
   produced 79,800 rows across 21 conditions, `onpolicy` died with
   `ValueError: The decoder prompt cannot be empty` at
   `vllm/v1/engine/processor.py:488`. Root cause (found in R8): chat-framing rows emit
   `messages: [...]` but no `prompt_text`, and the consumer called
   `row.get("prompt_text")` — **33% of rows resolved to empty**. The failure landed
   *after* vLLM initialization, so it cost a pod cycle rather than seconds of CPU.
3. **Interpreter-finalization crash turned a completed phase into a pipeline abort.**
   `[corpus] done: scanned=5132 kept=3800` — output written — then
   `Fatal Python error: PyGILState_Release … Python runtime state: finalizing`, a
   C-extension atexit race across 203 loaded modules (torch/scipy/sklearn/pandas/
   pyarrow). Under the dispatcher's `set -euo pipefail`, the nonzero rc from a phase
   whose WORK was complete killed phases B–E. R6 fixed it by adding an explicit
   process exit to 8 entrypoints.

All three share one shape: a phase boundary with no contract. (1) has no "already
done" contract, (2) has no input contract, (3) has no "work complete ≠ exit code"
contract.

- **Confidence (emitter):** high on all three (each carries the task's own
  `epm:failure` / implementer-report text). Medium on where the enforcement belongs.
- verified-at-filing: `grep -rn --exclude-dir=worktrees 'checkpoint-per-phase'
  .claude/rules/code-style.md CLAUDE.md` → the rule EXISTS (CLAUDE.md § Code Style
  names "**checkpoint-per-phase**" in the code-style pointer list), which is the point:
  the rule is present and was not enforced, so the gap is a missing GATE, not missing
  guidance — clause (c) context read confirms the existing text is advisory prose, not
  a reviewer-binding item. Incident text quoted above read from #1689's own
  `epm:failure` markers (`reason: vllm-empty-prompt`,
  `reason: python-interpreter-shutdown-crash`) and the R5/R8 implementer reports, not
  from recall. Landed-fix history check `git log --oneline --since='7 days ago' --
  .claude/agents/code-reviewer.md .claude/rules/code-style.md` → no idempotency or
  inter-phase-contract gate landed. (2026-07-25)

## Proposed change (refine in planning)

```
  code-reviewer gate (kind: experiment diffs adding/altering a phased dispatcher):
+ (1) every phase skips when its declared output artifact already exists
+     (or an explicit --force is passed). BLOCKER when the phase makes paid
+     API calls or holds a GPU; CONCERN otherwise.
+ (2) the CONSUMER phase asserts its inter-phase contract at the top —
+     required fields non-empty, row count, fail-loud drop report — BEFORE
+     any model/vLLM/accelerator initialization.

  gotchas.md entry:
+ a phase entrypoint importing torch/scipy/sklearn/pyarrow should exit the
+ process explicitly after flushing; a nondeterministic interpreter-finalization
+ crash otherwise converts a COMPLETED phase into a whole-pipeline abort under
+ `set -euo pipefail`.  (#1689: Phase A wrote its output, then died in finalize.)
```

## Scope / surfaces

- Primary target: `.claude/agents/code-reviewer.md` (the two gate items).
- `.claude/rules/gotchas.md` (the finalization entry) and/or
  `.claude/rules/code-style.md` § checkpoint-per-phase — the planner picks one home
  and cross-references, rather than duplicating.
- Consider whether the idempotency gate belongs partly in `planner.md` §9 (declare each
  phase's output artifact so the reviewer has something to check against). A gate that
  cannot name the expected artifact will degrade to a judgement call.
- Do NOT hard-code #1689 specifics; the schema mismatch is task-specific, the
  *unasserted contract crossing a phase boundary* is the recurring class.

## Constraints / invariants

- The idempotency requirement must not break deliberate re-runs: `--force` (or an
  equivalent) stays first-class, and a partial/corrupt output must not read as "done"
  (prefer a completion sentinel over bare file existence — the project's own
  `.claude/rules/` Monitoring rule already bans keying "done" on file existence).
- The consumer-side assertion must fail LOUD (no silent drop, no default fill) —
  CLAUDE.md § Critical Rules "Fail fast".
- `scripts/workflow_lint.py --check-references` / `--check-asks` pass; ruff passes;
  agent-spec size ratchet stays under budget.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/agents/code-reviewer.md
- fingerprint: 4cb38a3bc0db
- Source: `/daily` 2026-07-25 transcript sweep, session `5c5a89e8` (#1689) @
  2026-07-26T05:29Z/06:20Z (re-runs), 05:40:29Z (vllm-empty-prompt), 02:31:13Z
  (finalization crash).
