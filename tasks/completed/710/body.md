---
title: 'workflow-fix: add stall_reason to _PR struct in backends/runpod.py'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9359cb308676
created_at: '2026-06-28T08:39:17Z'
has_clean_result: false
origin_prompt: 'Claude code-reviewer #701 round 2 surfaced-prose follow-up: ''separate
  kind: infra fix for the runpod.py stall_reason regression on main'''
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised by the Claude `code-reviewer` during /issue #701 round 2 (`tasks/completed/701/events.jsonl` → `epm:code-review v2`, ts 2026-06-28T08:17:04Z, emitting agent: code-reviewer).

## Goal

Restore `stall_reason` on the `_PR` struct so `RunPodBackend.poll` no longer raises `AttributeError: '_PR' object has no attribute 'stall_reason'` when constructing a `PollResult` at `src/explore_persona_space/backends/runpod.py:389`.

## Workflow gap

- **Bug observed:** `tests/test_router*.py` / `tests/test_backend_*.py` produces 2 failing tests with `AttributeError: '_PR' object has no attribute 'stall_reason'` at `src/explore_persona_space/backends/runpod.py:389` (`stall_reason=raw.stall_reason`). The line is identical on `main`; `backends/` was never touched by #701. The Claude code-reviewer flagged this when running the full pytest suite during /issue #701 round-2 review.
- **Why it is a workflow gap:** the `_PR` (poll-result) struct is missing the `stall_reason` attribute the `RunPodBackend.poll` method assumes — a workflow-helper code-shape regression on the unified backend router (per `.claude/rules/workflow-fix-on-bug.md` § "Workflow surface" / "src/explore_persona_space/backends/*.py" listing). A planner has no way to satisfy the line without this fix.
- **Confidence (emitter):** medium (the symptom is clear, the fix is to add the attribute; whether `stall_reason` should be sourced from somewhere else or defaulted is the planner's call).

## Proposed change (candidate diff sketch — refine in planning)

Add the missing `stall_reason` attribute to the `_PR` (poll-result raw) struct so the `RunPodBackend.poll` consumer at runpod.py:389 can read it. Two plausible patch shapes for the planner to choose between:

- (A) Add `stall_reason: str | None = None` as a dataclass field on `_PR` (the raw poll-result struct populated by the SSH/SQL probe) — same default as the `PollResult` field — and have `poll_pipeline.py` (or whatever populates `_PR`) thread the actual stall-reason value when known.
- (B) Default-source `stall_reason` from `raw.__dict__.get("stall_reason")` at the `PollResult(stall_reason=...)` construction site so a missing attribute degrades to `None`.

Option (A) is the cleaner structural fix (the field belongs on `_PR`); (B) is a one-line `getattr(raw, "stall_reason", None)` defensive patch.

## Scope / surfaces

- Primary target: `src/explore_persona_space/backends/runpod.py` (the `_PR` struct definition + the `RunPodBackend.poll` construction site at line 389).
- Probably also the `_PR` populator (`poll_pipeline.py` or the SSH probe path that builds `_PR`) — the planner should grep for `_PR(` constructors and confirm whether the new attribute needs threading or just a default.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `tests/test_router*.py` + `tests/test_backend_*.py` must PASS after the fix.
- `scripts/workflow_lint.py` PASS.
- If `workflow.yaml` or `CLAUDE.md` need to change to document the field's semantics, they stay consistent with the existing `poll_pipeline.py` stall-reason taxonomy (see SKILL.md Step 7 § "Missing `failure_class` — invoke the classifier script" + `.claude/rules/failure_patterns.md`).

## Provenance

- workflow_fix_target: src/explore_persona_space/backends/runpod.py
- fingerprint: 9359cb308676
- related_task: #701 (the code-reviewer that surfaced this was reviewing #701; the bug itself predates #701 and is on `main`)

The Claude code-reviewer's full verbatim surfaced-prose:

> **One note on the test claim**
> The implementer's "356 tests passed" was a targeted subset. The **full** suite shows `5 failed, 5391 passed`. I verified all 5 are unrelated to this doc-only patch:
> - 2 are a **pre-existing `main` bug** — `AttributeError: '_PR' object has no attribute 'stall_reason'` at `src/explore_persona_space/backends/runpod.py:389` (line identical on `main`; `backends/` never touched by the #701 branch).
> - 3 are **test-isolation artifacts** (they pass when the 5 are run alone).
> - None reference `planner.md`.
>
> These do not block the merge but the orchestrator may want a separate `kind: infra` fix for the `runpod.py` `stall_reason` regression on `main`.
