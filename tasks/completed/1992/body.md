---
title: 'workflow-fix: flaky settle-sleep equality assert in inline-l'
kind: infra
tags:
- wf-fix
- wf-fix-fp:81fc8443f020
- daily-auto-filed
created_at: '2026-08-02T07:04:37Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-01 problem sweep (route 2): test_write_cert_malformed_rehash_delay_falls_back_and_still_toctous
  fails intermittently under load (2 of 3 single-test runs in the issue-1977 worktree;
  passes at the main checkout) because monkeypatch.setattr(ilg.time, ''sleep'', ...)
  patches the GLOBAL time module and captures interpreter-internal backoff sleeps
  (a 0.001*2^k series, 105+ entries) alongside write_cert''s single settle sleep,
  breakin'
workflow: v1
---
# workflow-fix: flaky settle-sleep equality assert in inline-lint-gate test

## Overview / Motivation

Auto-filed by the /daily 2026-08-01 Step C parked-candidate sweep from a workflow-fix candidate parked on task #1977 (emitting agent: implementer round 1, recursion-guarded; formal candidate block, fingerprint ffff2ff64004).

## Goal

Harden `test_write_cert_malformed_rehash_delay_falls_back_and_still_toctous` (and its sibling assertion) against interpreter-internal sleeps captured by the process-global `time.sleep` monkeypatch, so unrelated machine load cannot flip the Step 9c gate red on `tests/test_inline_lint_gate.py`.

## Workflow gap

- **Bug observed:** `test_write_cert_malformed_rehash_delay_falls_back_and_still_toctous` fails intermittently under load (emitter: 2 of 3 single-test runs in the issue-1977 worktree; passes at the main checkout) because `monkeypatch.setattr(ilg.time, "sleep", ...)` patches the GLOBAL time module and captures interpreter-internal backoff sleeps (a 0.001*2^k series, 105+ entries) alongside write_cert's single settle sleep, breaking the exact `slept == [2.0]` assertion.
- **Why it is a workflow gap:** the test pins the #1857 cert-retry workflow invariant on the Step 9c gate surface with an over-strict equality on a process-globally patched function, so unrelated machine load can flip the gate red on this file.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'slept' tests/test_inline_lint_gate.py` around the cited lines → `assert slept == [2.0], slept` present at ~L594 and `assert len(slept) == 1, slept` at ~L573, both via `monkeypatch.setattr(ilg.time, "sleep", ...)` (2026-08-02 UTC). Landed-fix check: `git log --oneline --since='7 days ago' -- tests/test_inline_lint_gate.py` → 2 commits (#1889, #1857), neither filtering the sleeps. The intermittency itself is the emitter's measurement (`unverified hypothesis — verify at plan time: 2-of-3 under-load failure reproduces`; the mechanism — global patch capturing interpreter-internal 0.001*2^k backoff sleeps — is the emitter's traced diagnosis, re-verify by inspecting the captured `slept` list on a failing run).

## Proposed change (candidate diff sketch — refine in planning)

```diff
- assert slept == [2.0], slept
+ settle = [s for s in slept if s >= 1.0]
+ assert settle == [2.0], slept
```
(same hardening for the sibling `len(slept) == 1` assertion at ~L573.)

## Scope / surfaces

- Primary target: `tests/test_inline_lint_gate.py`

## Constraints / invariants

- Workflow-surface only. Ruff on touched files passes; the hardened asserts still pin the #1857 cert-retry invariant (exactly one 2.0s settle sleep from write_cert).
- Recursion guard applies (workflow_fix_target Provenance line below).

## Provenance

- workflow_fix_target: tests/test_inline_lint_gate.py
- fingerprint: 81fc8443f020 (tag-authoritative; supersedes body-carried fingerprint: ffff2ff64004)
- origin: parked candidate on task #1977, ts 2026-08-01T08:06:03Z, routed by /daily 2026-08-01 Step C.

<!-- workflow-fix-candidate v1 -->
target_file: tests/test_inline_lint_gate.py
bug_observed: test_write_cert_malformed_rehash_delay_falls_back_and_still_toctous fails intermittently under load (2 of 3 single-test runs in the issue-1977 worktree; passes at the main checkout) because monkeypatch.setattr(ilg.time, "sleep", ...) patches the GLOBAL time module and captures interpreter-internal backoff sleeps (a 0.001*2^k series, 105+ entries) alongside write_cert's single settle sleep, breaking the exact `slept == [2.0]` assertion at tests/test_inline_lint_gate.py:594.
why_workflow_gap: the test pins the #1857 cert-retry workflow invariant on the Step 9c gate surface with an over-strict equality on a process-globally patched function, so unrelated machine load can flip the gate red on this file.
proposed_change: assert the settle sleep by filtered membership (settle-scale sleeps only) instead of exact list equality, and apply the same hardening to the sibling len(slept)==1 assertion at tests/test_inline_lint_gate.py:573.
confidence: high
related_task: #1977
<!-- /workflow-fix-candidate -->
