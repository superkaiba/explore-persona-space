---
title: 'daily-fix: make the implementer NOT-RUN escape pre-emptive f'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9ba4711d501b
- daily-auto-filed
created_at: '2026-07-27T07:15:44Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 2): an implementer burned ~51
  min on four timeout-killed local runs of a test file the selector already reports
  as slow, ending in the same NOT-RUN escape it could have taken at minute zero'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 2). Surfaced by 3 independent
miner group(s) over the 2026-07-26 session transcripts.

## Goal

Make the implementer's NOT-RUN escape PRE-EMPTIVE — when the Step 9c selector reports a
selected file in its `slow_tests_selected` list, route it straight to NOT-RUN + defer to
Step 9c with zero local attempts.

## Workflow gap

- **Bug observed:** Implementers attempted `tests/test_workflow_lint.py` locally under sub-600 s fences at least six times across three sessions on 2026-07-26; every attempt was timeout-killed and every session ended in the same NOT-RUN escape it could have taken at minute zero.
- **Why it is a workflow gap:** `.claude/agents/implementer.md` frames the escape post-hoc ("if a mandatory-set file genuinely cannot finish in-turn … the existing NOT-RUN escape applies"), and cites a stale 319-771 s runtime, while the selector already publishes the machine-readable signal (`slow_tests_selected`, and a 2400 s surcharge for this exact file) that neither implementer spec reads.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'NOT-RUN\|test_workflow_lint\|slow_tests_selected' .claude/agents/implementer.md .claude/agents/experiment-implementer.md` → implementer.md 3 hits (L174 post-hoc escape naming `tests/test_workflow_lint.py` at "319-771 s"; L197 generic NOT-RUN escape; L254 report field), experiment-implementer.md 1 hit (L953, generic NOT-RUN, file not named); `slow_tests_selected` and `recommended_timeout_s` → 0 hits in BOTH specs. `grep -n 'SLOW_TESTS\|slow_tests_selected' scripts/select_step9c_tests.py` → the selector emits `"slow_tests_selected"` in its `--json` payload (L1797) and carries `SLOW_TESTS = {"tests/test_workflow_lint.py": 2400}` (L740-746) whose comment records the wall growing "771 -> 1819 s max" and a measured 1188.62 s standalone. `git log --oneline --since='7 days ago' -- .claude/agents/implementer.md .claude/agents/experiment-implementer.md` → 6 commits, none making the escape pre-emptive (2026-07-26)

## Evidence

- Session `0e2c3b21` (implementer subagent `agent-a2759a68efd2d1b7f`), 09:32:52Z→10:23:44Z: four attempts on `tests/test_workflow_lint.py` (9,088 lines / 531 tests) — `timeout 500` SIGTERM'd, `timeout 850` inside a 900 s Bash (harness-converted to background, `Terminated`), a detached `timeout 900` + poll loop, plus two intervening `pgrep`/`sleep` poll loops — before taking the documented NOT-RUN escape. Its own report: `"rc=124 = timeout hit (900s exceeded …). Made it to ~54% (~380/700+ tests) before the 900s cap. Under this fleet load test_workflow_lint.py needs >900s … The result: NOT-RUN due to timeout."` Measured cost ~51 min of implementer wall-time, 4 discarded runs, on a shared VM already running 3 concurrent Step 9c gates. Step 9c covered the file 40 min later with a 5310 s budget.
- Session `8571eca6`, 12:14:20Z: the #1698 implementer reported the same file `"timed out at both 540s foreground and 1400s background under fleet load from 2 concurrent Step 9c gate runs on the shared VM"`, naming it NOT-RUN in the report.
- Session `a5a4b7bd` (subagent `a510618c`), 07:48:25Z→07:57:25Z: `timeout --kill-after=30s 540s uv run pytest tests/test_workflow_lint.py -x -q` returned `Exit code 143 Terminated`; the implementer then ran a `-k` subset (29 passed / 535 deselected in 4 s) and deferred the remainder. Measured cost ~9 min on a killed run, and the implementer-side verification of a file it had just edited rested on 29 of 564 tests.
- The failure was predictable at dispatch in every case: the file is the sole entry in the selector's `SLOW_TESTS` surcharge table, whose in-tree comment records a measured max of 1819 s — above every fence any of the three sessions could set, since the Bash tool cap is 600 s foreground.
- Net information gained across all three sessions: zero. Every run ended at the same NOT-RUN escape, and Step 9c ran the file to completion in each case.

## Proposed change

- In `.claude/agents/implementer.md` § "Run tests — gate-matched scope (#1288)" (L174), replace the post-hoc sentence with a pre-emptive rule: any selected file appearing in the selector's `--json` `slow_tests_selected` list is routed to NOT-RUN + Step 9c deferral with ZERO local attempts; report it in `(c) How to verify` with the exact copy-pasteable command and the selector's `recommended_timeout_s`.
- Drop the stale "319-771 s" figure — it is contradicted by the selector's own `SLOW_TESTS` comment (max 1819 s, 1188.62 s standalone) and reads as "worth one try". Cite the selector field, not a frozen number, so the spec cannot drift again.
- Keep the existing distinction intact: a pin-sweep HIT left NOT-RUN stays presumptively blocker-adjacent for the code-reviewer; the pre-emptive route changes WHEN the escape is taken, not its downstream treatment.
- Add the corollary that a `-k` subset is an acceptable local substitute ONLY when it is named explicitly in the `epm:results` marker with its deselected count, so a 29-of-564 subset can never read as a full local run.
- Mirror the same sentence in `.claude/agents/experiment-implementer.md` (its NOT-RUN escape at L953 is generic and names no slow file).
- unverified hypothesis — verify at plan time: a fleet-wide single-flight around Step 9c / pre-push lint-gate launches would remove the contention that pushed this file past its historical band (session `8571eca6` proposed it). That is a larger change than the pre-emptive escape and should be scoped separately; the escape is correct regardless of contention.

## Scope / surfaces

- Primary target: `.claude/agents/implementer.md`
- `.claude/agents/experiment-implementer.md` (mirror sentence)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `uv run python scripts/workflow_lint.py` passes (no-flags); ruff clean on touched files.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 9ba4711d501b

- workflow_fix_target: .claude/agents/implementer.md
- fingerprint: PENDING

/daily 2026-07-26 route-2 filing. Miner refs: H-P1, E-P15, J-P8.
