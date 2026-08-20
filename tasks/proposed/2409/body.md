---
title: 'Step 10d pre-push lint gate: 900s per-leg fence is sized off the IDLE range
  the same section forbids (rc 124 reads as a payload crash)'
kind: infra
tags:
- workflow-fix
- step10d
created_at: '2026-08-20T02:21:08Z'
has_clean_result: false
parent_id: 2204
origin_prompt: '#2204 Step 10d lint gate returned crash on GATED_RC=124 (timeout)
  with all other legs green; load 27 (15min avg), 3 foreign gates live'
workflow: v1
---
# Step 10d pre-push lint gate: the 900s per-leg fence is sized off the IDLE range the same section says not to use

## Goal

The Step 10d pre-push workflow-lint gate bounds each `workflow_lint.py` invocation at `timeout --kill-after=60s 900s`. Re-size that fence off the LOADED range — which the same subsection explicitly instructs — or make the timeout-vs-violation distinction visible in the verdict so a contention timeout is not diagnosed as a payload defect. Decide which of the two (or both) in the plan.

## Why (incident, measured)

`.claude/skills/issue/steps/18-step-10d.md` § Pre-push workflow-lint gate states, verbatim:

> Measured wall ~4.5-6 min (no-flags) + ~1.4 s (parity leg) ... The two leg pairs + TG legs total ~9-12+ min on an IDLE VM, but **30-40 min under typical fleet load (3+ concurrent gates)** — measured (#1690/#1694/#1711). Size any wall-time-derived fence off the LOADED range, not the idle one.

The gate's own per-leg fence is **900s**, which is ~2.5-3x the 4.5-6 min IDLE figure — i.e. sized off precisely the range the paragraph forbids. The instruction and the implementation contradict each other in the same subsection.

Realized on #2204 (2026-08-19/20), a workflow-surface round:

- `[diag] verdict inputs: GT_RC=0 BASE_RC=0 **GATED_RC=124** TG_CRASH=no TG_RC=0 TG_BASE_RC=0`
- Verdict: **`crash`** → merge blocked, correctly fail-closed.
- rc 124 is `timeout`'s signature. Every other leg passed: gate-tree construction, BOTH baseline lint legs, and BOTH mapped-invariant legs (`TG_BASE_RC=0`, `TG_RC=0`).
- Load averages at the verdict: **16.61 / 23.50 / 27.15**; the 15-min average of 27 covers the gated leg's window. Three foreign gate trees were live (issues 2147, 2201, 2205).
- Total gate wall before the crash: **1h37m** (the whole gate, spec-quoted at 30-40 min loaded).

So the payload was never implicated: the baseline leg ran the SAME two invocations on the same tree minus the payload and finished inside 900s, and the payload is a 139-line addition to a 13k-line file plus a one-constant change. The differentiator was contention, not content.

## Why this is worse than a slow gate

Verdict case 3 (`crash`) instructs: *"fix the crash cause in the worktree, re-run the gate ONCE; still crashing → `epm:merge-failed v1`"*, and describes the crash class as *"the linter itself CRASHED ... import error, missing dep, sparse-worktree crash ... the crash is payload-inducible"*. A contention timeout is none of those, but it lands in the same bucket with the same instruction — so the prescribed response is to go looking for a defect in the worktree that does not exist. On #2204 the diagnosis was only separable because `[diag] verdict inputs` prints the raw rc and 124 is recognizable; a reader who trusted the verdict label alone would have spent a round hunting a phantom linter crash.

Second-order cost: the sanctioned recovery is a full gate re-run. At ~1.5h per attempt under load, and with the fence most likely to blow exactly when the fleet is busy, the retry is disproportionately likely to hit the same wall — a doomed-retry shape the fleet-arbitration queue mitigates but does not fix.

## Acceptance

- The per-leg lint fence is re-derived from the LOADED measurement (the subsection's own 30-40 min total, and the #1690/#1694/#1711 figures it cites), with the arithmetic written down next to the constant so the next reader can check it rather than re-measuring. State the chosen value and its dispersion margin (the repo's p90-style x2 convention is the obvious candidate).
- A timeout is DISTINGUISHABLE from a linter crash at the verdict layer: either a distinct verdict token, or a mandatory `timeout` annotation in the crash line naming which leg hit 124 and its fence, so case 3's "fix the crash cause in the worktree" is not applied to a contention wall. State which mechanism and why.
- The fix does NOT weaken fail-closed: a timeout must still block the merge (no trustworthy compare exists). This task is about correct DIAGNOSIS and a correctly-sized fence, never about letting an uncertified payload through.
- Same treatment audited for the sibling fences in the same gate: the mapped-baseline leg (5040s), the gated TG leg (4620s), and the Step 9c gate's `recommended_timeout_s()`. Report which are loaded-range-derived and which are not; fix or explicitly justify each.
- Cross-check the sibling wall-time claim while there: Step 9c's docstring quotes median ~18 min / max ~38 min for its gate; #2204 realized **1:29:45** on a 183-file selection with `7024 passed`. Both overruns trace to the same cause — the invariant/mapped sets now contain many tests that each shell out a full no-flags lint run. Update the quoted ranges or state why they still stand.

## Provenance

Surfaced by the #2204 orchestrator during its own Step 10d merge (session 9e938266, 2026-08-19/20) when the pre-push lint gate returned `crash` on `GATED_RC=124` with every other leg green. Filed per `.claude/rules/workflow-fix-on-bug.md` — a gap in the workflow surface itself (`.claude/skills/issue/steps/18-step-10d.md`), distinct target and fingerprint from #2204's `scripts/verify_plan.py` deliverable, from #2402 (`guard_skill_doc_headroom.sh` raise-time validation), and from #2404 (c67 negation scoping).

Reference points: `.claude/skills/issue/steps/18-step-10d.md` § Pre-push workflow-lint gate (the "size off the LOADED range" instruction and the 900s legs), § Verdict bullet case 3 (the crash-class definition and its worktree-fix instruction), `.claude/skills/issue/steps/13-step-9.md` § 9c 1b (`recommended_timeout_s()`), #1690/#1694/#1711 (the loaded-range measurements), #2204 events.jsonl (the `[diag] verdict inputs` line and the load averages).
