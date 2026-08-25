---
title: 'judge_pilot transport-parity check is opt-in so it sits unarmed by default:
  one round had two drivers, one armed and one not, with a 34.1%-censoring batch route
  riding on an unprotected sync pin'
kind: infra
tags: []
created_at: '2026-08-21T07:59:48Z'
has_clean_result: false
origin_prompt: /issue 2329
workflow: v1
---
# `judge_pilot`'s transport-parity check is OPT-IN, so it sits unarmed by default — two drivers in one round, one armed and one not, with a third of a wave's censoring riding on an unprotected constant

## Goal

Make the rule-26(c) / #2152 transport-parity check ARMED BY DEFAULT in
`src/explore_persona_space/eval/judge_pilot.py`, or require callers to waive it explicitly. Today an
UNDECLARED wave "keeps today's verdicts byte-identically and records ONE warning" — which means the
common path is a pilot gate that PASSes while stating, in its own artifact, that it did not verify the
property the gate exists to verify.

## Evidence (#2329, 2026-08-21) — same round, two drivers, inconsistent arming

`eval_results/issue_2329/q35_ladder_decay/judge/gates/pilot_gate_report.json` — `passed: true`, 8/8
rubrics PASS at 56 draws each — carries ONE warning, identical on every rubric:

> "wave transport UNDECLARED — pilot ran sync; transport parity NOT verified (rule 26 / #2152):
> declare wave_n_calls / wave_threshold_base / wave_force_sync to arm the mismatch FAIL"

- `scripts/issue2329_ladder_judge.py` (the ladder driver, which produced that gate): does NOT declare
  the wave transport => check unarmed.
- `scripts/issue2329_decay.py:815-816` (the Leg B driver, SAME round, same issue): DOES declare
  `wave_n_calls=...` + `wave_threshold_base=0` => check armed.

Nothing distinguishes the two cases methodologically. One author threaded it, one didn't.

## Why the exposure is severe rather than cosmetic

The ladder driver pins `FORCE_SYNC = J62.FORCE_SYNC_THRESHOLD_BASE = 10**9` at all seven dispatch
sites, so `decide_route` yields `effective_threshold = 1e9` and every wave routes sync — confirmed live
in the production log (`route: N=2160 | path=sync | cost_pref=latency | forced`). So the property HOLDS
today.

But it holds by an unprotected constant. Under the library default
`DEFAULT_THRESHOLD_BASE = 2000`, the two 2,160-item waves (`coherence.grid`, `hol-plain.grid`) route
**batch** — and per rule 28 / #1739 the batch path measured **34.1% censoring** against 0/14,887 sync
re-refusals. So an edit removing or lowering that pin would route roughly a third of the wave into
censoring, certified by a sync-validated pilot, and **the gate would not FAIL** because the check was
never armed. The gate cannot protect the very constant the run depends on.

That is the same defect shape as an assertion whose comparison set is empty: a check that reads as
verifying a property while structurally not evaluating it. The difference here is that the artifact
says so out loud, in a warning field nobody is required to read.

## Proposed fix

Preferred: **invert the default.** `judge_pilot` REQUIRES the wave-transport declaration
(`wave_n_calls` + one of `wave_threshold_base` / `wave_force_sync`) and raises `ValueError` before any
API spend when absent — matching the module's existing fail-fast posture for unsatisfiable
configurations (the #2124 sizing clause and the #2152 MF-1 OTPM-probe refusal both already refuse
before spend, so this is consistent rather than novel). An explicit
`allow_undeclared_wave_transport=True` (reason recorded at the caller site, mirroring the existing
`waive_api_refusal_arms` / `allow_subresolution_pilot` patterns) preserves the escape hatch.

Weaker fallback if inversion is too disruptive: keep the warning but make an undeclared wave FAIL the
gate when the pilot's realized route differs from the route `decide_route` would pick for the
caller's own default `threshold_base` — i.e. derive the comparison instead of requiring it.

Either way, thread the declaration through `scripts/issue2329_ladder_judge.py`'s pilot call so this
round's gate is armed on re-run.

## Acceptance criteria

1. A pilot invoked with no wave-transport declaration either raises before any API call, or FAILs when
   the realized route differs from the derived expected route. Reproduce with the #2329 ladder
   configuration (pilot sync, `threshold_base` default 2000, wave n=2160 => derived route batch =>
   must not silently PASS).
2. A declared-and-matching wave still PASSes byte-identically (regression: `issue2329_decay.py`'s
   existing armed call must be unaffected).
3. The explicit waiver path records its reason in the gate report.
4. No existing caller silently changes verdict without either raising or recording — enumerate the
   `judge_pilot` call sites and state each one's post-change disposition.
5. Tests failing before and passing after; no new red in the no-flags `workflow_lint.py` run or the
   mapped-test selection.

## Candidate metadata

- target_file: src/explore_persona_space/eval/judge_pilot.py
- fingerprint: judge-pilot-transport-parity-opt-in-unarmed-by-default
- confidence: high — the unarmed gate, the armed sibling call site, the pin, and the live sync routing
  were all read directly in #2329; the batch-censoring magnitude is the recorded #1739 measurement

## Provenance

workflow_fix_target: src/explore_persona_space/eval/judge_pilot.py

Auto-filed by the `/issue 2329` orchestrator during the L5 judge-wave pilot gate (2026-08-21).
Evidence: #2329 `events.jsonl` `epm:progress` v190; the gate artifact's own warning field;
`scripts/issue2329_ladder_judge.py:122/813/816/835`; `scripts/issue2329_decay.py:815-816`;
`src/explore_persona_space/eval/judge_dispatch.py:329/454-489`.

Note on scope: the per-issue script's missing declaration is an experiment-code gap and NOT itself a
workflow-fix candidate; what is filed here is the SHARED-library design choice that makes the unarmed
state the default for every caller.
