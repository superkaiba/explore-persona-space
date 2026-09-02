---
name: judge-pilot-report-resume-fields
description: PilotGateReport JSON lacks instrument n_draws — derive it per arm as n_draws // n_items; wave_transport/rubric_hash/judge_model/max_tokens ARE persisted; key resume dirs on every re-run-triggering constant
metadata:
  type: reference
---

For a #2479-style consumer-side resume of `eval.judge_pilot.judge_pilot_gate`
(the rule-26 PASS-report skip; built on #2658 group-E, commit `f3827fc7787`):

- `pilot_gate_report.json` (PilotGateReport.to_json) persists `passed`,
  `verdict`, `judge_model`, `max_tokens`, `rubric_hash`
  (= sha256(eval_prompt)[:16]), `parse_fail_threshold`,
  `api_refusal_threshold` (rule 26(d), verdict-bearing — compare it AND fold
  it into the gate-dir key; NEITHER waiver tuple (`waive_api_refusal_arms`
  26(d), `waive_parse_fail_arms` 26(b)) is persisted by PilotGateReport, so
  waiver tracking is key-only — #2658 E rounds 2/3, commits `103747b429b` /
  `1db6b076f79`; round 3 added a mechanical enumeration guard
  (test_pilot_gate_parameter_enumeration_guard, unit6) walking the live
  signature + report fields against an introspectable
  `pilot_gate_key_payload(row)` — extend THAT payload/allow-list rather than
  re-deriving the classification), `wave_transport`, and
  per-arm `n_items`/`n_draws` — but NOT the instrument n_draws. Derive it:
  `arm.n_draws // arm.n_items` (every subsampled item gets exactly n_draws
  draws), consistent across arms; a non-integer ratio = mismatch.
- Compute the expected declared transport via `judge_pilot._wave_routing(
  wave_n_calls, threshold_base, force_sync).path` — never hardcode "batch".
- The gate FAILs any wave-declared arm with `n_cached > 0`, so a resume
  MISMATCH that re-runs at the SAME dir wedges on its own cache: fold every
  re-run-triggering constant that is OUTSIDE the instrument fingerprint
  (declared transport, parse-fail threshold, effective-draws floor) into the
  gate dir's path key so a genuine re-run always gets a fresh cache.
- Test-side: a fake gate must WRITE a production-format report JSON at
  `report_path` (same field names, incl. realized per-arm n_items/n_draws via
  the library's floor-division sizing) or the resume path never executes.

Related: [[worktree-commit-and-selector-vintage]].
