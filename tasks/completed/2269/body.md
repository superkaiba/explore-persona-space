---
title: 'verify_plan.py: regen-headroom arithmetic check (armed ≥2× re-gen vs max_model_len
  pin) — third recurrence of the #505/#601 class'
kind: infra
tags: []
created_at: '2026-08-13T07:27:03Z'
has_clean_result: false
origin_prompt: 'Methodology critic workflow-fix prose follow-up on #2221 plan v9 (2026-08-13):
  armed P6 re-gen at 2x2048 against the max_model_len=4096 pin leaves zero prompt
  headroom; verify_plan v9 run was PASS n_warn=0 — class invisible to the mechanical
  checker'
workflow: v1
---
# verify_plan.py: add a regen-headroom arithmetic check (armed ≥2× re-gen vs max_model_len pin)

## Goal

Add a `verify_plan.py` WARN-class check: for any plan declaring an ARMED ≥2× cap-hit re-gen trigger alongside a `max_model_len` pin, assert `max_model_len − 2×cap ≥ stated prompt-token bound` — WARN when the arithmetic is non-positive or the prompt bound is unstated. Demonstrated miss (third recurrence of the #505/#601 cap-raise-vs-`max_model_len` class): #2221 plan v9 armed the P6 re-gen at regen_cap = 2×2048 = 4096 against `build_vllm_engine`'s hard `max_model_len=4096` pin — zero prompt headroom, so the reused `_regen_cell` mechanism would skip EVERY row as `regen_overlong_skipped`, write `regen_applied: true` with n_regen=0, and silently re-commit the parent's flagged cap-hit deviation while the plan claimed to fix it. verify_plan v9 pass: PASS n_fail=0 n_warn=0 — the class is currently invisible to the mechanical checker; the Methodology critic caught it by reading the engine builder and the regen docstring.

Sketch (from the critic's mechanizable note): detect (a) re-gen arming vocabulary ("re-gen trigger ARMED", "regenerate ... at ≥2× the cap", `phase_rollouts_regen`/`_regen_cell` mentions) with a numeric cap, (b) a `max_model_len` (or `VLLM_MAX_MODEL_LEN`) numeric pin in the same plan, (c) a stated prompt-token bound (e.g. "≤1,900 prompt tokens"); WARN when 2×cap + prompt_bound > max_model_len or when (a)+(b) present with no stated prompt bound. Standalone N/A escape line for incidental mentions (e.g. `N/A — no armed re-gen trigger`), per the existing check grammar. Include a pin test reproducing the #2221 v9 shape and a clean case (regen engine at 8192).

## Provenance

Surfaced as a workflow-fix prose follow-up by the Methodology `critic` on #2221 plan v9 (2026-08-13, Phase 2 of the specialized_corpus_remine amendment round); routed by the #2221 orchestrator per `.claude/rules/workflow-fix-on-bug.md`. Target file: `scripts/verify_plan.py`. Also worth a one-line extension to the `gotchas.md` cap-raise rule naming the REGEN leg explicitly. Candidate fingerprint: verify-plan-regen-headroom-arithmetic.
