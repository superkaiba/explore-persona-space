---
title: 'Thicken #528''s segmentation CI with a validating-only sweep at n=10 seeds
  (role-header vs system-prompt)'
kind: experiment
tags: []
created_at: '2026-06-10T12:10:48Z'
has_clean_result: false
parent_id: 528
goal: 'Re-test whether the role-header encoding segments validating from off-target
  personas more cleanly than a system-prompt encoding (paired leakage gap ≤ -0.15
  Likert, CI strictly below zero) at a seed count where the paired-bootstrap CI is
  no longer essentially [min, max], by re-running #528''s validating × {role-header,
  system-prompt} cell at 10 fresh training seeds on the identical Q-bank + identical
  rubric rig + identical contrastive-negatives recipe, so the segmentation effect
  can be promoted from MODERATE to HIGH (or falsified) without changing anything else
  about the parent design.'
---
---
title: "Thicken #528's segmentation CI with a validating-only sweep at n=10 seeds (role-header vs system-prompt)"
kind: experiment
parent_id: 528
goal: "Re-test whether the role-header encoding segments validating from off-target personas more cleanly than a system-prompt encoding (paired leakage gap ≤ -0.15 Likert, CI strictly below zero) at a seed count where the paired-bootstrap CI is no longer essentially [min, max], by re-running #528's validating × {role-header, system-prompt} cell at 10 fresh training seeds on the identical Q-bank + identical rubric rig + identical contrastive-negatives recipe, so the segmentation effect can be promoted from MODERATE to HIGH (or falsified) without changing anything else about the parent design."
tags: []
auto_run: yes
auto_run_reason: "corrective re-run of #528's H2 PASS that fixes the named 'N=3 paired-bootstrap CI is essentially [min, max]' validity defect with one variable changed (seed count) and a fully-grounded recipe inherited verbatim from #528's validating row; no human design call needed; cost ~3 GPU-h, well under the auto-approve cap; same canonical shape as task #520 → #527."
cost_class: needs-gpu
headline_affecting: yes
---

## Goal

Re-test whether the role-header encoding segments validating from off-target personas more cleanly than a system-prompt encoding (paired leakage gap ≤ -0.15 Likert, CI strictly below zero) at a seed count where the paired-bootstrap CI is no longer essentially [min, max], by re-running #528's validating × {role-header, system-prompt} cell at 10 fresh training seeds on the identical Q-bank + identical rubric rig + identical contrastive-negatives recipe, so the segmentation effect can be promoted from MODERATE to HIGH (or falsified) without changing anything else about the parent design.

## Hypothesis

The +0.155 Likert role-vs-system segmentation gap with 3/3 seeds in the negative direction in #528 reflects a real role-header advantage and will replicate at higher N: the n=10 across-seed mean leakage gap stays ≤ -0.10 Likert with a paired-bootstrap CI that strictly excludes zero, and ≥ 8 of 10 seeds carry the negative sign.

## Falsification

If the n=10 across-seed mean leakage gap drifts above -0.05 Likert, or the bootstrap CI crosses zero, or fewer than 6 of 10 seeds carry the negative sign, #528's segmentation PASS is downgraded to a 3-seed coincidence rather than a real segmentation effect, and the role-header surface buys nothing measurable beyond noise on this trait.

## Differs from parent

- Seeds: 3 → 10 fresh seeds (11, 23, 73, 191, 257, 401, 503, 631, 757, 911 — distinct from #528's {42, 137, 1337}).
- Trait set: validating only (no conciseness / asks-clarifying-first / calibrated-uncertainty re-train — those cells stayed saturated in #528).
- Everything else held identical to #528's validating row: same Q-bank sha256s, same Sonnet 4.5 rubric, same LoRA r=32/alpha=64/lr=1e-5/5-epoch recipe, same contrastive-negatives composition, same per-encoding saturation gate, same H2 PASS bar.

## Reuse from parent

- Q-banks (sha256-pinned from `data/issue_528/validating/`).
- R_pos + R_neg generations from #528's `issue528_role_header_traits/` HF data repo path (no fresh Sonnet generation needed — those are model-agnostic teacher responses).
- Base headroom probe (#528's `eval_results/issue_528/base_headroom_judge.json` covers validating own_scenario base; reuse unless we change the eval slice).

## Spec

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Seeds: 11, 23, 73, 191, 257, 401, 503, 631, 757, 911 (10 fresh, distinct from #528's 3)
- Encodings: system-prompt + role-header (parent's "system" and "role" arms)
- Total cells: 10 seeds × 2 encodings × 1 trait = 20 trained cells
- Eval: vLLM batched greedy, temperature 0, max_new_tokens 2048; 5 eval contexts × 40 prompts per cell; Sonnet 4.5 Likert 1-5, 3 judge calls averaged at temp 0
- Saturation gate: per-encoding (per #528's free-analysis follow-up) — validating-role expected NOT saturated (base CI [3.08, 3.67]), validating-system expected saturated (base CI [3.78, 4.02])

## Estimated cost

~7 H100 wall-min per cell × 20 cells = ~2.4 GPU-h train; ~3,400 trained generations × 3 judge calls + reuse of #528's base = ~10,200 Sonnet calls × ~0.6 s = ~1.7 h API; total ~3 GPU-h on 1× H100 (`lora-7b` intent) + ~2 h Sonnet API.

## PASS / KILL

- **PASS (segmentation effect confirmed):** n=10 across-seed mean leakage gap ≤ -0.10 Likert AND paired-bootstrap CI strictly excludes zero AND ≥ 8 of 10 seeds carry the negative sign → q:spec-role-header flips from MODERATE (3-seed coincidence not ruled out) to HIGH for validating.
- **KILL (3-seed PASS was a coincidence):** mean leakage gap > -0.05 Likert OR CI crosses zero OR < 6 of 10 seeds negative → q:spec-role-header reads as twice-unverified (#498 null + #528 N=3 coincidence + this null), role-header story parked pending a fundamentally different segmentation probe.

## Relates to

q:spec-role-header (`docs/open_questions.md`)
