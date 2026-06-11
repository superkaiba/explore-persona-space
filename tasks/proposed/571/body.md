---
title: Does negative-panel breadth cause the universal end-of-answer clamp?
kind: experiment
tags: []
created_at: '2026-06-11T01:44:03Z'
has_clean_result: false
parent_id: 560
goal: Test whether negative-panel breadth causes the universal end-of-answer clamp
  by training matched marker adapters on one source context (software-engineer query,
  the i474 recipe held fixed) that differ only in negative-panel size — 15 bystander
  conditions vs a 4-condition class-spanning subset, total negative rows held at 300
  — and reading the end-token logit change on the never-negative held-out personas.
---
## Goal

Test whether negative-panel breadth causes the universal end-of-answer clamp by training matched marker adapters on one source context (software-engineer query, the i474 recipe held fixed) that differ only in negative-panel size — 15 bystander conditions vs a 4-condition class-spanning subset, total negative rows held at 300 — and reading the end-token logit change on the never-negative held-out personas.


### 1. Does negative-panel breadth cause the universal end-of-answer clamp? — Type: Ablation (causal)

**Parent:** #560
**question_relation:** substantially-different
**Goal:** Test whether negative-panel breadth causes the universal end-of-answer clamp by training matched marker adapters on one source context (software-engineer query, the i474 recipe held fixed) that differ only in negative-panel size — 15 bystander conditions vs a 4-condition class-spanning subset, total negative rows held at 300 — and reading the end-token logit change on the never-negative held-out personas.
**Hypothesis:** #560's registered falsification F2 fired — personas the training never mentioned get +13.7 logits of end-token push, breaking the exposure account — and the body explicitly states breadth-causes-clamp is NOT identified by that design. Prediction: the 15-negative arm reproduces the universal clamp (~+13 logits on never-negative personas) while the 4-negative arm lands materially lower, toward the 4-negative recipe's near-zero held-out clamp (#553/#478 anchor −3.1, cross-panel regime caveats acknowledged).
**Falsification:** The between-arm never-negative dz_eos contrast has a persona-cluster CI containing 0 with both arms near +13 → breadth is NOT the cause; the clamp comes from something else in the broad recipe (negative-row count, question coverage, training regime), and the exposure-story autopsy has to look elsewhere.
**Differs from parent:** Exactly ONE variable vs the i474 training recipe: negative-panel breadth (15 conditions x 20 rows vs 4 conditions {A1,B1,C1,D1} x 75 rows; positives, source context A2, total negatives 300, marker-only loss, lr, r=32/alpha=64, ep1 checkpoint all held fixed). The #560 eval surface is reused unchanged.

**Pre-filled spec (from parent):**
- Model: Qwen/Qwen2.5-7B-Instruct (same)
- Data: i474 recipe inputs — frozen R artifacts `issue460_marker_at_end/on_policy_R/` + #406 condition definitions (same; Hub-verified above); eval = same 35 personas x 20 questions @ rev a9fc5a9
- Seeds: 2 training seeds per arm (42, 43) — 4 new LoRAs total
- Eval: same four-float panel driver (`scripts/issue560_crossrecipe_panel.py` machinery), 4 adapters x 35 personas x 20 q; primary DV = never-negative mean dz_eos (32-persona set from #560's classification, held fixed across arms); dz/dmargin reported alongside
- Config: same as i474 loc arm (`scripts/i474_phase23_train.py`) EXCEPT: `_build_negative_rows()` negative panel = {A1,B1,C1,D1} at 75 rows each in the narrow arm

Regime-vs-DV compatibility: training stop is the i474 ep1-checkpoint convention (not a [5,12]-nat band-stop), and the primary DV is the off-source end-token/marker LOGIT read, which has demonstrated dynamic range in exactly this regime — #560 measured graded dz_eos +2.8..+21.0 and dz +5.7..+25.7 across 557 off-source cells, all below the argmax ceiling per `.claude/rules/marker-training-recipe.md` (only source-resident cells saturate).

**Estimated cost:** ~6 GPU-hours on 1x H100 (lora-7b intent): 4 small trains (600 rows, ep1) + 2,800 generations + four-float scoring + smoke.
**If it works:** Breadth is established as the causal lever behind the universal clamp — the recipe-level side-effect a pre-training audit must price in, and the exposure story gets a mechanistic replacement (broad suppression teaches "end your answer" globally).
**If it fails:** Breadth is exonerated; the next suspect is per-bystander row count or the suppression-target composition, and the clamp becomes a property of the marker-only-loss negative rows themselves — either way the F2 autopsy advances one causal step.

**auto_run:** yes
**auto_run_reason:** Clean single-variable ablation with a fully grounded inherited recipe (training script + frozen R + condition defs + eval rig all Hub/main-verified above); cost known and far under cap; no remaining human design choice (source, subset, seeds, DV all pinned); fresh complete-sentence Goal present.

**cost_class:** needs-gpu
**headline_affecting:** no

---
