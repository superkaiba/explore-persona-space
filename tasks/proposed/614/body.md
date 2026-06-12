---
title: 'Assistant-twin immunity: role gating vs trained-negative suppression generalization
  (canned software_engineer retrain with assistant swapped out of the negative set)'
kind: experiment
tags: []
created_at: '2026-06-12T16:17:24Z'
has_clean_result: false
parent_id: 612
goal: Determine whether the assistant-twin immunity to implanted sycophancy is role/prior
  gating or suppression generalizing from the trained default-assistant negative,
  by retraining the canned software-engineer cells with the assistant persona replaced
  in the contrastive negative set and re-reading the twin, daycare-teacher, and gradient
  cells.
---
## Goal

Determine whether the assistant-twin immunity to implanted sycophancy is role/prior gating or suppression generalizing from the trained default-assistant negative, by retraining the canned software-engineer cells with the assistant persona replaced in the contrastive negative set and re-reading the twin, daycare-teacher, and gradient cells.


### 3. Assistant-twin immunity: role gating vs trained-negative suppression generalization — Type: Ablation

**Parent:** #612
**question_relation:** substantially-different
**Goal:** Determine whether the assistant-twin immunity to implanted sycophancy is role/prior gating or suppression generalizing from the trained default-assistant negative, by retraining the canned software-engineer cells with the assistant persona replaced in the contrastive negative set and re-reading the twin, daycare-teacher, and gradient cells.
**Hypothesis:** Suppression generalization: the default `assistant` is a trained negative for software_engineer (cosine 0.969 to the source), the probe twins are its 0.979-cosine near-twins, and their mildly negative lifts (-0.034/-0.035, CIs wholly below zero) are trained-down corrections generalizing. With `assistant` swapped out of the negative set, the twins' lift rises to >= +0.10 (joining daycare_teacher +0.13 and school_principal +0.43 at the same cosine band).
**Falsification:** Twins stay flat-to-negative (|Delta| < 0.05) without the assistant negative — the immunity is a property of the personas (role/prior gating), not of the training design, and the standing cross-task anomaly (#411/#591/#612) survives its strongest alternative explanation.
**Differs from parent:** Exactly ONE thing — negative-set membership for the canned software_engineer cells: `assistant` replaced by `french_person` (selection rule: drawn from the existing #411 negative roster, maximally far from the probe twins, with written correction rows over the same 200 claims already present in the frozen kindergarten_teacher pool — keeps counts, ratio, and claims identical). Everything else (canned positives, recipe, seeds, panel, eval, judge) verbatim from the parent's canned arm.

**Pre-filled spec (from parent):**
- Model: Qwen/Qwen2.5-7B-Instruct, LoRA r=32/alpha=64 rsLoRA all-linear, lr 1e-5, 3 epochs, batch 4x4 — parent recipe verbatim
- Data: frozen #411 canned pools (Hub-verified above), software_engineer positives + negatives with the assistant rows swapped for french_person correction rows on the same claims; 700 rows, 200:500
- Seeds: {42, 137}
- Eval: same 30-persona panel x 60 claims x 10 rollouts + Haiku judge; primary cells = virtual_assistant, digital_helper, daycare_teacher; full panel kept for the inverted-gradient (rho -0.47) re-read
- Config: same as parent EXCEPT the negative-panel swap (and the disjointness assert updated accordingly)

**Estimated cost:** ~3 GPU-hours (2 train+merge+eval cells on the parent's per-cell budget) + ~36k Haiku judge calls
**If it works (twins light up):** The anomaly relocates from "persona property" to "training-design artifact" — every prior immunity read (#411, #591, #612) gets reinterpreted as suppression generalization, and contrastive-negative placement becomes a recognized lever on apparent leakage topology (directly relevant to the contrastive-negatives rule's overclaim caveats).
**If it fails (twins stay flat):** Role/prior gating survives its strongest mechanical alternative, hardening the parent's same-cosine-spread finding into a genuine geometry-is-not-sufficient claim; next probe is twins far from every trained negative.

**auto_run:** yes
**auto_run_reason:** Fully specified single-variable ablation with a pinned replacement rule and Hub-verified inputs; as a substantially-different proposal it is FILED as a proposed child for manual triage, never auto-spawned. Files via `task.py new --parent 612 --goal "<Goal above>"`.

**cost_class:** needs-gpu
**headline_affecting:** no

---

Not proposed as top-3 (lower info gain, already named in the body): the sycophantic-affect second-judge pass and the per-claim-family breakdown (both cost_class: free-analysis, headline_affecting: no — neither can move a registered verdict per the idx48/idx8 bound, so neither qualifies for the Step 9a-ter auto-run, and both rank below the three above on information gain).

**To create any of these as issues, reply on this issue with `create N`
(e.g., `create 1` or `create 1,3`).**
<!-- /epm:follow-ups -->
