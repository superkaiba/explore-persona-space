---
title: 'Did #496 warmth training actually implant warmth? (manipulation check on existing
  adapters)'
kind: experiment
tags:
- roadmap-jun05
created_at: '2026-06-08T06:58:29Z'
has_clean_result: false
parent_id: 496
goal: 'Measure whether #496''s six warmth-trained Qwen-2.5-7B adapters actually became
  warmer than base (SocioT Warmth + Claude judge on held-out vulnerability prompts),
  to determine whether #496''s warmth->sycophancy null is a real null or an artifact
  of warmth never being implanted.'
---
# Did #496 warmth training actually implant warmth? (manipulation check on the existing adapters)

## Goal

Measure whether #496's six warmth-trained Qwen-2.5-7B adapters actually became warmer than base (SocioT Warmth + Claude judge on held-out vulnerability prompts), to determine whether #496's warmth->sycophancy null is a real null or an artifact of warmth never being implanted.

The parent task [#496](https://eps.superkaiba.com/tasks/496) trained warmth into six Qwen-2.5-7B source personas and found no source cleared the +0.10 warmth→sycophancy gate. But #496 **never measured whether the warmth-trained models actually became warmer** — the only warmth-side evidence was a qualitative eyeball of completions ("empathic prosody"), and `report_to="none"` was hardcoded so there are no loss curves to inspect convergence. The null is therefore confounded: "warmth doesn't leak into sycophancy" is currently indistinguishable from "warmth was never implanted." This task closes that gap by scoring the **existing** #496 warmth adapters — no retraining.

The original paper (Ibrahim, Hafner & Rocher, [arXiv 2507.21919](https://arxiv.org/abs/2507.21919)) confirmed warmth took via the **SocioT Warmth** metric (their Figure 1A: all five models get progressively warmer, plateauing by epoch 2). This task runs that missing manipulation check.

## What to do

- Load the 6 warmth-trained adapters from #496 (HF `superkaiba1/explore-persona-space/adapters/issue_496/warmth_<source>_seed42`) plus base `Qwen/Qwen2.5-7B-Instruct`.
- Generate completions for each (6 adapters + base) on a held-out set of vulnerability-and-advice prompts (reuse the 50 held-out warmth ablation prompts already uploaded with #496's corpus, `issue496_warmth_sycophancy/warmth_prompts/`; do NOT reuse training prompts).
- Score warmth two ways: (1) the paper's **SocioT Warmth** log-likelihood-ratio metric (primary, matches the paper's manipulation check), and (2) a Claude Sonnet 4.5 judge warmth rating (1-5) as a cross-check on a sample.
- DV: per-source warmth(trained) − warmth(base), with claim-cluster / prompt bootstrap CIs.

## Why it matters / decision it informs

- If warmth clearly went up on all 6 sources → #496's sycophancy null is a **real** narrow→broad null (warmth doesn't drag sycophancy along on this rig), which is the interesting finding.
- If warmth did NOT go up (or only weakly) → #496's null is an **artifact** of under-implanted warmth, and the sycophancy conclusion must be retracted/softened. This directly tells us whether the sibling replication task is needed to settle it.

## Reuse / scope

- Reuses #496's adapters, eval prompts, and judge harness. Cheap: ~1-3 GPU-h (inference + scoring only, no training).
- Single seed (42), inherited from #496. This is a measurement on fixed artifacts, not a new training sweep.
- Not a behavior-implantation experiment (no new training) → contrastive-negatives rule is N/A.

Parent: #496. Relates to the B→B′ (warmth→sycophancy) line (#446).
