---
title: 'Marker implant with 9 contrastive negatives vs #416''s 2: does broader negative
  coverage make ※ source-specific and attenuate the global shift? (software_engineer
  source)'
kind: experiment
tags: []
created_at: '2026-05-29T07:05:08Z'
has_clean_result: false
parent_id: 416
---
---
kind: experiment
parent_id: 416
goal: "Re-run #416's software_engineer ※ marker implant with all 9 other source personas as contrastive negatives (vs #416's 2) and test whether broader negative coverage elevates the source on the marker-logprob leaderboard / attenuates the global marker-affinity shift — isolating whether the global shift is an artifact of under-constrained (2-negative) contrastive training."
---

## Goal

Re-run #416's software_engineer ※ marker implant with **all 9 other source personas** as contrastive negatives (vs #416's **2**), holding everything else identical, and test whether broader negative coverage makes the marker source-specific (software_engineer rises toward the top of the leaderboard) and/or attenuates the global marker-affinity shift — isolating whether the "global shift / marker leaks to all bystanders" finding from #385→#416 is partly an artifact of under-constrained (2-negative) contrastive training.

## Background

[#416](https://eps.superkaiba.com/tasks/416) found that training ※ into software_engineer left it at rank 25/28 (transient bump that washed out), confirming #398's global-marker-affinity-shift reading. But the whole #385→#416 marker line trains against only **2** contrastive negatives (generator default `n_neg=2`, never overridden) while evaluating leakage across **27** bystanders. For 25 of 27 bystanders there is no suppression signal — so the global shift may be a consequence of under-constrained contrastive training, not a fundamental property of marker implants.

## What this tests

- Does software_engineer (rank 25/28 with 2 negatives) rise toward the top with 9 negatives? (source-specificity rescued by broader coverage)
- Does the panel-wide global shift attenuate (smaller panel-mean rise, lower ρ, source separates from bystanders)?
- Direct comparison to #416 (2-negative) — single variable: negative coverage 2→9.

## Design

- Source = software_engineer (positive rows byte-identical to #416). Negatives = all 9 other source personas (librarian, kindergarten_teacher, data_scientist, medical_doctor, french_person, villain, comedian, police_officer, zelthari_scholar). 200 positive + 9×200 negative = 2000 rows, reusing #416's cached generic answers (no new Claude calls).
- Identical recipe: Qwen-2.5-7B-Instruct, LoRA r=32 α=64, lr=1e-5, 1600 steps, seed=42, bare ※ id 63680, 22-checkpoint schedule, same 28-persona eval panel, dual-probe (pos0+endpos) teacher-forced logp eval ONLY.
- Confound (flag in writeup): at fixed 1600 steps the source sees ~12.8 epochs vs #416's ~42.7 (positive-exposure dilution).

## Cross-references

- Parent: #416 (software_engineer, 2 negatives). Grandparent: #398 (librarian, 2 negatives).
