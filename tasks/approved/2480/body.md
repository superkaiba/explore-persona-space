---
title: 'Harden #2394 jailbreak-mining pilot to paper-citable grade (commit JSONs +
  verify headline numbers + audit constructions)'
kind: analysis
tags: []
created_at: '2026-08-22T20:24:25Z'
has_clean_result: false
parent_id: 2394
origin_prompt: 'Critique F10: #2394 cited at paper grade from unreviewed scratch report'
workflow: v1
---
## Goal

Harden #2394 (jailbreak-context mining pilot) to paper-citable grade, 0 GPU: (1) commit its headline eval JSONs (map_arms_results.json, label_efficiency_results.json, and any companions) from the /mnt/eps-data staging into eval_results/issue_2394/; (2) re-read and verify the headline numbers (probe PR-AUC 0.973 = oracle 0.974; map arms <=0.43; labels-to-PR-0.80: 10 vs ~47-51; benign-fit reconstruction R2 -0.12..-0.88 vs in-domain +0.33..+0.62) against the committed artifacts; (3) sanity-audit the two constructions a reviewer will poke: the 5% base-rate composition and the same-family failed-jailbreak negatives — state whether either inflates the probe's PR-AUC.

## Provenance

Paper outline restructure 2026-08-22; adversarial critique F10 (docs/paper_context_answer_map/outline_critique_2026-08-22.md): #2394 is cited at paper grade in Results III from an unreviewed scratch report (docs/scratch/jailbreak_mining_pilot.md @ cb1f5f836c) with JSONs verified only on staging — every other number in the section passed the clean-result + critic pipeline.

## Design notes

- Read-only analysis + artifact commits by explicit path; no new generation or judging. If a number fails re-verification, flag it in the report and in a marker on #2394 — do not silently correct the scratch report.
