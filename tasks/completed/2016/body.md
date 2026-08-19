---
title: 'workflow-fix: figure-PNG-vs-own-sidecar render consistency check'
kind: infra
tags:
- wf-fix
- wf-fix-fp:bdb686fc0d92
created_at: '2026-08-02T08:29:24Z'
has_clean_result: false
origin_prompt: 'clean-result-critic round-3 prose follow-up on #1768, 2026-08-02 (verbatim
  in body Provenance)'
workflow: v1
---
## Overview / Motivation
Auto-filed from a prose follow-up surfaced by clean-result-critic round 3 on task #1768 (2026-08-02).
## Goal
Add a verify_task_body check comparing each committed `figures/issue_<N>/**/*.png` against its OWN `.meta.json` sidecar — count rendered bar groups / plotted series (colour-agnostic column runs) vs the sidecar's `points` group sizes and `xticklabels` length; FAIL on mismatch.
## Workflow gap
- **Bug observed:** `figures/issue_1768/map_augmentation/operator_kv_read.png` at HEAD (8dec46caa6) was a degraded re-render drawing 3 of 8 arm groups (one behavior colour) while its OWN sidecar at the same commit declared 8 points per series and 8 x-tick labels; caught only at adversarial review round 3. Root context: the figure was regenerated from a partial cell set mid-shared-root-race (#2015's mechanism).
- **Why it is a workflow gap:** the verifier has figure checks for URL/SHA pinning (22), text tokens (24/28), panel prose (26), tracked-at-HEAD (29), prose numerics (33) — but nothing compares a PNG's rendered content against its own committed sidecar, so a partial re-render ships silently; generalizes to any figure regenerated from an incomplete cell set.
- **Confidence (emitter):** high (mechanizable: yes, per the emitting critic)
- verified-at-filing: `grep -n "check_figure" scripts/verify_task_body.py` → checks 22/24/26/28/29/33 present, none reads PNG pixel content vs sidecar structure (context read of each check's doc line, 2026-08-02); the degraded-render instance verified by the critic reading three commits' PNGs (8/8/3 bar groups at ee236b8d/1ac5e238/HEAD) with byte-identical sidecars.
## Proposed change (candidate diff sketch — refine in planning)
diff_sketch: |
  New WARN-or-FAIL check (~"check_figure_render_vs_sidecar"): for each body-linked PNG with a
  sidecar, load PNG (PIL), count distinct bar-group column runs (colour-agnostic) and/or plotted
  series; compare against len(sidecar xticklabels) and points-per-series; mismatch -> finding
  naming the figure + counts. Conservative: only fire when the sidecar declares countable
  structure (bar charts); skip continuous plots.
## Scope / surfaces
- Primary target: `scripts/verify_task_body.py` (+ tests with a synthetic degraded-render fixture).
## Constraints / invariants
- WARN-first for grandfathered bodies if FAIL proves noisy; never blocks on figures without sidecars; workflow-surface only; recursion guard applies.
## Provenance
- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: bdb686fc0d92
Verbatim surfaced prose: "scripts/verify_task_body.py has no check that a committed figure PNG agrees with its OWN sidecar. The #1768 case is mechanizable — for each figures/issue_<N>/**/*.png with a .meta.json, count rendered bar groups / plotted series in the PNG (colour-agnostic column runs) and compare against the sidecar's points group sizes and xticklabels length, FAILing on a mismatch. That would have caught this degraded re-render at commit time rather than at adversarial review, and it generalizes to any figure re-rendered from a partial cell set."
